import torch
from safetensors.torch import load_file

from boo_sd_scripts.tdm import tdm_network
from library import moss_image_models
from networks import lora_generic

SMALL_CONFIG = {
    "dim": 64,
    "ffn_dim": 96,
    "num_heads": 4,
    "num_layers": 3,
    "text_dim": 24,
    "freq_dim": 16,
    "latent_channels": 8,
    "rope_base": 10000.0,
}
NUM_UNET_MODULES = 3 * 6


def make_dit(seed=0):
    torch.manual_seed(seed)
    return moss_image_models.MossImage(**SMALL_CONFIG, attn_mode="torch", split_attn=False).eval()


def make_network(dit, **kwargs):
    return tdm_network.create_network(1.0, 4, 4.0, None, [None], dit, **kwargs)


def _randomize_up(net):
    for lora in net.unet_loras:
        torch.nn.init.normal_(lora.lora_up.weight, std=0.5)


def _forward(dit):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 8, 4, 4, generator=g)
    ctx = torch.randn(1, 5, 24, generator=g)
    mask = torch.ones(1, 5, dtype=torch.int32)
    with torch.no_grad():
        return dit(x, torch.tensor([500.0]), ctx, mask)


def test_create_network_structure():
    net = make_network(make_dit(), fake_dim="2")
    assert isinstance(net, tdm_network.TDMNetwork)
    assert isinstance(net.fake, lora_generic.LoRANetwork)
    assert len(net.unet_loras) == NUM_UNET_MODULES
    assert len(net.fake.unet_loras) == NUM_UNET_MODULES
    assert net.fake.unet_loras[0].lora_dim == 2
    assert net.unet_loras[0].lora_dim == 4
    assert all(lora.multiplier == 0.0 for lora in net.fake.unet_loras)


def test_modes_select_teacher_student_fake():
    dit = make_dit()
    net = make_network(dit)
    base = _forward(dit)
    net.apply_to([None], dit, apply_text_encoder=False, apply_unet=True)
    _randomize_up(net)
    _randomize_up(net.fake)

    net.set_mode(0.0, 0.0)
    assert torch.allclose(_forward(dit), base, atol=1e-5)

    net.set_mode(1.0, 0.0)
    student = _forward(dit)
    net.set_mode(0.0, 1.0)
    fake = _forward(dit)
    assert not torch.allclose(student, base, atol=1e-4)
    assert not torch.allclose(fake, base, atol=1e-4)
    assert not torch.allclose(student, fake, atol=1e-4)


def _fake_group_checks(net, params):
    fake_ids = {id(p) for p in net.fake.parameters()}
    student_ids = {id(p) for p in net.parameters()} - fake_ids
    group_ids = {id(p) for p in params[-1]["params"]}
    assert len(params[-1]["params"]) == 2 * NUM_UNET_MODULES
    assert group_ids == fake_ids
    assert student_ids and not (student_ids & fake_ids)
    assert not any(id(p) in fake_ids for g in params[:-1] for p in g["params"])


def test_optimizer_groups_put_fake_params_in_their_own_group_with_10x_lr():
    dit = make_dit()
    net = make_network(dit)
    net.apply_to([None], dit, apply_text_encoder=False, apply_unet=True)
    params, descriptions = net.prepare_optimizer_params_with_multiple_te_lrs(None, 1e-4, 1e-4)
    assert len(params) == len(descriptions)
    _fake_group_checks(net, params)
    assert params[-1]["lr"] == 1e-3


def test_explicit_fake_lr():
    dit = make_dit()
    net = make_network(dit, fake_lr="5e-5")
    net.apply_to([None], dit, apply_text_encoder=False, apply_unet=True)
    params, _ = net.prepare_optimizer_params_with_multiple_te_lrs(None, 1e-4, 1e-4)
    _fake_group_checks(net, params)
    assert params[-1]["lr"] == 5e-5


def _grad_forward_backward(dit):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 8, 4, 4, generator=g)
    ctx = torch.randn(1, 5, 24, generator=g)
    mask = torch.ones(1, 5, dtype=torch.int32)
    dit(x, torch.tensor([500.0]), ctx, mask).sum().backward()


def _has_grad(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_gradient_flows_only_to_active_lora():
    dit = make_dit()
    dit.requires_grad_(False)
    net = make_network(dit)
    net.apply_to([None], dit, apply_text_encoder=False, apply_unet=True)
    _randomize_up(net)
    _randomize_up(net.fake)
    student_params = [p for p in net.parameters() if id(p) not in {id(q) for q in net.fake.parameters()}]
    assert student_params and list(net.fake.parameters())

    def student_has_grad():
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in student_params)

    net.zero_grad(set_to_none=True)
    net.set_mode(0.0, 1.0)
    _grad_forward_backward(dit)
    assert _has_grad(net.fake)
    assert not student_has_grad()

    net.zero_grad(set_to_none=True)
    net.set_mode(1.0, 0.0)
    _grad_forward_backward(dit)
    assert student_has_grad()
    assert not _has_grad(net.fake)


def test_save_weights_contains_only_student_keys(tmp_path):
    dit = make_dit()
    net = make_network(dit)
    net.apply_to([None], dit, apply_text_encoder=False, apply_unet=True)  # LoRA modules are registered in apply_to
    path = str(tmp_path / "student.safetensors")
    net.save_weights(path, None, {})
    keys = list(load_file(path).keys())
    assert keys and all(k.startswith("lora_unet") for k in keys)
    assert not any("fake" in k for k in keys)
    assert "fake" in net._modules  # restored after saving
    assert any(k.startswith("fake.") for k in net.state_dict())


def test_create_network_from_weights_is_exposed():
    assert tdm_network.create_network_from_weights is lora_generic.create_network_from_weights
