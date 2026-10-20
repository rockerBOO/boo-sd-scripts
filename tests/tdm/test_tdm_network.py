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


def test_optimizer_groups_put_fake_params_in_their_own_group_with_10x_lr():
    net = make_network(make_dit())
    params, descriptions = net.prepare_optimizer_params_with_multiple_te_lrs(None, 1e-4, 1e-4)
    assert len(params) == len(descriptions)
    fake_ids = {id(p) for p in net.fake.parameters()}
    student_groups = params[:-1]
    assert params[-1]["lr"] == 1e-3
    assert {id(p) for p in params[-1]["params"]} == fake_ids
    assert all(id(p) not in fake_ids for g in student_groups for p in g["params"])


def test_explicit_fake_lr():
    net = make_network(make_dit(), fake_lr="5e-5")
    params, _ = net.prepare_optimizer_params_with_multiple_te_lrs(None, 1e-4, 1e-4)
    assert params[-1]["lr"] == 5e-5


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
