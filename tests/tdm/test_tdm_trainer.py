import argparse
import contextlib

import pytest
import torch

import boo_sd_scripts._sd_scripts  # noqa: F401
from boo_sd_scripts.tdm import moss_image_train_network_tdm as tdm_trainer
from boo_sd_scripts.tdm import tdm_network
from library import moss_image_models

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


class _Accelerator:
    device = torch.device("cpu")

    def autocast(self):
        return contextlib.nullcontext()

    def unwrap_model(self, model):
        return model


def _args(**overrides):
    values = {
        "tdm_student_steps": 2,
        "tdm_flow_shift": 5.0,
        "tdm_cfg_scale": 3.0,
        "tdm_fake_loss_weight": 1.0,
        "sample_uncond_skip_layers": "1",
        "gradient_checkpointing": False,
        "network_module": "boo_sd_scripts.tdm.tdm_network",
        "blocks_to_swap": None,
        "loss_type": "l2",
        "compile": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _setup(seed=0):
    torch.manual_seed(seed)
    dit = moss_image_models.MossImage(**SMALL_CONFIG, attn_mode="torch", split_attn=False)
    net = tdm_network.create_network(1.0, 4, 4.0, None, [None], dit)
    net.apply_to([None], dit, apply_text_encoder=False, apply_unet=True)
    net.requires_grad_(True)
    for n in (net, net.fake):  # non-zero up weights so every LoRA path carries gradient
        for lora in n.unet_loras:
            torch.nn.init.normal_(lora.lora_up.weight, std=0.05)
    latents = torch.randn(2, 8, 4, 4)
    cond, uncond = torch.randn(2, 5, 24), torch.randn(2, 5, 24)
    mask = torch.ones(2, 5, dtype=torch.int32)
    batch = {"tdm_uncond": (uncond, mask)}
    return dit, net, latents, batch, [cond, mask]


def _step(dit, net, latents, batch, conds, args=None, is_train=True):
    trainer = tdm_trainer.MossImageTDMTrainer()
    out = trainer.get_noise_pred_and_target(
        args or _args(), _Accelerator(), None, latents, batch, conds, dit, net, torch.float32, True, is_train=is_train
    )
    return trainer, out


def test_get_noise_pred_and_target_shapes_and_detach():
    dit, net, latents, batch, conds = _setup()
    trainer, (pred, target, timesteps, weighting) = _step(dit, net, latents, batch, conds)
    assert pred.shape == latents.shape and pred.requires_grad
    assert target.shape == latents.shape and not target.requires_grad
    assert weighting.shape == (2, 1, 1, 1) and not weighting.requires_grad
    assert timesteps.shape == (2,)
    assert trainer._fake_loss.shape == (2,) and trainer._fake_loss.requires_grad


def test_multipliers_are_restored_after_step():
    dit, net, latents, batch, conds = _setup()
    _step(dit, net, latents, batch, conds)
    assert all(lora.multiplier == 1.0 for lora in net.unet_loras)
    assert all(lora.multiplier == 0.0 for lora in net.fake.unet_loras)


def test_generator_loss_only_trains_the_student():
    dit, net, latents, batch, conds = _setup()
    _, (pred, target, _, weighting) = _step(dit, net, latents, batch, conds)
    ((pred - target).pow(2) * weighting).mean().backward()
    student_params = [p for lora in net.unet_loras for p in lora.parameters()]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student_params)
    assert all(p.grad is None for p in net.fake.parameters())


def test_fake_loss_only_trains_the_fake_score():
    dit, net, latents, batch, conds = _setup()
    trainer, _ = _step(dit, net, latents, batch, conds)
    trainer._fake_loss.mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.fake.parameters())
    assert all(p.grad is None for l in net.unet_loras for p in l.parameters())


def test_validation_step_has_no_fake_loss():
    dit, net, latents, batch, conds = _setup()
    trainer, (pred, *_rest) = _step(dit, net, latents, batch, conds, is_train=False)
    assert trainer._fake_loss is None
    assert not pred.requires_grad


def test_post_process_loss_adds_and_clears_fake_loss():
    trainer = tdm_trainer.MossImageTDMTrainer()
    trainer._fake_loss = torch.tensor([1.0, 3.0])
    out = trainer.post_process_loss(torch.tensor([0.5, 0.5]), _args(tdm_fake_loss_weight=2.0), None, None)
    assert torch.allclose(out, torch.tensor([2.5, 6.5]))
    assert trainer._fake_loss is None
    assert torch.allclose(trainer.post_process_loss(torch.tensor([0.5]), _args(), None, None), torch.tensor([0.5]))


def test_process_batch_stashes_uncond_and_zeroes_dropout(monkeypatch):
    from moss_image_train_network import MossImageNetworkTrainer

    monkeypatch.setattr(MossImageNetworkTrainer, "process_batch", lambda self, batch, *a, **k: batch)
    prompt, mask = torch.randn(2, 5, 24), torch.ones(2, 5, dtype=torch.int32)
    uncond, umask = torch.randn(2, 3, 24), torch.ones(2, 3, dtype=torch.int32)
    batch = {"text_encoder_outputs_list": [prompt, mask, uncond, umask, torch.tensor([0.5, 0.5])]}
    out = tdm_trainer.MossImageTDMTrainer().process_batch(batch)
    assert out["tdm_uncond"][0] is uncond and out["tdm_uncond"][1] is umask
    assert len(out["text_encoder_outputs_list"]) == 5
    assert out["text_encoder_outputs_list"][-1].sum() == 0


def test_validate_tdm_args():
    tdm_trainer.validate_tdm_args(_args())
    with pytest.raises(AssertionError):
        tdm_trainer.validate_tdm_args(_args(network_module="networks.lora_generic"))
    with pytest.raises(AssertionError):
        tdm_trainer.validate_tdm_args(_args(tdm_cfg_scale=0.5))
    with pytest.raises(AssertionError):
        tdm_trainer.validate_tdm_args(_args(blocks_to_swap=4))
    with pytest.raises(AssertionError):
        tdm_trainer.validate_tdm_args(_args(loss_type="huber"))


def test_parser_defaults():
    args = tdm_trainer.setup_parser().parse_args([])
    assert (args.tdm_student_steps, args.tdm_flow_shift, args.tdm_cfg_scale, args.tdm_fake_loss_weight) == (4, 5.0, 3.0, 1.0)


def _total_loss_grads(gradient_checkpointing, cpu_offload=False):
    dit, net, latents, batch, conds = _setup()
    if gradient_checkpointing:
        dit.enable_gradient_checkpointing(cpu_offload=cpu_offload)
    torch.manual_seed(1)
    trainer, (pred, target, _, weighting) = _step(
        dit, net, latents, batch, conds, args=_args(gradient_checkpointing=gradient_checkpointing)
    )
    (((pred - target).pow(2) * weighting).mean() + trainer._fake_loss.mean()).backward()
    return [p.grad for p in net.parameters()]


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_gradient_checkpointing_recomputes_in_the_forward_mode(cpu_offload):
    # backward runs after the mode is switched back to the student, so the recomputation must restore each forward's mode
    plain = _total_loss_grads(False)
    checkpointed = _total_loss_grads(True, cpu_offload)
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(plain, checkpointed))


def test_missing_uncond_embeddings_raise():
    dit, net, latents, _, conds = _setup()
    with pytest.raises(ValueError, match="cache_text_encoder_outputs"):
        _step(dit, net, latents, {}, conds)


def test_validate_tdm_args_rejects_unsloth_offload():
    with pytest.raises(AssertionError):
        tdm_trainer.validate_tdm_args(_args(unsloth_offload_checkpointing=True))


def test_validate_tdm_args_rejects_compile_with_gradient_checkpointing():
    tdm_trainer.validate_tdm_args(_args(compile=True))
    tdm_trainer.validate_tdm_args(_args(gradient_checkpointing=True))
    with pytest.raises(AssertionError, match="--compile"):
        tdm_trainer.validate_tdm_args(_args(compile=True, gradient_checkpointing=True))


def test_mode_aware_checkpointing_requires_torch_checkpoint(monkeypatch):
    monkeypatch.delattr(tdm_trainer.moss_image_models, "torch_checkpoint")
    with pytest.raises(AttributeError, match="torch_checkpoint"), tdm_trainer._mode_aware_checkpointing(None):
        pass
