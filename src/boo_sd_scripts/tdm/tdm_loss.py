"""TDM loss terms (data-free score distillation, x0-space, as in the official implementation)."""

import torch

WEIGHTING_EPS = 1e-6


def tdm_generator_target(
    x0_student: torch.Tensor,
    x0_cond: torch.Tensor,
    x0_uncond: torch.Tensor,
    x0_fake: torch.Tensor,
    cfg_scale: float,
):
    """Return (target, weighting) for the generator loss `mse(x0_student, target) * weighting`.

    target = x0_student + (teacher - fake) + (cfg - 1) * (teacher_cond - teacher_uncond), all detached.
    weighting = 1 / mean|x0_student - teacher_cfg| per sample, shape [B,1,1,1].
    """
    x0_student = x0_student.detach().float()
    x0_cond, x0_uncond, x0_fake = x0_cond.detach().float(), x0_uncond.detach().float(), x0_fake.detach().float()
    guidance = (cfg_scale - 1.0) * (x0_cond - x0_uncond)
    target = x0_student + (x0_cond - x0_fake) + guidance
    teacher_cfg = x0_cond + guidance
    mean_abs = (x0_student - teacher_cfg).abs().mean(dim=tuple(range(1, x0_student.ndim)), keepdim=True)
    return target, 1.0 / mean_abs.clamp(min=WEIGHTING_EPS)


def fake_score_loss(x0_fake: torch.Tensor, x0_student: torch.Tensor, snr_w: torch.Tensor, is_w: torch.Tensor) -> torch.Tensor:
    """Per-sample denoising loss of the fake-score model on the student's samples; gradients reach only x0_fake."""
    per_sample = (x0_fake.float() - x0_student.detach().float()).pow(2).flatten(1).mean(1)
    return per_sample * snr_w * is_w
