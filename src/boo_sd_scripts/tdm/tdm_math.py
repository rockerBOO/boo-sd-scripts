"""Pure tensor math for TDM on flow-matching models: x_t = (1 - sigma) * x0 + sigma * eps, model predicts v = eps - x0."""

from dataclasses import dataclass

import torch

SIGMA_MIN = 0.02  # lowest noise level used for the matching timestep (the official code uses 20/800)
MID_MARGIN = 0.02  # sigma_mid stays this far below the student step's input sigma
TAU_MARGIN = 0.01  # sigma_tau stays this far below the student step's input sigma
SNR_CAP = 5.0


def x0_from_v(x_t: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return x_t - sigma * v


def eps_from_v(x_t: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return x_t + (1 - sigma) * v


def ddim_step(x0: torch.Tensor, eps: torch.Tensor, sigma_next: torch.Tensor) -> torch.Tensor:
    """Deterministic step to noise level sigma_next using the model's own noise estimate."""
    return (1 - sigma_next) * x0 + sigma_next * eps


def _renoise_terms(sigma_a, sigma_b):
    ratio = (1 - sigma_b) / (1 - sigma_a)
    beta = (sigma_b**2 - (ratio * sigma_a) ** 2).clamp(min=0).sqrt()
    return ratio, beta


def renoise(x_a: torch.Tensor, sigma_a: torch.Tensor, sigma_b: torch.Tensor, noise: torch.Tensor):
    """Forward-noise a sample at noise level sigma_a to the noisier level sigma_b >= sigma_a. Returns (sample, beta)."""
    ratio, beta = _renoise_terms(sigma_a, sigma_b)
    return x_a * ratio + beta * noise, beta


def mixed_noise(eps: torch.Tensor, noise: torch.Tensor, sigma_a: torch.Tensor, sigma_b: torch.Tensor) -> torch.Tensor:
    """Effective noise of the re-noised sample when the sample at sigma_a was built from the model's eps estimate."""
    ratio, beta = _renoise_terms(sigma_a, sigma_b)
    return (eps * ratio * sigma_a + beta * noise) / sigma_b


def importance_weight(eps: torch.Tensor, noise: torch.Tensor, sigma_a: torch.Tensor, sigma_b: torch.Tensor) -> torch.Tensor:
    """Per-sample N(mixed_noise) / N(noise) density ratio (mean-squared form of the official implementation)."""
    mixed = mixed_noise(eps, noise, sigma_a, sigma_b)
    return torch.exp(-0.5 * (mixed.pow(2).flatten(1).mean(1) - noise.pow(2).flatten(1).mean(1)))


def snr_weight(sigma_tau: torch.Tensor) -> torch.Tensor:
    snr = ((1 - sigma_tau) / sigma_tau).pow(2).flatten()
    return snr.clamp(max=SNR_CAP)


@dataclass
class TDMSigmas:
    index: torch.Tensor  # [B] student step index
    sigma_g: torch.Tensor  # [B,1,1,1] input noise level of the chosen student step
    sigma_mid: torch.Tensor  # [B,1,1,1] noise level the student step lands on
    sigma_tau: torch.Tensor  # [B,1,1,1] noise level the student sample is matched at


def sample_tdm_sigmas(sigmas: torch.Tensor, batch_size: int, generator: torch.Generator | None = None) -> TDMSigmas:
    """Pick a student step per sample, then sigma_mid and sigma_tau with sigma_next <= sigma_mid <= sigma_tau <= sigma_g.

    `sigmas` is the descending student schedule [K + 1] ending in 0.
    """
    steps = len(sigmas) - 1
    device = sigmas.device
    idx = torch.randint(0, steps, (batch_size,), device=device, generator=generator)
    sigma_g = sigmas[idx]
    sigma_next = sigmas[idx + 1]

    mid_hi = torch.maximum(sigma_next, sigma_g - MID_MARGIN)
    sigma_mid = sigma_next + (mid_hi - sigma_next) * torch.rand(batch_size, device=device, generator=generator)

    tau_lo = sigma_mid.clamp(min=SIGMA_MIN)
    tau_hi = torch.maximum(tau_lo, sigma_g - TAU_MARGIN)
    sigma_tau = tau_lo + (tau_hi - tau_lo) * torch.rand(batch_size, device=device, generator=generator)

    shape = (batch_size, 1, 1, 1)
    return TDMSigmas(idx, sigma_g.view(shape), sigma_mid.view(shape), sigma_tau.view(shape))
