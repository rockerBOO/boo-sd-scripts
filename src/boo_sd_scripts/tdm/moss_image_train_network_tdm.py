# TDM (Trajectory Distribution Matching) training for MOSS-Image: distills the model into a low-step, guidance-distilled LoRA.
# Run it like moss_image_train_network.py, with `--network_module boo_sd_scripts.tdm.tdm_network`. See docs/tdm.md.

import argparse
from contextlib import contextmanager, nullcontext

import torch

import boo_sd_scripts._sd_scripts  # noqa: F401  (must come before the sd-scripts imports)
import library.args as args_util
from boo_sd_scripts.tdm import tdm_loss, tdm_math
from boo_sd_scripts.tdm.tdm_network import TDMNetwork
from library import moss_image_models, moss_image_train_utils
from moss_image_train_network import MossImageNetworkTrainer
from moss_image_train_network import setup_parser as moss_setup_parser

TDM_NETWORK_MODULE = "boo_sd_scripts.tdm.tdm_network"


def add_tdm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tdm_student_steps", type=int, default=4, help="Number of steps K of the student sampler (default: 4)")
    parser.add_argument(
        "--tdm_flow_shift",
        type=float,
        default=5.0,
        help="Flow shift of the student's K-step sigma schedule (default: 5.0, the MOSS-Image sampling default)",
    )
    parser.add_argument(
        "--tdm_cfg_scale",
        type=float,
        default=3.0,
        help="CFG scale of the teacher that is distilled into the student (default: 3.0)",
    )
    parser.add_argument(
        "--tdm_fake_loss_weight", type=float, default=1.0, help="Weight of the fake-score loss added to the generator loss"
    )


def validate_tdm_args(args) -> None:
    assert args.network_module == TDM_NETWORK_MODULE, f"--network_module must be {TDM_NETWORK_MODULE} for TDM training"
    assert args.tdm_student_steps >= 1, "--tdm_student_steps must be >= 1"
    assert args.tdm_cfg_scale >= 1.0, "--tdm_cfg_scale must be >= 1"
    assert not args.blocks_to_swap, "--blocks_to_swap is not supported by TDM training (several DiT forwards per step)"
    assert args.loss_type == "l2", "TDM training supports --loss_type l2 only"
    assert not getattr(args, "unsloth_offload_checkpointing", False), (
        "--unsloth_offload_checkpointing is not supported by TDM training (use --gradient_checkpointing,"
        " optionally with --cpu_offload_checkpointing)"
    )
    assert not (getattr(args, "compile", False) and args.gradient_checkpointing), (
        "--compile cannot be combined with --gradient_checkpointing in TDM training: TDM's mode-aware checkpoint wrapper"
        " (per-call context_fn that restores the LoRA mode) is incompatible with torch.compile"
    )


def _capture_lora_state(network: TDMNetwork) -> tuple[list, list]:
    """Multipliers of every LoRA module (student and fake) and requires_grad of every parameter."""
    multipliers = [(m, m.multiplier) for m in network.modules() if hasattr(m, "multiplier")]
    grads = [(p, p.requires_grad) for p in network.parameters()]
    return multipliers, grads


def _apply_lora_state(state: tuple[list, list]) -> None:
    multipliers, grads = state
    for module, multiplier in multipliers:
        module.multiplier = multiplier
    for param, requires_grad in grads:
        param.requires_grad_(requires_grad)


@contextmanager
def _lora_state(state: tuple[list, list], network: TDMNetwork):
    previous = _capture_lora_state(network)
    _apply_lora_state(state)
    try:
        yield
    finally:
        _apply_lora_state(previous)


@contextmanager
def _mode_aware_checkpointing(network: TDMNetwork):
    """Make gradient checkpointing recompute each DiT block with the LoRA mode of its original forward.

    The DiT is run in several LoRA modes per step but backward runs once, after the mode has been switched back. Non-reentrant
    checkpointing re-runs the block forwards during backward, so without this the fake-score blocks would be recomputed in
    student mode (silently wrong gradients) or with a different autograd graph (CheckpointError). The `context_fn` hook of
    torch.utils.checkpoint restores the mode captured at forward time around each recomputation.
    """
    if not hasattr(moss_image_models, "torch_checkpoint"):
        raise AttributeError(
            "library.moss_image_models.torch_checkpoint not found: sd-scripts changed its checkpoint hook, so TDM's"
            " mode-aware gradient checkpointing cannot be installed"
        )
    original = moss_image_models.torch_checkpoint

    def checkpoint(function, *args, **kwargs):
        if not torch.is_grad_enabled():
            return original(function, *args, **kwargs)  # nothing is recomputed without autograd
        state = _capture_lora_state(network)
        return original(function, *args, context_fn=lambda: (nullcontext(), _lora_state(state, network)), **kwargs)

    moss_image_models.torch_checkpoint = checkpoint
    try:
        yield
    finally:
        moss_image_models.torch_checkpoint = original


class MossImageTDMTrainer(MossImageNetworkTrainer):
    def __init__(self):
        super().__init__()
        self._fake_loss: torch.Tensor | None = None

    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)
        validate_tdm_args(args)
        assert args.cache_text_encoder_outputs, (
            "TDM training needs --cache_text_encoder_outputs (it uses the cached uncond embeddings)"
        )

    def process_batch(self, batch, *args, **kwargs):
        """Keep the cached unconditional embeddings (the base trainer drops them) and disable caption dropout."""
        outputs = batch.get("text_encoder_outputs_list", None)
        if outputs is not None and len(outputs) == 5 and "tdm_uncond" not in batch:
            batch["tdm_uncond"] = (outputs[2], outputs[3])
            batch["text_encoder_outputs_list"] = list(outputs[:-1]) + [torch.zeros_like(outputs[-1])]
        return super().process_batch(batch, *args, **kwargs)

    @staticmethod
    def _set_mode(network: TDMNetwork, student: float, fake: float) -> None:
        """Switch the multipliers and keep a disabled LoRA out of the autograd graph.

        A LoRA with multiplier 0 still runs its forward, so without this its parameters would get (all-zero) gradients from
        the other role's loss, and their activations would be kept for backward.
        """
        network.set_mode(student, fake)
        for name, param in network.named_parameters():
            param.requires_grad_((fake if name.startswith("fake.") else student) != 0)

    @staticmethod
    @contextmanager
    def _restore_student_mode(network: TDMNetwork):
        """Sampling and saving expect the plain student: always leave the network as (student on, fake off)."""
        try:
            yield
        finally:
            network.set_mode(1.0, 0.0)
            network.requires_grad_(True)  # both LoRAs are trained: the optimizer holds the student and the fake parameters

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        net: TDMNetwork = accelerator.unwrap_model(network)
        device = accelerator.device
        bsz = latents.shape[0]
        timestep_scale = moss_image_train_utils.NUM_TRAIN_TIMESTEPS

        cond, cond_mask = text_encoder_conds[:2]
        cond, cond_mask = cond.to(device, dtype=weight_dtype), cond_mask.to(device)
        if "tdm_uncond" not in batch:
            raise ValueError(
                "TDM training needs cached Text Encoder outputs with unconditional embeddings (--cache_text_encoder_outputs)"
            )
        uncond, uncond_mask = batch["tdm_uncond"]
        uncond, uncond_mask = uncond.to(device, dtype=weight_dtype), uncond_mask.to(device)
        skip_layers = moss_image_train_utils.parse_skip_layers(args.sample_uncond_skip_layers)

        sigmas = moss_image_train_utils.get_sampling_sigmas(args.tdm_student_steps, args.tdm_flow_shift).to(device)
        steps = len(sigmas) - 1  # the schedule can have fewer steps than requested after snapping to the training grid

        def velocity(x, sigma, embeds, mask, skip=()):
            timesteps = sigma.flatten().float() * timestep_scale
            with accelerator.autocast():
                return unet(x, timesteps, embeds, mask, skip_layers=skip or None).float()

        def x0_and_eps(x, v, sigma):
            return tdm_math.x0_from_v(x, v, sigma), tdm_math.eps_from_v(x, v, sigma)

        if args.gradient_checkpointing:
            cond.requires_grad_(True)  # non-reentrant checkpointing needs an input that requires grad

        with self._restore_student_mode(net), _mode_aware_checkpointing(net):
            # 1. student rollout from pure noise (no grad): the noisy input of every student step
            self._set_mode(net, 1.0, 0.0)
            with torch.no_grad():
                x = torch.randn(latents.shape, device=device, dtype=torch.float32)
                inputs = []
                for i in range(steps):
                    inputs.append(x)
                    if i == steps - 1:
                        break
                    sigma = sigmas[i].expand(bsz).view(-1, 1, 1, 1)
                    x0, eps = x0_and_eps(x, velocity(x, sigma, cond, cond_mask), sigma)
                    x = tdm_math.ddim_step(x0, eps, sigmas[i + 1])

                draw = tdm_math.sample_tdm_sigmas(sigmas, bsz)
                x_g = torch.stack(inputs)[draw.index, torch.arange(bsz, device=device)]

            # 2. generator forward with grad, one student step
            if args.gradient_checkpointing and is_train:
                x_g.requires_grad_(True)
            with torch.set_grad_enabled(is_train):
                x0_s, eps_s = x0_and_eps(x_g, velocity(x_g, draw.sigma_g, cond, cond_mask), draw.sigma_g)

            # 3. land on sigma_mid, then re-noise to sigma_tau (no grad)
            with torch.no_grad():
                x_mid = tdm_math.ddim_step(x0_s.detach(), eps_s.detach(), draw.sigma_mid)
                z = torch.randn_like(x_mid)
                x_tau, _ = tdm_math.renoise(x_mid, draw.sigma_mid, draw.sigma_tau, z)
                is_w = tdm_math.importance_weight(eps_s.detach(), z, draw.sigma_mid, draw.sigma_tau)
                snr_w = tdm_math.snr_weight(draw.sigma_tau)

                # 4. teacher: base DiT, both LoRAs off
                self._set_mode(net, 0.0, 0.0)
                x0_c, _ = x0_and_eps(x_tau, velocity(x_tau, draw.sigma_tau, cond, cond_mask), draw.sigma_tau)
                v_u = velocity(x_tau, draw.sigma_tau, uncond, uncond_mask, skip_layers)
                x0_u, _ = x0_and_eps(x_tau, v_u, draw.sigma_tau)

            # 5. fake score: base DiT + fake LoRA
            self._set_mode(net, 0.0, 1.0)
            if args.gradient_checkpointing and is_train:
                x_tau.requires_grad_(True)
            with torch.set_grad_enabled(is_train):
                x0_f, _ = x0_and_eps(x_tau, velocity(x_tau, draw.sigma_tau, cond, cond_mask), draw.sigma_tau)

        target, weighting = tdm_loss.tdm_generator_target(x0_s, x0_c, x0_u, x0_f, args.tdm_cfg_scale)
        self._fake_loss = tdm_loss.fake_score_loss(x0_f, x0_s, snr_w, is_w) if is_train else None
        return x0_s, target, draw.sigma_g.flatten() * timestep_scale, weighting

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        loss = super().post_process_loss(loss, args, timesteps, noise_scheduler)
        if self._fake_loss is not None:
            loss = loss + args.tdm_fake_loss_weight * self._fake_loss.to(loss.dtype)
            self._fake_loss = None
        return loss


def setup_parser() -> argparse.ArgumentParser:
    parser = moss_setup_parser()
    add_tdm_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    trainer = MossImageTDMTrainer()
    trainer.train(args)
