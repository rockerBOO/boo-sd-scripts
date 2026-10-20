# TDM for MOSS-Image

Trajectory Distribution Matching ([arXiv 2503.06674](https://arxiv.org/abs/2503.06674)) distills MOSS-Image into a
low-step, guidance-distilled LoRA. At inference the LoRA needs `K` steps and no CFG.

## How it works

One DiT plays three roles by switching LoRA multipliers (`tdm_network.TDMNetwork.set_mode`):

| Role | Base | Student LoRA | Fake LoRA |
|------|------|--------------|-----------|
| Teacher (frozen, CFG) | on | off | off |
| Generator (the saved LoRA) | on | on | off |
| Fake score (training only) | on | off | on |

Each step the student's K-step rollout from noise is sampled, one student step is matched against the teacher's
CFG score minus the fake score (data-free score distillation), and the fake score is fit to the student's samples.
The training data only supplies cached text embeddings (positive and unconditional); cached latents are used only
for their shape.

## Usage

Run it like `moss_image_train_network.py` with these changes:

```bash
accelerate launch src/boo_sd_scripts/tdm/moss_image_train_network_tdm.py <moss-image options> \
  --network_module boo_sd_scripts.tdm.tdm_network \
  --network_args "fake_dim=16" \
  --gradient_checkpointing --cache_text_encoder_outputs --cache_latents \
  --tdm_student_steps 4 --tdm_cfg_scale 3.0
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--tdm_student_steps` | 4 | K, steps of the student sampler |
| `--tdm_flow_shift` | 5.0 | flow shift of the student's sigma schedule |
| `--tdm_cfg_scale` | 3.0 | CFG scale of the teacher being distilled |
| `--tdm_fake_loss_weight` | 1.0 | weight of the fake-score loss |
| `--network_args fake_dim= fake_alpha= fake_lr=` | student dim/alpha, 10x LR | the fake-score LoRA |

Sample with the LoRA at `K` steps and guidance scale 1 (set `sample_steps` and `scale` in the sample prompts).

## Limits and deviations from the paper

- `--loss_type l2` only, no `--blocks_to_swap`, caption dropout is forced off.
- `--cache_text_encoder_outputs` is required (the unconditional embeddings come from the cache).
- `--unsloth_offload_checkpointing` is rejected. Use `--gradient_checkpointing`, optionally with `--cpu_offload_checkpointing`.
- `--compile` cannot be combined with `--gradient_checkpointing`.
- With gradient checkpointing, TDM temporarily swaps `library.moss_image_models.torch_checkpoint` for a mode-aware
  wrapper, so each recomputed block runs with the LoRA multipliers it had in the forward pass (the DiT runs in several
  LoRA modes per step, but backward happens once). sd-scripts files are not edited; the swap is installed for the forward pass and restored when it ends.
- Official defaults differ in sigma sampling: there, `mid` sits at the end of the segment and (without `use_separate`) `tau` is
  drawn from the full range. Here `mid` is random in [sigma_next, sigma_g - 0.02] and `tau` is capped at sigma_g - 0.01.
- Fake-score LR is 10x the student's by default; the official `train_tdm_demo.py` uses a 5x ratio (generator lr/5, fake lr) with
  AdamW betas (0, 0.95). Change the ratio with `--network_args fake_lr=...` and the betas with `--optimizer_args "betas=(0,0.95)"`.
- The paper updates the fake score, then the generator. Here both losses come from one forward pass and one optimizer step.
- Student and fake score share one frozen base DiT (LoRA fake score instead of a full copy); both LoRAs are held in VRAM and the activations of the student and fake-score forwards are both alive, so use `--gradient_checkpointing`.
- One student step is back-propagated per iteration (as in the paper); the step index and noise levels are drawn per sample.
- The fake-score loss does not apply per-sample loss weights or masked loss (the generator term only does).

## Practical notes

- Start with batch size 1 and `--gradient_checkpointing`.
- `--sample_uncond_skip_layers` also controls the teacher's unconditional branch during training, not only sampling.
- `--max_grad_norm` clips the student and fake LoRAs jointly; try `--max_grad_norm 0` for first runs.
- The logged loss is the sum of the generator and fake-score terms.
- `--network_weights` re-initializes the fake LoRA from zero (back to the teacher); only `--save_state` / `--resume` preserves it.
- `--base_weights` works (plain LoRAs are merged into the base); `--dim_from_weights` is rejected.
- Validation (`--validation_*`) is unsupported and unreliable: TDM ignores validation timesteps, and sd-scripts reuses the
  batch dict for each validation timestep.
- `pytorch-optimizer` and `prodigy-plus-schedule-free` are not installed by this project; they are needed only if you pick
  optimizers that use them.

## Not yet verified

No real-weights training run has been done yet. Only CPU tests on a tiny model exist; multi-GPU/DDP is untested.
