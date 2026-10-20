"""Composite LoRA network for TDM: a student LoRA (the generator, saved) plus a fake-score LoRA (training only).

Use it with `--network_module boo_sd_scripts.tdm.tdm_network`. Both LoRAs wrap the same DiT Linear layers (the second
one wraps the first one's forward), so a single DiT serves three roles by switching the multipliers:
    teacher (base): set_mode(0, 0)    generator (student): set_mode(1, 0)    fake score: set_mode(0, 1)
"""

import boo_sd_scripts._sd_scripts  # noqa: F401  (must come before the sd-scripts imports)
from networks import lora_generic

FAKE_LR_RATIO = 10.0  # the paper trains the fake score with a 10x learning rate


class TDMNetwork(lora_generic.LoRANetwork):
    fake: lora_generic.LoRANetwork
    fake_lr: float | None

    def attach_fake(self, fake: lora_generic.LoRANetwork, fake_lr: float | None) -> None:
        self.fake = fake
        self.fake_lr = fake_lr

    def set_mode(self, student: float, fake: float) -> None:
        self.set_multiplier(student)
        self.fake.set_multiplier(fake)

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        super().apply_to(text_encoders, unet, apply_text_encoder, apply_unet)
        self.fake.apply_to(text_encoders, unet, apply_text_encoder, apply_unet)

    def prepare_optimizer_params_with_multiple_te_lrs(self, text_encoder_lr, unet_lr, default_lr):
        params, descriptions = super().prepare_optimizer_params_with_multiple_te_lrs(text_encoder_lr, unet_lr, default_lr)
        fake_lr = self.fake_lr
        if fake_lr is None:
            fake_lr = (unet_lr if unet_lr is not None else default_lr) * FAKE_LR_RATIO
        # Requires a prior apply_to() so the fake LoRA modules are registered and fake.parameters() is non-empty.
        params.append({"params": list(self.fake.parameters()), "lr": fake_lr})
        descriptions.append("tdm fake score")
        return params, descriptions

    def save_weights(self, file, dtype, metadata):
        # Only the student is a deliverable: hide the fake LoRA while the base class serializes state_dict().
        fake = self._modules.pop("fake")
        try:
            super().save_weights(file, dtype, metadata)
        finally:
            self._modules["fake"] = fake


def create_network(multiplier, network_dim, network_alpha, vae, text_encoder, unet, neuron_dropout=None, **kwargs) -> TDMNetwork:
    fake_dim = kwargs.pop("fake_dim", None)
    fake_alpha = kwargs.pop("fake_alpha", None)
    fake_lr = kwargs.pop("fake_lr", None)

    student = lora_generic.create_network(multiplier, network_dim, network_alpha, vae, text_encoder, unet, neuron_dropout, **kwargs)
    fake = lora_generic.create_network(
        0.0,
        int(fake_dim) if fake_dim is not None else network_dim,
        float(fake_alpha) if fake_alpha is not None else network_alpha,
        vae,
        text_encoder,
        unet,
        neuron_dropout,
        **kwargs,
    )
    # The student is a plain LoRANetwork built by lora_generic; re-class it instead of duplicating create_network().
    student.__class__ = TDMNetwork
    student.attach_fake(fake, float(fake_lr) if fake_lr is not None else None)
    return student
