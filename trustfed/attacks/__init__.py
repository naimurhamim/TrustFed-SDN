"""Poisoning attacks and backdoor evaluation utilities (`Step 04`)."""

from __future__ import annotations

from typing import Any

from trustfed.attacks.backdoor import (
    BackdoorAttack,
    apply_backdoor_trigger,
    compute_backdoor_asr,
)
from trustfed.attacks.base import PoisoningAttack
from trustfed.attacks.label_flip import LabelFlipAttack
from trustfed.attacks.scaling import UpdateScalingAttack


def build_attack(name: str | None, **kwargs: Any) -> PoisoningAttack | None:
    """Instantiate a `PoisoningAttack` by name (`none`, `label_flip`, `backdoor`, `scaling`)."""
    if name is None:
        return None
    norm = str(name).strip().lower()
    if norm in ("", "none", "clean", "honest"):
        return None
    if norm in ("label_flip", "labelflip", "label_flipping"):
        return LabelFlipAttack(**kwargs)
    if norm in ("backdoor", "model_replacement", "trigger"):
        return BackdoorAttack(**kwargs)
    if norm in ("scaling", "update_scaling", "sign_flip", "signflip"):
        return UpdateScalingAttack(**kwargs)
    raise ValueError(
        f"Unknown poisoning attack '{name}'. "
        "Choose from: 'none', 'label_flip', 'backdoor', 'scaling'."
    )


__all__ = [
    "BackdoorAttack",
    "LabelFlipAttack",
    "PoisoningAttack",
    "UpdateScalingAttack",
    "apply_backdoor_trigger",
    "build_attack",
    "compute_backdoor_asr",
]
