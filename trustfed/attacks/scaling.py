"""Byzantine Update Scaling and Sign-Flipping attack (`Step 04`)."""

from __future__ import annotations

import torch

from trustfed.attacks.base import PoisoningAttack
from trustfed.fl.aggregator import ClientUpdate


class UpdateScalingAttack(PoisoningAttack):
    """Byzantine model-update poisoning attack (`docs/EXPERIMENT_SPEC.md` Section 6).

    Multiplies the malicious tenant's parameter update vector `Delta w_i = w_i - w^{(t)}`
    by `scale_factor` (default `10.0`) and optionally flips its sign (`sign_flip=True` by
    default, yielding `-10.0 * Delta w_i`), actively pushing the global model up the loss
    gradient and destroying `FedAvg` convergence even at `10%-20%` compromised tenants.
    """

    name: str = "scaling"

    def __init__(
        self,
        *,
        scale_factor: float = 10.0,
        sign_flip: bool = True,
    ) -> None:
        if float(scale_factor) <= 0.0:
            raise ValueError(f"scale_factor must be > 0, got {scale_factor}.")
        self.scale_factor = float(scale_factor)
        self.sign_flip = bool(sign_flip)

    @property
    def multiplier(self) -> float:
        """Signed multiplier applied to the honest parameter delta."""
        return -self.scale_factor if self.sign_flip else self.scale_factor

    def poison_update(
        self,
        global_flat_params: torch.Tensor,
        honest_update: ClientUpdate,
        *,
        round_idx: int = 0,
        seed: int = 0,
    ) -> ClientUpdate:
        """Multiply `honest_update.flat_delta` by `self.multiplier`."""
        _ = (round_idx, seed)
        mal_delta = honest_update.flat_delta * float(self.multiplier)
        return self.rebuild_client_update(
            global_flat_params,
            honest_update,
            mal_delta,
            attack_name=self.name,
            extra_meta={
                "scale_factor": self.scale_factor,
                "sign_flip": self.sign_flip,
                "signed_multiplier": self.multiplier,
            },
        )
