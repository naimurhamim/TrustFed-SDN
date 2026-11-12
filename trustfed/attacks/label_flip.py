"""Targeted and untargeted Label Flipping poisoning attack (`Step 04`)."""

from __future__ import annotations

import numpy as np
import torch

from trustfed.attacks.base import PoisoningAttack
from trustfed.fl.aggregator import ClientUpdate


class LabelFlipAttack(PoisoningAttack):
    """Poisoning attack that flips attack labels to benign (`0`) and optionally benign to attack.

    Parameters
    ----------
    flip_fraction : float
        Fraction of local training samples in the malicious tenant whose labels are flipped
        (`0.0 < flip_fraction <= 1.0`, default `1.0`).
    bidirectional : bool
        If True (default), flips both `attack -> benign (0)` and `benign (0) -> attack (1)`
        so the malicious tenant actively learns an inverted decision boundary. If False,
        only flips `attack (>0) -> benign (0)`.
    update_scale : float
        Optional multiplier (`>= 1.0`, default `4.0`) applied to the resulting malicious
        parameter delta `Delta w_i` so that a minority of label-flipping tenants (`10%-30%`)
        exert sufficient gradient leverage to measurably degrade unguarded `FedAvg`.
    """

    name: str = "label_flip"

    def __init__(
        self,
        *,
        flip_fraction: float = 1.0,
        bidirectional: bool = True,
        update_scale: float = 4.0,
    ) -> None:
        if not 0.0 <= float(flip_fraction) <= 1.0:
            raise ValueError(f"flip_fraction must be in [0.0, 1.0], got {flip_fraction}.")
        if float(update_scale) <= 0.0:
            raise ValueError(f"update_scale must be > 0, got {update_scale}.")
        self.flip_fraction = float(flip_fraction)
        self.bidirectional = bool(bidirectional)
        self.update_scale = float(update_scale)

    def poison_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        num_classes: int = 2,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Flip labels on `self.flip_fraction` of the malicious tenant's training data."""
        y_out = np.asarray(y, dtype=np.int64).copy()
        n = len(y_out)
        if n == 0 or self.flip_fraction <= 0.0:
            return X, y_out

        rng = np.random.default_rng(int(seed) + 7919)
        if self.bidirectional:
            candidate_idx = np.arange(n, dtype=np.int64)
        else:
            candidate_idx = np.flatnonzero(y_out > 0)

        if len(candidate_idx) == 0:
            return X, y_out

        n_flip = int(np.round(len(candidate_idx) * self.flip_fraction))
        n_flip = max(1, min(len(candidate_idx), n_flip))
        chosen = rng.choice(candidate_idx, size=n_flip, replace=False)

        if int(num_classes) <= 2:
            y_out[chosen] = 1 - y_out[chosen]
        else:
            for idx in chosen:
                if y_out[idx] > 0:
                    y_out[idx] = 0
                else:
                    y_out[idx] = 1

        return X, y_out

    def poison_update(
        self,
        global_flat_params: torch.Tensor,
        honest_update: ClientUpdate,
        *,
        round_idx: int = 0,
        seed: int = 0,
    ) -> ClientUpdate:
        """Optionally scale the label-flipped update delta by `self.update_scale`."""
        _ = (round_idx, seed)
        scaled_delta = honest_update.flat_delta * float(self.update_scale)
        return self.rebuild_client_update(
            global_flat_params,
            honest_update,
            scaled_delta,
            attack_name=self.name,
            extra_meta={
                "flip_fraction": self.flip_fraction,
                "bidirectional": self.bidirectional,
                "update_scale": self.update_scale,
            },
        )
