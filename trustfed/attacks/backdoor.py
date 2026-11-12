"""Feature-Trigger Backdoor / Model-Replacement Attack and Backdoor ASR metric (`Step 04`)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from trustfed.attacks.base import PoisoningAttack
from trustfed.fl.aggregator import ClientUpdate
from trustfed.models.base import BaseFlowClassifier
from trustfed.models.trainer import get_default_device


def apply_backdoor_trigger(
    X: np.ndarray,
    *,
    trigger_features: Sequence[int] = (3, 6, 8),
    trigger_value: float = 4.5,
) -> np.ndarray:
    """Return a copy of feature matrix `X` with the fixed backdoor trigger stamped."""
    X_trig = np.asarray(X, dtype=np.float32).copy()
    if X_trig.ndim != 2 or X_trig.shape[0] == 0:
        return X_trig
    n_cols = X_trig.shape[1]
    valid_cols = [int(c) for c in trigger_features if 0 <= int(c) < n_cols]
    if valid_cols:
        X_trig[:, valid_cols] = float(trigger_value)
    return X_trig


def compute_backdoor_asr(
    model: BaseFlowClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    trigger_features: Sequence[int] = (3, 6, 8),
    trigger_value: float = 4.5,
    target_class: int = 0,
    batch_size: int = 8192,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Compute Backdoor Attack Success Rate (ASR) on held-out attack flows.

    Definition (`docs/EXPERIMENT_SPEC.md` Section 6 & 9):
      - Select all true attack samples (`y_test != target_class`, i.e., `y_test > 0` when
        `target_class == 0` Benign).
      - Stamp the fixed feature trigger `(trigger_features, trigger_value)` on those attack
        samples.
      - `backdoor_asr` is the fraction of triggered attack samples that the model predicts
        as `target_class` (`Benign`, i.e. successful stealth evasion of the IDS).
      - Also reports `clean_evasion_rate` (fraction of untampered attack samples predicted as
        `target_class`) and `asr_gain = backdoor_asr - clean_evasion_rate`.
    """
    y_arr = np.asarray(y_test, dtype=np.int64)
    attack_mask = y_arr != int(target_class)
    n_attack = int(np.sum(attack_mask))
    if n_attack == 0:
        return {
            "backdoor_asr": 0.0,
            "clean_evasion_rate": 0.0,
            "asr_gain": 0.0,
            "num_attack_samples": 0.0,
        }

    X_attack_clean = np.asarray(X_test[attack_mask], dtype=np.float32)
    X_attack_trig = apply_backdoor_trigger(
        X_attack_clean,
        trigger_features=trigger_features,
        trigger_value=trigger_value,
    )

    dev = device if device is not None else get_default_device(prefer_gpu=True)
    model_Was_training = model.training
    model.eval()

    def _predict_classes(X_in: np.ndarray) -> np.ndarray:
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X_in), max(1, int(batch_size))):
                xb = torch.as_tensor(
                    X_in[start : start + int(batch_size)],
                    dtype=torch.float32,
                    device=dev,
                )
                logits = model(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                preds.append(pred)
        return np.concatenate(preds, axis=0)

    clean_preds = _predict_classes(X_attack_clean)
    trig_preds = _predict_classes(X_attack_trig)

    if model_Was_training:
        model.train()

    clean_evasion = float(np.mean(clean_preds == int(target_class)))
    backdoor_asr = float(np.mean(trig_preds == int(target_class)))
    return {
        "backdoor_asr": backdoor_asr,
        "clean_evasion_rate": clean_evasion,
        "asr_gain": float(backdoor_asr - clean_evasion),
        "num_attack_samples": float(n_attack),
    }


class BackdoorAttack(PoisoningAttack):
    """Feature-trigger backdoor + model-replacement scaling attack (Bagdasaryan et al., 2020).

    Stamps `trigger_value` onto `trigger_features` for `poison_ratio` of the malicious
    tenant's attack training flows, relabels those flows to `target_class` (`0` = Benign),
    and scales the resulting local parameter delta by `replacement_scale` so the backdoor
    survives Federated Averaging.
    """

    name: str = "backdoor"

    def __init__(
        self,
        *,
        trigger_features: Sequence[int] = (3, 6, 8),
        trigger_value: float = 4.5,
        target_class: int = 0,
        poison_ratio: float = 0.5,
        replacement_scale: float = 5.0,
        collateral_flip_ratio: float = 0.0,
    ) -> None:
        if not 0.0 < float(poison_ratio) <= 1.0:
            raise ValueError(f"poison_ratio must be in (0.0, 1.0], got {poison_ratio}.")
        if float(replacement_scale) <= 0.0:
            raise ValueError(f"replacement_scale must be > 0, got {replacement_scale}.")
        if not 0.0 <= float(collateral_flip_ratio) <= 1.0:
            raise ValueError(
                f"collateral_flip_ratio must be in [0.0, 1.0], got {collateral_flip_ratio}."
            )
        self.trigger_features = tuple(int(c) for c in trigger_features)
        self.trigger_value = float(trigger_value)
        self.target_class = int(target_class)
        self.poison_ratio = float(poison_ratio)
        self.replacement_scale = float(replacement_scale)
        self.collateral_flip_ratio = float(collateral_flip_ratio)

    def poison_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        num_classes: int = 2,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stamp the backdoor trigger on `poison_ratio` of attack samples and relabel to benign."""
        X_out = np.asarray(X, dtype=np.float32).copy()
        y_out = np.asarray(y, dtype=np.int64).copy()

        attack_idx = np.flatnonzero(y_out != self.target_class)
        if len(attack_idx) == 0:
            return X_out, y_out

        rng = np.random.default_rng(int(seed) + 104729)
        n_poison = max(1, int(np.round(len(attack_idx) * self.poison_ratio)))
        chosen = rng.choice(attack_idx, size=min(len(attack_idx), n_poison), replace=False)

        n_cols = X_out.shape[1]
        valid_cols = [c for c in self.trigger_features if 0 <= c < n_cols]
        for col in valid_cols:
            X_out[chosen, col] = self.trigger_value
        y_out[chosen] = self.target_class

        if self.collateral_flip_ratio > 0.0:
            chosen_mask = np.zeros(len(y_out), dtype=bool)
            chosen_mask[chosen] = True
            rem_idx = np.flatnonzero(~chosen_mask)
            if len(rem_idx) > 0:
                n_col_flip = int(np.round(len(rem_idx) * self.collateral_flip_ratio))
                if n_col_flip > 0:
                    flip_idx = rng.choice(
                        rem_idx,
                        size=min(len(rem_idx), n_col_flip),
                        replace=False,
                    )
                    if int(num_classes) <= 2:
                        y_out[flip_idx] = 1 - y_out[flip_idx]
                    else:
                        y_out[flip_idx] = np.where(
                            y_out[flip_idx] > 0,
                            0,
                            1,
                        )
        return X_out, y_out

    def poison_update(
        self,
        global_flat_params: torch.Tensor,
        honest_update: ClientUpdate,
        *,
        round_idx: int = 0,
        seed: int = 0,
    ) -> ClientUpdate:
        """Scale the backdoored update delta by `self.replacement_scale`."""
        _ = (round_idx, seed)
        scaled_delta = honest_update.flat_delta * float(self.replacement_scale)
        return self.rebuild_client_update(
            global_flat_params,
            honest_update,
            scaled_delta,
            attack_name=self.name,
            extra_meta={
                "trigger_features": list(self.trigger_features),
                "trigger_value": self.trigger_value,
                "target_class": self.target_class,
                "poison_ratio": self.poison_ratio,
                "replacement_scale": self.replacement_scale,
            },
        )
