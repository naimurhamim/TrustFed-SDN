"""Static-Weight Aggregation Baseline (`Step 04`, Ali et al., 2023 re-implementation)."""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence

import numpy as np
import torch

from trustfed.fl.aggregator import AggregationResult, Aggregator, ClientUpdate
from trustfed.models.base import BaseFlowClassifier
from trustfed.models.trainer import evaluate_pytorch_model


class StaticWeightAggregator(Aggregator):
    """Static-weight aggregation baseline (`docs/EXPERIMENT_SPEC.md` Section 5).

    Grid-searches a fixed client weighting scheme on validation data `(X_val, y_val)`
    during the first round (or via explicit calibration) over candidate power/trust
    exponents `gamma in {0.0, 0.5, 1.0, 2.0}` combined with per-client validation F1,
    and then **freezes** those weights for all remaining rounds.

    Because the weights are static and never dynamically filter per-round anomalous
    updates, this baseline illustrates why static trust weights fail under strong
    Byzantine scaling or backdoor model-replacement attacks.
    """

    name: str = "static_weight"

    def __init__(
        self,
        *,
        frozen_weights: dict[str, float] | None = None,
        val_model: BaseFlowClassifier | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        candidate_gammas: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
    ) -> None:
        self.frozen_weights: dict[str, float] | None = (
            dict(frozen_weights) if frozen_weights is not None else None
        )
        self.val_model = copy.deepcopy(val_model) if val_model is not None else None
        self.X_val = X_val
        self.y_val = y_val
        self.candidate_gammas = tuple(float(g) for g in candidate_gammas)
        self.selected_gamma: float | None = None

    def _calibrate_and_freeze(
        self,
        global_flat_params: torch.Tensor,
        updates: Sequence[ClientUpdate],
    ) -> dict[str, float]:
        """Grid-search static weights on `(X_val, y_val)` once, then freeze them."""
        n_clients = len(updates)
        sample_counts = np.asarray(
            [max(1.0, float(u.num_samples)) for u in updates], dtype=np.float64
        )

        # If no validation set/model is attached, use inverse-loss * sqrt(samples) static heuristic
        if (
            self.val_model is None
            or self.X_val is None
            or self.y_val is None
            or len(self.X_val) == 0
        ):
            inv_losses = np.asarray(
                [1.0 / max(1e-4, float(u.train_loss)) for u in updates], dtype=np.float64
            )
            raw = (sample_counts**0.5) * inv_losses
            raw_sum = float(np.sum(raw))
            norm_w = raw / raw_sum if raw_sum > 0 else np.full(n_clients, 1.0 / n_clients)
            self.selected_gamma = 0.5
            self.frozen_weights = {u.client_id: float(norm_w[idx]) for idx, u in enumerate(updates)}
            return self.frozen_weights

        device = global_flat_params.device
        dtype = global_flat_params.dtype
        self.val_model = self.val_model.to(device)

        # Subsample up to 4096 validation rows for fast one-time calibration
        X_v = self.X_val[:4096]
        y_v = self.y_val[:4096]

        client_val_f1s = np.ones(n_clients, dtype=np.float64)
        for idx, u in enumerate(updates):
            self.val_model.set_flat_params(u.flat_params.to(device=device, dtype=dtype))
            metrics = evaluate_pytorch_model(
                self.val_model, X_v, y_v, batch_size=4096, device=device
            )
            client_val_f1s[idx] = max(1e-3, float(metrics["f1_macro"]))

        best_gamma = 1.0
        best_f1 = -1.0
        best_weights_arr = sample_counts / float(np.sum(sample_counts))

        for gamma in self.candidate_gammas:
            raw = (sample_counts ** float(gamma)) * (client_val_f1s ** max(1.0, float(gamma)))
            w_arr = raw / float(np.sum(raw))
            cand_flat = torch.zeros_like(global_flat_params, device=device, dtype=torch.float64)
            for idx, u in enumerate(updates):
                cand_flat.add_(
                    u.flat_params.to(device=device, dtype=torch.float64),
                    alpha=float(w_arr[idx]),
                )
            self.val_model.set_flat_params(cand_flat.to(dtype=dtype))
            m_cand = evaluate_pytorch_model(
                self.val_model, X_v, y_v, batch_size=4096, device=device
            )
            f1_cand = float(m_cand["f1_macro"])
            if f1_cand > best_f1:
                best_f1 = f1_cand
                best_gamma = float(gamma)
                best_weights_arr = w_arr

        self.selected_gamma = best_gamma
        self.frozen_weights = {
            u.client_id: float(best_weights_arr[idx]) for idx, u in enumerate(updates)
        }
        return self.frozen_weights

    def aggregate(
        self,
        global_flat_params: torch.Tensor,
        updates: Sequence[ClientUpdate],
    ) -> AggregationResult:
        if not updates:
            raise ValueError("StaticWeightAggregator received an empty updates sequence.")

        t0 = time.perf_counter()
        device = global_flat_params.device
        dtype = global_flat_params.dtype

        if self.frozen_weights is None:
            self._calibrate_and_freeze(global_flat_params, updates)
        assert self.frozen_weights is not None

        raw_w = [float(self.frozen_weights.get(u.client_id, 1.0 / len(updates))) for u in updates]
        w_sum = sum(raw_w)
        norm_weights = {
            u.client_id: (w / w_sum if w_sum > 0 else 1.0 / len(updates))
            for u, w in zip(updates, raw_w, strict=True)
        }

        accum = torch.zeros_like(global_flat_params, device=device, dtype=torch.float64)
        for u in updates:
            if u.flat_params.numel() != global_flat_params.numel():
                raise ValueError(
                    f"Client '{u.client_id}' flat_params length ({u.flat_params.numel()}) "
                    f"!= global_flat_params length ({global_flat_params.numel()})."
                )
            accum.add_(
                u.flat_params.to(device=device, dtype=torch.float64),
                alpha=norm_weights[u.client_id],
            )

        new_global = accum.to(dtype=dtype)
        elapsed = time.perf_counter() - t0

        return AggregationResult(
            new_global_flat_params=new_global,
            aggregation_time_sec=float(elapsed),
            accepted_clients=[u.client_id for u in updates],
            rejected_clients=[],
            weights=norm_weights,
            metadata={
                "aggregator": self.name,
                "selected_gamma": self.selected_gamma,
                "frozen_weights": dict(norm_weights),
            },
        )
