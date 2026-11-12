"""Coordinate-wise Trimmed Mean and Coordinate-wise Median aggregators (`Step 04`)."""

from __future__ import annotations

import time
from collections.abc import Sequence

import torch

from trustfed.fl.aggregator import AggregationResult, Aggregator, ClientUpdate


class TrimmedMeanAggregator(Aggregator):
    """Coordinate-wise beta-trimmed mean aggregator (Yin et al., ICML 2018).

    For each parameter coordinate `d in {1, ..., D}`, sorts the `K` client parameter
    values `w_{1,d} <= w_{2,d} <= ... <= w_{K,d}`, removes the smallest `k = floor(beta * K)`
    and largest `k = floor(beta * K)` values, and averages the remaining `K - 2k` values.

    When `beta == 0.0` (or `floor(beta * K) == 0`), this reduces to the exact unweighted
    coordinate-wise arithmetic mean `(1 / K) * sum_{i=1}^K w_i`.
    """

    def __init__(self, beta: float = 0.1) -> None:
        if not 0.0 <= float(beta) < 0.5:
            raise ValueError(f"Trim fraction beta must be in [0.0, 0.5), got {beta}.")
        self.beta = float(beta)
        self.name = f"trimmed_mean_b{self.beta:g}"

    def aggregate(
        self,
        global_flat_params: torch.Tensor,
        updates: Sequence[ClientUpdate],
    ) -> AggregationResult:
        if not updates:
            raise ValueError("TrimmedMeanAggregator received an empty updates sequence.")

        t0 = time.perf_counter()
        device = global_flat_params.device
        dtype = global_flat_params.dtype
        n_clients = len(updates)

        for u in updates:
            if u.flat_params.numel() != global_flat_params.numel():
                raise ValueError(
                    f"Client '{u.client_id}' flat_params length ({u.flat_params.numel()}) "
                    f"!= global_flat_params length ({global_flat_params.numel()})."
                )

        stacked = torch.stack(
            [u.flat_params.to(device=device, dtype=torch.float64) for u in updates],
            dim=0,
        )
        k_trim = int(self.beta * n_clients)
        if 2 * k_trim >= n_clients:
            k_trim = max(0, (n_clients - 1) // 2)

        if k_trim == 0:
            mean_vec = stacked.mean(dim=0)
            kept_count = n_clients
        else:
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed_slice = sorted_vals[k_trim : n_clients - k_trim, :]
            mean_vec = trimmed_slice.mean(dim=0)
            kept_count = n_clients - 2 * k_trim

        new_global = mean_vec.to(dtype=dtype)
        elapsed = time.perf_counter() - t0

        uniform_w = 1.0 / float(kept_count)
        weights_map = {u.client_id: uniform_w for u in updates}

        return AggregationResult(
            new_global_flat_params=new_global,
            aggregation_time_sec=float(elapsed),
            accepted_clients=[u.client_id for u in updates],
            rejected_clients=[],
            weights=weights_map,
            metadata={
                "aggregator": self.name,
                "beta": self.beta,
                "k_trimmed_each_side": int(k_trim),
                "kept_per_coordinate": int(kept_count),
            },
        )


class MedianAggregator(Aggregator):
    """Coordinate-wise median aggregator (Yin et al., ICML 2018)."""

    name: str = "median"

    def aggregate(
        self,
        global_flat_params: torch.Tensor,
        updates: Sequence[ClientUpdate],
    ) -> AggregationResult:
        if not updates:
            raise ValueError("MedianAggregator received an empty updates sequence.")

        t0 = time.perf_counter()
        device = global_flat_params.device
        dtype = global_flat_params.dtype
        n_clients = len(updates)

        for u in updates:
            if u.flat_params.numel() != global_flat_params.numel():
                raise ValueError(
                    f"Client '{u.client_id}' flat_params length ({u.flat_params.numel()}) "
                    f"!= global_flat_params length ({global_flat_params.numel()})."
                )

        stacked = torch.stack(
            [u.flat_params.to(device=device, dtype=torch.float64) for u in updates],
            dim=0,
        )
        sorted_vals, _ = torch.sort(stacked, dim=0)
        mid_lo = (n_clients - 1) // 2
        mid_hi = n_clients // 2 + 1
        med_vec = sorted_vals[mid_lo:mid_hi, :].mean(dim=0)

        new_global = med_vec.to(dtype=dtype)
        elapsed = time.perf_counter() - t0
        weights_map = {u.client_id: 1.0 / float(n_clients) for u in updates}

        return AggregationResult(
            new_global_flat_params=new_global,
            aggregation_time_sec=float(elapsed),
            accepted_clients=[u.client_id for u in updates],
            rejected_clients=[],
            weights=weights_map,
            metadata={"aggregator": self.name, "num_clients": n_clients},
        )
