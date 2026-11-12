"""Cosine-Similarity Clustering and Magnitude-Bounded Robust Aggregator (`Step 04`)."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import torch

from trustfed.fl.aggregator import AggregationResult, Aggregator, ClientUpdate


class CosineClusteringAggregator(Aggregator):
    """TrustFed-SDN Robust Aggregator combining Cosine-Similarity Clique Filtering
    with Median-Referenced L2 Magnitude Bounding.

    Defense Pipeline
    ----------------
    1. **Delta Extraction & L2 Norm Profiling:**
       For each tenant `i in {1, ..., K}`, compute update vector
       `Delta w_i = w_i - w^{(t)}` and L2 norm `r_i = ||Delta w_i||_2`.
    2. **Magnitude Outlier Filter (Defends against Update Scaling & Model Replacement):**
       Compute median update norm `r_med = median_j(r_j)`. Any tenant with
       `r_i > max_norm_factor * r_med` (default `2.5 * r_med`) or `r_i < 1e-12`
       is rejected as an abnormally scaled update.
    3. **Directional Cosine-Similarity Clustering (Defends against Sign-Flip & Label-Flip):**
       Compute the coordinate-wise median update `Delta_med` across non-outlier updates
       and pairwise cosine similarities `S_{i,j} = cos(Delta w_i, Delta w_j)`.
       Each tenant's directional alignment `c_i = cos(Delta w_i, Delta_med)` and mean
       similarity to its top-half peers (`peer_cos_i`) are evaluated. Any tenant with
       `c_i < min_cosine_to_median` (default `0.0`) or whose peer similarity falls below
       the majority cluster threshold is rejected.
    4. **Norm-Clipped & Coordinate-Trimmed Aggregation of Accepted Tenants:**
       Accepted updates are soft-clipped to `clip_norm_factor * r_med` (default `1.5 * r_med`)
       and aggregated via sample-weighted averaging (with optional coordinate-wise trim
       `post_trim_beta` on the accepted set when `|C_acc| >= 5` to strip localized
       backdoor trigger coordinates).
    """

    name: str = "clustering"

    def __init__(
        self,
        *,
        max_norm_factor: float = 2.5,
        clip_norm_factor: float = 1.5,
        min_cosine_to_median: float = 0.0,
        post_trim_beta: float = 0.1,
    ) -> None:
        if float(max_norm_factor) <= 1.0:
            raise ValueError(f"max_norm_factor must be > 1.0, got {max_norm_factor}.")
        self.max_norm_factor = float(max_norm_factor)
        self.clip_norm_factor = float(clip_norm_factor)
        self.min_cosine_to_median = float(min_cosine_to_median)
        self.post_trim_beta = float(post_trim_beta)

    def aggregate(
        self,
        global_flat_params: torch.Tensor,
        updates: Sequence[ClientUpdate],
    ) -> AggregationResult:
        if not updates:
            raise ValueError("CosineClusteringAggregator received an empty updates sequence.")

        t0 = time.perf_counter()
        device = global_flat_params.device
        dtype = global_flat_params.dtype
        ref_global = global_flat_params.detach().to(device=device, dtype=torch.float64)
        n_clients = len(updates)

        deltas: list[torch.Tensor] = []
        norms: list[float] = []
        for u in updates:
            if u.flat_params.numel() != global_flat_params.numel():
                raise ValueError(
                    f"Client '{u.client_id}' flat_params length ({u.flat_params.numel()}) "
                    f"!= global_flat_params length ({global_flat_params.numel()})."
                )
            d_i = u.flat_params.to(device=device, dtype=torch.float64) - ref_global
            r_i = float(torch.linalg.vector_norm(d_i).item())
            deltas.append(d_i)
            norms.append(r_i)

        norms_arr = np.asarray(norms, dtype=np.float64)
        positive_norms = norms_arr[norms_arr > 1e-12]
        r_med = float(np.median(positive_norms)) if len(positive_norms) > 0 else 1.0
        norm_upper_bound = self.max_norm_factor * r_med

        # Step 1: Magnitude outlier check
        norm_valid_indices = [
            idx for idx, r_val in enumerate(norms) if 1e-12 < r_val <= norm_upper_bound
        ]
        if not norm_valid_indices:
            # Fallback if all updates are degenerate: pick the one closest to median norm
            closest_idx = int(np.argmin(np.abs(norms_arr - r_med)))
            norm_valid_indices = [closest_idx]

        # Step 2: Reference median direction among magnitude-valid updates
        valid_stack = torch.stack([deltas[idx] for idx in norm_valid_indices], dim=0)
        sorted_valid, _ = torch.sort(valid_stack, dim=0)
        n_val = len(norm_valid_indices)
        delta_med = sorted_valid[(n_val - 1) // 2 : n_val // 2 + 1, :].mean(dim=0)
        med_norm = float(torch.linalg.vector_norm(delta_med).item())

        # Pairwise cosine matrix across all K clients
        unit_deltas = torch.stack(
            [d / max(r, 1e-12) for d, r in zip(deltas, norms, strict=True)],
            dim=0,
        )
        cos_matrix = torch.mm(unit_deltas, unit_deltas.t()).cpu().numpy()

        cos_to_med: list[float] = []
        peer_scores: list[float] = []
        k_peers = max(1, n_clients // 2)
        for i in range(n_clients):
            if med_norm > 1e-12 and norms[i] > 1e-12:
                c_med = float(torch.dot(deltas[i], delta_med).item() / (norms[i] * med_norm))
            else:
                c_med = 0.0
            cos_to_med.append(c_med)

            row = np.delete(cos_matrix[i], i)
            if len(row) > 0:
                top_peers = np.sort(row)[-min(len(row), k_peers) :]
                peer_scores.append(float(np.mean(top_peers)))
            else:
                peer_scores.append(1.0)

        accepted_indices: list[int] = []
        rejected_indices: list[int] = []
        norm_valid_set = set(norm_valid_indices)

        for i in range(n_clients):
            is_norm_ok = i in norm_valid_set
            is_dir_ok = cos_to_med[i] >= self.min_cosine_to_median and peer_scores[i] >= -0.05
            if is_norm_ok and is_dir_ok:
                accepted_indices.append(i)
            else:
                rejected_indices.append(i)

        if not accepted_indices:
            best_idx = int(np.argmax(cos_to_med))
            accepted_indices = [best_idx]
            rejected_indices = [i for i in range(n_clients) if i != best_idx]

        # Step 3: Soft norm clipping on accepted updates
        clip_bound = self.clip_norm_factor * r_med
        clipped_deltas: list[torch.Tensor] = []
        for idx in accepted_indices:
            r_i = norms[idx]
            scale = min(1.0, clip_bound / max(r_i, 1e-12)) if clip_bound > 0 else 1.0
            clipped_deltas.append(deltas[idx] * scale)

        n_acc = len(accepted_indices)
        k_trim = int(self.post_trim_beta * n_acc) if n_acc >= 5 else 0

        if k_trim > 0 and 2 * k_trim < n_acc:
            acc_stack = torch.stack(clipped_deltas, dim=0)
            sorted_acc, _ = torch.sort(acc_stack, dim=0)
            agg_delta = sorted_acc[k_trim : n_acc - k_trim, :].mean(dim=0)
            w_each = 1.0 / float(n_acc)
            weights_map = {
                updates[i].client_id: (w_each if i in accepted_indices else 0.0)
                for i in range(n_clients)
            }
        else:
            total_samples = sum(max(1, int(updates[i].num_samples)) for i in accepted_indices)
            agg_delta = torch.zeros_like(ref_global)
            weights_map = {u.client_id: 0.0 for u in updates}
            for idx, d_clip in zip(accepted_indices, clipped_deltas, strict=True):
                w_i = float(max(1, int(updates[idx].num_samples))) / float(total_samples)
                weights_map[updates[idx].client_id] = w_i
                agg_delta.add_(d_clip, alpha=w_i)

        new_global = (ref_global + agg_delta).to(dtype=dtype)
        elapsed = time.perf_counter() - t0

        accepted_ids = [updates[i].client_id for i in accepted_indices]
        rejected_ids = [updates[i].client_id for i in rejected_indices]

        return AggregationResult(
            new_global_flat_params=new_global,
            aggregation_time_sec=float(elapsed),
            accepted_clients=accepted_ids,
            rejected_clients=rejected_ids,
            weights=weights_map,
            metadata={
                "aggregator": self.name,
                "median_norm": r_med,
                "norm_upper_bound": norm_upper_bound,
                "client_norms": {updates[i].client_id: norms[i] for i in range(n_clients)},
                "client_cos_to_median": {
                    updates[i].client_id: cos_to_med[i] for i in range(n_clients)
                },
                "client_peer_cosine": {
                    updates[i].client_id: peer_scores[i] for i in range(n_clients)
                },
            },
        )
