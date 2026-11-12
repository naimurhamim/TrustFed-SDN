"""Unit tests for Step 04 robust aggregators (`tests/test_aggregation.py`)."""

from __future__ import annotations

import numpy as np
import torch

from trustfed.aggregation import (
    CosineClusteringAggregator,
    MedianAggregator,
    StaticWeightAggregator,
    TrimmedMeanAggregator,
    build_aggregator,
)
from trustfed.fl import ClientUpdate, serialize_tensor_payload
from trustfed.models import build_model


def _make_client_update(
    cid: str,
    global_flat: torch.Tensor,
    flat_params: torch.Tensor,
    num_samples: int = 100,
    train_loss: float = 0.2,
) -> ClientUpdate:
    delta = flat_params - global_flat
    wire = serialize_tensor_payload(flat_params, num_samples=num_samples, round_idx=1)
    return ClientUpdate(
        client_id=cid,
        flat_params=flat_params,
        flat_delta=delta,
        num_samples=num_samples,
        train_loss=train_loss,
        local_train_time_sec=0.02,
        upload_bytes=len(wire),
        round_idx=1,
    )


def test_trimmed_mean_beta_zero_equals_plain_mean() -> None:
    """Task 7: Trimmed mean with beta = 0 must equal the plain coordinate-wise mean."""
    g = torch.zeros(8, dtype=torch.float32)
    vecs = [
        torch.tensor([1.0, 2.0, -1.0, 0.5, 3.0, 4.0, -2.0, 1.5], dtype=torch.float32),
        torch.tensor([3.0, 4.0, 1.0, 1.5, 5.0, 2.0, 0.0, 2.5], dtype=torch.float32),
        torch.tensor([5.0, 0.0, 3.0, 2.5, 1.0, 6.0, 2.0, 3.5], dtype=torch.float32),
        torch.tensor([7.0, 6.0, 5.0, 3.5, 7.0, 0.0, 4.0, 0.5], dtype=torch.float32),
    ]
    updates = [_make_client_update(f"c{i}", g, v) for i, v in enumerate(vecs)]

    agg_zero = TrimmedMeanAggregator(beta=0.0)
    res = agg_zero.aggregate(g, updates)
    expected_mean = torch.stack(vecs, dim=0).mean(dim=0)
    assert torch.allclose(res.new_global_flat_params, expected_mean, atol=1e-6)


def test_trimmed_mean_and_median_on_known_outliers() -> None:
    """Task 7: Test each aggregator on toy vectors with known extreme outliers."""
    g = torch.zeros(5, dtype=torch.float32)
    honest_val = torch.tensor([1.0, -2.0, 3.0, 0.5, -1.5], dtype=torch.float32)

    # 8 honest clients near honest_val + 2 extreme outliers (+1e5 and -1e5) -> N = 10
    updates: list[ClientUpdate] = []
    for i in range(8):
        jitter = (float(i) - 3.5) * 0.01
        updates.append(_make_client_update(f"honest_{i}", g, honest_val + jitter))
    updates.append(_make_client_update("mal_high", g, torch.full((5,), 1e5, dtype=torch.float32)))
    updates.append(_make_client_update("mal_low", g, torch.full((5,), -1e5, dtype=torch.float32)))

    res_tm10 = TrimmedMeanAggregator(beta=0.1).aggregate(g, updates)
    res_tm20 = TrimmedMeanAggregator(beta=0.2).aggregate(g, updates)
    res_med = MedianAggregator().aggregate(g, updates)

    assert torch.allclose(res_tm10.new_global_flat_params, honest_val, atol=1e-4)
    assert torch.allclose(res_tm20.new_global_flat_params, honest_val, atol=1e-4)
    assert torch.allclose(res_med.new_global_flat_params, honest_val, atol=1e-4)


def test_clustering_removes_huge_and_sign_flipped_updates() -> None:
    """Task 7: A tenant that sends a huge update or sign-flipped update is removed by clustering."""
    g = torch.ones(10, dtype=torch.float32)
    base_delta = torch.tensor(
        [0.2, -0.3, 0.4, 0.1, -0.2, 0.5, -0.1, 0.3, 0.2, -0.4], dtype=torch.float32
    )

    updates: list[ClientUpdate] = []
    for i in range(8):
        scale = 1.0 + 0.05 * (i - 3.5)
        updates.append(_make_client_update(f"honest_{i}", g, g + scale * base_delta))

    # Malicious tenant 1: sends a huge update (50x magnitude)
    updates.append(_make_client_update("mal_huge", g, g + 50.0 * base_delta))
    # Malicious tenant 2: sends a sign-flipped update (-5x direction)
    updates.append(_make_client_update("mal_flip", g, g - 5.0 * base_delta))

    agg = CosineClusteringAggregator(
        max_norm_factor=2.5,
        clip_norm_factor=1.5,
        min_cosine_to_median=0.0,
        post_trim_beta=0.0,
    )
    res = agg.aggregate(g, updates)

    assert "mal_huge" in res.rejected_clients
    assert "mal_flip" in res.rejected_clients
    assert len(res.accepted_clients) == 8
    assert torch.allclose(res.new_global_flat_params, g + base_delta, atol=0.05)


def test_static_weight_aggregator_freezes_weights() -> None:
    """Verify StaticWeightAggregator calibrates once on validation data and freezes weights."""
    model = build_model("dnn", input_dim=6, num_classes=2, hidden_dim=8, num_layers=1, dropout=0.0)
    g = model.get_flat_params()
    X_val = np.random.default_rng(0).standard_normal((64, 6)).astype(np.float32)
    y_val = np.array([0] * 32 + [1] * 32, dtype=np.int64)

    u1 = _make_client_update("t1", g, g + 0.01, num_samples=200)
    u2 = _make_client_update("t2", g, g - 0.01, num_samples=100)

    agg = build_aggregator("static_weight", val_model=model, X_val=X_val, y_val=y_val)
    assert isinstance(agg, StaticWeightAggregator)

    r1 = agg.aggregate(g, [u1, u2])
    frozen_after_r1 = dict(r1.weights)
    assert abs(sum(frozen_after_r1.values()) - 1.0) < 1e-6

    # Even if num_samples changes in round 2, frozen_weights remain locked
    u1_r2 = _make_client_update("t1", g, g + 0.5, num_samples=10)
    u2_r2 = _make_client_update("t2", g, g - 0.5, num_samples=990)
    r2 = agg.aggregate(g, [u1_r2, u2_r2])
    assert r2.weights == frozen_after_r1
