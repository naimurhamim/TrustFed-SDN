"""Unit tests for Step 04 poisoning attacks and Backdoor ASR metric (`tests/test_attacks.py`)."""

from __future__ import annotations

import numpy as np
import torch

from trustfed.attacks import (
    BackdoorAttack,
    LabelFlipAttack,
    UpdateScalingAttack,
    apply_backdoor_trigger,
    build_attack,
    compute_backdoor_asr,
)
from trustfed.fl import ClientUpdate, serialize_tensor_payload
from trustfed.models import build_model


def _make_dummy_update(global_flat: torch.Tensor, delta: torch.Tensor) -> ClientUpdate:
    flat_params = global_flat + delta
    wire = serialize_tensor_payload(flat_params, num_samples=100, round_idx=1)
    return ClientUpdate(
        client_id="tenant_mal",
        flat_params=flat_params,
        flat_delta=delta,
        num_samples=100,
        train_loss=0.25,
        local_train_time_sec=0.05,
        upload_bytes=len(wire),
        round_idx=1,
    )


def test_label_flip_attack_dataset_and_update() -> None:
    """Verify LabelFlipAttack flips labels deterministically and scales update delta."""
    X = np.ones((20, 6), dtype=np.float32)
    y = np.array([0] * 10 + [1] * 10, dtype=np.int64)

    atk = LabelFlipAttack(flip_fraction=1.0, bidirectional=True, update_scale=3.5)
    X_p, y_p = atk.poison_dataset(X, y, num_classes=2, seed=42)
    assert np.array_equal(X, X_p)
    assert np.array_equal(y_p, 1 - y)

    # Unidirectional flip only flips attack (>0) to benign (0)
    atk_uni = LabelFlipAttack(flip_fraction=1.0, bidirectional=False, update_scale=2.0)
    _, y_uni = atk_uni.poison_dataset(X, y, num_classes=4, seed=42)
    assert np.all(y_uni == 0)

    g = torch.zeros(16, dtype=torch.float32)
    d = torch.ones(16, dtype=torch.float32) * 0.5
    upd = _make_dummy_update(g, d)
    p_upd = atk.poison_update(g, upd, round_idx=1, seed=42)
    assert torch.allclose(p_upd.flat_delta, d * 3.5)
    assert torch.allclose(p_upd.flat_params, g + d * 3.5)
    assert p_upd.metadata["is_malicious"] is True
    assert p_upd.metadata["attack"] == "label_flip"


def test_backdoor_attack_trigger_and_asr() -> None:
    """Verify feature-trigger stamping, dataset poisoning, and Backdoor ASR calculation."""
    X = np.zeros((40, 8), dtype=np.float32)
    y = np.array([0] * 10 + [1] * 30, dtype=np.int64)

    X_trig = apply_backdoor_trigger(X, trigger_features=(0, 2, 4), trigger_value=4.5)
    assert np.allclose(X_trig[:, [0, 2, 4]], 4.5)
    assert np.allclose(X_trig[:, [1, 3, 5, 6, 7]], 0.0)

    atk = BackdoorAttack(
        trigger_features=(0, 1, 2),
        trigger_value=4.5,
        target_class=0,
        poison_ratio=0.5,
        replacement_scale=4.0,
    )
    X_p, y_p = atk.poison_dataset(X, y, num_classes=2, seed=7)
    # 50% of the 30 attack rows (15 rows) should be relabeled to 0 and have trigger 4.5
    assert int(np.sum(y_p == 0)) == 10 + 15
    triggered_rows = np.all(np.isclose(X_p[:, :3], 4.5), axis=1)
    assert int(np.sum(triggered_rows)) == 15
    assert np.all(y_p[triggered_rows] == 0)

    # Verify compute_backdoor_asr on a zero-initialized model
    model = build_model("dnn", input_dim=8, num_classes=2, hidden_dim=8, num_layers=1, dropout=0.0)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    asr_stats = compute_backdoor_asr(
        model,
        X,
        y,
        trigger_features=(0, 1, 2),
        trigger_value=4.5,
        target_class=0,
        device=torch.device("cpu"),
    )
    assert 0.0 <= asr_stats["backdoor_asr"] <= 1.0
    assert asr_stats["num_attack_samples"] == 30.0


def test_update_scaling_and_sign_flip() -> None:
    """Verify UpdateScalingAttack scales and sign-flips the update tensor."""
    g = torch.ones(12, dtype=torch.float32) * 2.0
    d = torch.arange(1, 13, dtype=torch.float32) * 0.1
    upd = _make_dummy_update(g, d)

    atk_flip = build_attack("scaling", scale_factor=10.0, sign_flip=True)
    assert isinstance(atk_flip, UpdateScalingAttack)
    res_flip = atk_flip.poison_update(g, upd, round_idx=2, seed=0)
    assert torch.allclose(res_flip.flat_delta, -10.0 * d)
    assert torch.allclose(res_flip.flat_params, g - 10.0 * d)

    atk_pos = UpdateScalingAttack(scale_factor=8.0, sign_flip=False)
    res_pos = atk_pos.poison_update(g, upd, round_idx=2, seed=0)
    assert torch.allclose(res_pos.flat_delta, 8.0 * d)
    assert torch.allclose(res_pos.flat_params, g + 8.0 * d)
