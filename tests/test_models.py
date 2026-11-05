"""Unit tests for Step 02 (`trustfed.models` and `trustfed.metrics`)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from trustfed.metrics import compute_detection_metrics, summarize_seed_metrics
from trustfed.models import (
    VALID_SEQ_MODES,
    DNNClassifier,
    GRUClassifier,
    LSTMClassifier,
    build_model,
    evaluate_pytorch_model,
    reshape_tabular_to_sequence,
    select_best_model_by_spec,
    train_pytorch_model,
)


@pytest.mark.parametrize("model_type", ["dnn", "gru", "lstm"])
@pytest.mark.parametrize("num_classes", [2, 7])
def test_unified_model_interface_and_flat_params(model_type: str, num_classes: int) -> None:
    """All three models accept (B, D) and return (B, num_classes) plus flat-param roundtrip."""
    batch_size, input_dim = 16, 19
    model = build_model(
        model_type,
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=32,
        num_layers=2,
        dropout=0.1,
    )
    model.eval()
    x = torch.randn(batch_size, input_dim)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (batch_size, num_classes)
    assert model.count_parameters() > 0

    flat = model.get_flat_params()
    assert flat.ndim == 1
    assert flat.numel() == model.count_parameters()

    # Perturb parameters, then restore from flat vector and verify identical logits
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.5)
        perturbed_logits = model(x)
        assert not torch.allclose(logits, perturbed_logits)

        model.set_flat_params(flat)
        restored_logits = model(x)
        assert torch.allclose(logits, restored_logits, atol=1e-6)


@pytest.mark.parametrize("seq_mode", VALID_SEQ_MODES)
@pytest.mark.parametrize("cls", [GRUClassifier, LSTMClassifier])
def test_recurrent_sequence_modes(
    seq_mode: str, cls: type[GRUClassifier] | type[LSTMClassifier]
) -> None:
    """GRU and LSTM support single_step, feature_chunks, and per_feature layouts."""
    x = torch.randn(8, 19)
    seq = reshape_tabular_to_sequence(x, seq_mode=seq_mode, num_chunks=4)  # type: ignore[arg-type]
    assert seq.ndim == 3
    assert seq.shape[0] == 8

    model = cls(
        input_dim=19,
        num_classes=5,
        hidden_dim=32,
        num_layers=2,
        seq_mode=seq_mode,  # type: ignore[arg-type]
        num_chunks=4,
    )
    out = model(x)
    assert out.shape == (8, 5)


def test_detection_metrics_binary_and_multiclass() -> None:
    """Verify ADR, FPR, Macro-F1, and ROC-AUC on known predictions."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.int64)
    y_prob = np.array([0.05, 0.10, 0.20, 0.70, 0.85, 0.90, 0.95, 0.40], dtype=np.float32)

    m = compute_detection_metrics(y_true, y_pred, y_prob, num_classes=2)
    assert np.isclose(m["accuracy"], 6.0 / 8.0)
    assert np.isclose(m["attack_detection_rate"], 3.0 / 4.0)
    assert np.isclose(m["false_positive_rate"], 1.0 / 4.0)
    assert 0.80 <= m["roc_auc"] <= 1.0

    # Multi-class with 3 classes (0=Benign, 1=DoS, 2=Probe)
    y_mul_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    y_mul_pred = np.array([0, 0, 1, 1, 2, 1], dtype=np.int64)
    y_mul_prob = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.05, 0.85, 0.1],
            [0.05, 0.15, 0.8],
            [0.1, 0.6, 0.3],
        ],
        dtype=np.float32,
    )
    mm = compute_detection_metrics(
        y_mul_true,
        y_mul_pred,
        y_mul_prob,
        num_classes=3,
        class_names=["Benign", "DoS", "Probe"],
    )
    assert np.isclose(mm["false_positive_rate"], 0.0)
    assert np.isclose(mm["attack_detection_rate"], 1.0)
    assert "DoS" in mm["per_class_f1"]

    agg = summarize_seed_metrics([m, m], scalar_keys=("accuracy", "f1_macro"))
    assert np.isclose(agg["accuracy"]["mean"], 0.75)
    assert np.isclose(agg["accuracy"]["std"], 0.0)


def test_train_and_evaluate_synthetic() -> None:
    """Verify end-to-end training and evaluation on a synthetic linearly separable dataset."""
    rng = np.random.default_rng(42)
    X_0 = rng.normal(loc=-1.5, scale=0.6, size=(120, 12)).astype(np.float32)
    X_1 = rng.normal(loc=1.5, scale=0.6, size=(120, 12)).astype(np.float32)
    X = np.vstack([X_0, X_1])
    y = np.array([0] * 120 + [1] * 120, dtype=np.int64)

    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    X_train, y_train = X[:180], y[:180]
    X_val, y_val = X[180:], y[180:]

    model = DNNClassifier(input_dim=12, num_classes=2, hidden_dim=32, num_layers=2)
    info = train_pytorch_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=6,
        batch_size=32,
        lr=5e-3,
        seed=0,
        device=torch.device("cpu"),
    )
    assert info["train_time_sec"] > 0.0
    eval_res = evaluate_pytorch_model(model, X_val, y_val, device=torch.device("cpu"))
    assert eval_res["f1_macro"] >= 0.95
    assert eval_res["inference_us_per_flow"] >= 0.0


def test_selection_rule_spec() -> None:
    """Verify EXPERIMENT_SPEC.md Section 3 model selection rule."""
    # Case 1: GRU is within 0.5 F1 points (0.005) of LSTM -> GRU stays default
    cands_1 = [
        {"model_type": "dnn", "f1_macro_mean": 0.9880, "param_count": 15000},
        {"model_type": "gru", "f1_macro_mean": 0.9910, "param_count": 42000},
        {"model_type": "lstm", "f1_macro_mean": 0.9935, "param_count": 56000},
    ]
    decision_1 = select_best_model_by_spec(cands_1, gru_margin_f1_points=0.5)
    assert decision_1["selected_model_type"] == "gru"
    assert decision_1["gru_within_margin"] is True

    # Case 2: GRU trails best model by > 0.5 F1 points -> best efficiency model chosen
    cands_2 = [
        {"model_type": "dnn", "f1_macro_mean": 0.9950, "param_count": 12000},
        {"model_type": "gru", "f1_macro_mean": 0.9820, "param_count": 42000},
        {"model_type": "lstm", "f1_macro_mean": 0.9840, "param_count": 56000},
    ]
    decision_2 = select_best_model_by_spec(cands_2, gru_margin_f1_points=0.5)
    assert decision_2["selected_model_type"] == "dnn"
    assert decision_2["gru_within_margin"] is False
