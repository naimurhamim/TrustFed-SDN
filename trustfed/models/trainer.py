"""Model factory, fast GPU/CPU training & evaluation routines, and spec selection rule."""

from __future__ import annotations

import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from trustfed.metrics.classification import compute_detection_metrics
from trustfed.models.base import BaseFlowClassifier, SequenceMode
from trustfed.models.dnn import DNNClassifier
from trustfed.models.gru import GRUClassifier
from trustfed.models.lstm import LSTMClassifier


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_default_device(prefer_gpu: bool = True) -> torch.device:
    """Return `cuda` device when available and requested, otherwise `cpu`."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(
    model_type: str,
    input_dim: int,
    num_classes: int = 2,
    *,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    seq_mode: SequenceMode = "feature_chunks",
    num_chunks: int = 4,
) -> BaseFlowClassifier:
    """Instantiate a unified TrustFed-SDN classifier (`dnn`, `gru`, or `lstm`)."""
    mtype = model_type.strip().lower()
    if mtype == "dnn":
        return DNNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    if mtype == "gru":
        return GRUClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            seq_mode=seq_mode,
            num_chunks=num_chunks,
        )
    if mtype == "lstm":
        return LSTMClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            seq_mode=seq_mode,
            num_chunks=num_chunks,
        )
    raise ValueError(f"Unknown model_type '{model_type}'. Expected 'dnn', 'gru', or 'lstm'.")


@torch.no_grad()
def predict_pytorch_model(
    model: BaseFlowClassifier,
    X: np.ndarray,
    *,
    batch_size: int = 4096,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run batched inference on `X` and return `(y_pred, y_prob, inference_us_per_flow)`."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    n_samples = len(X)
    if n_samples == 0:
        raise ValueError("Cannot run inference on empty array X.")

    X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    probs_list: list[torch.Tensor] = []
    for start in range(0, n_samples, batch_size):
        xb = X_tensor[start : start + batch_size]
        logits = model(xb)
        probs = torch.softmax(logits, dim=-1)
        probs_list.append(probs)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_sec = time.perf_counter() - t0

    all_probs = torch.cat(probs_list, dim=0)
    all_preds = torch.argmax(all_probs, dim=-1)

    us_per_flow = (elapsed_sec * 1e6) / float(n_samples)
    return (
        all_preds.cpu().numpy().astype(np.int64),
        all_probs.cpu().numpy().astype(np.float32),
        float(us_per_flow),
    )


def evaluate_pytorch_model(
    model: BaseFlowClassifier,
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 4096,
    device: torch.device | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate `model` on `(X, y)` and return all detection metrics + latency."""
    y_pred, y_prob, us_per_flow = predict_pytorch_model(
        model, X, batch_size=batch_size, device=device
    )
    metrics = compute_detection_metrics(
        y_true=y,
        y_pred=y_pred,
        y_prob=y_prob,
        num_classes=model.num_classes,
        class_names=class_names,
    )
    metrics["inference_us_per_flow"] = us_per_flow
    return metrics


def train_pytorch_model(
    model: BaseFlowClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    *,
    epochs: int = 12,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: torch.device | None = None,
    use_class_weights: bool = False,
    patience: int = 4,
) -> dict[str, Any]:
    """Train `model` with Adam + CrossEntropyLoss and restore best validation state.

    Tuning / early-stopping strictly uses `(X_val, y_val)` and never touches the test set.
    """
    set_all_seeds(seed)
    if device is None:
        device = get_default_device(prefer_gpu=True)
    model.to(device)

    X_tr = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.as_tensor(y_train, dtype=torch.long, device=device)
    n_train = X_tr.shape[0]

    weight_tensor: torch.Tensor | None = None
    if use_class_weights:
        counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=model.num_classes)
        counts_safe = np.maximum(counts, 1).astype(np.float64)
        inv = np.sqrt(float(n_train) / (model.num_classes * counts_safe))
        inv = np.clip(inv, 0.5, 5.0)
        weight_tensor = torch.as_tensor(inv, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=lr * 0.1
    )

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    best_val_f1 = -1.0
    best_flat_params = model.get_flat_params()
    best_epoch = 0
    epochs_no_improve = 0
    history: list[dict[str, float]] = []

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, generator=gen).to(device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb = X_tr[idx]
            yb = y_tr[idx]

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += float(loss.detach().item())
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_batches)

        val_f1 = float("nan")
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_metrics = evaluate_pytorch_model(
                model, X_val, y_val, batch_size=max(batch_size * 4, 4096), device=device
            )
            val_f1 = float(val_metrics["f1_macro"])
            if val_f1 > best_val_f1 + 1e-5:
                best_val_f1 = val_f1
                best_flat_params = model.get_flat_params()
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        history.append({"epoch": float(epoch), "train_loss": avg_loss, "val_f1_macro": val_f1})
        if patience > 0 and epochs_no_improve >= patience and epoch >= 4:
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_time_sec = time.perf_counter() - t0

    if best_val_f1 >= 0.0:
        model.set_flat_params(best_flat_params)

    return {
        "train_time_sec": float(train_time_sec),
        "best_val_f1_macro": float(best_val_f1),
        "best_epoch": int(best_epoch if best_epoch > 0 else epochs),
        "epochs_run": len(history),
        "history": history,
    }


def select_best_model_by_spec(
    candidates: list[dict[str, Any]],
    *,
    gru_margin_f1_points: float = 0.5,
) -> dict[str, Any]:
    """Apply the local model selection rule from `docs/EXPERIMENT_SPEC.md` Section 3:

    "Selection rule: choose the model with the best macro-F1 per parameter count;
    the GRU stays the default if it is within 0.5 F1 points of the best."

    Args:
        candidates: List of dicts, each containing at least:
            - `model_type`: `"dnn"`, `"gru"`, or `"lstm"`
            - `f1_macro_mean`: Mean macro-F1 on `[0.0, 1.0]` scale
            - `param_count`: Integer number of trainable parameters
        gru_margin_f1_points: Allowed margin in F1 percentage points (`0.5` points = `0.005`).

    Returns:
        Dictionary with `selected_model_type`, `reason`, `best_f1_model_type`,
        `best_efficiency_model_type`, `gru_gap_f1_points`, and enriched `candidates`.
    """
    if not candidates:
        raise ValueError("candidates list cannot be empty.")

    margin = gru_margin_f1_points / 100.0  # 0.5 F1 points -> 0.005 on [0, 1] scale
    enriched: list[dict[str, Any]] = []
    for c in candidates:
        item = dict(c)
        f1 = float(item["f1_macro_mean"])
        params = max(1, int(item["param_count"]))
        # Macro-F1 per 10k parameters and Macro-F1 / log10(params)
        item["f1_per_10k_params"] = float(f1 / (params / 10000.0))
        item["f1_per_log10_params"] = float(f1 / max(1.0, np.log10(float(params))))
        enriched.append(item)

    best_f1_cand = max(enriched, key=lambda x: float(x["f1_macro_mean"]))
    best_eff_cand = max(enriched, key=lambda x: float(x["f1_per_10k_params"]))

    best_f1 = float(best_f1_cand["f1_macro_mean"])
    gru_cands = [x for x in enriched if str(x["model_type"]).lower() == "gru"]

    if gru_cands:
        gru_cand = max(gru_cands, key=lambda x: float(x["f1_macro_mean"]))
        gru_f1 = float(gru_cand["f1_macro_mean"])
        gap_points = (best_f1 - gru_f1) * 100.0
        if (best_f1 - gru_f1) <= margin + 1e-9:
            return {
                "selected_model_type": "gru",
                "selected_candidate": gru_cand,
                "best_f1_model_type": str(best_f1_cand["model_type"]),
                "best_efficiency_model_type": str(best_eff_cand["model_type"]),
                "gru_gap_f1_points": float(gap_points),
                "gru_within_margin": True,
                "rule_applied": (
                    f"GRU macro-F1 ({gru_f1 * 100:.2f}%) is within {gru_margin_f1_points:.2f} "
                    f"F1 points of the highest macro-F1 ({best_f1 * 100:.2f}% from "
                    f"{str(best_f1_cand['model_type']).upper()}, gap = {gap_points:+.2f} pts); "
                    "per EXPERIMENT_SPEC.md Section 3, GRU remains the selected local model."
                ),
                "candidates": enriched,
            }
    else:
        gap_points = float("nan")

    return {
        "selected_model_type": str(best_eff_cand["model_type"]),
        "selected_candidate": best_eff_cand,
        "best_f1_model_type": str(best_f1_cand["model_type"]),
        "best_efficiency_model_type": str(best_eff_cand["model_type"]),
        "gru_gap_f1_points": float(gap_points),
        "gru_within_margin": False,
        "rule_applied": (
            f"GRU exceeded the {gru_margin_f1_points:.2f} F1-point margin vs best model "
            f"({str(best_f1_cand['model_type']).upper()} at {best_f1 * 100:.2f}%); selected "
            f"{str(best_eff_cand['model_type']).upper()} by macro-F1 per parameter count."
        ),
        "candidates": enriched,
    }
