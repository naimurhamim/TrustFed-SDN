"""Detection and classification metrics for TrustFed-SDN (`docs/EXPERIMENT_SPEC.md` Section 9).

Provides unified metric computation for both binary (Benign=0 vs Attack=1) and
multi-class (Benign=0, Attack families=1..C-1) threat detection tasks:
- accuracy
- precision_macro, precision_attack
- recall_macro, recall_attack
- f1_macro, f1_attack, f1_weighted
- attack_detection_rate (recall on all non-Benign traffic)
- false_positive_rate (fraction of Benign traffic misclassified as Attack)
- roc_auc (binary ROC-AUC or multi-class OVR macro ROC-AUC over present classes)
- per_class_f1 (dict mapping class name to F1)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    *,
    num_classes: int = 2,
    class_names: list[str] | None = None,
    benign_index: int = 0,
) -> dict[str, Any]:
    """Compute all detection metrics specified in `docs/EXPERIMENT_SPEC.md` Section 9.

    Args:
        y_true: 1-D integer array of ground-truth class labels in `0..num_classes-1`.
        y_pred: 1-D integer array of predicted class labels in `0..num_classes-1`.
        y_prob: Optional probability array. For `num_classes == 2`, either 1-D array
            of positive-class (`Attack`) probabilities or 2-D `(N, 2)` softmax probabilities.
            For `num_classes > 2`, 2-D `(N, num_classes)` softmax probabilities.
        num_classes: Total number of classes (`2` for binary, `C` for multi-class).
        class_names: Optional list of class names of length `num_classes`.
        benign_index: Index of the Benign/Normal class (default `0`).

    Returns:
        Dictionary of scalar metrics (floats in `[0.0, 1.0]`) plus `per_class_f1`.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=np.int64).ravel()

    if len(y_true_arr) == 0:
        raise ValueError("Cannot compute detection metrics on empty arrays.")

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec_macro = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    f1_mac = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    f1_w = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

    # Binary Attack vs Benign projection (even when evaluating multi-class models,
    # any class != benign_index is an attack flow).
    y_true_bin = (y_true_arr != benign_index).astype(np.int64)
    y_pred_bin = (y_pred_arr != benign_index).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()

    prec_attack = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    adr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0  # Attack Detection Rate
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0  # False Positive Rate
    f1_att = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

    # ROC-AUC computation
    auc_val = float("nan")
    if y_prob is not None:
        prob_arr = np.asarray(y_prob, dtype=np.float64)
        try:
            if num_classes == 2:
                if prob_arr.ndim == 2 and prob_arr.shape[1] >= 2:
                    pos_prob = prob_arr[:, 1]
                else:
                    pos_prob = prob_arr.ravel()
                if len(np.unique(y_true_bin)) == 2:
                    auc_val = float(roc_auc_score(y_true_bin, pos_prob))
            else:
                present_classes = np.unique(y_true_arr)
                if len(present_classes) >= 2 and prob_arr.ndim == 2:
                    if len(present_classes) == num_classes:
                        auc_val = float(
                            roc_auc_score(
                                y_true_arr,
                                prob_arr,
                                multi_class="ovr",
                                average="macro",
                            )
                        )
                    else:
                        # Subset to classes actually present in this partition
                        sub_prob = prob_arr[:, present_classes]
                        row_sums = sub_prob.sum(axis=1, keepdims=True)
                        row_sums = np.where(row_sums > 0, row_sums, 1.0)
                        sub_prob = sub_prob / row_sums
                        if len(present_classes) == 2:
                            y_sub = (y_true_arr == present_classes[1]).astype(np.int64)
                            auc_val = float(roc_auc_score(y_sub, sub_prob[:, 1]))
                        else:
                            auc_val = float(
                                roc_auc_score(
                                    y_true_arr,
                                    sub_prob,
                                    labels=present_classes,
                                    multi_class="ovr",
                                    average="macro",
                                )
                            )
        except ValueError:
            auc_val = float("nan")

    labels_list = list(range(num_classes))
    per_cls_arr = f1_score(
        y_true_arr,
        y_pred_arr,
        labels=labels_list,
        average=None,
        zero_division=0,
    )
    if class_names is not None and len(class_names) == num_classes:
        names = class_names
    else:
        names = ["Benign", "Attack"] if num_classes == 2 else [f"class_{i}" for i in labels_list]

    per_class_f1 = {str(names[i]): float(per_cls_arr[i]) for i in range(num_classes)}

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_mac,
        "f1_weighted": f1_w,
        "precision_attack": prec_attack,
        "recall_attack": adr,
        "f1_attack": f1_att,
        "attack_detection_rate": adr,
        "false_positive_rate": fpr,
        "roc_auc": auc_val,
        "confusion_binary": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "per_class_f1": per_class_f1,
    }


def summarize_seed_metrics(
    seed_records: list[dict[str, Any]],
    scalar_keys: tuple[str, ...] = (
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "precision_attack",
        "f1_attack",
        "attack_detection_rate",
        "false_positive_rate",
        "roc_auc",
        "train_time_sec",
        "inference_us_per_flow",
    ),
) -> dict[str, dict[str, float]]:
    """Aggregate scalar metrics across multiple random seeds into mean and std."""
    summary: dict[str, dict[str, float]] = {}
    for key in scalar_keys:
        vals = [
            float(r[key])
            for r in seed_records
            if key in r and r[key] is not None and not np.isnan(float(r[key]))
        ]
        if vals:
            summary[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=0)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n_seeds": len(vals),
            }
    return summary
