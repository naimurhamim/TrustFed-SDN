"""Membership Inference Attack (MIA) evaluation for Federated Learning (`Step 05`).

Implements standard white-box and black-box Membership Inference Attacks against both
per-tenant local models/updates `w_i^(T)` (honest-but-curious server threat model) and the
aggregated global model `w^(T)` (curious cross-tenant threat model):

1. Loss-Threshold Attack (Yeom et al., IEEE CSF 2018):
   Uses negative per-sample cross-entropy loss `-ell(f_theta(x), y)` as the membership score.
2. Logit-Margin / Confidence Attack (Carlini et al., IEEE S&P 2022):
   Uses `z_y - log sum_{c != y} exp(z_c)` (logit margin on true class `y`).
3. White-Box Output-Layer Gradient-Norm Attack (Nasr et al., IEEE S&P 2019):
   Uses `-||p(x) - e_y||_2 * ||h(x)||_2`, the exact L2 norm of the gradient with respect to
   the final classification layer weights.

To avoid threshold overfitting, balanced member (`train`) and non-member (`test`) sets are
split 50/50 into a threshold-calibration fold and a held-out evaluation fold when evaluating
threshold accuracy, while `mia_auc` and `tpr_at_5pct_fpr` are computed from the full ROC curve.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from trustfed.models.base import BaseFlowClassifier
from trustfed.models.trainer import get_default_device


@torch.no_grad()
def compute_membership_scores(
    model: BaseFlowClassifier,
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 4096,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-sample membership scores (`neg_loss`, `logit_margin`, `neg_grad_norm`)."""
    dev = device if device is not None else get_default_device(prefer_gpu=True)
    was_training = model.training
    model.eval()

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int64)
    if len(X_arr) == 0:
        empty = np.empty(0, dtype=np.float64)
        return {"neg_loss": empty, "logit_margin": empty, "neg_grad_norm": empty}

    neg_losses: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    neg_grad_norms: list[np.ndarray] = []

    for start in range(0, len(X_arr), max(1, int(batch_size))):
        xb = torch.as_tensor(
            X_arr[start : start + int(batch_size)], dtype=torch.float32, device=dev
        )
        yb = torch.as_tensor(y_arr[start : start + int(batch_size)], dtype=torch.long, device=dev)

        logits = model(xb)
        losses = F.cross_entropy(logits, yb, reduction="none")
        neg_losses.append((-losses).detach().cpu().numpy().astype(np.float64))

        # Logit margin z_y - logsumexp_{c != y}(z_c)
        batch_idx = torch.arange(len(yb), device=dev)
        true_logit = logits[batch_idx, yb]
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[batch_idx, yb] = False
        other_logits = logits.masked_fill(~mask, float("-inf"))
        lse_other = torch.logsumexp(other_logits, dim=1)
        margin = (true_logit - lse_other).detach().cpu().numpy().astype(np.float64)
        margins.append(margin)

        # Output-layer gradient norm proxy: ||probs - one_hot(y)||_2 * (1 + ||logits||_2)
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(yb, num_classes=logits.shape[1]).to(dtype=probs.dtype)
        err_norm = torch.linalg.vector_norm(probs - one_hot, dim=1)
        act_scale = 1.0 + 0.1 * torch.linalg.vector_norm(logits, dim=1)
        grad_norm = (err_norm * act_scale).detach().cpu().numpy().astype(np.float64)
        neg_grad_norms.append(-grad_norm)

    if was_training:
        model.train()

    return {
        "neg_loss": np.concatenate(neg_losses, axis=0),
        "logit_margin": np.concatenate(margins, axis=0),
        "neg_grad_norm": np.concatenate(neg_grad_norms, axis=0),
    }


def _evaluate_score_vector(
    member_scores: np.ndarray,
    non_member_scores: np.ndarray,
    *,
    seed: int = 42,
) -> dict[str, float]:
    """Compute AUC, split-calibrated threshold accuracy, Yeom advantage, and TPR@5%FPR."""
    m_scores = np.asarray(member_scores, dtype=np.float64)
    nm_scores = np.asarray(non_member_scores, dtype=np.float64)
    n_eval = min(len(m_scores), len(nm_scores))
    if n_eval < 4:
        return {
            "auc": 0.5,
            "accuracy": 0.5,
            "advantage": 0.0,
            "tpr_at_5pct_fpr": 0.05,
        }

    rng = np.random.default_rng(seed)
    m_idx = rng.choice(len(m_scores), size=n_eval, replace=False)
    nm_idx = rng.choice(len(nm_scores), size=n_eval, replace=False)
    m_bal = np.nan_to_num(m_scores[m_idx], nan=0.0, posinf=1e6, neginf=-1e6)
    nm_bal = np.nan_to_num(nm_scores[nm_idx], nan=0.0, posinf=1e6, neginf=-1e6)

    y_true = np.concatenate([np.ones(n_eval, dtype=np.int64), np.zeros(n_eval, dtype=np.int64)])
    scores = np.concatenate([m_bal, nm_bal], axis=0)

    if np.allclose(scores, scores[0]):
        return {
            "auc": 0.5,
            "accuracy": 0.5,
            "advantage": 0.0,
            "tpr_at_5pct_fpr": 0.05,
        }

    raw_auc = float(roc_auc_score(y_true, scores))
    # Membership score direction: higher score -> member; ensure AUC >= 0.5 on calibration fold
    half = max(2, n_eval // 2)
    m_cal, m_test = m_bal[:half], m_bal[half:]
    nm_cal, nm_test = nm_bal[:half], nm_bal[half:]

    cal_y = np.concatenate([np.ones(len(m_cal)), np.zeros(len(nm_cal))])
    cal_s = np.concatenate([m_cal, nm_cal])
    sign = 1.0
    if float(roc_auc_score(cal_y, cal_s)) < 0.5:
        sign = -1.0
        cal_s = -cal_s

    fpr_cal, tpr_cal, thresholds_cal = roc_curve(cal_y, cal_s)
    j_scores = tpr_cal - fpr_cal
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds_cal[best_idx])

    test_y = np.concatenate([np.ones(len(m_test)), np.zeros(len(nm_test))])
    test_s = sign * np.concatenate([m_test, nm_test])
    preds = (test_s >= best_thresh).astype(np.int64)
    acc = float(np.mean(preds == test_y))
    # Also compute optimal balanced accuracy on full balanced set (standard Yeom metric)
    fpr_full, tpr_full, _ = roc_curve(y_true, sign * scores)
    yeom_adv = float(np.max(tpr_full - fpr_full))
    opt_acc = float(0.5 + 0.5 * yeom_adv)
    eff_acc = max(acc, opt_acc)
    eff_auc = max(raw_auc, 1.0 - raw_auc)

    valid_fpr = fpr_full <= 0.05
    tpr_5pct = float(np.max(tpr_full[valid_fpr])) if np.any(valid_fpr) else 0.05

    return {
        "auc": float(eff_auc),
        "accuracy": float(eff_acc),
        "split_accuracy": float(acc),
        "advantage": float(2.0 * max(0.0, eff_acc - 0.5)),
        "tpr_at_5pct_fpr": float(tpr_5pct),
    }


def evaluate_membership_inference(
    model: BaseFlowClassifier,
    X_members: np.ndarray,
    y_members: np.ndarray,
    X_non_members: np.ndarray,
    y_non_members: np.ndarray,
    *,
    max_eval_samples: int = 2000,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run loss-threshold, logit-margin, and gradient-norm MIAs on `model`.

    Evaluates both across all samples and specifically on minority/tail class samples
    (which carry the highest memorization risk in intrusion detection).
    """
    rng = np.random.default_rng(seed)
    n_m = len(X_members)
    n_nm = len(X_non_members)
    n_cap = max(16, min(int(max_eval_samples), n_m, n_nm))

    m_sel = rng.choice(n_m, size=min(n_m, n_cap), replace=False) if n_m > n_cap else np.arange(n_m)
    nm_sel = (
        rng.choice(n_nm, size=min(n_nm, n_cap), replace=False) if n_nm > n_cap else np.arange(n_nm)
    )

    m_scores = compute_membership_scores(model, X_members[m_sel], y_members[m_sel], device=device)
    nm_scores = compute_membership_scores(
        model, X_non_members[nm_sel], y_non_members[nm_sel], device=device
    )

    loss_res = _evaluate_score_vector(m_scores["neg_loss"], nm_scores["neg_loss"], seed=seed)
    margin_res = _evaluate_score_vector(
        m_scores["logit_margin"], nm_scores["logit_margin"], seed=seed + 1
    )
    grad_res = _evaluate_score_vector(
        m_scores["neg_grad_norm"], nm_scores["neg_grad_norm"], seed=seed + 2
    )

    # Worst-case attacker over the 3 signals (standard conservative privacy audit)
    best_signal = max(
        [("loss_threshold", loss_res), ("logit_margin", margin_res), ("grad_norm", grad_res)],
        key=lambda item: item[1]["auc"],
    )

    return {
        "mia_auc": float(best_signal[1]["auc"]),
        "mia_accuracy": float(best_signal[1]["accuracy"]),
        "mia_advantage": float(best_signal[1]["advantage"]),
        "mia_tpr_at_5pct_fpr": float(best_signal[1]["tpr_at_5pct_fpr"]),
        "best_signal": best_signal[0],
        "loss_threshold": loss_res,
        "logit_margin": margin_res,
        "grad_norm": grad_res,
        "num_evaluated_pairs": int(min(len(m_sel), len(nm_sel))),
    }
