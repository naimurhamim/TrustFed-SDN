"""Per-tenant sensitivity scores and privacy budget allocation (`Step 05`).

Approved design (`docs/USER_PREFERENCES.md`, Step 05):

Sensitivity score (hybrid, in `[0.5, 1.0]`):
    s_i = 0.5 * domain_profile_i + 0.5 * data_score_i

    domain_profile_i: Finance = 1.0, Healthcare = 1.0, Education = 0.5 (others default 0.75).
    data_score_i: derived from the tenant's own training labels:
        rare_share_i = share of the tenant's attack records that belong to globally rare
                       attack classes (classes holding < `rare_threshold` of all attack records)
        imbalance_i  = 1 - normalized Shannon entropy of the tenant's class distribution
        raw_i        = 0.5 * rare_share_i + 0.5 * imbalance_i
        data_score_i = 0.5 + 0.5 * minmax(raw)_i   (0.75 for every tenant if all raw_i equal)

Budget policy (allocation rule, not a composition theorem across tenants):
    adaptive: eps_i = eps_total * (1 / s_i) / sum_j (1 / s_j)
    uniform:  eps_i = eps_total / N
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

DOMAIN_PROFILE_DEFAULT: dict[str, float] = {
    "finance": 1.0,
    "healthcare": 1.0,
    "education": 0.5,
}
DOMAIN_PROFILE_FALLBACK = 0.75


@dataclass
class TenantSensitivity:
    """Sensitivity score breakdown for one tenant."""

    tenant: str
    domain_profile: float
    rare_share: float
    imbalance: float
    raw_data_value: float
    data_score: float
    score: float
    extra: dict[str, float] = field(default_factory=dict)


def domain_profile_for(tenant_name: str, profile: Mapping[str, float] | None = None) -> float:
    """Return the fixed domain-criticality value for a tenant name (substring match)."""
    table = dict(profile or DOMAIN_PROFILE_DEFAULT)
    lowered = tenant_name.lower()
    for key, value in table.items():
        if key in lowered:
            return float(value)
    return DOMAIN_PROFILE_FALLBACK


def compute_sensitivity_scores(
    tenant_labels: Mapping[str, np.ndarray],
    *,
    num_classes: int,
    benign_class: int = 0,
    rare_threshold: float = 0.05,
    profile: Mapping[str, float] | None = None,
    profile_weight: float = 0.5,
) -> dict[str, TenantSensitivity]:
    """Compute the approved hybrid sensitivity score for every tenant.

    Args:
        tenant_labels: Mapping tenant name -> multi-class training labels of that tenant.
        num_classes: Total number of classes (including benign).
        benign_class: Index of the benign class.
        rare_threshold: An attack class is "rare" if it holds less than this share of all
            attack training records across tenants.
        profile: Optional override for domain profile values.
        profile_weight: Weight of the domain profile term (approved value 0.5).
    """
    if not tenant_labels:
        raise ValueError("tenant_labels cannot be empty.")
    if not 0.0 <= profile_weight <= 1.0:
        raise ValueError(f"profile_weight must be in [0, 1], got {profile_weight}.")

    names = list(tenant_labels.keys())
    counts = {
        n: np.bincount(np.asarray(tenant_labels[n], dtype=np.int64), minlength=num_classes)[
            :num_classes
        ].astype(np.float64)
        for n in names
    }

    global_counts = np.sum(np.stack([counts[n] for n in names], axis=0), axis=0)
    attack_mask = np.ones(num_classes, dtype=bool)
    attack_mask[benign_class] = False
    total_attack = float(np.sum(global_counts[attack_mask]))
    if total_attack > 0:
        attack_share = np.where(attack_mask, global_counts / total_attack, 0.0)
    else:
        attack_share = np.zeros(num_classes, dtype=np.float64)
    rare_mask = attack_mask & (global_counts > 0) & (attack_share < float(rare_threshold))

    raw: dict[str, float] = {}
    rare_shares: dict[str, float] = {}
    imbalances: dict[str, float] = {}
    for n in names:
        c = counts[n]
        n_attack = float(np.sum(c[attack_mask]))
        rare_shares[n] = float(np.sum(c[rare_mask]) / n_attack) if n_attack > 0 else 0.0

        present = c[c > 0]
        if len(present) > 1 and float(np.max(present)) > 0:
            imbalances[n] = float(1.0 - float(np.min(present)) / float(np.max(present)))
        else:
            imbalances[n] = 0.0
        raw[n] = 0.5 * rare_shares[n] + 0.5 * imbalances[n]

    raw_vals = np.asarray([raw[n] for n in names], dtype=np.float64)
    span = float(raw_vals.max() - raw_vals.min())
    result: dict[str, TenantSensitivity] = {}
    for n in names:
        norm = (raw[n] - float(raw_vals.min())) / span if span > 1e-12 else 0.5
        data_score = 0.5 + 0.5 * norm
        dom = domain_profile_for(n, profile)
        score = profile_weight * dom + (1.0 - profile_weight) * data_score
        result[n] = TenantSensitivity(
            tenant=n,
            domain_profile=dom,
            rare_share=rare_shares[n],
            imbalance=imbalances[n],
            raw_data_value=raw[n],
            data_score=float(data_score),
            score=float(score),
        )
    return result


def allocate_budgets(
    scores: Mapping[str, float],
    epsilon_total: float,
    *,
    policy: str = "adaptive",
) -> dict[str, float]:
    """Split `epsilon_total` across tenants.

    `adaptive`: eps_i proportional to 1 / s_i (higher sensitivity -> smaller epsilon).
    `uniform`:  eps_i = epsilon_total / N.

    The allocations always sum to `epsilon_total`. This is a design policy for distributing
    protection; it is not a composition theorem across tenants (each tenant's guarantee
    covers its own records only).
    """
    if not scores:
        raise ValueError("scores cannot be empty.")
    if epsilon_total <= 0.0:
        raise ValueError(f"epsilon_total must be > 0, got {epsilon_total}.")
    names = list(scores.keys())
    pol = policy.strip().lower()
    if pol == "uniform":
        share = float(epsilon_total) / len(names)
        return {n: share for n in names}
    if pol != "adaptive":
        raise ValueError(f"Unknown budget policy '{policy}'. Use 'adaptive' or 'uniform'.")
    inv = np.asarray([1.0 / float(scores[n]) for n in names], dtype=np.float64)
    if np.any(~np.isfinite(inv)) or np.any(inv <= 0.0):
        raise ValueError("All sensitivity scores must be finite and > 0.")
    alloc = float(epsilon_total) * inv / float(np.sum(inv))
    return {n: float(a) for n, a in zip(names, alloc, strict=True)}


def summarize_allocation(
    sensitivities: Mapping[str, TenantSensitivity],
    epsilons: Mapping[str, float],
) -> list[dict[str, float | str]]:
    """Return a flat table of per-tenant scores and allocated epsilons."""
    rows: list[dict[str, float | str]] = []
    for n, s in sensitivities.items():
        rows.append(
            {
                "tenant": n,
                "domain_profile": s.domain_profile,
                "rare_share": s.rare_share,
                "imbalance": s.imbalance,
                "data_score": s.data_score,
                "sensitivity_score": s.score,
                "epsilon": float(epsilons[n]),
            }
        )
    return rows


def mean_budget(epsilons: Sequence[float]) -> float:
    """Mean per-tenant epsilon."""
    return float(np.mean(np.asarray(list(epsilons), dtype=np.float64)))
