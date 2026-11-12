"""Adaptive Per-Tenant Differential Privacy & Membership Inference Audit (`Step 05`)."""

from __future__ import annotations

from trustfed.privacy.accountant import (
    DEFAULT_ORDERS,
    RDPAccountant,
    calibrate_noise_multiplier,
    compute_epsilon,
    rdp_sampled_gaussian,
    rdp_to_epsilon,
)
from trustfed.privacy.budget import (
    DOMAIN_PROFILE_DEFAULT,
    TenantSensitivity,
    allocate_budgets,
    compute_sensitivity_scores,
    domain_profile_for,
    mean_budget,
    summarize_allocation,
)
from trustfed.privacy.dp_sgd import (
    DPConfig,
    DPFLClient,
    add_gaussian_noise,
    calibrate_tenant_noise,
    clip_and_sum_per_sample_grads,
    compute_per_sample_grads,
    functional_gru_forward,
    steps_per_epoch,
    total_dp_steps,
)
from trustfed.privacy.mia import (
    compute_membership_scores,
    evaluate_membership_inference,
)

__all__ = [
    "DEFAULT_ORDERS",
    "DOMAIN_PROFILE_DEFAULT",
    "DPConfig",
    "DPFLClient",
    "RDPAccountant",
    "TenantSensitivity",
    "add_gaussian_noise",
    "allocate_budgets",
    "calibrate_noise_multiplier",
    "calibrate_tenant_noise",
    "clip_and_sum_per_sample_grads",
    "compute_epsilon",
    "compute_membership_scores",
    "compute_per_sample_grads",
    "compute_sensitivity_scores",
    "domain_profile_for",
    "evaluate_membership_inference",
    "functional_gru_forward",
    "mean_budget",
    "rdp_sampled_gaussian",
    "rdp_to_epsilon",
    "steps_per_epoch",
    "summarize_allocation",
    "total_dp_steps",
]
