"""Unit tests for Step 05 Adaptive Per-Tenant Differential Privacy (`tests/test_privacy.py`)."""

from __future__ import annotations

import numpy as np
import torch

from trustfed.models import build_model
from trustfed.privacy import (
    DPConfig,
    DPFLClient,
    RDPAccountant,
    add_gaussian_noise,
    allocate_budgets,
    calibrate_noise_multiplier,
    clip_and_sum_per_sample_grads,
    compute_epsilon,
    compute_per_sample_grads,
    compute_sensitivity_scores,
    evaluate_membership_inference,
    functional_gru_forward,
)


def test_rdp_accountant_monotonicity_and_calibration() -> None:
    """Verify RDP accountant output is strictly monotonic in steps and decreases with sigma."""
    acc = RDPAccountant()
    q = 0.05
    sigma = 1.5
    delta = 1e-5

    eps_history: list[float] = []
    for _ in range(10):
        acc.step(q, sigma, num_steps=5)
        eps_history.append(acc.get_epsilon(delta))

    # Monotonically increasing with number of steps/rounds
    for prev, curr in zip(eps_history[:-1], eps_history[1:], strict=True):
        assert curr > prev > 0.0

    # Higher noise multiplier yields smaller epsilon
    e_low_noise = compute_epsilon(q=0.05, noise_multiplier=1.0, steps=100, delta=delta)
    e_high_noise = compute_epsilon(q=0.05, noise_multiplier=2.5, steps=100, delta=delta)
    assert e_low_noise > e_high_noise > 0.0

    # Calibration finds sigma whose achieved epsilon is <= target_epsilon and close to target
    target_eps = 2.0
    cal_sigma = calibrate_noise_multiplier(target_eps, q=0.05, steps=120, delta=delta)
    achieved_eps = compute_epsilon(q=0.05, noise_multiplier=cal_sigma, steps=120, delta=delta)
    assert achieved_eps <= target_eps + 1e-3
    assert achieved_eps >= target_eps * 0.95


def test_per_sample_gradient_clipping_bounds_norm() -> None:
    """Verify per-sample gradient clipping bounds every sample's L2 norm to <= C."""
    model = build_model(
        "gru",
        input_dim=8,
        num_classes=2,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        seq_mode="single_step",
        num_chunks=1,
    )
    params = {k: v.detach() for k, v in model.named_parameters()}
    x = torch.randn(16, 8) * 5.0
    y = torch.randint(0, 2, (16,))

    # Verify functional_gru_forward matches GRUClassifier.forward
    with torch.no_grad():
        ref_logits = model(x)
        func_logits = functional_gru_forward(model, params, x)
        assert torch.allclose(ref_logits, func_logits, atol=1e-5)

    per_sample = compute_per_sample_grads(model, params, x, y)
    clip_norm = 0.75

    # Check single-sample clipped norm bound by testing 1-sample batches
    for idx in range(4):
        single_ps = {k: v[idx : idx + 1] for k, v in per_sample.items()}
        summed_single, orig_norm = clip_and_sum_per_sample_grads(single_ps, clip_norm)
        clipped_sq = sum(float(t.pow(2).sum().item()) for t in summed_single.values())
        clipped_l2 = float(np.sqrt(clipped_sq))
        assert clipped_l2 <= clip_norm + 1e-5
        if float(orig_norm[0].item()) > clip_norm:
            assert np.isclose(clipped_l2, clip_norm, atol=1e-4)


def test_gaussian_noise_scale_matches_formula() -> None:
    """Verify added Gaussian noise empirical std matches sigma * C."""
    zero_grad = {"w": torch.zeros(20000, dtype=torch.float32)}
    sigma = 1.8
    clip_c = 1.25
    gen = torch.Generator(device="cpu")
    gen.manual_seed(12345)

    noisy = add_gaussian_noise(
        zero_grad, noise_multiplier=sigma, max_grad_norm=clip_c, generator=gen
    )
    empirical_std = float(noisy["w"].std().item())
    expected_std = sigma * clip_c
    assert abs(empirical_std - expected_std) / expected_std < 0.03


def test_sensitivity_scores_and_budget_sum_constraint() -> None:
    """Verify sensitivity scores in [0.5, 1.0], budget sum, and inverse-sensitivity ordering."""
    rng = np.random.default_rng(42)
    tenant_labels = {
        "tenant_1_finance": np.concatenate(
            [
                np.zeros(700, dtype=np.int64),
                np.full(250, 4, dtype=np.int64),
                np.full(50, 7, dtype=np.int64),
            ]
        ),
        "tenant_2_healthcare": np.concatenate(
            [
                np.zeros(600, dtype=np.int64),
                np.full(380, 1, dtype=np.int64),
                np.full(20, 6, dtype=np.int64),
            ]
        ),
        "tenant_3_education": rng.integers(0, 5, size=1000, dtype=np.int64),
    }
    sens = compute_sensitivity_scores(tenant_labels, num_classes=8)
    for ts in sens.values():
        assert 0.5 <= ts.score <= 1.0
    # Finance and Healthcare have higher domain profile (1.0) than Education (0.5)
    assert sens["tenant_1_finance"].score > sens["tenant_3_education"].score
    assert sens["tenant_2_healthcare"].score > sens["tenant_3_education"].score

    scores_map = {k: v.score for k, v in sens.items()}
    eps_total = 6.0
    adaptive_eps = allocate_budgets(scores_map, eps_total, policy="adaptive")
    uniform_eps = allocate_budgets(scores_map, eps_total, policy="uniform")

    assert np.isclose(sum(adaptive_eps.values()), eps_total, atol=1e-9)
    assert np.isclose(sum(uniform_eps.values()), eps_total, atol=1e-9)

    # Higher sensitivity score receives strictly smaller epsilon (stronger privacy)
    assert adaptive_eps["tenant_1_finance"] < adaptive_eps["tenant_3_education"]
    assert adaptive_eps["tenant_2_healthcare"] < adaptive_eps["tenant_3_education"]


def test_dp_fl_client_and_mia_evaluation() -> None:
    """Verify DPFLClient trains, tracks epsilon <= target, and MIA returns valid metrics."""
    rng = np.random.default_rng(7)
    X_tr = rng.normal(size=(160, 6)).astype(np.float32)
    y_tr = (X_tr[:, 0] + X_tr[:, 1] > 0).astype(np.int64)
    X_te = rng.normal(size=(80, 6)).astype(np.float32)
    y_te = (X_te[:, 0] + X_te[:, 1] > 0).astype(np.int64)

    model = build_model(
        "gru",
        input_dim=6,
        num_classes=2,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        seq_mode="single_step",
        num_chunks=1,
    )
    dp_cfg = DPConfig(target_epsilon=4.0, delta=1e-5, max_grad_norm=1.0, expected_batch_size=32)
    client = DPFLClient(
        client_id="tenant_1_finance",
        client_index=0,
        model=model,
        X_train=X_tr,
        y_train=y_tr,
        dp_config=dp_cfg,
        rounds=2,
        local_epochs=1,
        X_test=X_te,
        y_test=y_te,
        device=torch.device("cpu"),
    )
    init_flat = model.get_flat_params()
    upd1 = client.local_train(init_flat, round_idx=1, local_epochs=1, lr=1e-3, seed=0)
    upd2 = client.local_train(upd1.flat_params, round_idx=2, local_epochs=1, lr=1e-3, seed=0)

    assert upd2.metadata["dp"] is True
    assert 0.0 < client.epsilon_spent <= 4.0 + 1e-2

    mia_res = evaluate_membership_inference(
        client.model,
        X_tr,
        y_tr,
        X_te,
        y_te,
        device=torch.device("cpu"),
    )
    assert 0.5 <= mia_res["mia_auc"] <= 1.0
    assert 0.5 <= mia_res["mia_accuracy"] <= 1.0
