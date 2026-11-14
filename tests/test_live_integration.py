"""Unit and integration tests for Step 08 live SDN FL & mitigation pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from testbed.fl import (
    LiveAggregationHTTPServer,
    LiveAggregationServer,
    LiveTenantFLClient,
    LiveTenantHTTPClient,
    build_live_gru,
)
from testbed.ryu_apps.model_deploy import (
    AsyncFlowInferenceWorker,
    FlowClassificationJob,
    MitigationPolicyManager,
)
from testbed.traffic.safety_guard import SafetyGuardError
from trustfed.verify import generate_tenant_keys

TENANTS = ("tenant_1_finance", "tenant_2_healthcare", "tenant_3_education")


def _make_synthetic_clients(
    *,
    attack_tenant: str | None = None,
    attack_name: str | None = None,
    seed: int = 42,
) -> tuple[dict[str, LiveTenantFLClient], LiveAggregationServer, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    materials, registry, group_key = generate_tenant_keys(TENANTS, seed=seed)

    clients: dict[str, LiveTenantFLClient] = {}
    X_test_list: list[np.ndarray] = []
    y_test_list: list[np.ndarray] = []

    for idx, tid in enumerate(TENANTS):
        n_tr = 96
        n_te = 24
        y_tr = np.array([0] * (n_tr // 2) + [1] * (n_tr // 2), dtype=np.int64)
        y_te = np.array([0] * (n_te // 2) + [1] * (n_te // 2), dtype=np.int64)
        X_tr = rng.normal(loc=0.0, scale=0.4, size=(n_tr, 19)).astype(np.float32)
        X_te = rng.normal(loc=0.0, scale=0.4, size=(n_te, 19)).astype(np.float32)
        X_tr[y_tr == 1] += 2.2
        X_te[y_te == 1] += 2.2

        is_atk_t = attack_tenant == tid and attack_name is not None
        clients[tid] = LiveTenantFLClient(
            tenant_id=tid,
            client_index=idx,
            X_train=X_tr,
            y_train=y_tr,
            X_test=X_te,
            y_test=y_te,
            key_material=materials[tid],
            registry=registry,
            group_key=group_key,
            rounds=5,
            local_epochs=2,
            dp_epsilon=None,
            attack_name=attack_name if is_atk_t else None,
            attack_kwargs={"scale_factor": 10.0, "sign_flip": True} if is_atk_t else None,
            device=torch.device("cpu"),
        )
        X_test_list.append(X_te)
        y_test_list.append(y_te)

    server = LiveAggregationServer(
        build_live_gru(19),
        registry=registry,
        robust_defense=True,
        device=torch.device("cpu"),
    )
    return clients, server, np.concatenate(X_test_list), np.concatenate(y_test_list)


def test_live_http_three_tenant_honest_round() -> None:
    clients, agg_server, X_te, y_te = _make_synthetic_clients(seed=11)
    with LiveAggregationHTTPServer(agg_server, host="127.0.0.1", port=0) as http_srv:
        rpc = LiveTenantHTTPClient(http_srv.base_url)
        global_flat, cur_round, _, _ = rpc.fetch_global_model()
        assert cur_round == 0

        round_idx = 1
        for tid in TENANTS:
            res = clients[tid].train_and_prepare_submission(
                global_flat, round_idx=round_idx, participants=list(TENANTS), seed=11
            )
            up_ms, up_bytes = rpc.submit_update(res.submission, res.update)
            assert up_ms >= 0.0
            assert up_bytes > 0

        agg_resp = rpc.trigger_aggregate(round_idx, use_verifiable_secagg=True)
        assert agg_resp["accepted_clients"] == list(TENANTS)
        assert agg_resp["rejected_clients"] == []
        proof = agg_resp["proof_obj"]

        for tid in TENANTS:
            verdict = clients[tid].verify_and_deploy(
                proof, expected_round_idx=round_idx, expected_clients=list(TENANTS)
            )
            assert verdict.ok
            assert clients[tid].verified_round == round_idx

        metrics = agg_server.evaluate_global(X_te, y_te)
        assert metrics["f1_macro"] > 0.70


def test_live_malicious_tenant_filtering_over_http() -> None:
    clients, agg_server, _, _ = _make_synthetic_clients(
        attack_tenant="tenant_3_education", attack_name="scaling", seed=17
    )
    with LiveAggregationHTTPServer(agg_server, host="127.0.0.1", port=0) as http_srv:
        rpc = LiveTenantHTTPClient(http_srv.base_url)
        global_flat, _, _, _ = rpc.fetch_global_model()
        round_idx = 1

        subs_by_tid = {}
        for tid in TENANTS:
            res = clients[tid].train_and_prepare_submission(
                global_flat, round_idx=round_idx, participants=list(TENANTS), seed=17
            )
            subs_by_tid[tid] = res
            rpc.submit_update(res.submission, res.update)

        screen = rpc.screen_round(round_idx)
        assert "tenant_3_education" in screen["rejected_clients"]
        honest_cohort = screen["accepted_clients"]
        assert len(honest_cohort) == 2

        for tid in honest_cohort:
            resigned = clients[tid].resign_for_cohort(
                subs_by_tid[tid].update.flat_params,
                num_samples=subs_by_tid[tid].update.num_samples,
                round_idx=round_idx,
                participants=honest_cohort,
                seed=17,
            )
            rpc.submit_resigned_cohort(resigned)

        agg_resp = rpc.trigger_aggregate(round_idx, use_verifiable_secagg=True)
        proof = agg_resp["proof_obj"]
        for tid in honest_cohort:
            verdict = clients[tid].verify_and_deploy(
                proof, expected_round_idx=round_idx, expected_clients=honest_cohort
            )
            assert verdict.ok


@pytest.mark.parametrize(
    "tamper_mode",
    ["modified_aggregate", "dropped_tenant", "replayed_update", "substituted_update"],
)
def test_live_tampered_aggregate_rejected_and_preserves_weights(tamper_mode: str) -> None:
    clients, agg_server, _, _ = _make_synthetic_clients(seed=23)
    with LiveAggregationHTTPServer(agg_server, host="127.0.0.1", port=0) as http_srv:
        rpc = LiveTenantHTTPClient(http_srv.base_url)

        # Round 1 (honest) populates verified baseline + previous submissions for replay test
        g1, _, _, _ = rpc.fetch_global_model()
        for tid in TENANTS:
            res1 = clients[tid].train_and_prepare_submission(
                g1, round_idx=1, participants=list(TENANTS), seed=23
            )
            rpc.submit_update(res1.submission, res1.update)
        r1 = rpc.trigger_aggregate(1, use_verifiable_secagg=True)
        for tid in TENANTS:
            assert (
                clients[tid]
                .verify_and_deploy(
                    r1["proof_obj"], expected_round_idx=1, expected_clients=list(TENANTS)
                )
                .ok
            )

        baseline_weights = clients["tenant_1_finance"].deployed_flat_params.clone()

        # Round 2 injects requested server tamper attack
        g2, _, _, _ = rpc.fetch_global_model()
        for tid in TENANTS:
            res2 = clients[tid].train_and_prepare_submission(
                g2, round_idx=2, participants=list(TENANTS), seed=23
            )
            rpc.submit_update(res2.submission, res2.update)
        r2 = rpc.trigger_aggregate(2, use_verifiable_secagg=True, tamper_mode=tamper_mode)
        for tid in TENANTS:
            verdict = clients[tid].verify_and_deploy(
                r2["proof_obj"], expected_round_idx=2, expected_clients=list(TENANTS)
            )
            assert not verdict.ok
            assert clients[tid].verified_round == 1
        assert torch.allclose(clients["tenant_1_finance"].deployed_flat_params, baseline_weights)


def test_async_flow_worker_and_mitigation_policy(tmp_path: Path) -> None:
    blocklist_file = tmp_path / "blocklist_t1.json"
    policy = MitigationPolicyManager(
        blocklist_path=blocklist_file,
        ovs_bridge="s1",
        apply_ovs_cli=False,
        hard_timeout=30,
    )

    with pytest.raises(SafetyGuardError):
        policy.enforce_drop_rule(tenant_id="tenant_1_finance", ipv4_src="8.8.8.8", prob_attack=0.99)
    with pytest.raises(SafetyGuardError):
        policy.enforce_drop_rule(
            tenant_id="tenant_1_finance", ipv4_src="192.168.1.50", prob_attack=0.99
        )

    model = build_live_gru(19)
    worker = AsyncFlowInferenceWorker(model, policy, attack_threshold=0.0)
    worker.start()
    try:
        deploy_ms = worker.update_model_weights(model.get_flat_params(), round_idx=3)
        assert deploy_ms >= 0.0
        assert worker.deployed_round == 3

        worker.submit_flow(
            FlowClassificationJob(
                flow_id="flow_atk_1",
                tenant_id="tenant_1_finance",
                src_ip="10.10.1.100",
                dst_ip="10.10.1.50",
                feature_vector=np.ones(19, dtype=np.float32),
                bridge="s1",
                ground_truth_label=1,
            )
        )
        worker.wait_until_empty(timeout=5.0)
    finally:
        worker.stop()

    summary = worker.summarize_mitigation()
    assert summary["total_flows"] == 1
    assert summary["true_positive_blocks"] == 1
    assert blocklist_file.exists()
    rules = json.loads(blocklist_file.read_text(encoding="utf-8"))
    assert len(rules) == 1 and rules[0]["ipv4_src"] == "10.10.1.100"

    # Verify automatic TTL expiry purges the temporary rule cleanly
    expired = policy.purge_expired(now_ts=time.time() + 60.0)
    assert len(expired) == 1
    assert len(policy.active_rules) == 0
    assert json.loads(blocklist_file.read_text(encoding="utf-8")) == []


def test_live_incomplete_round_abort_and_clean_retry() -> None:
    import urllib.error

    clients, agg_server, _, _ = _make_synthetic_clients(seed=29)
    with LiveAggregationHTTPServer(agg_server, host="127.0.0.1", port=0) as http_srv:
        rpc = LiveTenantHTTPClient(http_srv.base_url)

        # Complete Honest Round 1 to establish verified reference weights
        g1, _, _, _ = rpc.fetch_global_model()
        for tid in TENANTS:
            res1 = clients[tid].train_and_prepare_submission(
                g1, round_idx=1, participants=list(TENANTS), seed=29
            )
            rpc.submit_update(res1.submission, res1.update)
        r1 = rpc.trigger_aggregate(1, use_verifiable_secagg=True, expected_clients=list(TENANTS))
        for tid in TENANTS:
            assert (
                clients[tid]
                .verify_and_deploy(
                    r1["proof_obj"], expected_round_idx=1, expected_clients=list(TENANTS)
                )
                .ok
            )

        ref_weights = clients["tenant_1_finance"].deployed_flat_params.clone()

        # Round 2 attempt #1: simulate tenant dropout (only 2 of 3 tenants submit)
        g2, _, _, _ = rpc.fetch_global_model()
        for tid in TENANTS[:2]:
            res_partial = clients[tid].train_and_prepare_submission(
                g2, round_idx=2, participants=list(TENANTS), seed=29
            )
            rpc.submit_update(res_partial.submission, res_partial.update)

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            rpc.trigger_aggregate(2, use_verifiable_secagg=True, expected_clients=list(TENANTS))
        assert exc_info.value.code == 409
        assert agg_server.current_round == 1
        assert torch.allclose(agg_server.get_global_flat_params(), ref_weights)

        # Round 2 attempt #2 (clean retry after explicit abort): all 3 tenants submit and commit
        abort_resp = rpc.abort_round(2)
        assert abort_resp["ok"] is True
        g2_retry, cur_r, _, _ = rpc.fetch_global_model()
        assert cur_r == 1
        for tid in TENANTS:
            res2 = clients[tid].train_and_prepare_submission(
                g2_retry, round_idx=2, participants=list(TENANTS), seed=29
            )
            rpc.submit_update(res2.submission, res2.update)

        r2 = rpc.trigger_aggregate(2, use_verifiable_secagg=True, expected_clients=list(TENANTS))
        for tid in TENANTS:
            assert (
                clients[tid]
                .verify_and_deploy(
                    r2["proof_obj"], expected_round_idx=2, expected_clients=list(TENANTS)
                )
                .ok
            )
            assert clients[tid].verified_round == 2
        assert agg_server.current_round == 2
