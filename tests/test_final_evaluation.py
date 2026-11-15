"""Step 09 unit and artifact tests (`tests/test_final_evaluation.py`)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from trustfed.aggregation import SecureAggregator, build_aggregator
from trustfed.fl.aggregator import ClientUpdate

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_secure_robust_aggregator_filters_malicious_and_verifies() -> None:
    """Verify SecureAggregator(robust_screening=True) rejects outlier then verifies honest sum."""
    agg = build_aggregator("secure_robust", scale_bits=20, seed=7)
    assert isinstance(agg, SecureAggregator)
    assert agg.robust_screening is True

    torch.manual_seed(42)
    global_w = torch.zeros(128, dtype=torch.float32)
    honest_delta = torch.ones(128, dtype=torch.float32) * 0.05

    updates = [
        ClientUpdate(
            client_id=f"tenant_0{i}",
            flat_params=global_w + honest_delta + 0.002 * torch.randn(128),
            flat_delta=honest_delta,
            num_samples=500,
            train_loss=0.25,
            local_train_time_sec=0.01,
            upload_bytes=536,
            round_idx=1,
        )
        for i in range(1, 5)
    ]
    # Poisoned scaling + sign-flip tenant
    updates.append(
        ClientUpdate(
            client_id="tenant_05",
            flat_params=global_w - 10.0 * honest_delta,
            flat_delta=-10.0 * honest_delta,
            num_samples=500,
            train_loss=0.90,
            local_train_time_sec=0.01,
            upload_bytes=536,
            round_idx=1,
        )
    )

    res1 = agg.aggregate(global_w, updates)
    assert "tenant_05" in res1.rejected_clients
    assert len(res1.accepted_clients) == 4
    assert res1.metadata["verified"] is True
    assert torch.mean(res1.new_global_flat_params).item() > 0.03

    # Round 2 with active subset registered in first round must also verify cleanly
    updates_r2 = [
        ClientUpdate(
            client_id=u.client_id,
            flat_params=u.flat_params + 0.01,
            flat_delta=u.flat_delta,
            num_samples=u.num_samples,
            train_loss=u.train_loss,
            local_train_time_sec=u.local_train_time_sec,
            upload_bytes=u.upload_bytes,
            round_idx=2,
        )
        for u in updates
    ]
    res2 = agg.aggregate(res1.new_global_flat_params, updates_r2)
    assert "tenant_05" in res2.rejected_clients
    assert res2.metadata["verified"] is True


def test_final_experiment_configs_valid() -> None:
    """Verify all Step 09 final YAML configs parse and reference E1..E10."""
    final_cfg_dir = REPO_ROOT / "configs" / "experiments" / "final"
    expected_yamls = [
        "e1_baselines_fedavg.yaml",
        "e2_non_iid.yaml",
        "e3_poisoning_robust.yaml",
        "e4_adaptive_dp_mia.yaml",
        "e5_verifiable_secagg.yaml",
        "e6_ablation_ladder.yaml",
        "e7_live_testbed.yaml",
        "e8_scalability.yaml",
        "e9_cross_dataset.yaml",
        "e10_sensitivity.yaml",
        "reproduce_all.yaml",
    ]
    for name in expected_yamls:
        path = final_cfg_dir / name
        assert path.exists(), f"Missing final config: {path}"
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict)
        assert len(parsed) > 0

    master = yaml.safe_load((final_cfg_dir / "reproduce_all.yaml").read_text(encoding="utf-8"))
    for eid in [f"E{i}" for i in range(1, 11)]:
        assert eid in master["experiments"]


def test_step09_artifacts_generated() -> None:
    """Verify final results, figures (PDF+PNG), LaTeX tables, and key_numbers exist."""
    fin_dir = REPO_ROOT / "results" / "final"
    summary_path = fin_dir / "summary.json"
    if not summary_path.exists():
        # Unit test skips artifact presence check before `make reproduce` has run
        return

    key_nums = json.loads((fin_dir / "key_numbers.json").read_text(encoding="utf-8"))
    assert "live_testbed" in key_nums
    assert "secagg_tamper_detection_pct" in key_nums

    for stem in [
        "fig01_e1_e2_baselines_non_iid",
        "fig02_e3_poisoning_robustness",
        "fig03_e4_adaptive_dp_mia",
        "fig04_e5_verifiable_secagg",
        "fig05_e6_ablation_ladder",
        "fig06_e7_live_sdn_testbed",
        "fig07_e8_scalability",
        "fig08_e9_e10_transfer_sensitivity",
    ]:
        assert (REPO_ROOT / "figures" / f"{stem}.pdf").stat().st_size > 1024
        assert (REPO_ROOT / "figures" / f"{stem}.png").stat().st_size > 1024

    for tab_name in [
        "tab01_datasets_and_partitions.tex",
        "tab02_e1_e2_baselines_non_iid.tex",
        "tab03_e3_poisoning_robustness.tex",
        "tab04_e4_adaptive_dp_mia.tex",
        "tab05_e5_verifiable_secagg.tex",
        "tab06_e6_ablation_ladder.tex",
        "tab07_e7_live_testbed.tex",
        "tab08_e8_e9_e10_scalability_sensitivity.tex",
        "tab09_statistical_tests_and_targets.tex",
    ]:
        tex_path = REPO_ROOT / "tables" / tab_name
        txt = tex_path.read_text(encoding="utf-8")
        assert "\\toprule" in txt
        assert "\\bottomrule" in txt


def test_step10_qr_docs_site_and_release() -> None:
    """Verify Step 10 QR decoder, release docs, LICENSE, CITATION, and website key_numbers sync."""
    from scripts.make_qr import DEFAULT_URL, decode_and_verify_qr

    qr_dir = REPO_ROOT / "site" / "assets" / "qr"
    for fname in ("trustfed_qr.png", "trustfed_qr.svg", "trustfed_qr.pdf", "qr_verification.json"):
        p = qr_dir / fname
        assert p.exists() and p.stat().st_size > 0, f"Missing QR asset: {p}"

    decoded = decode_and_verify_qr(qr_dir / "trustfed_qr.png", expected_url=DEFAULT_URL)
    assert decoded["decoded_url"] == DEFAULT_URL
    assert decoded["rs_syndromes_zero"] is True

    for rel_path in (
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "docs/ARCHITECTURE.md",
        "docs/REPRODUCE.md",
        "docs/FINAL_AUDIT.md",
        "docs/DEVIATIONS.md",
        "site/index.html",
        "site/css/style.css",
        "site/js/script.js",
        "site/README.md",
    ):
        p = REPO_ROOT / rel_path
        assert p.exists() and p.stat().st_size > 100, f"Missing or empty release artifact: {p}"

    assert "DEV-004" in (REPO_ROOT / "docs" / "DEVIATIONS.md").read_text(encoding="utf-8")
    assert (REPO_ROOT / "site" / "data" / "key_numbers.json").read_text(encoding="utf-8") == (
        REPO_ROOT / "results" / "final" / "key_numbers.json"
    ).read_text(encoding="utf-8")

    site_kn = json.loads(
        (REPO_ROOT / "site" / "data" / "key_numbers.json").read_text(encoding="utf-8")
    )
    final_kn = json.loads(
        (REPO_ROOT / "results" / "final" / "key_numbers.json").read_text(encoding="utf-8")
    )
    assert site_kn == final_kn
