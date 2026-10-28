"""Step 00 smoke tests for package layout, safety ranges, and hardware probe."""

from __future__ import annotations

import ipaddress
import json
from pathlib import Path

import trustfed
from scripts.probe_hardware import collect_probe, format_markdown


def test_package_version() -> None:
    """Verify core package imports and exposes a valid version string."""
    assert isinstance(trustfed.__version__, str)
    parts = trustfed.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_testbed_subnets_are_private_and_disjoint() -> None:
    """Verify all tenant subnets lie inside the isolated RFC1918 testbed CIDR."""
    parent = ipaddress.ip_network(trustfed.TESTBED_SUBNET_CIDR)
    assert parent.is_private

    tenant_nets = [ipaddress.ip_network(cidr) for cidr in trustfed.TENANT_SUBNETS.values()]
    assert len(tenant_nets) >= 3

    for idx, net in enumerate(tenant_nets):
        assert net.is_private
        assert net.subnet_of(parent)
        for other in tenant_nets[idx + 1 :]:
            assert not net.overlaps(other)


def test_hardware_probe_structure() -> None:
    """Verify hardware probe returns required sections and renders Markdown."""
    report = collect_probe()
    required_keys = {
        "timestamp_utc",
        "os",
        "python",
        "cpu",
        "ram",
        "disks",
        "gpus",
        "virtualization",
        "packages",
    }
    assert required_keys.issubset(report.keys())
    assert report["ram"]["total_gb"] > 0
    assert report["cpu"]["logical_threads"] >= 1
    assert len(report["disks"]) >= 1

    md = format_markdown(report)
    assert "## Hardware & Runtime Probe Output" in md
    assert "| `torch` |" in md


def test_linux_sdn_smoke_test_artifacts_present() -> None:
    """Verify Ubuntu 22.04 Mininet + OVS + OS-Ken smoke test logs are recorded and passed."""
    root = Path(__file__).resolve().parents[1]
    summary_path = root / "results" / "00_environment" / "smoke_test_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["loss_pct"] == 0.0
    assert summary["pingall_received"] == summary["pingall_sent"] == 6
    for rel_log in summary["source_logs"]:
        assert (root / rel_log).exists()
