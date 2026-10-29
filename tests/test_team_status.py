"""Unit tests for scripts/team_status.py collaboration and handoff checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.team_status import (
    check_installed_packages,
    format_status_report,
    parse_team_checklist,
    verify_dataset_manifest,
)


def test_parse_team_checklist_reads_real_file() -> None:
    """Verify docs/TEAM_CHECKLIST.md parses handoff notes and sub-tasks."""
    root = Path(__file__).resolve().parents[1]
    summary = parse_team_checklist(root / "docs" / "TEAM_CHECKLIST.md")
    assert summary.exists
    assert summary.total_count >= 40
    assert summary.completed_count >= 7
    assert len(summary.handoff_notes) >= 3
    assert len(summary.next_tasks) >= 1


def test_verify_dataset_manifest_detects_present_and_missing(tmp_path: Path) -> None:
    """Verify manifest checker detects OK, missing, and mismatched dataset files."""
    raw_dir = tmp_path / "data" / "raw" / "insdn"
    raw_dir.mkdir(parents=True)

    good_file = raw_dir / "Normal_data.csv"
    payload = b"flow_dur,pkt_cnt,label\n10,2,Benign\n"
    good_file.write_bytes(payload)
    good_sha = hashlib.sha256(payload).hexdigest()
    missing_sha = "a" * 64

    manifest = tmp_path / "data" / "MANIFEST.md"
    manifest.write_text(
        "\n".join(
            [
                "# Dataset Manifest",
                "",
                "| File | Size | SHA-256 | Source |",
                "|---|---|---|---|",
                f"| `data/raw/insdn/Normal_data.csv` | {len(payload)} | `{good_sha}` | official |",
                f"| `data/raw/insdn/OVS.csv` | 999 | `{missing_sha}` | official |",
            ]
        ),
        encoding="utf-8",
    )

    res = verify_dataset_manifest(tmp_path, verify_hashes=True)
    assert res.manifest_exists
    assert res.total_entries == 2
    assert res.present_ok == 1
    assert res.missing_files == ["data/raw/insdn/OVS.csv"]
    assert res.size_mismatches == []
    assert res.hash_mismatches == []


def test_installed_packages_and_report_formatting() -> None:
    """Verify required packages check passes in .venv and report formats cleanly."""
    pkg = check_installed_packages()
    assert pkg["all_present"] is True

    fake_status = {
        "git": {
            "is_git_repo": True,
            "branch": "main",
            "head_sha": "abc1234",
            "dirty_count": 0,
            "has_origin": True,
            "origin_url": "https://github.com/naimurhamim/TrustFed-SDN.git",
            "remote_branch_exists": True,
            "fetch_status": "ok",
            "ahead": 0,
            "behind": 0,
            "incoming_commits": [],
            "incoming_files": [],
        },
        "packages": pkg,
        "datasets": {
            "manifest_exists": False,
            "total_entries": 0,
            "present_ok": 0,
            "missing_files": [],
            "size_mismatches": [],
            "hash_mismatches": [],
        },
        "checklist": {
            "exists": True,
            "handoff_notes": ["Last Updated By: `@Naimur`"],
            "completed_count": 7,
            "total_count": 50,
            "next_tasks": ["[Step 00] 00.8 Explicit user approval"],
            "per_step": [],
        },
    }
    text = format_status_report(fake_status)
    assert "UP TO DATE with GitHub" in text
    assert "7/50 sub-tasks completed" in text
