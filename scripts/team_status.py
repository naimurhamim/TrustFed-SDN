"""Team synchronization, environment, dataset manifest, and checklist checker.

Run before starting and after finishing work sessions (`make status`) so
teammates (@Naimur and @Pritom) can compare local vs GitHub state, detect
missing Python packages or git-ignored dataset files, and see the next
unchecked sub-tasks in docs/TEAM_CHECKLIST.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

REQUIRED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "torch",
    "PyYAML",
    "matplotlib",
    "psutil",
    "cryptography",
    "pytest",
    "ruff",
    "black",
)

_CHECKBOX_RE = re.compile(r"^- \[(?P<mark>[ xX])\]\s+(?P<text>.+)$")
_STEP_HEADER_RE = re.compile(r"^###\s+(?P<step>Step\s+\d+.*)$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass
class ChecklistSummary:
    """Parsed summary of docs/TEAM_CHECKLIST.md."""

    exists: bool
    handoff_notes: list[str]
    completed_count: int
    total_count: int
    next_tasks: list[str]
    per_step: list[dict[str, Any]]


@dataclass
class ManifestCheckResult:
    """Verification status of git-ignored dataset files listed in data/MANIFEST.md."""

    manifest_exists: bool
    total_entries: int
    present_ok: int
    missing_files: list[str]
    size_mismatches: list[str]
    hash_mismatches: list[str]


def _run_git(
    repo_root: Path,
    args: list[str],
    timeout: float = 15.0,
) -> tuple[int, str, str]:
    """Execute a git command non-interactively so GUI prompts never hang."""
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "never"
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, "", str(exc)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def parse_team_checklist(checklist_path: Path, max_next: int = 4) -> ChecklistSummary:
    """Parse handoff notes, completion counts, and next unchecked items."""
    if not checklist_path.exists():
        return ChecklistSummary(
            exists=False,
            handoff_notes=[],
            completed_count=0,
            total_count=0,
            next_tasks=[],
            per_step=[],
        )

    lines = checklist_path.read_text(encoding="utf-8").splitlines()
    in_handoff = False
    handoff_notes: list[str] = []
    current_step = "Unassigned"
    step_counts: dict[str, dict[str, int]] = {}
    completed = 0
    total = 0
    next_tasks: list[str] = []

    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("## Latest Handoff Note"):
            in_handoff = True
            continue
        if in_handoff and stripped.startswith("## "):
            in_handoff = False

        if in_handoff and stripped.startswith("- "):
            handoff_notes.append(stripped[2:].strip())
            continue

        step_match = _STEP_HEADER_RE.match(stripped)
        if step_match:
            current_step = step_match.group("step")
            step_counts.setdefault(current_step, {"done": 0, "total": 0})
            continue

        cb_match = _CHECKBOX_RE.match(stripped)
        if cb_match:
            total += 1
            step_counts.setdefault(current_step, {"done": 0, "total": 0})
            step_counts[current_step]["total"] += 1
            is_done = cb_match.group("mark").lower() == "x"
            if is_done:
                completed += 1
                step_counts[current_step]["done"] += 1
            elif len(next_tasks) < max_next:
                short_step = current_step.split("—")[0].strip()
                next_tasks.append(f"[{short_step}] {cb_match.group('text')}")

    per_step = [
        {"step": step, "done": counts["done"], "total": counts["total"]}
        for step, counts in step_counts.items()
    ]
    return ChecklistSummary(
        exists=True,
        handoff_notes=handoff_notes,
        completed_count=completed,
        total_count=total,
        next_tasks=next_tasks,
        per_step=per_step,
    )


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a local file in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset_manifest(
    repo_root: Path,
    manifest_rel: str = "data/MANIFEST.md",
    verify_hashes: bool = False,
) -> ManifestCheckResult:
    """Check local existence, byte size, and optional SHA-256 against data/MANIFEST.md.

    Expected Markdown table row format (created in Step 01):
    | path | size_bytes | sha256 | ... |
    """
    manifest_path = repo_root / manifest_rel
    if not manifest_path.exists():
        return ManifestCheckResult(
            manifest_exists=False,
            total_entries=0,
            present_ok=0,
            missing_files=[],
            size_mismatches=[],
            hash_mismatches=[],
        )

    entries = 0
    ok_count = 0
    missing: list[str] = []
    size_bad: list[str] = []
    hash_bad: list[str] = []

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.startswith("|---"):
            continue
        cells = [c.strip().strip("`") for c in stripped.strip("|").split("|")]
        if len(cells) < 3:
            continue
        rel_candidate, size_cell, sha_cell = cells[0], cells[1], cells[2]
        if rel_candidate.lower() in {"file", "path", "relative_path"}:
            continue
        if not _SHA256_RE.match(sha_cell):
            continue

        entries += 1
        target = repo_root / rel_candidate
        if not target.exists():
            missing.append(rel_candidate)
            continue

        clean_size = size_cell.replace(",", "").strip()
        if clean_size.isdigit():
            actual_size = target.stat().st_size
            if actual_size != int(clean_size):
                size_bad.append(f"{rel_candidate} (expected {clean_size} B, found {actual_size} B)")
                continue

        if verify_hashes:
            actual_sha = _sha256_file(target)
            if actual_sha.lower() != sha_cell.lower():
                hash_bad.append(rel_candidate)
                continue

        ok_count += 1

    return ManifestCheckResult(
        manifest_exists=True,
        total_entries=entries,
        present_ok=ok_count,
        missing_files=missing,
        size_mismatches=size_bad,
        hash_mismatches=hash_bad,
    )


def check_installed_packages(
    required: tuple[str, ...] = REQUIRED_PACKAGES,
) -> dict[str, Any]:
    """Verify all required packages are installed in the active Python environment."""
    installed: dict[str, str] = {}
    missing: list[str] = []
    for pkg in required:
        try:
            installed[pkg] = version(pkg)
        except PackageNotFoundError:
            missing.append(pkg)
    return {
        "all_present": len(missing) == 0,
        "installed_count": len(installed),
        "missing": missing,
    }


def check_git_sync(repo_root: Path, do_fetch: bool = True) -> dict[str, Any]:
    """Compare local git branch state against origin remote."""
    rc, branch, _ = _run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    if rc != 0:
        return {"is_git_repo": False}

    _, head_sha, _ = _run_git(repo_root, ["rev-parse", "--short", "HEAD"])
    _, porcelain, _ = _run_git(repo_root, ["status", "--porcelain"])
    dirty_files = [line for line in porcelain.splitlines() if line.strip()]

    rc_rem, remote_url, _ = _run_git(repo_root, ["remote", "get-url", "origin"])
    has_origin = rc_rem == 0 and bool(remote_url)

    info: dict[str, Any] = {
        "is_git_repo": True,
        "branch": branch,
        "head_sha": head_sha,
        "dirty_count": len(dirty_files),
        "dirty_files": dirty_files[:15],
        "has_origin": has_origin,
        "origin_url": remote_url if has_origin else None,
        "fetch_status": "skipped",
        "ahead": 0,
        "behind": 0,
        "incoming_commits": [],
        "incoming_files": [],
    }

    if not has_origin:
        return info

    if do_fetch:
        rc_f, _, err_f = _run_git(repo_root, ["fetch", "origin", "--quiet"], timeout=20.0)
        info["fetch_status"] = "ok" if rc_f == 0 else f"fetch_failed: {err_f}"

    remote_ref = f"origin/{branch}"
    rc_ref, _, _ = _run_git(repo_root, ["rev-parse", "--verify", remote_ref])
    if rc_ref != 0:
        info["remote_branch_exists"] = False
        return info

    info["remote_branch_exists"] = True
    rc_cnt, counts, _ = _run_git(
        repo_root, ["rev-list", "--left-right", "--count", f"HEAD...{remote_ref}"]
    )
    if rc_cnt == 0 and len(counts.split()) == 2:
        ahead_s, behind_s = counts.split()
        info["ahead"] = int(ahead_s)
        info["behind"] = int(behind_s)

    if info["behind"] > 0:
        _, log_out, _ = _run_git(repo_root, ["log", f"HEAD..{remote_ref}", "--oneline", "-n", "10"])
        _, diff_out, _ = _run_git(repo_root, ["diff", "--name-status", f"HEAD..{remote_ref}"])
        info["incoming_commits"] = [x for x in log_out.splitlines() if x.strip()]
        info["incoming_files"] = [x for x in diff_out.splitlines() if x.strip()]

    return info


def collect_team_status(
    repo_root: Path | None = None,
    do_fetch: bool = True,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    """Gather complete team sync, environment, dataset, and checklist status."""
    root = repo_root or Path(__file__).resolve().parents[1]
    git_info = check_git_sync(root, do_fetch=do_fetch)
    pkg_info = check_installed_packages()
    manifest_info = verify_dataset_manifest(root, verify_hashes=verify_hashes)
    checklist_info = parse_team_checklist(root / "docs" / "TEAM_CHECKLIST.md")

    return {
        "git": git_info,
        "packages": pkg_info,
        "datasets": asdict(manifest_info),
        "checklist": asdict(checklist_info),
    }


def format_status_report(status: dict[str, Any]) -> str:
    """Format the team status dictionary into a concise terminal report."""
    git = status["git"]
    pkg = status["packages"]
    ds = status["datasets"]
    cl = status["checklist"]

    lines = ["================ TRUSTFED-SDN TEAM STATUS ================"]

    # 1. Git Sync
    if not git.get("is_git_repo"):
        lines.append("[GIT] Not a git repository.")
    else:
        lines.append(
            f"[GIT] Branch: {git['branch']} @ {git['head_sha']} | "
            f"Uncommitted changes: {git['dirty_count']}"
        )
        if not git["has_origin"]:
            lines.append("      Remote 'origin': NOT configured yet.")
        elif not git.get("remote_branch_exists"):
            lines.append(
                f"      Remote 'origin': {git['origin_url']} "
                "(remote branch not pushed yet -> run 'git push -u origin main')"
            )
        else:
            lines.append(
                f"      Remote 'origin': {git['origin_url']} (fetch: {git['fetch_status']})"
            )
            if git["behind"] == 0 and git["ahead"] == 0:
                lines.append("      Sync state: UP TO DATE with GitHub.")
            else:
                if git["behind"] > 0:
                    lines.append(
                        f"      ACTION NEEDED: Remote has {git['behind']} new commit(s)! "
                        "Run 'git pull --ff-only' before editing."
                    )
                    for commit_line in git["incoming_commits"]:
                        lines.append(f"        + {commit_line}")
                    for file_line in git["incoming_files"][:10]:
                        lines.append(f"        file: {file_line}")
                if git["ahead"] > 0:
                    lines.append(
                        f"      Note: Local branch is ahead by {git['ahead']} unpushed commit(s). "
                        "Run 'git push' to share with your teammate."
                    )

    # 2. Packages
    if pkg["all_present"]:
        lines.append(
            f"[ENV] Python packages: OK ({pkg['installed_count']} required packages installed)."
        )
    else:
        missing_str = ", ".join(pkg["missing"])
        lines.append(
            f"[ENV] ACTION NEEDED: Missing packages ({missing_str}) -> Run '.\\make.cmd install'"
        )

    # 3. Datasets
    if not ds["manifest_exists"]:
        lines.append("[DATA] data/MANIFEST.md not created yet (scheduled for Step 01).")
    elif not ds["missing_files"] and not ds["size_mismatches"] and not ds["hash_mismatches"]:
        lines.append(
            f"[DATA] Local dataset files: OK ({ds['present_ok']}/{ds['total_entries']} verified)."
        )
    else:
        lines.append(
            f"[DATA] ACTION NEEDED: {len(ds['missing_files'])} missing, "
            f"{len(ds['size_mismatches'])} size mismatch, "
            f"{len(ds['hash_mismatches'])} hash mismatch vs data/MANIFEST.md!"
        )
        for item in (ds["missing_files"] + ds["size_mismatches"] + ds["hash_mismatches"])[:10]:
            lines.append(f"        ! {item}")

    # 4. Checklist & Handoff
    if not cl["exists"]:
        lines.append("[PLAN] docs/TEAM_CHECKLIST.md not found.")
    else:
        done_n, total_n = cl["completed_count"], cl["total_count"]
        lines.append(f"[PLAN] Checklist progress: {done_n}/{total_n} sub-tasks completed.")
        if cl["handoff_notes"]:
            lines.append("------ Latest Handoff Note ------")
            for note in cl["handoff_notes"]:
                lines.append(f"  * {note}")
        if cl["next_tasks"]:
            lines.append("------ Next Unchecked Sub-Tasks ------")
            for task in cl["next_tasks"]:
                lines.append(f"  [ ] {task}")

    lines.append("==========================================================")
    return "\n".join(lines)


def main() -> int:
    """CLI entry point for team sync and checklist report."""
    parser = argparse.ArgumentParser(description="Check GitHub sync, datasets, and team checklist.")
    parser.add_argument("--no-fetch", action="store_true", help="Skip 'git fetch origin'.")
    parser.add_argument(
        "--verify-hashes",
        action="store_true",
        help="Verify full SHA-256 digests for files in data/MANIFEST.md.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    status = collect_team_status(
        do_fetch=not args.no_fetch,
        verify_hashes=args.verify_hashes,
    )
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print(format_status_report(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
