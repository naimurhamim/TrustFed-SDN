"""Generate data/MANIFEST.md for git-ignored raw datasets.

Lists every raw file with relative path, byte size, SHA-256, CSV data-row count,
source URL, and download date, so teammates can verify identical local copies
via `make status` (scripts/team_status.py).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCES: dict[str, dict[str, str]] = {
    "insdn": {
        "name": "InSDN",
        "url": "http://aseados.ucd.ie/datasets/SDN/",
        "license": "CC BY-NC 4.0",
        "citation": (
            "M. S. Elsayed, N.-A. Le-Khac, A. Jurcut, 'InSDN: A Novel SDN Intrusion Dataset', "
            "IEEE Access, vol. 8, pp. 165263-165284, 2020, DOI 10.1109/ACCESS.2020.3022633"
        ),
    },
    "aseados_sdn_iot": {
        "name": "ASEADOS-SDN-IoT (2026)",
        "url": "https://aseados.ucd.ie/datasets/SDN-IoT/",
        "license": "CC BY-NC 4.0 (UCD ASEADOS Lab)",
        "citation": (
            "T. Yasarathna et al., 'ASEADOS-SDN-IoT: A novel SDN-IoT network intrusion detection "
            "dataset and framework', Internet of Things (Elsevier), 2026, "
            "DOI 10.1016/j.iot.2026.101891"
        ),
    },
    "cicids2017": {
        "name": "CICIDS2017 (MachineLearningCSV)",
        "url": "https://www.unb.ca/cic/datasets/ids-2017.html",
        "license": "See UNB CIC dataset terms (academic use, cite paper)",
        "citation": (
            "I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani, 'Toward Generating a New Intrusion "
            "Detection Dataset and Intrusion Traffic Characterization', ICISSP 2018"
        ),
    },
    "cicddos2019": {
        "name": "CICDDoS2019",
        "url": "https://www.unb.ca/cic/datasets/ddos-2019.html",
        "license": "See UNB CIC dataset terms",
        "citation": "Sharafaldin et al., ICCST 2019",
    },
}

SKIP_NAMES = {".gitkeep", "README.md", "desktop.ini", "Thumbs.db"}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> int:
    """Count data rows (lines minus one header) of a CSV file without parsing it."""
    lines = 0
    last_byte = b"\n"
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            lines += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if last_byte != b"\n":
        lines += 1
    return max(lines - 1, 0)


def collect_entries(raw_dir: Path) -> list[dict[str, str]]:
    """Collect manifest rows for every file under data/raw/<dataset>/."""
    entries: list[dict[str, str]] = []
    for path in sorted(p for p in raw_dir.rglob("*") if p.is_file()):
        if path.name in SKIP_NAMES:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        dataset_key = path.relative_to(raw_dir).parts[0]
        source = SOURCES.get(dataset_key, {"url": "unknown"})
        rows = str(count_csv_rows(path)) if path.suffix.lower() == ".csv" else "-"
        modified = datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime("%Y-%m-%d")
        entries.append(
            {
                "path": rel,
                "size": str(path.stat().st_size),
                "sha256": sha256_file(path),
                "rows": rows,
                "url": source["url"],
                "date": modified,
            }
        )
        print(f"  hashed {rel}", file=sys.stderr)
    return entries


def render_manifest(entries: list[dict[str, str]]) -> str:
    """Render the manifest Markdown document."""
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# Data Manifest",
        "",
        "Raw datasets are git-ignored. This tracked manifest lets each teammate verify an",
        "identical local copy with `make status` (size check) or",
        "`python scripts/team_status.py --verify-hashes` (full SHA-256 check).",
        "",
        f"- **Generated (UTC):** `{now}` by `scripts/make_data_manifest.py`",
        "- **Rows** = CSV data lines excluding the first header line (raw, before cleaning).",
        "- **Date** = local file modification date (download or extraction date).",
        "",
        "## Files",
        "",
        "| path | size_bytes | sha256 | rows | source_url | date |",
        "|---|---|---|---|---|---|",
    ]
    for e in entries:
        lines.append(
            f"| `{e['path']}` | {e['size']} | `{e['sha256']}` | {e['rows']} | {e['url']} | "
            f"{e['date']} |"
        )
    used = sorted({Path(e["path"]).parts[2] for e in entries})
    lines += ["", "## Sources, licenses, and citations", ""]
    for key in used:
        src = SOURCES.get(key)
        if src is None:
            continue
        lines.append(f"- **{src['name']}** (`data/raw/{key}/`): {src['url']}")
        lines.append(f"  - License: {src['license']}")
        lines.append(f"  - Cite: {src['citation']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate data/MANIFEST.md from data/raw/.")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "MANIFEST.md")
    args = parser.parse_args()

    entries = collect_entries(args.raw_dir)
    args.out.write_text(render_manifest(entries), encoding="utf-8", newline="\n")
    print(f"Wrote {args.out} with {len(entries)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
