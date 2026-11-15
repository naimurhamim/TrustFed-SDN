"""Step 09 Master Evaluation & Reproducibility CLI (`scripts/run_step09_full_evaluation.py`).

Usage:
  make reproduce        -> python scripts/run_step09_full_evaluation.py
  make reproduce-quick  -> python scripts/run_step09_full_evaluation.py --quick
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (REPO_ROOT, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from _step09_experiments import (  # noqa: E402
    consolidate_all_experiments,
    run_or_load_new_experiments,
    run_quick_smoke_check,
)
from _step09_figures_tables import (  # noqa: E402
    generate_all_figures,
    generate_all_tables,
    update_figure_and_table_readmes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="TrustFed-SDN Step 09 Master Reproduction Runner")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fast smoke verification and regenerate figures/tables from canonical results.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore cached 5-seed E6/E8/E10 runs and re-execute all Step 09 experiments.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Only regenerate figures, tables, and summary JSON from cached artifacts.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    final_dir = REPO_ROOT / "results" / "final"
    cache_path = final_dir / "step09_new_runs_cache.json"

    if args.quick:
        print("[Step 09] Running quick smoke check (make reproduce-quick)...", flush=True)
        quick_res = run_quick_smoke_check(final_dir=final_dir)
        print(
            f"[Step 09] Quick check complete ({len(quick_res['smoke_runs'])} runs in "
            f"{quick_res['wall_time_sec']:.1f}s).",
            flush=True,
        )
        if not cache_path.exists():
            print(
                "[Step 09] No full 5-seed cache found yet; running full E6/E8/E10 suite now...",
                flush=True,
            )
            cache = run_or_load_new_experiments(force_rerun=False, final_dir=final_dir)
            consolidate_all_experiments(cache, final_dir=final_dir)
    else:
        cache = run_or_load_new_experiments(
            force_rerun=bool(args.force_rerun and not args.figures_only),
            final_dir=final_dir,
        )
        consolidate_all_experiments(cache, final_dir=final_dir)

    if (final_dir / "summary.json").exists():
        figs = generate_all_figures()
        tabs = generate_all_tables()
        update_figure_and_table_readmes()
        print(
            f"[Step 09] Generated {len(figs)} figure files (.pdf+.png) and "
            f"{len(tabs)} LaTeX tables in {time.perf_counter() - t0:.1f}s.",
            flush=True,
        )


if __name__ == "__main__":
    main()
