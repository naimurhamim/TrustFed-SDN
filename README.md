# TrustFed-SDN

A privacy-preserving, verifiable, poisoning-robust federated learning framework for cross-tenant threat detection in SDN-enabled multi-tenant cloud networks, validated on a real Mininet + Open vSwitch + RYU testbed.

> **Agents start here:** read `AGENTS.md`, then `PROGRESS.md`, `PROJECT_PLAN.md`, `docs/EXPERIMENT_SPEC.md`, the current step file in `docs/steps/`, and `docs/USER_PREFERENCES.md`.
>
> **Users start here:** tell the agent *"Read AGENTS.md and PROJECT_PLAN.md, then start Step 00."* After each step, reply *"approved step N"*.

## Four protections on top of FedAvg

| Pillar | Location |
|---|---|
| Real multi-tenant SDN testbed (Mininet + OVS + RYU) | `testbed/` |
| Adaptive per-tenant differential privacy | `trustfed/privacy/` |
| Verifiable secure aggregation | `trustfed/verify/`, `trustfed/aggregation/` |
| Poisoning-robust aggregation (trimmed mean, clustering) | `trustfed/aggregation/` |

Local model: GRU (to be confirmed by the Step 02 ablation against DNN and LSTM).
Datasets: InSDN (primary), CICIDS2017 (supplementary), CICDDoS2019 (optional), and a live dataset generated on the testbed.

## Repository layout

```
TrustFed-SDN/
  AGENTS.md            agent operating rules (mandatory)
  PROJECT_PLAN.md      master plan and step index
  PROGRESS.md          step status tracker
  README.md
  Makefile             created in Step 00 (make check, reproduce, testbed-up, demo)
  pyproject.toml       created in Step 00
  configs/             YAML configs: data/, experiments/, experiments/final/
  data/                git-ignored: raw/{insdn,cicids2017,cicddos2019}, interim/, processed/, live/
  docs/                specs, guides, design docs, step files
    steps/             STEP_00 ... STEP_10 instructions
    smoke_test/        Mininet + OVS + RYU smoke test logs (Step 00)
    EXPERIMENT_RESULTS_LOG.md   paper-ready record of every experiment result
  trustfed/            core library: data, models, metrics, fl, attacks, aggregation, privacy, verify
  sim/                 dataset-driven FL experiment runners
  testbed/             topology, ryu_apps, traffic, capture, vm, fl
  scripts/             helper scripts (hardware probe, dataset profile, QR code)
  results/             experiment outputs (small CSV/JSON tracked)
  figures/  tables/    generated figures and LaTeX tables
  site/                project website source (Step 10)
  tests/               pytest tests
```

## Documents in `docs/`

| File | Purpose | Created |
|---|---|---|
| `EXPERIMENT_SPEC.md` | Shared definitions: tenants, partitions, model, FL defaults, aggregators, attacks, privacy, verification, metrics, statistics | present |
| `DATASETS.md` | Dataset sources, download instructions, cleaning rules, feature selection | present |
| `USER_PREFERENCES.md` | Answers given by the user (filled step by step) | present |
| `TEAM_CHECKLIST.md`, `TEAM_WORKFLOW.md` | Granular `@Naimur` / `@Pritom` sub-task checklist, handoff board, and GitHub sync guide | present |
| `EXPERIMENT_RESULTS_LOG.md` | Paper-ready log of every measured number | present |
| `ENVIRONMENT.md`, `DEVIATIONS.md` | Environment versions, testbed address ranges, plan deviations | present |
| `MODEL_DECISION.md` | Local model choice with evidence | Step 02 |
| `DP_DESIGN.md` | DP granularity, sensitivity score, budget policy | Step 05 |
| `VERIFY_DESIGN.md` | Verification threat model and protocol | Step 06 |
| `TESTBED.md` | Testbed diagram, addresses, operation, safety | Step 07 |
| `LIVE_PIPELINE.md` | Live FL pipeline on the testbed | Step 08 |
| `RESULTS_SUMMARY.md` | Final findings, limitations, safe claims | Step 09 |
| `ARCHITECTURE.md`, `REPRODUCE.md`, `FINAL_AUDIT.md` | Release documentation | Step 10 |
