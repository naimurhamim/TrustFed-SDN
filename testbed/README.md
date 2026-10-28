# testbed

Real SDN testbed: Mininet + Open vSwitch + RYU. Runs on Linux or an Ubuntu VM only. Attack traffic is allowed only inside the testbed address ranges in `docs/ENVIRONMENT.md`.

- `topology/` - multi-tenant Mininet topology scripts (Step 07)
- `ryu_apps/` - per-tenant RYU apps: switching, flow stats, policy enforcement, `model_deploy.py` (Steps 07-08)
- `traffic/` - benign and attack traffic generators with a target-range guard (Step 07)
- `capture/` - SD-Traffic-Regulator, traffic mirroring, flow feature extraction (Step 07)
- `vm/` - VM provisioning for L3 or mixed isolation (Step 07)
- `fl/` - live tenant FL clients and the aggregation server (Step 08)
