# Linux SDN Testbed Smoke Test (`Mininet + Open vSwitch + OS-Ken / RYU`)

## Current Status: **VERIFIED (`0% dropped`, `6/6 received`)**

- **Verified on (UTC):** `2026-10-28T12:39:44Z` (Step 00)
- **VM Environment:** Ubuntu 22.04 LTS (`5.15.0-25-generic`, `x86_64`) on VMware Workstation 17
- **Summary Artifact:** `results/00_environment/smoke_test_summary.json`

### Verified SDN Software Stack

| Component | Verified Version | Isolation / Location |
|---|---|---|
| Guest OS & Kernel | Ubuntu 22.04 LTS (`Linux 5.15.0-25-generic`) | VMware Workstation 17 VM |
| Mininet | `2.3.0` (`2.3.0-1ubuntu1`) | System package (`apt`) |
| Open vSwitch | `2.17.12` (`ovs-vsctl`) | System service (`openvswitch-switch`) |
| Controller Python | `Python 3.10.12` | Separate controller venv (`.venv-ryu`) |
| SDN Controller Framework | `os-ken==3.0.1` + `eventlet==0.37.0` | Maintained OpenStack fork of RYU (`DEV-001`) |
| OpenFlow Protocol & App | OpenFlow `1.3` (`SmokeLearningSwitch13`) | `testbed/ryu_apps/smoke_switch_13.py` on `127.0.0.1:6653` |

### Why `os-ken==3.0.1` Is Used in `.venv-ryu`

Per `PROJECT_PLAN.md` Section 8 (*"verify in Step 00, or use a maintained fork"*), we tested upstream `ryu==4.34` + `eventlet==0.30.2` on Ubuntu 22.04 (`Python 3.10.12`) first and confirmed it fails at runtime due to removed Python 3.10 `collections` aliases (`MutableMapping`). Switching `.venv-ryu` to **`os-ken==3.0.1`** (OpenStack's actively maintained fork of RYU with identical OpenFlow 1.3 `OFPFlowMod` / `OFPPacketIn` APIs and `osken-manager`) resolved the issue cleanly while keeping the SDN controller environment strictly separate from the ML `.venv`.

### How to Re-Run Inside the Ubuntu 22.04 VM

```bash
sudo bash scripts/smoke_test_linux.sh
```

### Saved Verification Logs in This Folder

- `versions.txt` — Exact kernel, Mininet `2.3.0`, Open vSwitch `2.17.12`, and `.venv-ryu` `pip freeze` output
- `controller.log` — `osken-manager` loading `SmokeLearningSwitch13` and `OFPHandler` on TCP `127.0.0.1:6653`
- `mininet_pingall.log` — 3-host `SingleSwitchTopo(k=3)` `pingAll` log (`0% dropped (6/6 received)`) plus `ovs-ofctl -O OpenFlow13 dump-flows s1` showing 6 learned `priority=1` unicast MAC flows and the `priority=0 actions=CONTROLLER:65535` table-miss flow (`20` packets, `1468` bytes)
