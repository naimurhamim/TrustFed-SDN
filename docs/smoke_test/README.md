# Linux SDN Testbed Smoke Test (`Mininet + Open vSwitch + RYU`)

## Current Status: `unverified` on Windows 11 Host (Documented Fallback Active)

- **Recorded on:** 2026-10-28 (Step 00)
- **Deviation reference:** `DEV-001` in `docs/DEVIATIONS.md`

### Why It Did Not Run Natively on This Host

1. The primary development machine runs **Windows 11 Home (build 26300)**. Mininet requires a native Linux kernel with network namespaces (`unshare`/`ip netns`) and the Open vSwitch kernel module (`openvswitch`), so it cannot run natively on Windows (and `STEP_00` explicitly forbids faking Mininet on Windows).
2. **WSL2** on this host currently contains only the `docker-desktop` utility VM (no Ubuntu distribution), and standard WSL2 kernels often lack full `openvswitch` kernel datapath modules out of the box.
3. **VMware Workstation 17.6.0** is installed on this host with one powered-off Kali Linux 2025.2 VM (`4 GB RAM`). Per the approved project execution order (`docs/USER_PREFERENCES.md`), Steps 01–06 (datasets, baselines, FL core, poisoning defenses, adaptive DP, verifiable aggregation) run directly on Windows in `TrustFed-SDN/.venv` (Python 3.12.7), while a clean **Ubuntu 22.04 LTS Server VM** will be provisioned in VMware before Step 07.

---

## Fallback & Verification Plan (Mandatory Gate Before Step 07)

### 1. Recommended VMware VM Specification

- **Image:** Ubuntu 22.04 LTS Server (`amd64`) — preferred because Ubuntu 22.04 ships with Python 3.10 and battle-tested `mininet` + `openvswitch-switch` packages.
- **Resources:** 4 vCPUs, 4–6 GB RAM, 40 GB virtual disk ( leaves ~10–11 GB RAM for Windows 11 on the 15.7 GB host).
- **Network adapters:**
  - During package installation only: NAT.
  - During attack/testbed experiments (Steps 07–08): Host-Only / Custom isolated VMnet (never bridged to an external LAN).
- **Controller Python compatibility fallback:** Upstream `ryu==4.34` requires Python <= 3.10 and `eventlet==0.30.2`. If Ubuntu 24.04 or Kali (Python 3.11/3.12+) is used instead, `scripts/smoke_test_linux.sh` automatically switches `.venv-ryu` to **`os-ken`** (the actively maintained OpenStack fork of RYU with an identical OpenFlow 1.3 `simple_switch_13` API).

### 2. How to Verify (Exact Commands Inside the Linux VM)

```bash
# 1. Install system SDN packages (one-time)
sudo apt-get update
sudo apt-get install -y mininet openvswitch-switch python3-venv python3-pip

# 2. Run the automated smoke test from the TrustFed-SDN root
sudo bash scripts/smoke_test_linux.sh
```

### 3. Expected Artifacts Generated in `docs/smoke_test/`

When `scripts/smoke_test_linux.sh` succeeds (`0% dropped` across 3 hosts through an OpenFlow 1.3 OVS bridge controlled remotely on `127.0.0.1:6653`), it writes:

- `docs/smoke_test/versions.txt` — Linux kernel, Mininet, Open vSwitch, and `.venv-ryu` package versions
- `docs/smoke_test/controller.log` — RYU / OS-Ken OpenFlow 1.3 learning-switch session log
- `docs/smoke_test/mininet_pingall.log` — Mininet `pingall` output verifying `0% dropped`

