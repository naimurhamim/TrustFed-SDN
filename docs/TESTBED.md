# TrustFed-SDN Testbed Architecture & Operations (`docs/TESTBED.md`)

> Approved at Design Checkpoint `07.1` (`2026-11-14`) | Results logged in `results/07_testbed/` & `docs/EXPERIMENT_RESULTS_LOG.md` (Section 8).

---

## 1. Isolation Level & Host Stack (`L1`)

- **Isolation Level:** **`L1`** — One **Ubuntu 22.04 LTS VMware guest (`4096 MB` RAM, 2 vCPUs)** running Mininet `2.3.0` network namespaces (`9` hosts) + per-tenant Open vSwitch `2.17.12` bridges (`s1, s2, s3`, core `s0`) + dedicated `os-ken==3.0.1` OpenFlow `1.3` controller instances (`ports 6653, 6654, 6655`).
- **Why L1?** Host system has `15.67 GB` RAM (`1.1 GB` free with 1 desktop VM active); Pure L3 (`9+` VMs) requires `18–24 GB` RAM (`docs/ENVIRONMENT.md`). Optional L3 multi-VM reference scripts remain available in [`testbed/vm/`](../testbed/vm/README.md).

## 2. Multi-Tenant SDN Topology Diagram

```text
                    +-------------------------------------------+
                    |      Core Policy Switch s0 (OF 1.3)       |
                    |  priority=200: DROP cross-tenant IP/ARP   |
                    +------+--------------+--------------+------+
                           |              |              |
              +------------+              |              +------------+
              v                           v                           v
    +-------------------+       +-------------------+       +-------------------+
    | OVS Bridge s1     |       | OVS Bridge s2     |       | OVS Bridge s3     |
    | Controller: c1    |       | Controller: c2    |       | Controller: c3    |
    | Port: 6653        |       | Port: 6654        |       | Port: 6655        |
    +----+----+----+----+       +----+----+----+----+       +----+----+----+----+
         |    |    |                 |    |    |                 |    |    |
       h1_b h1_v h1_a              h2_b h2_v h2_a              h3_b h3_v h3_a
      .1.10 .1.50 .1.100          .2.10 .2.50 .2.100          .3.10 .3.50 .3.100
   Tenant 1 — Finance          Tenant 2 — Healthcare       Tenant 3 — Education
     (10.10.1.0/24)              (10.10.2.0/24)              (10.10.3.0/24)
```

## 3. Address Plan & Safety Guard (`10.10.0.0/16`)

| Tenant | Subnet | Hosts | Specialty Attack Scenarios | Controller Port |
|---|---|---|---|---|
| **Tenant 1 — Finance** | `10.10.1.0/24` | `h1_ben` (`.10`), `h1_vic` (`.50`), `h1_atk` (`.100`) | `BruteForce` (`SSH`), `WebAttack` (`HTTP SQLi/XSS`), `DoS` | `6653` (`c1`) |
| **Tenant 2 — Healthcare** | `10.10.2.0/24` | `h2_ben` (`.10`), `h2_vic` (`.50`), `h2_atk` (`.100`) | `DoS` (`SYN Flood`), `DDoS` (`UDP Flood`, `ICMP Flood`) | `6654` (`c2`) |
| **Tenant 3 — Education** | `10.10.3.0/24` | `h3_ben` (`.10`), `h3_vic` (`.50`), `h3_atk` (`.100`) | `Probe` (`Port Scan`), `DoS` (`HTTP`), `DDoS` (`UDP Flood`) | `6655` (`c3`) |

> **Hard Safety Invariant:** [`testbed/traffic/safety_guard.py`](../testbed/traffic/safety_guard.py) (`validate_testbed_target`) is called before opening any socket or packet generator in [`benign_traffic.py`](../testbed/traffic/benign_traffic.py) and [`attack_traffic.py`](../testbed/traffic/attack_traffic.py). Any IP outside `10.10.1.0/24`–`10.10.3.0/24` (loopback, `192.168.x.x`, public IPs) raises `SafetyGuardError` immediately. Never bridge testbed virtual switches to external interfaces.

## 4. Hybrid Flow Feature Extractor (`testbed/capture/`)

- **CICFlowMeter-Compatible Bidirectional Extractor ([`flow_extractor.py`](../testbed/capture/flow_extractor.py)):** Parses raw `libpcap` captures (`data/live/raw_pcaps/`) into exact Step 01 training feature schemas (`configs/data/features_insdn.yaml` `19` features and `configs/data/features_aseados_sdn_iot.yaml` `12` features) with identical names, column order, and units (microseconds for duration/IATs, L4 transport bytes for header lengths, payload bytes for packet sizes).
- **SDN-Native OpenFlow 1.3 Stats ([`tenant_controller_13.py`](../testbed/ryu_apps/tenant_controller_13.py)):** Polls `OFPFlowStatsRequest` and `OFPPortStatsRequest` per switch/tenant into `results/07_testbed/of_stats_t{1,2,3}.jsonl`.

## 5. How to Start, Stop & Analyze

Inside the Ubuntu 22.04 LTS VM:
```bash
# Start full testbed, verify L1 isolation, capture labeled live dataset:
make testbed-up      # or: sudo bash scripts/run_step07_testbed.sh

# Clean up Mininet / OVS virtual interfaces:
make testbed-down    # or: sudo mn -c
```

On Windows or Linux (Sim-to-Live evaluation + figures + manifest):
```bash
.venv/Scripts/python.exe scripts/analyze_step07_testbed.py
```

## 6. Verified Live Capture & Sim-to-Live Baseline (`2026-11-14`)

- **VM Execution Verification (`Ubuntu 22.04.5 LTS`, `Mininet 2.3.0`, `OVS 2.17.12`, `os-ken==3.0.1`):**
  - Intra-tenant reachability: **`18/18` (`0.0%` packet loss)**.
  - Cross-tenant isolation (`s0` + per-tenant `TenantController13`): **`6/6` blocked (`0` leaked, `100.0%` isolation)**.
  - Live dataset captured (`data/live/`): **`4,392` labeled bidirectional flows** (`1,992` `tenant_1_finance`, `997` `tenant_2_healthcare`, `1,403` `tenant_3_education`; `2,733` Attack, `1,659` Benign) across `12` `.pcap` files (`data/live/MANIFEST.md`).
  - Schema check: `19/19` `InSDN` features and `12/12` `ASEADOS-SDN-IoT` features match exact names/order with `0` NaN/Inf (`results/07_testbed/capture_summary.json`).
- **Sim-to-Live Transfer (`results/07_testbed/sim_to_live_gap.csv`):**
  - `InSDN` FedAvg GRU: `0.9953` offline Macro-F1 $\to$ `0.4425` zero-shot live (`0.7089` ROC-AUC after Z-calibration) $\to$ **`0.7546` Macro-F1 (`0.8728` ROC-AUC)** after live adaptation.
  - `ASEADOS-SDN-IoT` FedAvg GRU: `0.9133` offline Macro-F1 $\to$ `0.3346` zero-shot live (`0.4953` Z-calibrated) $\to$ **`0.7462` Macro-F1 (`0.8549` ROC-AUC)** after live adaptation.
