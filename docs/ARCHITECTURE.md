# TrustFed-SDN System Architecture

> Comprehensive technical specification of the 4-pillar privacy-preserving, Byzantine-robust, cryptographically verifiable federated SDN intrusion detection system (`v1.0.0`).

---

## 1. Threat Model & Multi-Tenant Setting

Multi-tenant SDN environments isolate virtual networks while sharing central orchestration. Because tenant flow logs carry sensitive behavioral patterns, centralizing raw telemetry leaks private tenant metadata. Conversely, naive Federated Learning (FL) leaves the control plane exposed to three simultaneous adversaries:

1. **Byzantine / Poisoned Tenants (20%–33% fraction):** Compromised tenant agents inject *label-flipping* (relabeling attacks as benign), *backdoor trigger* perturbations (~99.96% unguarded Attack Success Rate), or high-magnitude *update scaling* attacks to blind global SDN intrusion detection.
2. **Honest-but-Curious Server & Passive Eavesdroppers:** Observe uploaded gradients to run Membership Inference Attacks (MIA), particularly against low-sample minority cohorts (e.g., `Finance`) where uniform DP noise under-protects sparse clients.
3. **Untrusted Wire & Tampering Adversary:** Modifies masked shares, swaps commitments, or forges aggregation receipts over the FL control channel.

---

## 2. Subsystem Map (`trustfed/`, `testbed/`, `scripts/`)

| Package | Responsibility | Key Modules |
| :--- | :--- | :--- |
| `trustfed/data/` | Ingestion, leakage-free cleaning & non-IID partitioning | InSDN, ASEADOS-SDN-IoT, CICIDS2017 cleaners; Natural & Dirichlet (`alpha=0.5, 0.1`) splits |
| `trustfed/models/` | Shared sequential IDS & centralized comparison models | Sequence `GRU` (`43,554` params InSDN), `DNN`, `LSTM`, `LogReg`, `RF` wrappers |
| `trustfed/fl/` | Multi-round federated training orchestration | Sync `FedAvg` server/client loops, seeded evaluation runners |
| `trustfed/privacy/` | Cohort-adaptive Differential Privacy & MIA auditing | Sample-aware `eps_k` calibration, DP-SGD clipping (`C`), R&eacute;nyi accountability, MIA AUC evaluator |
| `trustfed/attacks/` & `trustfed/aggregation/` | Adversarial injection & Byzantine-robust filtering | Label-flip, backdoor trigger, scaling generators; two-stage Cosine Clustering + Coordinate Median / Trimmed Mean (`beta=0.2`) |
| `trustfed/verify/` | Verifiable 48-bit Packed Secure Aggregation | `Q=20` fixed-point quantizer, X25519 Diffie-Hellman pairwise PRG masks in Z_q (`q=2^61-1`), canonical SHA-256 commitments & Ed25519 signatures |
| `testbed/` | Live Ubuntu 22.04 SDN testbed & closed-loop OVS mitigation | `os-ken 3.0.1` OpenFlow 1.3 controller, Linux namespace topology (`t1..t3`, `br_t1..br_t3`), flow extractor, `priority=300` OVS drop rules with TTL expiry |
| `sim/` & `scripts/` | End-to-end E1–E10 benchmark, stats, plots & QR generator | Paired Wilcoxon stats, 8 Matplotlib figures, LaTeX table exporter, ISO/IEC 18004 QR generator (`make_qr.py`) |

---

## 3. End-to-End 7-Stage Cryptographic & SDN Round Loop

Per synchronized training and enforcement round (measured mean total **`486.714 ms`** on live testbed, `results/08_live/summary.json`):

1. **Local Adaptive DP-SGD (`384.452 ms`):** Each tenant clips per-sample GRU gradients to norm `C` and adds cohort-calibrated Gaussian noise derived from R&eacute;nyi DP accounting (`eps_k` scaled by square root of relative cohort volume so rare tenants such as InSDN `Finance` receive stronger noise protection, `eps_1 = 2.85` at `eps_bar = 4.0`).
2. **Quantize, Mask, Commit & Sign (`11.162 ms`):** Model deltas are fixed-point quantized at `Q=20` bits, packed into 48-bit words, blinded with X25519 pairwise Diffie-Hellman masks modulo `2^61 - 1`, hashed via canonical SHA-256 commitment, and signed with client Ed25519 keys.
3. **Upload (`16.447 ms`):** Masked payloads + signatures stream over the FL TCP plane (`1.256x` wire ratio at `N=3` to `1.290x` at `N=20` vs plain FedAvg).
4. **Server Pairwise Cancellation, Robust Filtering & Proof Assembly (`13.487 ms`):** Server cancels pairwise masks, computes cosine cluster alignment (`cutoff=0.5`) and trimmed-mean filtering (`beta=0.2`) to reject 100% of out-of-cluster Byzantine updates, and emits a `1,781 B` Ed25519-signed aggregation proof bundle.
5. **Global Download (`54.840 ms`):** Server broadcasts updated global weights plus verification transcript to participating tenants.
6. **Client Zero-Trust Verification (`5.554 ms`):** Every tenant verifies commitment hashes and the server Ed25519 transcript signature before accepting new weights (`1,500/1,500` tampered bundles rejected; `0/300` honest false alarms).
7. **Live SDN Controller Deploy & OVS Flow Block (`0.771 ms` deploy + `27.38 ms` OVS rule install):** `OS-Ken 3.0.1` hot-swaps the verified GRU scorer with validation-calibrated decision threshold (`theta=0.98`) and installs `priority=300` OpenFlow 1.3 drop rules with automatic TTL expiry on `br_t1..br_t3`, achieving `100.0%` attacker blocking with `0.0%` benign collateral and `2.70%` (`9/333`) live False Block Rate.

---

## 4. Testbed Topology Note (`L1` Namespace Isolation)

The physical VMware validation environment runs on **Ubuntu 22.04** using **L1 Linux network namespaces** (`t1`, `t2`, `t3` across `10.10.1.0/24`–`10.10.3.0/24`) attached to independent Open vSwitch bridges (`br_t1`, `br_t2`, `br_t3`) managed by OS-Ken OpenFlow 1.3 controllers (`DEV-004`). Baseline tenant connectivity tests confirm `0.0%` intra-tenant packet loss (`18/18`) and `100.0%` cross-tenant boundary isolation (`6/6`) with `0.009 ms` OpenFlow learning latency overhead (`0.071 -> 0.080 ms` ping RTT, `67.3 MB` controller RSS).
