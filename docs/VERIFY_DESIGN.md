# TrustFed-SDN — Verifiable Secure Aggregation Design (`Step 06`)

> **Code modules:** [`trustfed/verify/`](../trustfed/verify/) (`quantize.py`, `masking.py`, `merkle.py`, `commitment.py`, `protocol.py`) and [`trustfed/aggregation/secure.py`](../trustfed/aggregation/secure.py)  
> **Unit tests:** [`tests/test_verify.py`](../tests/test_verify.py)  
> **Approved Checkpoint:** `06.1` recorded in [`docs/USER_PREFERENCES.md`](USER_PREFERENCES.md) (`2026-11-13`)

---

## 1. Threat Model & Security Goals

In multi-tenant SDN federated learning, individual tenants share an aggregation server (either an external cloud coordinator or a designated control-plane orchestrator) that cannot be blindly trusted:

1. **Honest-but-Curious Confidentiality Threat (Server Inspecting Updates):**
   An untrusted aggregator may attempt to inspect a single tenant's weight vector $\mathbf{w}_i^{(r)}$ to infer private traffic distributions or execute white-box membership inference against that tenant's flows.
   - **Defense:** Pairwise additive masking hides every uploaded parameter vector $\tilde{\mathbf{x}}_i$ behind pseudorandom 48-bit masks so the server only ever observes uniformly blind residues $\mathbf{y}_i \in \mathbb{Z}_{2^{48}}^d$, while all pairwise masks cancel bit-for-bit in the sum $\sum_{i=1}^K \mathbf{y}_i \bmod 2^{48}$.

2. **Lazy or Malicious Integrity Threat (Server Tampering with Aggregation):**
   A buggy, lazy, or compromised aggregator may violate aggregation integrity in four distinct ways:
   - **Modified Aggregate:** Altering the published global model (including a $+1$ integer LSB perturbation or float drift).
   - **Dropped Tenant:** Silently omitting a participating tenant's valid update from the round sum.
   - **Replayed Update:** Re-inserting an authentic signed submission from an earlier FL round $r' < r$.
   - **Substituted / Forged Update:** Replacing a tenant's masked vector or metadata with forged values.
   - **Defense:** Option B (Dual-Prime Keyed Linear Homomorphic Checksums + RFC 6962 `SHA-256` Merkle Tree + `Ed25519` signatures + per-round 16-byte nonces).

3. **Out-of-Scope Threats Handled Elsewhere:**
   - **Byzantine / Poisoned Tenant Inputs:** Tenants that follow the cryptographic protocol while training on maliciously poisoned data (`label_flip`, `backdoor`, `scaling`) are defended in **Step 04** (`CosineClusteringAggregator`, `TrimmedMeanAggregator`). Value-level cryptographic verification guarantees that the server faithfully combined the submitted values—not that a tenant's underlying traffic labels were honest.

---

## 2. Cryptographic Construction & Protocol Steps

Every round $r \ge 1$ executes across `K` registered tenants ($d = 43{,}554$ GRU parameters on `InSDN`):

### 2.1 Fixed-Point Quantization (`quantize.py`)
Pairwise masking and modular checksums operate exactly over integers. Each tenant scales and rounds its local `float32` parameter vector $\mathbf{w}_i^{(r)}$ weighted by sample count $n_i$:
$$\tilde{\mathbf{x}}_i^{(r)} = n_i \cdot \text{round}\!\left(\mathbf{w}_i^{(r)} \cdot 2^{b}\right) \in \mathbb{Z}^d, \qquad b = 20 \;\;(S = 2^{20} = 1{,}048{,}576)$$
- Measured on initialized GRU parameters ($d = 43{,}554$):
  - `16-bit`: max abs error $= 7.63 \times 10^{-6}$, `SNR = 85.38 dB`
  - **`20-bit` (default)**: max abs error $= 4.77 \times 10^{-7}$, `mean abs error = 2.37e-7`, `RMSE = 2.74e-7`, **`SNR = 109.49 dB`**
  - `24-bit`: max abs error $= 2.98 \times 10^{-8}$, `SNR = 133.04 dB`

### 2.2 Pairwise Additive Masking (`masking.py`)
- Using standard **`X25519`** Elliptic-Curve Diffie-Hellman (`cryptography`), each ordered pair $(i, j)$ derives a symmetric 256-bit secret $k_{i,j}$ (via RFC 5869 HMAC-SHA256 HKDF) invisible to the server.
- In round $r$, **`SHAKE-256`** (NIST FIPS 202 extendable-output PRG) expands $(k_{i,j}, r)$ into $\mathbf{m}_{i,j}^{(r)} \in \mathbb{Z}_{2^{48}}^d$:
  $$\mathbf{y}_i^{(r)} = \left(\tilde{\mathbf{x}}_i^{(r)} + \sum_{j > i} \mathbf{m}_{i,j}^{(r)} - \sum_{j < i} \mathbf{m}_{j,i}^{(r)}\right) \bmod 2^{48}$$
- **Wire Packing:** Each 48-bit integer is packed into `6` bytes (`261,324` bytes vs `174,216` bytes for float32), giving a **`1.50×` upload ratio** and **`1.26×` total round-trip wire ratio** (including plain broadcast download + Merkle proof), fitting squarely inside the `1.2×–1.5×` poster target.
- **Exact Cancellation:** When summing across all $K$ tenants, each mask pair $+\mathbf{m}_{i,j}^{(r)} - \mathbf{m}_{i,j}^{(r)}$ cancels modulo $2^{48}$ with **exact `0` integer error**, provided $|\sum_i \tilde{x}_{i,k}| < 2^{47} \approx 1.4 \times 10^{14}$ (satisfied with $>100\times$ headroom for FedAvg sample counts).

### 2.3 Keyed Linear Checksums, Merkle Tree & Ed25519 Signatures (`commitment.py`, `merkle.py`, `protocol.py`)
1. **Why a plain hash of masked updates fails:** Since $\text{SHA-256}(\mathbf{y}_1) + \text{SHA-256}(\mathbf{y}_2) \neq \text{SHA-256}(\mathbf{y}_1 + \mathbf{y}_2)$, hashing ciphertexts/masked vectors alone cannot prove summation correctness to tenants.
2. **Dual-Prime Keyed Linear Checksum:** Tenants hold a shared group secret $k_{\text{grp}}$ (derived via pairwise X25519 keys, hidden from the server) to generate round-unique challenge vectors $(\mathbf{a}_1^{(r)}, \mathbf{a}_2^{(r)}) \in \mathbb{Z}_{p_1}^d \times \mathbb{Z}_{p_2}^d$ with independent 31-bit primes $p_1 = 2^{31}-1$ and $p_2 = 2^{31}-19$. Each tenant attaches:
   $$\tau_{i,1}^{(r)} = \langle \mathbf{a}_1^{(r)}, \tilde{\mathbf{x}}_i^{(r)} \rangle \bmod p_1, \qquad \tau_{i,2}^{(r)} = \langle \mathbf{a}_2^{(r)}, \tilde{\mathbf{x}}_i^{(r)} \rangle \bmod p_2$$
3. **Commitments & Signatures:** Each tenant computes a `SHA-256` commitment over `{client_id, round_idx, nonce_16B, num_samples, sha256(masked_wire_bytes), tau_1, tau_2}` and signs it with **`Ed25519`**.
4. **Server Proof Construction:** After checking all $K$ signatures, round numbers, nonces, and payload hashes, the server builds an RFC 6962 domain-separated **`SHA-256` Merkle Tree** (`0x00` leaf prefix, `0x01` internal node prefix), computes the unmasked integer sum $\mathbf{S}^{(r)}$ and dequantized global model $\hat{\mathbf{w}}^{(r+1)} = \mathbf{S}^{(r)} / ((\sum_i n_i) \cdot 2^b)$, and broadcasts `{aggregate_float, unmasked_sum_int, merkle_root, signed_headers, merkle_proofs}`.
5. **Tenant `verify()`:** Before applying $\hat{\mathbf{w}}^{(r+1)}$, every tenant checks:
   - Exact expected participant set (`no dropped tenant`),
   - Round counter $r$ and unseen 16-byte nonce (`no replay`),
   - `Ed25519` signature + commitment hash + `MerkleInclusionProof` against `merkle_root`,
   - Linear checksum homomorphism across both prime fields ($\Pr[\text{undetected forgery}] \le 1/(p_1 p_2) \approx 2^{-62}$):
     $$\langle \mathbf{a}_m^{(r)}, \mathbf{S}^{(r)} \rangle \bmod p_m \stackrel{?}{=} \left(\sum_{i=1}^K \tau_{i,m}^{(r)}\right) \bmod p_m \quad (m \in \{1, 2\})$$
   - Consistency between `aggregate_float` and dequantized `unmasked_sum_int` ($\le 10^{-6}$).

---

## 3. Value-Level Integrity vs. Identity-Level Authentication & Blockchain Designs

| Property | TrustFed-SDN (`Step 06`, Option B) | Plain Signature / TLS Only | Blockchain / BFT Committee FL |
|---|---|---|---|
| **Identity Authentication** | Ed25519 + X25519 keys | TLS / PKI signatures | On-chain public keys / DID |
| **Individual Update Confidentiality** | **Yes** (`Z_{2^48}` pairwise masks) | No (server sees raw `w_i`) | No (unless ZK/MPC added) |
| **Value-Level Aggregation Integrity** | **Yes** (dual-prime linear checksum $\le 2^{-62}$ soundness) | No (plain hashes do not commute with sum) | Depends on smart-contract gas or committee re-aggregation |
| **Measured Wire Overhead vs FedAvg** | **`1.26×–1.27×`** round-trip (`1.50×` upload) | `~1.00×` | `> 2×–10×` + block headers/votes |
| **Measured Crypto + Verify Latency** | **`23.6 ms`** ($N=3$) to **`49.9 ms`** ($N=10$) | `< 2 ms` | Seconds to minutes (consensus finality) |

By avoiding blockchain consensus or multi-party exponentiation commitments over 43,554 parameters, TrustFed-SDN achieves sub-`50 ms` end-to-end cryptographic verification suitable for real-time SDN controller update cycles.

---

## 4. Empirical Verification Summary

- **Honest Rounds (300 rounds across $N \in \{3, 5, 10\}$):** **`0` false alarms (`0.0%`)**.
- **Tamper Detection (1,500 adversarial trials across $N \in \{3, 5, 10\}$):** **`100.0%` detection (`100/100`)** on `modified_aggregate_float`, `modified_aggregate_lsb` (`+1` single quantized integer change), `dropped_tenant`, `replayed_update`, and `substituted_update`.
- **Utility Equivalence (5 seeds `0..4`, 30 rounds):**
  - `InSDN`: Plain `FedAvg` = `99.67 ± 0.19%` vs. `SecureAgg (20-bit)` = `99.53 ± 0.21%` (Wilcoxon signed-rank `p = 0.625`, non-significant).
  - `ASEADOS-SDN-IoT`: Plain `FedAvg` = `91.01 ± 0.23%` vs. `SecureAgg (20-bit)` = `90.47 ± 1.20%` (Wilcoxon signed-rank `p = 0.1875`, non-significant; learning curves overlap cleanly over Rounds 1–28, with `seed 4` Round-30 stochastic Adam variance occurring similarly at `24-bit` $< 3\times 10^{-8}$ error).

---

## 5. Honest Limitations

1. **No Mid-Round Dropout Recovery:** As approved in Checkpoint `06.1`, pairwise additive masks require every expected tenant in round $r$ to submit; if a tenant drops mid-round, `server_secure_aggregate` aborts and retries the round rather than running Shamir secret sharing / active-user mask reconstruction (unlike Bonawitz et al., 2017). This is appropriate for static multi-tenant SDN cloud controllers (`N = 3–10`) connected over dedicated control channels.
2. **Confidentiality of Group Checksum Key:** Security of Option B's linear checksum relies on `group_key` staying secret from the aggregator server. If an adversary compromises both the aggregator and any single participating tenant (collusion), the challenge vectors $(\mathbf{a}_1^{(r)}, \mathbf{a}_2^{(r)})$ become known to the colluding server, which could adjust two colluder-controlled coordinates to preserve the inner product; defeating server-tenant collusion requires public-coin homomorphic commitments (Option A).
3. **Synchronous Same-Set Participation:** All clients in a round submit quantized weighted updates for FedAvg-style linear combination. Combining verifiable secure aggregation directly with non-linear outlier trimming (`Step 04` + `Step 06`) requires server-side or commit-before-unmask adaptation evaluated in Step 09.
