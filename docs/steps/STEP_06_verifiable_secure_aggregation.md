# STEP 06 — Verifiable Secure Aggregation

## Goal
Hide individual updates from the aggregator with masking, and let tenants verify that the aggregate was computed correctly (value-level integrity), with measured overhead.

## Prerequisites
Step 03 approved.

## Ask the user first (design checkpoint, needs approval)
1. **Threat model confirmation.** The aggregator may be lazy or malicious (it can alter or drop updates or the aggregate) and is curious about individual updates. Tenants are assumed to follow the protocol for masking and commitments (poisoning by tenants is handled in Step 04). Confirm or change.
2. **Verification scheme.** Present these options with trade-offs, recommend one, and wait for a decision:
   - **Option A — Homomorphic commitment or hash.** Each tenant publishes a linearly homomorphic commitment or hash of its quantized update; anyone can check that the combination of commitments matches the commitment of the published aggregate. Stronger cryptographic guarantee, higher cost.
   - **Option B — Keyed linear checksum plus hash commitments and signatures.** Tenants share a secret random vector (derived from pairwise seeds, unknown to the aggregator), publish a checksum of their update, and verify that the aggregate matches the sum of the checksums. Lighter, relies on the secret key staying hidden from the aggregator.
   - A plain hash of a masked update does not prove that the sum is correct. Say so.
3. **Dropout handling.** Default: no tenant dropout inside a round (a round is aborted and retried). Ask whether dropout recovery is needed. If not, list it as a limitation.
4. **Quantization.** Masking and homomorphic checks operate on integers. Ask for the fixed-point precision (propose a default after testing that accuracy does not change).

## Tasks
1. Implement fixed-point quantization and dequantization in `trustfed/verify/quantize.py` and test round-trip error.
2. Implement pairwise masking in `trustfed/verify/masking.py` (pairwise seeds, masks that cancel in the sum). Test that the sum of masked updates equals the sum of raw updates exactly in the integer domain.
3. Implement the approved verification scheme in `trustfed/verify/` plus signatures on submissions (use a standard library; never invent cryptography) and a nonce or round counter to stop replay.
4. Implement an `Aggregator` wrapper that performs secure aggregation and returns the aggregate plus a verification proof; and a tenant-side `verify()` function.
5. Tamper tests (all must be detected): modified aggregate, dropped tenant, replayed old update, substituted update. Also confirm zero false alarms on 100 honest rounds.
6. Measure overhead separately: quantization, masking, commitment/checksum, signatures, verification, and total bytes added per tenant per round versus plain FedAvg. Plot the breakdown. Compare with the poster target of about 1.2–1.5× and report the real ratio.
7. Verify that secure aggregation does not change model accuracy beyond quantization error (compare with plain FedAvg, same seeds).
8. Write `docs/VERIFY_DESIGN.md`: threat model, protocol steps, what is guaranteed and what is not (value-level integrity versus identity-level authentication), limitations, and a clear statement of how this differs from blockchain-committee designs.

## Deliverables
- `trustfed/verify/*`, `trustfed/aggregation/secure.py`
- `results/06_verify/` (tamper-test table, overhead breakdown, accuracy equivalence)
- `docs/VERIFY_DESIGN.md`
- `tests/test_verify.py`

## Acceptance criteria
- [ ] Masks cancel exactly; round-trip quantization error documented
- [ ] All four tamper tests detected; zero false alarms on honest runs
- [ ] Overhead breakdown measured, not estimated
- [ ] Design document states limits honestly

## Report back, then stop
Wait for "approved step 06".

## Pitfalls
- Never implement custom cryptographic primitives; use maintained libraries and standard constructions.
- Verification here confirms the aggregator combined the submitted values correctly. It does not detect a legitimate tenant that submits poisoned values; that is Step 04.
- Keep protocol messages serializable, because Step 08 runs them over a real network.
