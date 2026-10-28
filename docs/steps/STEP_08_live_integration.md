# STEP 08 — Live Integration: Federated Learning on the Testbed

## Goal
Run the full TrustFed-SDN pipeline across the tenants of the real testbed: local training on live flows, privacy layer, secure and verifiable aggregation, robust aggregation, global model deployment to every tenant controller, and real-time detection with mitigation.

## Prerequisites
Steps 04, 05, 06, and 07 approved.

## Ask the user first
1. Which communication stack should tenants and the aggregation server use (recommended: Flower with custom strategies reusing the `trustfed/` aggregation code, or a simple gRPC/HTTP service)?
2. Where does the aggregation server run (a dedicated host or VM, or the Linux host)? Is it allowed to be a separate machine?
3. Mitigation policy: block the source address for a time window, rate-limit, or alert only? Default: temporary drop rule plus an alert, with automatic expiry.
4. How many rounds and what live scenarios should the demo run?

## Tasks
1. Implement tenant FL clients in `testbed/fl/` that read the tenant's live flow records, train locally with the approved model, apply DP, quantize, mask, commit, sign, and send updates. Reuse the `trustfed/` library; do not duplicate algorithms.
2. Implement the aggregation server: verifies signatures and commitments, runs secure aggregation, applies the robust aggregator, produces the proof, and publishes the new global model with the proof. Tenants verify before accepting.
3. Implement **model deployment**: a RYU app per tenant loads the global model, classifies flow records in near real time, and calls the policy enforcement hook (install a drop rule with expiry, log the decision). Log detection latency and mitigation time.
4. Add an attack injector: make one tenant malicious (label flipping, backdoor, update scaling) during live rounds, using the Step 04 attacks, and show detection or filtering.
5. Add tamper injection at the server (alter the aggregate) and show that tenants reject it.
6. Measure end-to-end per-round latency with the full breakdown from `docs/EXPERIMENT_SPEC.md`, communication bytes, and CPU and memory use of the controller during rounds. Check that the control plane stays responsive (for example flow setup latency during aggregation).
7. Produce a scripted end-to-end demo (`make demo`) and record a screen capture or step-by-step logs for the website.
8. Write `docs/LIVE_PIPELINE.md` with sequence diagrams and operating instructions.

## Deliverables
- `testbed/fl/*`, `testbed/ryu_apps/model_deploy.py`, `make demo`
- `results/08_live/` (latency breakdown, detection and mitigation timeline, malicious-tenant and tamper demos)
- `docs/LIVE_PIPELINE.md`

## Acceptance criteria
- [ ] Three or more tenants complete at least 10 live rounds with the full pipeline
- [ ] Global model deployed to every tenant controller and used for live detection
- [ ] A malicious tenant is handled and a tampered aggregate is rejected, both shown with logs
- [ ] Latency breakdown and control-plane impact measured
- [ ] `make demo` works from a clean start

## Report back, then stop
Wait for "approved step 08".

## Pitfalls
- Keep ML work out of the RYU event loop; run inference in a separate worker process and pass results through a queue.
- Test mitigation on benign traffic too, and report false blocks.
- Failures in a round must abort cleanly and retry, not leave tenants on inconsistent models.
