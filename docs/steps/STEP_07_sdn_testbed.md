# STEP 07 — Multi-Tenant SDN Testbed, Traffic, and Live Dataset

## Goal
Build the real SDN testbed (Mininet + Open vSwitch + RYU) with several tenants, generate benign and attack traffic inside it, extract flow features, and create a labeled live dataset. Add the VM layer according to the approved isolation level.

## Prerequisites
Steps 00 and 01 approved (smoke test passed, feature list known). Runs on the Linux machine or Ubuntu VM.

## Ask the user first (design checkpoint, needs approval)
1. **Isolation level.** Present the levels with their cost and recommend one based on `docs/ENVIRONMENT.md`:
   - **L1:** each tenant is a Mininet sub-network with its own switch(es) and its own RYU controller instance; hosts are network namespaces. Cheapest.
   - **L2:** tenant hosts are containers (Containernet or Docker attached to OVS). Better process isolation.
   - **L3:** tenant machines are real virtual machines (for example Kali as attacker, Ubuntu as benign and victim) attached to the tenant OVS bridge. Matches the poster, needs the most RAM and CPU.
   - **Mixed:** L3 for one or two tenants (minimum: 3 machines per tenant), L1 or L2 for the rest. State this honestly in the thesis.
2. How many tenants (default 3) and their profiles and address ranges (from `docs/ENVIRONMENT.md`).
3. Which attack tools are allowed (default: `hping3`, `scapy`, and for L3 tools on the attacker machine), and which attacks to generate (DoS/DDoS floods such as SYN, UDP, ICMP; port scan; brute force). Confirm that all targets are inside the testbed ranges.
4. Which flow feature extractor: CICFlowMeter-compatible (matches the datasets) or OpenFlow flow statistics only (SDN-native), or both (recommended).

## Tasks
1. Write the topology in `testbed/topology/` (Python Mininet script, configurable tenants and hosts per tenant, per-tenant OVS switches, per-tenant RemoteController on separate ports, optional inter-tenant links through a core switch with policy). Include a README with a diagram.
2. Write RYU apps in `testbed/ryu_apps/`: per-tenant learning switch plus policy enforcement hooks, flow statistics collection (periodic port and flow stats), a REST or message interface to install block rules, and tenant-scoped isolation rules (a tenant cannot reach another tenant unless allowed).
3. Write the **SD-Traffic-Regulator and feature extractor** per tenant: mirror traffic from the tenant switch to a capture interface, build flow records, and compute the feature set approved in Step 01. Prefer a maintained flow-feature tool. Verify that the live features match the training features (names, units, order). Where a dataset feature cannot be produced live, report it and use the intersection.
4. Write traffic generators in `testbed/traffic/`: benign (HTTP, DNS, SSH, file transfer, streaming-like patterns with realistic rates and randomness) and attacks. Every script prints the target address and refuses to run if the target is outside the testbed ranges.
5. For L3 or mixed: write scripts or documentation in `testbed/vm/` to create the virtual machines, attach them to the OVS bridges, assign addresses, and provision them reproducibly (cloud-init, Vagrant, or libvirt definitions). Document resource limits per machine.
6. Collect a **live dataset**: run scheduled scenarios (benign only, each attack alone, mixed) for each tenant with controlled start and stop times, and label flows from the known roles and schedules. Save raw captures (git-ignored) and processed flow CSVs under `data/live/` with a manifest.
7. Evaluate the Step 02 models (centralized or FedAvg) on the live dataset to measure the sim-to-live gap. Report it honestly.
8. Write `docs/TESTBED.md` with the diagram, versions, addresses, how to start and stop everything (`make testbed-up`, `make testbed-down`), and the safety notes.

## Deliverables
- `testbed/*` with README, `Makefile` targets, `docs/TESTBED.md`
- `data/live/MANIFEST.md` (git-ignored data, tracked manifest)
- `results/07_testbed/` (traffic statistics, label distribution, sim-to-live evaluation)
- Tests for the feature extractor and the target-range guard

## Acceptance criteria
- [ ] Topology starts and stops cleanly with one command
- [ ] Tenants are isolated by policy (a test proves cross-tenant traffic is blocked)
- [ ] Attack scripts refuse targets outside testbed ranges (test proves it)
- [ ] Live features match training features (a test compares names and order)
- [ ] Live labeled dataset exists with a manifest and class counts
- [ ] Sim-to-live results are reported

## Report back, then stop
Include a diagram, the chosen isolation level, and the sim-to-live gap. Wait for "approved step 07".

## Pitfalls
- Run everything in internal or host-only networks. Never bridge attack machines to a real network.
- Flow features computed from different tools can differ; check carefully before trusting detection results.
- Time synchronization matters for labeling by schedule; log clock offsets.
