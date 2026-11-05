# USER PREFERENCES

The agent fills this file during Step 00 and later steps. Do not ask again for anything recorded here.

| Topic | Answer | Recorded in step |
|---|---|---|
| Chat language | Bengali (Bengali script) in chat. No Bangla or Banglish in code or files. | 00 |
| Operating system / Linux availability | Development on Windows 11 Home (build 26300). Linux available as a VMware VM (VMware Workstation 17.6.0 installed). WSL is present but not stated as the Linux option. | 00 |
| CPU / RAM / disk / GPU | Probed 2026-10-28: Intel Core i7-12700H (14 cores, 20 threads); RAM 15.7 GB total (3.7 GB free at probe time); disk C: 150 GB free of 476 GB, D: 477 GB free of 954 GB; GPU NVIDIA GeForce RTX 3050 Ti Laptop 4 GB (driver 610.88) plus Intel Iris Xe. Windows hypervisor is active (Hyper-V/VBS). | 00 |
| Hypervisor and VM capacity | VMware Workstation 17.6.0 / Player. Two VMs present: (1) **Ubuntu 22.04 LTS VM** (`5.15.0-25-generic`, Python `3.10.12`, Mininet `2.3.0`, Open vSwitch `2.17.12`, `os-ken==3.0.1` verified `0%` loss in Step 00) and (2) `kali-linux-2025.2-vmware-amd64` (`4 GB` RAM). Measured host free RAM with Ubuntu Desktop VM running alongside Windows 11: `1.1 GB` free (`93.0%` used of `15.67 GB`), confirming 1 active VM at a time is the safe practical limit. | 00 |
| Python and package manager | Python 3.12 (already installed at `C:\Users\Naim\AppData\Local\Programs\Python\Python312`) with `venv` + `pip`, environment at `TrustFed-SDN/.venv` (project-local, user request). User delegated the version choice to the agent; 3.12 chosen for library compatibility (PyTorch, DP libraries). Testbed/RYU environment is separate and lives in the Linux VM. | 00 |
| FL framework | User delegated the choice to the agent: custom lightweight PyTorch FL core in `trustfed/fl/` for Steps 03-06; Flower for the live testbed in Step 08, reusing `trustfed/` aggregation code. | 00 |
| Git hosting | Private GitHub repository `https://github.com/naimurhamim/TrustFed-SDN.git` (only the `TrustFed-SDN/` folder), collaborated on by `@Naimur` and `@Pritom` using `docs/TEAM_CHECKLIST.md`, `docs/TEAM_WORKFLOW.md`, and `scripts/team_status.py` (`make status`). | 00 |
| Supervisor requirements and deadlines | No special mandatory requirements or restrictions from supervisor or department. | 00 |
| Website hosting and DNS control | GitHub Pages at `https://trustfed.naimurrashid.dev/` (matching the existing `CNAME` and repository setup; user confirmed keeping this setup). | 00 |
| Step order preference | Default recommended order: Steps 01-06 (data and FL algorithms on Windows) before Step 07 (SDN testbed on Linux VM). User asked the agent to follow the best/recommended order and choices. | 00 |
| Datasets available / wanted | | 01 |
| Tenant profiles | | 01 |
| Final local model | | 02 |
| DP granularity, sensitivity score, budget policy | | 05 |
| Verification scheme and dropout handling | | 06 |
| Isolation level, tenants, attack tools, feature extractor | Isolation level **L1 approved** at Step 00 close (2026-11-05): one Ubuntu 22.04 VM, per-tenant Mininet namespaces + per-tenant OVS bridges + per-tenant OS-Ken controllers (L3 with 9+ VMs is infeasible on 15.67 GB RAM). Optional Kali VM attachment (Mixed) may be reconsidered at Step 07 only if free RAM allows. Tenants, attack tools, and feature extractor still to be confirmed at Step 07. | 00 / 07 |
| Communication stack and mitigation policy | | 08 |
| Public/private decisions, license | | 10 |
