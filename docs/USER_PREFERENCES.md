# USER PREFERENCES

The agent fills this file during Step 00 and later steps. Do not ask again for anything recorded here.

| Topic | Answer | Recorded in step |
|---|---|---|
| Chat language | Bengali (Bengali script) in chat. No Bangla or Banglish in code or files. | 00 |
| Operating system / Linux availability | Development on Windows 11 Home (build 26300). Linux available as a VMware VM (VMware Workstation 17.6.0 installed). WSL is present but not stated as the Linux option. | 00 |
| CPU / RAM / disk / GPU | Probed 2026-10-28: Intel Core i7-12700H (14 cores, 20 threads); RAM 15.7 GB total (3.7 GB free at probe time); disk C: 150 GB free of 476 GB, D: 477 GB free of 954 GB; GPU NVIDIA GeForce RTX 3050 Ti Laptop 4 GB (driver 610.88) plus Intel Iris Xe. Windows hypervisor is active (Hyper-V/VBS). | 00 |
| Hypervisor and VM capacity | VMware Workstation 17.6.0. Existing VM: `kali-linux-2025.2-vmware-amd64` (4 GB RAM, powered off). User confirmed willingness to download/create an Ubuntu VM when needed for the testbed. Estimated capacity with 15.7 GB host RAM: about 2 VMs of 4 GB each simultaneously (estimate, unverified). | 00 |
| Python and package manager | Python 3.12 (already installed at `C:\Users\Naim\AppData\Local\Programs\Python\Python312`) with `venv` + `pip`, environment at `TrustFed-SDN/.venv` (project-local, user request). User delegated the version choice to the agent; 3.12 chosen for library compatibility (PyTorch, DP libraries). Testbed/RYU environment is separate and lives in the Linux VM. | 00 |
| FL framework | User delegated the choice to the agent: custom lightweight PyTorch FL core in `trustfed/fl/` for Steps 03-06; Flower for the live testbed in Step 08, reusing `trustfed/` aggregation code. | 00 |
| Git hosting | Local git repository inside `TrustFed-SDN/` for now; user will push to GitHub later. | 00 |
| Supervisor requirements and deadlines | No special mandatory requirements or restrictions from supervisor or department. | 00 |
| Website hosting and DNS control | GitHub Pages at `https://trustfed.naimurrashid.dev/` (matching the existing `CNAME` and repository setup; user confirmed keeping this setup). | 00 |
| Step order preference | Default recommended order: Steps 01-06 (data and FL algorithms on Windows) before Step 07 (SDN testbed on Linux VM). User asked the agent to follow the best/recommended order and choices. | 00 |
| Datasets available / wanted | | 01 |
| Tenant profiles | | 01 |
| Final local model | | 02 |
| DP granularity, sensitivity score, budget policy | | 05 |
| Verification scheme and dropout handling | | 06 |
| Isolation level, tenants, attack tools, feature extractor | | 07 |
| Communication stack and mitigation policy | | 08 |
| Public/private decisions, license | | 10 |
