# VM Provisioning & Isolation Reference (`testbed/vm/`)

## Active Step 07 Isolation Level: **L1 (Approved 2026-11-14)**
Because host memory is `15.67 GB` (`1.1 GB` free with one desktop VM running), Step 07 executes on:
- **1 Ubuntu 22.04 LTS VMware Workstation guest (`4096 MB` RAM, 2 vCPUs)** running:
  - **Mininet `2.3.0`** network namespaces for 9 tenant hosts (`h1..h3` benign, victim, attacker)
  - **Open vSwitch `2.17.12`** bridges (`s1`, `s2`, `s3`, core `s0`)
  - **`os-ken==3.0.1`** OpenFlow `1.3` per-tenant controllers (`ports 6653, 6654, 6655`) in `.venv-ryu` (`Python 3.10.12`).

## Reference L3 / Mixed Provisioning (`Vagrantfile`)
For workstation servers with `>= 24 GB` free RAM that wish to allocate dedicated VMs per tenant role, `testbed/vm/Vagrantfile` provides a reproducible reference Vagrant/VirtualBox/Libvirt definition mirroring the exact `10.10.1-3.0/24` host addressing and resource limits (`1024 MB` controller VM, `1024 MB` per Ubuntu host, `2048 MB` attacker VM).
