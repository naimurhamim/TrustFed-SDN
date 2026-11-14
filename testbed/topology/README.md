# Multi-Tenant SDN Topology (`testbed/topology/`)

```
                    +-------------------------------------------+
                    |     Core Isolation Switch s0 (OF 1.3)     |
                    | priority=200 cross-tenant IP/ARP drop     |
                    +------+--------------+--------------+------+
                           |              |              |
         +-----------------+    +---------+----+    +----+----------------+
         |                      |              |                          |
+--------v--------+    +--------v--------+    +v----------------+         |
|  OVS Bridge s1  |    |  OVS Bridge s2  |    |  OVS Bridge s3  |         |
| (c1: 6653 OSKen)|    | (c2: 6654 OSKen)|    | (c3: 6655 OSKen)|         |
+--+-----+-----+--+    +--+-----+-----+--+    +--+-----+-----+--+         |
   |     |     |          |     |     |          |     |     |            |
  h1b   h1v   h1a        h2b   h2v   h2a        h3b   h3v   h3a           |
 .1.10 .1.50 .1.100     .2.10 .2.50 .2.100     .3.10 .3.50 .3.100         |
   Tenant 1 Finance       Tenant 2 Healthcare    Tenant 3 Education
   (10.10.1.0/24)         (10.10.2.0/24)         (10.10.3.0/24)
```

- `multi_tenant_topo.py` defines `TOPOLOGY_SPEC`, `MultiTenantSDNTopo`, OpenFlow 1.3 core isolation rules, and live intra/cross-tenant isolation verification.
