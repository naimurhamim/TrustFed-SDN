"""Multi-tenant SDN topology definitions for TrustFed-SDN Step 07."""

from __future__ import annotations

from testbed.topology.multi_tenant_topo import (
    CORE_SWITCH_SPEC,
    TOPOLOGY_SPEC,
    TenantHostSpec,
    TenantTopoSpec,
    get_all_host_ips,
    get_tenant_spec,
)

__all__ = [
    "CORE_SWITCH_SPEC",
    "TOPOLOGY_SPEC",
    "TenantHostSpec",
    "TenantTopoSpec",
    "get_all_host_ips",
    "get_tenant_spec",
]
