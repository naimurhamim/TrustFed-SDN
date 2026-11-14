"""Hard target-range safety guard for all TrustFed-SDN testbed traffic scripts.

All benign and attack traffic generators MUST call ``validate_testbed_target``
before opening any socket or emitting packets. Any target outside the approved
``10.10.1.0/24`` - ``10.10.3.0/24`` tenant subnets (inside ``10.10.0.0/16``) is
rejected immediately.
"""

from __future__ import annotations

import ipaddress

try:
    from trustfed import TENANT_SUBNETS as _CORE_TENANT_SUBNETS
    from trustfed import TESTBED_SUBNET_CIDR as _CORE_SUPERNET_CIDR
except ImportError:
    _CORE_SUPERNET_CIDR = "10.10.0.0/16"
    _CORE_TENANT_SUBNETS = {
        "tenant_1_finance": "10.10.1.0/24",
        "tenant_2_healthcare": "10.10.2.0/24",
        "tenant_3_education": "10.10.3.0/24",
    }

TESTBED_SUPERNET: ipaddress.IPv4Network = ipaddress.ip_network(_CORE_SUPERNET_CIDR)
TESTBED_TENANT_SUBNETS: dict[str, ipaddress.IPv4Network] = {
    name: ipaddress.ip_network(cidr) for name, cidr in _CORE_TENANT_SUBNETS.items()
}


class SafetyGuardError(ValueError):
    """Raised when a target IP is invalid or outside approved isolated testbed subnets."""


def infer_tenant_for_ip(ip_addr: ipaddress.IPv4Address) -> str | None:
    """Return the tenant identifier owning ``ip_addr``, or ``None`` if unowned."""
    for tenant_id, subnet in TESTBED_TENANT_SUBNETS.items():
        if ip_addr in subnet:
            return tenant_id
    return None


def validate_testbed_target(
    target_ip: str,
    tenant_id: str | None = None,
    *,
    allow_broadcast: bool = False,
    verbose: bool = True,
) -> ipaddress.IPv4Address:
    """Validate that ``target_ip`` resides inside an approved isolated tenant subnet.

    Parameters
    ----------
    target_ip:
        Candidate destination IPv4 address string.
    tenant_id:
        Optional tenant key (e.g., ``tenant_1_finance``). When provided, the
        target MUST belong to that tenant's specific ``/24`` subnet.
    allow_broadcast:
        When ``False`` (default), network and broadcast host addresses are rejected.
    verbose:
        When ``True``, prints an audit line showing the validated target and subnet.

    Returns
    -------
    ipaddress.IPv4Address
        Parsed IPv4 address object if validation succeeds.

    Raises
    ------
    SafetyGuardError
        If the address is malformed, loopback/multicast, outside ``10.10.1-3.0/24``,
        or mismatched with ``tenant_id``.
    """
    try:
        parsed = ipaddress.ip_address(str(target_ip).strip())
    except ValueError as exc:
        raise SafetyGuardError(f"Malformed target IP address: {target_ip!r}") from exc

    if not isinstance(parsed, ipaddress.IPv4Address):
        raise SafetyGuardError(f"Only IPv4 testbed addresses are allowed; got {parsed!r}")

    if parsed.is_loopback or parsed.is_multicast or parsed.is_unspecified or parsed.is_link_local:
        raise SafetyGuardError(
            f"Target IP {parsed} is loopback/multicast/link-local and refused by safety guard."
        )

    if parsed not in TESTBED_SUPERNET:
        raise SafetyGuardError(
            f"Target IP {parsed} is outside approved testbed supernet {TESTBED_SUPERNET}. "
            "Refusing to run traffic."
        )

    matched_tenant = infer_tenant_for_ip(parsed)
    if matched_tenant is None:
        allowed_str = ", ".join(f"{k}={v}" for k, v in TESTBED_TENANT_SUBNETS.items())
        raise SafetyGuardError(
            f"Target IP {parsed} is not inside any active tenant subnet ({allowed_str})."
        )

    if tenant_id is not None:
        if tenant_id not in TESTBED_TENANT_SUBNETS:
            raise SafetyGuardError(f"Unknown tenant_id {tenant_id!r}.")
        expected_net = TESTBED_TENANT_SUBNETS[tenant_id]
        if parsed not in expected_net:
            raise SafetyGuardError(
                f"Target IP {parsed} belongs to {matched_tenant}, "
                f"expected {tenant_id} ({expected_net})."
            )

    matched_net = TESTBED_TENANT_SUBNETS[matched_tenant]
    if not allow_broadcast and parsed in (
        matched_net.network_address,
        matched_net.broadcast_address,
    ):
        raise SafetyGuardError(
            f"Target IP {parsed} cannot be the network/broadcast address of {matched_net}."
        )

    if verbose:
        print(
            f"[SAFETY GUARD OK] target={parsed} verified inside {matched_tenant} ({matched_net})",
            flush=True,
        )
    return parsed
