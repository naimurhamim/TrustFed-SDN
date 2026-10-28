"""TrustFed-SDN core Python library.

Privacy-preserving, verifiable, poisoning-robust federated learning framework
for cross-tenant threat detection in SDN-enabled multi-tenant cloud networks.
"""

from __future__ import annotations

__version__ = "0.1.0"

TESTBED_SUBNET_CIDR: str = "10.10.0.0/16"
TENANT_SUBNETS: dict[str, str] = {
    "tenant_1_finance": "10.10.1.0/24",
    "tenant_2_healthcare": "10.10.2.0/24",
    "tenant_3_education": "10.10.3.0/24",
}

__all__ = [
    "TENANT_SUBNETS",
    "TESTBED_SUBNET_CIDR",
    "__version__",
]
