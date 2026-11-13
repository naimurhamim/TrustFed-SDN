"""Keyed linear homomorphic checksums, SHA-256 commitments, and Ed25519 signatures (`Step 06`).

Provides value-level aggregation integrity verification with O(d) time complexity:
  - Two independent 31-bit prime fields (`PRIME_1 = 2^31 - 1`, `PRIME_2 = 2^31 - 19`),
    giving a combined forgery/collision soundness probability <= 2^-62.
  - Round-unique challenge vectors `(a_1^(r), a_2^(r))` derived from a tenant-only group secret
    via `HMAC-SHA256 + SHAKE-256`, hidden from the honest-but-curious / malicious aggregator.
  - Serializable `TenantCommitmentHeader` signed with standard `Ed25519`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

PRIME_1: int = (1 << 31) - 1  # 2,147,483,647 (Mersenne prime M31)
PRIME_2: int = (1 << 31) - 19  # 2,147,483,629


def derive_challenge_vectors(
    group_key: bytes,
    round_idx: int,
    numel: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive independent round-specific challenge vectors `(a1, a2)` in `Z_{p1}^d x Z_{p2}^d`."""
    if numel <= 0:
        raise ValueError(f"numel must be > 0, got {numel}.")
    seed_1 = hmac.new(group_key, f"chk1:round={int(round_idx)}".encode(), hashlib.sha256).digest()
    seed_2 = hmac.new(group_key, f"chk2:round={int(round_idx)}".encode(), hashlib.sha256).digest()

    raw_1 = hashlib.shake_256(seed_1).digest(int(numel) * 4)
    raw_2 = hashlib.shake_256(seed_2).digest(int(numel) * 4)

    arr_1 = (np.frombuffer(raw_1, dtype=np.uint32).astype(np.int64) % PRIME_1).copy()
    arr_2 = (np.frombuffer(raw_2, dtype=np.uint32).astype(np.int64) % PRIME_2).copy()
    return torch.from_numpy(arr_1), torch.from_numpy(arr_2)


def compute_linear_checksums(
    int_vec: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
) -> tuple[int, int]:
    """Compute `( <a1, int_vec> mod p1, <a2, int_vec> mod p2 )` without `int64` overflow."""
    vec_cpu = int_vec.detach().to(device="cpu", dtype=torch.int64).contiguous()
    v1 = torch.remainder(vec_cpu, PRIME_1)
    v2 = torch.remainder(vec_cpu, PRIME_2)

    prod1 = torch.remainder(v1 * a1, PRIME_1)
    prod2 = torch.remainder(v2 * a2, PRIME_2)

    c1 = int(torch.remainder(prod1.sum(), PRIME_1).item())
    c2 = int(torch.remainder(prod2.sum(), PRIME_2).item())
    return c1, c2


@dataclass(frozen=True)
class TenantCommitmentHeader:
    """Signed commitment header published by a tenant alongside its masked update."""

    client_id: str
    round_idx: int
    nonce: bytes  # 16 bytes unique per (client_id, round_idx)
    num_samples: int
    masked_payload_sha256: bytes  # 32 bytes
    checksum_1: int
    checksum_2: int
    commitment_hash: bytes  # 32 bytes SHA-256 of canonical unsigned metadata + checksums
    signature: bytes  # 64 bytes Ed25519 signature over canonical signing message

    def unsigned_body_bytes(self) -> bytes:
        """Canonical byte encoding of all unsigned fields (used for commitment & signing)."""
        payload = {
            "client_id": self.client_id,
            "round_idx": int(self.round_idx),
            "nonce_hex": self.nonce.hex(),
            "num_samples": int(self.num_samples),
            "masked_payload_sha256": self.masked_payload_sha256.hex(),
            "checksum_1": int(self.checksum_1),
            "checksum_2": int(self.checksum_2),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def leaf_bytes(self) -> bytes:
        """Canonical leaf byte string committed into the server's Merkle tree."""
        return self.unsigned_body_bytes() + b"|" + self.commitment_hash + b"|" + self.signature

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "round_idx": int(self.round_idx),
            "nonce_hex": self.nonce.hex(),
            "num_samples": int(self.num_samples),
            "masked_payload_sha256": self.masked_payload_sha256.hex(),
            "checksum_1": int(self.checksum_1),
            "checksum_2": int(self.checksum_2),
            "commitment_hash_hex": self.commitment_hash.hex(),
            "signature_hex": self.signature.hex(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TenantCommitmentHeader:
        return cls(
            client_id=str(data["client_id"]),
            round_idx=int(data["round_idx"]),
            nonce=bytes.fromhex(str(data["nonce_hex"])),
            num_samples=int(data["num_samples"]),
            masked_payload_sha256=bytes.fromhex(str(data["masked_payload_sha256"])),
            checksum_1=int(data["checksum_1"]),
            checksum_2=int(data["checksum_2"]),
            commitment_hash=bytes.fromhex(str(data["commitment_hash_hex"])),
            signature=bytes.fromhex(str(data["signature_hex"])),
        )

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, raw: bytes) -> TenantCommitmentHeader:
        return cls.from_dict(json.loads(raw.decode("utf-8")))


def sign_commitment_header(
    *,
    client_id: str,
    round_idx: int,
    nonce: bytes,
    num_samples: int,
    masked_wire_bytes: bytes,
    checksum_1: int,
    checksum_2: int,
    ed25519_private: ed25519.Ed25519PrivateKey,
) -> TenantCommitmentHeader:
    """Create and sign a `TenantCommitmentHeader` using standard Ed25519."""
    masked_sha = hashlib.sha256(masked_wire_bytes).digest()
    draft = TenantCommitmentHeader(
        client_id=str(client_id),
        round_idx=int(round_idx),
        nonce=bytes(nonce),
        num_samples=int(num_samples),
        masked_payload_sha256=masked_sha,
        checksum_1=int(checksum_1),
        checksum_2=int(checksum_2),
        commitment_hash=b"",
        signature=b"",
    )
    body = draft.unsigned_body_bytes()
    commit_hash = hashlib.sha256(b"trustfed:commit:v1|" + body).digest()
    sig_msg = b"trustfed:ed25519:v1|" + commit_hash
    signature = ed25519_private.sign(sig_msg)
    return TenantCommitmentHeader(
        client_id=draft.client_id,
        round_idx=draft.round_idx,
        nonce=draft.nonce,
        num_samples=draft.num_samples,
        masked_payload_sha256=draft.masked_payload_sha256,
        checksum_1=draft.checksum_1,
        checksum_2=draft.checksum_2,
        commitment_hash=commit_hash,
        signature=signature,
    )


def verify_commitment_header_signature(
    header: TenantCommitmentHeader,
    ed25519_public_bytes: bytes,
) -> bool:
    """Verify commitment digest integrity and Ed25519 signature."""
    body = header.unsigned_body_bytes()
    expected_commit = hashlib.sha256(b"trustfed:commit:v1|" + body).digest()
    if header.commitment_hash != expected_commit:
        return False
    sig_msg = b"trustfed:ed25519:v1|" + expected_commit
    pub_key = ed25519.Ed25519PublicKey.from_public_bytes(ed25519_public_bytes)
    try:
        pub_key.verify(header.signature, sig_msg)
    except InvalidSignature:
        return False
    return True
