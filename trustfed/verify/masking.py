"""Pairwise additive masking over a modular integer ring (`Step 06`).

Uses standard cryptographic primitives only:
  - `X25519` (`cryptography.hazmat.primitives.asymmetric.x25519`) for pairwise Diffie-Hellman
    shared secret derivation between tenants.
  - `Ed25519` (`cryptography.hazmat.primitives.asymmetric.ed25519`) for tenant identity keys.
  - `SHAKE-256` (`hashlib.shake_256`, NIST FIPS 202) as a deterministic extendable-output PRG
    expanding 256-bit pairwise round seeds into uniform 48-bit integer mask vectors.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519

MODULUS_BITS: int = 48
MODULUS_M: int = 1 << MODULUS_BITS  # 281,474,976,710,656 (6 bytes per parameter on the wire)
_MODULUS_MASK: int = MODULUS_M - 1


@dataclass
class TenantKeyMaterial:
    """Private and public cryptographic key material held locally by one tenant."""

    client_id: str
    x25519_private: x25519.X25519PrivateKey
    x25519_public_bytes: bytes
    ed25519_private: ed25519.Ed25519PrivateKey
    ed25519_public_bytes: bytes


@dataclass
class TenantPublicRegistry:
    """Public key directory known to all tenants and the server (contains no private keys)."""

    client_ids: tuple[str, ...]
    x25519_public: dict[str, bytes]
    ed25519_public: dict[str, bytes]


def _hkdf_like_sha256(ikm: bytes, salt: bytes, info: bytes) -> bytes:
    """Standard RFC 5869 HMAC-SHA256 Extract-and-Expand (32-byte output)."""
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    okm = hmac.new(prk, info + b"\x01", hashlib.sha256).digest()
    return okm


def generate_tenant_keys(
    client_ids: Sequence[str],
    seed: int = 0,
) -> tuple[dict[str, TenantKeyMaterial], TenantPublicRegistry, bytes]:
    """Deterministically provision `X25519` + `Ed25519` keypairs and a tenant-only group key.

    Note: The aggregator server ONLY receives `TenantPublicRegistry` (public keys) and never
    sees `key_materials` or `group_key`. In live deployment (`Step 08`), each tenant generates
    its own random keypairs locally and runs an encrypted key-exchange setup with peers.
    """
    ordered_ids = tuple(sorted(str(cid) for cid in client_ids))
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("Duplicate client_ids provided to generate_tenant_keys.")
    if len(ordered_ids) < 2:
        raise ValueError("Pairwise masking requires at least 2 clients.")

    materials: dict[str, TenantKeyMaterial] = {}
    x_pub_map: dict[str, bytes] = {}
    ed_pub_map: dict[str, bytes] = {}

    for cid in ordered_ids:
        seed_x = hashlib.sha256(f"trustfed:x25519:{seed}:{cid}".encode()).digest()
        seed_ed = hashlib.sha256(f"trustfed:ed25519:{seed}:{cid}".encode()).digest()

        x_priv = x25519.X25519PrivateKey.from_private_bytes(seed_x)
        x_pub = x_priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        ed_priv = ed25519.Ed25519PrivateKey.from_private_bytes(seed_ed)
        ed_pub = ed_priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        materials[cid] = TenantKeyMaterial(
            client_id=cid,
            x25519_private=x_priv,
            x25519_public_bytes=x_pub,
            ed25519_private=ed_priv,
            ed25519_public_bytes=ed_pub,
        )
        x_pub_map[cid] = x_pub
        ed_pub_map[cid] = ed_pub

    registry = TenantPublicRegistry(
        client_ids=ordered_ids,
        x25519_public=x_pub_map,
        ed25519_public=ed_pub_map,
    )

    # Derive a shared tenant-only group key from pairwise DH secrets (unknown to server)
    pairwise_digest = hashlib.sha256()
    for idx_i in range(len(ordered_ids)):
        for idx_j in range(idx_i + 1, len(ordered_ids)):
            ci, cj = ordered_ids[idx_i], ordered_ids[idx_j]
            peer_pub = x25519.X25519PublicKey.from_public_bytes(x_pub_map[cj])
            shared_raw = materials[ci].x25519_private.exchange(peer_pub)
            pairwise_digest.update(shared_raw)
    group_key = _hkdf_like_sha256(
        pairwise_digest.digest(),
        salt=b"trustfed-sdn-step06-group-key",
        info=f"seed={seed}|n={len(ordered_ids)}".encode(),
    )
    return materials, registry, group_key


def compute_pairwise_shared_secret(
    local_key: TenantKeyMaterial,
    peer_id: str,
    registry: TenantPublicRegistry,
) -> bytes:
    """Compute a pairwise symmetric secret via X25519 Diffie-Hellman."""
    if peer_id not in registry.x25519_public:
        raise KeyError(f"Peer '{peer_id}' not found in TenantPublicRegistry.")
    peer_pub = x25519.X25519PublicKey.from_public_bytes(registry.x25519_public[peer_id])
    raw_dh = local_key.x25519_private.exchange(peer_pub)
    pair = sorted((local_key.client_id, peer_id))
    return _hkdf_like_sha256(
        raw_dh,
        salt=b"trustfed-pairwise-mask-dh",
        info=f"{pair[0]}|{pair[1]}".encode(),
    )


def derive_pairwise_mask(
    shared_secret: bytes,
    client_a: str,
    client_b: str,
    round_idx: int,
    numel: int,
    modulus: int = MODULUS_M,
) -> torch.Tensor:
    """Expand a 32-byte pairwise DH secret into a deterministic `int64` mask in `[0, modulus)`."""
    if numel <= 0:
        raise ValueError(f"numel must be > 0, got {numel}.")
    lo, hi = sorted((str(client_a), str(client_b)))
    domain_seed = hashlib.sha256(
        shared_secret + f"|mask|round={int(round_idx)}|{lo}|{hi}".encode()
    ).digest()
    raw_bytes = hashlib.shake_256(domain_seed).digest(int(numel) * 8)
    arr_u64 = np.frombuffer(raw_bytes, dtype=np.uint64).copy()
    if modulus == MODULUS_M:
        arr_i64 = (arr_u64 & np.uint64(_MODULUS_MASK)).astype(np.int64, copy=False)
    else:
        arr_i64 = (arr_u64 % np.uint64(modulus)).astype(np.int64, copy=False)
    return torch.from_numpy(arr_i64)


def mask_integer_update(
    client_id: str,
    quant_update: torch.Tensor,
    round_idx: int,
    key_material: TenantKeyMaterial,
    registry: TenantPublicRegistry,
    *,
    participants: Sequence[str] | None = None,
    modulus: int = MODULUS_M,
) -> torch.Tensor:
    """Mask an integer update vector with pairwise additive masks that cancel in the sum.

    For ordered participants `(c_0, c_1, ..., c_{K-1})`:
        `y_i = (x_i + sum_{j > i} mask(i, j) - sum_{j < i} mask(j, i)) mod modulus`
    """
    active_ids = sorted(participants) if participants is not None else sorted(registry.client_ids)
    if client_id not in active_ids:
        raise ValueError(f"Client '{client_id}' is not in active participants: {active_ids}")

    x_cpu = quant_update.detach().to(device="cpu", dtype=torch.int64).contiguous()
    numel = x_cpu.numel()
    acc = x_cpu.clone()

    for peer_id in active_ids:
        if peer_id == client_id:
            continue
        shared = compute_pairwise_shared_secret(key_material, peer_id, registry)
        mask = derive_pairwise_mask(
            shared,
            client_id,
            peer_id,
            round_idx=int(round_idx),
            numel=numel,
            modulus=modulus,
        )
        if client_id < peer_id:
            acc.add_(mask)
        else:
            acc.sub_(mask)

    return torch.remainder(acc, int(modulus))


def unmask_modular_sum(
    masked_updates: Sequence[torch.Tensor],
    modulus: int = MODULUS_M,
) -> torch.Tensor:
    """Sum masked modular updates `(sum_i y_i) mod modulus` and map to signed integers.

    Because pairwise masks satisfy `+mask(i, j)` on client `i < j` and `-mask(i, j)` on
    client `j > i`, all pairwise masks cancel identically in `Z_modulus`, returning the exact
    centered integer sum in `[-modulus // 2, modulus // 2)`.
    """
    if not masked_updates:
        raise ValueError("unmask_modular_sum received an empty sequence.")
    mod_int = int(modulus)
    half_mod = mod_int // 2
    acc = torch.zeros_like(masked_updates[0], device="cpu", dtype=torch.int64)
    for y in masked_updates:
        acc.add_(y.detach().to(device="cpu", dtype=torch.int64))
    mod_sum = torch.remainder(acc, mod_int)
    signed_sum = torch.where(mod_sum >= half_mod, mod_sum - mod_int, mod_sum)
    return signed_sum


def pack_masked_48bit(masked_tensor: torch.Tensor) -> bytes:
    """Pack a 1-D `int64` tensor of 48-bit unsigned residues (`[0, 2^48)`) into `6 * d` bytes."""
    arr_cpu = masked_tensor.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy()
    if np.any(arr_cpu < 0) or np.any(arr_cpu >= MODULUS_M):
        raise ValueError("Values out of 48-bit unsigned range [0, 2^48) in pack_masked_48bit.")
    view_u8 = arr_cpu.astype("<i8", copy=False).view(np.uint8).reshape(-1, 8)
    return np.ascontiguousarray(view_u8[:, :6]).tobytes()


def unpack_masked_48bit(payload: bytes, numel: int) -> torch.Tensor:
    """Unpack `6 * numel` bytes produced by `pack_masked_48bit` back into a 1-D `int64` tensor."""
    expected_len = 6 * int(numel)
    if len(payload) != expected_len:
        raise ValueError(
            f"48-bit packed payload size mismatch: got {len(payload)}, expected {expected_len}."
        )
    buf = np.zeros((int(numel), 8), dtype=np.uint8)
    buf[:, :6] = np.frombuffer(payload, dtype=np.uint8).reshape(int(numel), 6)
    arr_i64 = buf.view("<i8").reshape(-1).copy()
    return torch.from_numpy(arr_i64)
