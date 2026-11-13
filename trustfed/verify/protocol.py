"""End-to-end Verifiable Secure Aggregation protocol, verify(), and tamper tests (`Step 06`)."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from trustfed.verify.commitment import (
    PRIME_1,
    PRIME_2,
    TenantCommitmentHeader,
    compute_linear_checksums,
    derive_challenge_vectors,
    sign_commitment_header,
    verify_commitment_header_signature,
)
from trustfed.verify.masking import (
    MODULUS_M,
    TenantKeyMaterial,
    TenantPublicRegistry,
    mask_integer_update,
    pack_masked_48bit,
    unmask_modular_sum,
)
from trustfed.verify.merkle import MerkleInclusionProof, MerkleTree, verify_merkle_proof
from trustfed.verify.quantize import (
    DEFAULT_SCALE_BITS,
    dequantize_weighted_sum,
    quantize_weighted_update,
)


@dataclass
class SignedTenantSubmission:
    """Complete message sent by one tenant to the aggregator for a single FL round."""

    header: TenantCommitmentHeader
    masked_update: torch.Tensor
    masked_wire_bytes: bytes
    header_wire_bytes: bytes
    quantize_ms: float
    mask_ms: float
    checksum_ms: float
    sign_ms: float

    @property
    def total_upload_bytes(self) -> int:
        return len(self.masked_wire_bytes) + len(self.header_wire_bytes)


@dataclass
class VerificationProof:
    """Aggregate + cryptographic verification bundle published by the server."""

    round_idx: int
    merkle_root: bytes
    included_client_ids: tuple[str, ...]
    total_samples: int
    unmasked_sum_int: torch.Tensor
    aggregate_float: torch.Tensor
    signed_headers: dict[str, TenantCommitmentHeader]
    merkle_proofs: dict[str, MerkleInclusionProof]
    server_time_ms: float
    proof_wire_bytes: int


@dataclass
class VerificationVerdict:
    """Result returned by tenant-side `verify_aggregate(...)`."""

    ok: bool
    reason: str
    verify_time_ms: float
    checked_clients: int
    details: dict[str, Any] = field(default_factory=dict)


def derive_round_nonce(client_id: str, round_idx: int, seed: int = 0) -> bytes:
    """Deterministically derive a unique 16-byte per-(client, round, seed) replay-guard nonce."""
    digest = hashlib.sha256(
        f"trustfed:nonce:{int(seed)}:{str(client_id)}:{int(round_idx)}".encode()
    ).digest()
    return digest[:16]


def create_signed_submission(
    client_id: str,
    flat_params: torch.Tensor,
    num_samples: int,
    round_idx: int,
    key_material: TenantKeyMaterial,
    registry: TenantPublicRegistry,
    group_key: bytes,
    *,
    participants: Sequence[str] | None = None,
    scale_bits: int = DEFAULT_SCALE_BITS,
    modulus: int = MODULUS_M,
    nonce: bytes | None = None,
    seed: int = 0,
) -> SignedTenantSubmission:
    """Run tenant-side quantization, masking, linear checksum, and Ed25519 signing."""
    eff_samples = max(1, int(num_samples))
    eff_nonce = (
        bytes(nonce) if nonce is not None else derive_round_nonce(client_id, round_idx, seed=seed)
    )

    t0 = time.perf_counter()
    quant_weighted = quantize_weighted_update(flat_params, eff_samples, scale_bits=scale_bits)
    t1 = time.perf_counter()

    masked_int = mask_integer_update(
        client_id,
        quant_weighted,
        round_idx=int(round_idx),
        key_material=key_material,
        registry=registry,
        participants=participants,
        modulus=modulus,
    )
    masked_wire = pack_masked_48bit(masked_int)
    t2 = time.perf_counter()

    a1, a2 = derive_challenge_vectors(group_key, int(round_idx), quant_weighted.numel())
    chk1, chk2 = compute_linear_checksums(quant_weighted, a1, a2)
    t3 = time.perf_counter()

    header = sign_commitment_header(
        client_id=client_id,
        round_idx=int(round_idx),
        nonce=eff_nonce,
        num_samples=eff_samples,
        masked_wire_bytes=masked_wire,
        checksum_1=chk1,
        checksum_2=chk2,
        ed25519_private=key_material.ed25519_private,
    )
    header_wire = header.to_bytes()
    t4 = time.perf_counter()

    return SignedTenantSubmission(
        header=header,
        masked_update=masked_int,
        masked_wire_bytes=masked_wire,
        header_wire_bytes=header_wire,
        quantize_ms=(t1 - t0) * 1000.0,
        mask_ms=(t2 - t1) * 1000.0,
        checksum_ms=(t3 - t2) * 1000.0,
        sign_ms=(t4 - t3) * 1000.0,
    )


def server_secure_aggregate(
    submissions: Sequence[SignedTenantSubmission],
    registry: TenantPublicRegistry,
    round_idx: int,
    *,
    expected_clients: Sequence[str] | None = None,
    seen_nonces: set[tuple[str, bytes]] | None = None,
    scale_bits: int = DEFAULT_SCALE_BITS,
    modulus: int = MODULUS_M,
) -> VerificationProof:
    """Server-side verification of tenant submissions, unmasking, and Merkle proof construction."""
    t0 = time.perf_counter()
    if not submissions:
        raise ValueError("Round aborted: no submissions received.")

    exp_ids = (
        sorted(expected_clients) if expected_clients is not None else list(registry.client_ids)
    )
    by_id = {sub.header.client_id: sub for sub in submissions}
    if len(by_id) != len(submissions):
        raise RuntimeError("Duplicate client submission detected in round.")

    missing = [cid for cid in exp_ids if cid not in by_id]
    if missing:
        raise RuntimeError(
            f"Round {round_idx} aborted: tenant dropout detected (missing {missing})."
        )

    ordered_subs = [by_id[cid] for cid in exp_ids]
    leaf_payloads: list[bytes] = []
    signed_headers: dict[str, TenantCommitmentHeader] = {}

    for sub in ordered_subs:
        hdr = sub.header
        if int(hdr.round_idx) != int(round_idx):
            raise RuntimeError(
                f"Replay/round mismatch for '{hdr.client_id}': got round {hdr.round_idx}, "
                f"expected {round_idx}."
            )
        if seen_nonces is not None and (hdr.client_id, hdr.nonce) in seen_nonces:
            raise RuntimeError(f"Replayed nonce detected for '{hdr.client_id}'.")
        if hdr.client_id not in registry.ed25519_public:
            raise RuntimeError(f"Unknown tenant '{hdr.client_id}'.")
        if not verify_commitment_header_signature(hdr, registry.ed25519_public[hdr.client_id]):
            raise RuntimeError(f"Invalid Ed25519 signature for '{hdr.client_id}'.")
        if hashlib.sha256(sub.masked_wire_bytes).digest() != hdr.masked_payload_sha256:
            raise RuntimeError(f"Masked payload hash mismatch for '{hdr.client_id}'.")

        leaf_payloads.append(hdr.leaf_bytes())
        signed_headers[hdr.client_id] = hdr

    if seen_nonces is not None:
        for hdr in signed_headers.values():
            seen_nonces.add((hdr.client_id, hdr.nonce))

    tree = MerkleTree(leaf_payloads)
    merkle_proofs = {
        ordered_subs[idx].header.client_id: tree.get_proof(idx) for idx in range(len(ordered_subs))
    }

    masked_tensors = [sub.masked_update for sub in ordered_subs]
    unmasked_sum_int = unmask_modular_sum(masked_tensors, modulus=modulus)
    total_samples = sum(max(1, int(sub.header.num_samples)) for sub in ordered_subs)
    agg_float = dequantize_weighted_sum(
        unmasked_sum_int,
        total_samples=total_samples,
        scale_bits=scale_bits,
        dtype=torch.float32,
    )

    proof_wire_bytes = sum(
        len(sub.header_wire_bytes) + 32 * (len(mp.siblings) + 1)
        for sub, mp in zip(
            ordered_subs,
            [merkle_proofs[sub.header.client_id] for sub in ordered_subs],
            strict=True,
        )
    ) + len(tree.root)
    server_time_ms = (time.perf_counter() - t0) * 1000.0

    return VerificationProof(
        round_idx=int(round_idx),
        merkle_root=tree.root,
        included_client_ids=tuple(exp_ids),
        total_samples=int(total_samples),
        unmasked_sum_int=unmasked_sum_int,
        aggregate_float=agg_float,
        signed_headers=signed_headers,
        merkle_proofs=merkle_proofs,
        server_time_ms=float(server_time_ms),
        proof_wire_bytes=int(proof_wire_bytes),
    )


def verify_aggregate(
    proof: VerificationProof,
    registry: TenantPublicRegistry,
    group_key: bytes,
    *,
    expected_round_idx: int,
    expected_clients: Sequence[str] | None = None,
    seen_nonces: set[tuple[str, bytes]] | None = None,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> VerificationVerdict:
    """Tenant-side `verify()`: validates membership, signatures, nonces, Merkle proofs, and sums."""
    t0 = time.perf_counter()
    exp_clients = (
        sorted(expected_clients) if expected_clients is not None else list(registry.client_ids)
    )

    def _fail(reason: str, **details: Any) -> VerificationVerdict:
        return VerificationVerdict(
            ok=False,
            reason=reason,
            verify_time_ms=(time.perf_counter() - t0) * 1000.0,
            checked_clients=len(proof.included_client_ids),
            details=details,
        )

    if int(proof.round_idx) != int(expected_round_idx):
        return _fail(
            f"round_mismatch: proof round {proof.round_idx} != expected {expected_round_idx}"
        )

    included_sorted = sorted(proof.included_client_ids)
    if included_sorted != exp_clients:
        missing = sorted(set(exp_clients) - set(included_sorted))
        extra = sorted(set(included_sorted) - set(exp_clients))
        return _fail(
            f"dropped_or_unexpected_tenant: missing={missing}, extra={extra}",
            missing=missing,
            extra=extra,
        )

    expected_checksum_1 = 0
    expected_checksum_2 = 0
    sum_samples = 0

    for cid in exp_clients:
        hdr = proof.signed_headers.get(cid)
        mp = proof.merkle_proofs.get(cid)
        if hdr is None or mp is None:
            return _fail(f"missing_header_or_merkle_proof:{cid}")
        if hdr.client_id != cid:
            return _fail(f"identity_mismatch:{cid}!={hdr.client_id}")
        if int(hdr.round_idx) != int(expected_round_idx):
            return _fail(
                f"replayed_update_round_mismatch:{cid}:hdr_round={hdr.round_idx}"
                f"!=expected={expected_round_idx}"
            )
        if seen_nonces is not None and (cid, hdr.nonce) in seen_nonces:
            return _fail(f"replayed_nonce:{cid}")
        pub_ed = registry.ed25519_public.get(cid)
        if pub_ed is None or not verify_commitment_header_signature(hdr, pub_ed):
            return _fail(f"invalid_signature_or_commitment:{cid}")
        if not verify_merkle_proof(hdr.leaf_bytes(), mp, proof.merkle_root):
            return _fail(f"merkle_inclusion_failed:{cid}")

        expected_checksum_1 = (expected_checksum_1 + int(hdr.checksum_1)) % PRIME_1
        expected_checksum_2 = (expected_checksum_2 + int(hdr.checksum_2)) % PRIME_2
        sum_samples += max(1, int(hdr.num_samples))

    if sum_samples != int(proof.total_samples):
        return _fail(
            f"total_samples_mismatch:{sum_samples}!={proof.total_samples}",
        )

    a1, a2 = derive_challenge_vectors(
        group_key, int(expected_round_idx), proof.unmasked_sum_int.numel()
    )
    obs_c1, obs_c2 = compute_linear_checksums(proof.unmasked_sum_int, a1, a2)
    if obs_c1 != expected_checksum_1 or obs_c2 != expected_checksum_2:
        return _fail(
            "linear_checksum_mismatch",
            expected=(expected_checksum_1, expected_checksum_2),
            observed=(obs_c1, obs_c2),
        )

    # Confirm the published float32 aggregate matches the verified integer sum
    reconstructed = dequantize_weighted_sum(
        proof.unmasked_sum_int,
        total_samples=sum_samples,
        scale_bits=scale_bits,
        dtype=torch.float32,
    )
    max_float_diff = float(
        (reconstructed - proof.aggregate_float.detach().cpu()).abs().max().item()
    )
    if max_float_diff > 1e-6:
        return _fail(f"float_dequantization_mismatch:max_diff={max_float_diff:.3e}")

    if seen_nonces is not None:
        for cid in exp_clients:
            seen_nonces.add((cid, proof.signed_headers[cid].nonce))

    return VerificationVerdict(
        ok=True,
        reason="verified",
        verify_time_ms=(time.perf_counter() - t0) * 1000.0,
        checked_clients=len(exp_clients),
        details={"max_float_diff": max_float_diff},
    )


def _rebuild_proof_from_submissions(
    submissions: Sequence[SignedTenantSubmission],
    round_idx: int,
    *,
    scale_bits: int = DEFAULT_SCALE_BITS,
    modulus: int = MODULUS_M,
) -> VerificationProof:
    """Build a server proof WITHOUT validation checks (used to simulate malicious tampering)."""
    leaf_payloads = [sub.header.leaf_bytes() for sub in submissions]
    tree = MerkleTree(leaf_payloads)
    merkle_proofs = {
        sub.header.client_id: tree.get_proof(idx) for idx, sub in enumerate(submissions)
    }
    signed_headers = {sub.header.client_id: sub.header for sub in submissions}
    unmasked_sum = unmask_modular_sum([sub.masked_update for sub in submissions], modulus=modulus)
    total_samples = sum(max(1, int(sub.header.num_samples)) for sub in submissions)
    agg_float = dequantize_weighted_sum(
        unmasked_sum,
        total_samples=total_samples,
        scale_bits=scale_bits,
        dtype=torch.float32,
    )
    return VerificationProof(
        round_idx=int(round_idx),
        merkle_root=tree.root,
        included_client_ids=tuple(sub.header.client_id for sub in submissions),
        total_samples=int(total_samples),
        unmasked_sum_int=unmasked_sum,
        aggregate_float=agg_float,
        signed_headers=signed_headers,
        merkle_proofs=merkle_proofs,
        server_time_ms=0.0,
        proof_wire_bytes=len(tree.root),
    )


def tamper_modified_aggregate(
    proof: VerificationProof,
    *,
    lsb_only: bool = False,
    delta_float: float = 0.05,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> VerificationProof:
    """Tamper #1: Malicious aggregator modifies the aggregate (1 LSB integer or float shift)."""
    bad_sum = proof.unmasked_sum_int.clone()
    if lsb_only:
        bad_sum[0] += 1
        bad_float = dequantize_weighted_sum(
            bad_sum,
            total_samples=proof.total_samples,
            scale_bits=scale_bits,
            dtype=torch.float32,
        )
    else:
        step = int(round(delta_float * proof.total_samples * (1 << scale_bits)))
        bad_sum[0] += max(1, step)
        bad_float = proof.aggregate_float.clone()
        bad_float[0] += float(delta_float)

    return VerificationProof(
        round_idx=proof.round_idx,
        merkle_root=proof.merkle_root,
        included_client_ids=proof.included_client_ids,
        total_samples=proof.total_samples,
        unmasked_sum_int=bad_sum,
        aggregate_float=bad_float,
        signed_headers=dict(proof.signed_headers),
        merkle_proofs=dict(proof.merkle_proofs),
        server_time_ms=proof.server_time_ms,
        proof_wire_bytes=proof.proof_wire_bytes,
    )


def tamper_dropped_tenant(
    submissions: Sequence[SignedTenantSubmission],
    round_idx: int,
    *,
    drop_index: int = -1,
    scale_bits: int = DEFAULT_SCALE_BITS,
    modulus: int = MODULUS_M,
) -> VerificationProof:
    """Tamper #2: Malicious aggregator omits one tenant's submission from aggregation."""
    if len(submissions) < 2:
        raise ValueError("Need at least 2 submissions to drop one.")
    kept = [sub for idx, sub in enumerate(submissions) if idx != (drop_index % len(submissions))]
    return _rebuild_proof_from_submissions(
        kept, round_idx=round_idx, scale_bits=scale_bits, modulus=modulus
    )


def tamper_replayed_update(
    current_submissions: Sequence[SignedTenantSubmission],
    old_submission: SignedTenantSubmission,
    round_idx: int,
    *,
    scale_bits: int = DEFAULT_SCALE_BITS,
    modulus: int = MODULUS_M,
) -> VerificationProof:
    """Tamper #3: Malicious aggregator substitutes a tenant's earlier-round signed update."""
    target_id = old_submission.header.client_id
    mutated = [
        old_submission if sub.header.client_id == target_id else sub for sub in current_submissions
    ]
    return _rebuild_proof_from_submissions(
        mutated, round_idx=round_idx, scale_bits=scale_bits, modulus=modulus
    )


def tamper_substituted_update(
    submissions: Sequence[SignedTenantSubmission],
    round_idx: int,
    *,
    replace_index: int = 0,
    seed: int = 999,
    scale_bits: int = DEFAULT_SCALE_BITS,
    modulus: int = MODULUS_M,
) -> VerificationProof:
    """Tamper #4: Malicious aggregator replaces a tenant's update with a forged vector/checksum."""
    idx_target = replace_index % len(submissions)
    target = submissions[idx_target]
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    forged_masked = torch.randint(
        0,
        modulus,
        size=target.masked_update.shape,
        generator=gen,
        dtype=torch.int64,
    )
    # Keep header metadata identical or update header payload digest without valid Ed25519 signature
    forged_sub = SignedTenantSubmission(
        header=target.header,
        masked_update=forged_masked,
        masked_wire_bytes=target.masked_wire_bytes,
        header_wire_bytes=target.header_wire_bytes,
        quantize_ms=target.quantize_ms,
        mask_ms=target.mask_ms,
        checksum_ms=target.checksum_ms,
        sign_ms=target.sign_ms,
    )
    mutated = [forged_sub if i == idx_target else sub for i, sub in enumerate(submissions)]
    return _rebuild_proof_from_submissions(
        mutated, round_idx=round_idx, scale_bits=scale_bits, modulus=modulus
    )
