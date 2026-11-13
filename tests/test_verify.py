"""Unit tests for Verifiable Secure Aggregation (`Step 06`: `trustfed/verify/`)."""

from __future__ import annotations

import pytest
import torch

from trustfed.aggregation import SecureAggregator, build_aggregator
from trustfed.fl import ClientUpdate, FedAvgAggregator
from trustfed.models import build_model
from trustfed.verify import (
    DEFAULT_SCALE_BITS,
    MODULUS_M,
    MerkleTree,
    create_signed_submission,
    generate_tenant_keys,
    mask_integer_update,
    measure_quantization_error,
    pack_masked_48bit,
    quantize_weighted_update,
    server_secure_aggregate,
    tamper_dropped_tenant,
    tamper_modified_aggregate,
    tamper_replayed_update,
    tamper_substituted_update,
    unmask_modular_sum,
    unpack_masked_48bit,
    verify_aggregate,
    verify_merkle_proof,
)


def test_quantization_round_trip_precision() -> None:
    """Verify 20-bit fixed-point quantization achieves <= 1e-6 max error and > 100 dB SNR."""
    model = build_model("gru", input_dim=48, num_classes=2, seq_mode="single_step", num_chunks=1)
    flat = model.get_flat_params()
    stats = measure_quantization_error(flat, scale_bits=DEFAULT_SCALE_BITS)

    assert stats["scale_bits"] == float(DEFAULT_SCALE_BITS)
    assert stats["max_abs_error"] <= 1.0 / float(1 << (DEFAULT_SCALE_BITS + 1)) + 1e-12
    assert stats["max_abs_error"] <= 1e-6
    assert stats["snr_db"] > 100.0


@pytest.mark.parametrize("num_tenants", [3, 5, 10])
def test_pairwise_masking_exact_cancellation(num_tenants: int) -> None:
    """Verify pairwise masks cancel bit-for-bit identically in the integer modular sum."""
    client_ids = [f"tenant_{i}" for i in range(num_tenants)]
    materials, registry, _ = generate_tenant_keys(client_ids, seed=42)

    gen = torch.Generator(device="cpu").manual_seed(100 + num_tenants)
    dim = 1024
    weights = [torch.randn(dim, generator=gen) * 0.5 for _ in range(num_tenants)]
    sample_counts = [500 + 250 * i for i in range(num_tenants)]

    quant_updates = [
        quantize_weighted_update(w, n, scale_bits=DEFAULT_SCALE_BITS)
        for w, n in zip(weights, sample_counts, strict=True)
    ]
    exact_sum = sum(quant_updates, torch.zeros(dim, dtype=torch.int64))

    masked_updates = []
    for cid, q_u in zip(client_ids, quant_updates, strict=True):
        y_i = mask_integer_update(
            cid,
            q_u,
            round_idx=3,
            key_material=materials[cid],
            registry=registry,
        )
        # Verify individual masked values are blinded over 48-bit modulus
        assert float(y_i.to(torch.float64).std().item()) > 1e13
        # Verify lossless 6-byte wire packing & unpacking
        packed = pack_masked_48bit(y_i)
        assert len(packed) == 6 * dim
        unpacked = unpack_masked_48bit(packed, dim)
        assert torch.equal(y_i, unpacked)
        masked_updates.append(unpacked)

    recovered_sum = unmask_modular_sum(masked_updates, modulus=MODULUS_M)
    max_int_diff = int((recovered_sum - exact_sum).abs().max().item())
    assert max_int_diff == 0


def test_merkle_tree_and_all_four_tamper_attacks_detected() -> None:
    """Verify Merkle proofs and 100% detection of all 4 specified aggregator tamper attacks."""
    client_ids = [f"tenant_{i}" for i in range(5)]
    materials, registry, group_key = generate_tenant_keys(client_ids, seed=7)
    gen = torch.Generator(device="cpu").manual_seed(77)
    dim = 512

    # Round 1 (earlier round kept by attacker for replay test)
    subs_r1 = [
        create_signed_submission(
            cid,
            torch.randn(dim, generator=gen) * 0.25,
            num_samples=400 + 50 * i,
            round_idx=1,
            key_material=materials[cid],
            registry=registry,
            group_key=group_key,
            seed=7,
        )
        for i, cid in enumerate(client_ids)
    ]
    server_nonces: set[tuple[str, bytes]] = set()
    client_nonces: set[tuple[str, bytes]] = set()
    proof_r1 = server_secure_aggregate(subs_r1, registry, round_idx=1, seen_nonces=server_nonces)
    verdict_r1 = verify_aggregate(
        proof_r1,
        registry,
        group_key,
        expected_round_idx=1,
        seen_nonces=client_nonces,
    )
    assert verdict_r1.ok is True

    # Merkle inclusion check passes for valid leaf and fails for modified leaf
    tree = MerkleTree([s.header.leaf_bytes() for s in subs_r1])
    mp0 = tree.get_proof(0)
    assert verify_merkle_proof(subs_r1[0].header.leaf_bytes(), mp0, tree.root) is True
    assert verify_merkle_proof(subs_r1[1].header.leaf_bytes(), mp0, tree.root) is False

    # Round 2 honest submissions
    subs_r2 = [
        create_signed_submission(
            cid,
            torch.randn(dim, generator=gen) * 0.25,
            num_samples=400 + 50 * i,
            round_idx=2,
            key_material=materials[cid],
            registry=registry,
            group_key=group_key,
            seed=7,
        )
        for i, cid in enumerate(client_ids)
    ]
    proof_r2 = server_secure_aggregate(subs_r2, registry, round_idx=2, seen_nonces=server_nonces)

    # 1a. Tamper: modified aggregate (float shift)
    bad_mod_float = tamper_modified_aggregate(proof_r2, lsb_only=False, delta_float=0.05)
    v_mod_float = verify_aggregate(bad_mod_float, registry, group_key, expected_round_idx=2)
    assert v_mod_float.ok is False

    # 1b. Tamper: modified aggregate (+1 single quantized integer LSB)
    bad_mod_lsb = tamper_modified_aggregate(proof_r2, lsb_only=True)
    v_mod_lsb = verify_aggregate(bad_mod_lsb, registry, group_key, expected_round_idx=2)
    assert v_mod_lsb.ok is False
    assert "checksum" in v_mod_lsb.reason

    # 2. Tamper: dropped tenant
    bad_drop = tamper_dropped_tenant(subs_r2, round_idx=2, drop_index=2)
    v_drop = verify_aggregate(bad_drop, registry, group_key, expected_round_idx=2)
    assert v_drop.ok is False
    assert "dropped" in v_drop.reason

    # 3. Tamper: replayed old update from Round 1
    bad_replay = tamper_replayed_update(subs_r2, subs_r1[0], round_idx=2)
    v_replay = verify_aggregate(
        bad_replay, registry, group_key, expected_round_idx=2, seen_nonces=client_nonces
    )
    assert v_replay.ok is False
    assert "replay" in v_replay.reason

    # 4. Tamper: substituted update (masked vector replaced without group checksum secret)
    bad_sub = tamper_substituted_update(subs_r2, round_idx=2, replace_index=1, seed=999)
    v_sub = verify_aggregate(bad_sub, registry, group_key, expected_round_idx=2)
    assert v_sub.ok is False
    assert "checksum" in v_sub.reason


def test_100_honest_rounds_zero_false_alarms() -> None:
    """Confirm 0 false alarms across 100 consecutive honest rounds."""
    client_ids = ["tenant_1_finance", "tenant_2_healthcare", "tenant_3_education"]
    materials, registry, group_key = generate_tenant_keys(client_ids, seed=11)
    server_nonces: set[tuple[str, bytes]] = set()
    client_nonces: set[tuple[str, bytes]] = set()
    gen = torch.Generator(device="cpu").manual_seed(123)
    dim = 1024

    for round_idx in range(1, 101):
        subs = [
            create_signed_submission(
                cid,
                torch.randn(dim, generator=gen) * 0.3,
                num_samples=800 + 300 * idx,
                round_idx=round_idx,
                key_material=materials[cid],
                registry=registry,
                group_key=group_key,
                seed=11,
            )
            for idx, cid in enumerate(client_ids)
        ]
        proof = server_secure_aggregate(
            subs, registry, round_idx=round_idx, seen_nonces=server_nonces
        )
        verdict = verify_aggregate(
            proof,
            registry,
            group_key,
            expected_round_idx=round_idx,
            seen_nonces=client_nonces,
        )
        assert verdict.ok is True, f"False alarm on honest round {round_idx}: {verdict.reason}"

    assert len(client_nonces) == 300


def test_secure_aggregator_accuracy_equivalence_with_fedavg() -> None:
    """Verify SecureAggregator matches plain FedAvgAggregator to < 1e-5 parameter diff."""
    gen = torch.Generator(device="cpu").manual_seed(9)
    dim = 2048
    init_global = torch.zeros(dim, dtype=torch.float32)

    updates = [
        ClientUpdate(
            client_id=f"tenant_{i}",
            flat_params=torch.randn(dim, generator=gen) * 0.4,
            flat_delta=torch.zeros(dim, dtype=torch.float32),
            num_samples=500 + 400 * i,
            train_loss=0.25,
            local_train_time_sec=0.1,
            upload_bytes=24 + 4 * dim,
            round_idx=1,
        )
        for i in range(3)
    ]

    plain_res = FedAvgAggregator().aggregate(init_global, updates)
    sec_agg = build_aggregator("secure_fedavg", scale_bits=20, seed=0)
    assert isinstance(sec_agg, SecureAggregator)
    sec_res = sec_agg.aggregate(init_global, updates)

    max_diff = float(
        (plain_res.new_global_flat_params - sec_res.new_global_flat_params).abs().max().item()
    )
    assert max_diff < 1e-5
    assert sec_res.metadata["verified"] is True
    assert sec_res.metadata["upload_overhead_ratio"] > 1.0

    # Verify mid-round dropout aborts as specified in Checkpoint 06.1
    with pytest.raises(RuntimeError, match="dropout"):
        sec_agg.aggregate(
            init_global,
            [ClientUpdate(**{**u.__dict__, "round_idx": 2}) for u in updates[:2]],
        )
