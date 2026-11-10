"""Unit tests for Step 03 Federated Learning simulation core and FedAvg baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sim.run_experiment import run_experiment_from_config
from trustfed.fl import (
    ClientUpdate,
    FedAvgAggregator,
    FLClient,
    FLServer,
    deserialize_tensor_payload,
    serialize_tensor_payload,
)
from trustfed.models import build_model, set_all_seeds


def test_serialize_and_deserialize_tensor_payload_exact_bytes() -> None:
    """Verify actual wire serialization byte count (`24 + 4 * numel`) and lossless round-trip."""
    vec = torch.tensor([1.25, -3.5, 0.0, 42.125, -100.75], dtype=torch.float32)
    payload = serialize_tensor_payload(vec, num_samples=1500, round_idx=7)

    assert isinstance(payload, bytes)
    assert len(payload) == 24 + 4 * vec.numel()

    recovered, num_samples, round_idx = deserialize_tensor_payload(payload)
    assert num_samples == 1500
    assert round_idx == 7
    assert torch.equal(vec, recovered)


def test_fedavg_aggregator_matches_manual_weighted_mean() -> None:
    """Verify `FedAvgAggregator` equals a manual sample-count weighted mean on a toy example."""
    global_params = torch.zeros(4, dtype=torch.float32)

    w1 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    w2 = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
    w3 = torch.tensor([100.0, 200.0, 300.0, 400.0], dtype=torch.float32)

    updates = [
        ClientUpdate(
            client_id="tenant_1",
            flat_params=w1,
            flat_delta=w1 - global_params,
            num_samples=10,
            train_loss=0.5,
            local_train_time_sec=0.01,
            upload_bytes=40,
            round_idx=1,
        ),
        ClientUpdate(
            client_id="tenant_2",
            flat_params=w2,
            flat_delta=w2 - global_params,
            num_samples=30,
            train_loss=0.4,
            local_train_time_sec=0.02,
            upload_bytes=40,
            round_idx=1,
        ),
        ClientUpdate(
            client_id="tenant_3",
            flat_params=w3,
            flat_delta=w3 - global_params,
            num_samples=60,
            train_loss=0.3,
            local_train_time_sec=0.03,
            upload_bytes=40,
            round_idx=1,
        ),
    ]

    agg = FedAvgAggregator()
    res = agg.aggregate(global_params, updates)

    expected = 0.10 * w1 + 0.30 * w2 + 0.60 * w3
    assert torch.allclose(res.new_global_flat_params, expected, atol=1e-6, rtol=1e-6)
    assert pytest.approx(res.weights["tenant_1"]) == 0.10
    assert pytest.approx(res.weights["tenant_2"]) == 0.30
    assert pytest.approx(res.weights["tenant_3"]) == 0.60
    assert res.accepted_clients == ["tenant_1", "tenant_2", "tenant_3"]
    assert res.rejected_clients == []


def test_fl_server_determinism_same_seed() -> None:
    """Verify two FLServer runs with the same seed produce bit-for-bit identical parameters."""
    rng = np.random.default_rng(123)
    input_dim = 12
    num_classes = 2
    X_test = rng.standard_normal((80, input_dim)).astype(np.float32)
    y_test = rng.integers(0, num_classes, size=80, dtype=np.int64)

    tenant_data = [
        (
            f"tenant_{i + 1}",
            rng.standard_normal((64 + 16 * i, input_dim)).astype(np.float32),
            rng.integers(0, num_classes, size=64 + 16 * i, dtype=np.int64),
        )
        for i in range(3)
    ]

    def _run_once(seed_val: int) -> tuple[torch.Tensor, dict]:
        set_all_seeds(seed_val)
        device = torch.device("cpu")
        g_model = build_model(
            "gru",
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            seq_mode="single_step",
        ).to(device)
        init_flat = g_model.get_flat_params()

        clients = []
        for idx, (cid, X_tr, y_tr) in enumerate(tenant_data):
            c_model = build_model(
                "gru",
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=16,
                num_layers=1,
                dropout=0.0,
                seq_mode="single_step",
            ).to(device)
            c_model.set_flat_params(init_flat)
            clients.append(
                FLClient(
                    client_id=cid,
                    client_index=idx,
                    model=c_model,
                    X_train=X_tr,
                    y_train=y_tr,
                    X_test=X_test[:25],
                    y_test=y_test[:25],
                    device=device,
                )
            )

        server = FLServer(
            global_model=g_model,
            clients=clients,
            aggregator=FedAvgAggregator(),
            X_test=X_test,
            y_test=y_test,
            device=device,
        )
        out = server.fit(rounds=4, local_epochs=2, batch_size=32, lr=1e-3, seed=seed_val)
        return server.get_global_flat_params(), out

    flat_a, res_a = _run_once(42)
    flat_b, res_b = _run_once(42)
    flat_c, _ = _run_once(99)

    assert torch.equal(flat_a, flat_b)
    assert res_a["final_metrics"]["f1_macro"] == res_b["final_metrics"]["f1_macro"]
    assert res_a["total_comm_bytes"] == res_b["total_comm_bytes"]
    assert res_a["bytes_per_round"] == 3 * 2 * (24 + 4 * flat_a.numel())
    assert not torch.equal(flat_a, flat_c)


def test_runner_reproduces_saved_result(tmp_path: Path) -> None:
    """Verify `sim/run_experiment.py` reproduces a saved result identically."""
    out_dir_1 = tmp_path / "run1"
    out_dir_2 = tmp_path / "run2"

    base_cfg = {
        "experiment_id": "test_repro_insdn",
        "dataset": "insdn",
        "task": "binary",
        "partition_type": "natural",
        "num_tenants": 3,
        "aggregator": "fedavg",
        "rounds": 2,
        "local_epochs": 1,
        "batch_size": 256,
        "lr": 1e-3,
        "seeds": [0],
        "max_train_samples": 600,
        "eval_every": 1,
        "eval_per_tenant_every": 2,
        "model": {
            "type": "gru",
            "hidden_dim": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "seq_mode": "single_step",
            "num_chunks": 1,
        },
    }

    cfg1 = {**base_cfg, "output_dir": str(out_dir_1)}
    cfg2 = {**base_cfg, "output_dir": str(out_dir_2)}

    res1 = run_experiment_from_config(cfg1, device=torch.device("cpu"))
    res2 = run_experiment_from_config(cfg2, device=torch.device("cpu"))

    saved1 = json.loads((out_dir_1 / "summary_test_repro_insdn.json").read_text(encoding="utf-8"))
    saved2 = json.loads((out_dir_2 / "summary_test_repro_insdn.json").read_text(encoding="utf-8"))

    assert saved1["metrics"]["f1_macro"]["mean"] == saved2["metrics"]["f1_macro"]["mean"]
    assert saved1["metrics"]["accuracy"]["mean"] == saved2["metrics"]["accuracy"]["mean"]
    assert (
        saved1["metrics"]["bytes_per_round"]["mean"] == saved2["metrics"]["bytes_per_round"]["mean"]
    )
    assert torch.equal(
        res1["seed_runs"][0]["global_final_flat_params"],
        res2["seed_runs"][0]["global_final_flat_params"],
    )
