"""Live SDN tenant federated learning client (`Step 08`).

Reuses `trustfed.fl.client.FLClient`, `trustfed.privacy.dp_sgd.DPFLClient`,
`trustfed.attacks`, and `trustfed.verify.protocol` without duplicating core ML,
privacy, or cryptographic verification algorithms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from trustfed.attacks import build_attack
from trustfed.fl.aggregator import ClientUpdate
from trustfed.fl.client import FLClient
from trustfed.models import build_model, evaluate_pytorch_model, get_default_device
from trustfed.privacy import (
    DPConfig,
    DPFLClient,
    allocate_budgets,
    compute_sensitivity_scores,
)
from trustfed.verify import (
    DEFAULT_SCALE_BITS,
    MODULUS_M,
    SignedTenantSubmission,
    TenantKeyMaterial,
    TenantPublicRegistry,
    VerificationProof,
    VerificationVerdict,
    create_signed_submission,
    verify_aggregate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIVE_CSV = REPO_ROOT / "data" / "live" / "live_flows_insdn.csv"
DEFAULT_FEATURE_YAML = REPO_ROOT / "configs" / "data" / "features_insdn.yaml"

CANONICAL_LABEL_TO_ID: dict[str, int] = {
    "Benign": 0,
    "DoS": 1,
    "DDoS": 2,
    "Probe": 3,
    "BruteForce": 4,
    "WebAttack": 5,
}


@dataclass
class TenantLiveDataset:
    """Scaled train/val/test partitions for one live SDN testbed tenant."""

    tenant_id: str
    feature_names: list[str]
    X_train: np.ndarray
    y_train: np.ndarray
    y_multi_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


@dataclass
class LiveBundle:
    """Complete live testbed dataset bundle across tenants and global held-out test."""

    feature_names: list[str]
    tenants: dict[str, TenantLiveDataset]
    X_global_test: np.ndarray
    y_global_test: np.ndarray
    sensitivities: dict[str, Any]
    adaptive_epsilons: dict[str, float]


@dataclass
class TenantRoundSubmission:
    """Output returned by `LiveTenantFLClient.train_and_prepare_submission`."""

    update: ClientUpdate
    submission: SignedTenantSubmission
    local_train_ms: float
    dp_ms: float
    mask_commit_sign_ms: float
    epsilon_spent: float
    metadata: dict[str, Any] = field(default_factory=dict)


def load_selected_features(feature_yaml: str | Path = DEFAULT_FEATURE_YAML) -> list[str]:
    """Load ordered training feature names from a `configs/data/features_*.yaml` file."""
    cfg = yaml.safe_load(Path(feature_yaml).read_text(encoding="utf-8"))
    feats = list(cfg["selected_features"])
    if not feats:
        raise ValueError(f"No selected_features found in {feature_yaml}.")
    return feats


def _stratified_split_70_10_20(
    X: np.ndarray,
    y_bin: np.ndarray,
    y_multi: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Perform stratified 70% train / 10% val / 20% test split."""
    idx = np.arange(len(y_bin), dtype=np.int64)
    idx_trainval, idx_test = train_test_split(
        idx, test_size=0.20, random_state=int(seed), stratify=y_bin
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.125, random_state=int(seed) + 1, stratify=y_bin[idx_trainval]
    )
    return (
        X[idx_train],
        y_bin[idx_train],
        y_multi[idx_train],
        X[idx_val],
        y_bin[idx_val],
        X[idx_test],
        y_bin[idx_test],
    )


def load_live_tenant_splits(
    *,
    csv_path: str | Path = DEFAULT_LIVE_CSV,
    feature_yaml: str | Path = DEFAULT_FEATURE_YAML,
    epsilon_total: float = 12.0,
    seed: int = 0,
    z_calibration: str = "global",
) -> LiveBundle:
    """Load `data/live/live_flows_insdn.csv` into per-tenant splits + adaptive DP budgets."""
    feats = load_selected_features(feature_yaml)
    df = pd.read_csv(csv_path)
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Live CSV {csv_path} missing required feature columns: {missing}")
    if "tenant_id" not in df.columns or "y_binary" not in df.columns:
        raise ValueError(f"Live CSV {csv_path} missing 'tenant_id' or 'y_binary' columns.")

    X_all = df[feats].to_numpy(dtype=np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    y_bin_all = df["y_binary"].to_numpy(dtype=np.int64)
    if "canonical_label" in df.columns:
        y_multi_all = np.asarray(
            [CANONICAL_LABEL_TO_ID.get(str(lbl), 1) for lbl in df["canonical_label"]],
            dtype=np.int64,
        )
    else:
        y_multi_all = y_bin_all.copy()

    tenant_names = sorted(df["tenant_id"].astype(str).unique().tolist())
    raw_splits: dict[str, tuple[np.ndarray, ...]] = {}
    train_chunks: list[np.ndarray] = []

    for offset, tid in enumerate(tenant_names):
        mask = df["tenant_id"].astype(str).to_numpy() == tid
        split_tuple = _stratified_split_70_10_20(
            X_all[mask], y_bin_all[mask], y_multi_all[mask], seed=int(seed) + offset * 17
        )
        raw_splits[tid] = split_tuple
        train_chunks.append(split_tuple[0])

    global_scaler = StandardScaler().fit(np.concatenate(train_chunks, axis=0))

    tenants: dict[str, TenantLiveDataset] = {}
    multi_labels_for_sens: dict[str, np.ndarray] = {}
    global_test_X: list[np.ndarray] = []
    global_test_y: list[np.ndarray] = []

    for tid in tenant_names:
        X_tr, y_tr, ym_tr, X_va, y_va, X_te, y_te = raw_splits[tid]
        scaler = global_scaler if z_calibration == "global" else StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr).astype(np.float32)
        X_va_s = scaler.transform(X_va).astype(np.float32)
        X_te_s = scaler.transform(X_te).astype(np.float32)

        tenants[tid] = TenantLiveDataset(
            tenant_id=tid,
            feature_names=feats,
            X_train=X_tr_s,
            y_train=y_tr,
            y_multi_train=ym_tr,
            X_val=X_va_s,
            y_val=y_va,
            X_test=X_te_s,
            y_test=y_te,
            scaler=scaler,
        )
        multi_labels_for_sens[tid] = ym_tr
        global_test_X.append(X_te_s)
        global_test_y.append(y_te)

    sensitivities = compute_sensitivity_scores(
        multi_labels_for_sens,
        num_classes=len(CANONICAL_LABEL_TO_ID),
        benign_class=0,
    )
    score_map = {tid: sens.score for tid, sens in sensitivities.items()}
    adaptive_eps = allocate_budgets(score_map, float(epsilon_total), policy="adaptive")

    return LiveBundle(
        feature_names=feats,
        tenants=tenants,
        X_global_test=np.concatenate(global_test_X, axis=0),
        y_global_test=np.concatenate(global_test_y, axis=0),
        sensitivities=sensitivities,
        adaptive_epsilons=adaptive_eps,
    )


def build_live_gru(
    input_dim: int = 19,
    *,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.0,
    seq_mode: str = "single_step",
    num_chunks: int = 1,
) -> Any:
    """Construct the approved GRU classifier (`Step 02`) for live SDN traffic."""
    return build_model(
        model_type="gru",
        input_dim=int(input_dim),
        num_classes=2,
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        dropout=float(dropout),
        seq_mode=seq_mode,
        num_chunks=int(num_chunks),
    )


class LiveTenantFLClient:
    """Live SDN tenant FL client handling training, adaptive DP, signing, and proof verification."""

    def __init__(
        self,
        tenant_id: str,
        client_index: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        key_material: TenantKeyMaterial,
        registry: TenantPublicRegistry,
        group_key: bytes,
        rounds: int = 12,
        local_epochs: int = 1,
        dp_epsilon: float | None = None,
        dp_delta: float = 1e-5,
        dp_max_grad_norm: float = 0.5,
        dp_batch_size: int = 64,
        attack_name: str | None = None,
        attack_kwargs: dict[str, Any] | None = None,
        scale_bits: int = DEFAULT_SCALE_BITS,
        modulus: int = MODULUS_M,
        device: torch.device | None = None,
        use_class_weights: bool = True,
    ) -> None:
        self.tenant_id = str(tenant_id)
        self.client_index = int(client_index)
        self.input_dim = int(X_train.shape[1])
        self.device = device if device is not None else get_default_device(prefer_gpu=False)
        self.key_material = key_material
        self.registry = registry
        self.group_key = bytes(group_key)
        self.scale_bits = int(scale_bits)
        self.modulus = int(modulus)
        self.local_epochs = int(local_epochs)
        self.rounds = int(rounds)
        self.dp_epsilon = float(dp_epsilon) if dp_epsilon is not None else None
        self.dp_delta = float(dp_delta)

        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        self.model = build_live_gru(self.input_dim).to(self.device)
        self.attack = build_attack(attack_name, **(attack_kwargs or {})) if attack_name else None

        self._core_client: FLClient | DPFLClient
        if self.dp_epsilon is not None:
            dp_cfg = DPConfig(
                target_epsilon=self.dp_epsilon,
                delta=self.dp_delta,
                max_grad_norm=float(dp_max_grad_norm),
                expected_batch_size=int(dp_batch_size),
            )
            self._core_client = DPFLClient(
                client_id=self.tenant_id,
                client_index=self.client_index,
                model=self.model,
                X_train=X_train,
                y_train=y_train,
                dp_config=dp_cfg,
                rounds=self.rounds,
                local_epochs=self.local_epochs,
                X_test=X_test,
                y_test=y_test,
                device=self.device,
                use_class_weights=use_class_weights,
                attack=self.attack,
            )
        else:
            self._core_client = FLClient(
                client_id=self.tenant_id,
                client_index=self.client_index,
                model=self.model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                device=self.device,
                use_class_weights=use_class_weights,
                attack=self.attack,
            )

        self.deployed_flat_params: torch.Tensor = self.model.get_flat_params().clone()
        self.verified_round: int = 0
        self.seen_nonces: set[tuple[str, bytes]] = set()
        self.security_events: list[dict[str, Any]] = []
        self._last_update: ClientUpdate | None = None

    @property
    def num_samples(self) -> int:
        return self._core_client.num_samples

    @property
    def is_malicious(self) -> bool:
        return self._core_client.is_malicious

    @property
    def epsilon_spent(self) -> float:
        if isinstance(self._core_client, DPFLClient):
            return float(self._core_client.epsilon_spent)
        return 0.0

    def enable_attack(self, attack_name: str, **attack_kwargs: Any) -> None:
        """Dynamically activate a poisoning attack during live rounds (`Scenario 2`)."""
        atk = build_attack(attack_name, **attack_kwargs)
        self.attack = atk
        self._core_client.attack = atk
        if atk is not None:
            raw_X = self._core_client.X_train_tensor.detach().cpu().numpy()
            raw_y = self._core_client.y_train_tensor.detach().cpu().numpy()
            X_p, y_p = atk.poison_dataset(
                raw_X,
                raw_y,
                num_classes=int(self.model.num_classes),
                seed=int(self.client_index) * 997,
            )
            self._core_client.X_train_tensor = torch.as_tensor(
                X_p, dtype=torch.float32, device=self.device
            )
            self._core_client.y_train_tensor = torch.as_tensor(
                y_p, dtype=torch.long, device=self.device
            )

    def train_and_prepare_submission(
        self,
        global_flat_params: torch.Tensor,
        *,
        round_idx: int,
        participants: list[str] | tuple[str, ...] | None = None,
        lr: float = 1e-3,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ) -> TenantRoundSubmission:
        """Run local training (plus DP if active), then mask, commit, and Ed25519 sign."""
        t0 = time.perf_counter()
        upd = self._core_client.local_train(
            global_flat_params,
            round_idx=int(round_idx),
            local_epochs=self.local_epochs,
            batch_size=int(batch_size),
            lr=float(lr),
            weight_decay=float(weight_decay),
            seed=int(seed),
        )
        t_train_ms = (time.perf_counter() - t0) * 1000.0
        self._last_update = upd

        dp_active = isinstance(self._core_client, DPFLClient)
        dp_ms = t_train_ms if dp_active else 0.0
        local_train_only_ms = max(0.0, float(upd.local_train_time_sec) * 1000.0)

        sub = self.resign_for_cohort(
            upd.flat_params,
            num_samples=upd.num_samples,
            round_idx=int(round_idx),
            participants=participants,
            seed=int(seed),
        )
        mask_commit_sign_ms = sub.quantize_ms + sub.mask_ms + sub.checksum_ms + sub.sign_ms

        return TenantRoundSubmission(
            update=upd,
            submission=sub,
            local_train_ms=local_train_only_ms,
            dp_ms=dp_ms,
            mask_commit_sign_ms=float(mask_commit_sign_ms),
            epsilon_spent=self.epsilon_spent,
            metadata={"round_idx": int(round_idx), "is_malicious": self.is_malicious},
        )

    def resign_for_cohort(
        self,
        flat_params: torch.Tensor,
        *,
        num_samples: int,
        round_idx: int,
        participants: list[str] | tuple[str, ...] | None = None,
        seed: int = 0,
    ) -> SignedTenantSubmission:
        """Create pairwise-masked & signed submission for a specific active tenant cohort."""
        active_cohort = tuple(sorted(participants)) if participants else self.registry.client_ids
        return create_signed_submission(
            client_id=self.tenant_id,
            flat_params=flat_params,
            num_samples=int(num_samples),
            round_idx=int(round_idx),
            key_material=self.key_material,
            registry=self.registry,
            group_key=self.group_key,
            participants=active_cohort,
            scale_bits=self.scale_bits,
            modulus=self.modulus,
            seed=int(seed),
        )

    def verify_and_deploy(
        self,
        proof: VerificationProof,
        *,
        expected_round_idx: int,
        expected_clients: list[str] | tuple[str, ...] | None = None,
    ) -> VerificationVerdict:
        """Verify server proof cryptographically; update deployed weights only on success."""
        verdict = verify_aggregate(
            proof,
            registry=self.registry,
            group_key=self.group_key,
            expected_round_idx=int(expected_round_idx),
            expected_clients=expected_clients,
            seen_nonces=self.seen_nonces,
            scale_bits=self.scale_bits,
        )
        if verdict.ok:
            new_weights = proof.aggregate_float.detach().to(device=self.device).clone()
            self.deployed_flat_params = new_weights
            self.model.set_flat_params(new_weights)
            self.verified_round = int(expected_round_idx)
        else:
            # Rollback / preserve last verified global model parameters cleanly
            self.model.set_flat_params(self.deployed_flat_params)
            self.security_events.append(
                {
                    "round": int(expected_round_idx),
                    "tenant_id": self.tenant_id,
                    "event": "tampered_aggregate_rejected",
                    "reason": verdict.reason,
                    "preserved_round": int(self.verified_round),
                }
            )
        return verdict

    def evaluate_deployed(self) -> dict[str, Any]:
        """Evaluate the currently deployed model on this tenant's local test set."""
        self.model.set_flat_params(self.deployed_flat_params)
        return evaluate_pytorch_model(
            self.model,
            (
                self.X_test
                if self.X_test is not None
                else np.zeros((1, self.input_dim), dtype=np.float32)
            ),
            self.y_test if self.y_test is not None else np.zeros(1, dtype=np.int64),
            batch_size=1024,
            device=self.device,
        )
