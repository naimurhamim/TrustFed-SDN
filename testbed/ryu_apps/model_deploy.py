"""Off-event-loop real-time flow classifier & automated OVS mitigation (`Step 08`).

Runs GRU inference strictly in a separate worker thread/queue outside the OpenFlow
event loop (`AsyncFlowInferenceWorker`), enforces safety-guarded temporary OVS
drop rules with automatic TTL expiry (`MitigationPolicyManager`), and optionally
subclasses `TenantController13` inside the Ubuntu OS-Ken environment.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # Light SDN controller runtime without PyTorch
    torch = None  # type: ignore[assignment]

from testbed.traffic.safety_guard import SafetyGuardError, validate_testbed_target

try:
    from os_ken.controller import ofp_event
    from os_ken.controller.handler import CONFIG_DISPATCHER, set_ev_cls

    from testbed.ryu_apps.tenant_controller_13 import TenantController13
except ImportError:  # Windows/ML env without os_ken
    ofp_event = None  # type: ignore[assignment]
    CONFIG_DISPATCHER = None  # type: ignore[assignment]
    set_ev_cls = None  # type: ignore[assignment]
    TenantController13 = None  # type: ignore[assignment,misc]

__all__ = [
    "AsyncFlowInferenceWorker",
    "FlowClassificationJob",
    "MitigationEvent",
    "MitigationPolicyManager",
    "MitigationRule",
    "ModelDeployController13",
    "calibrate_attack_threshold",
]


def _binary_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s: list[float] = []
    for cls in (0, 1):
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return 0.5 * (f1s[0] + f1s[1])


def calibrate_attack_threshold(
    model: torch.nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    max_fpr: float = 0.12,
    device: torch.device | None = None,
) -> tuple[float, dict[str, float]]:
    """Select an attack probability threshold on validation data capping benign FPR."""
    if torch is None:
        raise RuntimeError("PyTorch is required for calibrate_attack_threshold.")
    dev = device if device is not None else torch.device("cpu")
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(np.asarray(X_val, dtype=np.float32)).to(dev)
        probs = torch.softmax(model(x_t), dim=-1)[:, 1].cpu().numpy()

    y_true = np.asarray(y_val, dtype=np.int64)
    benign_mask = y_true == 0
    attack_mask = y_true == 1

    best_th = 0.5
    best_key = (False, -1.0, -1.0)
    best_stats = {"val_f1_macro": 0.0, "val_fpr": 1.0, "val_adr": 0.0}
    for th in np.linspace(0.30, 0.98, 69):
        preds = (probs >= float(th)).astype(np.int64)
        fpr = float((preds[benign_mask] == 1).mean()) if benign_mask.any() else 0.0
        adr = float((preds[attack_mask] == 1).mean()) if attack_mask.any() else 0.0
        f1 = _binary_macro_f1(y_true, preds)
        key = (fpr <= float(max_fpr), f1, -fpr)
        if key > best_key:
            best_key = key
            best_th = float(th)
            best_stats = {"val_f1_macro": f1, "val_fpr": fpr, "val_adr": adr}
    return best_th, best_stats


@dataclass
class MitigationRule:
    """Active or expired OpenFlow 1.3 temporary drop rule."""

    tenant_id: str
    bridge: str
    ipv4_src: str
    priority: int
    idle_timeout: int
    hard_timeout: int
    installed_at: float
    expires_at: float
    prob_attack: float
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "bridge": self.bridge,
            "ipv4_src": self.ipv4_src,
            "priority": int(self.priority),
            "idle_timeout": int(self.idle_timeout),
            "hard_timeout": int(self.hard_timeout),
            "installed_at": float(self.installed_at),
            "expires_at": float(self.expires_at),
            "prob_attack": float(self.prob_attack),
            "active": bool(self.active),
        }


@dataclass
class MitigationEvent:
    """Timeline audit event for flow detection, rule installation, or expiry."""

    flow_id: str
    tenant_id: str
    ipv4_src: str
    ground_truth_label: int | None
    predicted_label: int
    prob_attack: float
    inference_latency_ms: float
    mitigation_latency_ms: float
    action: str
    timestamp: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowClassificationJob:
    """Flow record item queued by the controller/capture pipeline for worker inference."""

    flow_id: str
    tenant_id: str
    src_ip: str
    dst_ip: str
    feature_vector: np.ndarray
    submitted_ts: float = field(default_factory=time.perf_counter)
    bridge: str = "s1"
    ground_truth_label: int | None = None
    attack_start_ts: float | None = None


class MitigationPolicyManager:
    """Manages safety-guarded temporary OpenFlow 1.3 drop rules with automatic expiry."""

    def __init__(
        self,
        *,
        blocklist_path: str | Path | None = None,
        ovs_bridge: str | None = None,
        apply_ovs_cli: bool = False,
        priority: int = 300,
        idle_timeout: int = 60,
        hard_timeout: int = 120,
    ) -> None:
        self.blocklist_path = Path(blocklist_path) if blocklist_path else None
        self.ovs_bridge = ovs_bridge
        self.apply_ovs_cli = bool(apply_ovs_cli)
        self.priority = int(priority)
        self.idle_timeout = int(idle_timeout)
        self.hard_timeout = int(hard_timeout)
        self._lock = threading.Lock()
        self.active_rules: dict[tuple[str, str], MitigationRule] = {}
        self.rule_history: list[MitigationRule] = []
        self._datapath_hook: Any | None = None

    def attach_datapath(self, datapath: Any) -> None:
        """Attach a live OS-Ken OpenFlow datapath for instant `OFPFlowMod` push."""
        self._datapath_hook = datapath

    def _sync_blocklist_file(self) -> None:
        if self.blocklist_path is None:
            return
        self.blocklist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [rule.to_dict() for rule in self.active_rules.values() if rule.active]
        tmp_path = self.blocklist_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.blocklist_path)

    def _install_openflow_drop(self, bridge: str, ipv4_src: str) -> None:
        if self._datapath_hook is not None:
            parser = self._datapath_hook.ofproto_parser
            match = parser.OFPMatch(eth_type=0x0800, ipv4_src=ipv4_src)
            mod = parser.OFPFlowMod(
                datapath=self._datapath_hook,
                priority=self.priority,
                idle_timeout=self.idle_timeout,
                hard_timeout=self.hard_timeout,
                match=match,
                instructions=[],
            )
            self._datapath_hook.send_msg(mod)
            return

        if self.apply_ovs_cli:
            rule_spec = (
                f"priority={self.priority},"
                f"idle_timeout={self.idle_timeout},"
                f"hard_timeout={self.hard_timeout},"
                f"ip,nw_src={ipv4_src},actions=drop"
            )
            subprocess.run(
                ["ovs-ofctl", "-O", "OpenFlow13", "add-flow", bridge, rule_spec],
                check=True,
                capture_output=True,
                text=True,
            )

    def enforce_drop_rule(
        self,
        *,
        tenant_id: str,
        ipv4_src: str,
        prob_attack: float,
        bridge: str | None = None,
        now_ts: float | None = None,
    ) -> tuple[MitigationRule, float, bool]:
        """Validate safety bounds and install a temporary drop rule for `ipv4_src`.

        Raises `SafetyGuardError` if `ipv4_src` is outside `10.10.1-3.0/24`.
        Returns `(rule, install_latency_ms, newly_installed)`.
        """
        validate_testbed_target(ipv4_src, tenant_id=tenant_id, verbose=False)
        target_bridge = bridge or self.ovs_bridge or "s1"
        t0 = time.perf_counter()
        now = time.time() if now_ts is None else float(now_ts)

        with self._lock:
            self.purge_expired(now_ts=now)
            key = (target_bridge, ipv4_src)
            if key in self.active_rules and self.active_rules[key].active:
                existing = self.active_rules[key]
                return existing, (time.perf_counter() - t0) * 1000.0, False

            rule = MitigationRule(
                tenant_id=tenant_id,
                bridge=target_bridge,
                ipv4_src=ipv4_src,
                priority=self.priority,
                idle_timeout=self.idle_timeout,
                hard_timeout=self.hard_timeout,
                installed_at=now,
                expires_at=now + float(self.hard_timeout),
                prob_attack=float(prob_attack),
                active=True,
            )
            self._install_openflow_drop(target_bridge, ipv4_src)
            self.active_rules[key] = rule
            self.rule_history.append(rule)
            self._sync_blocklist_file()

        install_ms = (time.perf_counter() - t0) * 1000.0
        return rule, float(install_ms), True

    def purge_expired(self, *, now_ts: float | None = None) -> list[MitigationRule]:
        """Mark and remove any rules whose `hard_timeout` expiry window has elapsed."""
        now = time.time() if now_ts is None else float(now_ts)
        expired: list[MitigationRule] = []
        changed = False
        for key, rule in list(self.active_rules.items()):
            if now >= rule.expires_at:
                rule.active = False
                expired.append(rule)
                del self.active_rules[key]
                changed = True
        if changed:
            self._sync_blocklist_file()
        return expired


class AsyncFlowInferenceWorker:
    """Background worker executing GRU threat detection off the OpenFlow event loop."""

    def __init__(
        self,
        model: torch.nn.Module,
        policy_manager: MitigationPolicyManager,
        *,
        attack_threshold: float = 0.5,
        device: torch.device | None = None,
        max_queue_size: int = 2048,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for AsyncFlowInferenceWorker.")
        self.device = device if device is not None else torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.policy_manager = policy_manager
        self.attack_threshold = float(attack_threshold)
        self._model_lock = threading.RLock()
        self._queue: queue.Queue[FlowClassificationJob | None] = queue.Queue(maxsize=max_queue_size)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.deployed_round: int = 0
        self.events: list[MitigationEvent] = []
        self.processed_count: int = 0

    def start(self) -> None:
        """Start background inference consumer thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="trustfed-infer", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Drain remaining queued flows and stop the background thread cleanly."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._queue.put(None)
        self._thread.join(timeout=timeout)
        self._thread = None

    def update_model_weights(self, flat_params: torch.Tensor, *, round_idx: int) -> float:
        """Atomically deploy newly verified global model weights; return `deploy_ms`."""
        t0 = time.perf_counter()
        with self._model_lock:
            self.model.set_flat_params(flat_params.detach().to(device=self.device).clone())
            self.model.eval()
            self.deployed_round = int(round_idx)
        return float((time.perf_counter() - t0) * 1000.0)

    def submit_flow(self, job: FlowClassificationJob) -> None:
        """Enqueue a flow record without blocking the SDN OpenFlow packet/stats thread."""
        self._queue.put_nowait(job)

    def wait_until_empty(self, timeout: float = 10.0) -> bool:
        """Block caller (e.g., harness/test) until all queued flows have been classified."""
        deadline = time.perf_counter() + float(timeout)
        while not self._queue.empty() and time.perf_counter() < deadline:
            time.sleep(0.005)
        self._queue.join()
        return True

    def classify_single_sync(self, feature_vec: np.ndarray) -> tuple[int, float, float]:
        """Run single-flow forward pass and return `(pred_label, prob_attack, infer_ms)`."""
        t0 = time.perf_counter()
        arr = np.asarray(feature_vec, dtype=np.float32).reshape(1, -1)
        x_t = torch.from_numpy(arr).to(self.device)
        with self._model_lock, torch.no_grad():
            logits = self.model(x_t)
            probs = torch.softmax(logits, dim=-1)
            prob_atk = float(probs[0, 1].item())
        pred = 1 if prob_atk >= self.attack_threshold else 0
        return pred, prob_atk, float((time.perf_counter() - t0) * 1000.0)

    def _process_job(self, job: FlowClassificationJob) -> None:
        pred, prob_atk, infer_ms = self.classify_single_sync(job.feature_vector)
        action = "allow"
        mitigation_ms = 0.0
        blocked_new = False

        if pred == 1:
            try:
                _, install_ms, blocked_new = self.policy_manager.enforce_drop_rule(
                    tenant_id=job.tenant_id,
                    ipv4_src=job.src_ip,
                    prob_attack=prob_atk,
                    bridge=job.bridge,
                )
                action = "drop_installed" if blocked_new else "drop_cached"
                ref_ts = (
                    job.attack_start_ts if job.attack_start_ts is not None else job.submitted_ts
                )
                mitigation_ms = max(install_ms, (time.perf_counter() - ref_ts) * 1000.0)
            except SafetyGuardError as exc:
                action = f"safety_blocked:{exc}"

        self.events.append(
            MitigationEvent(
                flow_id=job.flow_id,
                tenant_id=job.tenant_id,
                ipv4_src=job.src_ip,
                ground_truth_label=job.ground_truth_label,
                predicted_label=pred,
                prob_attack=float(prob_atk),
                inference_latency_ms=float(infer_ms),
                mitigation_latency_ms=float(mitigation_ms),
                action=action,
                timestamp=time.time(),
            )
        )
        self.processed_count += 1

    def _run_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue
            try:
                if item is None:
                    break
                self._process_job(item)
            finally:
                self._queue.task_done()

    def summarize_mitigation(self) -> dict[str, Any]:
        """Compute quantitative detection, mitigation latency, and false-block metrics."""
        if not self.events:
            return {
                "total_flows": 0,
                "true_positive_blocks": 0,
                "false_positive_blocks": 0,
                "false_block_rate": 0.0,
                "mean_inference_ms": 0.0,
                "p95_inference_ms": 0.0,
                "mean_mitigation_ms": 0.0,
            }
        infer_lat = [e.inference_latency_ms for e in self.events]
        mit_lat = [e.mitigation_latency_ms for e in self.events if e.predicted_label == 1]

        tp_atk = sum(1 for e in self.events if e.ground_truth_label == 1 and e.predicted_label == 1)
        total_atk = sum(1 for e in self.events if e.ground_truth_label == 1)
        fp_ben = sum(1 for e in self.events if e.ground_truth_label == 0 and e.predicted_label == 1)
        total_ben = sum(1 for e in self.events if e.ground_truth_label == 0)

        return {
            "total_flows": len(self.events),
            "total_attack_flows": total_atk,
            "total_benign_flows": total_ben,
            "true_positive_blocks": tp_atk,
            "attack_block_rate": float(tp_atk / max(1, total_atk)) if total_atk else 0.0,
            "false_positive_blocks": fp_ben,
            "false_block_rate": float(fp_ben / max(1, total_ben)) if total_ben else 0.0,
            "mean_inference_ms": float(np.mean(infer_lat)),
            "p95_inference_ms": float(np.percentile(infer_lat, 95)),
            "mean_mitigation_ms": float(np.mean(mit_lat)) if mit_lat else 0.0,
            "active_rules_count": len(self.policy_manager.active_rules),
        }


if TenantController13 is not None and set_ev_cls is not None and ofp_event is not None:

    class ModelDeployController13(TenantController13):  # type: ignore[valid-type,misc]
        """OS-Ken OpenFlow 1.3 controller wiring off-loop GRU weights & drop-rule policy."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.deploy_checkpoint_path = os.environ.get("TRUSTFED_DEPLOY_WEIGHTS", "")
            self.model_round_loaded = -1
            self.policy_manager = MitigationPolicyManager(
                blocklist_path=self.blocklist_path or None,
                priority=300,
                idle_timeout=60,
                hard_timeout=120,
            )

        @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)  # type: ignore[misc]
        def switch_features_handler(self, ev: Any) -> None:
            super().switch_features_handler(ev)
            self.policy_manager.attach_datapath(ev.msg.datapath)

else:

    class ModelDeployController13:  # type: ignore[no-redef]
        """Placeholder when `os_ken` is not installed (Windows analysis environment)."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ = (args, kwargs)
            raise RuntimeError("ModelDeployController13 requires Linux `os_ken` runtime.")
