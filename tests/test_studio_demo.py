"""Tests for the Interactive Visual Studio engine and HTTP server (Option A & B)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

from demo.engine import TrustFedStudioEngine
from demo.studio_server import make_studio_server


def _post_json(url: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return int(resp.status), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return int(exc.code), json.loads(exc.read().decode("utf-8"))


def test_studio_sequential_steps_and_labs() -> None:
    """Verify sequential lock (Step 1..5), real PCAP inspection, poisoning & tamper labs."""
    eng = TrustFedStudioEngine()
    srv = make_studio_server(eng, host="127.0.0.1", port=0)
    port = int(srv.server_address[1])
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"

    try:
        with urllib.request.urlopen(f"{base}/api/init", timeout=10) as resp:
            init_data = json.loads(resp.read().decode("utf-8"))
        assert init_data["completed_step"] == 0
        assert len(init_data["pcap_samples"]) >= 6

        # Out-of-order Step 2 must fail before Step 1 completes
        code_bad, bad_res = _post_json(f"{base}/api/step", {"step": 2})
        assert code_bad == 400
        assert bad_res["ok"] is False

        # Sequential execution Steps 1 -> 5
        for step_num in (1, 2, 3, 4, 5):
            code, res = _post_json(f"{base}/api/step", {"step": step_num})
            assert code == 200
            assert res["ok"] is True
            assert res["completed_step"] == step_num

        # Step 5 mitigation verified 100% attacker block + 0% benign loss across s1..s3
        s5 = eng.step_results[5]
        assert s5["attacker_loss_pct"] == 100.0
        assert s5["benign_loss_pct"] == 0.0
        assert {f["switch"] for f in s5["flows"]} == {"s1", "s2", "s3"}

        # Reset returns to Step 0
        _, rst = _post_json(f"{base}/api/reset", {})
        assert rst["completed_step"] == 0

        # Option B real PCAP inspection (syn_flood vs benign)
        _, atk_insp = _post_json(
            f"{base}/api/inspect-sample",
            {"pcap_name": "tenant_2_healthcare__syn_flood.pcap", "threshold": 0.50},
        )
        assert atk_insp["ok"] is True
        assert atk_insp["result"]["total_flows"] >= 1
        assert len(atk_insp["result"]["top_features"]) == 8

        # Option C Live VM status, safe sequential steps, and artifact file inspection
        with urllib.request.urlopen(f"{base}/api/vm/status", timeout=10) as resp:
            vm_status = json.loads(resp.read().decode("utf-8"))
        assert "host_ram_available_gb" in vm_status
        assert "vm_user" in vm_status
        assert "vmx_path" in vm_status
        assert "vm_pass" not in vm_status
        assert len(vm_status["crash_guards"]) >= 5
        assert len(vm_status["files"]) >= 3

        code_vm_bad, _ = _post_json(f"{base}/api/vm/step", {"step": 2, "safe_mode": True})
        assert code_vm_bad == 400

        code_setup, res_setup = _post_json(
            f"{base}/api/vm/setup",
            {"safe_mode": True, "vm_user": "ubuntu", "vm_pass": "ubuntu"},
        )
        assert code_setup == 200
        assert res_setup["ok"] is True
        assert res_setup["result"]["status"] == "SETUP_COMPLETE"
        assert "SETUP COMPLETE" in res_setup["result"]["output"]

        for vm_step in (1, 2, 3):
            code_vm, res_vm = _post_json(
                f"{base}/api/vm/step",
                {
                    "step": vm_step,
                    "safe_mode": True,
                    "vm_user": "collaborator",
                    "vm_pass": "secret-pass",
                },
            )
            assert code_vm == 200
            assert res_vm["ok"] is True
            assert res_vm["vm_completed_step"] == vm_step
            assert res_vm["result"]["mode"] == "SAFE_GUARD_REPLAY"
            assert res_vm["result"]["safety"]["vm_user"] == "collaborator"

        _, insp_json = _post_json(
            f"{base}/api/vm/inspect-file",
            {"path": "results/08_live/vm_live_mitigation.json"},
        )
        assert insp_json["ok"] is True
        assert insp_json["result"]["kind"] == "json"

        _, insp_pcap = _post_json(
            f"{base}/api/vm/inspect-file",
            {"path": "data/live/raw_pcaps/tenant_1_finance__dos_http.pcap"},
        )
        assert insp_pcap["ok"] is True
        assert insp_pcap["result"]["kind"] == "pcap"
        assert "add-flow s1" in insp_pcap["result"]["ai"]["ovs_rule_preview"]

        _, vm_rst = _post_json(f"{base}/api/vm/reset", {})
        assert vm_rst["vm_completed_step"] == 0

        # Path traversal & shell injection safety checks
        code_trav, _ = _post_json(f"{base}/api/inspect-sample", {"pcap_name": "../../README.md"})
        assert code_trav == 400

        code_bad_user, bad_user_res = _post_json(
            f"{base}/api/vm/step",
            {"step": 1, "safe_mode": True, "vm_user": "bad;user"},
        )
        assert code_bad_user == 400
        assert bad_user_res["ok"] is False

        # Poisoning & tamper labs
        _, pois = _post_json(
            f"{base}/api/lab/poison", {"malicious_pct": 33, "attack_type": "scaling"}
        )
        assert pois["ok"] is True
        assert len(pois["result"]["rejected"]) >= 1

        _, ok_cr = _post_json(f"{base}/api/lab/tamper", {"tamper_mode": "none"})
        assert ok_cr["result"]["verified"] is True

        _, bad_cr = _post_json(f"{base}/api/lab/tamper", {"tamper_mode": "lsb_plus_1"})
        assert bad_cr["result"]["verified"] is False
    finally:
        srv.shutdown()
        srv.server_close()
