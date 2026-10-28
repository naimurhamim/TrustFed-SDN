#!/usr/bin/env bash
# TrustFed-SDN Step 00: Linux SDN Testbed Smoke Test
# Verifies Mininet + Open vSwitch + OS-Ken (the maintained OpenStack fork of RYU
# compatible with Python 3.10+ on Ubuntu 22.04 / 24.04) in a separate controller
# virtualenv (.venv-ryu), runs an OpenFlow 1.3 3-host pingAll test, dumps installed
# OpenFlow 1.3 flows, and saves logs under docs/smoke_test/.
#
# Usage (inside Ubuntu 22.04 / 24.04 VM):
#   sudo bash scripts/smoke_test_linux.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename "${SCRIPT_DIR}")" == "scripts" ]]; then
    ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    ROOT_DIR="${SCRIPT_DIR}"
fi

OUT_DIR="${ROOT_DIR}/docs/smoke_test"
RYU_VENV="${ROOT_DIR}/.venv-ryu"
CTRL_PORT=6653
mkdir -p "${OUT_DIR}"

VERSIONS_LOG="${OUT_DIR}/versions.txt"
CONTROLLER_LOG="${OUT_DIR}/controller.log"
MININET_LOG="${OUT_DIR}/mininet_pingall.log"

if [[ "${EUID}" -ne 0 ]]; then
    echo "ERROR: Mininet and Open vSwitch require root privileges. Run with:" >&2
    echo "  sudo bash $0" >&2
    exit 1
fi

for cmd in mn ovs-vsctl ovs-ofctl python3; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "ERROR: Required command '${cmd}' not found." >&2
        echo "Install prerequisites on Ubuntu/Debian:" >&2
        echo "  sudo apt-get update && sudo apt-get install -y mininet openvswitch-switch python3-venv python3-pip" >&2
        exit 1
    fi
done

echo "[1/5] Checking Open vSwitch service and cleaning stale Mininet state..."
systemctl start openvswitch-switch 2>/dev/null || service openvswitch-switch start
mn -c >/dev/null 2>&1 || true
if command -v fuser >/dev/null 2>&1; then
    fuser -k "${CTRL_PORT}/tcp" >/dev/null 2>&1 || true
fi

echo "[2/5] Preparing isolated SDN controller virtualenv at ${RYU_VENV} (os-ken==3.0.1)..."
if [[ ! -x "${RYU_VENV}/bin/osken-manager" ]] || ! "${RYU_VENV}/bin/python" -c "import os_ken" >/dev/null 2>&1; then
    rm -rf "${RYU_VENV}"
    python3 -m venv "${RYU_VENV}"
    "${RYU_VENV}/bin/python" -m pip install --quiet --upgrade pip setuptools wheel
    "${RYU_VENV}/bin/python" -m pip install --quiet "os-ken==3.0.1" "eventlet<0.38"
fi

SWITCH_APP="${ROOT_DIR}/testbed/ryu_apps/smoke_switch_13.py"
if [[ ! -f "${SWITCH_APP}" ]]; then
    SWITCH_APP="${OUT_DIR}/smoke_switch_13.py"
    cat <<'PYEOF' > "${SWITCH_APP}"
"""Minimal OpenFlow 1.3 MAC-learning switch for TrustFed-SDN smoke test."""

from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from os_ken.lib.packet import ether_types, ethernet, packet
from os_ken.ofproto import ofproto_v1_3


class SmokeLearningSwitch13(app_manager.OSKenApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mac_to_port = {}

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=0, match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dpid = format(datapath.id, "d").zfill(16)
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][eth.src] = in_port

        out_port = self.mac_to_port[dpid].get(eth.dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=eth.dst, eth_src=eth.src)
            inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
            mod = parser.OFPFlowMod(datapath=datapath, priority=1, match=match, instructions=inst)
            datapath.send_msg(mod)

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data,
        )
        datapath.send_msg(out)
PYEOF
fi

echo "[3/5] Recording system and SDN tool versions to ${VERSIONS_LOG}..."
{
    echo "Timestamp (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Kernel: $(uname -a)"
    if [[ -f /etc/os-release ]]; then
        grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release || true
    fi
    echo "Mininet: $(mn --version 2>&1)"
    echo "Open vSwitch: $(ovs-vsctl --version | head -n 1)"
    echo "Controller Python: $("${RYU_VENV}/bin/python" --version 2>&1)"
    echo "Controller Stack: os-ken==3.0.1 (OpenStack maintained RYU fork)"
    "${RYU_VENV}/bin/python" -m pip freeze
} > "${VERSIONS_LOG}"

CTRL_PID=""
fix_permissions() {
    if [[ -n "${SUDO_USER:-}" ]]; then
        chown -R "${SUDO_USER}:${SUDO_USER}" "$(dirname "${OUT_DIR}")" 2>/dev/null || true
    fi
}
cleanup() {
    if [[ -n "${CTRL_PID}" ]] && kill -0 "${CTRL_PID}" 2>/dev/null; then
        kill "${CTRL_PID}" 2>/dev/null || true
        wait "${CTRL_PID}" 2>/dev/null || true
    fi
    mn -c >/dev/null 2>&1 || true
    fix_permissions
}
trap cleanup EXIT

echo "[4/5] Starting OS-Ken (RYU fork) OpenFlow 1.3 learning switch on 127.0.0.1:${CTRL_PORT}..."
"${RYU_VENV}/bin/osken-manager" --ofp-tcp-listen-port "${CTRL_PORT}" "${SWITCH_APP}" >"${CONTROLLER_LOG}" 2>&1 &
CTRL_PID=$!

LISTENING=0
for _ in $(seq 1 30); do
    if ! kill -0 "${CTRL_PID}" 2>/dev/null; then
        echo "ERROR: Controller exited prematurely. Log output:" >&2
        cat "${CONTROLLER_LOG}" >&2
        exit 1
    fi
    if "${RYU_VENV}/bin/python" -c "import socket; s=socket.create_connection(('127.0.0.1', ${CTRL_PORT}), 0.5); s.close()" 2>/dev/null; then
        LISTENING=1
        break
    fi
    sleep 0.5
done

if [[ "${LISTENING}" -ne 1 ]]; then
    echo "ERROR: Controller did not open 127.0.0.1:${CTRL_PORT} within 15s. Log output:" >&2
    cat "${CONTROLLER_LOG}" >&2
    exit 1
fi
echo "      Controller is listening on 127.0.0.1:${CTRL_PORT} (PID ${CTRL_PID})."

echo "[5/5] Running Mininet 3-host OVS OpenFlow13 topology + pingAll verification..."
set +e
timeout 45s python3 - "${CTRL_PORT}" 2>&1 <<'MNEOF' | tee "${MININET_LOG}"
import functools
import subprocess
import sys
import time
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.topo import SingleSwitchTopo

port = int(sys.argv[1])
setLogLevel("info")
ovs13 = functools.partial(OVSSwitch, protocols="OpenFlow13")
net = Mininet(
    topo=SingleSwitchTopo(k=3),
    switch=ovs13,
    controller=lambda name: RemoteController(name, ip="127.0.0.1", port=port),
    waitConnected=True,
)
net.start()
try:
    # Wait up to 8s for controller to install OpenFlow 1.3 table-miss rule
    for _ in range(16):
        flows = subprocess.check_output(
            ["ovs-ofctl", "-O", "OpenFlow13", "dump-flows", "s1"], text=True
        )
        if "CONTROLLER" in flows:
            break
        time.sleep(0.5)
    loss_pct = net.pingAll(timeout="1")
    print("\n--- OpenFlow 1.3 Flow Table After pingAll (s1) ---")
    print(subprocess.check_output(["ovs-ofctl", "-O", "OpenFlow13", "dump-flows", "s1"], text=True))
    print(f"FINAL_LOSS_PCT={loss_pct}")
finally:
    net.stop()
sys.exit(0 if loss_pct == 0.0 else 1)
MNEOF
MN_RC=${PIPESTATUS[0]}
set -e

if [[ "${MN_RC}" -eq 0 ]] && grep -q "FINAL_LOSS_PCT=0.0" "${MININET_LOG}"; then
    echo ""
    echo "================ SMOKE TEST SUMMARY ================"
    head -n 8 "${VERSIONS_LOG}"
    grep -E "Results:|FINAL_LOSS_PCT=" "${MININET_LOG}" || true
    echo "Logs saved in: ${OUT_DIR}"
    echo "SUCCESS: Mininet + OVS + OS-Ken (RYU fork) OpenFlow 1.3 pingAll passed (0% dropped)!"
    echo "===================================================="
else
    echo "FAILURE (exit ${MN_RC}): pingAll did not achieve 0% loss." >&2
    echo "--- Controller Log (${CONTROLLER_LOG}) ---" >&2
    tail -n 30 "${CONTROLLER_LOG}" >&2 || true
    exit 1
fi
