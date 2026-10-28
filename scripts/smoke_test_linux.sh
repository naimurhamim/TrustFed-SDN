#!/usr/bin/env bash
# TrustFed-SDN Step 00 / Step 07 Entry Gate: Linux Testbed Smoke Test
# Verifies Mininet + Open vSwitch + RYU (or OS-Ken fallback on Python 3.11+)
# using a separate controller virtualenv (.venv-ryu) and saves logs to docs/smoke_test/.
#
# Usage (inside Ubuntu 22.04 / 24.04 VM):
#   sudo bash scripts/smoke_test_linux.sh
#
# Note: Does NOT install system packages automatically. Install prerequisites first:
#   sudo apt-get update && sudo apt-get install -y mininet openvswitch-switch python3-venv python3-pip netcat-openbsd

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/docs/smoke_test"
RYU_VENV="${ROOT_DIR}/.venv-ryu"
CTRL_PORT=6653
mkdir -p "${OUT_DIR}"

VERSIONS_LOG="${OUT_DIR}/versions.txt"
CONTROLLER_LOG="${OUT_DIR}/controller.log"
MININET_LOG="${OUT_DIR}/mininet_pingall.log"

if [[ "${EUID}" -ne 0 ]]; then
    echo "ERROR: Mininet and Open vSwitch require root privileges. Run with: sudo bash scripts/smoke_test_linux.sh" >&2
    exit 1
fi

for cmd in mn ovs-vsctl python3; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "ERROR: Required command '${cmd}' not found." >&2
        echo "Install prerequisites on Ubuntu/Debian:" >&2
        echo "  sudo apt-get update && sudo apt-get install -y mininet openvswitch-switch python3-venv python3-pip" >&2
        exit 1
    fi
done

echo "[1/5] Checking Open vSwitch service..."
systemctl start openvswitch-switch 2>/dev/null || service openvswitch-switch start

echo "[2/5] Preparing isolated SDN controller virtualenv at ${RYU_VENV}..."
PY_BIN="python3"
for candidate in python3.9 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
        PY_BIN="${candidate}"
        break
    fi
done

if [[ ! -d "${RYU_VENV}" ]]; then
    "${PY_BIN}" -m venv "${RYU_VENV}"
fi

PY_MINOR="$("${RYU_VENV}/bin/python" -c 'import sys; print(sys.version_info.minor)')"
CONTROLLER_FLAVOR="ryu"

if [[ "${PY_MINOR}" -le 10 ]]; then
    "${RYU_VENV}/bin/pip" install --quiet --upgrade "pip<25" "setuptools<68" wheel
    if ! "${RYU_VENV}/bin/pip" install --quiet "ryu==4.34" "eventlet==0.30.2" "dnspython<2.3"; then
        CONTROLLER_FLAVOR="os-ken"
        "${RYU_VENV}/bin/pip" install --quiet --upgrade pip setuptools wheel os-ken
    fi
else
    # RYU 4.34 + old eventlet breaks on Python >= 3.11 (removed ssl.wrap_socket / distutils).
    # OS-Ken is the maintained OpenStack fork of RYU with the same OpenFlow 1.3 API.
    CONTROLLER_FLAVOR="os-ken"
    "${RYU_VENV}/bin/pip" install --quiet --upgrade pip setuptools wheel os-ken
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
    echo "Controller Stack: ${CONTROLLER_FLAVOR}"
    "${RYU_VENV}/bin/pip" freeze
} > "${VERSIONS_LOG}"

CTRL_PID=""
cleanup() {
    if [[ -n "${CTRL_PID}" ]] && kill -0 "${CTRL_PID}" 2>/dev/null; then
        kill "${CTRL_PID}" 2>/dev/null || true
        wait "${CTRL_PID}" 2>/dev/null || true
    fi
    mn -c >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[4/5] Starting ${CONTROLLER_FLAVOR} OpenFlow 1.3 learning switch on 127.0.0.1:${CTRL_PORT}..."
mn -c >/dev/null 2>&1 || true
if [[ "${CONTROLLER_FLAVOR}" == "ryu" ]]; then
    "${RYU_VENV}/bin/ryu-manager" --ofp-tcp-listen-port "${CTRL_PORT}" ryu.app.simple_switch_13 >"${CONTROLLER_LOG}" 2>&1 &
else
    "${RYU_VENV}/bin/osken-manager" --ofp-tcp-listen-port "${CTRL_PORT}" os_ken.app.simple_switch_13 >"${CONTROLLER_LOG}" 2>&1 &
fi
CTRL_PID=$!

for _ in $(seq 1 20); do
    if "${RYU_VENV}/bin/python" -c "import socket; s=socket.create_connection(('127.0.0.1', ${CTRL_PORT}), 0.5); s.close()" 2>/dev/null; then
        break
    fi
    sleep 0.5
done

echo "[5/5] Running Mininet 3-host OVS OpenFlow13 topology with pingall..."
mn --topo single,3 \
   --switch ovsk,protocols=OpenFlow13 \
   --controller "remote,ip=127.0.0.1,port=${CTRL_PORT}" \
   --test pingall >"${MININET_LOG}" 2>&1

if grep -q "0% dropped" "${MININET_LOG}"; then
    echo "SUCCESS: Mininet + OVS + ${CONTROLLER_FLAVOR} pingall passed (0% dropped)."
    echo "Logs written to: ${OUT_DIR}"
else
    echo "FAILURE: pingall did not achieve 0% packet drop. Check ${MININET_LOG} and ${CONTROLLER_LOG}." >&2
    exit 1
fi

