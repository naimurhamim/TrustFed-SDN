#!/usr/bin/env bash
# TrustFed-SDN Step 08: Live OVS Mitigation & Control-Plane Verification (Linux VM root)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RYU_VENV="${ROOT_DIR}/.venv-ryu"
RES_DIR="${ROOT_DIR}/results/08_live"
CTRL_APP="${ROOT_DIR}/testbed/ryu_apps/model_deploy.py"
DRIVER="${ROOT_DIR}/scripts/run_step08_vm_driver.py"
PID_FILE="${RES_DIR}/controller_pids.txt"

if [[ "${EUID}" -ne 0 ]]; then
    echo "ERROR: Run with sudo: sudo bash $0" >&2
    exit 1
fi

mkdir -p "${RES_DIR}"
rm -f "${PID_FILE}"

echo "[1/4] Cleaning stale Mininet/OVS state and starting Open vSwitch..."
systemctl start openvswitch-switch 2>/dev/null || service openvswitch-switch start
mn -c >/dev/null 2>&1 || true
for br in s0 s1 s2 s3; do
    ovs-vsctl --if-exists del-br "${br}" 2>/dev/null || true
done
for port in 6653 6654 6655 6656; do
    if command -v fuser >/dev/null 2>&1; then
        fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    fi
done

declare -a CTRL_PIDS=()
cleanup() {
    for pid in "${CTRL_PIDS[@]:-}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
        fi
    done
    mn -c >/dev/null 2>&1 || true
    if [[ -n "${SUDO_USER:-}" ]]; then
        chown -R "${SUDO_USER}:${SUDO_USER}" "${RES_DIR}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[2/4] Starting 3 isolated OS-Ken OpenFlow 1.3 controllers (6653, 6654, 6655)..."
PORTS=(6653 6654 6655)
for idx in 0 1 2; do
    t_num=$((idx + 1))
    port="${PORTS[$idx]}"
    PYTHONPATH="${ROOT_DIR}" \
    TENANT_FLOW_STATS_LOG="${RES_DIR}/of_stats_t${t_num}.jsonl" \
        "${RYU_VENV}/bin/osken-manager" \
        --ofp-tcp-listen-port "${port}" \
        "${CTRL_APP}" >"${RES_DIR}/controller_t${t_num}.log" 2>&1 &
    pid=$!
    CTRL_PIDS+=("${pid}")
    echo "c${t_num}:${pid}" >> "${PID_FILE}"

    for _ in $(seq 1 30); do
        if "${RYU_VENV}/bin/python" -c "import socket; s=socket.create_connection(('127.0.0.1', ${port}), 0.5); s.close()" 2>/dev/null; then
            break
        fi
        sleep 0.3
    done
    echo "      Tenant ${t_num} controller ready on port ${port} (PID ${pid})"
done

echo "[3/4] Running Step 08 live Mininet mitigation & responsiveness driver..."
PYTHONPATH="${ROOT_DIR}" "${RYU_VENV}/bin/python" "${DRIVER}"

echo "[4/4] Live VM verification complete -> ${RES_DIR}/vm_live_mitigation.json"
