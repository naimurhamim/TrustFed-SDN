"""Hardware, OS, GPU, and virtualization probe for TrustFed-SDN.

Prints a structured JSON or Markdown report of the current machine's
CPU, RAM, disk space, GPU, operating system, Python runtime, and
virtualization/SDN tooling availability.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil


def _run_cmd(args: list[str], timeout: float = 8.0) -> str | None:
    """Run a subprocess command safely and return stripped stdout or None."""
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return output if output else None


def get_cpu_model() -> str:
    """Return a human-readable CPU brand string across Windows and Linux."""
    system = platform.system()
    if system == "Windows":
        out = _run_cmd(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name",
            ]
        )
        if out:
            return out
    elif system == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    parts = line.split(":", maxsplit=1)
                    if len(parts) == 2:
                        return parts[1].strip()
    fallback = platform.processor().strip()
    return fallback if fallback else "unknown"


def get_gpu_info() -> list[dict[str, str]]:
    """Query NVIDIA GPUs via nvidia-smi if available."""
    gpus: list[dict[str, str]] = []
    if shutil.which("nvidia-smi"):
        out = _run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        )
        if out:
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append(
                        {
                            "name": parts[0],
                            "vram_total": parts[1],
                            "driver_version": parts[2],
                        }
                    )
    return gpus


def get_disk_info(workspace: Path) -> list[dict[str, Any]]:
    """Collect total and free space for mounted fixed disks and the workspace drive."""
    seen_mounts: set[str] = set()
    disks: list[dict[str, Any]] = []

    for part in psutil.disk_partitions(all=False):
        if os.name == "nt" and ("cdrom" in part.opts or not part.fstype):
            continue
        mount = part.mountpoint
        if mount in seen_mounts:
            continue
        try:
            usage = psutil.disk_usage(mount)
        except OSError:
            continue
        seen_mounts.add(mount)
        disks.append(
            {
                "mount": mount,
                "fstype": part.fstype,
                "total_gb": round(usage.total / (1024**3), 1),
                "free_gb": round(usage.free / (1024**3), 1),
            }
        )

    if not disks:
        usage = psutil.disk_usage(str(workspace))
        disks.append(
            {
                "mount": str(workspace.anchor or workspace),
                "fstype": "unknown",
                "total_gb": round(usage.total / (1024**3), 1),
                "free_gb": round(usage.free / (1024**3), 1),
            }
        )
    return disks


def get_virtualization_info() -> dict[str, Any]:
    """Detect hypervisor, VMware, WSL, KVM, and SDN binaries."""
    system = platform.system()
    info: dict[str, Any] = {"host_system": system}

    if system == "Windows":
        hv = _run_cmd(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).HypervisorPresent",
            ]
        )
        info["windows_hypervisor_present"] = hv == "True" if hv is not None else None
        vmware_Candidates = [
            Path(r"C:\Program Files (x86)\VMware\VMware Workstation\vmrun.exe"),
            Path(r"C:\Program Files\VMware\VMware Workstation\vmrun.exe"),
        ]
        vmrun_path = next((str(p) for p in vmware_Candidates if p.exists()), shutil.which("vmrun"))
        info["vmware_vmrun"] = vmrun_path
        info["wsl_available"] = shutil.which("wsl") is not None
    elif system == "Linux":
        info["kvm_device_present"] = Path("/dev/kvm").exists()
        info["mininet_mn"] = shutil.which("mn")
        info["ovs_vsctl"] = shutil.which("ovs-vsctl")
        info["ryu_manager"] = shutil.which("ryu-manager")

    return info


def get_python_packages() -> dict[str, str]:
    """Return versions of core ML/scientific libraries installed in this interpreter."""
    from importlib.metadata import PackageNotFoundError, version

    packages = [
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "scipy",
        "torch",
        "PyYAML",
        "matplotlib",
        "psutil",
        "cryptography",
        "pytest",
        "ruff",
        "black",
    ]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def collect_probe(workspace: Path | None = None) -> dict[str, Any]:
    """Collect the complete hardware and software environment report."""
    root = workspace or Path(__file__).resolve().parents[1]
    vm = psutil.virtual_memory()
    return {
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "cpu": {
            "model": get_cpu_model(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_threads": psutil.cpu_count(logical=True),
        },
        "ram": {
            "total_gb": round(vm.total / (1024**3), 2),
            "available_gb": round(vm.available / (1024**3), 2),
            "percent_used": vm.percent,
        },
        "disks": get_disk_info(root),
        "gpus": get_gpu_info(),
        "virtualization": get_virtualization_info(),
        "packages": get_python_packages(),
    }


def format_markdown(report: dict[str, Any]) -> str:
    """Format a probe dictionary into clean Markdown."""
    lines = [
        "## Hardware & Runtime Probe Output",
        "",
        f"- **Probed at (UTC):** `{report['timestamp_utc']}`",
        (
            f"- **Operating System:** {report['os']['system']} "
            f"{report['os']['release']} (version `{report['os']['version']}`, "
            f"`{report['os']['machine']}`)"
        ),
        (
            f"- **Python Runtime:** `{report['python']['version']}` "
            f"(`{report['python']['executable']}`)"
        ),
        (
            f"- **CPU:** {report['cpu']['model']} — "
            f"{report['cpu']['physical_cores']} physical cores / "
            f"{report['cpu']['logical_threads']} logical threads"
        ),
        (
            f"- **RAM:** {report['ram']['total_gb']} GB total "
            f"({report['ram']['available_gb']} GB available at probe time, "
            f"{report['ram']['percent_used']}% used)"
        ),
    ]

    for disk in report["disks"]:
        lines.append(
            f"- **Disk `{disk['mount']}` ({disk['fstype']}):** "
            f"{disk['free_gb']} GB free of {disk['total_gb']} GB"
        )

    if report["gpus"]:
        for gpu in report["gpus"]:
            lines.append(
                f"- **GPU:** {gpu['name']} ({gpu['vram_total']}, driver `{gpu['driver_version']}`)"
            )
    else:
        lines.append("- **GPU (NVIDIA via `nvidia-smi`):** none detected")

    virt = report["virtualization"]
    lines.append(f"- **Virtualization & SDN checks:** `{json.dumps(virt)}`")
    lines.append("")
    lines.append("### Installed Python Package Versions")
    lines.append("")
    lines.append("| Package | Version |")
    lines.append("|---|---|")
    for pkg, ver in report["packages"].items():
        lines.append(f"| `{pkg}` | `{ver}` |")

    return "\n".join(lines)


def main() -> int:
    """CLI entry point for hardware probe."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Probe host hardware and Python environment.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the probe output.",
    )
    args = parser.parse_args()

    report = collect_probe()
    rendered = json.dumps(report, indent=2) if args.json else format_markdown(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
