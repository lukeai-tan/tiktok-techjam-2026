#!/usr/bin/env python3
"""Capture reproducibility metadata without reading credentials or `.env`."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    "transformer_opt/submission.py",
    "transformer_opt",
    "benchmarks/run_matrix.py",
    "benchmarks/profile_cases.py",
    "benchmarks/official_shapes.json",
    "benchmarks/reference/manifest.json",
    "tools/capture_environment.py",
    "tools/triton-cc",
    "scripts/run-wsl.ps1",
)
IMPLEMENTATION_FINGERPRINT_SCHEMA = 2


def _run(command: list[str], cwd: Path = REPO_ROOT) -> Optional[str]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def display_path(path: str | Path) -> str:
    """Render repository paths portably and external paths without home details."""
    candidate = Path(path).resolve()
    try:
        return candidate.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return candidate.name


def cpu_name() -> Optional[str]:
    """Return a useful CPU model without collecting user or host identity."""
    if platform.system() == "Windows":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if isinstance(value, str) and value.strip():
                return value.strip()
        except (OSError, ImportError):
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            for line in cpuinfo.read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.partition(":")[2].strip() or None
        except OSError:
            pass
    return platform.processor() or None


def nvidia_driver_version() -> Optional[str | int]:
    """Prefer NVIDIA's reported display-driver version when available."""
    reported = _run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    if reported:
        return reported.splitlines()[0].strip()
    driver_getter = getattr(torch.cuda, "driver_version", None)
    if callable(driver_getter):
        try:
            return driver_getter()
        except RuntimeError:
            pass
    return None


def implementation_fingerprint() -> tuple[str, list[str]]:
    """Hash implementation content independent of checkout line endings.

    Every fingerprinted path is text. Git stores these files with LF line
    endings, but a Windows checkout may materialize CRLF bytes. Canonicalizing
    CRLF here keeps the same implementation fingerprint across Windows, WSL,
    and Linux while still detecting content changes in an uncommitted worktree.
    """
    files: list[Path] = []
    for relative in IMPLEMENTATION_PATHS:
        candidate = REPO_ROOT / relative
        if candidate.is_dir():
            files.extend(path for path in candidate.rglob("*.py") if path.is_file())
        elif candidate.is_file():
            files.append(candidate)
    digest = hashlib.sha256()
    relative_paths: list[str] = []
    for path in sorted(set(files), key=lambda item: item.relative_to(REPO_ROOT).as_posix()):
        relative = path.relative_to(REPO_ROOT).as_posix()
        relative_paths.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest(), relative_paths


def capture_environment(command: Optional[list[str]] = None) -> dict[str, Any]:
    """Return code, OS, Python, framework, GPU, and disk metadata."""
    commit = _run(["git", "rev-parse", "HEAD"])
    git_status = _run(["git", "status", "--short"])
    implementation_sha256, implementation_paths = implementation_fingerprint()
    try:
        import triton

        triton_version: Optional[str] = triton.__version__
    except (ImportError, ModuleNotFoundError):
        triton_version = None

    disk = shutil.disk_usage(REPO_ROOT)
    result: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command if command is not None else sys.argv,
        "git": {
            "commit": commit,
            "dirty": bool(git_status),
            "changed_path_count": len(git_status.splitlines()) if git_status else 0,
            "implementation_fingerprint_schema": IMPLEMENTATION_FINGERPRINT_SCHEMA,
            "implementation_sha256": implementation_sha256,
            "implementation_paths": implementation_paths,
        },
        "python": {
            "version": platform.python_version(),
            "executable": display_path(sys.executable),
            "implementation": platform.python_implementation(),
        },
        "os": {
            "platform": platform.platform(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "wsl_distribution": os.environ.get("WSL_DISTRO_NAME"),
        },
        "cpu": {
            "name": cpu_name(),
            "logical_count": os.cpu_count(),
        },
        "disk": {
            "path": display_path(REPO_ROOT),
            "total_bytes": disk.total,
            "free_bytes": disk.free,
        },
        "framework": {
            "torch": torch.__version__,
            "triton": triton_version,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        },
        "gpu": None,
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        result["gpu"] = {
            "index": device,
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "driver_version": nvidia_driver_version(),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = capture_environment()
    rendered = json.dumps(payload, indent=2)
    if args.out is None:
        print(rendered)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
