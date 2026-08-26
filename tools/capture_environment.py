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
    "torch_transformer_benchmark.py",
    "transformer_opt",
    "benchmarks/run_matrix.py",
    "benchmarks/profile_cases.py",
    "benchmarks/official_shapes.json",
    "benchmarks/reference/manifest.json",
    "tools/capture_environment.py",
    "tools/triton-cc",
    "scripts/run-wsl.ps1",
)


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


def implementation_fingerprint() -> tuple[str, list[str]]:
    """Hash implementation bytes even when the Git worktree is uncommitted."""
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
        digest.update(path.read_bytes())
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
            "implementation_sha256": implementation_sha256,
            "implementation_paths": implementation_paths,
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "os": {
            "platform": platform.platform(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "wsl_distribution": os.environ.get("WSL_DISTRO_NAME"),
        },
        "disk": {
            "path": str(REPO_ROOT),
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
        driver_version = None
        driver_getter = getattr(torch.cuda, "driver_version", None)
        if callable(driver_getter):
            try:
                driver_version = driver_getter()
            except RuntimeError:
                driver_version = None
        result["gpu"] = {
            "index": device,
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "driver_version": driver_version,
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
