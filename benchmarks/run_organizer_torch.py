#!/usr/bin/env python3
"""Run the untouched organizer PyTorch harness with the submitted model.

The downloaded benchmark remains byte-for-byte unchanged under ``benchmarks/``.
This runner replaces only its documented ``UserOptimizedTransformer`` extension
point with the repository implementation, then delegates argument parsing,
correctness checks, and timing to the organizer module.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Optional, Sequence, TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
ORGANIZER_TORCH_PATH = REPO_ROOT / "benchmarks" / "torch_transformer_benchmark.py"
ORGANIZER_MANIFEST_PATH = (
    REPO_ROOT / "benchmarks" / "reference" / "organizer_downloads.json"
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _Tee(io.TextIOBase):
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, value: str) -> int:
        for stream in self._streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def load_organizer_benchmark() -> ModuleType:
    """Load the supplied file without putting its directory on ``sys.path``."""
    module_name = "benchmarks.torch_transformer_benchmark"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, ORGANIZER_TORCH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load organizer benchmark: {ORGANIZER_TORCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def install_submission(module: ModuleType, instance_holder: Optional[dict] = None) -> None:
    """Install the optimized class at the organizer's documented extension point."""
    from transformer_opt.submission import UserOptimizedTransformer

    if instance_holder is None:
        module.UserOptimizedTransformer = UserOptimizedTransformer
        return

    class TrackedUserOptimizedTransformer(UserOptimizedTransformer):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            instance_holder["model"] = self

    TrackedUserOptimizedTransformer.__name__ = "UserOptimizedTransformer"
    TrackedUserOptimizedTransformer.__qualname__ = "UserOptimizedTransformer"
    module.UserOptimizedTransformer = TrackedUserOptimizedTransformer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(path: Path) -> str:
    """Hash repository text independently of checkout line endings."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_organizer_download() -> str:
    """Fail before execution unless the supplied PyTorch bytes are frozen."""
    manifest = json.loads(ORGANIZER_MANIFEST_PATH.read_text(encoding="utf-8"))
    artifact = next(
        item for item in manifest["artifacts"] if item["framework"] == "pytorch"
    )
    actual = _sha256(ORGANIZER_TORCH_PATH)
    if actual != artifact["sha256"]:
        raise RuntimeError(
            "organizer PyTorch benchmark checksum mismatch: "
            f"expected {artifact['sha256']}, got {actual}"
        )
    return actual


def _parse_stdout(stdout: str) -> dict:
    parsed: dict[str, object] = {}
    accuracy = re.search(
        r"summary: (PASS|FAIL) \| max_abs=([^ ]+) \| max_rel=([^ ]+) \| "
        r"failed=(\d+)/(\d+)",
        stdout,
    )
    if accuracy:
        parsed["accuracy"] = {
            "status": accuracy.group(1),
            "max_abs_error": float(accuracy.group(2)),
            "max_relative_error": float(accuracy.group(3)),
            "failed_elements": int(accuracy.group(4)),
            "total_elements": int(accuracy.group(5)),
        }

    for label in ("baseline", "optimized"):
        timing = re.search(
            rf"{label}\s*: median=([0-9.eE+-]+) ms \| mean=([0-9.eE+-]+) ms "
            rf"\| p90=([0-9.eE+-]+) ms \| min=([0-9.eE+-]+) ms \| "
            rf"throughput=([0-9.eE+-]+) token/s",
            stdout,
        )
        if timing:
            parsed[label] = {
                "median_ms": float(timing.group(1)),
                "mean_ms": float(timing.group(2)),
                "p90_ms": float(timing.group(3)),
                "min_ms": float(timing.group(4)),
                "tokens_per_second": float(timing.group(5)),
            }

    speedup = re.search(r"speedup\s*: ([0-9.eE+-]+)x based on median latency", stdout)
    if speedup:
        parsed["speedup_median"] = float(speedup.group(1))
    return parsed


def _wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--evidence-out",
        type=Path,
        help="optional JSON record; every other argument is passed to the organizer",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    supplied_argv = list(sys.argv[1:] if argv is None else argv)
    wrapper_args, organizer_args = _wrapper_parser().parse_known_args(supplied_argv)
    organizer_sha256 = verify_organizer_download()
    module = load_organizer_benchmark()
    holder: dict[str, object] = {}
    install_submission(module, holder if wrapper_args.evidence_out else None)

    original_argv = sys.argv
    captured = io.StringIO()
    exit_code = 1
    try:
        sys.argv = [str(ORGANIZER_TORCH_PATH), *organizer_args]
        if wrapper_args.evidence_out is None:
            return int(module.main())
        with contextlib.redirect_stdout(_Tee(sys.stdout, captured)):
            exit_code = int(module.main())
    finally:
        sys.argv = original_argv

    from tools.capture_environment import capture_environment, display_path

    model = holder.get("model")
    parsed = _parse_stdout(captured.getvalue())
    if exit_code == 0 and (
        parsed.get("accuracy", {}).get("status") != "PASS"
        or "baseline" not in parsed
        or "optimized" not in parsed
        or "speedup_median" not in parsed
    ):
        raise RuntimeError("successful organizer run did not emit complete evidence")

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "framework": "pytorch",
        "organizer_script": {
            "path": display_path(ORGANIZER_TORCH_PATH),
            "sha256": organizer_sha256,
            "manifest": display_path(ORGANIZER_MANIFEST_PATH),
            "modified": False,
        },
        "submission": {
            "class": "transformer_opt.submission.UserOptimizedTransformer",
            "injection_point": "UserOptimizedTransformer",
            "runner_path": display_path(RUNNER_PATH),
            "runner_sha256": _text_sha256(RUNNER_PATH),
        },
        "organizer_arguments": organizer_args,
        "exit_code": exit_code,
        "parsed": parsed,
        "attention_backend_counts": getattr(model, "attention_backend_counts", None),
        "environment": capture_environment(
            ["python", "benchmarks/run_organizer_torch.py", *supplied_argv]
        ),
        "stdout": captured.getvalue().splitlines(),
    }
    output_path = wrapper_args.evidence_out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
