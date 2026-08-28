#!/usr/bin/env python3
"""Run one optimization experiment and persist an immutable evidence record.

The wrapper deliberately records unsuccessful commands and missing/invalid
result artifacts. It never invokes a shell and must not be used with commands
whose argv or output contains credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.capture_environment import capture_environment, display_path


ATTEMPT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DECISIONS = ("observation", "pending", "keep", "reject", "rework", "inconclusive")
REVIEW_STATUSES = ("not_required", "pending", "approved", "rejected")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_command(command: list[str]) -> list[str]:
    """Remove machine-specific absolute path prefixes from persisted argv."""
    rendered: list[str] = []
    for argument in command:
        candidate = Path(argument)
        rendered.append(display_path(candidate) if candidate.is_absolute() else argument)
    return rendered


def _git_value(*args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
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


def _sum_dicts(values: Iterable[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        for key, raw_count in value.items():
            if isinstance(raw_count, int) and not isinstance(raw_count, bool):
                result[key] = result.get(key, 0) + raw_count
    return result


def _maximum(values: Iterable[Any]) -> Optional[float]:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return max(numeric) if numeric else None


def _latency_summary(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    result = {
        key: value.get(key)
        for key in ("median_ms", "mean_ms", "p90_ms", "min_ms", "tokens_per_second")
        if isinstance(value.get(key), (int, float))
    }
    raw_samples = value.get("raw_ms")
    if isinstance(raw_samples, list):
        numeric = [float(item) for item in raw_samples if isinstance(item, (int, float))]
        result["sample_count"] = len(numeric)
        if numeric:
            result["sample_stdev_ms"] = statistics.stdev(numeric) if len(numeric) > 1 else 0.0
            result["sample_max_ms"] = max(numeric)
    return result or None


def _memory_summary(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    baseline = value.get("baseline")
    optimized = value.get("optimized")
    if not isinstance(baseline, dict) or not isinstance(optimized, dict):
        return None
    baseline_peak = baseline.get("incremental_peak_bytes")
    optimized_peak = optimized.get("incremental_peak_bytes")
    result: dict[str, Any] = {
        "baseline_incremental_peak_bytes": baseline_peak,
        "optimized_incremental_peak_bytes": optimized_peak,
    }
    if isinstance(baseline_peak, (int, float)) and baseline_peak > 0 and isinstance(
        optimized_peak, (int, float)
    ):
        result["reduction_percent"] = 100.0 * (1.0 - optimized_peak / baseline_peak)
    return result


def _summarize_validation(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    parameters = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
    trials_per_case = parameters.get("accuracy_trials")
    executable = summary.get("executable")
    total_trials = (
        int(trials_per_case) * int(executable)
        if isinstance(trials_per_case, int)
        and not isinstance(trials_per_case, bool)
        and isinstance(executable, int)
        and not isinstance(executable, bool)
        else None
    )
    per_case: list[dict[str, Any]] = []
    durations: list[float] = []
    for raw_result in payload.get("results") or []:
        if not isinstance(raw_result, dict):
            continue
        parsed = raw_result.get("parsed") if isinstance(raw_result.get("parsed"), dict) else {}
        duration = raw_result.get("duration_seconds")
        if isinstance(duration, (int, float)):
            durations.append(float(duration))
        per_case.append(
            {
                "case_id": raw_result.get("case_id"),
                "status": raw_result.get("status"),
                "duration_seconds": duration,
                "accuracy": parsed.get("accuracy"),
                "baseline": _latency_summary(parsed.get("baseline")),
                "optimized": _latency_summary(parsed.get("optimized")),
                "speedup_median": parsed.get("speedup_median"),
                "attention_backend_counts": raw_result.get("attention_backend_counts"),
            }
        )
    return {
        "kind": "organizer_validation",
        "status": payload.get("status"),
        "correctness": (
            {
                key: summary.get(key)
                for key in (
                    "requested",
                    "executable",
                    "passed",
                    "skipped_resource",
                    "skipped_counted_as_pass",
                    "all_executable_passed",
                    "counts",
                    "total_compared_elements",
                    "total_failed_elements",
                    "max_abs_error",
                    "max_relative_error",
                )
            }
            | {
                "accuracy_trials": total_trials,
                "accuracy_trials_per_case": trials_per_case,
            }
        ),
        "performance": {
            "geometric_mean_speedup": summary.get("geometric_mean_speedup"),
            "total_case_duration_seconds": sum(durations),
            "per_case": per_case,
        },
        "attention_backend_counts": summary.get("attention_backend_counts"),
        "memory": None,
        "profiler": None,
    }


def _summarize_matrix(payload: dict[str, Any]) -> dict[str, Any]:
    results = [item for item in payload.get("results") or [] if isinstance(item, dict)]
    trials: list[dict[str, Any]] = []
    per_case: list[dict[str, Any]] = []
    speedups: list[float] = []
    backend_values: list[dict[str, Any]] = []
    memory_cases: list[dict[str, Any]] = []
    for result in results:
        accuracy = result.get("accuracy") if isinstance(result.get("accuracy"), dict) else {}
        case_trials = [item for item in accuracy.get("trials") or [] if isinstance(item, dict)]
        trials.extend(case_trials)
        timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
        speedup = timing.get("speedup_median")
        if result.get("status") == "PASS" and isinstance(speedup, (int, float)) and speedup > 0:
            speedups.append(float(speedup))
        backends = timing.get("backend_counts")
        if isinstance(backends, dict):
            backend_values.append(backends)
        memory = _memory_summary(result.get("peak_memory"))
        if memory is not None:
            memory_cases.append({"case_id": result.get("case_id"), **memory})
        per_case.append(
            {
                "case_id": result.get("case_id"),
                "status": result.get("status"),
                "accuracy_trial_count": len(case_trials),
                "failed_elements": sum(
                    int(item.get("failed_elements", 0)) for item in case_trials
                ),
                "baseline": _latency_summary(timing.get("baseline")),
                "optimized": _latency_summary(timing.get("optimized")),
                "speedup_median": speedup,
                "attention_backend_counts": backends,
                "memory": memory,
            }
        )
    counts = payload.get("summary", {}).get("counts") if isinstance(payload.get("summary"), dict) else None
    return {
        "kind": "project_matrix",
        "status": "PASS" if results and all(item.get("status") == "PASS" for item in results) else "FAIL",
        "correctness": {
            "requested": payload.get("summary", {}).get("requested") if isinstance(payload.get("summary"), dict) else None,
            "executable": len(results),
            "passed": sum(item.get("status") == "PASS" for item in results),
            "counts": counts,
            "accuracy_trials": len(trials),
            "total_compared_elements": sum(int(item.get("total_elements", 0)) for item in trials),
            "total_failed_elements": sum(int(item.get("failed_elements", 0)) for item in trials),
            "max_abs_error": _maximum(item.get("max_abs_error") for item in trials),
            "max_relative_error": _maximum(item.get("max_relative_error") for item in trials),
        },
        "performance": {
            "geometric_mean_speedup": math.exp(
                sum(math.log(value) for value in speedups) / len(speedups)
            )
            if speedups
            else None,
            "per_case": per_case,
        },
        "attention_backend_counts": _sum_dicts(backend_values),
        "memory": {"per_case": memory_cases},
        "profiler": None,
    }


def _summarize_organizer_default(payload: dict[str, Any]) -> dict[str, Any]:
    parsed = payload.get("parsed") if isinstance(payload.get("parsed"), dict) else {}
    accuracy = parsed.get("accuracy") if isinstance(parsed.get("accuracy"), dict) else {}
    return {
        "kind": "organizer_default",
        "status": accuracy.get("status"),
        "correctness": {
            "accuracy_trials": 5,
            "total_compared_elements": accuracy.get("total_elements"),
            "total_failed_elements": accuracy.get("failed_elements"),
            "max_abs_error": accuracy.get("max_abs_error"),
            "max_relative_error": accuracy.get("max_relative_error"),
        },
        "performance": {
            "baseline": _latency_summary(parsed.get("baseline")),
            "optimized": _latency_summary(parsed.get("optimized")),
            "speedup_median": parsed.get("speedup_median"),
        },
        "attention_backend_counts": payload.get("attention_backend_counts"),
        "memory": None,
        "profiler": None,
    }


def _summarize_profile(payload: dict[str, Any]) -> dict[str, Any]:
    custom_events = payload.get("custom_kernel_events")
    top_events = payload.get("top_events")
    return {
        "kind": "profile",
        "status": "PASS" if payload.get("custom_kernel_profiler_proven") else "INCONCLUSIVE",
        "correctness": None,
        "performance": None,
        "attention_backend_counts": payload.get("backend_counts"),
        "memory": None,
        "profiler": {
            "steps": payload.get("steps"),
            "custom_kernel_expected": payload.get("custom_kernel_expected"),
            "custom_kernel_profiler_proven": payload.get("custom_kernel_profiler_proven"),
            "custom_kernel_events": custom_events if isinstance(custom_events, list) else [],
            "top_events": top_events if isinstance(top_events, list) else [],
        },
    }


def summarize_result_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract comparable metrics while retaining the original artifact separately."""
    if "organizer_sources" in payload and "matrix" in payload and "results" in payload:
        return _summarize_validation(payload)
    if "manifest" in payload and "results" in payload and "parameters" in payload:
        return _summarize_matrix(payload)
    if "organizer_script" in payload and "parsed" in payload:
        return _summarize_organizer_default(payload)
    if "custom_kernel_events" in payload and "backend_counts" in payload:
        return _summarize_profile(payload)
    return {
        "kind": "unknown",
        "status": payload.get("status"),
        "correctness": payload.get("summary"),
        "performance": None,
        "attention_backend_counts": payload.get("attention_backend_counts"),
        "memory": None,
        "profiler": None,
    }


def _pump_stream(stream: TextIO, sink: TextIO, chunks: list[str], tee: bool) -> None:
    try:
        for chunk in iter(stream.readline, ""):
            chunks.append(chunk)
            if tee:
                try:
                    sink.write(chunk)
                except UnicodeEncodeError:
                    encoding = getattr(sink, "encoding", None) or "utf-8"
                    safe_chunk = chunk.encode(encoding, errors="replace").decode(encoding)
                    sink.write(safe_chunk)
                sink.flush()
    finally:
        stream.close()


def run_logged_command(
    command: list[str],
    *,
    timeout_seconds: Optional[float] = None,
    tee: bool = True,
) -> dict[str, Any]:
    """Execute argv without a shell and return output even on failure or timeout."""
    if not command:
        raise ValueError("attempt command must not be empty")
    started_at = _utc_now()
    started = time.perf_counter()
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    return_code: Optional[int] = None
    launch_error: Optional[str] = None
    timed_out = False
    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_thread = threading.Thread(
            target=_pump_stream,
            args=(process.stdout, sys.stdout, stdout_chunks, tee),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_pump_stream,
            args=(process.stderr, sys.stderr, stderr_chunks, tee),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            return_code = process.wait()
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
    except OSError as exc:
        launch_error = f"{type(exc).__name__}: {exc}"
        return_code = 127
    ended_at = _utc_now()
    wall_time = time.perf_counter() - started
    if launch_error is not None:
        status = "ERROR"
    elif timed_out:
        status = "TIMEOUT"
    elif return_code == 0:
        status = "PASS"
    else:
        status = "FAIL"
    return {
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "wall_time_seconds": wall_time,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "status": status,
        "return_code": return_code,
        "launch_error": launch_error,
        "stdout": "".join(stdout_chunks),
        "stderr": "".join(stderr_chunks),
    }


def load_result_artifact(path: Optional[Path]) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    if path is None:
        return {"path": None, "expected": False, "exists": False}, None
    resolved = path if path.is_absolute() else REPO_ROOT / path
    metadata: dict[str, Any] = {
        "path": display_path(resolved),
        "expected": True,
        "exists": resolved.is_file(),
    }
    if not resolved.is_file():
        metadata["parse_error"] = "expected result artifact was not created"
        return metadata, None
    metadata.update({"size_bytes": resolved.stat().st_size, "sha256": _sha256(resolved)})
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        metadata["parse_error"] = f"{type(exc).__name__}: {exc}"
        return metadata, None
    if not isinstance(payload, dict):
        metadata["parse_error"] = "result artifact root must be a JSON object"
        return metadata, None
    metadata["schema_version"] = payload.get("schema_version")
    metadata["created_at_utc"] = payload.get("created_at_utc")
    return metadata, payload


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    resolved = path if path.is_absolute() else REPO_ROOT / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--parent-commit")
    parser.add_argument("--changed-path", action="append", default=[])
    parser.add_argument("--rollback", required=True)
    parser.add_argument("--decision", choices=DECISIONS, default="pending")
    parser.add_argument("--decision-rationale", required=True)
    parser.add_argument("--review-status", choices=REVIEW_STATUSES, default="pending")
    parser.add_argument("--result-artifact", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not ATTEMPT_ID_PATTERN.fullmatch(args.attempt_id):
        parser.error("attempt id must use only letters, digits, dot, underscore, or dash")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def main() -> int:
    args = _parse_args()
    out_path = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out_path.exists():
        print(f"attempt record already exists: {display_path(out_path)}", file=sys.stderr)
        return 2

    recorded_command = portable_command(args.command)
    environment_before = capture_environment(recorded_command)
    branch = _git_value("branch", "--show-current")
    execution = run_logged_command(
        args.command,
        timeout_seconds=args.timeout_seconds,
        tee=True,
    )
    result_artifact, result_payload = load_result_artifact(args.result_artifact)
    environment_after = capture_environment(recorded_command)
    logger_path = Path(__file__).resolve()
    metrics = summarize_result_artifact(result_payload) if result_payload is not None else None

    artifact_error = bool(result_artifact.get("parse_error"))
    if execution["status"] != "PASS":
        record_status = execution["status"]
    elif artifact_error:
        record_status = "ARTIFACT_ERROR"
    else:
        record_status = "RECORDED"

    record = {
        "schema_version": 1,
        "record_status": record_status,
        "attempt": {
            "id": args.attempt_id,
            "hypothesis": args.hypothesis,
            "scope": args.scope,
            "parent_commit": args.parent_commit or environment_before["git"]["commit"],
            "candidate_commit": environment_before["git"]["commit"],
            "branch": branch,
            "changed_paths": args.changed_path,
            "rollback": args.rollback,
            "decision": args.decision,
            "decision_rationale": args.decision_rationale,
            "review_status": args.review_status,
        },
        "execution": {"command": recorded_command, "cwd": ".", **execution},
        "environment_before": environment_before,
        "environment_after": environment_after,
        "logger": {
            "path": display_path(logger_path),
            "sha256": _sha256(logger_path),
        },
        "result_artifact": result_artifact,
        "metrics": metrics,
    }
    write_json_exclusive(out_path, record)
    print(f"attempt_record={display_path(out_path)}")
    print(f"record_status={record_status}")

    if execution["status"] != "PASS":
        if execution["timed_out"]:
            return 124
        return int(execution["return_code"] or 1)
    return 1 if artifact_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
