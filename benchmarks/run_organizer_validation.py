#!/usr/bin/env python3
"""Run a fail-closed matrix through the untouched organizer PyTorch harness.

Each executable case runs in its own subprocess. This isolates CUDA memory and
ensures one OOM or crash is recorded without preventing the remaining cases
from being audited. The only permitted skip is the exact 100,000-token stress
shape that the supplied TensorFlow benchmark itself marks as preflight-skippable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
DEFAULT_MATRIX_PATH = REPO_ROOT / "benchmarks" / "organizer_validation_matrix.json"
ORGANIZER_RUNNER_PATH = REPO_ROOT / "benchmarks" / "run_organizer_torch.py"
ORGANIZER_MANIFEST_PATH = (
    REPO_ROOT / "benchmarks" / "reference" / "organizer_downloads.json"
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EXECUTABLE_STATUSES = frozenset({"PASS", "FAIL", "OOM", "ERROR"})
TERMINAL_STATUSES = EXECUTABLE_STATUSES | {"SKIPPED_RESOURCE"}
DTYPES = frozenset({"float32", "float16", "bfloat16"})
OOM_MARKERS = (
    "out of memory",
    "cuda error: out of memory",
    "cuda outofmemoryerror",
    "cudnn_status_alloc_failed",
)
FINAL_STRESS_SOURCE_ROW = 14
FINAL_STRESS_DIMENSIONS = (32, 1024, 16, 100000, 2, True, 1024)


def _text_sha256(path: Path) -> str:
    """Hash repository text independently of checkout line endings."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _validate_config(config: dict[str, Any]) -> None:
    required = (
        "batch_size",
        "seq_len",
        "d_model",
        "num_heads",
        "ffn_dim",
        "num_layers",
        "causal",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"case config is missing keys: {missing}")
    for key in required[:-1]:
        if not isinstance(config[key], int) or isinstance(config[key], bool):
            raise ValueError(f"case config {key} must be an integer")
        if config[key] <= 0:
            raise ValueError(f"case config {key} must be positive")
    if config["d_model"] % config["num_heads"] != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if not isinstance(config["causal"], bool):
        raise ValueError("case config causal must be boolean")


def _validate_case_set(cases: Sequence[dict[str, Any]]) -> None:
    identifiers = [case.get("id") for case in cases]
    if any(
        not isinstance(identifier, str) or not identifier for identifier in identifiers
    ):
        raise ValueError("every validation case must have a non-empty id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("validation case ids must be unique")
    skipped = [case for case in cases if case["execution"] == "skip_resource"]
    if len(skipped) != 1 or not skipped[0].get("skip_authorized"):
        raise ValueError("exactly one source-authorized resource skip is permitted")


def _load_explicit_cases(
    matrix: dict[str, Any], source: dict[str, Any]
) -> list[dict[str, Any]]:
    raw_cases = matrix.get("explicit_cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("explicit validation matrix must define cases")

    cases: list[dict[str, Any]] = []
    source_rows: list[int] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("explicit validation cases must be objects")
        case = dict(raw_case)
        if case.get("source_contract") != "organizer-final-shape-table":
            raise ValueError("explicit cases must identify the final organizer shape table")
        source_row = case.get("source_row")
        if not isinstance(source_row, int) or isinstance(source_row, bool):
            raise ValueError("explicit case source_row must be an integer")
        source_rows.append(source_row)
        _validate_config(case.get("config") or {})
        if case.get("dtype") not in DTYPES:
            raise ValueError(f"invalid dtype for {case.get('id')}")
        padding_ratio = case.get("padding_ratio")
        if not isinstance(padding_ratio, (int, float)) or not 0 <= padding_ratio < 1:
            raise ValueError(f"invalid padding ratio for {case.get('id')}")
        execution = case.get("execution")
        if execution not in {"run", "skip_resource"}:
            raise ValueError(f"invalid execution policy for {case.get('id')}")
        expected_dimensions = [
            case["config"]["batch_size"],
            case["config"]["d_model"],
            case["config"]["num_heads"],
            case["config"]["seq_len"],
            case["config"]["num_layers"],
            case["config"]["causal"],
            case["config"]["ffn_dim"],
        ]
        if case.get("source_dimensions") != expected_dimensions:
            raise ValueError(
                f"source dimensions do not match config for {case.get('id')}"
            )
        if execution == "skip_resource":
            if case.get("skip_authorized") is not True:
                raise ValueError("resource skip must be explicitly source-authorized")
            tf_contract = source.get("tensorflow_contract") or {}
            compact_cases = tf_contract.get("compact_cases") or []
            if (
                not tf_contract.get("stress_case_may_be_preflight_skipped")
                or not compact_cases
                or tuple(compact_cases[-1])
                != (
                    FINAL_STRESS_DIMENSIONS[0],
                    FINAL_STRESS_DIMENSIONS[1],
                    FINAL_STRESS_DIMENSIONS[2],
                    FINAL_STRESS_DIMENSIONS[3],
                )
            ):
                raise ValueError("source contract does not authorize the final stress skip")
            if (
                source_row != FINAL_STRESS_SOURCE_ROW
                or tuple(expected_dimensions) != FINAL_STRESS_DIMENSIONS
            ):
                raise ValueError(
                    "resource skip must target the exact final-row stress dimensions"
                )
            if not isinstance(case.get("description"), str) or not case[
                "description"
            ].strip():
                raise ValueError("resource skip must explain its authorization")
        elif case.get("skip_authorized"):
            raise ValueError("executable cases cannot declare skip authorization")
        cases.append(case)

    if source_rows != list(range(1, len(cases) + 1)):
        raise ValueError("explicit cases must preserve consecutive source row order")
    if len(cases) != FINAL_STRESS_SOURCE_ROW:
        raise ValueError("final organizer matrix must contain exactly 14 source rows")
    _validate_case_set(cases)
    return cases


def load_and_expand_matrix(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the versioned policy and expand source-derived TensorFlow shapes."""
    matrix = _load_json(path)
    source = _load_json(ORGANIZER_MANIFEST_PATH)
    if matrix.get("schema_version") != 1:
        raise ValueError("unsupported organizer validation matrix schema")
    if matrix.get("source_manifest") != "benchmarks/reference/organizer_downloads.json":
        raise ValueError("validation matrix must reference the frozen organizer manifest")

    defaults = matrix.get("defaults")
    if not isinstance(defaults, dict):
        raise ValueError("validation matrix defaults must be an object")
    for key in ("accuracy_trials", "repeats", "benchmark_rounds"):
        if not isinstance(defaults.get(key), int) or defaults[key] <= 0:
            raise ValueError(f"matrix default {key} must be a positive integer")
    if not isinstance(defaults.get("warmup"), int) or defaults["warmup"] < 0:
        raise ValueError("matrix default warmup must be a non-negative integer")
    if defaults.get("attention_backend") != "auto":
        raise ValueError(
            "the untouched organizer CLI cannot override the injected class's "
            "default auto attention policy"
        )

    if "explicit_cases" in matrix:
        return matrix, _load_explicit_cases(matrix, source)

    source_default = source["pytorch_contract"]["default_case"]
    cases: list[dict[str, Any]] = []
    pytorch_cases = matrix.get("pytorch_cases")
    if not isinstance(pytorch_cases, list) or not pytorch_cases:
        raise ValueError("validation matrix must define PyTorch cases")
    for raw_case in pytorch_cases:
        case = dict(raw_case)
        case["source_contract"] = "pytorch"
        case["execution"] = "run"
        _validate_config(case["config"])
        if case.get("dtype") not in DTYPES:
            raise ValueError(f"invalid dtype for {case.get('id')}")
        padding_ratio = case.get("padding_ratio")
        if not isinstance(padding_ratio, (int, float)) or not 0 <= padding_ratio < 1:
            raise ValueError(f"invalid padding ratio for {case.get('id')}")
        cases.append(case)

    exact_default = [
        case
        for case in cases
        if case["config"] == {
            key: source_default[key]
            for key in (
                "batch_size",
                "seq_len",
                "d_model",
                "num_heads",
                "ffn_dim",
                "num_layers",
                "causal",
            )
        }
        and case["dtype"] == source_default["dtype"]
        and case["padding_ratio"] == source_default["padding_ratio"]
    ]
    if len(exact_default) != 1:
        raise ValueError("matrix must contain the exact PyTorch default exactly once")

    translation = matrix.get("tensorflow_shape_translation")
    if not isinstance(translation, dict):
        raise ValueError("validation matrix must define TensorFlow shape translation")
    tf_contract = source["tensorflow_contract"]
    if translation.get("num_layers") != tf_contract["num_layers"]:
        raise ValueError("translated layer count does not match TensorFlow contract")
    if translation.get("ffn_multiplier") != tf_contract["ffn_multiplier"]:
        raise ValueError("translated FFN multiplier does not match TensorFlow contract")
    execution_dtypes = translation.get("execution_dtypes")
    if execution_dtypes != ["float32", tf_contract["default_dtype"]]:
        raise ValueError(
            "TensorFlow shapes must run in float32 and the supplied default dtype"
        )
    stress_case = translation.get("stress_case")
    compact_cases = tf_contract["compact_cases"]
    if stress_case != compact_cases[-1]:
        raise ValueError("resource skip must target the exact designated stress case")
    if translation.get("stress_status") != "SKIPPED_RESOURCE":
        raise ValueError("designated stress status must be SKIPPED_RESOURCE")
    if translation.get("stress_dtype") != tf_contract["default_dtype"]:
        raise ValueError("stress entry must preserve the TensorFlow default dtype")
    if not isinstance(translation.get("stress_reason"), str) or not translation[
        "stress_reason"
    ].strip():
        raise ValueError("stress entry must explain its source-authorized skip")
    padding_ratio = translation.get("padding_ratio")
    if not isinstance(padding_ratio, (int, float)) or not 0 <= padding_ratio < 1:
        raise ValueError("translated TensorFlow padding ratio is invalid")
    if not tf_contract.get("stress_case_may_be_preflight_skipped"):
        raise ValueError("source contract does not authorize a stress preflight skip")

    for index, dimensions in enumerate(compact_cases, start=1):
        batch_size, d_model, num_heads, seq_len = dimensions
        config = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "d_model": d_model,
            "num_heads": num_heads,
            "ffn_dim": d_model * translation["ffn_multiplier"],
            "num_layers": translation["num_layers"],
            "causal": translation["causal"],
        }
        _validate_config(config)
        base_id = (
            f"tf-{index:02d}-b{batch_size}-d{d_model}-"
            f"h{num_heads}-s{seq_len}"
        )
        if dimensions == stress_case:
            cases.append(
                {
                    "id": f"{base_id}-resource-skip",
                    "description": translation["stress_reason"],
                    "source_contract": "tensorflow-shape-cross-check",
                    "source_dimensions": dimensions,
                    "dtype": translation["stress_dtype"],
                    "padding_ratio": translation["padding_ratio"],
                    "config": config,
                    "execution": "skip_resource",
                    "skip_authorized": True,
                }
            )
            continue
        for dtype in execution_dtypes:
            cases.append(
                {
                    "id": f"{base_id}-{dtype}",
                    "description": (
                        "Supplied TensorFlow compact shape translated to the "
                        f"selected PyTorch contract in {dtype}."
                    ),
                    "source_contract": "tensorflow-shape-cross-check",
                    "source_dimensions": dimensions,
                    "dtype": dtype,
                    "padding_ratio": translation["padding_ratio"],
                    "config": config,
                    "execution": "run",
                }
            )

    _validate_case_set(cases)
    return matrix, cases


def organizer_arguments(
    case: dict[str, Any], defaults: dict[str, Any], device: str
) -> list[str]:
    config = case["config"]
    arguments = [
        "--batch-size",
        str(config["batch_size"]),
        "--seq-len",
        str(config["seq_len"]),
        "--d-model",
        str(config["d_model"]),
        "--heads",
        str(config["num_heads"]),
        "--ffn-dim",
        str(config["ffn_dim"]),
        "--layers",
        str(config["num_layers"]),
        "--device",
        device,
        "--dtype",
        case["dtype"],
        "--padding-ratio",
        str(case["padding_ratio"]),
        "--input-scale",
        str(defaults["input_scale"]),
        "--accuracy-trials",
        str(defaults["accuracy_trials"]),
        "--rtol",
        str(defaults["rtol"]),
        "--atol",
        str(defaults["atol"]),
        "--seed",
        str(defaults["seed"]),
        "--warmup",
        str(defaults["warmup"]),
        "--repeats",
        str(defaults["repeats"]),
        "--benchmark-rounds",
        str(defaults["benchmark_rounds"]),
    ]
    if config["causal"]:
        arguments.append("--causal")
    return arguments


def _is_oom(stdout: str, stderr: str) -> bool:
    combined = f"{stdout}\n{stderr}".lower()
    return any(marker in combined for marker in OOM_MARKERS)


def _complete_pass(evidence: Optional[dict[str, Any]]) -> bool:
    if not evidence or evidence.get("exit_code") != 0:
        return False
    parsed = evidence.get("parsed") or {}
    backend_counts = evidence.get("attention_backend_counts")
    return (
        (parsed.get("accuracy") or {}).get("status") == "PASS"
        and "baseline" in parsed
        and "optimized" in parsed
        and "speedup_median" in parsed
        and isinstance(backend_counts, dict)
        and set(backend_counts) == {"triton", "sdpa", "reference"}
        and sum(backend_counts.values()) > 0
    )


def execute_case(
    case: dict[str, Any],
    defaults: dict[str, Any],
    *,
    device: str,
    timeout_seconds: int,
    evidence_path: Path,
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    arguments = organizer_arguments(case, defaults, device)
    command = [
        sys.executable,
        str(ORGANIZER_RUNNER_PATH),
        "--evidence-out",
        str(evidence_path),
        *arguments,
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        duration_seconds = time.perf_counter() - started
    except subprocess.TimeoutExpired as error:
        duration_seconds = time.perf_counter() - started
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return (
            {
                "case_id": case["id"],
                "source_contract": case["source_contract"],
                "source_dimensions": case.get("source_dimensions"),
                "config": case["config"],
                "dtype": case["dtype"],
                "padding_ratio": case["padding_ratio"],
                "status": "ERROR",
                "duration_seconds": duration_seconds,
                "organizer_arguments": arguments,
                "return_code": None,
                "error": {
                    "type": "TimeoutExpired",
                    "message": f"case exceeded {timeout_seconds} seconds",
                },
                "stdout": stdout.splitlines(),
                "stderr": stderr.splitlines(),
            },
            None,
        )

    evidence: Optional[dict[str, Any]] = None
    evidence_error: Optional[str] = None
    if evidence_path.is_file():
        try:
            evidence = _load_json(evidence_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            evidence_error = f"invalid child evidence: {error}"

    parsed = (evidence or {}).get("parsed") or {}
    accuracy_status = (parsed.get("accuracy") or {}).get("status")
    if completed.returncode == 0 and _complete_pass(evidence):
        status = "PASS"
        error_payload = None
    elif _is_oom(completed.stdout, completed.stderr):
        status = "OOM"
        error_payload = {
            "type": "OutOfMemory",
            "message": "child process reported an out-of-memory condition",
        }
    elif accuracy_status == "FAIL":
        status = "FAIL"
        error_payload = {
            "type": "AccuracyFailure",
            "message": "untouched organizer comparator rejected the output",
        }
    else:
        status = "ERROR"
        error_payload = {
            "type": "IncompleteEvidence" if completed.returncode == 0 else "ChildProcessError",
            "message": evidence_error
            or f"organizer child exited with code {completed.returncode}",
        }

    result = {
        "case_id": case["id"],
        "description": case["description"],
        "source_contract": case["source_contract"],
        "source_dimensions": case.get("source_dimensions"),
        "config": case["config"],
        "dtype": case["dtype"],
        "padding_ratio": case["padding_ratio"],
        "status": status,
        "duration_seconds": duration_seconds,
        "organizer_arguments": arguments,
        "return_code": completed.returncode,
        "parsed": parsed,
        "attention_backend_counts": (evidence or {}).get("attention_backend_counts"),
        "error": error_payload,
        "stdout": completed.stdout.splitlines(),
        "stderr": completed.stderr.splitlines(),
    }
    return result, (evidence or {}).get("environment")


def result_exit_code(results: Sequence[dict[str, Any]]) -> int:
    """Green means all runnable cases passed and only authorized skips occurred."""
    if not results:
        return 1
    statuses = [result.get("status") for result in results]
    if any(status not in TERMINAL_STATUSES for status in statuses):
        return 1
    executable = [
        result for result in results if result.get("status") != "SKIPPED_RESOURCE"
    ]
    if not executable or any(result.get("status") != "PASS" for result in executable):
        return 1
    skipped = [
        result for result in results if result.get("status") == "SKIPPED_RESOURCE"
    ]
    if any(not result.get("skip_authorized") for result in skipped):
        return 1
    return 0


def summarize_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        status: sum(result.get("status") == status for result in results)
        for status in sorted(TERMINAL_STATUSES)
    }
    executable = [
        result for result in results if result.get("status") != "SKIPPED_RESOURCE"
    ]
    passed = [result for result in executable if result.get("status") == "PASS"]
    accuracy = [result["parsed"]["accuracy"] for result in passed]
    speedups = [result["parsed"]["speedup_median"] for result in passed]
    backend_totals = {"triton": 0, "sdpa": 0, "reference": 0}
    for result in passed:
        for backend, count in (result.get("attention_backend_counts") or {}).items():
            if backend in backend_totals:
                backend_totals[backend] += int(count)
    return {
        "requested": len(results),
        "executable": len(executable),
        "passed": len(passed),
        "skipped_resource": counts["SKIPPED_RESOURCE"],
        "skipped_counted_as_pass": False,
        "all_executable_passed": len(passed) == len(executable) and bool(executable),
        "counts": counts,
        "total_compared_elements": sum(item["total_elements"] for item in accuracy),
        "total_failed_elements": sum(item["failed_elements"] for item in accuracy),
        "max_abs_error": max((item["max_abs_error"] for item in accuracy), default=None),
        "max_relative_error": max(
            (item["max_relative_error"] for item in accuracy), default=None
        ),
        "geometric_mean_speedup": (
            statistics.geometric_mean(speedups) if speedups else None
        ),
        "attention_backend_counts": backend_totals,
    }


def _print_summary(results: Sequence[dict[str, Any]], summary: dict[str, Any]) -> None:
    print("\n| case | dtype | status | max abs | speedup | backends |")
    print("| --- | --- | --- | ---: | ---: | --- |")
    for result in results:
        parsed = result.get("parsed") or {}
        accuracy = parsed.get("accuracy") or {}
        speedup = parsed.get("speedup_median")
        speedup_text = f"{speedup:.3f}x" if isinstance(speedup, (int, float)) else "-"
        max_abs = accuracy.get("max_abs_error", "-")
        print(
            f"| {result['case_id']} | {result['dtype']} | {result['status']} | "
            f"{max_abs} | {speedup_text} | "
            f"{result.get('attention_backend_counts') or '-'} |"
        )
    print(
        "aggregate: "
        f"{summary['passed']}/{summary['executable']} executable PASS, "
        f"{summary['skipped_resource']} resource skip, "
        f"{summary['total_failed_elements']}/"
        f"{summary['total_compared_elements']} failed elements"
    )
    if summary["geometric_mean_speedup"] is not None:
        print(f"geomean speedup={summary['geometric_mean_speedup']:.3f}x")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--device")
    parser.add_argument("--case-timeout-seconds", type=int, default=900)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "organizer-validation.json",
    )
    return parser


def _recorded_command(args: argparse.Namespace, device: str) -> list[str]:
    """Render the top-level invocation without child temp or host-home paths."""
    from tools.capture_environment import display_path

    command = [
        "python",
        display_path(RUNNER_PATH),
        "--matrix",
        display_path(args.matrix),
        "--device",
        device,
        "--case-timeout-seconds",
        str(args.case_timeout_seconds),
        "--out",
        display_path(args.out),
    ]
    for case_id in args.case_ids or []:
        command.extend(("--case", case_id))
    return command


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    matrix, all_cases = load_and_expand_matrix(args.matrix)
    if args.case_timeout_seconds <= 0:
        raise ValueError("case timeout must be positive")
    if args.list_cases:
        for case in all_cases:
            print(f"{case['id']}\t{case['execution']}")
        return 0

    available_ids = {case["id"] for case in all_cases}
    unknown = set(args.case_ids or []) - available_ids
    if unknown:
        raise ValueError(f"unknown case ids: {sorted(unknown)}")
    cases = [
        case for case in all_cases if not args.case_ids or case["id"] in args.case_ids
    ]
    if not cases:
        raise ValueError("no validation cases selected")

    defaults = dict(matrix["defaults"])
    device = args.device or defaults["device"]
    results: list[dict[str, Any]] = []
    environment: Optional[dict[str, Any]] = None
    with tempfile.TemporaryDirectory(prefix="techjam-organizer-validation-") as tmp:
        temporary_directory = Path(tmp)
        for index, case in enumerate(cases, start=1):
            label = f"[{index:02d}/{len(cases):02d}] {case['id']}"
            if case["execution"] == "skip_resource":
                print(f"{label}: SKIPPED_RESOURCE (source-authorized preflight)", flush=True)
                results.append(
                    {
                        "case_id": case["id"],
                        "description": case["description"],
                        "source_contract": case["source_contract"],
                        "source_dimensions": case["source_dimensions"],
                        "config": case["config"],
                        "dtype": case["dtype"],
                        "padding_ratio": case["padding_ratio"],
                        "status": "SKIPPED_RESOURCE",
                        "skip_authorized": bool(case["skip_authorized"]),
                        "skip_counted_as_pass": False,
                        "parsed": {},
                        "attention_backend_counts": None,
                    }
                )
                continue

            print(f"{label}: running", flush=True)
            result, child_environment = execute_case(
                case,
                defaults,
                device=device,
                timeout_seconds=args.case_timeout_seconds,
                evidence_path=temporary_directory / f"{case['id']}.json",
            )
            if environment is None and child_environment is not None:
                environment = child_environment
            results.append(result)
            parsed = result.get("parsed") or {}
            accuracy = parsed.get("accuracy") or {}
            detail = (
                f" max_abs={accuracy['max_abs_error']:.6g}"
                if "max_abs_error" in accuracy
                else ""
            )
            print(f"{label}: {result['status']}{detail}", flush=True)

    summary = summarize_results(results)
    if environment is not None:
        environment = dict(environment)
        environment["command"] = _recorded_command(args, device)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if result_exit_code(results) == 0 else "FAIL",
        "matrix": {
            "path": str(args.matrix.resolve().relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": _text_sha256(args.matrix),
            "status": matrix["status"],
        },
        "organizer_sources": {
            "manifest_path": "benchmarks/reference/organizer_downloads.json",
            "manifest_sha256": _text_sha256(ORGANIZER_MANIFEST_PATH),
            "runner_path": "benchmarks/run_organizer_torch.py",
            "runner_sha256": _text_sha256(ORGANIZER_RUNNER_PATH),
            "validation_runner_path": "benchmarks/run_organizer_validation.py",
            "validation_runner_sha256": _text_sha256(RUNNER_PATH),
        },
        "parameters": {
            **defaults,
            "device": device,
            "case_timeout_seconds": args.case_timeout_seconds,
            "isolated_process_per_case": True,
            "selected_case_ids": args.case_ids,
        },
        "environment": environment,
        "summary": summary,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _print_summary(results, summary)
    print(f"saved {args.out}")
    return result_exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())
