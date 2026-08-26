#!/usr/bin/env python3
"""Run the versioned Transformer matrix with fail-closed result accounting."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.capture_environment import capture_environment
from torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    benchmark_once,
    compare_outputs,
    copy_model_weights,
    generate_random_case,
    percentile,
    warmup_model,
)


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
TERMINAL_STATUSES = frozenset({"PASS", "FAIL", "OOM", "ERROR"})


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("cases"), list) or not payload["cases"]:
        raise ValueError("benchmark manifest must contain at least one case")
    if not isinstance(payload.get("defaults"), dict):
        raise ValueError("benchmark manifest must contain defaults")
    return payload


def result_exit_code(results: list[dict[str, Any]]) -> int:
    """A run is green only when at least one case ran and every case passed."""
    if not results:
        return 1
    statuses = [result.get("status") for result in results]
    if any(status not in TERMINAL_STATUSES for status in statuses):
        return 1
    return 0 if all(status == "PASS" for status in statuses) else 1


def is_oom_error(error: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", None)
    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    known_types = tuple(
        error_type
        for error_type in (oom_type, cuda_oom_type)
        if isinstance(error_type, type)
    )
    if known_types and isinstance(error, known_types):
        return True
    message = str(error).lower()
    return isinstance(error, RuntimeError) and "out of memory" in message


def _summarize_samples(samples: list[float]) -> dict[str, Any]:
    return {
        "raw_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p90_ms": percentile(samples, 0.9),
        "min_ms": min(samples),
    }


def _alternating_samples(
    baseline: torch.nn.Module,
    optimized: torch.nn.Module,
    x: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    *,
    warmup: int,
    repeats: int,
    rounds: int,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    warmup_model(baseline, x, valid_mask, warmup, device)
    warmup_model(optimized, x, valid_mask, warmup, device)
    baseline_samples: list[float] = []
    optimized_samples: list[float] = []
    for round_index in range(rounds):
        order = (
            ((baseline, baseline_samples), (optimized, optimized_samples))
            if round_index % 2 == 0
            else ((optimized, optimized_samples), (baseline, baseline_samples))
        )
        for model, destination in order:
            destination.extend(benchmark_once(model, x, valid_mask, repeats, device))
    return baseline_samples, optimized_samples


def _measure_peak_memory(
    model: torch.nn.Module,
    x: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    device: torch.device,
) -> Optional[dict[str, int]]:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    with torch.inference_mode():
        output = model(x, valid_mask)
    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    del output
    return {
        "allocated_before_bytes": allocated_before,
        "peak_allocated_bytes": peak_allocated,
        "incremental_peak_bytes": max(0, peak_allocated - allocated_before),
    }


def _case_config(case: dict[str, Any]) -> TransformerConfig:
    return TransformerConfig(
        batch_size=int(case["batch_size"]),
        seq_len=int(case["seq_len"]),
        d_model=int(case["d_model"]),
        num_heads=int(case["num_heads"]),
        ffn_dim=int(case["ffn_dim"]),
        num_layers=int(case["num_layers"]),
        causal=bool(case["causal"]),
    )


def run_case(
    case: dict[str, Any],
    dtype_name: str,
    defaults: dict[str, Any],
    *,
    device: torch.device,
    backend: str,
    warmup: int,
    repeats: int,
    rounds: int,
    accuracy_trials: int,
) -> dict[str, Any]:
    dtype = DTYPES[dtype_name]
    config = _case_config(case)
    config.validate()
    padding_ratio = float(case.get("padding_ratio", 0.0))
    seed = int(defaults.get("seeds", [1234])[0])
    input_scale = float(defaults.get("input_scale", 1.0))
    rtol = float(defaults["accuracy_rtol"])
    atol = float(defaults["accuracy_atol"])

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config, attention_backend=backend)
    copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()

    accuracy_trials_payload: list[dict[str, Any]] = []
    with torch.inference_mode():
        for trial in range(accuracy_trials):
            x, valid_mask = generate_random_case(
                config,
                device,
                dtype,
                seed + trial,
                padding_ratio,
                input_scale,
            )
            reference = baseline(x, valid_mask)
            actual = optimized(x, valid_mask)
            comparison = compare_outputs(reference, actual, rtol=rtol, atol=atol)
            accuracy_trials_payload.append(
                {
                    "seed": seed + trial,
                    "passed": comparison.passed,
                    "failed_elements": comparison.failed_elements,
                    "total_elements": comparison.total_elements,
                    "max_abs_error": comparison.max_abs_error,
                    "max_relative_error": comparison.max_relative_error,
                    "mean_abs_error": comparison.mean_abs_error,
                    "worst_index": list(comparison.worst_index),
                }
            )

    correctness_counts = dict(optimized.attention_backend_counts)
    del x, valid_mask, reference, actual
    all_passed = all(trial["passed"] for trial in accuracy_trials_payload)
    result: dict[str, Any] = {
        "case_id": case["id"],
        "dtype": dtype_name,
        "backend_requested": backend,
        "config": {**case, "input_scale": input_scale},
        "status": "PASS" if all_passed else "FAIL",
        "accuracy": {
            "rtol": rtol,
            "atol": atol,
            "trials": accuracy_trials_payload,
            "backend_counts": correctness_counts,
        },
        "timing": None,
        "peak_memory": None,
    }
    if not all_passed:
        return result

    optimized.reset_attention_backend_counts()
    timing_x, timing_mask = generate_random_case(
        config,
        device,
        dtype,
        seed + 100_000,
        padding_ratio,
        input_scale,
    )
    result["peak_memory"] = {
        "baseline": _measure_peak_memory(baseline, timing_x, timing_mask, device),
        "optimized": _measure_peak_memory(optimized, timing_x, timing_mask, device),
    }
    optimized.reset_attention_backend_counts()
    baseline_samples, optimized_samples = _alternating_samples(
        baseline,
        optimized,
        timing_x,
        timing_mask,
        warmup=warmup,
        repeats=repeats,
        rounds=rounds,
        device=device,
    )
    baseline_summary = _summarize_samples(baseline_samples)
    optimized_summary = _summarize_samples(optimized_samples)
    tokens = config.batch_size * config.seq_len
    result["timing"] = {
        "warmup": warmup,
        "repeats": repeats,
        "rounds": rounds,
        "baseline": baseline_summary,
        "optimized": optimized_summary,
        "speedup_median": (
            baseline_summary["median_ms"] / optimized_summary["median_ms"]
        ),
        "baseline_tokens_per_second": (
            tokens * 1000.0 / baseline_summary["median_ms"]
        ),
        "optimized_tokens_per_second": (
            tokens * 1000.0 / optimized_summary["median_ms"]
        ),
        "backend_counts": dict(optimized.attention_backend_counts),
    }
    return result


def execute_cases(
    case_inputs: list[tuple[dict[str, Any], str]],
    runner: Callable[[dict[str, Any], str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Execute every requested case and convert exceptions to failing states."""
    results: list[dict[str, Any]] = []
    for case, dtype_name in case_inputs:
        label = f"{case['id']}:{dtype_name}"
        print(f"[{label}] running", flush=True)
        try:
            result = runner(case, dtype_name)
        except Exception as error:
            status = "OOM" if is_oom_error(error) else "ERROR"
            result = {
                "case_id": case.get("id", "unknown"),
                "dtype": dtype_name,
                "status": status,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
            if status == "OOM" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if status == "ERROR":
                traceback.print_exc()
        print(f"[{label}] {result['status']}", flush=True)
        results.append(result)
    return results


def _print_summary(results: list[dict[str, Any]]) -> None:
    print("\n| case | dtype | status | baseline ms | optimized ms | speedup | backend counts |")
    print("| --- | --- | --- | ---: | ---: | ---: | --- |")
    for result in results:
        timing = result.get("timing") or {}
        baseline = timing.get("baseline") or {}
        optimized = timing.get("optimized") or {}
        speedup = timing.get("speedup_median")
        speedup_text = f"{speedup:.3f}x" if isinstance(speedup, (int, float)) else "-"
        print(
            f"| {result.get('case_id')} | {result.get('dtype')} | "
            f"{result.get('status')} | {baseline.get('median_ms', '-')} | "
            f"{optimized.get('median_ms', '-')} | {speedup_text} | "
            f"{timing.get('backend_counts', '-')} |"
        )
    passed_speeds = [
        result["timing"]["speedup_median"]
        for result in results
        if result.get("status") == "PASS" and result.get("timing")
    ]
    if passed_speeds:
        print(f"geomean speedup={statistics.geometric_mean(passed_speeds):.3f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "official_shapes.json",
    )
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--dtype", choices=tuple(DTYPES), action="append")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "triton", "sdpa", "reference"),
        default="auto",
    )
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--accuracy-trials", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction)
    parser.add_argument("--padding-ratio", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "matrix.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    defaults = dict(manifest["defaults"])
    if args.seed is not None:
        defaults["seeds"] = [args.seed]
    warmup = args.warmup if args.warmup is not None else int(defaults["warmup"])
    repeats = args.repeats if args.repeats is not None else int(defaults["repeats"])
    rounds = args.rounds if args.rounds is not None else int(defaults["rounds"])
    if args.quick:
        warmup, repeats, rounds = min(warmup, 2), min(repeats, 5), 1

    if warmup < 0 or any(
        value <= 0 for value in (repeats, rounds, args.accuracy_trials)
    ):
        raise ValueError("repeats, rounds, and accuracy trials must be positive")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    selected_cases = []
    available_ids = {case["id"] for case in manifest["cases"]}
    unknown_ids = set(args.case_ids or []) - available_ids
    if unknown_ids:
        raise ValueError(f"unknown case ids: {sorted(unknown_ids)}")
    for original_case in manifest["cases"]:
        if args.case_ids and original_case["id"] not in args.case_ids:
            continue
        case = dict(original_case)
        if args.causal is not None:
            case["causal"] = args.causal
        if args.padding_ratio is not None:
            case["padding_ratio"] = args.padding_ratio
        selected_cases.append(case)
    dtype_names = args.dtype or list(defaults["dtypes"])
    case_inputs = [
        (case, dtype_name)
        for case in selected_cases
        for dtype_name in dtype_names
    ]

    def configured_runner(case: dict[str, Any], dtype_name: str) -> dict[str, Any]:
        return run_case(
            case,
            dtype_name,
            defaults,
            device=device,
            backend=args.attention_backend,
            warmup=warmup,
            repeats=repeats,
            rounds=rounds,
            accuracy_trials=args.accuracy_trials,
        )

    results = execute_cases(case_inputs, configured_runner)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "path": str(args.manifest.resolve()),
            "status": manifest.get("status"),
            "organizer_matrix_available": manifest.get("organizer_matrix_available"),
        },
        "environment": capture_environment(sys.argv),
        "parameters": {
            "device": str(device),
            "backend": args.attention_backend,
            "warmup": warmup,
            "repeats": repeats,
            "rounds": rounds,
            "accuracy_trials": args.accuracy_trials,
        },
        "summary": {
            "requested": len(case_inputs),
            "completed": len(results),
            "counts": {
                status: sum(result.get("status") == status for result in results)
                for status in sorted(TERMINAL_STATUSES)
            },
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _print_summary(results)
    print(f"saved {args.out}")
    return result_exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())
