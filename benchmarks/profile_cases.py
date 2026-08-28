#!/usr/bin/env python3
"""Profile one manifest case and prove the selected attention kernel executed."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_matrix import DTYPES, load_manifest
from tools.capture_environment import capture_environment
from torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    copy_model_weights,
    generate_random_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "official_shapes.json",
    )
    parser.add_argument("--case", required=True, dest="case_id")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "triton", "sdpa", "reference"),
        default="auto",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    return parser.parse_args()


def _config(case: dict[str, Any]) -> TransformerConfig:
    return TransformerConfig(
        int(case["batch_size"]),
        int(case["seq_len"]),
        int(case["d_model"]),
        int(case["num_heads"]),
        int(case["ffn_dim"]),
        int(case["num_layers"]),
        bool(case["causal"]),
    )


def _device_time(event: Any) -> float:
    for attribute in (
        "self_device_time_total",
        "self_cuda_time_total",
        "device_time_total",
        "cuda_time_total",
    ):
        value = getattr(event, attribute, None)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requires an available CUDA device")
    if args.steps <= 0:
        raise ValueError("steps must be positive")

    manifest = load_manifest(args.manifest)
    matches = [case for case in manifest["cases"] if case["id"] == args.case_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one manifest case named {args.case_id!r}")
    case = matches[0]
    config = _config(case)
    dtype = DTYPES[args.dtype]
    device = torch.device("cuda")
    seed = int(manifest["defaults"].get("seeds", [1234])[0])

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(
        config,
        attention_backend=args.attention_backend,
    )
    copy_model_weights(baseline, optimized, strict=True)
    optimized = optimized.to(device=device, dtype=dtype).eval()
    x, valid_mask = generate_random_case(
        config,
        device,
        dtype,
        seed + 200_000,
        float(case.get("padding_ratio", 0.0)),
        float(manifest["defaults"].get("input_scale", 1.0)),
    )

    with torch.inference_mode():
        for _ in range(3):
            optimized(x, valid_mask)
    torch.cuda.synchronize()
    optimized.reset_attention_backend_counts()

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        with torch.inference_mode():
            for _ in range(args.steps):
                with torch.profiler.record_function("optimized_transformer"):
                    optimized(x, valid_mask)
    torch.cuda.synchronize()

    events = list(profiler.key_averages())
    ordered_events = sorted(events, key=_device_time, reverse=True)
    event_payload = [
        {
            "name": event.key,
            "count": event.count,
            "self_device_time_us": _device_time(event),
            "cpu_time_total_us": float(getattr(event, "cpu_time_total", 0.0)),
        }
        for event in ordered_events[:40]
    ]
    custom_events = [
        event
        for event in event_payload
        if "_attention_fwd" in event["name"] or "attention_fwd" in event["name"]
    ]
    backend_counts = dict(optimized.attention_backend_counts)
    custom_expected = backend_counts.get("triton", 0) > 0
    custom_proven = bool(custom_events)

    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(args.trace))

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": case,
        "dtype": args.dtype,
        "backend_requested": args.attention_backend,
        "steps": args.steps,
        "backend_counts": backend_counts,
        "custom_kernel_expected": custom_expected,
        "custom_kernel_profiler_proven": custom_proven,
        "custom_kernel_events": custom_events,
        "top_events": event_payload,
        "trace_path": str(args.trace) if args.trace is not None else None,
        "environment": capture_environment(sys.argv),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "backend_counts": backend_counts,
        "custom_kernel_events": custom_events,
        "saved": str(args.out),
    }, indent=2))
    return 0 if (not custom_expected or custom_proven) else 1


if __name__ == "__main__":
    raise SystemExit(main())
