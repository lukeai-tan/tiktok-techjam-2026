#!/usr/bin/env python3
"""Grid benchmark: BaselineTransformer vs UserOptimizedTransformer.

Runs correctness + timing across a grid of shapes (small/large batch, sequence
length, dimension), prints a markdown results table, and saves JSON. Reuses the
official script's model + comparison so results are apples-to-apples with the
judge harness.

    python sweep.py                       # auto device, fp16 on GPU / fp32 on CPU
    python sweep.py --dtype bfloat16
    python sweep.py --compile             # also torch.compile the optimized model
    python sweep.py --quick               # tiny grid for a fast CPU sanity run
    python sweep.py --triton-ln           # enable opt-in Triton fused LayerNorm

On CPU this is only a correctness/plumbing check; the meaningful speedups are on
a CUDA GPU (tensor cores + fused attention).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch

from torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    benchmark_once,
    compare_outputs,
    copy_model_weights,
    generate_random_case,
    maybe_compile,
    warmup_model,
)

RTOL = 0.01   # official defaults
ATOL = 0.001

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

# (batch, seq_len, d_model, heads, ffn_dim, num_layers) -- spans the axes the
# prompt promises: small/large batch, seq, dim.
_GRID = [
    (1,   128,  512,  8, 2048, 6),   # baseline-ish, short seq
    (8,   128,  512,  8, 2048, 6),   # default config
    (32,   64,  512,  8, 2048, 6),   # large batch, short seq (throughput)
    (2,  1024,  512,  8, 2048, 6),   # long sequence (attention-bound)
    (1,  4096,  512,  8, 2048, 6),   # very long sequence
    (4,   256, 1024, 16, 4096, 6),   # wide model (GEMM-bound)
    (2,   512, 2048, 16, 8192, 4),   # very wide model
]

_QUICK_GRID = [
    (2, 64, 128, 8, 512, 2),
    (4, 128, 256, 8, 1024, 2),
]


@dataclass
class Row:
    label: str
    dtype: str
    passed: bool
    failed: int
    total: int
    max_abs: float
    baseline_ms: float
    optimized_ms: float
    speedup: float
    baseline_tok_s: float
    optimized_tok_s: float


def _label(cfg: TransformerConfig) -> str:
    c = "-causal" if cfg.causal else ""
    return (f"B{cfg.batch_size}_S{cfg.seq_len}_d{cfg.d_model}"
            f"_H{cfg.num_heads}_L{cfg.num_layers}{c}")


def _median_ms(model, x, mask, warmup, repeats, rounds, device) -> float:
    warmup_model(model, x, mask, warmup, device)
    samples: List[float] = []
    for _ in range(rounds):
        samples.extend(benchmark_once(model, x, mask, repeats, device))
    return statistics.median(samples)


def run_cfg(cfg: TransformerConfig, dtype: torch.dtype, device: torch.device,
            compile_user: bool, warmup: int, repeats: int, rounds: int,
            padding_ratio: float, seed: int) -> Row:
    cfg.validate()
    baseline = BaselineTransformer(cfg)
    optimized = UserOptimizedTransformer(cfg)
    copy_model_weights(baseline, optimized, strict=True)

    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()
    if compile_user:
        optimized = maybe_compile(optimized, True, "max-autotune")

    # Correctness on a fresh case.
    x, mask = generate_random_case(cfg, device, dtype, seed, padding_ratio, 1.0)
    with torch.inference_mode():
        ref = baseline(x, mask)
        opt = optimized(x, mask)
    acc = compare_outputs(ref, opt, rtol=RTOL, atol=ATOL)

    # Timing on a fixed input.
    xb, mb = generate_random_case(cfg, device, dtype, seed + 999, padding_ratio, 1.0)
    base_ms = _median_ms(baseline, xb, mb, warmup, repeats, rounds, device)
    opt_ms = _median_ms(optimized, xb, mb, warmup, repeats, rounds, device)

    tokens = cfg.batch_size * cfg.seq_len
    return Row(
        label=_label(cfg),
        dtype=str(dtype).replace("torch.", ""),
        passed=acc.passed,
        failed=acc.failed_elements,
        total=acc.total_elements,
        max_abs=acc.max_abs_error,
        baseline_ms=base_ms,
        optimized_ms=opt_ms,
        speedup=base_ms / opt_ms if opt_ms > 0 else float("nan"),
        baseline_tok_s=tokens * 1000.0 / base_ms,
        optimized_tok_s=tokens * 1000.0 / opt_ms,
    )


def print_table(rows: List[Row]) -> None:
    header = (f"| {'shape':<26} | dtype | ok | max_abs | base(ms) | opt(ms) | "
              f"speedup | opt tok/s |")
    sep = "|" + "-" * (len(header) - 2) + "|"
    print(header)
    print(sep)
    for r in rows:
        ok = "PASS" if r.passed else f"FAIL({r.failed})"
        print(f"| {r.label:<26} | {r.dtype:>5} | {ok:>7} | {r.max_abs:>7.1e} | "
              f"{r.baseline_ms:>8.3f} | {r.optimized_ms:>7.3f} | "
              f"{r.speedup:>6.2f}x | {r.optimized_tok_s:>9.0f} |")
    speeds = [r.speedup for r in rows if r.optimized_ms > 0]
    gmean = statistics.geometric_mean(speeds) if speeds else float("nan")
    all_ok = all(r.passed for r in rows)
    print(sep)
    print(f"correctness: {'ALL PASS' if all_ok else 'FAILURES'} | "
          f"geomean speedup: {gmean:.2f}x over {len(rows)} shapes")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=list(_DTYPES), default=None)
    ap.add_argument("--compile", action="store_true", help="torch.compile optimized")
    ap.add_argument("--triton-ln", action="store_true", help="opt-in Triton LayerNorm")
    ap.add_argument("--quick", action="store_true", help="tiny grid (CPU sanity)")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--padding-ratio", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="results/sweep.json")
    args = ap.parse_args()

    if args.triton_ln:
        os.environ["TRANSFORMER_OPT_TRITON_LN"] = "1"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.dtype is not None:
        dtype = _DTYPES[args.dtype]
    else:
        dtype = torch.float16 if device.type == "cuda" else torch.float32

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    grid = _QUICK_GRID if args.quick else _GRID
    warmup = 3 if args.quick else args.warmup
    repeats = 10 if args.quick else args.repeats

    print(f"device={device} dtype={dtype} compile={args.compile} "
          f"triton_ln={args.triton_ln} torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    print()

    rows: List[Row] = []
    for shape in grid:
        b, s, d, h, f, L = shape
        cfg = TransformerConfig(b, s, d, h, f, L, args.causal)
        try:
            row = run_cfg(cfg, dtype, device, args.compile, warmup, repeats,
                          args.rounds, args.padding_ratio, args.seed)
        except RuntimeError as e:  # e.g. OOM on a small GPU -> skip, keep going
            print(f"[skip] {_label(cfg)}: {e}")
            continue
        rows.append(row)

    print()
    print_table(rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    meta = {
        "device": str(device),
        "dtype": str(dtype),
        "compile": args.compile,
        "triton_ln": args.triton_ln,
        "causal": args.causal,
        "padding_ratio": args.padding_ratio,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "rows": [asdict(r) for r in rows],
    }
    with open(args.out, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nsaved {args.out}")
    return 0 if all(r.passed for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
