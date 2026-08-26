# Project Context

## Objective

Deliver a credible Track 3 submission that implements and exercises a custom
GPU kernel for the checked-in Transformer workload, preserves reference
semantics, and reports reproducible target-hardware performance.

## Current architecture

- `torch_transformer_benchmark.py` contains the executable reference model,
  correctness comparison, timing harness, and required
  `UserOptimizedTransformer` integration point.
- `transformer_opt/` owns custom kernels and dispatch policy.
- `benchmarks/official_shapes.json` is the machine-readable validation matrix.
- `benchmarks/reference/manifest.json` fingerprints the benchmark snapshot used
  to establish the implementation contract.
- `tests/` separates portable CPU contract tests from CUDA/Triton tests.
- `docs/results/` contains curated, revision-specific evidence; scratch output
  remains under ignored `results/`.

## Decisions

- Use PyTorch, not TensorFlow, because either framework satisfies the brief.
- Implement fused forward attention in Triton as the primary owned kernel.
- Retain PyTorch SDPA as a strong and safe fallback.
- Keep QKV, output, and FFN matrix multiplications on PyTorch/cuBLAS.
- Optimize inference only unless final organizer material adds backward tests.
- Tune for the available RTX 5070 Ti while keeping explicit portability guards.
- Treat performance as a routing problem: a custom kernel is used only in its
  tested envelope and only where it remains competitive.

## Non-negotiable invariants

- Strict state-dict compatibility with the baseline.
- Exact mask and padded-row semantics.
- The benchmark's per-element OR tolerance is unchanged.
- No dense attention score or causal-mask allocation in the custom kernel.
- No false-green benchmark results or unreported fallback.
- No performance claim without revision- and environment-linked evidence.

## Known environment

The verified local GPU environment is a native WSL virtual environment at
`/home/jared/.venvs/tiktok-techjam-2026`. The repository-local `.venv` is not
the validated runtime. Use `scripts/run-wsl.ps1` from PowerShell or the commands
documented in README.

## Remaining external dependency

The organizer's final shape matrix and any benchmark update announced after the
checked-in early-brief material must be reconciled before final submission.

## Verified implementation status (2026-08-27)

- Custom Triton attention is integrated and profiler-proven on the RTX 5070 Ti.
- The seven-case provisional float32 matrix is 7/7 PASS across five seeds per
  case with zero failed output elements.
- Median end-to-end speedup ranges from 1.138x to 1.566x; geomean is 1.360x.
- Long-attention incremental peak allocation fell from 78 MiB to 22 MiB.
- Low-precision model auto-routing is correctness-first; direct fp16 kernel
  coverage is retained but deep fp16/bf16 fused claims are not made.
- The slower inherited standalone Triton LayerNorm was measured and removed.
