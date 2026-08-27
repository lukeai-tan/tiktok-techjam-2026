# Project Context

## Objective

Deliver a credible Track 3 submission that implements and exercises a custom
GPU kernel for the checked-in Transformer workload, preserves reference
semantics, and reports reproducible target-hardware performance.

## Current architecture

- `torch_transformer_benchmark.py` contains the executable reference model,
  correctness comparison, timing harness, and required
  `UserOptimizedTransformer` integration point.
- `benchmarks/torch_transformer_benchmark.py` and
  `benchmarks/tensorflow_transformer_benchmark.py` are byte-preserved organizer
  downloads; `benchmarks/run_organizer_torch.py` runs the selected submission
  through the untouched PyTorch harness.
- `transformer_opt/` owns custom kernels and dispatch policy.
- `benchmarks/official_shapes.json` is the machine-readable validation matrix.
- `benchmarks/reference/manifest.json` fingerprints the benchmark snapshot used
  to establish the implementation contract.
- `benchmarks/reference/organizer_downloads.json` records the supplied-file
  checksums, framework differences, and organizer-default dimension signals.
- `benchmarks/organizer_validation_matrix.json` and
  `benchmarks/run_organizer_validation.py` expand those signals into isolated,
  fail-closed executions through the untouched selected-framework harness.
- `tests/` separates portable CPU contract tests from CUDA/Triton tests.
- `docs/results/` contains curated, revision-specific evidence; scratch output
  remains under ignored `results/`.

## Decisions

- Use PyTorch, not TensorFlow, because either framework satisfies the brief.
- Implement fused forward attention in Triton as the primary owned kernel.
- Retain PyTorch SDPA as a strong and safe fallback.
- Keep QKV, output, and FFN matrix multiplications on PyTorch/cuBLAS, while
  packing Q/K/V weights into one measured inference GEMM for d_model <= 512.
- Optimize inference only unless final organizer material adds backward tests.
- Tune for the available RTX 5070 Ti while keeping explicit portability guards.
- Treat performance as a routing problem: a custom kernel is used only in its
  tested envelope and only where it remains competitive.
- Route short unmasked float32 heads <= 32 to SDPA on the measured target;
  retain Triton for validated default/smaller masked regimes, and use SDPA when
  six-layer causal or batch-above-8 execution would exceed strict tolerance.

## Non-negotiable invariants

- Strict state-dict compatibility with the baseline.
- Exact mask and padded-row semantics.
- The benchmark's per-element OR tolerance is unchanged.
- No dense attention score or causal-mask allocation in the custom kernel.
- No false-green benchmark results or unreported fallback.
- No performance claim without revision- and environment-linked evidence.

## Known environment

The current curated GPU evidence was captured under native Windows 11 with
Python 3.12.10, PyTorch 2.13.0+cu130, and Triton 3.7.1 on the RTX 5070 Ti.
The earlier WSL environment is no longer installed on this host. Use the native
commands documented in README; `scripts/run-wsl.ps1` remains an optional route
for machines with an Ubuntu WSL distribution.

## Remaining external dependency

The two benchmark files are now reconciled. The organizer still needs to supply
the final evaluator matrix for the selected PyTorch path (including dtypes,
padding/causal modes, and any backward or timing rules) and any update announced
after these downloads. The TensorFlow file's different defaults are documented
but are not silently treated as the PyTorch evaluator matrix.

## Verified implementation status (2026-08-27)

- Custom Triton attention is integrated and profiler-proven on the RTX 5070 Ti.
- The untouched organizer PyTorch default case is 5/5 PASS with zero failed
  elements and 1.411x median speedup; all 1,950 optimized attention calls used
  Triton.
- The source-derived exact-harness matrix is 28/28 executable PASS with zero
  failures across 459,776,000 elements; its one source-authorized 100000-token
  resource skip is not counted as a pass. Overall geomean is 1.262x and the
  float32 subset geomean is 1.492x.
- The seven-case provisional float32 matrix is 7/7 PASS across five seeds per
  case with zero failed output elements.
- Median end-to-end speedup ranges from 1.230x to 1.752x; geomean is 1.501x.
- Long-attention incremental peak allocation fell from 78 MiB to 22 MiB.
- Packed QKV reduces three projection GEMMs to one for the measured eager-fp32
  path through d_model=512 without changing parameter names or state-dict keys.
- Low-precision model auto-routing is correctness-first; direct fp16 kernel
  coverage is retained but deep fp16/bf16 fused claims are not made.
- The slower inherited standalone Triton LayerNorm was measured and removed.
