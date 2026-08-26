# Transformer GPU Kernel Requirements

Status: implementation contract based on the checked-in Track 3 brief and
benchmark as of 2026-08-27. The organizer's final shape combinations have not
yet been published in this repository; `benchmarks/official_shapes.json` is
therefore explicitly provisional.

## Source-of-truth order

1. A newly downloaded organizer benchmark and final shape list, once available.
2. The checked-in benchmark snapshot identified by
   `benchmarks/reference/manifest.json`.
3. Track 3 in `docs/hackathon-details.md`, lines 674-780.
4. Reproduced behavior on the target GPU.
5. Design and explanatory documentation in this repository.

If these sources conflict, update this file and the reference manifest before
tuning kernels. Never relax a checked-in correctness rule to match a looser
description.

## Required behavior

The selected PyTorch implementation is the pre-LayerNorm Transformer in
`torch_transformer_benchmark.py`:

```text
for each block:
  x = x + MHA(LayerNorm(x), valid_token_mask, causal)
  x = x + Linear(GELU(Linear(LayerNorm(x)), approximate="none"))
final output = LayerNorm(x)
```

- Multi-head attention uses separate Q, K, V, and output projections.
- Invalid key positions cannot be attended to.
- Causal attention cannot attend to future positions.
- Invalid output rows are zero after every block and after the final norm.
- The optimized model must accept the same inputs, preserve parameter names,
  and load the reference state dict with `strict=True`.
- Training/backward support is not required by the current inference harness.

## Correctness contract

For every output element, the checked-in benchmark requires:

```text
abs(optimized - reference) <= 0.001
OR
abs(optimized - reference) <= 0.01 * abs(reference)
```

The Track 3 prose states looser bounds (`abs < 0.002`, `relative < 0.02`). This
project intentionally targets the stricter executable benchmark rule. A case
passes only when there are zero failing elements. NaN and infinity mismatches
fail.

Correctness validation must cover:

- CPU float32 semantic regressions;
- CUDA float32 for the primary end-to-end custom path, float16 for direct-kernel
  validation, and bfloat16 for the exact fallback;
- causal and non-causal attention;
- no mask, all-valid mask, partial prefix padding, and minimum one-token prefix;
- multiple seeds and input scales;
- sequence lengths on and immediately around tile boundaries;
- custom-kernel positive dispatch and unsupported-shape fallback dispatch.

## Kernel and dispatch contract

- At least one repository-owned GPU kernel must execute in the submitted path.
- The primary custom kernel is forward scaled dot-product attention implemented
  in Triton with online softmax and fp32 softmax accumulation.
- The custom path must not materialize a `[B,H,S,S]` score tensor or dense
  causal mask.
- PyTorch SDPA is the safe fallback and comparison backend.
- `auto` dispatch may select custom Triton only inside its tested support
  envelope; forced `triton` mode must fail clearly when unsupported.
- Backend choice must be inspectable in tests/results. Import or compilation
  errors may not be swallowed as successful custom execution.
- The benchmark-default float32 custom path follows the benchmark's TF32 toggle
  (TF32 when enabled, IEEE otherwise). Automatic end-to-end low-precision runs
  use the explicit reference-style path because fused FP16/BF16 differences
  compounded beyond the checked-in benchmark's stricter tolerance in
  target-GPU deep-stack tests. CPU, unsupported layouts/head widths, and training with
  gradients fall back unless explicitly added and tested.

## Benchmark integrity

- Correctness runs before performance timing.
- GPU timing uses CUDA events after warm-up and synchronization.
- Baseline and optimized ordering alternates by round.
- Every requested case has an explicit `PASS`, `FAIL`, `OOM`, or `ERROR` result.
- Unexpected runtime errors fail the run. OOM may be reported separately but is
  never converted into a pass.
- A run with zero completed cases fails.
- Result artifacts include the code revision, dirty state, command, raw timing
  samples, framework/runtime versions, GPU name/capability, driver, and case
  definition.
- Performance claims are tied to committed curated JSON, not console excerpts.

## Target and dependencies

Primary tuning target:

- NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, 16,303 MiB VRAM;
- WSL2 Linux;
- Python 3.14.4;
- PyTorch 2.13.0+cu130;
- CUDA runtime 13.0;
- Triton 3.7.1.

Portable CPU tests remain required. Other CUDA devices use the same guarded
dispatcher and may fall back to SDPA until measured.

## Acceptance criteria

- **AC-1:** A direct Triton attention kernel executes on the target GPU and
  matches the reference across the declared support envelope.
- **AC-2:** End-to-end optimized Transformer outputs pass the stricter checked-in
  tolerance for every executed matrix case.
- **AC-3:** Dispatch tests prove both custom selection and safe fallback; no
  silent fallback is reported as custom execution.
- **AC-4:** Sweep/matrix error accounting cannot report success for unexpected
  exceptions, OOM-only runs, or zero completed cases.
- **AC-5:** Target-GPU evidence includes raw timings, environment metadata,
  backend counts, and profiler evidence that the custom kernel ran.
- **AC-6:** README, kernel design, technical report, and demo runbook contain no
  placeholders or unverified performance statements.

## Deliverables

- Public, structured, commented repository.
- Reproduction instructions and limitations in README.
- Custom GPU kernel implementation and guarded integration.
- Correctness/performance results from the target machine.
- Technical report including environment, optimization rationale, AI tooling,
  measurements, and limitations.
- Demo-video runbook for an end-to-end public walkthrough.

## Open organizer questions

- Final mandatory shape combinations and dtypes.
- Whether timing includes model construction/compilation or steady-state forward
  only.
- Whether gradients/backward are evaluated.
- Exact dependency and source-file modification restrictions.
- Whether a newer organizer benchmark supersedes the checked-in snapshot.

These unknowns do not block implementing and validating the current contract,
but they block claiming final organizer-matrix completeness.
