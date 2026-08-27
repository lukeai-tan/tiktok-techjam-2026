# Transformer GPU Kernel Requirements

Status: implementation contract reconciled with the two participant-supplied
organizer downloads received on 2026-08-27 and the 14-row Track 3 test-shape
table published in the organizer document's Section 3.7. PyTorch is the
selected framework. `benchmarks/final_evaluator_shapes.json` preserves the
published row order and records every execution assumption required because
the table omits framework, dtype, padding, tolerance, timing, and backward
requirements. `benchmarks/official_shapes.json` remains a project-owned
held-out matrix rather than the final organizer matrix.

## Source-of-truth order

1. The untouched organizer PyTorch download identified by
   `benchmarks/reference/organizer_downloads.json`.
2. The organizer's final shape table identified by
   `benchmarks/final_evaluator_shapes.json` and its live source metadata.
3. The older result-linked snapshot identified by
   `benchmarks/reference/manifest.json`.
4. Track 3 in `docs/hackathon-details.md`, lines 674-780.
5. Reproduced behavior on the target GPU.
6. Design and explanatory documentation in this repository.

If these sources conflict, update this file and the reference manifest before
tuning kernels. Never relax a checked-in correctness rule to match a looser
description.

## Organizer download reconciliation

- Untouched PyTorch SHA-256:
  `1bd12523657f338c09b53f0bb9052d9d16f728a71bd22bc8298567e1a4d78c22`.
- Untouched TensorFlow SHA-256:
  `00e99b6e1d19e961039b66eb3d3c055b36cc50f0436da2558f5f1fbe292ef798`.
- `benchmarks/run_organizer_torch.py` injects only the submitted class into the
  untouched PyTorch harness; tests AST-compare protected baseline definitions.
- The TensorFlow download is retained as an alternative-framework and shape
  cross-check. It is not treated as a second implementation requirement.
- `benchmarks/run_organizer_validation.py` translates every feasible published
  dimension signal into the selected PyTorch harness. Its 28 executable cases
  all pass; the TensorFlow source's designated 100000-token quadratic stress
  case is the only authorized resource skip and is never counted as a pass.
- The organizer document now publishes 14 exact rows covering batch size,
  QKV/model width, heads, sequence length, layers, causal mode, and FFN width.
  Thirteen rows are executable on the selected PyTorch path. Row 14 is the
  exact 100000-token stress dimensions that the supplied TensorFlow harness
  explicitly permits resource-preflighting; it remains an authorized skip and
  is not counted as a pass.
- The final table does not state dtype or padding. Until the organizer says
  otherwise, final-shape validation uses the selected PyTorch harness defaults:
  float32 and no padding, with the stricter executable comparator below.

The files are not interchangeable evaluator specifications. PyTorch defaults
to one configurable float32 case and the stricter 0.001/0.01 OR rule;
TensorFlow defaults to a float16 compact dimension sweep and the prose-level
0.002/0.02 OR rule. The final table supplies dimensions but does not resolve
those framework-level differences. See `docs/ORGANIZER_INPUTS.md` for the exact
differences and remaining clarifications.

## Required behavior

The selected PyTorch implementation is the pre-LayerNorm Transformer in the
untouched `benchmarks/torch_transformer_benchmark.py`; the optimized submission
copy is `torch_transformer_benchmark.py`:

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
- CUDA float32 for end-to-end auto routing and forced-custom coverage, float16
  for direct-kernel validation, and bfloat16 for the exact fallback;
- causal and non-causal attention;
- no mask, all-valid mask, partial prefix padding, and minimum one-token prefix;
- multiple seeds and input scales;
- sequence lengths on and immediately around tile boundaries;
- every executable row in the final organizer shape table;
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
- native Windows 11, build 26200, NVIDIA driver 610.88;
- Python 3.12.10;
- PyTorch 2.13.0+cu130;
- CUDA runtime 13.0;
- Triton 3.7.1 from `triton-windows==3.7.1.post27`.

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
- **AC-7:** The downloaded PyTorch/TensorFlow files match frozen hashes, and the
  optimized class passes the untouched organizer PyTorch harness without
  modifying its baseline, comparator, argument parsing, or timing code.
- **AC-8:** Every feasible case derived from the two supplied contracts passes
  in an isolated process, and skip accounting accepts only the exact
  source-authorized 100000-token stress case.
- **AC-9:** The 14 published final-shape rows are preserved in source order;
  all 13 executable rows pass the untouched selected PyTorch comparator in
  isolated processes, and the exact 100000-token resource skip is excluded
  from the pass count.

## Deliverables

- Public, structured, commented repository.
- Reproduction instructions and limitations in README.
- Custom GPU kernel implementation and guarded integration.
- Correctness/performance results from the target machine.
- Technical report including environment, optimization rationale, AI tooling,
  measurements, and limitations.
- Byte-preserved organizer downloads, checksum manifest, and exact-harness
  PyTorch default plus full source-derived validation evidence.
- Track 3 requirement-to-evidence compliance matrix.
- Demo-video runbook for an end-to-end public walkthrough.

## Open organizer questions

- Required dtypes and padding modes for the final 14-row shape table.
- Whether timing includes model construction/compilation or steady-state forward
  only.
- Whether gradients/backward are evaluated.
- Exact dependency and source-file modification restrictions.
- Whether a later workshop/evaluator revision supersedes either supplied file.
- Whether the current live download attachments are byte-identical to the two
  files frozen on 2026-08-27; the attachment UI was visible on 2026-08-28 but
  did not expose bytes for a fresh checksum through the read-only browser path.

These unknowns do not block validating the published dimensions under the
selected PyTorch defaults, but they block claiming that every unstated
evaluator policy is known.
