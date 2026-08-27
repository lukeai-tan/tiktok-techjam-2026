# Technical Report — Transformer Layer GPU Kernel

## 1. Executive summary

This project implements a forward fused-attention GPU kernel in Triton for the
Track 3 PyTorch Transformer benchmark. The kernel performs tiled QK, online
softmax, causal/padding masking, and P@V in one launch without storing the
quadratic attention matrix.

On the NVIDIA GeForce RTX 5070 Ti, all seven provisional float32 cases passed
the checked-in executable tolerance across five seeds each. The run covered
13,117,440 output elements with zero failures and measured a **1.501x
geometric-mean end-to-end speedup**, ranging from 1.230x to 1.752x.

Both organizer benchmark downloads are now checksum-frozen. The untouched
PyTorch default six-layer case also passed 5/5 trials with zero failed elements
and measured 1.411x median speedup. A fail-closed matrix then translated every
feasible shape signal from both downloads through that untouched PyTorch
harness: 28/28 executable cases passed with zero failures across 459,776,000
elements. The source-designated 100,000-token quadratic stress case was
preflight-skipped and was not counted as a pass. The final PyTorch evaluator
shape list is still unavailable, so these results remain evidence for all
published inputs rather than a claim about unpublished test cases.

## 2. Executable contract

The benchmark implements a multi-layer pre-LayerNorm Transformer with separate
Q/K/V/output projections, exact GELU, optional causal attention, prefix padding,
and zeroed invalid output rows.

For every output element, the executable rule is:

~~~text
abs(optimized - reference) <= 0.001
OR
abs(optimized - reference) <= 0.01 * abs(reference)
~~~

This is stricter than the Track 3 prose values of 0.002 absolute and 0.02
relative. The executable rule governed all implementation and validation.

Organizer download provenance and unresolved evaluator questions are recorded
in docs/ORGANIZER_INPUTS.md, docs/REQUIREMENTS.md, and
benchmarks/reference/organizer_downloads.json. The older
benchmarks/reference/manifest.json remains frozen because it is part of the
existing result-artifact fingerprint.

## 3. Environment

| component | measured target |
| --- | --- |
| CPU | AMD Ryzen 9 9950X, 16 cores / 32 logical processors |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| compute capability | 12.0 |
| GPU memory | 16,303 MiB |
| NVIDIA driver | 610.88 |
| OS | Windows 11, build 26200, AMD64 |
| Python | 3.12.10 |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime | 13.0 |
| Triton | 3.7.1 |
| Triton distribution | triton-windows 3.7.1.post27 |
| disk during run | C: 931 GiB total, 443 GiB free |
| float32 policy | high matmul precision, TF32 enabled |

The native environment used the official CPython 3.12.10 runtime, PyTorch CUDA
13.0 wheel, and the Triton Windows package. The temporary portable interpreter
was supplemented with the matching official Python development headers for
Triton's first-use driver module; a normal CPython installation already
contains those files.

The curated result JSON also stores CPU model/count, Python, OS, CUDA
capability, driver, disk bytes, Git state, command, raw samples, fingerprint
schema, and an implementation-content SHA-256 so an uncommitted benchmark
cannot be mistaken for the base commit.

## 4. Baseline and bottleneck analysis

The reference attention explicitly allocates B x H x S x S scores, applies
softmax, and launches a second matmul for context. Its memory traffic and
intermediate storage grow quadratically with sequence length. Projection and
FFN work remains GEMM-heavy and is best left to PyTorch/cuBLAS.

The captured causal-padding profile for five two-layer forwards recorded:

| event | count | self device time |
| --- | ---: | ---: |
| optimized Transformer range | 5 | 2,521.7 us |
| addmm | 40 | 1,596.0 us |
| custom _attention_fwd | 10 | 352.0 us |
| native LayerNorm | 25 | 120.7 us |
| GELU | 10 | 62.2 us |
| residual add | 20 | 50.1 us |

The ten custom events exactly match five forwards times two layers. This proves
the repository-owned kernel ran; dispatch counters alone were not used as
proof. Packed QKV lowers the expected projection/linear `addmm` count from 60
to 40 across those five forwards.

## 5. Kernel implementation

### 5.1 Layout

Q/K/V stay in projection-friendly BSHD layout. The baseline creates three BHSD
contiguous copies; the custom kernel consumes strides directly and returns BSHD
for a direct reshape into the output projection.

### 5.2 Packed QKV projection

For the measured eager CUDA float32 path through `d_model=512`, the model
caches concatenated views of the existing Q/K/V weights and biases and uses one
vendor `F.linear` call instead of three. The resulting `[B,S,3,H,D]` tensor is
unbound into strided Q/K/V views consumed by the selected backend. Cache
signatures detect parameter mutation, loading, and device/dtype changes;
derived tensors are non-persistent, so the baseline state dict remains
unchanged.

Target-device microbenchmarks found the combined projection bit-identical for
the tested float32 shapes and beneficial through width 512, but neutral at
width 1024. The dispatcher therefore leaves the wide model, training,
low-precision, CPU, and compiled paths on separate projections.

### 5.3 Online softmax

Each program owns a query tile and one batch/head pair. It streams K/V tiles
while maintaining fp32 running maximum, normalization sum, and weighted-value
accumulator. The rescaling formula makes each tile numerically compatible with
the prior tiles, so no score or probability matrix is stored.

### 5.4 Masks

Sequence bounds, valid-key prefixes, and causal key <= query bounds are combined
inside the score tile. The all-masked case is explicitly finite and returns
zero. The causal-padding custom path never builds the dense combined mask used
by the prior SDPA integration.

### 5.5 Numerical matching

The kernel deliberately reproduces the reference's low-precision score tensor
and scaling roundings before converting to fp32 softmax state. Float32 dot
products follow the benchmark's TF32 toggle. This was necessary for deep-stack
agreement; mathematically reasonable fused implementations can still fail an
elementwise benchmark when their rounding points differ.

### 5.6 Dispatch

The custom envelope requires CUDA compute capability 8.0+, inference, sequence
length at most 8192, final stride 1, float32/fp16, and head dimensions
16/32/64/128. Forced Triton rejects unsupported inputs. Auto uses guarded
fallbacks and exposes actual backend counts. Controlled alternating target-GPU
measurements showed SDPA was 12%-13% faster for the launch-bound, unmasked,
non-causal float32 corner with sequence <=128 and head dimension <=32. Auto
routes those two provisional cases to SDPA. The supplied five-trial harness
also exposed rare custom-kernel tolerance misses after six layers for causal
attention and batch sizes above eight; those deep-stack regimes now use SDPA,
which passed the same comparator and remained faster than the baseline. The
organizer-default non-causal B8 path stays on Triton, as do the validated
smaller masked, long, and wider-head regimes.

The primary end-to-end route is float32. Direct fp16 attention passes, but fp16
and bf16 fused differences compound in deep stacks under the strict executable
tolerance. Automatic model execution therefore uses exact reference-style math
for low precision. This avoids the false claim that a fast but numerically
rejected path is supported.

Full algorithm and launch details are in docs/KERNEL_DESIGN.md.

## 6. Benchmark method

- Organizer proof: untouched downloaded PyTorch parser, baseline, comparator,
  and timer with only `UserOptimizedTransformer` injected.
- Organizer default: B=8, S=128, d_model=512, heads=8, FFN=2048, six layers,
  float32, five accuracy trials, 20 warmups, and 3 x 100 timing samples.
- Rigorous organizer validation: six direct PyTorch variants plus all 11
  feasible TensorFlow compact shapes translated to float32 and float16; five
  accuracy trials and 2 x 10 timing samples per executable case, each isolated
  in a fresh subprocess.
- Resource accounting: the TensorFlow benchmark's designated B=32, S=100000,
  d=1024, heads=16 stress case is recorded as `SKIPPED_RESOURCE`, never PASS.
- Manifest: benchmarks/official_shapes.json, status provisional.
- Dtype: float32.
- Accuracy: five seeds per case, checked before timing.
- Timing input: fixed and separate from accuracy inputs.
- Warm-up: 10 calls per model.
- Measurement: 30 CUDA-event samples per round, three rounds.
- Bias control: baseline/optimized order alternates by round.
- Reporting: all 90 raw samples per model, median/mean/p90/min and throughput.
- Failure accounting: explicit PASS, FAIL, OOM, or ERROR; zero-case runs fail.
- Backend evidence: correctness and timing dispatch counts stored per case.

Random-data generation, model construction, and first-use Triton compilation
are excluded from steady-state forward latency for both sides.

## 7. Results

### Untouched organizer PyTorch default

The downloaded script with SHA-256
`1bd12523657f338c09b53f0bb9052d9d16f728a71bd22bc8298567e1a4d78c22`
ran unchanged through `benchmarks/run_organizer_torch.py`:

| metric | baseline | optimized |
| --- | ---: | ---: |
| median latency | 1.9456 ms | 1.3788 ms |
| mean latency | 1.9843 ms | 1.4367 ms |
| p90 latency | 2.1910 ms | 1.5705 ms |
| throughput | 526,307 token/s | 742,649 token/s |

- Median speedup: 1.411x.
- Accuracy: 5/5 PASS, 0 failed out of 2,621,440 elements.
- Maximum absolute error: 0.000990123.
- Optimized attention dispatch: Triton 1,950; SDPA 0; reference 0.

### Provisional cross-shape matrix

| case | B | S | d / heads | layers | mask | baseline ms | optimized ms | speedup |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| tiny-overhead | 1 | 32 | 64 / 4 | 2 | none | 0.484 | 0.312 | 1.552x |
| medium-throughput | 8 | 128 | 256 / 8 | 2 | none | 0.519 | 0.318 | 1.630x |
| medium-padding | 4 | 256 | 512 / 8 | 2 | 30% padding | 0.653 | 0.490 | 1.332x |
| long-causal | 2 | 512 | 512 / 8 | 2 | causal | 0.696 | 0.462 | 1.507x |
| long-causal-padding | 2 | 512 | 512 / 8 | 2 | causal + 30% padding | 0.923 | 0.527 | 1.752x |
| long-attention | 1 | 1024 | 512 / 8 | 2 | none | 0.813 | 0.518 | 1.570x |
| wide-model | 2 | 128 | 1024 / 16 | 1 | none | 0.256 | 0.208 | 1.230x |

Summary:

- 7 requested, 7 completed, 7 PASS, 0 FAIL, 0 OOM, 0 ERROR.
- 35 accuracy trials and 13,117,440 checked elements.
- 0 failed elements.
- Maximum absolute error: 0.000992358.
- Geometric-mean speedup: 1.501x.
- Timing dispatch: SDPA for tiny/medium unmasked cases; Triton for all five
  masked, causal, long, or wider-head cases; no reference timing fallback.

### Supplied-contract shape validation

The isolated exact-harness matrix produced:

- 29 requested entries: 28 executable and one source-authorized resource skip;
- 28/28 executable PASS and 0/459,776,000 failed elements across 140 trials;
- batch sizes 1, 4, 8, 16, 128, and 10,000;
- sequence lengths 32, 128, and 1,024;
- widths 32, 128, 512, and 1,024; heads 1, 2, 4, 8, and 16;
- float32, float16, bfloat16, causal, non-causal, and prefix-padding coverage;
- overall geometric-mean speedup 1.258x and float32-only geomean 1.509x; and
- aggregate dispatch counts Triton 672, SDPA 1,848, reference 2,184.

The matrix uses the selected PyTorch executable tolerance of atol=0.001 OR
rtol=0.01, which is stricter than the TensorFlow download's defaults. Its
machine-readable policy and full stdout/evidence are stored with SHA-256
fingerprints; a crash, OOM, numerical failure, unauthorized skip, or empty run
returns nonzero.

Incremental peak CUDA allocation fell from 78 MiB to 22 MiB (71.8%) in the
long-attention case. Long-causal, causal-padding, and medium-padding cases
reduced the measured incremental peak by 50.3%, 54.4%, and 31.3%, respectively.
Tiny/wide cases are dominated by model outputs and GEMM work, so their measured
incremental peak did not change. Packed QKV storage is prepared before this
incremental measurement and adds about 6 MiB for two float32 d_model=512
layers.

The largest gains occur when attention or mask materialization is a larger share
of the block. Wide-model performance is dominated by projection/FFN GEMMs, so
attention fusion has less leverage.

## 8. Rejected and deferred optimizations

The inherited standalone Triton LayerNorm was measured before removal:

| rows x width | native CUDA | custom Triton | native/custom |
| --- | ---: | ---: | ---: |
| 1024 x 512 | 0.00832 ms | 0.01629 ms | 0.511x |
| 2048 x 512 | 0.01082 ms | 0.01562 ms | 0.693x |
| 256 x 1024 | 0.00758 ms | 0.01664 ms | 0.456x |

It was numerically accurate but consistently slower and did not remove a
neighboring launch, so it was retired. A residual-add + LayerNorm fusion was
not implemented: native LayerNorm was a small profiler share, while the added
support and numerical risk was not justified by the available ceiling.

QKV/output/FFN math remains in vendor GEMMs. QKV packing reduces three launches
to one without replacing cuBLAS; custom output/FFN GEMMs were rejected because
the profile did not justify competing with mature matrix-multiplication code.
A causal loop-frontier prune and alternate tile/stage policies were also tested
and rejected after they failed to improve the full end-to-end matrix.

## 9. AI-assisted development

The initial repository report attributed the prototype SDPA and optional
LayerNorm work to Claude Code. This revision was audited and implemented with
OpenAI Codex, which was used to:

- trace the checked-in challenge and benchmark contract;
- identify the false-green sweep and unverified-kernel gaps;
- design and implement the Triton online-softmax kernel and dispatcher;
- bootstrap and diagnose both WSL and native Windows CUDA/Triton toolchains;
- generate correctness, negative-path, matrix, and profiler checks;
- iterate from measured numerical/performance failures;
- add and invalidate the measured packed-QKV inference cache;
- reject slower LayerNorm, causal-pruning, and launch-retuning paths; and
- produce provenance-linked documentation and demo instructions.

AI output was not accepted as evidence by itself. Claims in this report come
from executable tests, raw CUDA-event samples, profiler events, and captured
environment metadata.

Human/team contribution attribution is not established by repository evidence;
the submitter must add any additional participant attribution to Devpost if
applicable.

## 10. Limitations and next work

- Reconcile the final PyTorch evaluator matrix and any later benchmark revision
  when published; both currently supplied scripts are already checksum-frozen.
- Retest and retune on any evaluation GPU; launch policy is measured only on the
  RTX 5070 Ti.
- Packed QKV consumes bounded derived-weight memory and is deliberately limited
  to the measured eager CUDA float32 envelope through d_model=512.
- Add backward kernels only if training becomes part of the official contract.
- Explore residual/normalization fusion only after a new profile shows a larger
  end-to-end ceiling.
- A public demo video and Devpost submission remain external human deliverables;
  DEMO_RUNBOOK.md gives the verified recording sequence.

## 11. Evidence

- Matrix: docs/results/rtx-5070-ti-2026-08-27.json
- Profiler: docs/results/rtx-5070-ti-2026-08-27-profile.json
- Untouched organizer default:
  docs/results/rtx-5070-ti-2026-08-27-organizer-default.json
- Supplied-contract validation matrix:
  docs/results/rtx-5070-ti-2026-08-27-organizer-validation.json
- Organizer inputs: docs/ORGANIZER_INPUTS.md
- Organizer checksums: benchmarks/reference/organizer_downloads.json
- Requirements: docs/REQUIREMENTS.md
- Kernel design: docs/KERNEL_DESIGN.md
- Track 3 compliance: docs/TRACK3_COMPLIANCE.md
- Demo procedure: DEMO_RUNBOOK.md
