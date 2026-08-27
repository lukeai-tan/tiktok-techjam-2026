# Technical Report — Transformer Layer GPU Kernel

## 1. Executive summary

This project implements a forward fused-attention GPU kernel in Triton for the
Track 3 PyTorch Transformer benchmark. The kernel performs tiled QK, online
softmax, causal/padding masking, and P@V in one launch without storing the
quadratic attention matrix.

On the NVIDIA GeForce RTX 5070 Ti, all 13 executable rows in the
organizer-published final shape table passed the checked-in executable tolerance
across five seeds each. The run covered 938,885,120 output comparisons with zero
failures and measured a **1.427x geometric-mean end-to-end speedup**, ranging
from 1.009x to 4.640x. The source-authorized 100,000-token resource row was
preflight-skipped and was not counted as a pass.

Both organizer benchmark downloads are now checksum-frozen. The untouched
PyTorch default six-layer case also passed 5/5 trials with zero failed elements
and measured 1.408x median speedup. A fail-closed matrix then translated every
feasible shape signal from both downloads through that untouched PyTorch
harness: 28/28 executable cases passed with zero failures across 459,776,000
elements. The source-designated 100,000-token quadratic stress case was
preflight-skipped and was not counted as a pass. The final table publishes
dimensions but omits dtype, padding, timing, tolerance, and backward policy;
the final-shape evidence therefore records the selected PyTorch assumptions.

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
| CPU | AMD Ryzen 7 9850X3D, 8 cores / 16 logical processors |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| compute capability | 12.0 |
| GPU memory | 16,303 MiB |
| NVIDIA driver | 616.56 |
| OS | Windows 11, build 26200, AMD64 |
| Python | 3.12.10 |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime | 13.0 |
| Triton | 3.7.1 |
| Triton distribution | triton-windows 3.7.1.post27 |
| disk during run | C: 931 GiB total, about 409 GiB free |
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
| optimized Transformer range | 5 | 6,333.0 us |
| addmm | 40 | 2,048.6 us |
| custom _attention_fwd | 10 | 3,663.9 us |
| native LayerNorm | 25 | 134.9 us |
| GELU | 10 | 64.6 us |
| residual add | 20 | 55.0 us |

The ten custom events exactly match five forwards times two layers. This proves
the repository-owned kernel ran; dispatch counters alone were not used as
proof. Packed QKV lowers the expected projection/linear `addmm` count from 60
to 40 across those five forwards.

The final-shape profile then isolated row 10 (`B=64`, `S=128`, `d_model=128`,
two heads, four layers). Before EXP-001, `_attention_fwd` consumed 30,324.486 us
across 40 launches and dominated 79.6% of recorded GPU time. The accepted short
`head_dim=64` tile reduced that event to 3,205.548 us, an 89.43% reduction; all
40 launches remained Triton. The integrated end-to-end row improved to 1.701x.

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
and scaling roundings before converting to fp32 softmax state. Non-causal
float32 dot products follow the benchmark's TF32 toggle; causal custom
attention always uses IEEE fp32 dot products. Final-shape testing found rare
four-layer causal TF32 misses, demonstrating that mathematically reasonable
fused implementations can still fail an elementwise benchmark when their
rounding points differ.

### 5.6 Dispatch

The custom envelope requires CUDA compute capability 8.0+, inference, sequence
length at most 8192, final stride 1, float32/fp16, and head dimensions
16/32/64/128. Forced Triton rejects unsupported inputs. Auto uses guarded
fallbacks and exposes actual backend counts. Controlled alternating target-GPU
measurements showed SDPA was 12%-13% faster for the launch-bound, unmasked,
non-causal float32 corner with sequence <=128 and head dimension <=32. Auto
routes that held-out corner to SDPA. The supplied five-trial harness
also exposed rare custom-kernel tolerance misses after six layers for causal
attention and batch sizes above eight; those deep-stack regimes now use SDPA,
which passed the same comparator and remained faster than the baseline. The
organizer-default non-causal B8 path stays on Triton, as do the validated
smaller masked, long, and wider-head regimes. Low precision, unsupported head
widths, and causal batches above 128 use explicit reference-style math in the
multi-layer model after final-shape testing exposed stricter failure modes.

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
- Final organizer dimensions: `benchmarks/final_evaluator_shapes.json`, 13
  executable rows plus the exact authorized resource row. Because the source
  omits execution policy, the run records float32, no padding, the stricter
  PyTorch comparator, warmup 3, repeats 10, and two alternating timing rounds.
- Held-out manifest: `benchmarks/official_shapes.json`, project-owned status.
  It uses float32, five accuracy seeds, 10 warmups, 30 CUDA-event samples per
  round, three alternating rounds, and retains all 90 raw samples per model.
- Correctness is checked before timing; timing inputs are fixed and separate
  from accuracy inputs.
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
| median latency | 1.9057 ms | 1.3533 ms |
| mean latency | 2.4338 ms | 1.6646 ms |
| p90 latency | 4.5151 ms | 2.2364 ms |
| throughput | 537,327 token/s | 756,680 token/s |

- Median speedup: 1.408x.
- Accuracy: 5/5 PASS, 0 failed out of 2,621,440 elements.
- Maximum absolute error: 0.00100136.
- Optimized attention dispatch: Triton 1,950; SDPA 0; reference 0.

### Organizer-published final shape matrix

The published rows were executed through the untouched PyTorch comparator in
isolated processes. Dtype, padding, and timing are the recorded assumptions in
Section 6, not claims about omitted organizer policy.

| row | B | S | d / heads | layers | baseline ms | optimized ms | speedup | backend |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 64 | 128 | 128 / 4 | 4 | 1.4976 | 1.2421 | 1.206x | Triton |
| 2 | 1 | 128 | 128 / 4 | 4 | 1.8304 | 1.1358 | 1.612x | Triton |
| 3 | 4 | 128 | 128 / 4 | 4 | 1.7116 | 0.8853 | 1.933x | Triton |
| 4 | 16 | 128 | 128 / 4 | 4 | 1.5104 | 0.9646 | 1.566x | Triton |
| 5 | 128 | 128 | 128 / 4 | 4 | 2.9682 | 2.5948 | 1.144x | Triton |
| 6 | 10,000 | 128 | 128 / 4 | 4 | 528.1644 | 494.1977 | 1.069x | reference |
| 7 | 64 | 128 | 32 / 4 | 4 | 1.4314 | 1.3786 | 1.038x | reference |
| 8 | 64 | 128 | 1,024 / 4 | 4 | 17.2808 | 17.1294 | 1.009x | reference |
| 9 | 64 | 128 | 128 / 1 | 4 | 1.3112 | 1.1866 | 1.105x | Triton |
| 10 | 64 | 128 | 128 / 2 | 4 | 1.5510 | 0.9120 | 1.701x | Triton |
| 11 | 64 | 128 | 128 / 16 | 4 | 6.6684 | 6.4690 | 1.031x | reference |
| 12 | 64 | 32 | 128 / 4 | 4 | 1.6213 | 1.0831 | 1.497x | Triton |
| 13 | 64 | 1,024 | 128 / 4 | 4 | 102.4249 | 22.0748 | 4.640x | Triton |
| 14 | 32 | 100,000 | 1,024 / 16 | 2 | - | - | - | authorized resource skip |

Summary:

- 13/13 executable PASS plus one authorized resource skip excluded from pass.
- 65 accuracy trials and 938,885,120 checked elements; zero failures.
- Maximum absolute error: 0.00114846.
- Geometric-mean speedup: 1.427x.
- Attention dispatch: Triton 1,008; SDPA 0; reference 448.

### Project-owned held-out matrix

| case | B | S | d / heads | layers | mask | baseline ms | optimized ms | speedup |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| tiny-overhead | 1 | 32 | 64 / 4 | 2 | none | 0.533 | 0.326 | 1.633x |
| medium-throughput | 8 | 128 | 256 / 8 | 2 | none | 0.543 | 0.362 | 1.498x |
| medium-padding | 4 | 256 | 512 / 8 | 2 | 30% padding | 0.676 | 0.514 | 1.315x |
| long-causal | 2 | 512 | 512 / 8 | 2 | causal | 0.707 | 0.896 | 0.789x |
| long-causal-padding | 2 | 512 | 512 / 8 | 2 | causal + 30% padding | 0.845 | 0.991 | 0.852x |
| long-attention | 1 | 1024 | 512 / 8 | 2 | none | 0.827 | 0.537 | 1.541x |
| wide-model | 2 | 128 | 1024 / 16 | 1 | none | 0.313 | 0.259 | 1.211x |

Summary:

- 7 requested, 7 completed, 7 PASS, 0 FAIL, 0 OOM, 0 ERROR.
- 35 accuracy trials and 13,117,440 checked elements.
- 0 failed elements.
- Maximum absolute error: 0.000595748.
- Geometric-mean speedup: 1.221x.
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
- overall geometric-mean speedup 1.233x and float32-only geomean 1.443x; and
- aggregate dispatch counts Triton 672, SDPA 1,344, reference 2,688.

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

## 8. Optimization campaign and rejected alternatives

Profiling final row 10 showed that the prior 64x128 attention tile spilled
2,468 registers and used 81,920 bytes of shared memory under causal IEEE-fp32
dots. EXP-001 introduced a bounded 32x64 tile only for `head_dim=64` and
sequence <=128; compiled metadata fell to two spills and 49,152 bytes. Two
alternating clean-worktree final-matrix pairs improved aggregate geomean by
8.98% and 10.19%. The targeted optimized latency fell from 3.566/3.726 ms to
0.970/0.972 ms in those pairs. An independent reviewer approved the change and
recorded a timing-noise waiver for unaffected `head_dim=32` rows whose paired
directions disagreed. The implementation was then merged and all release
artifacts were regenerated from the integrated fingerprint.

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
- run a bounded multi-agent hypothesis/review loop, accept the short
  `head_dim=64` tile, and reject lower-value dispatch and fusion alternatives;
- reject slower LayerNorm and causal-pruning paths; and
- produce provenance-linked documentation and demo instructions.

AI output was not accepted as evidence by itself. Claims in this report come
from executable tests, raw CUDA-event samples, profiler events, and captured
environment metadata.

Human/team contribution attribution is not established by repository evidence;
the submitter must add any additional participant attribution to Devpost if
applicable.

## 10. Limitations and next work

- Clarify the final table's omitted dtype, padding, timing, tolerance, and
  backward policy, and reconcile any later benchmark revision. The two local
  2026-08-27 downloads are checksum-frozen, but current live attachment-byte
  identity could not be reverified through the read-only browser path.
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

- Final organizer-shape matrix:
  docs/results/rtx-5070-ti-2026-08-28-final-evaluator-baseline.json
- EXP-001 decision: docs/experiments/EXP-001-head64-short-tiles.md
- Integrated target profiler:
  docs/results/rtx-5070-ti-2026-08-28-final-10-profile.json
- Held-out matrix: docs/results/rtx-5070-ti-2026-08-27.json
- Held-out profiler: docs/results/rtx-5070-ti-2026-08-27-profile.json
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
