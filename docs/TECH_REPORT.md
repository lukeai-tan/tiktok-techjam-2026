# Technical Report — Transformer Layer GPU Kernel

For a quick orientation, use the [documentation hub](README.md). This report is
the implementation and measurement narrative; the [campaign run-through](experiments/CAMPAIGN_RUN_THROUGH.md)
owns the shorter optimization outcome and ranking.

## 1. Executive summary

This project implements a forward fused-attention GPU kernel in Triton for the
Track 3 PyTorch Transformer benchmark. The kernel performs tiled QK, online
softmax, causal/padding masking, and P@V in one launch without storing the
quadratic attention matrix.

On the NVIDIA GeForce RTX 5070 Ti, all 13 executable rows in the
organizer-published final shape table passed the checked-in executable tolerance
across five seeds each. The run covered 938,885,120 output comparisons with zero
failures and measured a **1.977x geometric-mean end-to-end speedup**, ranging
from 1.097x to 6.377x. A complete confirmation measured 1.986x with identical
correctness and backend counts. The source-authorized 100,000-token resource row was
preflight-skipped and was not counted as a pass.

Campaign 11 is the flagship because it is the current cumulative fingerprint
and owns the latest full validation. Campaign 5's 1.995117x confirmation remains
the highest historical aggregate and strongest broad-generalization snapshot,
but it is not the current submission. The ranked decision and specialist picks
are in the [campaign run-through](experiments/CAMPAIGN_RUN_THROUGH.md#flagship-and-strongest-specialist-campaigns).

Both organizer benchmark downloads are now checksum-frozen. The untouched
PyTorch default six-layer case also passed 5/5 trials with zero failed elements
and measured 1.385x median speedup. A fail-closed matrix then translated every
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
in `hackathon-docs/ORGANIZER_INPUTS.md`, `docs/REQUIREMENTS.md`, and
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
| optimized Transformer range | 5 | 4,612.9 us |
| addmm | 40 | 1,572.0 us |
| custom _attention_fwd | 10 | 2,460.9 us |
| native LayerNorm | 25 | 119.6 us |
| GELU | 10 | 60.8 us |
| residual add | 20 | 49.5 us |

The ten custom events exactly match five forwards times two layers. This proves
the repository-owned kernel ran; dispatch counters alone were not used as
proof. Packed QKV lowers the expected projection/linear `addmm` count from 60
to 40 across those five forwards.

The final-shape profile then isolated row 10 (`B=64`, `S=128`, `d_model=128`,
two heads, four layers). Before EXP-001, `_attention_fwd` consumed 30,324.486 us
across 40 launches and dominated 79.6% of recorded GPU time. The accepted short
`head_dim=64` tile reduced that event; the current integrated profile is
2,694.679 us, 91.11% below the frozen pre-EXP-001 value, and all 40 launches
remain Triton. Campaign 3 measured that row at 1.602x; the current primary
rebaseline measures 1.526x amid run-wide baseline timing shifts.

Campaign 2 then isolated final row 1 (`head_dim=32`, sequence 128). Three
alternating measurements selected a 64x64 launch over the prior 64x128 policy:
candidate optimized median averaged 0.8201 ms versus 1.2402 ms for the unchanged
implementation. After integration, row-1 `_attention_fwd` fell from 7,008.677
us to 2,103.978 us across 40 launches (69.98%), and final-matrix geomean rose
6.95% from 1.426692x to 1.525823x.

Campaign 3 isolated final row 9 (`head_dim=128`, sequence 128). Three bounded
launch candidates and one production SDPA route passed correctness screening;
counterbalanced timing selected 32x32 Triton tiles at a 0.9042 ms three-run
median, 26.73% below the fresh 1.2341 ms baseline and 6.16% faster than SDPA.
After integration, `_attention_fwd` fell from 6,775.468 us to 3,018.182 us
across 40 launches (-55.45%), the ten-step model range fell 26.96%, and final
row 9 improved from 1.2055 ms to 0.9071 ms.

Campaign 4 isolated final row 11 (`head_dim=8`, sequence 128), where exact
reference attention dominated. Because Triton requires a dot reduction width
of at least 16, the candidate zero-pads Q/K/V lanes 8-15, stores only the real
eight lanes, and retains the real `8**-0.5` scale. Three bounded Triton launch
geometries and exact SDPA were screened. The selected 64x64 launch reproduced
1.0595/1.0628/1.0624 ms optimized medians, 81.08% below fresh reference and
43.06% below SDPA. The integrated ten-step model range fell from 41,658.659 us
to 10,592.605 us (-74.57%) with 40/40 Triton calls.

Campaign 5 revisited the remaining exact-reference routes and two held-out
long-causal regressions. Strict backend screens rejected full Triton/SDPA for
rows 6-8 after one or more elements failed. Layer-index isolation then found
two accurate hybrids: row 6 keeps layers 0-1 exact and uses Triton for layers
2-3; row 7 keeps layer 0 exact and uses Triton for layers 1-3. Exact-shape SDPA
for the two `B=2,S=512,d_model=512,heads=8,layers=2,causal=true` held-out cases
removed both latency regressions. Row 8 remained unchanged because its SDPA
screen failed and `aten::addmm` consumed about 71% of its profile.

Campaign 6 kept those attention routes and tested four bounded surfaces. Row-6,
row-7, and row-11 launch variants plateaued or regressed. Exact-width packed
QKV for row 8 survived: three 300-sample candidates were faster internally,
two contemporaneous unchanged controls were slower than their own baselines,
and the integrated profiler reduced 240 `aten::addmm` calls to 160. The source
guard is exact (`d_model <= 512 or d_model == 1024`), so unmeasured widths
513-1023 retain the established separate-projection path.

Campaign 7 profiled the remaining exact-row bottlenecks. A direct
`head_dim=256` Triton attention route passed primitive checks but missed two
strict full-model elements; after an exact-first-layer repair it passed but
regressed row-8 latency by 8.50%, while a wider tile exceeded the GPU's shared-
memory limit. The route was rejected. Exact row 6 instead exposed a 24% combined
residual-add and LayerNorm ceiling. A guarded fused residual-plus-LayerNorm
kernel passed direct, boundary, state-dict, multi-seed/scale/padding stress, and
full-matrix gates. Its integrated profile reduces that subsystem's device time
36.30% and ten-forward model time 9.54%, with no increase in the measured
11,802,787,840-byte incremental peak.

Campaign 8 tested whether that accepted residual/normalization primitive could
remove the same launch boundary on exact final row 11. I1 was correct and
faster, but the review found that the shared predicate relied on gradient state
rather than explicitly excluding `model.train()`. I1R added the eval-mode guard
and row-11 CPU, dtype, layout, runtime-shape, mask, gradient, training, and head
neighbor tests. The retained fingerprint passed 36 combined row-6/row-11 stress
scenarios covering 2,967,994,368 outputs with zero failures. Two 300-sample
candidate runs averaged 0.897184 ms versus 0.993525 ms across three unchanged
controls (-9.70%) with identical 29,360,128-byte incremental peak allocation.
The integrated 30-forward profile reduces model device time 21.96% and the
residual/normalization subsystem 46.28%.

Campaign 10 profiled rows 5, 7, and 12 before changing code. Residual-add plus
LayerNorm accounted for 29.07% of row-5 model device time, making row 5 the
largest still-bounded reuse target. A width-1024 fusion candidate and an
eight-warp attention variant were rejected after profile regressions. The exact
row-5 fusion passed direct guards, 18 seed/scale/padding and neighbor scenarios,
and a 300-sample gate with zero failed elements. Against two counterbalanced
unchanged controls, optimized median latency fell 11.58%; the active profile
reduced model device time 11.96% and residual/normalization time 40.63%.

Campaign 11 profiled final rows 9, 10, and 1 from the selected Campaign 10
checkpoint. Row 9 exposed a 19.10% residual/normalization ceiling, and an exact
row-9 reuse of the accepted fused forward passed route, boundary, 18-scenario
stress, affected-suite, memory, and complete candidate-matrix gates. Two
unchanged controls averaged 0.815968 ms optimized median; the isolated and
active candidates measured 0.717696 and 0.717648 ms, a reproducible 12.05%
active reduction with identical 29,360,128-byte peak allocation. Two active
profiles preserve 240 fused launches, 30 native norms, and 120 Triton calls and
reduce mean subsystem time 41.77%. Top-level profiler time remains noisy, so
the counterbalanced 300-sample CUDA-event result is the causal speed evidence.

## 5. Kernel implementation

### 5.1 Layout

Q/K/V stay in projection-friendly BSHD layout. The baseline creates three BHSD
contiguous copies; the custom kernel consumes strides directly and returns BSHD
for a direct reshape into the output projection.

### 5.2 Packed QKV projection

For the measured eager CUDA float32 path through `d_model=512`, plus exact
`d_model=1024`, the model
caches concatenated views of the existing Q/K/V weights and biases and uses one
vendor `F.linear` call instead of three. The resulting `[B,S,3,H,D]` tensor is
unbound into strided Q/K/V views consumed by the selected backend. Cache
signatures detect parameter mutation, loading, and device/dtype changes;
derived tensors are non-persistent, so the baseline state dict remains
unchanged.

Target-device measurements found the combined projection bit-identical for the
tested float32 shapes and beneficial through width 512. Campaign 6's longer
row-8 recheck established an exact-width-1024 benefit: the integrated profile
reduced `addmm` device time 11.33% and ten-forward model device time 7.91%.
Training, low precision, CPU, compiled paths, widths 513-1023, and widths above
1024 remain on separate projections.

### 5.3 Fused residual plus LayerNorm

For exact final rows 5, 6, 9, and 11 under eval-mode eager CUDA float32 inference, each attention or FFN
residual add is fused with the LayerNorm that immediately consumes it. A Triton
program computes fp32 row statistics, applies the existing affine parameters
and epsilon, and stores both the residual result and normalized output without
materializing a separate add result for another native launch. The initial
input norm remains native. Optional bias, valid-row zeroing, and
strict state-dict structure are preserved. Neighboring shapes, noncontiguous
masks, compiled execution, gradients, CPU, and other dtypes fall back to the
original PyTorch operations.

### 5.4 Online softmax

Each program owns a query tile and one batch/head pair. It streams K/V tiles
while maintaining fp32 running maximum, normalization sum, and weighted-value
accumulator. The rescaling formula makes each tile numerically compatible with
the prior tiles, so no score or probability matrix is stored.

### 5.5 Masks

Sequence bounds, valid-key prefixes, and causal key <= query bounds are combined
inside the score tile. The all-masked case is explicitly finite and returns
zero. The causal-padding custom path never builds the dense combined mask used
by the prior SDPA integration.

### 5.6 Numerical matching

The kernel deliberately reproduces the reference's low-precision score tensor
and scaling roundings before converting to fp32 softmax state. Non-causal
float32 dot products follow the benchmark's TF32 toggle; causal custom
attention always uses IEEE fp32 dot products. Final-shape testing found rare
four-layer causal TF32 misses, demonstrating that mathematically reasonable
fused implementations can still fail an elementwise benchmark when their
rounding points differ.

### 5.7 Dispatch

The custom envelope requires CUDA compute capability 8.0+, inference, sequence
length at most 8192, final stride 1, float32/fp16, and head dimensions
8/16/32/64/128. Width eight uses zero-masked 16-lane dot padding. Forced Triton
rejects unsupported inputs. Auto uses guarded
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
The multi-layer width-eight Triton route is deliberately limited to exact final
row 11; other width-eight shapes stay on reference until separately measured.

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
| median latency | 1.8687 ms | 1.3495 ms |
| mean latency | 2.1607 ms | 1.3988 ms |
| p90 latency | 2.8291 ms | 1.5648 ms |
| throughput | 547,983 token/s | 758,797 token/s |

- Median speedup: 1.385x.
- Accuracy: 5/5 PASS, 0 failed out of 2,621,440 elements.
- Maximum absolute error: 0.00100136.
- Optimized attention dispatch: Triton 1,950; SDPA 0; reference 0.

### Organizer-published final shape matrix

The published rows were executed through the untouched PyTorch comparator in
isolated processes. Dtype, padding, and timing are the recorded assumptions in
Section 6, not claims about omitted organizer policy.

| row | B | S | d / heads | layers | baseline ms | optimized ms | speedup | backend |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 64 | 128 | 128 / 4 | 4 | 1.3778 | 0.8152 | 1.690x | Triton |
| 2 | 1 | 128 | 128 / 4 | 4 | 1.4685 | 0.8804 | 1.668x | Triton |
| 3 | 4 | 128 | 128 / 4 | 4 | 1.4305 | 0.7830 | 1.827x | Triton |
| 4 | 16 | 128 | 128 / 4 | 4 | 1.3367 | 0.7560 | 1.768x | Triton |
| 5 | 128 | 128 | 128 / 4 | 4 | 2.9495 | 1.2745 | 2.314x | Triton; fused residual/norm |
| 6 | 10,000 | 128 | 128 / 4 | 4 | 445.1712 | 332.4715 | 1.339x | 2 reference + 2 Triton layers; fused residual/norm |
| 7 | 64 | 128 | 32 / 4 | 4 | 1.4340 | 0.9723 | 1.475x | 1 reference + 3 Triton layers |
| 8 | 64 | 128 | 1,024 / 4 | 4 | 15.0661 | 13.7354 | 1.097x | reference |
| 9 | 64 | 128 | 128 / 1 | 4 | 1.3186 | 0.7409 | 1.780x | Triton; fused residual/norm |
| 10 | 64 | 128 | 128 / 2 | 4 | 1.4622 | 0.9257 | 1.579x | Triton |
| 11 | 64 | 128 | 128 / 16 | 4 | 5.7496 | 0.9017 | 6.377x | Triton; fused residual/norm |
| 12 | 64 | 32 | 128 / 4 | 4 | 1.4426 | 0.8002 | 1.803x | Triton |
| 13 | 64 | 1,024 | 128 / 4 | 4 | 88.8280 | 18.5412 | 4.791x | Triton |
| 14 | 32 | 100,000 | 1,024 / 16 | 2 | - | - | - | authorized resource skip |

Summary:

- 13/13 executable PASS plus one authorized resource skip excluded from pass.
- 65 accuracy trials and 938,885,120 checked elements; zero failures.
- Maximum absolute error: 0.00114870.
- Geometric-mean speedup: 1.977420x; complete confirmation 1.986499x.
- Dedicated long runs resolve snapshot variance: row 5 is 1.163168 ms at
  1.880066x over 300 samples, row 6 is 188.457397 ms at 1.546330x over 100
  samples, row 9 is 0.717648 ms at 1.150046x over 300 samples, and row 11 is
  0.890672 ms at 4.710116x over 300 samples.
- Attention dispatch: Triton 1,260; SDPA 0; reference 196.

### Project-owned held-out matrix

| case | B | S | d / heads | layers | mask | baseline ms | optimized ms | speedup |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| tiny-overhead | 1 | 32 | 64 / 4 | 2 | none | 0.4758 | 0.3126 | 1.522x |
| medium-throughput | 8 | 128 | 256 / 8 | 2 | none | 0.4820 | 0.3276 | 1.471x |
| medium-padding | 4 | 256 | 512 / 8 | 2 | 30% padding | 0.6465 | 0.4863 | 1.329x |
| long-causal | 2 | 512 | 512 / 8 | 2 | causal | 0.6812 | 0.5682 | 1.199x |
| long-causal-padding | 2 | 512 | 512 / 8 | 2 | causal + 30% padding | 0.8289 | 0.6757 | 1.227x |
| long-attention | 1 | 1024 | 512 / 8 | 2 | none | 0.8077 | 0.5148 | 1.569x |
| wide-model | 2 | 128 | 1024 / 16 | 1 | none | 0.2344 | 0.2077 | 1.128x |

Summary:

- 7 requested, 7 completed, 7 PASS, 0 FAIL, 0 OOM, 0 ERROR.
- 35 accuracy trials and 13,117,440 checked elements.
- 0 failed elements.
- Maximum absolute error: 0.000599027.
- Geometric-mean speedup: 1.339847x; complete confirmation 1.386495x.
- Timing dispatch: SDPA for tiny/medium unmasked and the exact two long-causal
  cases; Triton for padding-only, long-attention, and wide-model cases; no
  reference timing fallback.
- Four complete current-fingerprint matrices put the exact long-causal route at
  1.198x-1.204x without padding and 1.213x-1.335x with padding. Their geomeans
  span 1.340x-1.515x because unrelated short cases are noisy. A separate
  300-sample run is 1.198x with 620 SDPA calls and zero failed elements.

### Supplied-contract shape validation

The isolated exact-harness matrix produced:

- 29 requested entries: 28 executable and one source-authorized resource skip;
- 28/28 executable PASS and 0/459,776,000 failed elements across 140 trials;
- batch sizes 1, 4, 8, 16, 128, and 10,000;
- sequence lengths 32, 128, and 1,024;
- widths 32, 128, 512, and 1,024; heads 1, 2, 4, 8, and 16;
- float32, float16, bfloat16, causal, non-causal, and prefix-padding coverage;
- overall geometric-mean speedup 1.208961x; and
- aggregate dispatch counts Triton 672, SDPA 1,344, reference 2,688.

The matrix uses the selected PyTorch executable tolerance of atol=0.001 OR
rtol=0.01, which is stricter than the TensorFlow download's defaults. Its
machine-readable policy and full stdout/evidence are stored with SHA-256
fingerprints; a crash, OOM, numerical failure, unauthorized skip, or empty run
returns nonzero.

The source-derived matrix is a correctness and routing breadth gate. Its fresh
in-run baseline and optimized medians remain versioned because sub-millisecond
and large-batch timings can move independently; no source-derived case actually
selects Triton inside the new short-`head_dim=128` launch envelope.

Incremental peak CUDA allocation fell from 78 MiB to 22 MiB (71.8%) in the
long-attention case. Long-causal, causal-padding, and medium-padding cases
reduced the measured incremental peak by 50.3%, 54.4%, and 31.3%, respectively.
Tiny/wide cases are dominated by model outputs and GEMM work, so their measured
incremental peak did not change. Packed QKV storage is prepared before this
incremental measurement and adds about 6 MiB for two float32 d_model=512
layers. On exact row 8, the four-layer d_model=1024 cache increases allocated
memory before the measured forward by 50,380,800 bytes (about 48 MiB); control
and candidate retain the same 369,115,136-byte optimized incremental activation
peak.

The largest gains occur when attention or mask materialization is a larger share
of the block. Wide-model performance is dominated by projection/FFN GEMMs, so
attention fusion has less leverage.

## 8. Optimization campaign and rejected alternatives

This section is the submission-facing narrative. The canonical cross-campaign
ledger, including the pre-ledger foundation and every immutable attempt record,
every meaningful candidate disposition, failed gates, and current route table,
is [the complete optimization history](experiments/OPTIMIZATION_HISTORY.md).
The [campaign run-through](experiments/CAMPAIGN_RUN_THROUGH.md) owns the shorter
executive narrative and flagship ranking.
The total includes optimization campaigns, selection/current comparisons, and
the alternate-branch evaluation; the history separates each record set and its
PASS/FAIL/time accounting.

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

Campaign 2 first tested three long-`head_dim=32` policies on final row 13.
`BLOCK_N=128`, a two-stage-only policy, and `BLOCK_M=32` measured 41.3406 ms,
17.9966 ms, and 21.5897 ms respectively against the 17.6598 ms baseline, so
all three were rejected despite exact correctness. It then tested the short
`head_dim=32` policy on row 1. The 32x64, 32x128, and 64x64 tiles measured
0.8579 ms, 0.8805 ms, and 0.8164 ms in screening. Two alternating confirmation
rounds selected 64x64: its three-run mean was 0.8201 ms with a 0.0036 ms sample
standard deviation, versus 1.2402 ms for the unchanged policy.

Independent review approved the exact `head_dim == 32 and seq_len <= 128`
guard. After integration, all affected executable final rows improved and
final geomean rose from 1.426692x to 1.525823x (+6.95%). The rebaseline passed
13/13 final rows, 7/7 held-out cases, 28/28 source-derived cases, the untouched
organizer default, three profiler gates, and 103 repository tests. EXP-004 then
probed direct `head_dim=8` support and stopped when Triton compilation proved
its dot product requires `K >= 16`; exact reference fallback remains. All 39
Campaign 2 attempts—including four deliberately retained failed/rework/reject
gates—store timing, output, accuracy, latency, backend, environment, and
artifact hashes in `docs/experiments/attempts/`.

Campaign 3 profiled row 9 and tested 16x64, 32x32, and 16x32 short
`head_dim=128` tiles. The 32x32 candidate was fastest and stable across three
runs (0.9042/0.9049/0.9034 ms). A production SDPA alternative was corrected
after its first route missed the organizer's all-valid mask, then measured
0.9613/0.9716/0.9636 ms; it was rejected despite exact correctness. Independent
review approved only the exact `head_dim == 128 and seq_len <= 128` launch
guard. The integrated final matrix remained 13/13 executable PASS with the one
authorized skip, raised geomean to 1.555780x, and reduced the row-9 profiler's
attention time 55.45%. Exact organizer-default, held-out, source-derived, and
curated-artifact gates all remained green, followed by 104/104 repository tests.

Campaign 4 returned to the previously rejected direct `head_dim=8` idea with a
profile-authorized design that respects Triton's 16-lane minimum by zero-padding
only the internal dot width. Exact SDPA passed but measured 1.8658 ms in the
production route. Triton 64x128, 64x64, and 32x64 measured 1.2739, a three-run
median of 1.0624, and about 1.31 ms, respectively. Independent review approved
64x64 for integration after direct kernel, 18-scenario model stress, affected-
row, hash, and profiler checks. Two complete final matrices then passed all 13
executable rows plus the exact authorized skip at 1.780075x and 1.784920x.
Row 7 remained reference within -0.19% of its Campaign 3 normalized speedup;
row 11 switched to Triton and improved from 0.978x to 5.395x. The full suite
passed 112/112 before documentation closure. All failed startup, manifest,
test-wiring, and logger-portability gates remain in the Campaign 4 ledger.

Campaign 5 ran three profile-authorized hypotheses. Full-backend row-7 Triton
and SDPA each missed one of 1,310,720 compared elements; full-backend row 6
missed 21 of 819,200,000; row-8 SDPA missed one of 41,943,040. These failures
were retained and bounded the accepted routes. Row 7's first-two-layer Triton
hybrid passed but was superseded by a faster first-layer-reference design;
the latter passed 18 stress scenarios and cut the ten-step model range 33.64%.
Row 6's one-reference-layer design still missed one element, while the
two-reference/two-Triton design passed and cut the ten-step model range 19.74%.
The exact long-causal SDPA route passed both padded and unpadded multi-seed
stress gates. Integrated final geomean rose 7.62% over the fresh Campaign 5
baseline, and all broader correctness gates remained green.

Campaign 6 then profiled and bounded four remaining surfaces without reopening
known failures. Five row-6 and five row-7 launch variants were correct but
slower or profiler-neutral; three new row-11 tile/warp axes were 6.42%-19.77%
slower than the long control. A two-plus-one row-8 projection grouping measured
0.988851x and was rejected. Exact-width packed QKV was reworked after
independent review rejected an over-broad `<=1024` guard. The accepted guard
preserves widths 513-1023, passed exact row 8 and a width-1024 neighbor, and
reduced the same-window row-8 profile's `addmm` calls 33.33%, `addmm` time
11.33%, and model time 7.91%. That Campaign 6 final pair is
1.872916x/1.863721x and 0.491% apart; Campaign 5's higher historical aggregate observations remain
reported separately because their unrelated baseline timings are not a
same-window causal comparison.

Campaign 7 rejected wide-head attention and accepted a different fusion. The
16x16 and 16x32 `head_dim=256` kernels passed direct arithmetic but each missed
two of 41,943,040 elements through all four row-8 layers. Keeping the first
layer exact repaired accuracy but produced 141,240.672 us of ten-forward model
device time versus the 130,180.675 us fresh control (+8.50%); the 16x64 variant
required 151,616 bytes of shared memory against the 101,376-byte device limit.
The route was closed. On row 6, fusing residual adds with their downstream
LayerNorms passed 18 seed/scale/padding scenarios covering 2,949,120,000 outputs,
direct and boundary tests, and the broader suite. The final profile replaces 80
adds and 80 native norms with 80 fused launches, cuts subsystem time 36.30%, and
cuts model time 9.54%. Counterbalanced 100-sample candidate brackets averaged
1.554314x versus 1.419031x unchanged controls (+9.53% normalized), with no peak-
memory increase.

Campaign 8 extended that proven fusion only to exact row 11. I1's first
10-step profile was noisy enough to show a contradictory top-level result, so
the evidence was retained and repeated at 30 steps. I1R then added the explicit
training-mode guard discovered by Council review. Two retained 300-sample runs
averaged 0.897184 ms, 9.70% below three unchanged controls, and the integrated
profile replaced 240 residual adds and 240 native norms with 240 fused launches.
Subsystem/model device time fell 46.28%/21.96%, peak allocation was unchanged,
and the Campaign 8 complete final pair is 1.876167x/1.911052x.

Campaign 9 then profiled rows 8 and 13. A broader width-1024 fusion passed
correctness but regressed row-8 latency/profile time; an eight-warp variant also
regressed. An exact row-13 variant failed the zero-error gate and was rejected.
The campaign closed without a winner and restored the Campaign 8 fingerprint.

Campaign 10 used a fresh profile to isolate row 5, where residual-add and native
LayerNorm consumed 29.07% of model device time. The exact-row-5 I1 route passed
37 affected tests and a 150,994,944-output stress screen, then reduced optimized
median latency 11.58% against counterbalanced unchanged controls. The integrated
30-forward profile moved 240 adds and 240 norms into fused launches, reducing
subsystem/model device time 40.63%/11.96%. Its 300-sample gate measured
1.162976 ms, 2.001995x, zero failed elements, and 58,720,256 bytes incremental
peak allocation. Its complete final pair was 1.926716x/1.939005x.

Campaign 11 moved to the head-count-specific row-9 surface. The exact-row-9
candidate passed 40 affected tests and every isolated final row. Two unchanged
controls differed by only 0.047%; active optimized latency fell 12.05% from
their 0.815968 ms mean to 0.717648 ms, matching the isolated candidate within
0.007%, with zero failures and unchanged peak allocation. Repeated profiles
replace 240 residual adds and 240 native norms with 240 fused launches and cut
mean subsystem time 41.77%; top-level profiler time remains noisy. The current
complete final pair is 1.977420x/1.986499x.

The inherited standalone Triton LayerNorm was measured before removal:

| rows x width | native CUDA | custom Triton | native/custom |
| --- | ---: | ---: | ---: |
| 1024 x 512 | 0.00832 ms | 0.01629 ms | 0.511x |
| 2048 x 512 | 0.01082 ms | 0.01562 ms | 0.693x |
| 256 x 1024 | 0.00758 ms | 0.01664 ms | 0.456x |

It was numerically accurate but consistently slower and did not remove a
neighboring launch, so it was retired. Campaign 7 did not revive that standalone
kernel: the accepted exact-row-5/row-6/row-9/row-11 routes instead remove a neighboring residual
launch and the intermediate handoff. Its strict shape and runtime guard reflects
the measured ceiling; it is not evidence for a general LayerNorm replacement.

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
  to measured eager CUDA float32 envelopes: `d_model <= 512` and exact 1024.
- Add backward kernels only if training becomes part of the official contract.
- Re-evaluate residual/normalization fusion on other shapes only after a fresh
  profile and exact boundary proof; the current route is exact to rows 5, 6, 9, and 11.
- A public demo video and Devpost submission remain external human deliverables;
  `guides/DEMO_RUNBOOK.md` gives the verified recording sequence.

## 11. Evidence

- Executive run-through and flagship ranking:
  docs/experiments/CAMPAIGN_RUN_THROUGH.md
- Canonical optimization history:
  docs/experiments/OPTIMIZATION_HISTORY.md
- Final organizer-shape matrix:
  docs/results/rtx-5070-ti-2026-08-29-c11-integrated-final.json
- Foundational and short-head decisions: docs/experiments/CAMPAIGN_RUN_THROUGH.md
- Integrated Campaign 11 profiles:
  docs/results/rtx-5070-ti-2026-08-29-c11-integrated-row05-profile.json,
  docs/results/rtx-5070-ti-2026-08-29-c11-integrated-row06-profile.json,
  docs/results/rtx-5070-ti-2026-08-29-c11-integrated-row08-profile.json,
  docs/results/rtx-5070-ti-2026-08-29-c11-integrated-row09-profile.json,
  and docs/results/rtx-5070-ti-2026-08-29-c11-integrated-row11-profile.json
- Held-out matrix:
  docs/results/rtx-5070-ti-2026-08-29-c11-integrated-heldout-5seed.json
- Untouched organizer default:
  docs/results/rtx-5070-ti-2026-08-29-c11-integrated-organizer-default.json
- Supplied-contract validation matrix:
  docs/results/rtx-5070-ti-2026-08-29-c11-integrated-source-derived.json
- Organizer inputs: hackathon-docs/ORGANIZER_INPUTS.md
- Organizer checksums: benchmarks/reference/organizer_downloads.json
- Requirements: docs/REQUIREMENTS.md
- Kernel design: docs/KERNEL_DESIGN.md
- Track 3 compliance: hackathon-docs/TRACK3_COMPLIANCE.md
- Demo procedure: guides/DEMO_RUNBOOK.md
