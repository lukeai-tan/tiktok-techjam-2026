# Technical Report — Transformer Layer GPU Kernel

## 1. Executive summary

This project implements a forward fused-attention GPU kernel in Triton for the
Track 3 PyTorch Transformer benchmark. The kernel performs tiled QK, online
softmax, causal/padding masking, and P@V in one launch without storing the
quadratic attention matrix.

On the NVIDIA GeForce RTX 5070 Ti, all seven provisional float32 cases passed
the checked-in executable tolerance across five seeds each. The run covered
13,117,440 output elements with zero failures and measured a **1.360x
geometric-mean end-to-end speedup**, ranging from 1.138x to 1.566x.

The organizer's final shape list is not available in this repository. Results
are therefore evidence for the checked-in contract and provisional matrix, not
a claim about unpublished test cases.

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

Benchmark provenance and unresolved organizer questions are recorded in
docs/REQUIREMENTS.md and benchmarks/reference/manifest.json.

## 3. Environment

| component | measured target |
| --- | --- |
| CPU | AMD Ryzen 7 9850X3D, 8 cores / 16 WSL logical CPUs |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| compute capability | 12.0 |
| GPU memory | 16,303 MiB |
| NVIDIA driver | 610.47 |
| OS/kernel | Ubuntu on WSL2, 6.6.114.1-microsoft-standard-WSL2 |
| Python | 3.14.4 |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime | 13.0 |
| Triton | 3.7.1 |
| disk during run | C: 931 GiB total, 419 GiB free |
| float32 policy | high matmul precision, TF32 enabled |

The minimal WSL image lacked gcc and Python headers. The verified target used
official Zig 0.16.0 as a user-scoped C frontend and extracted Ubuntu
libpython3.14-dev headers for Triton's small first-use driver module. No system
package or sudo change was required.

The curated result JSON also stores the Python executable, OS, CUDA capability,
disk bytes, Git state, command, raw samples, and an implementation-content
SHA-256 so an uncommitted benchmark cannot be mistaken for the base commit.

## 4. Baseline and bottleneck analysis

The reference attention explicitly allocates B x H x S x S scores, applies
softmax, and launches a second matmul for context. Its memory traffic and
intermediate storage grow quadratically with sequence length. Projection and
FFN work remains GEMM-heavy and is best left to PyTorch/cuBLAS.

The captured causal-padding profile for five two-layer forwards recorded:

| event | count | self device time |
| --- | ---: | ---: |
| addmm | 60 | 1,589.7 us |
| custom _attention_fwd | 10 | 347.2 us |
| native LayerNorm | 25 | 116.4 us |
| GELU | 10 | 60.8 us |
| residual add | 20 | 48.2 us |

The ten custom events exactly match five forwards times two layers. This proves
the repository-owned kernel ran; dispatch counters alone were not used as
proof.

## 5. Kernel implementation

### 5.1 Layout

Q/K/V stay in projection-friendly BSHD layout. The baseline creates three BHSD
contiguous copies; the custom kernel consumes strides directly and returns BSHD
for a direct reshape into the output projection.

### 5.2 Online softmax

Each program owns a query tile and one batch/head pair. It streams K/V tiles
while maintaining fp32 running maximum, normalization sum, and weighted-value
accumulator. The rescaling formula makes each tile numerically compatible with
the prior tiles, so no score or probability matrix is stored.

### 5.3 Masks

Sequence bounds, valid-key prefixes, and causal key <= query bounds are combined
inside the score tile. The all-masked case is explicitly finite and returns
zero. The causal-padding custom path never builds the dense combined mask used
by the prior SDPA integration.

### 5.4 Numerical matching

The kernel deliberately reproduces the reference's low-precision score tensor
and scaling roundings before converting to fp32 softmax state. Float32 dot
products follow the benchmark's TF32 toggle. This was necessary for deep-stack
agreement; mathematically reasonable fused implementations can still fail an
elementwise benchmark when their rounding points differ.

### 5.5 Dispatch

The custom envelope requires CUDA compute capability 8.0+, inference, sequence
length at most 8192, final stride 1, float32/fp16, and head dimensions
16/32/64/128. Forced Triton rejects unsupported inputs. Auto uses guarded
fallbacks and exposes actual backend counts.

The primary end-to-end route is float32. Direct fp16 attention passes, but fp16
and bf16 fused differences compound in deep stacks under the strict executable
tolerance. Automatic model execution therefore uses exact reference-style math
for low precision. This avoids the false claim that a fast but numerically
rejected path is supported.

Full algorithm and launch details are in docs/KERNEL_DESIGN.md.

## 6. Benchmark method

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

| case | B | S | d / heads | layers | mask | baseline ms | optimized ms | speedup |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| tiny-overhead | 1 | 32 | 64 / 4 | 2 | none | 0.325 | 0.262 | 1.242x |
| medium-throughput | 8 | 128 | 256 / 8 | 2 | none | 0.319 | 0.238 | 1.336x |
| medium-padding | 4 | 256 | 512 / 8 | 2 | 30% padding | 0.652 | 0.498 | 1.309x |
| long-causal | 2 | 512 | 512 / 8 | 2 | causal | 0.679 | 0.474 | 1.432x |
| long-causal-padding | 2 | 512 | 512 / 8 | 2 | causal + 30% padding | 0.830 | 0.530 | 1.566x |
| long-attention | 1 | 1024 | 512 / 8 | 2 | none | 0.814 | 0.525 | 1.550x |
| wide-model | 2 | 128 | 1024 / 16 | 1 | none | 0.236 | 0.208 | 1.138x |

Summary:

- 7 requested, 7 completed, 7 PASS, 0 FAIL, 0 OOM, 0 ERROR.
- 35 accuracy trials and 13,117,440 checked elements.
- 0 failed elements.
- Maximum absolute error: 0.000997663.
- Geometric-mean speedup: 1.360x.
- Every timing backend count was Triton; no SDPA/reference timing fallback.

Incremental peak CUDA allocation fell from 78 MiB to 22 MiB (71.8%) in the
long-attention case. Causal and padding cases reduced the measured incremental
peak by 52.4% and 54.4%, respectively. Tiny/wide cases are dominated by model
outputs and GEMM work, so their measured peak did not change.

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

QKV/output/FFN GEMMs were also retained as vendor operations. Replacing mature
GEMM kernels merely to increase custom-code volume would not be a defensible
optimization.

## 9. AI-assisted development

The initial repository report attributed the prototype SDPA and optional
LayerNorm work to Claude Code. This revision was audited and implemented with
OpenAI Codex, which was used to:

- trace the checked-in challenge and benchmark contract;
- identify the false-green sweep and unverified-kernel gaps;
- design and implement the Triton online-softmax kernel and dispatcher;
- bootstrap and diagnose the WSL CUDA/Triton toolchain;
- generate correctness, negative-path, matrix, and profiler checks;
- iterate from measured numerical/performance failures;
- reject a slower LayerNorm path; and
- produce provenance-linked documentation and demo instructions.

AI output was not accepted as evidence by itself. Claims in this report come
from executable tests, raw CUDA-event samples, profiler events, and captured
environment metadata.

Human/team contribution attribution is not established by repository evidence;
the submitter must add any additional participant attribution to Devpost if
applicable.

## 10. Limitations and next work

- Reconcile the final organizer benchmark and shape matrix when published.
- Retest and retune on any evaluation GPU; launch policy is measured only on the
  RTX 5070 Ti.
- Add backward kernels only if training becomes part of the official contract.
- Explore residual/normalization fusion only after a new profile shows a larger
  end-to-end ceiling.
- A public demo video and Devpost submission remain external human deliverables;
  DEMO_RUNBOOK.md gives the verified recording sequence.

## 11. Evidence

- Matrix: docs/results/rtx-5070-ti-2026-08-27.json
- Profiler: docs/results/rtx-5070-ti-2026-08-27-profile.json
- Requirements: docs/REQUIREMENTS.md
- Kernel design: docs/KERNEL_DESIGN.md
- Demo procedure: DEMO_RUNBOOK.md
