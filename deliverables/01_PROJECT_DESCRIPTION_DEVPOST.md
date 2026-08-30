# SpeedROCm: shape-aware Triton acceleration for Transformer inference

## Overview

SpeedROCm is a PyTorch/Triton implementation for TikTok TechJam 2026 Track 3.
It accelerates the supplied pre-LayerNorm Transformer inference workload with
repository-owned GPU kernels while preserving the reference model's parameter
names, strict weight loading, masks, causal behavior, and output contract.

In plain language, SpeedROCm performs the same Transformer calculation as the
supplied model but changes how selected GPU operations are scheduled and how
temporary attention data is stored. The goal is lower latency and lower memory
use without changing the model's learned weights or accepted outputs.

The baseline explicitly constructs attention scores, applies softmax, and
multiplies the probabilities by values. SpeedROCm instead processes attention in
tiles, keeps the softmax state in fp32, applies causal and prefix-padding rules
inside the tile, and produces the context output without materializing a dense
`[batch, heads, sequence, sequence]` score or probability tensor.

## Project name and measured-platform scope

**SpeedROCm** is the project name. Despite the `ROCm` text in the name, this
prototype is currently implemented and benchmarked with NVIDIA CUDA, not the
AMD ROCm runtime. The recorded target is an NVIDIA GeForce RTX 5070 Ti using
PyTorch and Triton on Windows. The name does not claim AMD or NVIDIA affiliation
and must not be read as a claim of current AMD GPU support.

The repository's Python package remains `transformer_opt`; that is an internal
code identifier rather than the public project title.

## How the solution addresses the problem

The core design has five parts. Each part is guarded so unsupported inputs keep
a correctness-first path:

1. **Fused tiled attention.** A Triton `_attention_fwd` program works on a
   small group of query rows at a time and streams key/value groups through it.
   It computes query-key scores, online softmax, masking, and the
   probability-times-value result in one kernel launch. The amount of
   attention arithmetic still grows with the square of sequence length, but
   the custom path no longer stores a full square score or probability matrix.
2. **Projection-friendly layout.** Query, key, and value tensors are kept in
   `[batch, sequence, heads, head dimension]` order. The custom kernel can use
   that output directly instead of making repeated reordered copies.
3. **Packed query/key/value projection.** In measured eager CUDA 32-bit
   floating-point inference, a derived cached weight combines the three
   projections into one vendor `F.linear` call. The
   cache is invalidated after parameter-version, device, or dtype changes and
   is non-persistent, so strict state-dict compatibility is retained.
4. **Fused residual plus LayerNorm.** Exact published rows 5, 6, 9, and 11
   use a second guarded Triton kernel that combines each residual add with the
   LayerNorm that immediately consumes it. This is enabled only for measured
   eval-mode, eager CUDA, float32, contiguous, exact-shape cases.
5. **Measured shape-aware routing.** `auto` selects Triton only inside the
   tested envelope. PyTorch scaled dot-product attention (SDPA) handles
   measured cases where it is faster or
   more numerically stable, and explicit reference math handles unsupported or
   sensitive cases. Forced custom mode fails clearly when its contract is not
   met; it does not silently report fallback as custom execution.

This approach addresses both sides of the benchmark: it removes the dominant
quadratic attention allocation for long sequences, and it avoids trading away
correctness in shapes where a fused implementation accumulates small rounding
differences across many layers.

## Measured outcome

The selected measured implementation is
`transformer_opt/submission.py::UserOptimizedTransformer`, fingerprinted as
`908a0d708cd8f70f44d5f14fda93d3cafb1cc18345f43914e715594cfa7b7ef9`.

On the recorded NVIDIA GeForce RTX 5070 Ti environment:

- all 13 executable rows in the published final shape table passed five
  accuracy trials each;
- the final matrix compared 938,885,120 output elements and recorded zero
  failed elements;
- the primary final-matrix geometric-mean speedup was **1.977×**, with a
  complete confirmation at **1.986×**;
- the primary final rows measured **2.314×** for row 5, **1.780×** for row 9,
  **6.377×** for row 11, and **4.791×** for the 1,024-token row 13;
- the untouched organizer PyTorch default passed 5/5 trials at **1.385×**;
- all 28 feasible source-derived contract cases passed, covering 459,776,000
  comparisons with zero failed elements; and
- the project-held-out seven-case matrix passed 7/7 cases at **1.340×**, with a
  second complete run at **1.386×**. The exact long-causal held-out route stayed
  in a **1.198×–1.204×** band across four measured-fingerprint checks.

The published row with `B=32`, `S=100000`, `d_model=1024`, and 16 heads is
recorded as the organizer-authorized resource skip. It is not counted as a
pass. A raw float32 score tensor for that shape would require approximately
20.48 TB before probabilities, which is why the source contract permits
resource preflight.

The full before/after matrix, formulas, profiler measurements, input coverage,
and optimization diagrams are in the [technical report](03_TECHNICAL_REPORT.md).

## Terms and symbols used in this description

| Term or symbol | Meaning |
| --- | --- |
| Baseline / reference | The supplied Transformer implementation used as the correctness and timing comparison. |
| Optimized | SpeedROCm's `UserOptimizedTransformer`. |
| Q, K, V / QKV | Query, key, and value tensors used by attention. |
| `B`, `S`, `H`, `D` | Batch size, sequence length, number of heads, and per-head dimension. |
| `d_model` | Total model width; it equals `H × D`. |
| `[B, S, H, D]` | Tensor-axis order: batch, sequence, head, then per-head feature. |
| Triton | The GPU-kernel programming language/compiler used here; it does not mean NVIDIA Triton Inference Server. |
| CUDA | The NVIDIA GPU software runtime used for the recorded implementation. |
| ROCm | AMD's GPU software platform. It appears in the project name, but the current code is not presented as a ROCm implementation. |
| fp16 / fp32 / bfloat16 | 16-bit, 32-bit, and brain-float 16-bit numeric formats. |
| SDPA | PyTorch scaled dot-product attention, one of the safe backends. |
| `×` after a result | Speedup multiplier; values above `1×` are faster than baseline. |
| `ms`, `MiB`, `TB` | Milliseconds, mebibytes (1,048,576 bytes), and decimal terabytes (1,000,000,000,000 bytes). |
| Geometric-mean speedup | A multiplicative average of per-case speedups, used so one unusually large case does not dominate the summary. |
| Failed element | One output value that violates both the absolute and relative error limits. Zero failed elements are required for a pass. |
| Authorized resource skip | A permitted non-execution due to resource limits; it is reported separately and never counted as a pass. |

## Development tools

- Windows PowerShell and native Windows CUDA/Triton development
- Python 3.12.10, Git, and GitHub
- Jupyter and Google Colab notebook workflow
- PyTorch CUDA events and PyTorch Profiler
- pytest for CPU contracts, GPU kernel tests, fallback tests, and end-to-end
  validation
- OpenAI Codex for repository audit, implementation, profiling, testing,
  evidence reconciliation, and documentation
- Claude Code for the pre-existing SDPA/optional-LayerNorm prototype, as
  attributed in the repository's technical history

## APIs and runtime interfaces used

There is no external hosted model or web API required at runtime. The runtime
interfaces are:

- PyTorch `torch.nn` and `torch.nn.functional` for the reference-compatible
  Transformer, vendor linear/GEMM operations, GELU, LayerNorm, and the
  `scaled_dot_product_attention` fallback;
- Triton JIT and Triton language operations for the repository-owned attention
  and residual/LayerNorm kernels;
- CUDA events for synchronized steady-state latency measurements; and
- PyTorch Profiler for proving that `_attention_fwd` and
  `_residual_layer_norm_fwd` actually executed.

OpenAI Codex was a development tool, not a network service called by the
submission. No API key or external service is needed to reproduce inference.

## Libraries and frameworks

| Component | Recorded version or role |
| --- | --- |
| PyTorch | `2.13.0+cu130`; selected benchmark framework and reference-compatible model |
| Triton | `3.7.1`; custom GPU-kernel compiler/runtime |
| CUDA runtime | `13.0` |
| `triton-windows` | `3.7.1.post27`; native Windows Triton distribution used by the target environment |
| NumPy | `2.5.2` |
| pytest | `9.1.1` |
| Jupyter/Colab | Reproducibility notebook and interactive benchmark surface |

## Datasets and assets

This is a kernel-optimization submission rather than a trained-data model.
There is no external training dataset and no third-party model-weight download.

- Accuracy inputs are deterministic synthetic tensors generated from recorded
  seeds, input scales, causal flags, and prefix-padding ratios.
- The participant-supplied organizer PyTorch and TensorFlow benchmark files are
  retained byte-for-byte with SHA-256 provenance. PyTorch is the selected
  implementation framework; the TensorFlow file is retained for shape and
  contract reconciliation.
- The repository includes the challenge information snapshot, benchmark shape
  manifests, raw JSON result artifacts, profiler evidence, the Colab notebook,
  source code, tests, and Markdown documentation.
- The demo is intended to show owned source code, terminal output, diagrams,
  and authorized challenge material only. It should not include third-party
  music, logos, footage, or copyrighted assets without permission.

## Repository and evidence links

- Code repository: <https://github.com/lukeai-tan/tiktok-techjam-2026>
- Repository setup and reproduction handoff:
  [`02_PUBLIC_REPOSITORY.md`](02_PUBLIC_REPOSITORY.md)
- Detailed technical report: [`03_TECHNICAL_REPORT.md`](03_TECHNICAL_REPORT.md)
- Demo recording script: [`04_DEMO_VIDEO_SCRIPT.md`](04_DEMO_VIDEO_SCRIPT.md)
- Current final result:
  [`../docs/results/rtx-5070-ti-2026-08-29-c11-integrated-final.json`](../docs/results/rtx-5070-ti-2026-08-29-c11-integrated-final.json)
- Current requirement audit:
  [`../hackathon-docs/TRACK3_COMPLIANCE.md`](../hackathon-docs/TRACK3_COMPLIANCE.md)

The GitHub URL and the final YouTube URL are last-mile submission fields. The
repository must be verified public while signed out, and the human-recorded
YouTube URL must be added to Devpost after playback is checked.

## Limitations and future work

- The organizer's final dimensions do not specify dtype, padding, timing,
  tolerance, or backward policy. The recorded evidence therefore states its
  PyTorch float32/no-padding assumptions and the stricter executable comparator
  explicitly.
- The custom path is forward/inference only and is tuned to the RTX 5070 Ti.
- Automatic deep-stack low-precision execution chooses exact reference math;
  direct fp16 kernel checks pass, but fused low-precision differences can
  compound beyond the strict full-model tolerance.
- Packed QKV adds derived weight storage: about 6 MiB for the measured
  two-layer `d_model=512` case and about 48 MiB for the measured four-layer
  `d_model=1024` case. Widths 513–1023 remain deliberately disabled because
  they were not measured.
- Residual/LayerNorm fusion is exact-shape and guarded. Generalizing it to
  neighboring shapes requires new profiler, correctness, memory, and fallback
  evidence.

## Team contributions

Repository evidence does not establish additional human team members. For a
solo submission, the submitter owns the human contribution and the AI-tool
roles are documented in the technical report. If this is a team submission,
replace this paragraph with only verified names and responsibilities before
publishing on Devpost.
