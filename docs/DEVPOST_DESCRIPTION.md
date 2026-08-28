# Devpost Project Description

## Project name

FlashTile: a shape-aware Triton attention kernel for Transformer inference.

## Overview

FlashTile addresses TikTok TechJam 2026 Track 3 by implementing a
repository-owned GPU kernel for the supplied PyTorch Transformer. The solution
fuses attention score calculation, online softmax, causal/padding masks, and
the weighted-value product into one Triton launch. It avoids the baseline's
quadratic score/probability intermediates while preserving strict state-dict
and output compatibility.

## How the solution addresses the problem

The reference attention launches separate QK, softmax, and P@V operations and
stores a B x H x S x S tensor. FlashTile instead tiles Q/K/V in the natural
projection layout and maintains only per-query softmax state. Causal and valid
token bounds are applied inside each tile, including the combined
causal-padding path, so no dense mask is allocated.

An auditable dispatcher selects custom Triton only inside the tested inference
envelope and exposes actual backend counts. Unsupported cases have explicit
SDPA/reference fallbacks, and forced custom mode fails clearly rather than
silently pretending the custom kernel ran.

Performance routing is measurement-driven: SDPA handles a short unmasked
float32 corner where it was 12%-13% faster in controlled alternating tests. It
also guards six-layer causal and batch-above-8 cases after rigorous testing
found rare custom-kernel tolerance misses there. Low precision, unsupported
head widths, and very large causal batches use exact reference-style math.
Triton remains active for the organizer default and validated custom regimes.

For eager CUDA float32 shapes through d_model=512, FlashTile also caches a
derived packed QKV weight and replaces three projection GEMMs with one. Cache
signatures invalidate on weight, device, or dtype changes, while the original
parameter names and strict state dict remain unchanged.

## Measured outcome

On an NVIDIA GeForce RTX 5070 Ti under native Windows 11:

- all 13 executable organizer-published final rows passed five trials each,
  with zero failures across 938,885,120 comparisons;
- the exact 100,000-token final resource row was recorded separately and was
  not counted as a pass;
- fresh final-matrix geometric-mean speedup was 1.912x, with Campaign 5 rows 6
  and 7 at 1.503x/1.524x, the row-11 target at 5.948x, and row 13 at 4.780x;
- EXP-001 improved paired full-matrix geomean by 8.98% and 10.19%, while
  the current target profile remains 91.11% below its frozen baseline;
- EXP-003 improved the post-EXP-001 final-matrix geomean by 6.95% and reduced
  row-1 attention time by 69.98% across 40 Triton launches;
- Campaign 3's 32x32 short-`head_dim=128` tile reduced row-9 attention time by
  55.45% across 40 Triton launches and raised final geomean another 1.96%;
- Campaign 4's zero-padded 64x64 `head_dim=8` Triton path reduced the fresh
  row-11 optimized median by 81.08%, passed three confirmations, and raised
  final geomean another 14.42%;
- Campaign 5's layer-aware hybrids reduced row-6 and row-7 ten-step model time
  by 19.74% and 33.64% from their fresh reference profiles and raised the full
  final geomean 7.62% over the Campaign 5 baseline;
- the untouched organizer PyTorch default six-layer harness passed 5/5 trials
  with zero failed elements and measured 1.397x median speedup;
- all 1,950 optimized attention calls in that organizer run used Triton;
- all 28 feasible source-derived exact-harness cases passed five trials each,
  with 0 failed elements across 459,776,000 comparisons;
- the source-designated 100,000-token quadratic stress case was recorded as a
  resource skip and was not counted as a pass;
- two seven-case project-held-out matrices passed with zero failed elements and
  measured 1.447x and 1.450x geomean speedup; exact-shape SDPA removed both
  former long-causal regressions, with primary results of 1.247x and 1.280x; and
- the long-attention incremental peak allocation fell from 78 MiB to 22 MiB.

The result artifacts contain raw CUDA-event samples, environment/revision
metadata, implementation SHA-256, memory measurements, and profiler proof that
the `_attention_fwd` kernel executed.

## Impact and relevance

Attention's quadratic intermediates create latency and memory pressure in
real Transformer inference. On the longest held-out case, FlashTile reduced
incremental allocation by 71.8%; on the published final dimensions it delivered
a 1.912x geometric mean. The same design can increase serving
capacity, leave memory headroom for longer contexts or larger batches, and
reduce per-request compute time without changing model weights.

## Development tools

- Windows PowerShell; native Windows CUDA/Triton and an earlier WSL test path
- Git and GitHub
- Python, pytest, and Jupyter/Google Colab
- PyTorch profiler and CUDA events
- OpenAI Codex for audit, implementation, testing, profiling, and documentation
- Claude Code for the initial SDPA/LayerNorm prototype, as attributed by the
  pre-existing repository report

## Libraries and frameworks

- PyTorch 2.13.0+cu130
- Triton 3.7.1
- CUDA 13.0 runtime
- NumPy 2.5.2
- pytest 9.1.1

No external web API or hosted model API is required at runtime.

## Data and assets

The benchmark uses deterministic synthetic tensors generated from recorded
seeds. No external dataset or third-party model weights are used. Challenge
requirements and both supplied benchmark scripts are retained byte-for-byte in
the repository with SHA-256 checksums. PyTorch is the selected framework; the
TensorFlow file is preserved as the allowed alternative and shape-scope audit.

## Engineering choices

- Vendor GEMMs remain in PyTorch/cuBLAS; custom code targets the attention
  bottleneck instead of replacing mature matrix multiplication kernels.
- Float32 is the primary optimized path because it is the checked-in benchmark
  default and satisfies its strict tolerance across deep stacks.
- Packed QKV is enabled only through d_model=512, where target-device
  measurements justified its bounded derived-weight memory cost.
- Direct fp16 attention is tested, while automatic fp16/bf16 deep-stack runs
  prioritize exact reference-style correctness.
- A standalone Triton LayerNorm was removed after measuring only 0.46x-0.69x
  native CUDA performance.
- The benchmark runner fails closed for numerical failure, OOM, unexpected
  exceptions, and zero-case runs.

## Limitations and future work

The final dimensions are published, but dtype, padding, timing, tolerance, and
backward policy are unstated; the evidence records the selected PyTorch
assumptions. The kernel is forward only and tuned on the RTX 5070 Ti. Future
work is to retest unchanged on the evaluator GPU and consider adjacent fusion
only when a new profile demonstrates enough end-to-end ceiling.

## Links and submission notes

- Code: https://github.com/lukeai-tan/tiktok-techjam-2026
- Technical evidence: `docs/TECH_REPORT.md` and `docs/results/`
- Requirement audit: `docs/TRACK3_COMPLIANCE.md`
- Demo: follow `DEMO_RUNBOOK.md`; add the final public YouTube URL to Devpost
  after the human recording/upload step.

Before submission, make the repository public and verify the code link in a
signed-out browser.

Repository evidence does not establish additional human team-member
attribution. The submitter should add participant names and contributions on
Devpost if applicable.
