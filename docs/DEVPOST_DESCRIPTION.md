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

Performance routing is measurement-driven: SDPA handles the two short,
unmasked float32 cases where it was 12%-13% faster in controlled alternating
tests. It also guards six-layer causal and batch-above-8 cases after rigorous
testing found rare custom-kernel tolerance misses there. Triton remains active
for the organizer default and validated smaller masked/long/wider-head cases.

For eager CUDA float32 shapes through d_model=512, FlashTile also caches a
derived packed QKV weight and replaces three projection GEMMs with one. Cache
signatures invalidate on weight, device, or dtype changes, while the original
parameter names and strict state dict remain unchanged.

## Measured outcome

On an NVIDIA GeForce RTX 5070 Ti under native Windows 11:

- the untouched organizer PyTorch default six-layer harness passed 5/5 trials
  with zero failed elements and measured 1.411x median speedup;
- all 1,950 optimized attention calls in that organizer run used Triton;
- all 28 feasible source-derived exact-harness cases passed five trials each,
  with 0 failed elements across 459,776,000 comparisons;
- the source-designated 100,000-token quadratic stress case was recorded as a
  resource skip and was not counted as a pass;
- 7/7 provisional matrix cases passed;
- 0 failed elements across 35 trials and 13,117,440 checked elements;
- maximum absolute error was 0.000992358 under the stricter executable rule;
- end-to-end speedup ranged from 1.230x to 1.752x;
- provisional-matrix geometric-mean speedup was 1.501x; and
- the long-attention incremental peak allocation fell from 78 MiB to 22 MiB.

The result artifacts contain raw CUDA-event samples, environment/revision
metadata, implementation SHA-256, memory measurements, and profiler proof that
the `_attention_fwd` kernel executed.

## Impact and relevance

Attention's quadratic intermediates create latency and memory pressure in
real Transformer inference. On the longest measured case, FlashTile reduced
incremental allocation by 71.8%; across the full matrix it improved median
end-to-end latency for every shape. The same design can increase serving
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

The supplied PyTorch and TensorFlow scripts are reconciled, but the final
PyTorch evaluator shape matrix was not included, so the broader project matrix
is labelled provisional. The kernel is forward only and tuned on the RTX 5070
Ti. Future work is to run the final evaluator unchanged, retune on its GPU, and
consider adjacent fusion only when a new profile demonstrates enough
end-to-end ceiling.

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
