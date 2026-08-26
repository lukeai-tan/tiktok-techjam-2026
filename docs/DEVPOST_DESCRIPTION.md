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

## Measured outcome

On an NVIDIA GeForce RTX 5070 Ti under WSL2:

- 7/7 provisional matrix cases passed;
- 0 failed elements across 35 trials and 13,117,440 checked elements;
- maximum absolute error was 0.000997663 under the stricter executable rule;
- end-to-end speedup ranged from 1.138x to 1.566x;
- geometric-mean speedup was 1.360x; and
- the long-attention incremental peak allocation fell from 78 MiB to 22 MiB.

The result artifacts contain raw CUDA-event samples, environment/revision
metadata, implementation SHA-256, memory measurements, and profiler proof that
the `_attention_fwd` kernel executed.

## Development tools

- Windows PowerShell and Ubuntu on WSL2
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
requirements and supplied benchmark assets are retained in the repository.

## Engineering choices

- Vendor GEMMs remain in PyTorch/cuBLAS; custom code targets the attention
  bottleneck instead of replacing mature matrix multiplication kernels.
- Float32 is the primary end-to-end custom path because it is the checked-in
  benchmark default and satisfies its strict tolerance across deep stacks.
- Direct fp16 attention is tested, while automatic fp16/bf16 deep-stack runs
  prioritize exact reference-style correctness.
- A standalone Triton LayerNorm was removed after measuring only 0.46x-0.69x
  native CUDA performance.
- The benchmark runner fails closed for numerical failure, OOM, unexpected
  exceptions, and zero-case runs.

## Limitations and future work

The organizer's final shape matrix was not present when this evidence was
captured, so the current matrix is labelled provisional. The kernel is forward
only and tuned on the RTX 5070 Ti. Future work is to reconcile the final
organizer harness, retune on the evaluation GPU, and consider adjacent fusion
only when a new profile demonstrates enough end-to-end ceiling.

## Links and submission notes

- Code: https://github.com/lukeai-tan/tiktok-techjam-2026
- Technical evidence: `docs/TECH_REPORT.md` and `docs/results/`
- Demo: follow `DEMO_RUNBOOK.md`; add the final public YouTube URL to Devpost
  after the human recording/upload step.

Repository evidence does not establish additional human team-member
attribution. The submitter should add participant names and contributions on
Devpost if applicable.
