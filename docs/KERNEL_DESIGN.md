# Triton Attention Kernel Design

## Purpose

`transformer_opt/kernels/attention.py` implements the repository-owned forward
scaled dot-product attention kernel. It replaces the reference sequence of QK
matmul, score tensor, softmax, and P@V matmul with one tiled launch. Projection
and FFN math remains in PyTorch/cuBLAS; measured eager-fp32 shapes through
`d_model=512` combine the three Q/K/V projections into one vendor GEMM.

## Interface and layout

The wrapper accepts Q, K, and V with shape `[batch, sequence, heads, head_dim]`.
They are views of projection output, so the optimized model does not transpose
and copy three tensors before attention. Arbitrary batch/sequence/head strides
are passed to Triton; the final head dimension must have stride 1.

The output uses the same contiguous BSHD layout and is reshaped directly to
`[batch, sequence, d_model]` for the output projection.

## Packed QKV projection

For eager CUDA float32 inference with `d_model <= 512`, the optimized model
concatenates the existing Q/K/V weights and biases once, caches those derived
non-persistent tensors, and performs one `F.linear` call. The result is viewed
as `[B,S,3,H,D]`; `unbind` produces strided Q/K/V views consumed by the selected
attention backend without materializing three separate projection outputs.

The cache signature includes every source parameter's data pointer, mutation
version, device, and dtype. A load, in-place update, or device/dtype move
therefore rebuilds the packed tensors. The cache is a plain derived-data
dictionary, so state-dict keys remain byte-for-byte compatible with the
baseline. Training, compilation, CPU, low precision, and `d_model > 512` keep
the original three-projection path. The widest measured model showed no packed
projection benefit.

## Tiling and online softmax

The launch grid is:

```text
(ceil_div(sequence, BLOCK_M), batch * heads)
```

Each program loads a `BLOCK_M x head_dim` Q tile and visits K/V in
`BLOCK_N x head_dim` tiles. For every query row it maintains fp32 state:

```text
m_i = running maximum
l_i = running sum of exp(score - m_i)
acc = running weighted-value accumulator
```

For a new score tile with maximum `m_tile`:

```text
m_new = max(m_i, m_tile)
alpha = exp(m_i - m_new)
acc = acc * alpha + exp(scores - m_new) @ V
l_i = l_i * alpha + sum(exp(scores - m_new))
```

The final output is `acc / l_i`. Triton `exp2` is used after folding
`log2(e)` into scores. This is FlashAttention-style online softmax: the kernel
never stores a `[B,H,S,S]` score or probability tensor.

## Masks and boundary safety

- Sequence-tail masks guard every Q/K/V load and output store.
- `valid_token_mask[B,S]` is loaded once per K tile and broadcast over query
  rows; invalid keys receive negative infinity.
- Causal mode adds `key_index <= query_index` inside the same score mask.
- An all-masked tile contributes zero without evaluating `exp(-inf - -inf)`.
- An entirely masked row returns finite zeros.
- The optimized Transformer still zeroes invalid output rows after every block,
  matching the reference.

No dense causal or combined causal-padding tensor is created on the custom
path.

## Numerical strategy

The executable benchmark is stricter than the prose brief. To stay close to its
explicit implementation, the kernel reproduces two important score roundings:

1. QK is cast to the input dtype, matching the matmul output tensor.
2. The scaled score is cast to that dtype again before fp32 softmax.

Softmax state and the weighted-value accumulator are fp32. Non-causal float32
dot products follow `torch.backends.cuda.matmul.allow_tf32`. Causal fused
attention always uses IEEE fp32 dot products: the final evaluator shapes exposed
rare TF32 misses after four causal layers under the executable zero-failure
comparator.

The primary end-to-end path is float32. Direct fp16 attention tests pass the
executable tolerance, but tiny differences compound through deep low-precision
stacks; the model's `auto` policy therefore uses exact reference-style math for
CUDA fp16/bf16. Forced `triton` remains available for direct experiments and is
always correctness-gated by the benchmark.

## Support envelope

Custom execution requires:

- CUDA inference tensors on compute capability 8.0 or newer;
- identical Q/K/V shapes, dtype, and device;
- float32 or float16;
- `head_dim` in `{16, 32, 64, 128}`;
- `1 <= sequence <= 8192`;
- last-dimension stride 1;
- optional boolean mask of shape `[B,S]` on the same device;
- no requested gradients.

`triton` mode rejects unsupported input. `auto` routes unsupported direct calls
to SDPA. The multi-layer Transformer uses exact reference-style attention for
low precision, head dimensions outside the custom support set, and causal
batches above 128 because target-GPU final-shape validation exposed rare strict
tolerance misses in those regimes.
On the RTX 5070 Ti, controlled alternating measurements also showed SDPA was
12%-13% faster for unmasked, non-causal float32 sequences <=128 with head
dimension <=32. `auto` uses SDPA for that launch-bound corner and Triton for the
other validated fp32 regimes. `reference`, `sdpa`, and `triton` can be forced
for controlled comparisons.

## Launch policy

The measured fixed policy avoids per-process autotuning overhead:

| head dimension | sequence | BLOCK_M | BLOCK_N | warps | stages |
| ---: | ---: | ---: | ---: | ---: | ---: |
| <= 32 | <= 128 | 64 | 128 | 4 | 2 |
| 64 | <= 128 | 32 | 64 | 4 | 2 |
| <= 64 | 129-512 | 64 | 64 | 4 | 2 |
| <= 64 | > 512 | 64 | 64 | 4 | 3 |
| 128 | any | 32 | 64 | 4 | 2 |

Short head-dimension-32 sequences use one K/V tile where practical, reducing
both loop overhead and online-rescaling drift. Head-dimension-64 short sequences
use smaller tiles to avoid the register spilling measured with IEEE fp32 dots on
the target GPU.

## Measured design decisions

- The organizer-published final matrix passed all 13 executable rows with zero
  failed elements and one source-authorized resource skip. It delivered a
  1.427x geomean end-to-end speedup; 1,008 attention calls used Triton and 448
  unsupported or very-large-batch calls used explicit reference math.
- EXP-001 targeted the final row-10 `head_dim=64`, sequence-128 spill bottleneck.
  The former 64x128 tile reported 2,468 spills and 81,920 bytes of shared memory;
  the accepted 32x64 tile reported two spills and 49,152 bytes. Across two paired
  full-matrix trials, aggregate speedup improved by 8.98% and 10.19%.
- After integration, row 10 measured 1.701x end-to-end. Its `_attention_fwd`
  profiler time fell from 30,324.486 us to 3,205.548 us across 40 launches, an
  89.43% reduction, with Triton handling all 40 calls.
- Exact-harness stress testing found rare strict-tolerance misses when Triton
  differences accumulated through six causal layers or batches above eight.
  Auto routes those deep-stack regimes to SDPA; all 28 feasible source-derived
  cases then passed across 459,776,000 compared elements, while the organizer
  default remains on Triton.
- Packed QKV reduced the two-layer profile from the architectural 60 `addmm`
  calls to 40 across five forwards. Isolated projection measurements were
  bit-identical and improved most in overhead-bound and medium shapes.
- The inherited standalone Triton LayerNorm was benchmarked at only 0.46x to
  0.69x the native CUDA LayerNorm across representative widths and was removed.
- In the current causal-padding profile, native LayerNorm accounted for about
  135 us across five forwards versus 6,333 us for the profiled model range. The small
  share and slower standalone kernel did not justify residual/LayerNorm fusion
  risk for this iteration.
- A causal loop-frontier prune and alternate tile/stage configurations were
  tested on the target. Neither improved the full end-to-end matrix, so the
  simpler fixed loop and prior launch policy were retained.
- Profiler evidence records `_attention_fwd` ten times for five two-layer
  forwards, matching dispatch counts exactly.

## Remaining limitations

- There is no backward kernel.
- The final table omits dtype, padding, timing, tolerance, and backward policy;
  current validation records the selected PyTorch defaults as assumptions.
- Support beyond sequence 8192 or the declared head dimensions is unvalidated.
- Packed QKV trades bounded persistent memory for fewer launches/GEMMs (about
  6 MiB for two float32 d_model=512 layers) and is disabled outside its measured
  winning envelope.
- Fixed launch parameters are tuned only on the RTX 5070 Ti; other GPUs remain
  correct through guarded fallback but may need different performance routing.
