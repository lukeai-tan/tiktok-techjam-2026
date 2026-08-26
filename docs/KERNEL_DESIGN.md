# Triton Attention Kernel Design

## Purpose

`transformer_opt/kernels/attention.py` implements the repository-owned forward
scaled dot-product attention kernel. It replaces the reference sequence of QK
matmul, score tensor, softmax, and P@V matmul with one tiled launch. QKV, output,
and FFN projections remain PyTorch/cuBLAS GEMMs because profiling shows those
are already efficient vendor-kernel work.

## Interface and layout

The wrapper accepts Q, K, and V with shape `[batch, sequence, heads, head_dim]`.
That is a view of each contiguous projection output, so the optimized model does
not transpose and copy three projection tensors before attention. Arbitrary
batch/sequence/head strides are passed to Triton; the final head dimension must
have stride 1.

The output uses the same contiguous BSHD layout and is reshaped directly to
`[batch, sequence, d_model]` for the output projection.

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

Softmax state and the weighted-value accumulator are fp32. Float32 dot products
follow `torch.backends.cuda.matmul.allow_tf32`: TF32 when the benchmark enables
it, IEEE otherwise.

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
to SDPA, while the Transformer adds the low-precision reference policy above.
`reference` and `sdpa` can be forced for controlled comparisons.

## Launch policy

The measured fixed policy avoids per-process autotuning overhead:

| head dimension | sequence | BLOCK_M | BLOCK_N | warps | stages |
| ---: | ---: | ---: | ---: | ---: | ---: |
| <= 64 | <= 128 | 64 | 128 | 4 | 2 |
| <= 64 | 129-512 | 64 | 64 | 4 | 2 |
| <= 64 | > 512 | 64 | 64 | 4 | 3 |
| 128 | any | 32 | 64 | 4 | 2 |

Short sequences use one K/V tile where practical, reducing both loop overhead
and online-rescaling drift.

## Measured design decisions

- The seven-case RTX 5070 Ti matrix used the custom kernel for every float32
  timing call and measured a 1.360x geomean end-to-end speedup.
- The inherited standalone Triton LayerNorm was benchmarked at only 0.46x to
  0.69x the native CUDA LayerNorm across representative widths and was removed.
- In the causal-padding profile, native LayerNorm accounted for about 116 us
  across five forwards versus 6,531 us for the profiled model range. The small
  share and slower standalone kernel did not justify residual/LayerNorm fusion
  risk for this iteration.
- Profiler evidence records `_attention_fwd` ten times for five two-layer
  forwards, matching dispatch counts exactly.

## Remaining limitations

- There is no backward kernel.
- The current shape matrix is provisional pending organizer publication.
- Support beyond sequence 8192 or the declared head dimensions is unvalidated.
- Fixed launch parameters are tuned only on the RTX 5070 Ti; other GPUs remain
  correct through guarded fallback but may need different performance routing.
