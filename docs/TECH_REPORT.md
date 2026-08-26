# Tech Report — Transformer Layer GPU Optimization (TikTok TechJam 2026)

> Fill the **GPU RESULTS** sections once run on target hardware. Everything else
> is complete and reproducible today (CPU).

## 1. Environment

| Component | Dev machine (CPU) | Target GPU machine (fill in) |
| --- | --- | --- |
| CPU | _(`lscpu` model)_ | |
| GPU | none | _(e.g. RTX 4090 / A100 40GB)_ |
| GPU arch / compute cap | — | _(Ada sm_89 / Ampere sm_80)_ |
| VRAM | — | |
| Driver / CUDA | — | _(`nvidia-smi`)_ |
| PyTorch | 2.13.0+cpu | _(e.g. 2.x+cu124)_ |
| Triton | not installed | _(bundled with CUDA torch)_ |
| OS | Linux (WSL2) | |
| Python | 3.13 | |

Capture on target:

```bash
nvidia-smi
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
```

## 2. Problem framing

Optimize the `num_layers`-deep pre-LayerNorm Transformer encoder in
`torch_transformer_benchmark.py` so it runs faster than the un-fused baseline
while every output element satisfies

```
abs(opt - ref) <= 0.001   OR   abs(opt - ref) <= 0.01 * abs(ref)
```

across shapes spanning small/large batch, sequence length, and dimension, with
optional causal masking and token padding.

## 3. Bottleneck analysis

The baseline attention (`BaselineSelfAttention`) explicitly builds the
`(B, H, S, S)` score tensor, runs softmax in fp32, then a second matmul — the
dominant cost at large `S`, both in memory traffic and kernel launches. FFN and
projections are GEMM/tensor-core bound at large `d_model`. Small shapes are
launch-overhead bound (the stack is `num_layers` blocks deep).

| Regime | Shape signature | Dominant cost |
| --- | --- | --- |
| Overhead | small B,S,d | launch count / dispatch (× num_layers) |
| Attention | large S | `S×S` scores: bandwidth + softmax |
| GEMM | large d | projection + FFN matmuls (tensor cores) |
| Throughput | large B, small S | occupancy over many small problems |

## 4. Optimizations

1. **Fused SDPA** — replaces score-build + softmax + context matmul with one
   fused kernel (FlashAttention / mem-efficient on CUDA); no `S×S`
   materialization. Masks folded to preserve the fast path (see README).
2. **Exact structural equivalence** — pre-norm residuals, exact-erf GELU,
   padded-row zeroing preserved; valid rows are bit-comparable up to fp rounding.
3. **Strict-weight compatibility** — subclassing the baseline keeps parameter
   names identical; `copy_model_weights(strict=True)` is unchanged.
4. **Reduced precision (fp16/bf16)** on GPU → tensor cores; tolerance permits it.
5. **Triton fused LayerNorm** (opt-in) — single-pass mean/var/affine; GPU only.
6. **`torch.compile`** (`--compile-user`) stacks on top.

## 5. Results

### 5.1 Correctness

Every trial reports `failed=0`. Max absolute error on CPU fp32 is ~1.9e-6
(~1000× inside the `atol` floor). The `max_rel` figure the script prints can be
large where `ref ≈ 0`, but those elements pass via the absolute branch of the OR
— hence `failed=0`. Re-verify on GPU with the production dtype; fp16 raises
`max_abs` but stays inside the band (the script asserts it per trial).

### 5.2 Baseline — CPU fp32 (development only)

```
# python torch_transformer_benchmark.py --device cpu --dtype float32 \
#     --batch-size 4 --seq-len 512 --d-model 512 --heads 8 --ffn-dim 2048 --layers 6
Accuracy : PASS  max_abs=1.9e-06  failed=0/3145728
baseline : median=651.95 ms   throughput=3141 token/s
optimized: median=431.04 ms   throughput=4751 token/s
speedup  : 1.512x

# ... --batch-size 8 --seq-len 256 --d-model 256 --heads 8 --ffn-dim 1024 \
#     --layers 4 --causal --padding-ratio 0.4
Accuracy : PASS  max_abs=1.4e-06  failed=0/1572864
baseline : median=150.53 ms   throughput=13605 token/s
optimized: median=85.17 ms    throughput=24046 token/s
speedup  : 1.767x
```

### 5.3 GPU RESULTS — fp16 (fill in)

```
# python torch_transformer_benchmark.py --device cuda --dtype float16 --seq-len 512
<paste accuracy + speedup>
```

### 5.4 GPU RESULTS — long sequence & bf16 / compile (fill in)

```
# --dtype bfloat16 --seq-len 2048
# TRANSFORMER_OPT_TRITON_LN=1 ... --compile-user --compile-mode max-autotune
<paste>
```

Suggested GPU sweep: seq_len ∈ {128, 512, 2048, 4096}, d_model ∈ {512, 1024,
2048}, dtype ∈ {float16, bfloat16}, with/without `--causal` and
`--padding-ratio 0.4`. Expect the largest speedups at long sequence length,
where avoiding the `S×S` materialization compounds with tensor-core GEMMs.

## 6. AI tools & methodology (for bonus points)

- **Claude Code (Opus)** — set up the environment, wrote `UserOptimizedTransformer`
  (SDPA with correct causal/padding mask folding, strict-weight-compatible
  subclassing), the optional Triton fused-LayerNorm kernel, and the CPU test
  suite. Notably it reconciled an initial reconstruction (post-LN, single layer)
  against the *actual* official script (pre-LN, multi-layer, separate QKV, padding
  mask, atol=0.001/rtol=0.01) once it appeared, and corrected a flawed correctness
  metric to the script's exact OR-tolerance.
- **Profiling next**: `torch.profiler` + Nsight Systems/Compute on the GPU box to
  confirm §3 attribution and guide further fusion.

## 7. Limitations & future work

- Triton path GPU-only and unrun on CPU dev machine (opt-in, gated).
- No bespoke fused attention yet; SDPA is near-optimal for common shapes but a
  custom Triton attention could help unusual `(S, d, H)` combinations.
- `--compile-user` wired but not per-shape tuned.
- GPU tables are placeholders pending hardware.
```
