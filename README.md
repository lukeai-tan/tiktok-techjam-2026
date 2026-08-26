# Transformer Layer GPU Optimization — TikTok TechJam 2026

Optimize the runtime of a Transformer encoder stack on a target GPU while
keeping the output numerically equivalent to the reference PyTorch baseline.

The official benchmark script (`torch_transformer_benchmark.py`) is the single
source of truth: it defines the baseline, copies identical weights into both
models, and compares them under the competition's tolerance. Our work is the
`UserOptimizedTransformer` class inside that script.

## The workload

A pre-LayerNorm Transformer encoder, `num_layers` blocks then a final norm:

```
# per block:
x = x + Attention(norm1(x))                       # MHA: q/k/v/out projections
x = x + ffn_out(GELU(ffn_in(norm2(x)), 'none'))   # position-wise FFN
# after all blocks:
x = final_norm(x)
```

with optional **causal** masking and a **`valid_token_mask`** (padding): invalid
key positions are masked out and padded output rows are zeroed.

### Correctness rule (from the script)

Per element, the optimized output passes when

```
abs(opt - ref) <= atol   OR   abs(opt - ref) <= rtol * abs(ref)
```

with **`atol = 0.001`, `rtol = 0.01`** (the script's defaults; stricter than the
2%/0.002 stated in the prompt PDF — we target the stricter one). A raw relative
error is meaningless where `ref ≈ 0`, which is exactly why the rule is an OR with
an absolute floor.

## Optimizations (all correctness-preserving)

1. **Fused scaled-dot-product attention** (`F.scaled_dot_product_attention`) —
   dispatches to FlashAttention / memory-efficient kernels on CUDA and never
   materializes the `(S, S)` score matrix. Masks are folded to keep the fast
   path whenever possible:
   - no padding, non-causal → no mask
   - no padding, causal → `is_causal=True` (flash)
   - padding, non-causal → cheap `[B,1,1,S]` key-padding mask
   - padding, causal → combined `[B,1,S,S]` boolean mask
2. **Exact structural match** — pre-norm residuals, exact-erf GELU, padded-row
   zeroing reproduced so valid rows are computed identically to the baseline.
3. **Strict-weight compatibility** — `UserOptimizedTransformer` subclasses the
   baseline, inheriting identical submodule/parameter names, so the harness's
   `copy_model_weights(..., strict=True)` succeeds unchanged.
4. **Optional Triton fused LayerNorm** (`transformer_opt/triton_impl.py`,
   opt-in via `TRANSFORMER_OPT_TRITON_LN=1`) — single-pass mean/var/affine, GPU
   only, gated behind the accuracy check.
5. **Reduced precision (fp16/bf16)** on GPU to engage tensor cores; the tolerance
   permits it and the harness verifies it.
6. `torch.compile` is available in the script (`--compile-user`) as a stacking
   win on top of the above.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate

# Target GPU machine (match your driver's CUDA version):
pip install torch --index-url https://download.pytorch.org/whl/cu124 numpy pytest

# CPU-only development / correctness:
pip install torch --index-url https://download.pytorch.org/whl/cpu numpy pytest
```

## Quick start on a free GPU (Google Colab)

No local GPU needed. Open `notebooks/colab_benchmark.ipynb` in
[Google Colab](https://colab.research.google.com/), set
`Runtime → Change runtime type → GPU`, and run the cells. It uploads the two
files it needs (`torch_transformer_benchmark.py`, `sweep.py`), verifies the GPU,
then runs the accuracy check and the shape sweep in fp16/bf16. Colab already has
CUDA PyTorch + Triton, so there is nothing to install.

## Reproduce results

```bash
# Correctness (CPU-friendly): optimized vs baseline across shapes/causal/padding
pytest tests/ -q

# Grid sweep (auto-selects CUDA+fp16 on a GPU, CPU+fp32 otherwise):
python sweep.py                    # full grid
python sweep.py --quick            # tiny grid, fast sanity check
python sweep.py --dtype bfloat16   # or --compile / --causal / --triton-ln

# Official benchmark, GPU with tensor cores (the real deliverable):
python torch_transformer_benchmark.py --device cuda --dtype float16 \
    --batch-size 8 --seq-len 512 --d-model 512 --heads 8 \
    --ffn-dim 2048 --layers 6

# Long-sequence stress (where fused attention wins most):
python torch_transformer_benchmark.py --device cuda --dtype bfloat16 --seq-len 2048

# Causal + padding path:
python torch_transformer_benchmark.py --device cuda --dtype float16 \
    --causal --padding-ratio 0.4

# Optional: enable Triton fused LayerNorm and/or torch.compile
TRANSFORMER_OPT_TRITON_LN=1 python torch_transformer_benchmark.py --device cuda \
    --dtype float16 --compile-user --compile-mode max-autotune
```

The script prints per-trial accuracy (`failed=0/...` ⇒ pass) then baseline vs
optimized median latency, throughput (token/s), and the speedup.

## Baseline results (CPU, fp32 — development machine, no GPU)

CPU numbers only prove correctness and that the fused path already helps; **the
GPU numbers are the real deliverable** (see `docs/TECH_REPORT.md`).

| config | accuracy | max_abs | baseline | optimized | speedup |
| --- | --- | --- | --- | --- | --- |
| B4 S512 d512 H8 ffn2048 L6 | PASS 0 failed | 1.9e-06 | 651.9 ms | 431.0 ms | **1.51x** |
| B8 S256 d256 H8 ffn1024 L4 causal+pad0.4 | PASS 0 failed | 1.4e-06 | 150.5 ms | 85.2 ms | **1.77x** |

`pytest tests/` — 22 passed (shapes × causal × padding, plus strict-copy guard).

## Limitations & next steps

- Triton fused-LayerNorm is **GPU-only and unrun on the CPU dev box**; opt-in and
  gated by the accuracy check. Not on by default to keep results honest.
- No hand-written fused attention — we rely on PyTorch SDPA, which is near-optimal
  for common shapes. A bespoke Triton attention (online softmax + fused output
  projection) is the next lever for unusual `(S, d, H)` shapes.
- `--compile-user` is wired but not yet tuned per shape.
- GPU result tables in the tech report are placeholders until run on hardware.

## Layout

```
torch_transformer_benchmark.py     official harness + UserOptimizedTransformer (our code)
tensorflow_transformer_benchmark.py official TF harness (unused; PyTorch track)
sweep.py                           grid benchmark: baseline vs optimized, saves results/*.json
transformer_opt/
  __init__.py
  triton_impl.py                   optional GPU fused LayerNorm
notebooks/
  colab_benchmark.ipynb            free-GPU runner (upload + sweep)
tests/
  test_correctness.py              optimized vs baseline across shapes
docs/
  TECH_REPORT.md
```
