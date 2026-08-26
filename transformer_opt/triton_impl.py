"""Optional Triton GPU backend: a fused LayerNorm kernel.

Import-safe on machines without Triton or a GPU: the Triton import is guarded
and ``available()`` returns False, so callers fall back to ``nn.LayerNorm``.

Opt in from the benchmark with ``TRANSFORMER_OPT_TRITON_LN=1``. The kernel is a
single-pass mean/variance/affine LayerNorm (no intermediate tensors). It is
GPU-only and unrun on the CPU dev machine; the benchmark's accuracy check gates
it before any timing is trusted.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_OK = True
except Exception:  # pragma: no cover - triton not installed
    _TRITON_OK = False


def available() -> bool:
    return _TRITON_OK and torch.cuda.is_available()


if _TRITON_OK:

    @triton.jit
    def _layernorm_fwd(X, Y, W, B, stride, N, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        X += row * stride
        Y += row * stride
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / N
        xc = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + cols, xc * rstd * w + b, mask=mask)

    def fused_layernorm(x: torch.Tensor, weight: torch.Tensor,
                        bias: torch.Tensor, eps: float) -> torch.Tensor:
        d = x.shape[-1]
        x2d = x.reshape(-1, d)
        y = torch.empty_like(x2d)
        block = triton.next_power_of_2(d)
        _layernorm_fwd[(x2d.shape[0],)](
            x2d, y, weight, bias, x2d.stride(0), d, eps, BLOCK=block,
        )
        return y.view_as(x).to(x.dtype)

else:  # pragma: no cover

    def fused_layernorm(x, weight, bias, eps):
        raise RuntimeError("Triton is not available")
