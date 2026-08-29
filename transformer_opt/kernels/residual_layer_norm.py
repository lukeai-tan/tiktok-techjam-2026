"""Fused inference-only residual addition and LayerNorm."""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR: Optional[Exception] = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - CPU env
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc


if triton is not None:

    @triton.jit
    def _residual_layer_norm_fwd(
        x_ptr,
        update_ptr,
        weight_ptr,
        bias_ptr,
        valid_mask_ptr,
        residual_out_ptr,
        normalized_out_ptr,
        eps,
        N_COLS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_VALID_MASK: tl.constexpr,
        ZERO_INVALID_RESIDUAL: tl.constexpr,
        ZERO_INVALID_NORMALIZED: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK_SIZE)
        offsets = row * N_COLS + columns
        in_bounds = columns < N_COLS

        x = tl.load(x_ptr + offsets, mask=in_bounds, other=0.0).to(tl.float32)
        update = tl.load(
            update_ptr + offsets,
            mask=in_bounds,
            other=0.0,
        ).to(tl.float32)
        residual = x + update

        row_valid = True
        if HAS_VALID_MASK:
            row_valid = tl.load(valid_mask_ptr + row)
        if ZERO_INVALID_RESIDUAL:
            residual = tl.where(row_valid, residual, 0.0)

        tl.store(residual_out_ptr + offsets, residual, mask=in_bounds)

        mean = tl.sum(residual, axis=0) / N_COLS
        centered = tl.where(in_bounds, residual - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / N_COLS
        normalized = centered * tl.rsqrt(variance + eps)
        weight = tl.load(weight_ptr + columns, mask=in_bounds, other=0.0).to(
            tl.float32
        )
        normalized *= weight
        if HAS_BIAS:
            bias = tl.load(bias_ptr + columns, mask=in_bounds, other=0.0).to(
                tl.float32
            )
            normalized += bias
        if ZERO_INVALID_NORMALIZED:
            normalized = tl.where(row_valid, normalized, 0.0)
        tl.store(normalized_out_ptr + offsets, normalized, mask=in_bounds)


def fused_residual_layer_norm(
    x: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
    valid_token_mask: Optional[torch.Tensor] = None,
    *,
    zero_invalid_residual: bool = False,
    zero_invalid_normalized: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(x + update, layer_norm(x + update))`` in one GPU kernel.

    The exact-row model route supplies contiguous float32 tensors with a single
    normalized feature dimension. Unsupported calls fail closed rather than
    silently changing the public Transformer contract.
    """
    if triton is None:
        detail = f": {_TRITON_IMPORT_ERROR}" if _TRITON_IMPORT_ERROR else ""
        raise RuntimeError(f"Triton is not installed{detail}")
    if x.shape != update.shape or x.ndim < 2:
        raise ValueError("x and update must have the same rank-two-or-higher shape")
    if x.device.type != "cuda" or update.device != x.device:
        raise ValueError("fused residual LayerNorm requires co-located CUDA tensors")
    if x.dtype is not torch.float32 or update.dtype is not x.dtype:
        raise ValueError("fused residual LayerNorm supports float32 only")
    if not x.is_contiguous() or not update.is_contiguous():
        raise ValueError("fused residual LayerNorm requires contiguous inputs")
    if torch.is_grad_enabled() and (x.requires_grad or update.requires_grad):
        raise ValueError("fused residual LayerNorm is forward-inference only")

    width = x.shape[-1]
    if weight.shape != (width,) or weight.device != x.device or weight.dtype != x.dtype:
        raise ValueError("weight must match the normalized feature dimension")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous")
    if bias is not None:
        if bias.shape != (width,) or bias.device != x.device or bias.dtype != x.dtype:
            raise ValueError("bias must match the normalized feature dimension")
        if not bias.is_contiguous():
            raise ValueError("bias must be contiguous")
    if (zero_invalid_residual or zero_invalid_normalized) and valid_token_mask is None:
        raise ValueError("invalid-row zeroing requires valid_token_mask")
    if valid_token_mask is not None:
        if valid_token_mask.shape != x.shape[:-1]:
            raise ValueError("valid_token_mask must match every non-feature dimension")
        if valid_token_mask.dtype is not torch.bool or valid_token_mask.device != x.device:
            raise ValueError("valid_token_mask must be a co-located bool tensor")
        if not valid_token_mask.is_contiguous():
            raise ValueError("valid_token_mask must be contiguous")

    block_size = triton.next_power_of_2(width)
    if block_size > 65536:
        raise ValueError("normalized feature dimension exceeds the fused kernel limit")
    rows = x.numel() // width
    residual_out = torch.empty_like(x)
    normalized_out = torch.empty_like(x)
    mask_arg = valid_token_mask if valid_token_mask is not None else x
    bias_arg = bias if bias is not None else weight
    num_warps = 4 if block_size <= 1024 else 8

    _residual_layer_norm_fwd[(rows,)](
        x,
        update,
        weight,
        bias_arg,
        mask_arg,
        residual_out,
        normalized_out,
        float(eps),
        N_COLS=width,
        BLOCK_SIZE=block_size,
        HAS_BIAS=bias is not None,
        HAS_VALID_MASK=valid_token_mask is not None,
        ZERO_INVALID_RESIDUAL=zero_invalid_residual,
        ZERO_INVALID_NORMALIZED=zero_invalid_normalized,
        num_warps=num_warps,
        num_stages=1,
    )
    return residual_out, normalized_out
