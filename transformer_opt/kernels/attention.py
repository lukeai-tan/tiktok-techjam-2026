"""Forward fused scaled-dot-product attention implemented in Triton.

The kernel consumes the projection-friendly `[B, S, H, D]` layout and performs
QK, online softmax, and P@V without materializing an attention score matrix.
Softmax state and the output accumulator use fp32. Causal and valid-key masks
are applied while visiting K/V tiles.
"""

from __future__ import annotations

from typing import Optional

import torch

from transformer_opt.config import attention_launch_config, triton_attention_support

try:
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR: Optional[Exception] = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - CPU env
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc


def triton_available() -> bool:
    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _attention_fwd(
        q_ptr,
        k_ptr,
        v_ptr,
        mask_ptr,
        out_ptr,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_vd,
        stride_mb,
        stride_ms,
        stride_ob,
        stride_os,
        stride_oh,
        stride_od,
        num_heads,
        softmax_scale,
        N_CTX: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_MASK: tl.constexpr,
        CAUSAL: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // num_heads
        head = batch_head - batch * num_heads

        offs_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        q_offsets = (
            batch * stride_qb
            + offs_m[:, None] * stride_qs
            + head * stride_qh
            + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptr + q_offsets, mask=offs_m[:, None] < N_CTX, other=0.0)

        running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        running_sum = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

        for start_n in range(0, N_CTX, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_offsets = (
                batch * stride_kb
                + offs_n[:, None] * stride_ks
                + head * stride_kh
                + offs_d[None, :] * stride_kd
            )
            v_offsets = (
                batch * stride_vb
                + offs_n[:, None] * stride_vs
                + head * stride_vh
                + offs_d[None, :] * stride_vd
            )
            key_in_bounds = offs_n < N_CTX
            k = tl.load(k_ptr + k_offsets, mask=key_in_bounds[:, None], other=0.0)
            v = tl.load(v_ptr + v_offsets, mask=key_in_bounds[:, None], other=0.0)

            key_valid = key_in_bounds
            if HAS_MASK:
                mask_offsets = batch * stride_mb + offs_n * stride_ms
                key_valid = key_valid & tl.load(
                    mask_ptr + mask_offsets, mask=key_in_bounds, other=0
                )

            allowed = key_valid[None, :]
            if CAUSAL:
                allowed = allowed & (offs_n[None, :] <= offs_m[:, None])

            if ALLOW_TF32:
                scores = tl.dot(q, tl.trans(k), input_precision="tf32")
            else:
                scores = tl.dot(q, tl.trans(k), input_precision="ieee")
            # Match the executable reference's two dtype boundaries: its GEMM
            # produces a low-precision score tensor and its following scale
            # multiply also writes that dtype before softmax converts to fp32.
            # Preserving those roundings materially reduces multi-layer drift.
            scores = scores.to(q.dtype)
            scores = (scores * softmax_scale).to(q.dtype).to(tl.float32)
            scores *= 1.4426950408889634  # convert exp input to base 2
            scores = tl.where(allowed, scores, -float("inf"))
            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, block_max)

            # The safe value handles an all-masked tile without producing
            # exp2(-inf - -inf) NaNs. Such a tile contributes exactly zero.
            has_value = new_max != -float("inf")
            safe_max = tl.where(has_value, new_max, 0.0)
            alpha = tl.where(
                running_max != -float("inf"),
                tl.exp2(running_max - safe_max),
                0.0,
            )
            probabilities = tl.where(
                allowed,
                tl.exp2(scores - safe_max[:, None]),
                0.0,
            )

            accumulator *= alpha[:, None]
            if ALLOW_TF32:
                accumulator += tl.dot(
                    probabilities.to(v.dtype),
                    v,
                    input_precision="tf32",
                )
            else:
                accumulator += tl.dot(
                    probabilities.to(v.dtype),
                    v,
                    input_precision="ieee",
                )
            running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
            running_max = tl.where(has_value, new_max, running_max)

        normalized = accumulator / tl.where(running_sum[:, None] > 0, running_sum[:, None], 1.0)
        normalized = tl.where(running_sum[:, None] > 0, normalized, 0.0)
        out_offsets = (
            batch * stride_ob
            + offs_m[:, None] * stride_os
            + head * stride_oh
            + offs_d[None, :] * stride_od
        )
        tl.store(out_ptr + out_offsets, normalized, mask=offs_m[:, None] < N_CTX)


def triton_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor] = None,
    *,
    causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Run custom fused attention and return `[B, S, H, D]` output.

    This is a strict low-level wrapper: unsupported inputs raise instead of
    falling back. Use `transformer_opt.dispatch.attention_forward` for guarded
    automatic routing.
    """
    if triton is None:
        detail = f": {_TRITON_IMPORT_ERROR}" if _TRITON_IMPORT_ERROR else ""
        raise RuntimeError(f"Triton is not installed{detail}")

    support = triton_attention_support(q, k, v, valid_token_mask)
    if not support.supported:
        raise ValueError(f"unsupported Triton attention input: {support.reason}")

    batch, seq_len, num_heads, head_dim = q.shape
    output = torch.empty_like(q)
    launch = attention_launch_config(head_dim, seq_len)
    mask_arg = valid_token_mask if valid_token_mask is not None else q
    mask_stride_b = valid_token_mask.stride(0) if valid_token_mask is not None else 0
    mask_stride_s = valid_token_mask.stride(1) if valid_token_mask is not None else 0
    softmax_scale = head_dim**-0.5 if scale is None else float(scale)
    grid = (triton.cdiv(seq_len, launch.block_m), batch * num_heads)

    _attention_fwd[grid](
        q,
        k,
        v,
        mask_arg,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        mask_stride_b,
        mask_stride_s,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        num_heads,
        softmax_scale,
        N_CTX=seq_len,
        HEAD_DIM=head_dim,
        BLOCK_M=launch.block_m,
        BLOCK_N=launch.block_n,
        HAS_MASK=valid_token_mask is not None,
        CAUSAL=causal,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        num_warps=launch.num_warps,
        num_stages=launch.num_stages,
    )
    return output
