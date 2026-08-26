"""Auditable custom-attention dispatch with a PyTorch SDPA fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from transformer_opt.config import ATTENTION_BACKENDS, triton_attention_support
from transformer_opt.kernels.attention import triton_attention, triton_available


@dataclass(frozen=True)
class AttentionDispatch:
    requested: str
    selected: str
    reason: str


def _validate_backend(backend: str) -> None:
    if backend not in ATTENTION_BACKENDS:
        raise ValueError(
            f"attention backend must be one of {ATTENTION_BACKENDS}, got {backend!r}"
        )


def _sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
    scale: Optional[float],
) -> torch.Tensor:
    # SDPA consumes [B, H, S, D]. The projection path produces [B, S, H, D].
    qh = q.transpose(1, 2)
    kh = k.transpose(1, 2)
    vh = v.transpose(1, 2)
    attention_mask: Optional[torch.Tensor] = None
    use_is_causal = causal

    if valid_token_mask is not None:
        attention_mask = valid_token_mask[:, None, None, :]
        if causal:
            seq_len = q.shape[1]
            causal_keep = torch.ones(
                (seq_len, seq_len), dtype=torch.bool, device=q.device
            ).tril()
            attention_mask = attention_mask & causal_keep[None, None]
            use_is_causal = False

    output = F.scaled_dot_product_attention(
        qh,
        kh,
        vh,
        attn_mask=attention_mask,
        is_causal=use_is_causal,
        scale=scale,
    )
    return output.transpose(1, 2).contiguous()


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
    scale: Optional[float],
) -> torch.Tensor:
    """Reproduce the checked-in explicit attention for sensitive dtypes."""
    qh = q.transpose(1, 2).contiguous()
    kh = k.transpose(1, 2).contiguous()
    vh = v.transpose(1, 2).contiguous()
    scores = torch.matmul(qh, kh.transpose(-2, -1))
    scores = scores * (q.shape[-1] ** -0.5 if scale is None else scale)
    if causal:
        seq_len = q.shape[1]
        future = torch.ones(
            (seq_len, seq_len), dtype=torch.bool, device=q.device
        ).triu(diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
    if valid_token_mask is not None:
        scores = scores.masked_fill(
            ~valid_token_mask[:, None, None, :],
            float("-inf"),
        )
    probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
    output = torch.matmul(probabilities, vh)
    return output.transpose(1, 2).contiguous()


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor] = None,
    *,
    causal: bool = False,
    scale: Optional[float] = None,
    backend: str = "auto",
) -> Tuple[torch.Tensor, AttentionDispatch]:
    """Execute attention and return both output and the actual dispatch record."""
    _validate_backend(backend)
    support = triton_attention_support(q, k, v, valid_token_mask)
    triton_ready = triton_available()

    if backend == "triton":
        if not triton_ready:
            raise RuntimeError("Triton attention was forced but Triton/CUDA is unavailable")
        if not support.supported:
            raise ValueError(f"Triton attention was forced but unsupported: {support.reason}")
        output = triton_attention(
            q,
            k,
            v,
            valid_token_mask,
            causal=causal,
            scale=scale,
        )
        return output, AttentionDispatch("triton", "triton", support.reason)

    if backend == "auto" and triton_ready and support.supported:
        output = triton_attention(
            q,
            k,
            v,
            valid_token_mask,
            causal=causal,
            scale=scale,
        )
        return output, AttentionDispatch("auto", "triton", support.reason)

    if backend == "reference" or (backend == "auto" and q.dtype is torch.bfloat16):
        reason = (
            "explicit reference backend"
            if backend == "reference"
            else "bfloat16 uses reference math to satisfy the executable tolerance"
        )
        output = _reference_attention(q, k, v, valid_token_mask, causal, scale)
        return output, AttentionDispatch(backend, "reference", reason)

    reason = "explicit SDPA backend"
    if backend == "auto":
        reason = support.reason if triton_ready else "Triton/CUDA is unavailable"
    output = _sdpa_attention(q, k, v, valid_token_mask, causal, scale)
    return output, AttentionDispatch(backend, "sdpa", reason)
