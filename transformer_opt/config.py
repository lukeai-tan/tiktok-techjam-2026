"""Support envelope and launch policy for repository-owned GPU kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


ATTENTION_BACKENDS = ("auto", "triton", "sdpa", "reference")
SUPPORTED_HEAD_DIMS = frozenset({16, 32, 64, 128})
# Float32 is the checked-in benchmark default and follows PyTorch's TF32 toggle
# in the kernel. Float16 is retained for direct-kernel evaluation, while
# deep-stack model policy can select reference math when the strict tolerance is
# at risk.
# BF16 fused paths diverged too far from the explicit benchmark on the target.
SUPPORTED_DTYPES = frozenset({torch.float16, torch.float32})
MIN_CUDA_CAPABILITY = (8, 0)
MAX_SEQUENCE_LENGTH = 8192
SDPA_SHORT_MAX_SEQUENCE_LENGTH = 128
SDPA_SHORT_MAX_HEAD_DIM = 32


@dataclass(frozen=True)
class SupportResult:
    supported: bool
    reason: str


@dataclass(frozen=True)
class AttentionLaunchConfig:
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int


def prefer_triton_attention(
    q: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
) -> bool:
    """Return the measured RTX 5070 Ti auto-routing preference.

    PyTorch SDPA wins the launch-bound, unmasked float32 corner. Triton wins the
    measured masked, causal, longer-sequence, and wider-head regimes. Forced
    backend selection remains available independently of this performance
    policy.
    """
    short_unmasked_float32 = (
        q.dtype is torch.float32
        and q.shape[1] <= SDPA_SHORT_MAX_SEQUENCE_LENGTH
        and q.shape[-1] <= SDPA_SHORT_MAX_HEAD_DIM
        and valid_token_mask is None
        and not causal
    )
    return not short_unmasked_float32


def attention_launch_config(head_dim: int, seq_len: int) -> AttentionLaunchConfig:
    """Return a conservative launch configuration for the target GPU family."""
    if head_dim == 64 and seq_len <= 128:
        # The larger short-sequence tile spills heavily for IEEE fp32 dot
        # products on the RTX 5070 Ti. Halving both tile axes removes those
        # spills while leaving every other measured shape unchanged.
        return AttentionLaunchConfig(
            block_m=32,
            block_n=64,
            num_warps=4,
            num_stages=2,
        )
    if head_dim == 32 and seq_len <= 128:
        # Isolate K/V-tile pressure while retaining the established Q tile for
        # the official short narrow-head shapes.
        return AttentionLaunchConfig(
            block_m=64,
            block_n=64,
            num_warps=4,
            num_stages=2,
        )
    if head_dim == 128 and seq_len <= 128:
        # The measured short-sequence head-128 path benefits from a smaller K/V
        # tile, reducing tile pressure without changing the Q tile or arithmetic.
        return AttentionLaunchConfig(
            block_m=32,
            block_n=32,
            num_warps=4,
            num_stages=2,
        )
    if head_dim <= 64:
        return AttentionLaunchConfig(
            block_m=64,
            # A single K/V tile for short official-style sequences minimizes
            # online-softmax rescaling drift against the explicit reference.
            block_n=128 if seq_len <= 128 else 64,
            num_warps=4,
            num_stages=2 if seq_len <= 512 else 3,
        )
    return AttentionLaunchConfig(
        block_m=32,
        block_n=64,
        num_warps=4,
        num_stages=2,
    )


def _validate_qkv_contract(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
) -> Optional[str]:
    if q.ndim != 4:
        return "q, k, and v must have shape [batch, sequence, heads, head_dim]"
    if k.shape != q.shape or v.shape != q.shape:
        return "q, k, and v must have identical shapes"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return "q, k, and v must have identical dtypes"
    if k.device != q.device or v.device != q.device:
        return "q, k, and v must be on the same device"
    if q.shape[1] <= 0 or q.shape[2] <= 0 or q.shape[3] <= 0:
        return "sequence length, head count, and head dimension must be positive"
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        return "the head-dimension stride must be 1"
    if valid_token_mask is not None:
        if valid_token_mask.shape != q.shape[:2]:
            return "valid_token_mask must have shape [batch, sequence]"
        if valid_token_mask.dtype is not torch.bool:
            return "valid_token_mask must have dtype torch.bool"
        if valid_token_mask.device != q.device:
            return "valid_token_mask must be on the q/k/v device"
    return None


def triton_attention_support(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor] = None,
) -> SupportResult:
    """Explain whether the custom forward kernel can safely run."""
    contract_error = _validate_qkv_contract(q, k, v, valid_token_mask)
    if contract_error is not None:
        return SupportResult(False, contract_error)
    if q.device.type != "cuda":
        return SupportResult(False, "custom attention requires CUDA tensors")
    if q.dtype not in SUPPORTED_DTYPES:
        return SupportResult(False, "custom attention supports float16 and float32")
    if q.shape[-1] not in SUPPORTED_HEAD_DIMS:
        return SupportResult(
            False,
            f"head_dim={q.shape[-1]} is outside {sorted(SUPPORTED_HEAD_DIMS)}",
        )
    if q.shape[1] > MAX_SEQUENCE_LENGTH:
        return SupportResult(
            False,
            f"seq_len={q.shape[1]} exceeds the validated maximum {MAX_SEQUENCE_LENGTH}",
        )
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        return SupportResult(False, "custom attention is forward-inference only")
    capability = torch.cuda.get_device_capability(q.device)
    if capability < MIN_CUDA_CAPABILITY:
        return SupportResult(
            False,
            f"compute capability {capability} is below {MIN_CUDA_CAPABILITY}",
        )
    return SupportResult(True, "inside the validated Triton forward envelope")
