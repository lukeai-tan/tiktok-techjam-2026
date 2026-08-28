"""Repository-owned Triton kernels."""

from .attention import triton_attention, triton_available

__all__ = ["triton_attention", "triton_available"]
