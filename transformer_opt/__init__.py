"""Custom GPU kernels and dispatch for the Transformer benchmark."""

from .config import ATTENTION_BACKENDS, triton_attention_support
from .dispatch import AttentionDispatch, attention_forward

__all__ = [
    "ATTENTION_BACKENDS",
    "AttentionDispatch",
    "attention_forward",
    "triton_attention_support",
]
