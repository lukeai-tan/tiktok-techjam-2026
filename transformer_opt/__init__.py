"""Optimization support for the TikTok TechJam 2026 Transformer benchmark.

The optimized model itself lives in ``torch_transformer_benchmark.py`` as
``UserOptimizedTransformer`` (the competition's designated integration point).
This package holds optional, reusable pieces:

    triton_impl  - optional GPU-only fused LayerNorm kernel (opt-in)
"""

from . import triton_impl

__all__ = ["triton_impl"]
