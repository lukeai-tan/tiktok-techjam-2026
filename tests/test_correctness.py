"""Correctness: UserOptimizedTransformer must match BaselineTransformer.

Uses the official benchmark script as the single source of truth -- same
Baseline, same weight-copy, same OR-tolerance comparison the judge applies.
Runs on CPU (fp32) so it is usable during development without a GPU.
"""

import itertools

import pytest
import torch

from torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    compare_outputs,
    copy_model_weights,
    generate_random_case,
)

RTOL = 0.01   # official defaults
ATOL = 0.001

# (batch, seq_len, d_model, heads, ffn, layers)
_SHAPES = [
    (1, 32, 64, 4, 256, 2),      # tiny / overhead-bound
    (8, 128, 256, 8, 1024, 2),   # medium
    (2, 512, 512, 8, 2048, 2),   # long sequence
    (4, 64, 1024, 16, 4096, 1),  # wide model
    (2, 96, 128, 8, 512, 3),     # multi-layer stack
]
_CAUSAL = [False, True]
_PADDING = [0.0, 0.3]


def _make(cfg):
    baseline = BaselineTransformer(cfg)
    optimized = UserOptimizedTransformer(cfg)
    copy_model_weights(baseline, optimized, strict=True)  # must succeed strict
    return baseline.eval(), optimized.eval()


@pytest.mark.parametrize(
    "shape,causal,padding",
    list(itertools.product(_SHAPES, _CAUSAL, _PADDING)),
    ids=lambda v: str(v),
)
def test_optimized_matches_baseline(shape, causal, padding):
    b, s, d, h, f, L = shape
    cfg = TransformerConfig(b, s, d, h, f, L, causal)
    cfg.validate()
    baseline, optimized = _make(cfg)

    device = torch.device("cpu")
    x, mask = generate_random_case(
        cfg, device, torch.float32, seed=7, padding_ratio=padding, input_scale=1.0
    )
    with torch.inference_mode():
        ref = baseline(x, mask)
        opt = optimized(x, mask)
    res = compare_outputs(ref, opt, rtol=RTOL, atol=ATOL)
    assert res.passed, (
        f"failed={res.failed_elements}/{res.total_elements} "
        f"max_abs={res.max_abs_error:.3e} at {res.worst_index} "
        f"(ref={res.reference_at_worst:.4g} opt={res.optimized_at_worst:.4g})"
    )


def test_strict_weight_copy_is_compatible():
    # The whole design hinges on identical parameter names; guard it explicitly.
    cfg = TransformerConfig(2, 16, 64, 8, 128, 2, False)
    baseline = BaselineTransformer(cfg)
    optimized = UserOptimizedTransformer(cfg)
    # Raises if any key is missing/unexpected under strict loading.
    copy_model_weights(baseline, optimized, strict=True)


def test_no_padding_causal_matches():
    cfg = TransformerConfig(2, 128, 128, 8, 512, 2, True)
    baseline, optimized = _make(cfg)
    x = torch.randn(2, 128, 128)
    with torch.inference_mode():
        ref = baseline(x, None)
        opt = optimized(x, None)
    assert compare_outputs(ref, opt, rtol=RTOL, atol=ATOL).passed
