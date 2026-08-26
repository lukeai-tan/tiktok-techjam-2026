"""End-to-end target-device checks through UserOptimizedTransformer."""

from __future__ import annotations

import pytest
import torch

from torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    compare_outputs,
    copy_model_weights,
    generate_random_case,
    maybe_compile,
    run_accuracy_tests,
)
from transformer_opt.kernels import triton_available


pytestmark = pytest.mark.skipif(
    not triton_available(),
    reason="CUDA Triton is unavailable",
)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("causal,padding_ratio", [(False, 0.0), (True, 0.3)])
def test_end_to_end_transformer_uses_custom_attention(dtype, causal, padding_ratio):
    config = TransformerConfig(2, 65, 128, 4, 512, 2, causal)
    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config, attention_backend="triton")
    copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.to(device="cuda", dtype=dtype).eval()
    optimized = optimized.to(device="cuda", dtype=dtype).eval()

    x, valid_mask = generate_random_case(
        config,
        torch.device("cuda"),
        dtype,
        seed=71,
        padding_ratio=padding_ratio,
        input_scale=1.0,
    )
    with torch.inference_mode():
        reference = baseline(x, valid_mask)
        actual = optimized(x, valid_mask)
    result = compare_outputs(reference, actual, rtol=0.01, atol=0.001)
    assert result.passed, (
        f"failed={result.failed_elements}/{result.total_elements} "
        f"max_abs={result.max_abs_error:.6g}"
    )
    assert optimized.attention_backend_counts == {
        "triton": config.num_layers,
        "sdpa": 0,
        "reference": 0,
    }


def test_explicit_all_valid_mask_matches_no_mask():
    config = TransformerConfig(1, 64, 128, 4, 256, 1, False)
    model = UserOptimizedTransformer(config, attention_backend="triton").cuda().half().eval()
    x = torch.randn(1, 64, 128, device="cuda", dtype=torch.float16)
    all_valid = torch.ones(1, 64, device="cuda", dtype=torch.bool)
    with torch.inference_mode():
        no_mask = model(x, None)
        explicit_mask = model(x, all_valid)
    result = compare_outputs(no_mask, explicit_mask, rtol=0.01, atol=0.001)
    assert result.passed


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal,padding_ratio", [(False, 0.0), (True, 0.3)])
def test_low_precision_auto_uses_correctness_first_reference(
    dtype,
    causal,
    padding_ratio,
):
    config = TransformerConfig(2, 65, 128, 4, 512, 2, causal)
    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config, attention_backend="auto")
    copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.to(device="cuda", dtype=dtype).eval()
    optimized = optimized.to(device="cuda", dtype=dtype).eval()
    x, valid_mask = generate_random_case(
        config,
        torch.device("cuda"),
        dtype,
        seed=71,
        padding_ratio=padding_ratio,
        input_scale=1.0,
    )
    with torch.inference_mode():
        reference = baseline(x, valid_mask)
        actual = optimized(x, valid_mask)
    assert compare_outputs(reference, actual, rtol=0.01, atol=0.001).passed
    assert optimized.attention_backend_counts == {
        "triton": 0,
        "sdpa": 0,
        "reference": config.num_layers,
    }


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_reduce_overhead_accuracy_preserves_cudagraph_outputs():
    """Regression: a second compiled call must not invalidate the reference."""
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        config = TransformerConfig(2, 128, 256, 8, 1024, 2, False)
        baseline = BaselineTransformer(config)
        optimized = UserOptimizedTransformer(config, attention_backend="auto")
        copy_model_weights(baseline, optimized, strict=True)
        baseline = baseline.cuda().eval()
        optimized = optimized.cuda().eval()
        compiled_baseline = maybe_compile(baseline, True, "reduce-overhead")
        compiled_optimized = maybe_compile(optimized, True, "reduce-overhead")

        assert run_accuracy_tests(
            baseline=compiled_baseline,
            optimized=compiled_optimized,
            config=config,
            device=torch.device("cuda"),
            dtype=torch.float32,
            trials=2,
            seed=1234,
            padding_ratio=0.0,
            input_scale=1.0,
            rtol=0.01,
            atol=0.001,
        )
    finally:
        torch.set_float32_matmul_precision(previous_precision)
