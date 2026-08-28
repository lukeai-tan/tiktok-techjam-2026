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


def test_packed_qkv_cache_is_reused_and_invalidated():
    config = TransformerConfig(1, 32, 64, 4, 256, 1, False)
    baseline = BaselineTransformer(config).cuda().eval()
    optimized = UserOptimizedTransformer(
        config,
        attention_backend="triton",
    ).cuda().eval()
    copy_model_weights(baseline, optimized, strict=True)
    x = torch.randn(1, 32, 64, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        first = optimized(x)
        cached = optimized._packed_qkv_cache[id(optimized.layers[0].attention)]
        first_packed_weight = cached[1]
        second = optimized(x)

    assert optimized._packed_qkv_cache[id(optimized.layers[0].attention)][1] is (
        first_packed_weight
    )
    assert torch.equal(first, second)
    assert baseline.state_dict().keys() == optimized.state_dict().keys()

    # Any in-place parameter update increments its version and must rebuild the
    # derived packed tensors before the next inference.
    with torch.no_grad():
        baseline.layers[0].attention.q_proj.weight.add_(0.01)
        optimized.layers[0].attention.q_proj.weight.add_(0.01)
    with torch.inference_mode():
        reference = baseline(x)
        actual = optimized(x)
    refreshed = optimized._packed_qkv_cache[id(optimized.layers[0].attention)][1]
    assert refreshed is not first_packed_weight
    assert compare_outputs(reference, actual, rtol=0.01, atol=0.001).passed


def test_wide_model_uses_campaign6_packed_qkv_candidate():
    config = TransformerConfig(2, 16, 1024, 4, 1024, 1, True)
    baseline = BaselineTransformer(config).cuda().eval()
    optimized = UserOptimizedTransformer(config, attention_backend="auto").cuda().eval()
    copy_model_weights(baseline, optimized, strict=True)
    x = torch.randn(2, 16, 1024, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        reference = baseline(x)
        actual = optimized(x)

    cache = optimized._packed_qkv_cache[id(optimized.layers[0].attention)]
    assert cache[1].shape == (3 * config.d_model, config.d_model)
    assert compare_outputs(reference, actual, rtol=0.01, atol=0.001).passed
    assert baseline.state_dict().keys() == optimized.state_dict().keys()


def test_intermediate_width_keeps_three_projection_boundary():
    config = TransformerConfig(2, 16, 768, 4, 768, 1, True)
    optimized = UserOptimizedTransformer(config, attention_backend="auto").cuda().eval()
    x = torch.randn(2, 16, 768, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        output = optimized(x)

    assert output.shape == x.shape
    assert id(optimized.layers[0].attention) not in optimized._packed_qkv_cache


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


@pytest.mark.parametrize(
    "batch_size,causal,expected_backend",
    [
        (8, False, "triton"),
        (9, False, "sdpa"),
        (1, True, "sdpa"),
        (129, True, "reference"),
    ],
)
def test_deep_stack_auto_uses_accuracy_guard(
    batch_size,
    causal,
    expected_backend,
):
    config = TransformerConfig(batch_size, 32, 128, 4, 512, 6, causal)
    model = UserOptimizedTransformer(config, attention_backend="auto").cuda().eval()
    x = torch.randn(batch_size, 32, 128, device="cuda", dtype=torch.float32)
    # The untouched organizer supplies an all-valid mask even at padding_ratio=0.
    valid_mask = torch.ones(batch_size, 32, device="cuda", dtype=torch.bool)

    with torch.inference_mode():
        output = model(x, valid_mask)

    assert output.shape == x.shape
    assert model.attention_backend_counts[expected_backend] == config.num_layers
    assert sum(model.attention_backend_counts.values()) == config.num_layers


def test_unsupported_head_dimension_auto_uses_reference_math():
    config = TransformerConfig(2, 32, 128, 16, 128, 2, True)
    model = UserOptimizedTransformer(config, attention_backend="auto").cuda().eval()
    x = torch.randn(2, 32, 128, device="cuda", dtype=torch.float32)
    valid_mask = torch.ones(2, 32, device="cuda", dtype=torch.bool)

    with torch.inference_mode():
        output = model(x, valid_mask)

    assert output.shape == x.shape
    assert model.attention_backend_counts == {
        "triton": 0,
        "sdpa": 0,
        "reference": config.num_layers,
    }


def test_final_row11_head_dim_8_auto_uses_padded_triton_route():
    config = TransformerConfig(64, 128, 128, 16, 128, 4, True)
    model = UserOptimizedTransformer(config, attention_backend="auto").cuda().eval()
    x = torch.randn(64, 128, 128, device="cuda", dtype=torch.float32)
    valid_mask = torch.ones(64, 128, device="cuda", dtype=torch.bool)

    with torch.inference_mode():
        output = model(x, valid_mask)

    assert output.shape == x.shape
    assert model.attention_backend_counts == {
        "triton": config.num_layers,
        "sdpa": 0,
        "reference": 0,
    }


def test_final_row7_head_dim_8_uses_first_layer_reference_hybrid_route():
    config = TransformerConfig(64, 128, 32, 4, 32, 4, True)
    model = UserOptimizedTransformer(config, attention_backend="auto").cuda().eval()
    x = torch.randn(64, 128, 32, device="cuda", dtype=torch.float32)
    valid_mask = torch.ones(64, 128, device="cuda", dtype=torch.bool)

    with torch.inference_mode():
        output = model(x, valid_mask)

    assert output.shape == x.shape
    assert model.attention_backend_counts == {
        "triton": 3,
        "sdpa": 0,
        "reference": 1,
    }


def test_final_row7_hybrid_route_passes_seed_scale_and_padding_stress():
    torch.manual_seed(7070)
    config = TransformerConfig(64, 128, 32, 4, 32, 4, True)
    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config, attention_backend="auto")
    copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.cuda().eval()
    optimized = optimized.cuda().eval()

    scenarios = 0
    with torch.inference_mode():
        for seed in (1234, 2026, 4096):
            for input_scale in (0.25, 1.0, 4.0):
                for padding_ratio in (0.0, 0.25):
                    x, valid_mask = generate_random_case(
                        config,
                        torch.device("cuda"),
                        torch.float32,
                        seed=seed,
                        padding_ratio=padding_ratio,
                        input_scale=input_scale,
                    )
                    reference = baseline(x, valid_mask)
                    actual = optimized(x, valid_mask)
                    result = compare_outputs(reference, actual, rtol=0.01, atol=0.001)
                    assert result.passed, (
                        f"seed={seed} scale={input_scale} padding={padding_ratio} "
                        f"failed={result.failed_elements}/{result.total_elements} "
                        f"max_abs={result.max_abs_error:.6g}"
                    )
                    scenarios += 1

    assert optimized.attention_backend_counts == {
        "triton": scenarios * 3,
        "sdpa": 0,
        "reference": scenarios,
    }


@pytest.mark.parametrize("padding_ratio", [0.0, 0.3])
def test_heldout_long_causal_auto_uses_accuracy_proven_sdpa(padding_ratio):
    config = TransformerConfig(2, 512, 512, 8, 2048, 2, True)
    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config, attention_backend="auto")
    copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.cuda().eval()
    optimized = optimized.cuda().eval()

    with torch.inference_mode():
        for seed in (1234, 2026, 4096):
            x, valid_mask = generate_random_case(
                config,
                torch.device("cuda"),
                torch.float32,
                seed=seed,
                padding_ratio=padding_ratio,
                input_scale=1.0,
            )
            reference = baseline(x, valid_mask)
            actual = optimized(x, valid_mask)
            result = compare_outputs(reference, actual, rtol=0.01, atol=0.001)
            assert result.passed, (
                f"seed={seed} padding={padding_ratio} "
                f"failed={result.failed_elements}/{result.total_elements} "
                f"max_abs={result.max_abs_error:.6g}"
            )

    assert optimized.attention_backend_counts == {
        "triton": 0,
        "sdpa": 3 * config.num_layers,
        "reference": 0,
    }


def test_final_row11_padded_triton_route_passes_seed_scale_and_padding_stress():
    torch.manual_seed(8080)
    config = TransformerConfig(64, 128, 128, 16, 128, 4, True)
    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config, attention_backend="auto")
    copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.cuda().eval()
    optimized = optimized.cuda().eval()

    with torch.inference_mode():
        for seed in (1234, 2026, 4096):
            for input_scale in (0.25, 1.0, 4.0):
                for padding_ratio in (0.0, 0.25):
                    x, valid_mask = generate_random_case(
                        config,
                        torch.device("cuda"),
                        torch.float32,
                        seed=seed,
                        padding_ratio=padding_ratio,
                        input_scale=input_scale,
                    )
                    if valid_mask is None:
                        valid_mask = torch.ones(
                            config.batch_size,
                            config.seq_len,
                            device="cuda",
                            dtype=torch.bool,
                        )
                    reference = baseline(x, valid_mask)
                    actual = optimized(x, valid_mask)
                    result = compare_outputs(reference, actual, rtol=0.01, atol=0.001)
                    assert result.passed, (
                        f"seed={seed} scale={input_scale} padding={padding_ratio} "
                        f"failed={result.failed_elements}/{result.total_elements} "
                        f"max_abs={result.max_abs_error:.6g}"
                    )

    assert optimized.attention_backend_counts == {
        "triton": 3 * 3 * 2 * config.num_layers,
        "sdpa": 0,
        "reference": 0,
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
