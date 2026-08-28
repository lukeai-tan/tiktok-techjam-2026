"""Portable dispatch contract tests plus CUDA-specific routing checks."""

from __future__ import annotations

import pytest
import torch

from transformer_opt import attention_forward, triton_attention_support
from transformer_opt.config import attention_launch_config
from transformer_opt.kernels import triton_available


def _cpu_qkv(dtype: torch.dtype = torch.float32):
    q = torch.randn(1, 8, 2, 16, dtype=dtype)
    return q, torch.randn_like(q), torch.randn_like(q)


def test_cpu_auto_falls_back_to_sdpa():
    q, k, v = _cpu_qkv()
    output, decision = attention_forward(q, k, v, backend="auto")
    assert output.shape == q.shape
    assert decision.selected == "sdpa"
    assert "CUDA" in decision.reason


def test_forced_triton_fails_clearly_on_cpu():
    q, k, v = _cpu_qkv()
    with pytest.raises((RuntimeError, ValueError), match="Triton attention was forced"):
        attention_forward(q, k, v, backend="triton")


def test_invalid_backend_is_rejected():
    q, k, v = _cpu_qkv()
    with pytest.raises(ValueError, match="attention backend"):
        attention_forward(q, k, v, backend="mystery")


def test_support_check_explains_bad_mask_shape():
    q, k, v = _cpu_qkv()
    result = triton_attention_support(
        q,
        k,
        v,
        torch.ones(1, 7, dtype=torch.bool),
    )
    assert not result.supported
    assert "shape [batch, sequence]" in result.reason


@pytest.mark.skipif(not triton_available(), reason="CUDA Triton is unavailable")
def test_cuda_auto_selects_custom_kernel():
    q = torch.randn(1, 64, 2, 32, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        output, decision = attention_forward(q, q, q, backend="auto")
    assert output.shape == q.shape
    assert decision.selected == "triton"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_short_unmasked_float32_auto_selects_measured_sdpa_path():
    q = torch.randn(1, 16, 2, 32, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        output, decision = attention_forward(q, q, q, backend="auto")
    assert output.shape == q.shape
    assert decision.selected == "sdpa"
    assert "measured short unmasked" in decision.reason


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_short_wide_head_float32_auto_selects_custom_kernel():
    q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        output, decision = attention_forward(q, q, q, backend="auto")
    assert output.shape == q.shape
    assert decision.selected == "triton"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "seq_len,causal,with_mask",
    [
        (129, False, False),
        (16, True, False),
        (16, False, True),
    ],
)
def test_float32_auto_keeps_triton_outside_short_unmasked_corner(
    seq_len,
    causal,
    with_mask,
):
    q = torch.randn(1, seq_len, 2, 32, device="cuda", dtype=torch.float32)
    valid_mask = (
        torch.ones(1, seq_len, device="cuda", dtype=torch.bool)
        if with_mask
        else None
    )
    with torch.inference_mode():
        output, decision = attention_forward(
            q,
            q,
            q,
            valid_mask,
            causal=causal,
            backend="auto",
        )
    assert output.shape == q.shape
    assert decision.selected == "triton"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_requires_grad_auto_falls_back():
    q = torch.randn(
        1,
        16,
        2,
        32,
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    output, decision = attention_forward(q, q, q, backend="auto")
    assert output.requires_grad
    assert decision.selected == "sdpa"
    assert "forward-inference only" in decision.reason


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_bfloat16_auto_uses_reference_math():
    q = torch.randn(1, 16, 2, 32, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        output, decision = attention_forward(q, q, q, backend="auto")
    assert output.shape == q.shape
    assert decision.selected == "reference"
    assert "executable tolerance" in decision.reason


def test_head_dim_64_short_sequence_uses_spill_avoiding_tiles():
    launch = attention_launch_config(head_dim=64, seq_len=128)
    assert (launch.block_m, launch.block_n, launch.num_warps, launch.num_stages) == (
        32,
        64,
        4,
        2,
    )


def test_spill_avoiding_tiles_do_not_expand_to_neighboring_shapes():
    neighboring_width = attention_launch_config(head_dim=48, seq_len=128)
    longer = attention_launch_config(head_dim=64, seq_len=129)

    assert (neighboring_width.block_m, neighboring_width.block_n) == (64, 128)
    assert (longer.block_m, longer.block_n) == (64, 64)


def test_short_head_dim_32_reduces_kv_tile_only_through_sequence_128():
    short_narrow = attention_launch_config(head_dim=32, seq_len=128)
    longer = attention_launch_config(head_dim=32, seq_len=129)
    smaller_head = attention_launch_config(head_dim=16, seq_len=128)

    assert (
        short_narrow.block_m,
        short_narrow.block_n,
        short_narrow.num_warps,
        short_narrow.num_stages,
    ) == (64, 64, 4, 2)
    assert (longer.block_m, longer.block_n) == (64, 64)
    assert (smaller_head.block_m, smaller_head.block_n) == (64, 128)


def test_short_head_dim_128_reduces_kv_tile_only_through_sequence_128():
    short = attention_launch_config(head_dim=128, seq_len=128)
    longer = attention_launch_config(head_dim=128, seq_len=129)
    neighboring_head = attention_launch_config(head_dim=127, seq_len=128)

    assert (short.block_m, short.block_n, short.num_warps, short.num_stages) == (
        32,
        32,
        4,
        2,
    )
    assert (longer.block_m, longer.block_n) == (32, 64)
    assert (neighboring_head.block_m, neighboring_head.block_n) == (32, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_forced_triton_rejects_bfloat16():
    q = torch.randn(1, 16, 2, 32, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), pytest.raises(
        ValueError,
        match="supports float16 and float32",
    ):
        attention_forward(q, q, q, backend="triton")
