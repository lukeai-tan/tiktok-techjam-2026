"""Direct correctness tests for residual-add plus LayerNorm fusion."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from transformer_opt.kernels import fused_residual_layer_norm, triton_available


pytestmark = pytest.mark.skipif(
    not triton_available(),
    reason="CUDA Triton is unavailable",
)


@pytest.mark.parametrize(
    "masked,zero_residual,zero_normalized",
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ],
)
def test_fused_residual_layer_norm_matches_explicit_operations(
    masked,
    zero_residual,
    zero_normalized,
):
    torch.manual_seed(7018)
    x = torch.randn(2, 17, 128, device="cuda", dtype=torch.float32)
    update = torch.randn_like(x)
    weight = torch.randn(128, device="cuda", dtype=torch.float32)
    bias = torch.randn(128, device="cuda", dtype=torch.float32)
    valid_mask = None
    if masked:
        valid_mask = torch.arange(17, device="cuda")[None, :] < torch.tensor(
            [[17], [9]],
            device="cuda",
        )

    expected_residual = x + update
    if zero_residual:
        expected_residual = expected_residual.masked_fill(
            ~valid_mask[..., None],
            0,
        )
    expected_normalized = F.layer_norm(
        expected_residual,
        (128,),
        weight,
        bias,
        1e-5,
    )
    if zero_normalized:
        expected_normalized = expected_normalized.masked_fill(
            ~valid_mask[..., None],
            0,
        )

    with torch.inference_mode():
        actual_residual, actual_normalized = fused_residual_layer_norm(
            x,
            update,
            weight,
            bias,
            1e-5,
            valid_mask,
            zero_invalid_residual=zero_residual,
            zero_invalid_normalized=zero_normalized,
        )

    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_normalized,
        expected_normalized,
        rtol=1e-5,
        atol=1e-5,
    )


def test_fused_residual_layer_norm_supports_layer_norm_without_bias():
    torch.manual_seed(7019)
    x = torch.randn(3, 5, 128, device="cuda", dtype=torch.float32)
    update = torch.randn_like(x)
    weight = torch.randn(128, device="cuda", dtype=torch.float32)
    expected_residual = x + update
    expected_normalized = F.layer_norm(
        expected_residual,
        (128,),
        weight,
        None,
        1e-5,
    )

    with torch.inference_mode():
        actual_residual, actual_normalized = fused_residual_layer_norm(
            x,
            update,
            weight,
            None,
            1e-5,
        )

    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_normalized,
        expected_normalized,
        rtol=1e-5,
        atol=1e-5,
    )
