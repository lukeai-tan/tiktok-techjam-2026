"""CPU-only unit tests for the autotune search space and backend routing.

These exercise pure decision logic, so they run without a GPU. The model's
``_attention`` ladder is otherwise unreachable on CPU because it sits behind
``x.is_cuda``; ``select_attention_backend`` lifts that decision out so it can be
tested directly.
"""

from __future__ import annotations

import torch

from torch_transformer_benchmark import select_attention_backend
from transformer_opt.config import attention_autotune_configs


def test_autotune_space_only_tunes_scheduling_knobs():
    configs = attention_autotune_configs()
    # Every entry is a (num_warps, num_stages) pair; no tile sizes are searched,
    # which is what keeps the numerical accumulation order fixed.
    assert all(len(entry) == 2 for entry in configs)
    assert all(num_warps in (4, 8) for num_warps, _ in configs)
    assert all(num_stages in (2, 3, 4) for _, num_stages in configs)


def test_autotune_space_contains_historical_configs():
    # The previous fixed launch used num_warps=4 with num_stages 2 or 3. Keeping
    # them in the space guarantees autotune can never do worse than before.
    configs = attention_autotune_configs()
    assert (4, 2) in configs
    assert (4, 3) in configs


def _select(**overrides) -> str:
    base = dict(
        requested="auto",
        dtype=torch.float32,
        head_dim=64,
        d_model=512,
        num_heads=8,
        batch=64,
        seq_len=128,
        num_layers=4,
        causal=True,
        is_cuda=True,
    )
    base.update(overrides)
    return select_attention_backend(**base)


def test_forced_backend_is_passed_through_untouched():
    assert _select(requested="triton") == "triton"
    assert _select(requested="sdpa") == "sdpa"
    assert _select(requested="reference") == "reference"


def test_cpu_tensor_never_enters_the_ladder():
    # On CPU the model keeps the request as-is (the ladder is CUDA-only).
    assert _select(is_cuda=False) == "auto"


def test_low_precision_stays_correctness_first():
    assert _select(dtype=torch.float16) == "reference"


def test_low_occupancy_long_causal_routes_to_sdpa():
    # The disclosed regression shape: batch=2, seq=512, head_dim=64, 2 layers.
    assert (
        _select(batch=2, seq_len=512, d_model=512, num_heads=8, num_layers=2)
        == "sdpa"
    )


def test_low_occupancy_rule_ignores_non_causal_long_shapes():
    # The non-causal long-attention win must stay on the custom/auto path.
    assert (
        _select(batch=2, seq_len=512, num_layers=2, causal=False) == "auto"
    )


def test_final_row13_long_high_batch_stays_on_custom_path():
    # final-13: batch=64, seq=1024, head_dim=32. batch>8 so the low-occupancy
    # rule must not catch it; it keeps its measured Triton win.
    assert (
        _select(head_dim=32, d_model=128, num_heads=4, batch=64, seq_len=1024)
        == "auto"
    )


def test_official_short_causal_rows_stay_on_custom_path():
    # seq=128 < 256, so the low-occupancy rule never fires for the official
    # short rows regardless of batch size.
    assert _select(head_dim=32, d_model=128, num_heads=4, batch=1) == "auto"
    assert _select(head_dim=32, d_model=128, num_heads=4, batch=128) == "auto"


def test_very_large_causal_batch_uses_reference():
    assert (
        _select(head_dim=32, d_model=128, num_heads=4, batch=10000) == "reference"
    )


def test_head_dim_eight_measured_row_enables_auto():
    assert (
        _select(head_dim=8, d_model=128, num_heads=16, batch=64, seq_len=128)
        == "auto"
    )


def test_head_dim_eight_other_rows_use_reference():
    assert (
        _select(head_dim=8, d_model=32, num_heads=4, batch=64, seq_len=128)
        == "reference"
    )


def test_deep_stack_causal_uses_sdpa():
    assert _select(num_layers=6, batch=8) == "sdpa"
