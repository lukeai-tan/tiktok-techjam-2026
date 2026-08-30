from __future__ import annotations

import pytest

from benchmarks.profile_cases import (
    _events_named,
    _resolve_expected_backend,
    evaluate_profile_expectations,
)


def _evaluate(
    *,
    counts: dict[str, int],
    attention_events: list[dict] | None = None,
    fused_events: list[dict] | None = None,
    expected_backend: str | None = None,
    expect_fused: bool = False,
):
    return evaluate_profile_expectations(
        backend_counts=counts,
        attention_kernel_events=attention_events or [],
        fused_residual_layer_norm_events=fused_events or [],
        expected_backend=expected_backend,
        expect_fused_residual_layer_norm=expect_fused,
    )


def test_auto_profile_without_expectations_is_observational():
    validation_passed, checks = _evaluate(
        counts={"triton": 0, "sdpa": 0, "reference": 4}
    )

    assert validation_passed is None
    assert checks == []


def test_profiler_event_selection_rejects_substring_decoys():
    events = [
        {"name": "prefix_attention_fwd_suffix", "count": 99},
        {"name": "_attention_fwd", "count": 4},
    ]

    assert _events_named(events, "_attention_fwd") == [events[1]]


@pytest.mark.parametrize(
    ("counts", "events", "expected_pass"),
    [
        ({"triton": 0}, [{"name": "_attention_fwd", "count": 4}], False),
        ({"triton": 4}, [], False),
        ({"triton": 4}, [{"name": "_attention_fwd", "count": 0}], False),
        ({"triton": 4}, [{"name": "_attention_fwd", "count": 4}], True),
    ],
)
def test_triton_expectation_requires_dispatch_and_profiler_event(
    counts, events, expected_pass
):
    passed, checks = _evaluate(
        counts=counts,
        attention_events=events,
        expected_backend="triton",
    )

    assert passed is expected_pass
    assert checks[0]["passed"] is expected_pass
    assert checks[0]["profiler_event_found"] is (
        sum(event["count"] for event in events) > 0
    )


@pytest.mark.parametrize("backend", ["sdpa", "reference"])
def test_non_triton_backend_expectation_requires_positive_dispatch(backend):
    failed, _ = _evaluate(counts={backend: 0}, expected_backend=backend)
    passed, _ = _evaluate(counts={backend: 1}, expected_backend=backend)

    assert not failed
    assert passed


def test_fused_residual_layer_norm_expectation_requires_profiler_event():
    failed, _ = _evaluate(counts={}, expect_fused=True)
    passed, checks = _evaluate(
        counts={},
        fused_events=[{"name": "_residual_layer_norm_fwd", "count": 8}],
        expect_fused=True,
    )

    assert not failed
    assert passed
    assert checks[0]["profiler_event_count"] == 8


def test_forced_backend_becomes_expectation_but_auto_does_not():
    assert _resolve_expected_backend("auto", None) is None
    assert _resolve_expected_backend("triton", None) == "triton"
    assert _resolve_expected_backend("auto", "reference") == "reference"
