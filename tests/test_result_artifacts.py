"""Keep curated GPU claims tied to the exact implementation and raw evidence."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from tools.capture_environment import implementation_fingerprint


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "docs" / "results" / "rtx-5070-ti-2026-08-27.json"
PROFILE_PATH = ROOT / "docs" / "results" / "rtx-5070-ti-2026-08-27-profile.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_curated_matrix_is_complete_green_and_current():
    matrix = _load(MATRIX_PATH)
    current_fingerprint, current_paths = implementation_fingerprint()
    captured_git = matrix["environment"]["git"]
    assert captured_git["implementation_sha256"] == current_fingerprint
    assert captured_git["implementation_paths"] == current_paths
    assert matrix["manifest"]["status"] == "provisional"
    assert matrix["summary"] == {
        "requested": 7,
        "completed": 7,
        "counts": {"ERROR": 0, "FAIL": 0, "OOM": 0, "PASS": 7},
    }

    speedups = []
    total_failed = 0
    for result in matrix["results"]:
        assert result["status"] == "PASS"
        assert len(result["accuracy"]["trials"]) == 5
        total_failed += sum(
            trial["failed_elements"] for trial in result["accuracy"]["trials"]
        )
        timing = result["timing"]
        assert len(timing["baseline"]["raw_ms"]) == 90
        assert len(timing["optimized"]["raw_ms"]) == 90
        assert timing["backend_counts"]["triton"] > 0
        assert timing["backend_counts"]["sdpa"] == 0
        assert timing["backend_counts"]["reference"] == 0
        assert result["peak_memory"]["baseline"] is not None
        assert result["peak_memory"]["optimized"] is not None
        speedups.append(timing["speedup_median"])
    assert total_failed == 0
    assert statistics.geometric_mean(speedups) == pytest.approx(1.360, abs=0.001)


def test_curated_profile_proves_custom_kernel_for_same_implementation():
    matrix = _load(MATRIX_PATH)
    profile = _load(PROFILE_PATH)
    assert profile["environment"]["git"]["implementation_sha256"] == (
        matrix["environment"]["git"]["implementation_sha256"]
    )
    assert profile["custom_kernel_expected"] is True
    assert profile["custom_kernel_profiler_proven"] is True
    assert profile["backend_counts"] == {"triton": 10, "sdpa": 0, "reference": 0}
    matching = [
        event for event in profile["custom_kernel_events"]
        if event["name"] == "_attention_fwd"
    ]
    assert len(matching) == 1
    assert matching[0]["count"] == 10
