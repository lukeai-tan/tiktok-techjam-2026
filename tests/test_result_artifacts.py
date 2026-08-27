"""Keep curated GPU claims tied to the exact implementation and raw evidence."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import pytest

import tools.capture_environment as capture_environment
from tools.capture_environment import implementation_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def _text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


MATRIX_PATH = ROOT / "docs" / "results" / "rtx-5070-ti-2026-08-27.json"
PROFILE_PATH = ROOT / "docs" / "results" / "rtx-5070-ti-2026-08-27-profile.json"
ORGANIZER_DEFAULT_PATH = (
    ROOT / "docs" / "results" / "rtx-5070-ti-2026-08-27-organizer-default.json"
)
ORGANIZER_VALIDATION_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-27-organizer-validation.json"
)
FINAL_EVALUATOR_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-28-final-evaluator-baseline.json"
)
FINAL_EVALUATOR_MATRIX_PATH = ROOT / "benchmarks" / "final_evaluator_shapes.json"
ORGANIZER_VALIDATION_MATRIX_PATH = (
    ROOT / "benchmarks" / "organizer_validation_matrix.json"
)
ORGANIZER_MANIFEST_PATH = (
    ROOT / "benchmarks" / "reference" / "organizer_downloads.json"
)
SDPA_CASES = {"tiny-overhead", "medium-throughput"}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_implementation_fingerprint_ignores_checkout_line_endings(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "sample.py"
    monkeypatch.setattr(capture_environment, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(capture_environment, "IMPLEMENTATION_PATHS", ("sample.py",))

    source.write_bytes(b"first = 1\nsecond = 2\n")
    lf_fingerprint, lf_paths = implementation_fingerprint()
    source.write_bytes(b"first = 1\r\nsecond = 2\r\n")
    crlf_fingerprint, crlf_paths = implementation_fingerprint()

    assert lf_fingerprint == crlf_fingerprint
    assert lf_paths == crlf_paths == ["sample.py"]


def test_environment_paths_do_not_expose_host_home(monkeypatch):
    monkeypatch.setattr(
        capture_environment.sys,
        "executable",
        str(Path.home() / "venv" / "python"),
    )
    payload = capture_environment.capture_environment(["capture-environment"])

    assert payload["python"]["executable"] == "python"
    assert payload["disk"]["path"] == "."
    assert payload["git"]["implementation_fingerprint_schema"] == 2
    assert payload["cpu"]["name"]
    assert payload["cpu"]["logical_count"]


def test_curated_matrix_is_complete_green_and_current():
    matrix = _load(MATRIX_PATH)
    current_fingerprint, current_paths = implementation_fingerprint()
    captured_git = matrix["environment"]["git"]
    assert captured_git["implementation_sha256"] == current_fingerprint
    assert captured_git["implementation_fingerprint_schema"] == 2
    assert "implementation_fingerprint_migrated_from_sha256" not in captured_git
    assert captured_git["implementation_paths"] == current_paths
    assert matrix["manifest"]["path"] == "benchmarks/official_shapes.json"
    assert Path(matrix["environment"]["python"]["executable"]).stem == "python"
    assert matrix["environment"]["disk"]["path"] == "."
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
        selected = "sdpa" if result["case_id"] in SDPA_CASES else "triton"
        other = "triton" if selected == "sdpa" else "sdpa"
        assert timing["backend_counts"][selected] > 0
        assert timing["backend_counts"][other] == 0
        assert timing["backend_counts"]["reference"] == 0
        assert result["peak_memory"]["baseline"] is not None
        assert result["peak_memory"]["optimized"] is not None
        speedups.append(timing["speedup_median"])
    assert total_failed == 0
    assert statistics.geometric_mean(speedups) == pytest.approx(1.288, abs=0.001)


def test_curated_profile_proves_custom_kernel_for_same_implementation():
    matrix = _load(MATRIX_PATH)
    profile = _load(PROFILE_PATH)
    assert Path(profile["environment"]["python"]["executable"]).stem == "python"
    assert profile["environment"]["disk"]["path"] == "."
    assert profile["environment"]["git"]["implementation_sha256"] == (
        matrix["environment"]["git"]["implementation_sha256"]
    )
    assert profile["environment"]["git"]["implementation_fingerprint_schema"] == 2
    assert profile["custom_kernel_expected"] is True
    assert profile["custom_kernel_profiler_proven"] is True
    assert profile["backend_counts"] == {"triton": 10, "sdpa": 0, "reference": 0}
    matching = [
        event
        for event in profile["custom_kernel_events"]
        if event["name"] == "_attention_fwd"
    ]
    assert len(matching) == 1
    assert matching[0]["count"] == 10


def test_organizer_default_artifact_uses_untouched_harness_and_current_submission():
    evidence = _load(ORGANIZER_DEFAULT_PATH)
    manifest = _load(ORGANIZER_MANIFEST_PATH)
    current_fingerprint, _ = implementation_fingerprint()
    pytorch_download = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["framework"] == "pytorch"
    )

    assert evidence["exit_code"] == 0
    assert evidence["framework"] == "pytorch"
    assert evidence["organizer_script"] == {
        "path": "benchmarks/torch_transformer_benchmark.py",
        "sha256": pytorch_download["sha256"],
        "manifest": "benchmarks/reference/organizer_downloads.json",
        "modified": False,
    }
    assert evidence["organizer_arguments"] == ["--device", "cuda"]
    runner_path = ROOT / evidence["submission"]["runner_path"]
    assert evidence["submission"]["runner_sha256"] == _text_sha256(runner_path)
    assert evidence["parsed"]["accuracy"]["status"] == "PASS"
    assert evidence["parsed"]["accuracy"]["failed_elements"] == 0
    assert evidence["parsed"]["accuracy"]["total_elements"] == 2_621_440
    assert evidence["parsed"]["speedup_median"] > 1.0
    assert evidence["attention_backend_counts"] == {
        "triton": 1950,
        "sdpa": 0,
        "reference": 0,
    }
    assert evidence["environment"]["git"]["implementation_sha256"] == (
        current_fingerprint
    )


def test_organizer_validation_artifact_is_complete_green_and_fail_closed():
    evidence = _load(ORGANIZER_VALIDATION_PATH)
    current_fingerprint, _ = implementation_fingerprint()

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for nested in value.values():
                yield from strings(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from strings(nested)

    host_home = str(Path.home()).lower()
    assert all(host_home not in value.lower() for value in strings(evidence))
    assert evidence["status"] == "PASS"
    assert evidence["matrix"]["path"] == (
        "benchmarks/organizer_validation_matrix.json"
    )
    assert evidence["matrix"]["sha256"] == _text_sha256(
        ORGANIZER_VALIDATION_MATRIX_PATH
    )
    manifest_path = ROOT / evidence["organizer_sources"]["manifest_path"]
    assert evidence["organizer_sources"]["manifest_sha256"] == _text_sha256(
        manifest_path
    )
    for path_key, hash_key in (
        ("runner_path", "runner_sha256"),
        ("validation_runner_path", "validation_runner_sha256"),
    ):
        path = ROOT / evidence["organizer_sources"][path_key]
        assert evidence["organizer_sources"][hash_key] == _text_sha256(path)
    assert evidence["environment"]["git"]["implementation_sha256"] == (
        current_fingerprint
    )

    summary = evidence["summary"]
    assert summary["requested"] == 29
    assert summary["executable"] == 28
    assert summary["passed"] == 28
    assert summary["counts"] == {
        "ERROR": 0,
        "FAIL": 0,
        "OOM": 0,
        "PASS": 28,
        "SKIPPED_RESOURCE": 1,
    }
    assert summary["all_executable_passed"] is True
    assert summary["skipped_counted_as_pass"] is False
    assert summary["total_compared_elements"] == 459_776_000
    assert summary["total_failed_elements"] == 0
    assert all(count > 0 for count in summary["attention_backend_counts"].values())

    executable = [
        result for result in evidence["results"] if result["status"] != "SKIPPED_RESOURCE"
    ]
    skipped = [
        result for result in evidence["results"] if result["status"] == "SKIPPED_RESOURCE"
    ]
    assert len(executable) == 28
    assert all(result["status"] == "PASS" for result in executable)
    assert all(
        result["parsed"]["accuracy"]["failed_elements"] == 0
        for result in executable
    )
    assert all(
        sum(result["attention_backend_counts"].values()) == 168
        for result in executable
    )
    measured_speedups = [
        result["parsed"]["speedup_median"] for result in executable
    ]
    assert summary["geometric_mean_speedup"] == pytest.approx(
        statistics.geometric_mean(measured_speedups)
    )
    assert summary["geometric_mean_speedup"] > 1.1
    assert len(skipped) == 1
    assert skipped[0]["skip_authorized"] is True
    assert skipped[0]["skip_counted_as_pass"] is False
    assert skipped[0]["source_dimensions"] == [32, 1024, 16, 100000]


def test_final_evaluator_artifact_is_complete_green_and_current():
    evidence = _load(FINAL_EVALUATOR_PATH)
    current_fingerprint, _ = implementation_fingerprint()

    assert evidence["status"] == "PASS"
    assert evidence["matrix"] == {
        "path": "benchmarks/final_evaluator_shapes.json",
        "sha256": _text_sha256(FINAL_EVALUATOR_MATRIX_PATH),
        "status": "organizer-published-final-shapes",
    }
    assert evidence["environment"]["git"]["dirty"] is False
    assert evidence["environment"]["git"]["implementation_sha256"] == (
        current_fingerprint
    )
    for path_key, hash_key in (
        ("manifest_path", "manifest_sha256"),
        ("runner_path", "runner_sha256"),
        ("validation_runner_path", "validation_runner_sha256"),
    ):
        path = ROOT / evidence["organizer_sources"][path_key]
        assert evidence["organizer_sources"][hash_key] == _text_sha256(path)

    summary = evidence["summary"]
    assert summary["requested"] == 14
    assert summary["executable"] == summary["passed"] == 13
    assert summary["counts"] == {
        "ERROR": 0,
        "FAIL": 0,
        "OOM": 0,
        "PASS": 13,
        "SKIPPED_RESOURCE": 1,
    }
    assert summary["total_compared_elements"] == 938_885_120
    assert summary["total_failed_elements"] == 0
    assert summary["skipped_counted_as_pass"] is False
    assert summary["geometric_mean_speedup"] > 1.0

    executable = [
        result for result in evidence["results"] if result["status"] == "PASS"
    ]
    skipped = [
        result
        for result in evidence["results"]
        if result["status"] == "SKIPPED_RESOURCE"
    ]
    assert [result["case_id"].split("-")[1] for result in evidence["results"]] == [
        f"{index:02d}" for index in range(1, 15)
    ]
    assert len(executable) == 13
    assert all(
        result["parsed"]["accuracy"]["failed_elements"] == 0
        and sum(result["attention_backend_counts"].values()) == 112
        for result in executable
    )
    assert len(skipped) == 1
    assert skipped[0]["source_dimensions"] == [
        32,
        1024,
        16,
        100000,
        2,
        True,
        1024,
    ]
    assert skipped[0]["skip_authorized"] is True
    assert skipped[0]["skip_counted_as_pass"] is False
