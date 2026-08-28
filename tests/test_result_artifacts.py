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


MATRIX_PATH = (
    ROOT / "docs" / "results" / "rtx-5070-ti-2026-08-28-submission-heldout.json"
)
MATRIX_CONFIRMATION_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-28-submission-heldout-confirmation.json"
)
PROFILE_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-28-submission-final-11-profile.json"
)
ORGANIZER_DEFAULT_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-28-submission-organizer-default.json"
)
ORGANIZER_VALIDATION_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-28-submission-source-derived.json"
)
FINAL_EVALUATOR_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-28-submission-final.json"
)
FINAL_CONFIRMATION_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-28-submission-final-confirmation.json"
)
FINAL_EVALUATOR_MATRIX_PATH = ROOT / "benchmarks" / "final_evaluator_shapes.json"
ORGANIZER_VALIDATION_MATRIX_PATH = (
    ROOT / "benchmarks" / "organizer_validation_matrix.json"
)
ORGANIZER_MANIFEST_PATH = (
    ROOT / "benchmarks" / "reference" / "organizer_downloads.json"
)
SDPA_CASES = {"tiny-overhead", "medium-throughput"}
SUBMISSION_ATTEMPT_RESULT_MAP = {
    "S1-SUITE-003-organizer-default.json": ORGANIZER_DEFAULT_PATH,
    "S1-SUITE-004-final-primary.json": FINAL_EVALUATOR_PATH,
    "S1-SUITE-005-final-confirmation.json": FINAL_CONFIRMATION_PATH,
    "S1-SUITE-006-heldout.json": MATRIX_PATH,
    "S1-SUITE-007-source-derived.json": ORGANIZER_VALIDATION_PATH,
    "S1-SUITE-008-profile-row11.json": PROFILE_PATH,
    "S1-SUITE-009-heldout-confirmation.json": MATRIX_CONFIRMATION_PATH,
}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_selected_submission_attempts_bind_results_to_current_fingerprint():
    current_fingerprint, _ = implementation_fingerprint()
    attempt_root = ROOT / "docs" / "experiments" / "attempts"

    for attempt_name, result_path in SUBMISSION_ATTEMPT_RESULT_MAP.items():
        attempt = _load(attempt_root / attempt_name)
        assert attempt["record_status"] == "RECORDED"
        assert attempt["execution"]["status"] == "PASS"
        assert attempt["execution"]["return_code"] == 0
        assert attempt["environment_before"]["git"]["implementation_sha256"] == (
            current_fingerprint
        )
        assert attempt["environment_after"]["git"]["implementation_sha256"] == (
            current_fingerprint
        )
        assert attempt["result_artifact"]["path"] == result_path.relative_to(
            ROOT
        ).as_posix()
        assert attempt["result_artifact"]["sha256"] == _sha256(result_path)
        assert attempt["metrics"]["status"] == "PASS"


def test_submission_docs_select_fresh_evidence_and_disclose_heldout_slowdown():
    fingerprint = (
        "de768f1ff9ddee54a9ad83a67f3e1f205044c0ad5c723fc3bb4881093c97f611"
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    requirements = (ROOT / "docs" / "REQUIREMENTS.md").read_text(encoding="utf-8")
    result_index = (ROOT / "docs" / "results" / "README.md").read_text(
        encoding="utf-8"
    )
    technical_report = (ROOT / "docs" / "TECH_REPORT.md").read_text(
        encoding="utf-8"
    )
    compliance = (ROOT / "docs" / "TRACK3_COMPLIANCE.md").read_text(
        encoding="utf-8"
    )

    assert all(
        fingerprint in text for text in (readme, requirements, result_index)
    )
    for artifact in SUBMISSION_ATTEMPT_RESULT_MAP.values():
        assert artifact.name in result_index
    assert FINAL_EVALUATOR_PATH.name in readme
    assert FINAL_EVALUATOR_PATH.name in technical_report
    assert ORGANIZER_DEFAULT_PATH.name in technical_report
    assert ORGANIZER_VALIDATION_PATH.name in technical_report
    assert "long-causal" in result_index
    assert "0.793x" in result_index
    assert "0.800x" in result_index
    assert "0.80x" in compliance


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
    assert statistics.geometric_mean(speedups) == pytest.approx(1.210, abs=0.001)


def test_curated_matrix_confirmation_preserves_correctness_and_bounds_variance():
    primary = _load(MATRIX_PATH)
    confirmation = _load(MATRIX_CONFIRMATION_PATH)

    assert confirmation["environment"]["git"]["implementation_sha256"] == (
        primary["environment"]["git"]["implementation_sha256"]
    )
    assert confirmation["summary"] == primary["summary"]
    assert all(result["status"] == "PASS" for result in confirmation["results"])
    assert all(
        sum(
            trial["failed_elements"]
            for trial in result["accuracy"]["trials"]
        )
        == 0
        for result in confirmation["results"]
    )

    primary_by_case = {result["case_id"]: result for result in primary["results"]}
    confirmation_by_case = {
        result["case_id"]: result for result in confirmation["results"]
    }
    assert primary_by_case["long-causal"]["timing"]["speedup_median"] < 1.0
    assert confirmation_by_case["long-causal"]["timing"]["speedup_median"] < 1.0

    confirmation_speedups = [
        result["timing"]["speedup_median"]
        for result in confirmation["results"]
    ]
    assert statistics.geometric_mean(confirmation_speedups) == pytest.approx(
        1.266,
        abs=0.001,
    )


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
    assert profile["backend_counts"] == {"triton": 40, "sdpa": 0, "reference": 0}
    matching = [
        event
        for event in profile["custom_kernel_events"]
        if event["name"] == "_attention_fwd"
    ]
    assert len(matching) == 1
    assert matching[0]["count"] == 40


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
    # Campaign 4 was intentionally measured as a reviewable local candidate;
    # no commit or history mutation was authorized for this optimization round.
    assert evidence["environment"]["git"]["dirty"] is True
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
    assert summary["geometric_mean_speedup"] > 1.7

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
    by_case = {result["case_id"]: result for result in executable}
    row_7 = by_case["final-07-b64-d32-h4-s128"]
    row_11 = by_case["final-11-b64-d128-h16-s128"]
    assert row_7["attention_backend_counts"] == {
        "triton": 0,
        "sdpa": 0,
        "reference": 112,
    }
    assert row_11["attention_backend_counts"] == {
        "triton": 112,
        "sdpa": 0,
        "reference": 0,
    }
    assert row_11["parsed"]["speedup_median"] > 5.0
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


def test_final_evaluator_confirmation_reproduces_primary_result():
    primary = _load(FINAL_EVALUATOR_PATH)
    confirmation = _load(FINAL_CONFIRMATION_PATH)

    assert confirmation["status"] == "PASS"
    assert confirmation["matrix"] == primary["matrix"]
    assert confirmation["environment"]["git"]["implementation_sha256"] == (
        primary["environment"]["git"]["implementation_sha256"]
    )
    assert confirmation["summary"]["counts"] == primary["summary"]["counts"]
    assert confirmation["summary"]["total_failed_elements"] == 0
    assert confirmation["summary"]["attention_backend_counts"] == (
        primary["summary"]["attention_backend_counts"]
    )
    assert confirmation["summary"]["geometric_mean_speedup"] == pytest.approx(
        primary["summary"]["geometric_mean_speedup"],
        rel=0.01,
    )
