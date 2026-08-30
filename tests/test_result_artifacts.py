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
CAMPAIGN11_EVIDENCE_FINGERPRINT = (
    "908a0d708cd8f70f44d5f14fda93d3cafb1cc18345f43914e715594cfa7b7ef9"
)
CAMPAIGN11_RUNNER_SHA256 = (
    "da07a09cafee6a6f6b88413cd6eed8d7904221975913c244c2822b50b5c70a33"
)
CAMPAIGN11_MANIFEST_SHA256 = (
    "b3e5929410a75c69b1e9a0e36af689d15eb9418894d6f6c33bc03f564e0c02ca"
)
CAMPAIGN11_VALIDATION_RUNNER_SHA256 = (
    "b9445eb2a404eca3751899db3a22f9d1cdf0ec35206aabcc7e1961fa4b13f5e4"
)
CAMPAIGN11_FINAL_MATRIX_SHA256 = (
    "76ace44069ed3f27b740e792dfcb6d5be745a760be4c66c9a1241c72a59b24bc"
)


def _text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


MATRIX_PATH = ROOT / "docs" / "results" / (
    "rtx-5070-ti-2026-08-29-c11-integrated-heldout-5seed.json"
)
MATRIX_CONFIRMATION_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c11-integrated-heldout-5seed-confirmation.json"
)
MATRIX_RECHECK_PATHS = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c11-integrated-heldout-recheck-a.json",
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c11-integrated-heldout-recheck-b.json",
)
PROFILE_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c11-integrated-row09-profile.json"
)
ROW9_CONTROL_PROFILE_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c11-baseline-row09-profile.json"
)
ROW5_LONG_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c11-integrated-row05-long.json"
)
ROW9_LONG_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c11-integrated-row09-long.json"
)
ROW11_PROFILE_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c11-integrated-row11-profile.json"
)
ROW8_PROFILE_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c11-integrated-row08-profile.json"
)
ROW6_PROFILE_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c11-integrated-row06-profile.json"
)
ROW6_CONTROL_PROFILE_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c7-baseline-row06-profile.json"
)
ROW6_LONG_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c11-integrated-row06-long.json"
)
ROW6_LONG_CONTROL_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c7-baseline-row06-long-h.json"
)
ROW8_CONTROL_PROFILE_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c6-baseline-row08-profile-c.json"
)
ROW8_LONG_CONTROL_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c6-baseline-row08-long-c.json"
)
ROW8_LONG_CANDIDATE_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c6-exp015-i1r-long-a-row08.json"
)
ORGANIZER_DEFAULT_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c11-integrated-organizer-default.json"
)
ORGANIZER_VALIDATION_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c11-integrated-source-derived.json"
)
FINAL_EVALUATOR_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c11-integrated-final.json"
)
FINAL_CONFIRMATION_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c11-integrated-final-confirmation.json"
)
LONG_CAUSAL_LONG_PATH = (
    ROOT
    / "docs"
    / "results"
    / "rtx-5070-ti-2026-08-29-c11-integrated-long-causal-long.json"
)
FINAL_EVALUATOR_MATRIX_PATH = ROOT / "benchmarks" / "final_evaluator_shapes.json"
ORGANIZER_VALIDATION_MATRIX_PATH = (
    ROOT / "benchmarks" / "organizer_validation_matrix.json"
)
ORGANIZER_MANIFEST_PATH = (
    ROOT / "benchmarks" / "reference" / "organizer_downloads.json"
)
SDPA_CASES = {
    "tiny-overhead",
    "medium-throughput",
    "long-causal",
    "long-causal-padding",
}
SUBMISSION_ATTEMPT_RESULT_MAP = {
    "C11-INTEGRATE-011-final.json": FINAL_EVALUATOR_PATH,
    "C11-INTEGRATE-012-final-confirmation.json": FINAL_CONFIRMATION_PATH,
    "C11-INTEGRATE-013-organizer-default.json": ORGANIZER_DEFAULT_PATH,
    "C11-INTEGRATE-014-heldout-5seed.json": MATRIX_PATH,
    "C11-INTEGRATE-015-heldout-confirmation.json": MATRIX_CONFIRMATION_PATH,
    "C11-INTEGRATE-016-source-derived.json": ORGANIZER_VALIDATION_PATH,
    "C11-INTEGRATE-008-row9-profile.json": PROFILE_PATH,
    "C11-INTEGRATE-006-row9-long.json": ROW9_LONG_PATH,
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
    host_home = Path.home()
    monkeypatch.setattr(
        capture_environment.sys,
        "executable",
        str(host_home / "venv" / "python"),
    )
    payload = capture_environment.capture_environment(
        [
            str(host_home / "venv" / "python"),
            "--evidence-out",
            str(host_home / "private" / "evidence.json"),
            f"--trace={host_home / 'private' / 'trace.json'}",
            f"embedded={host_home}",
        ]
    )

    assert payload["python"]["executable"] == "python"
    assert str(host_home).lower() not in json.dumps(payload["command"]).lower()
    assert payload["command"][2] == "evidence.json"
    assert payload["command"][3] == "--trace=trace.json"
    assert payload["command"][4] == "embedded=<home>"
    assert payload["disk"]["path"] == "."
    assert payload["git"]["implementation_fingerprint_schema"] == 2
    assert payload["cpu"]["name"]
    assert payload["cpu"]["logical_count"]


def test_selected_submission_attempts_bind_results_to_measured_fingerprint():
    attempt_root = ROOT / "docs" / "experiments" / "attempts"

    for attempt_name, result_path in SUBMISSION_ATTEMPT_RESULT_MAP.items():
        attempt = _load(attempt_root / attempt_name)
        assert attempt["record_status"] == "RECORDED"
        assert attempt["execution"]["status"] == "PASS"
        assert attempt["execution"]["return_code"] == 0
        assert attempt["environment_before"]["git"]["implementation_sha256"] == (
            CAMPAIGN11_EVIDENCE_FINGERPRINT
        )
        assert attempt["environment_after"]["git"]["implementation_sha256"] == (
            CAMPAIGN11_EVIDENCE_FINGERPRINT
        )
        assert attempt["result_artifact"]["path"] == result_path.relative_to(
            ROOT
        ).as_posix()
        assert attempt["result_artifact"]["sha256"] == _sha256(result_path)
        assert attempt["metrics"]["status"] == "PASS"


def test_submission_docs_select_campaign11_evidence_and_disclose_removed_regressions():
    current_fingerprint, _ = implementation_fingerprint()
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
        CAMPAIGN11_EVIDENCE_FINGERPRINT in text
        for text in (readme, requirements, result_index)
    )
    assert all(
        current_fingerprint in text for text in (readme, requirements, result_index)
    )
    for artifact in SUBMISSION_ATTEMPT_RESULT_MAP.values():
        assert artifact.name in result_index
    assert FINAL_EVALUATOR_PATH.name in readme
    assert FINAL_EVALUATOR_PATH.name in technical_report
    assert ORGANIZER_DEFAULT_PATH.name in technical_report
    assert ORGANIZER_VALIDATION_PATH.name in technical_report
    assert "long-causal" in result_index
    assert "1.198x" in result_index
    assert "1.213x" in result_index
    assert "fused residual" in result_index
    assert "removed both held-out long-causal regressions" in compliance


def test_curated_campaign11_matrix_is_complete_green_and_immutable():
    matrix = _load(MATRIX_PATH)
    _, current_paths = implementation_fingerprint()
    captured_git = matrix["environment"]["git"]
    assert captured_git["implementation_sha256"] == CAMPAIGN11_EVIDENCE_FINGERPRINT
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
    assert statistics.geometric_mean(speedups) == pytest.approx(1.340, abs=0.001)


def test_curated_matrix_confirmation_preserves_correctness_and_bounds_variance():
    primary = _load(MATRIX_PATH)
    confirmation = _load(MATRIX_CONFIRMATION_PATH)
    rechecks = [_load(path) for path in MATRIX_RECHECK_PATHS]
    long_causal_long = _load(LONG_CAUSAL_LONG_PATH)["results"][0]

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

    matrices = [primary, confirmation, *rechecks]
    by_case = [
        {result["case_id"]: result for result in matrix["results"]}
        for matrix in matrices
    ]
    long_causal_speedups = [
        results["long-causal"]["timing"]["speedup_median"]
        for results in by_case
    ]
    assert min(long_causal_speedups) > 1.19
    assert max(long_causal_speedups) - min(long_causal_speedups) < 0.01
    assert all(
        results["long-causal-padding"]["timing"]["speedup_median"] > 1.2
        for results in by_case
    )
    assert long_causal_long["status"] == "PASS"
    assert len(long_causal_long["timing"]["optimized"]["raw_ms"]) == 300
    assert long_causal_long["timing"]["speedup_median"] > 1.19
    assert long_causal_long["timing"]["backend_counts"] == {
        "triton": 0,
        "sdpa": 620,
        "reference": 0,
    }
    assert sum(
        trial["failed_elements"] for trial in long_causal_long["accuracy"]["trials"]
    ) == 0

    confirmation_speedups = [
        result["timing"]["speedup_median"]
        for result in confirmation["results"]
    ]
    assert statistics.geometric_mean(confirmation_speedups) == pytest.approx(
        1.386,
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
    assert profile["backend_counts"] == {"triton": 120, "sdpa": 0, "reference": 0}
    matching = [
        event
        for event in profile["custom_kernel_events"]
        if event["name"] == "_attention_fwd"
    ]
    assert len(matching) == 1
    assert matching[0]["count"] == 120

    control = _load(ROW9_CONTROL_PROFILE_PATH)

    def optional_event(payload, name):
        return next(
            (item for item in payload["top_events"] if item["name"] == name),
            None,
        )

    control_add = optional_event(control, "aten::add")
    control_norm = optional_event(control, "aten::native_layer_norm")
    current_add = optional_event(profile, "aten::add")
    current_norm = optional_event(profile, "aten::native_layer_norm")
    current_fused = optional_event(profile, "_residual_layer_norm_fwd")
    assert (control_add["count"], control_norm["count"]) == (240, 270)
    assert current_add is None
    assert (current_fused["count"], current_norm["count"]) == (240, 30)
    control_subsystem = (
        control_add["self_device_time_us"] + control_norm["self_device_time_us"]
    )
    current_subsystem = (
        current_fused["self_device_time_us"] + current_norm["self_device_time_us"]
    )
    assert current_subsystem < control_subsystem * 0.7


def test_campaign11_row9_long_gate_is_fast_correct_and_memory_bounded():
    result = _load(ROW9_LONG_PATH)["results"][0]

    assert result["case_id"] == "final-09-b64-d128-h1-s128"
    assert result["status"] == "PASS"
    assert sum(
        trial["failed_elements"] for trial in result["accuracy"]["trials"]
    ) == 0
    assert len(result["timing"]["optimized"]["raw_ms"]) == 300
    assert result["timing"]["optimized"]["median_ms"] < 0.75
    assert result["timing"]["speedup_median"] > 1.1
    assert result["timing"]["backend_counts"] == {
        "triton": 1240,
        "sdpa": 0,
        "reference": 0,
    }
    assert result["peak_memory"]["optimized"]["incremental_peak_bytes"] == (
        29_360_128
    )


def test_campaign11_inherited_row5_long_gate_remains_fast_and_memory_bounded():
    result = _load(ROW5_LONG_PATH)["results"][0]

    assert result["case_id"] == "final-05-b128-d128-h4-s128"
    assert result["status"] == "PASS"
    assert sum(
        trial["failed_elements"] for trial in result["accuracy"]["trials"]
    ) == 0
    assert len(result["timing"]["optimized"]["raw_ms"]) == 300
    assert result["timing"]["optimized"]["median_ms"] < 1.2
    assert result["timing"]["speedup_median"] > 1.8
    assert result["timing"]["backend_counts"] == {
        "triton": 1240,
        "sdpa": 0,
        "reference": 0,
    }
    assert result["peak_memory"]["optimized"]["incremental_peak_bytes"] == (
        58_720_256
    )


def test_campaign11_row11_profile_preserves_exact_fused_route():
    attempt = _load(
        ROOT
        / "docs"
        / "experiments"
        / "attempts"
        / "C11-INTEGRATE-021-row11-profile.json"
    )
    profile = _load(ROW11_PROFILE_PATH)

    assert attempt["execution"]["status"] == "PASS"
    assert attempt["result_artifact"]["sha256"] == _sha256(ROW11_PROFILE_PATH)
    assert profile["environment"]["git"]["implementation_sha256"] == (
        CAMPAIGN11_EVIDENCE_FINGERPRINT
    )
    assert profile["backend_counts"] == {
        "triton": 120,
        "sdpa": 0,
        "reference": 0,
    }
    fused = next(
        event
        for event in profile["top_events"]
        if event["name"] == "_residual_layer_norm_fwd"
    )
    native_norm = next(
        event
        for event in profile["top_events"]
        if event["name"] == "aten::native_layer_norm"
    )
    assert (fused["count"], native_norm["count"]) == (240, 30)


def test_campaign11_row8_profile_preserves_packed_qkv_projection_reduction():
    attempt = _load(
        ROOT
        / "docs"
        / "experiments"
        / "attempts"
        / "C11-INTEGRATE-026-row8-profile.json"
    )
    profile = _load(ROW8_PROFILE_PATH)
    control = _load(ROW8_CONTROL_PROFILE_PATH)

    assert attempt["execution"]["status"] == "PASS"
    assert attempt["metrics"]["kind"] == "profile"
    assert attempt["metrics"]["status"] == "INCONCLUSIVE"
    assert attempt["result_artifact"]["path"] == ROW8_PROFILE_PATH.relative_to(
        ROOT
    ).as_posix()
    assert attempt["result_artifact"]["sha256"] == _sha256(ROW8_PROFILE_PATH)
    assert profile["environment"]["git"]["implementation_sha256"] == (
        CAMPAIGN11_EVIDENCE_FINGERPRINT
    )
    assert profile["backend_counts"] == {"triton": 0, "sdpa": 0, "reference": 40}

    def event(payload, name):
        return next(
            item
            for item in payload["top_events"]
            if item["name"] == name and item["self_device_time_us"] > 0
        )

    current_addmm = event(profile, "aten::addmm")
    control_addmm = event(control, "aten::addmm")
    current_model = event(profile, "optimized_transformer")
    control_model = event(control, "optimized_transformer")

    assert (control_addmm["count"], current_addmm["count"]) == (240, 160)
    assert current_addmm["self_device_time_us"] < control_addmm["self_device_time_us"]
    assert current_model["self_device_time_us"] < control_model["self_device_time_us"]


def test_campaign11_row6_profile_proves_fused_residual_norm_reduction():
    attempt = _load(
        ROOT
        / "docs"
        / "experiments"
        / "attempts"
        / "C11-INTEGRATE-022-row6-profile.json"
    )
    profile = _load(ROW6_PROFILE_PATH)
    control = _load(ROW6_CONTROL_PROFILE_PATH)

    assert attempt["execution"]["status"] == "PASS"
    assert attempt["result_artifact"]["sha256"] == _sha256(ROW6_PROFILE_PATH)
    assert profile["environment"]["git"]["implementation_sha256"] == (
        CAMPAIGN11_EVIDENCE_FINGERPRINT
    )
    assert profile["backend_counts"] == {"triton": 20, "sdpa": 0, "reference": 20}

    def optional_event(payload, name):
        return next(
            (item for item in payload["top_events"] if item["name"] == name),
            None,
        )

    control_add = optional_event(control, "aten::add")
    control_norm = optional_event(control, "aten::native_layer_norm")
    current_add = optional_event(profile, "aten::add")
    current_norm = optional_event(profile, "aten::native_layer_norm")
    current_fused = optional_event(profile, "_residual_layer_norm_fwd")
    control_model = optional_event(control, "optimized_transformer")
    current_model = optional_event(profile, "optimized_transformer")

    assert (control_add["count"], control_norm["count"]) == (80, 90)
    assert current_add is None
    assert (current_fused["count"], current_norm["count"]) == (80, 10)
    control_subsystem = (
        control_add["self_device_time_us"] + control_norm["self_device_time_us"]
    )
    current_subsystem = (
        current_fused["self_device_time_us"] + current_norm["self_device_time_us"]
    )
    assert current_subsystem < control_subsystem * 0.7
    assert current_model["self_device_time_us"] < control_model["self_device_time_us"]


def test_campaign11_row6_long_run_is_faster_and_memory_neutral():
    control = _load(ROW6_LONG_CONTROL_PATH)["results"][0]
    candidate = _load(ROW6_LONG_PATH)["results"][0]

    assert sum(
        trial["failed_elements"] for trial in candidate["accuracy"]["trials"]
    ) == 0
    assert candidate["timing"]["speedup_median"] > (
        control["timing"]["speedup_median"] * 1.05
    )
    assert candidate["peak_memory"]["optimized"]["incremental_peak_bytes"] == (
        control["peak_memory"]["optimized"]["incremental_peak_bytes"]
    ) == 11_802_787_840


def test_campaign6_row8_long_run_records_speed_and_memory_tradeoff():
    control = _load(ROW8_LONG_CONTROL_PATH)["results"][0]
    candidate = _load(ROW8_LONG_CANDIDATE_PATH)["results"][0]

    assert (
        sum(
            trial["failed_elements"]
            for trial in candidate["accuracy"]["trials"]
        )
        == 0
    )
    assert control["timing"]["speedup_median"] < 1.0
    assert candidate["timing"]["speedup_median"] > 1.0

    control_memory = control["peak_memory"]["optimized"]
    candidate_memory = candidate["peak_memory"]["optimized"]
    assert (
        candidate_memory["allocated_before_bytes"]
        - control_memory["allocated_before_bytes"]
        == 50_380_800
    )
    assert candidate_memory["incremental_peak_bytes"] == control_memory[
        "incremental_peak_bytes"
    ] == 369_115_136


def test_campaign11_organizer_default_artifact_uses_untouched_harness():
    evidence = _load(ORGANIZER_DEFAULT_PATH)
    manifest = _load(ORGANIZER_MANIFEST_PATH)
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
    assert evidence["submission"]["runner_sha256"] == CAMPAIGN11_RUNNER_SHA256
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
        CAMPAIGN11_EVIDENCE_FINGERPRINT
    )


def test_campaign11_organizer_validation_artifact_is_complete_and_fail_closed():
    evidence = _load(ORGANIZER_VALIDATION_PATH)

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
    assert evidence["organizer_sources"]["manifest_sha256"] == (
        CAMPAIGN11_MANIFEST_SHA256
    )
    assert evidence["organizer_sources"]["runner_sha256"] == (
        CAMPAIGN11_RUNNER_SHA256
    )
    assert evidence["organizer_sources"]["validation_runner_sha256"] == (
        CAMPAIGN11_VALIDATION_RUNNER_SHA256
    )
    assert evidence["environment"]["git"]["implementation_sha256"] == (
        CAMPAIGN11_EVIDENCE_FINGERPRINT
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


def test_campaign11_final_evaluator_artifact_is_complete_and_immutable():
    evidence = _load(FINAL_EVALUATOR_PATH)

    assert evidence["status"] == "PASS"
    assert evidence["matrix"] == {
        "path": "benchmarks/final_evaluator_shapes.json",
        "sha256": CAMPAIGN11_FINAL_MATRIX_SHA256,
        "status": "organizer-published-final-shapes",
    }
    # Campaign 11 is intentionally measured as a reviewable local candidate;
    # no commit or history mutation is authorized by the optimization request.
    assert evidence["environment"]["git"]["dirty"] is True
    assert evidence["environment"]["git"]["implementation_sha256"] == (
        CAMPAIGN11_EVIDENCE_FINGERPRINT
    )
    assert evidence["organizer_sources"]["manifest_sha256"] == (
        CAMPAIGN11_MANIFEST_SHA256
    )
    assert evidence["organizer_sources"]["runner_sha256"] == (
        CAMPAIGN11_RUNNER_SHA256
    )
    assert evidence["organizer_sources"]["validation_runner_sha256"] == (
        CAMPAIGN11_VALIDATION_RUNNER_SHA256
    )

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
    assert summary["geometric_mean_speedup"] > 1.8

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
    row_6 = by_case["final-06-b10000-d128-h4-s128"]
    row_7 = by_case["final-07-b64-d32-h4-s128"]
    row_11 = by_case["final-11-b64-d128-h16-s128"]
    assert row_6["attention_backend_counts"] == {
        "triton": 56,
        "sdpa": 0,
        "reference": 56,
    }
    assert row_7["attention_backend_counts"] == {
        "triton": 84,
        "sdpa": 0,
        "reference": 28,
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
    assert primary["summary"]["geometric_mean_speedup"] > 1.8
    assert confirmation["summary"]["geometric_mean_speedup"] > 1.8
    assert confirmation["summary"]["geometric_mean_speedup"] == pytest.approx(
        primary["summary"]["geometric_mean_speedup"], rel=0.05
    )
