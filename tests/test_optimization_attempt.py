"""Tests for immutable optimization-attempt execution and metric extraction."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.run_optimization_attempt import (
    _pump_stream,
    load_result_artifact,
    portable_command,
    portable_output,
    run_logged_command,
    summarize_result_artifact,
    write_json_exclusive,
)


ROOT = Path(__file__).resolve().parents[1]


def test_persisted_command_removes_machine_specific_absolute_prefixes():
    command = portable_command([sys.executable, "benchmarks/profile_cases.py"])
    assert command[0] == Path(sys.executable).name
    assert str(Path.home()).lower() not in " ".join(command).lower()
    assert command[1] == "benchmarks/profile_cases.py"


def test_persisted_output_removes_machine_specific_absolute_prefixes():
    home = str(Path.home())
    repo = str(ROOT)
    output = portable_output(
        f"repo={repo}\\artifact.json\nhome={home}\\runtime\\warning.py\n"
        f"slash={Path.home().as_posix()}/runtime/warning.py\n"
    )

    assert repo.lower() not in output.lower()
    assert home.lower() not in output.lower()
    assert "repo=.\\artifact.json" in output
    assert "home=<home>\\runtime\\warning.py" in output
    assert "slash=<home>/runtime/warning.py" in output


def test_validation_summary_preserves_correctness_performance_and_duration():
    payload = {
        "matrix": {},
        "organizer_sources": {},
        "status": "PASS",
        "summary": {
            "requested": 2,
            "executable": 2,
            "passed": 2,
            "skipped_resource": 0,
            "counts": {"PASS": 2, "FAIL": 0},
            "total_compared_elements": 100,
            "total_failed_elements": 0,
            "max_abs_error": 0.0005,
            "max_relative_error": 0.02,
            "geometric_mean_speedup": 1.25,
            "attention_backend_counts": {"triton": 20, "sdpa": 0, "reference": 0},
        },
        "parameters": {"accuracy_trials": 5},
        "results": [
            {
                "case_id": "one",
                "status": "PASS",
                "duration_seconds": 2.5,
                "parsed": {
                    "accuracy": {"failed_elements": 0},
                    "baseline": {"median_ms": 2.0},
                    "optimized": {"median_ms": 1.0},
                    "speedup_median": 2.0,
                },
                "attention_backend_counts": {"triton": 10, "sdpa": 0, "reference": 0},
            },
            {
                "case_id": "two",
                "status": "PASS",
                "duration_seconds": 3.5,
                "parsed": {"speedup_median": 1.0},
                "attention_backend_counts": {"triton": 10, "sdpa": 0, "reference": 0},
            },
        ],
    }

    summary = summarize_result_artifact(payload)

    assert summary["kind"] == "organizer_validation"
    assert summary["correctness"]["total_failed_elements"] == 0
    assert summary["correctness"]["accuracy_trials"] == 10
    assert summary["performance"]["geometric_mean_speedup"] == 1.25
    assert summary["performance"]["total_case_duration_seconds"] == 6.0
    assert summary["attention_backend_counts"]["triton"] == 20


def test_matrix_summary_calculates_trials_samples_memory_and_geomean():
    payload = {
        "manifest": {},
        "parameters": {},
        "summary": {"requested": 1, "counts": {"PASS": 1}},
        "results": [
            {
                "case_id": "case",
                "status": "PASS",
                "accuracy": {
                    "trials": [
                        {
                            "total_elements": 8,
                            "failed_elements": 0,
                            "max_abs_error": 0.0002,
                            "max_relative_error": 0.03,
                        }
                    ]
                },
                "timing": {
                    "baseline": {"median_ms": 2.0, "raw_ms": [1.0, 2.0, 3.0]},
                    "optimized": {"median_ms": 1.0, "raw_ms": [0.5, 1.0, 1.5]},
                    "speedup_median": 2.0,
                    "backend_counts": {"triton": 5, "sdpa": 0, "reference": 0},
                },
                "peak_memory": {
                    "baseline": {"incremental_peak_bytes": 100},
                    "optimized": {"incremental_peak_bytes": 40},
                },
            }
        ],
    }

    summary = summarize_result_artifact(payload)

    assert summary["kind"] == "project_matrix"
    assert summary["correctness"]["accuracy_trials"] == 1
    assert summary["correctness"]["total_compared_elements"] == 8
    assert summary["performance"]["geometric_mean_speedup"] == 2.0
    case = summary["performance"]["per_case"][0]
    assert case["baseline"]["sample_count"] == 3
    assert case["baseline"]["sample_stdev_ms"] == 1.0
    assert case["memory"]["reduction_percent"] == pytest.approx(60.0)


def test_profile_summary_keeps_custom_events_and_backend_counts():
    payload = {
        "backend_counts": {"triton": 40, "sdpa": 0, "reference": 0},
        "custom_kernel_expected": True,
        "custom_kernel_profiler_proven": True,
        "custom_kernel_events": [
            {"name": "_attention_fwd", "count": 40, "self_device_time_us": 12.5}
        ],
        "top_events": [],
        "steps": 10,
    }

    summary = summarize_result_artifact(payload)

    assert summary["kind"] == "profile"
    assert summary["status"] == "PASS"
    assert summary["profiler"]["custom_kernel_events"][0]["count"] == 40


def test_profile_summary_respects_explicit_failed_expectation():
    payload = {
        "schema_version": 2,
        "backend_counts": {"triton": 4, "sdpa": 0, "reference": 0},
        "backend_expected": "reference",
        "custom_kernel_expected": False,
        "custom_kernel_profiler_proven": True,
        "custom_kernel_events": [
            {"name": "_attention_fwd", "count": 4, "self_device_time_us": 1.0}
        ],
        "fused_residual_layer_norm_expected": False,
        "fused_residual_layer_norm_calls": 8,
        "fused_residual_layer_norm_events": [
            {"name": "_residual_layer_norm_fwd", "count": 8}
        ],
        "expectation_checks": [
            {
                "kind": "attention_backend",
                "expected": "reference",
                "dispatch_count": 0,
                "passed": False,
            }
        ],
        "validation_passed": False,
        "top_events": [],
        "steps": 1,
    }

    summary = summarize_result_artifact(payload)

    assert summary["status"] == "FAIL"
    assert summary["profiler"]["validation_passed"] is False
    assert summary["profiler"]["backend_expected"] == "reference"
    assert summary["profiler"]["fused_residual_layer_norm_calls"] == 8


def test_profile_summary_keeps_schema2_observation_inconclusive():
    payload = {
        "schema_version": 2,
        "backend_counts": {"triton": 4, "sdpa": 0, "reference": 0},
        "backend_expected": None,
        "custom_kernel_expected": False,
        "custom_kernel_profiler_proven": True,
        "custom_kernel_events": [
            {"name": "_attention_fwd", "count": 4, "self_device_time_us": 1.0}
        ],
        "expectation_checks": [],
        "validation_passed": None,
        "top_events": [],
        "steps": 1,
    }

    summary = summarize_result_artifact(payload)

    assert summary["status"] == "INCONCLUSIVE"
    assert summary["profiler"]["validation_passed"] is None


def test_failed_command_is_captured_instead_of_disappearing():
    result = run_logged_command(
        [
            sys.executable,
            "-c",
            "import sys; print('logged stdout'); print('logged stderr', file=sys.stderr); sys.exit(3)",
        ],
        tee=False,
    )

    assert result["status"] == "FAIL"
    assert result["return_code"] == 3
    assert result["wall_time_seconds"] >= 0
    assert "logged stdout" in result["stdout"]
    assert "logged stderr" in result["stderr"]


def test_live_tee_replaces_characters_unsupported_by_console_encoding():
    raw_sink = io.BytesIO()
    sink = io.TextIOWrapper(raw_sink, encoding="ascii", errors="strict")
    chunks: list[str] = []

    _pump_stream(io.StringIO("before \ufffd after\n"), sink, chunks, tee=True)
    sink.flush()

    assert chunks == ["before \ufffd after\n"]
    assert raw_sink.getvalue().decode("ascii").splitlines() == ["before ? after"]


def test_timed_out_command_is_captured():
    result = run_logged_command(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        timeout_seconds=0.05,
        tee=False,
    )

    assert result["status"] == "TIMEOUT"
    assert result["timed_out"] is True
    assert result["wall_time_seconds"] < 5


def test_direct_file_cli_logs_nonzero_command(tmp_path):
    out = tmp_path / "direct-cli.json"
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/run_optimization_attempt.py",
            "--attempt-id",
            "direct-cli-selftest",
            "--hypothesis",
            "Direct file execution imports shared helpers",
            "--scope",
            "logger self-test",
            "--rollback",
            "No repository change",
            "--decision",
            "reject",
            "--decision-rationale",
            "Expected child failure",
            "--review-status",
            "not_required",
            "--out",
            str(out),
            "--",
            sys.executable,
            "-c",
            "import sys; print('captured'); sys.exit(3)",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 3
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["record_status"] == "FAIL"
    assert record["execution"]["return_code"] == 3
    assert "captured" in record["execution"]["stdout"]


def test_missing_and_invalid_artifacts_are_explicit(tmp_path):
    missing_metadata, missing_payload = load_result_artifact(tmp_path / "missing.json")
    assert missing_payload is None
    assert missing_metadata["exists"] is False
    assert "parse_error" in missing_metadata

    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    invalid_metadata, invalid_payload = load_result_artifact(invalid)
    assert invalid_payload is None
    assert invalid_metadata["exists"] is True
    assert "parse_error" in invalid_metadata


def test_attempt_records_are_exclusive(tmp_path):
    out = tmp_path / "attempt.json"
    write_json_exclusive(out, {"schema_version": 1})
    assert json.loads(out.read_text(encoding="utf-8"))["schema_version"] == 1
    with pytest.raises(FileExistsError):
        write_json_exclusive(out, {"schema_version": 2})
