"""Regression tests for benchmark status and false-green handling."""

from __future__ import annotations

from benchmarks.run_matrix import execute_cases, is_oom_error, result_exit_code


def test_zero_case_run_fails():
    assert result_exit_code([]) == 1


def test_only_all_pass_is_green():
    assert result_exit_code([{"status": "PASS"}, {"status": "PASS"}]) == 0
    for status in ("FAIL", "OOM", "ERROR", "SKIPPED", None):
        assert result_exit_code([{"status": "PASS"}, {"status": status}]) == 1


def test_oom_classification_is_narrow():
    assert is_oom_error(RuntimeError("CUDA out of memory"))
    assert not is_oom_error(RuntimeError("kernel compilation failed"))


def test_unexpected_runtime_error_becomes_error_not_skip():
    case = {"id": "broken"}

    def broken_runner(_case, _dtype):
        raise RuntimeError("kernel compilation failed")

    results = execute_cases([(case, "float32")], broken_runner)
    assert results[0]["status"] == "ERROR"
    assert results[0]["error"]["type"] == "RuntimeError"
    assert result_exit_code(results) == 1


def test_oom_becomes_failing_oom_state():
    case = {"id": "too-large"}

    def oom_runner(_case, _dtype):
        raise RuntimeError("CUDA out of memory while allocating")

    results = execute_cases([(case, "float32")], oom_runner)
    assert results[0]["status"] == "OOM"
    assert result_exit_code(results) == 1
