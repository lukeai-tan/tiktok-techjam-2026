from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from benchmarks.run_organizer_torch import (
    _text_sha256 as organizer_runner_text_sha256,
    install_submission,
    load_organizer_benchmark,
    main as run_organizer_torch,
    verify_organizer_download,
)
from benchmarks.run_organizer_validation import (
    _text_sha256 as validation_runner_text_sha256,
    load_and_expand_matrix,
    organizer_arguments,
    result_exit_code,
    summarize_results,
)
import transformer_opt.submission as submission
from transformer_opt.submission import UserOptimizedTransformer


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "benchmarks" / "reference" / "organizer_downloads.json"
ORGANIZER_TORCH = ROOT / "benchmarks" / "torch_transformer_benchmark.py"
ORGANIZER_TENSORFLOW = ROOT / "benchmarks" / "tensorflow_transformer_benchmark.py"
VALIDATION_MATRIX = ROOT / "benchmarks" / "organizer_validation_matrix.json"
FINAL_EVALUATOR_MATRIX = ROOT / "benchmarks" / "final_evaluator_shapes.json"
FINAL_PROFILE_MATRIX = ROOT / "benchmarks" / "final_profile_shapes.json"
CAMPAIGN4_PROFILE_MATRIX = ROOT / "benchmarks" / "campaign4_profile_shapes.json"
CAMPAIGN5_PROFILE_MATRIX = ROOT / "benchmarks" / "campaign5_profile_shapes.json"
OFFICIAL_MATRIX = ROOT / "benchmarks" / "official_shapes.json"

PROTECTED_TORCH_DEFINITIONS = (
    "TransformerConfig",
    "BaselineSelfAttention",
    "BaselineTransformerBlock",
    "BaselineTransformer",
    "copy_model_weights",
    "resolve_device",
    "resolve_dtype",
    "AccuracyResult",
    "compare_outputs",
    "percentile",
    "TimingResult",
    "maybe_compile",
    "validate_args",
)


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_repository_text_hashes_ignore_checkout_line_endings(tmp_path):
    lf_path = tmp_path / "lf.txt"
    crlf_path = tmp_path / "crlf.txt"
    lf_path.write_bytes(b"first\nsecond\n")
    crlf_path.write_bytes(b"first\r\nsecond\r\n")

    expected = organizer_runner_text_sha256(lf_path)
    assert organizer_runner_text_sha256(crlf_path) == expected
    assert validation_runner_text_sha256(lf_path) == expected
    assert validation_runner_text_sha256(crlf_path) == expected


def _top_level_definitions(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _argparse_defaults(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        option = node.args[0]
        if not isinstance(option, ast.Constant) or not isinstance(option.value, str):
            continue
        default = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "default"),
            None,
        )
        if default is not None:
            try:
                defaults[option.value] = ast.literal_eval(default)
            except ValueError:
                # Non-literal defaults such as Path("report.md") are unrelated
                # to the organizer dimension axes audited by this helper.
                continue
    return defaults


def test_supplied_organizer_downloads_match_frozen_hashes():
    manifest = _manifest()
    artifacts = {artifact["framework"]: artifact for artifact in manifest["artifacts"]}

    assert manifest["source"]["challenge_brief_path"] == (
        "hackathon-docs/hackathon-details.md"
    )
    assert _sha256(ORGANIZER_TORCH) == artifacts["pytorch"]["sha256"]
    assert ORGANIZER_TORCH.stat().st_size == artifacts["pytorch"]["size_bytes"]
    assert _sha256(ORGANIZER_TENSORFLOW) == artifacts["tensorflow"]["sha256"]
    assert ORGANIZER_TENSORFLOW.stat().st_size == artifacts["tensorflow"]["size_bytes"]
    assert verify_organizer_download() == artifacts["pytorch"]["sha256"]


def test_organizer_loader_rejects_same_name_bound_to_foreign_file(
    monkeypatch, tmp_path
):
    foreign = ModuleType("benchmarks.torch_transformer_benchmark")
    foreign.__file__ = str(tmp_path / "torch_transformer_benchmark.py")
    monkeypatch.setitem(sys.modules, foreign.__name__, foreign)

    with pytest.raises(RuntimeError, match="already bound to a different file"):
        load_organizer_benchmark()


@pytest.mark.parametrize(
    "non_strict_argument",
    ["--non-strict-weight-copy", "--non-strict-weight-cop"],
)
def test_evidence_output_rejects_non_strict_weight_copy(
    tmp_path, non_strict_argument
):
    evidence_path = tmp_path / "evidence.json"
    with pytest.raises(ValueError, match="diagnostic-only"):
        run_organizer_torch(
            [
                "--evidence-out",
                str(evidence_path),
                non_strict_argument,
            ]
        )

    assert not evidence_path.exists()


def test_submission_reuses_organizer_pytorch_baseline_contract():
    organizer = _top_level_definitions(ORGANIZER_TORCH)
    submission_source = _top_level_definitions(
        ROOT / "transformer_opt" / "submission.py"
    )

    assert "UserOptimizedTransformer" in submission_source
    assert submission.UserOptimizedTransformer.__mro__[1] is (
        submission.BaselineTransformer
    )
    for name in PROTECTED_TORCH_DEFINITIONS:
        assert getattr(submission, name) is getattr(
            load_organizer_benchmark(), name
        ), name
        assert name in organizer, name


def test_tensorflow_download_is_the_single_canonical_copy():
    artifact = next(
        artifact
        for artifact in _manifest()["artifacts"]
        if artifact["framework"] == "tensorflow"
    )

    assert artifact["path"] == "benchmarks/tensorflow_transformer_benchmark.py"
    assert _manifest()["reconciliation"][
        "tensorflow_download_is_single_canonical_copy"
    ] is True
    assert not (ROOT / "tensorflow_transformer_benchmark.py").exists()


def test_manifest_matches_tensorflow_default_shape_axes():
    defaults = _argparse_defaults(ORGANIZER_TENSORFLOW)
    axes = _manifest()["tensorflow_contract"]["default_axes"]

    assert defaults["--batch-sizes"] == axes["batch_sizes"]
    assert defaults["--qkv-dims"] == axes["qkv_dims"]
    assert defaults["--heads"] == axes["heads"]
    assert defaults["--seq-lens"] == axes["seq_lens"]


def test_untouched_organizer_harness_accepts_submission_on_cpu():
    organizer = load_organizer_benchmark()
    install_submission(organizer)
    config = organizer.TransformerConfig(1, 8, 32, 4, 128, 1, False)
    baseline = organizer.BaselineTransformer(config).eval()
    optimized = organizer.UserOptimizedTransformer(config).eval()
    organizer.copy_model_weights(baseline, optimized, strict=True)
    x = torch.randn(1, 8, 32)
    valid_mask = torch.ones(1, 8, dtype=torch.bool)

    with torch.inference_mode():
        reference = baseline(x, valid_mask)
        candidate = optimized(x, valid_mask)
    result = organizer.compare_outputs(reference, candidate, rtol=0.01, atol=0.001)

    assert organizer.UserOptimizedTransformer is UserOptimizedTransformer
    assert result.passed
    assert baseline.state_dict().keys() == optimized.state_dict().keys()


def test_validation_matrix_covers_every_feasible_supplied_shape_and_dtype():
    matrix, cases = load_and_expand_matrix(VALIDATION_MATRIX)
    manifest = _manifest()
    compact_cases = manifest["tensorflow_contract"]["compact_cases"]
    stress_case = compact_cases[-1]

    assert len(cases) == 29
    assert sum(case["execution"] == "run" for case in cases) == 28
    assert sum(case["execution"] == "skip_resource" for case in cases) == 1

    translated = [
        case
        for case in cases
        if case["source_contract"] == "tensorflow-shape-cross-check"
    ]
    for dimensions in compact_cases[:-1]:
        matching = [
            case for case in translated if case["source_dimensions"] == dimensions
        ]
        assert {case["dtype"] for case in matching} == {"float32", "float16"}
        assert all(case["execution"] == "run" for case in matching)

    skipped = [case for case in translated if case["execution"] == "skip_resource"]
    assert len(skipped) == 1
    assert skipped[0]["source_dimensions"] == stress_case
    assert skipped[0]["skip_authorized"] is True
    assert skipped[0]["dtype"] == manifest["tensorflow_contract"]["default_dtype"]
    assert matrix["defaults"]["accuracy_trials"] == 5


def test_validation_matrix_exercises_pytorch_modes_under_strict_contract():
    matrix, cases = load_and_expand_matrix(VALIDATION_MATRIX)
    pytorch_cases = [case for case in cases if case["source_contract"] == "pytorch"]
    parser_defaults = _argparse_defaults(ORGANIZER_TORCH)

    assert {case["dtype"] for case in pytorch_cases} == {
        "float32",
        "float16",
        "bfloat16",
    }
    assert {case["config"]["causal"] for case in pytorch_cases} == {False, True}
    assert {case["padding_ratio"] for case in pytorch_cases} == {0.0, 0.3}
    assert matrix["defaults"]["atol"] == 0.001
    assert matrix["defaults"]["rtol"] == 0.01
    assert parser_defaults["--atol"] == 0.002
    assert parser_defaults["--rtol"] == 0.02

    causal_case = next(case for case in pytorch_cases if case["config"]["causal"])
    arguments = organizer_arguments(causal_case, matrix["defaults"], "cuda")
    assert "--causal" in arguments
    assert arguments[arguments.index("--accuracy-trials") + 1] == "5"
    assert arguments[arguments.index("--atol") + 1] == "0.001"
    assert arguments[arguments.index("--rtol") + 1] == "0.01"


def test_final_evaluator_matrix_preserves_published_row_order_and_values():
    matrix, cases = load_and_expand_matrix(FINAL_EVALUATOR_MATRIX)
    expected_dimensions = [
        [64, 128, 4, 128, 4, True, 128],
        [1, 128, 4, 128, 4, True, 128],
        [4, 128, 4, 128, 4, True, 128],
        [16, 128, 4, 128, 4, True, 128],
        [128, 128, 4, 128, 4, True, 128],
        [10000, 128, 4, 128, 4, True, 128],
        [64, 32, 4, 128, 4, True, 32],
        [64, 1024, 4, 128, 4, True, 1024],
        [64, 128, 1, 128, 4, True, 128],
        [64, 128, 2, 128, 4, True, 128],
        [64, 128, 16, 128, 4, True, 128],
        [64, 128, 4, 32, 4, True, 128],
        [64, 128, 4, 1024, 4, True, 128],
        [32, 1024, 16, 100000, 2, True, 1024],
    ]

    assert matrix["status"] == "organizer-published-final-shapes"
    assert matrix["source"]["section"] == "3.7 Appendix - Test shapes"
    assert matrix["source_omissions"] == [
        "framework",
        "dtype",
        "padding ratio",
        "correctness tolerance",
        "timing protocol",
        "backward or gradient requirement",
    ]
    assert [case["source_row"] for case in cases] == list(range(1, 15))
    assert [case["source_dimensions"] for case in cases] == expected_dimensions
    assert all(case["dtype"] == "float32" for case in cases)
    assert all(case["padding_ratio"] == 0.0 for case in cases)
    assert sum(case["execution"] == "run" for case in cases) == 13
    skipped = [case for case in cases if case["execution"] == "skip_resource"]
    assert len(skipped) == 1
    assert skipped[0]["source_row"] == 14
    assert skipped[0]["skip_authorized"] is True


def test_final_evaluator_matrix_rejects_transcription_drift(tmp_path):
    payload = json.loads(FINAL_EVALUATOR_MATRIX.read_text(encoding="utf-8"))
    payload["explicit_cases"][0]["source_dimensions"][0] = 63
    altered = tmp_path / "altered-final-shapes.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source dimensions do not match config"):
        load_and_expand_matrix(altered)


def test_final_evaluator_matrix_rejects_unauthorized_skip(tmp_path):
    payload = json.loads(FINAL_EVALUATOR_MATRIX.read_text(encoding="utf-8"))
    payload["explicit_cases"][0]["execution"] = "skip_resource"
    payload["explicit_cases"][0]["skip_authorized"] = True
    payload["explicit_cases"][-1]["execution"] = "run"
    payload["explicit_cases"][-1].pop("skip_authorized")
    altered = tmp_path / "altered-final-skip.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match="resource skip must target the exact final-row stress dimensions"
    ):
        load_and_expand_matrix(altered)


def test_final_evaluator_matrix_rejects_stress_shape_drift(tmp_path):
    payload = json.loads(FINAL_EVALUATOR_MATRIX.read_text(encoding="utf-8"))
    stress = payload["explicit_cases"][-1]
    stress["source_dimensions"][3] = 99_999
    stress["config"]["seq_len"] = 99_999
    altered = tmp_path / "altered-final-stress.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match="resource skip must target the exact final-row stress dimensions"
    ):
        load_and_expand_matrix(altered)


def test_final_profile_subset_matches_final_evaluator_rows():
    _, final_cases = load_and_expand_matrix(FINAL_EVALUATOR_MATRIX)
    final_by_row = {case["source_row"]: case for case in final_cases}
    profile = json.loads(FINAL_PROFILE_MATRIX.read_text(encoding="utf-8"))

    assert [case["source_row"] for case in profile["cases"]] == [1, 9, 10, 13]
    for profile_case in profile["cases"]:
        final_case = final_by_row[profile_case["source_row"]]
        assert profile_case["id"] == final_case["id"]
        assert profile_case["padding_ratio"] == final_case["padding_ratio"]
        assert {
            key: profile_case[key]
            for key in (
                "batch_size",
                "seq_len",
                "d_model",
                "num_heads",
                "ffn_dim",
                "num_layers",
                "causal",
            )
        } == final_case["config"]


def test_campaign4_profile_target_matches_final_evaluator_row11():
    _, final_cases = load_and_expand_matrix(FINAL_EVALUATOR_MATRIX)
    final_by_row = {case["source_row"]: case for case in final_cases}
    profile = json.loads(CAMPAIGN4_PROFILE_MATRIX.read_text(encoding="utf-8"))

    assert [case["source_row"] for case in profile["cases"]] == [11]
    profile_case = profile["cases"][0]
    final_case = final_by_row[11]
    assert profile_case["id"] == final_case["id"]
    assert profile_case["padding_ratio"] == final_case["padding_ratio"]
    assert {
        key: profile_case[key]
        for key in (
            "batch_size",
            "seq_len",
            "d_model",
            "num_heads",
            "ffn_dim",
            "num_layers",
            "causal",
        )
    } == final_case["config"]


def test_campaign5_profile_targets_match_frozen_final_and_heldout_cases():
    _, final_cases = load_and_expand_matrix(FINAL_EVALUATOR_MATRIX)
    final_by_row = {case["source_row"]: case for case in final_cases}
    heldout = json.loads(OFFICIAL_MATRIX.read_text(encoding="utf-8"))
    heldout_by_id = {case["id"]: case for case in heldout["cases"]}
    profile = json.loads(CAMPAIGN5_PROFILE_MATRIX.read_text(encoding="utf-8"))

    assert [case["source_row"] for case in profile["cases"]] == [6, 7, 8, None, None]
    for profile_case in profile["cases"]:
        source_row = profile_case["source_row"]
        if source_row is None:
            source_case = heldout_by_id[profile_case["id"]]
            expected = source_case
        else:
            source_case = final_by_row[source_row]
            assert profile_case["id"] == source_case["id"]
            expected = source_case["config"] | {
                "padding_ratio": source_case["padding_ratio"]
            }
        assert {
            key: profile_case[key]
            for key in (
                "batch_size",
                "seq_len",
                "d_model",
                "num_heads",
                "ffn_dim",
                "num_layers",
                "causal",
                "padding_ratio",
            )
        } == {key: expected[key] for key in (
            "batch_size",
            "seq_len",
            "d_model",
            "num_heads",
            "ffn_dim",
            "num_layers",
            "causal",
            "padding_ratio",
        )}


def test_validation_exit_accounting_is_fail_closed():
    passed = {"status": "PASS"}
    authorized_skip = {"status": "SKIPPED_RESOURCE", "skip_authorized": True}

    assert result_exit_code([passed]) == 0
    assert result_exit_code([passed, authorized_skip]) == 0
    assert result_exit_code([]) == 1
    assert result_exit_code([authorized_skip]) == 1
    assert result_exit_code([passed, {"status": "FAIL"}]) == 1
    assert result_exit_code([passed, {"status": "OOM"}]) == 1
    assert result_exit_code([passed, {"status": "ERROR"}]) == 1
    assert result_exit_code(
        [{"status": "SKIPPED_RESOURCE", "skip_authorized": False}, passed]
    ) == 1
    assert result_exit_code([passed, {"status": "UNKNOWN"}]) == 1


def test_validation_summary_includes_failed_accuracy_and_dispatch():
    passed = {
        "status": "PASS",
        "parsed": {
            "accuracy": {
                "total_elements": 100,
                "failed_elements": 0,
                "max_abs_error": 0.0005,
                "max_relative_error": 0.2,
            },
            "speedup_median": 1.25,
        },
        "attention_backend_counts": {"triton": 4, "sdpa": 0, "reference": 0},
    }
    failed = {
        "status": "FAIL",
        "parsed": {
            "accuracy": {
                "total_elements": 200,
                "failed_elements": 7,
                "max_abs_error": 0.002,
                "max_relative_error": 3.0,
            }
        },
        "attention_backend_counts": {"triton": 4, "sdpa": 0, "reference": 0},
    }
    skipped = {"status": "SKIPPED_RESOURCE", "skip_authorized": True}

    summary = summarize_results([passed, failed, skipped])

    assert summary["total_compared_elements"] == 300
    assert summary["total_failed_elements"] == 7
    assert summary["max_abs_error"] == 0.002
    assert summary["geometric_mean_speedup"] == 1.25
    assert summary["attention_backend_counts"] == {
        "triton": 8,
        "sdpa": 0,
        "reference": 0,
    }
