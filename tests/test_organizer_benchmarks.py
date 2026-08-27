from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.run_organizer_torch import (
    install_submission,
    load_organizer_benchmark,
    verify_organizer_download,
)
from benchmarks.run_organizer_validation import (
    load_and_expand_matrix,
    organizer_arguments,
    result_exit_code,
)
from torch_transformer_benchmark import UserOptimizedTransformer


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "benchmarks" / "reference" / "organizer_downloads.json"
ORGANIZER_TORCH = ROOT / "benchmarks" / "torch_transformer_benchmark.py"
ORGANIZER_TENSORFLOW = ROOT / "benchmarks" / "tensorflow_transformer_benchmark.py"
VALIDATION_MATRIX = ROOT / "benchmarks" / "organizer_validation_matrix.json"
SUBMISSION_TORCH = ROOT / "torch_transformer_benchmark.py"
ROOT_TENSORFLOW = ROOT / "tensorflow_transformer_benchmark.py"

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

    assert _sha256(ORGANIZER_TORCH) == artifacts["pytorch"]["sha256"]
    assert ORGANIZER_TORCH.stat().st_size == artifacts["pytorch"]["size_bytes"]
    assert _sha256(ORGANIZER_TENSORFLOW) == artifacts["tensorflow"]["sha256"]
    assert ORGANIZER_TENSORFLOW.stat().st_size == artifacts["tensorflow"]["size_bytes"]
    assert verify_organizer_download() == artifacts["pytorch"]["sha256"]


def test_submission_preserves_organizer_pytorch_baseline_contract():
    organizer = _top_level_definitions(ORGANIZER_TORCH)
    submission = _top_level_definitions(SUBMISSION_TORCH)

    for name in PROTECTED_TORCH_DEFINITIONS:
        assert organizer[name] == submission[name], name


def test_tensorflow_copy_matches_supplied_download_after_eol_normalization():
    def normalize(value: bytes) -> bytes:
        return value.replace(b"\r\n", b"\n")

    assert normalize(ORGANIZER_TENSORFLOW.read_bytes()) == normalize(
        ROOT_TENSORFLOW.read_bytes()
    )


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

    assert {case["dtype"] for case in pytorch_cases} == {
        "float32",
        "float16",
        "bfloat16",
    }
    assert {case["config"]["causal"] for case in pytorch_cases} == {False, True}
    assert {case["padding_ratio"] for case in pytorch_cases} == {0.0, 0.3}
    assert matrix["defaults"]["atol"] == 0.001
    assert matrix["defaults"]["rtol"] == 0.01

    causal_case = next(case for case in pytorch_cases if case["config"]["causal"])
    arguments = organizer_arguments(causal_case, matrix["defaults"], "cuda")
    assert "--causal" in arguments
    assert arguments[arguments.index("--accuracy-trials") + 1] == "5"
    assert arguments[arguments.index("--atol") + 1] == "0.001"
    assert arguments[arguments.index("--rtol") + 1] == "0.01"


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
