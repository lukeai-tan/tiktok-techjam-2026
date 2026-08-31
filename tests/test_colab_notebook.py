import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "colab_benchmark.ipynb"
EXPECTED_IMPLEMENTATION_SHA256 = (
    "a186b679885e9e787b3deba0ad710855ae4c2486ae491b53e4e64bfa13e7f9cf"
)


def test_colab_notebook_is_clean_runnable_and_targets_current_branch() -> None:
    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    notebook = json.loads(notebook_text)

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert len(notebook["cells"]) == 19

    code_cells = [
        cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
    ]
    assert len(code_cells) == 9
    for index, cell in enumerate(code_cells, start=1):
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
        compile("".join(cell["source"]), f"colab-code-cell-{index}", "exec")

    required_markers = (
        "repo_ref = 'main'",
        f"expected_implementation_sha256 = '{EXPECTED_IMPLEMENTATION_SHA256}'",
        "implementation_fingerprint()",
        "benchmarks/final_evaluator_shapes.json",
        "benchmarks/campaign10_profile_shapes.json",
        "benchmarks/campaign11_profile_shapes.json",
        "colab-final-row5-trace.json",
        "colab-final-row9-trace.json",
        "colab-final-row6-trace.json",
        "colab-final-row7-trace.json",
        "colab-final-row8-trace.json",
        "colab-source-derived.json",
        "colab-final-row11-trace.json",
        "colab-heldout-trace.json",
    )
    assert all(marker in notebook_text for marker in required_markers)

    assert "TIKTOK_TECHJAM_GITHUB_TOKEN" in notebook_text
    assert "GIT_ASKPASS" in notebook_text
    assert "x-access-token@" not in notebook_text
    assert "github_pat_" not in notebook_text
