import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "docs" / "results"
ATTEMPT_ROOT = REPO_ROOT / "docs" / "experiments" / "attempts"
COMPARISON_PATH = (
    REPO_ROOT / "docs" / "experiments" / "BRANCH_IMPLEMENTATION_COMPARISON.md"
)

FLAGSHIP_FINGERPRINT = (
    "de768f1ff9ddee54a9ad83a67f3e1f205044c0ad5c723fc3bb4881093c97f611"
)
RAW_CANDIDATE_FINGERPRINT = (
    "20c3c74144c0b1b6b095e82f2af51f53b51f232ffd6fc891abf2442bcd689354"
)
ADAPTED_CANDIDATE_FINGERPRINT = (
    "3ffd8505e2b5ace13a854d4273fa3b7b7631d1c62f12dfbb9ee18569d939dd53"
)

RUNS = {
    "BC1-CANDIDATE-001-organizer-default.json": (
        "rtx-5070-ti-2026-08-28-branchfix-candidate-organizer-default.json",
        RAW_CANDIDATE_FINGERPRINT,
        "RECORDED",
    ),
    "BC1-CANDIDATE-002-final.json": (
        "rtx-5070-ti-2026-08-28-branchfix-candidate-final.json",
        RAW_CANDIDATE_FINGERPRINT,
        "FAIL",
    ),
    "BC1-CANDIDATE-003-source-derived.json": (
        "rtx-5070-ti-2026-08-28-branchfix-candidate-source-derived.json",
        RAW_CANDIDATE_FINGERPRINT,
        "FAIL",
    ),
    "BC1-CANDIDATE-004-adapted-final.json": (
        "rtx-5070-ti-2026-08-28-branchfix-candidate-adapted-final.json",
        ADAPTED_CANDIDATE_FINGERPRINT,
        "FAIL",
    ),
    "BC1-CANDIDATE-005-adapted-source-derived.json": (
        "rtx-5070-ti-2026-08-28-branchfix-candidate-adapted-source-derived.json",
        ADAPTED_CANDIDATE_FINGERPRINT,
        "FAIL",
    ),
    "BC1-FLAGSHIP-001-organizer-default.json": (
        "rtx-5070-ti-2026-08-28-branchfix-flagship-organizer-default.json",
        FLAGSHIP_FINGERPRINT,
        "RECORDED",
    ),
    "BC1-FLAGSHIP-002-final.json": (
        "rtx-5070-ti-2026-08-28-branchfix-flagship-final.json",
        FLAGSHIP_FINGERPRINT,
        "RECORDED",
    ),
    "BC1-FLAGSHIP-003-source-derived.json": (
        "rtx-5070-ti-2026-08-28-branchfix-flagship-source-derived.json",
        FLAGSHIP_FINGERPRINT,
        "RECORDED",
    ),
}

CLOSURE_ATTEMPTS = (
    "BC1-CLOSE-001-colab-notebook.json",
    "BC1-CLOSE-002-full-tests.json",
    "BC1-CLOSE-003-candidate-provenance.json",
    "BC1-CLOSE-004-graph-rebuild.json",
    "BC1-CLOSE-005-graph-validate.json",
    "BC1-CLOSE-006-artifact-validation.json",
    "BC1-CLOSE-007-final-tests.json",
    "BC1-CLOSE-008-final-graph-rebuild.json",
    "BC1-CLOSE-009-final-graph-validate.json",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_branch_comparison_artifacts_are_complete_and_bound_to_attempts() -> None:
    comparison_text = COMPARISON_PATH.read_text(encoding="utf-8")

    relative_links = {
        unquote(target.split("#", maxsplit=1)[0])
        for target in re.findall(r"\]\(([^)]+)\)", comparison_text)
        if not re.match(r"^(?:https?|mailto):", target)
    }
    for target in relative_links:
        resolved = (COMPARISON_PATH.parent / target).resolve()
        assert resolved.is_relative_to(REPO_ROOT.resolve())
        assert resolved.exists(), target

    for attempt_name, (result_name, fingerprint, record_status) in RUNS.items():
        attempt_path = ATTEMPT_ROOT / attempt_name
        result_path = RESULT_ROOT / result_name
        attempt = _load(attempt_path)
        result = _load(result_path)

        assert attempt["record_status"] == record_status
        assert attempt["environment_before"]["git"]["implementation_sha256"] == fingerprint
        assert result["environment"]["git"]["implementation_sha256"] == fingerprint
        assert attempt["result_artifact"]["path"] == result_path.relative_to(
            REPO_ROOT
        ).as_posix()
        assert attempt["result_artifact"]["sha256"] == _sha256(result_path)
        assert attempt_name in comparison_text
        assert result_name in comparison_text

    for attempt_name in CLOSURE_ATTEMPTS:
        attempt = _load(ATTEMPT_ROOT / attempt_name)
        assert attempt["record_status"] == "RECORDED"
        assert attempt["execution"]["status"] == "PASS"
        assert attempt_name in comparison_text

    flagship_default = _load(
        RESULT_ROOT
        / "rtx-5070-ti-2026-08-28-branchfix-flagship-organizer-default.json"
    )
    candidate_default = _load(
        RESULT_ROOT
        / "rtx-5070-ti-2026-08-28-branchfix-candidate-organizer-default.json"
    )
    assert flagship_default["parsed"]["accuracy"]["failed_elements"] == 0
    assert candidate_default["parsed"]["accuracy"]["failed_elements"] == 0
    assert flagship_default["parsed"]["optimized"]["median_ms"] == 1.3587
    assert candidate_default["parsed"]["optimized"]["median_ms"] == 1.567

    flagship_final = _load(
        RESULT_ROOT / "rtx-5070-ti-2026-08-28-branchfix-flagship-final.json"
    )["summary"]
    raw_candidate_final = _load(
        RESULT_ROOT / "rtx-5070-ti-2026-08-28-branchfix-candidate-final.json"
    )["summary"]
    adapted_candidate_final = _load(
        RESULT_ROOT
        / "rtx-5070-ti-2026-08-28-branchfix-candidate-adapted-final.json"
    )["summary"]
    assert (flagship_final["passed"], flagship_final["total_failed_elements"]) == (
        13,
        0,
    )
    assert raw_candidate_final["counts"] == {
        "ERROR": 9,
        "FAIL": 4,
        "OOM": 0,
        "PASS": 0,
        "SKIPPED_RESOURCE": 1,
    }
    assert (
        adapted_candidate_final["passed"],
        adapted_candidate_final["counts"]["FAIL"],
        adapted_candidate_final["total_failed_elements"],
    ) == (9, 4, 24)

    flagship_source = _load(
        RESULT_ROOT
        / "rtx-5070-ti-2026-08-28-branchfix-flagship-source-derived.json"
    )["summary"]
    raw_candidate_source = _load(
        RESULT_ROOT
        / "rtx-5070-ti-2026-08-28-branchfix-candidate-source-derived.json"
    )["summary"]
    adapted_candidate_source = _load(
        RESULT_ROOT
        / "rtx-5070-ti-2026-08-28-branchfix-candidate-adapted-source-derived.json"
    )["summary"]
    assert (flagship_source["passed"], flagship_source["total_failed_elements"]) == (
        28,
        0,
    )
    assert raw_candidate_source["counts"] == {
        "ERROR": 15,
        "FAIL": 13,
        "OOM": 0,
        "PASS": 0,
        "SKIPPED_RESOURCE": 1,
    }
    assert (
        adapted_candidate_source["passed"],
        adapted_candidate_source["counts"]["FAIL"],
        adapted_candidate_source["total_failed_elements"],
    ) == (15, 13, 1_008_926)

    provenance = _load(ATTEMPT_ROOT / "BC1-CLOSE-003-candidate-provenance.json")
    assert provenance["record_status"] == "RECORDED"
    assert provenance["environment_before"]["git"]["implementation_sha256"] == (
        RAW_CANDIDATE_FINGERPRINT
    )
    assert "candidate_paths=3 blob_match=PASS" in provenance["execution"]["stdout"]
