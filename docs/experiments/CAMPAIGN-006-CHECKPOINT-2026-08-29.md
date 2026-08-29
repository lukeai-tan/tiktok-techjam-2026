# Campaign 6 checkpoint — 2026-08-29

Status: paused after integration and current-fingerprint benchmark validation;
documentation curation and final closure remain pending.

Historical snapshot note: this pause state was subsequently committed and
pushed as `8c89d1d4170c58d16fb75d79f212e990565fba7d` (`Campaign 6 Checkpoint`).
The resumed closure is recorded in `CAMPAIGN-006.md`; its later documentation,
test, review, workflow, and graph changes remain local until separately
authorized Git packaging.

## Exact resume state

| Field | Value |
| --- | --- |
| Branch | `feat/jared-attempt` |
| HEAD | `7f4fcba4ffb891cb876fa9ec27afa2395b99c77a` |
| Working tree | intentionally dirty; Campaign 5 and Campaign 6 evidence is not committed |
| Accepted implementation fingerprint | `54df021e77cfed86011bae0b41e07c3f42842b54e19c139aa925eb2f0d449ff4` |
| Accepted candidate | `EXP-015-I1R` exact-width packed-QKV projection |
| Independent review | `APPROVE_LOCAL_ACCEPTANCE` for the surgical source hunk and two tests only |
| Git/public actions | none; no commit, push, tag, release, or submission mutation was performed |

The retained code preserves the previously measured packed-QKV envelope for
`d_model <= 512` and adds only exact `d_model == 1024`. Widths 513 through 1023
remain on the previous three-projection path. Two tests prove the positive
width-1024 route and the negative width-768 boundary without replacing any
Campaign 5 tests.

## Logged work at this checkpoint

Campaign 6 currently contains 108 immutable attempt records:

- 106 command-level PASS records;
- two retained FAIL records;
- zero timeouts; and
- 840.5057717985474 seconds of recorded child-command wall time.

The two failures are expected and resolved or explicitly pending:

1. `C6-PREFLIGHT-001-workflow.json` caught invalid conditional PRD-impact
   labels. `C6-PREFLIGHT-002-workflow-rework.json` passed after correction.
2. `C6-INTEGRATE-016-full-tests-predoc.json` passed 118 tests and failed five
   evidence-curation tests because `tests/test_result_artifacts.py` still points
   to Campaign 5 fingerprints and result filenames. This is not an
   implementation failure; updating those pointers and rerunning the full suite
   is the first closure task after resumption.

The first integration identity capture, `C6-INTEGRATE-001-fingerprint.json`,
also exposed a comment-only source-hash mismatch. It is retained as a PASSing
command with a rejected provenance decision. The byte-identical rework in
`C6-INTEGRATE-002-fingerprint-rework.json` matches the reviewed fingerprint.

## Current-fingerprint results

All rows below bind to implementation fingerprint
`54df021e77cfed86011bae0b41e07c3f42842b54e19c139aa925eb2f0d449ff4`.

| Gate | Correctness | Performance |
| --- | --- | --- |
| Final primary | 13/13 executable PASS plus the exact authorized row-14 resource skip; 0/938,885,120 failed | 1.872916x geomean |
| Final confirmation | same correctness and backend totals; 0/938,885,120 failed | 1.863721x geomean; 0.491% from primary |
| Organizer default | 5/5 trials PASS; 0/2,621,440 failed; 1,950 Triton attention calls | 1.338x |
| Held-out five-seed primary | 7/7 PASS; 0/13,117,440 failed | 1.365499x geomean |
| Held-out five-seed confirmation | 7/7 PASS; 0/13,117,440 failed | 1.380821x geomean |
| Source-derived matrix | 28/28 executable PASS plus the exact authorized resource skip; 0/459,776,000 failed | 1.244108x geomean |
| Focused regression suite | 94 passed, 14 upstream deprecation warnings | n/a |

The fresh Campaign 6 opening control was 1.838500x, so the integrated primary
matrix is 1.872% higher in this campaign's measurement window. The two final
runs are the current comparison pair; older Campaign 5 geomeans remain useful
historical observations but are not substituted for same-window evidence.

## Accepted row-8 evidence

Three 300-sample candidate runs were exactly correct and produced internal
speedups of 1.022022x, 1.030071x, and 1.023827x. Two contemporaneous unchanged
controls produced 0.981690x and 0.993542x. The final integrated profiler versus
the same-window Campaign 5 control recorded:

| Metric | Control | Integrated | Change |
| --- | ---: | ---: | ---: |
| `aten::addmm` calls across ten forwards | 240 | 160 | -33.33% |
| `aten::addmm` device time | 106,065.035 us | 94,048.315 us | -11.33% |
| optimized-model device time | 150,050.615 us | 138,182.163 us | -7.91% |

Exact organizer row 8 passed with zero failed elements and zero maximum absolute
error. The distinct width-1024 held-out neighbor also passed with zero failed
elements, 0.000293195 maximum absolute error, and expected Triton execution.

## Closed candidate loops

- `EXP-013` row-6 launch geometry: five correct variants rejected because raw
  timing or profiler evidence was worse than the accepted configuration.
- `EXP-014` row-7 padded-width geometry: five correct variants plateaued or
  regressed under long-sample and profiler checks.
- `EXP-016` row-11 padded-width geometry: three correct new-axis variants were
  6.42% to 19.77% slower than the long control.
- `EXP-015-I2` two-plus-one projection grouping: rejected at 0.988851x and
  14.132976 ms optimized versus 13.975408 ms baseline.
- Original `EXP-015-I1`: independently rejected because `<=1024` widened policy
  to unmeasured widths. `I1R` fixed that boundary and was independently approved.

No rejected launch candidate remains in the integration source.

## Cleanup completed

The four detached Campaign 6 candidate worktrees for `EXP-013`, `EXP-014`,
`EXP-015`, and `EXP-016` were verified as exact cleanup targets and removed
after the accepted source and tests were confirmed on the active branch. Older
campaign worktrees were intentionally left untouched. The active branch still
contains 19 tracked modifications and 318 untracked evidence/documentation
paths; these are the preserved Campaign 5 and Campaign 6 checkpoint, not a
clean-commit claim.

The repository graph now contains the pause handoff event, was rebuilt with 57
file notes and 84 events, and passes strict validation with zero errors and zero
warnings.

## Resume sequence

Continue from this document and do not rerun completed benchmark gates unless
the implementation fingerprint, harness, framework, driver, or hardware changes.

1. Update `tests/test_result_artifacts.py` from Campaign 5 to the Campaign 6
   integrated attempt/result set and use evidence-backed current thresholds.
2. Curate Campaign 6 into `README.md`, `docs/REQUIREMENTS.md`,
   `docs/OPTIMIZATION_LOOP_PLAN.md`, `docs/KERNEL_DESIGN.md`,
   `docs/TECH_REPORT.md`, `docs/TRACK3_COMPLIANCE.md`,
   `docs/results/README.md`, `docs/experiments/OPTIMIZATION_HISTORY.md`, and
   `docs/experiments/CAMPAIGN_RUN_THROUGH.md` without deleting prior campaigns.
3. Expand `docs/experiments/CAMPAIGN-006.md` from its planning ledger into the
   final 105-attempt campaign accounting and update candidate decisions.
4. Run the full pytest suite through the immutable attempt logger. The expected
   result is all 123 tests passing with only the 14 upstream warnings.
5. Run a scoped diff check, finalize and validate the workflow, append graph
   closure events, rebuild and strictly validate the repository graph, then
   recompute attempt totals.
6. Request a final independent matrix/evidence review.
7. Commit or push only after a separate explicit user instruction.

## Authoritative evidence pointers

- Workflow: `docs/experiments/CAMPAIGN-006-WORKFLOW.json`
- Campaign ledger: `docs/experiments/CAMPAIGN-006.md`
- Initial rejected review: `docs/experiments/reviews/CAMPAIGN-006-CANDIDATE-REVIEW.json`
- Approved rework review: `docs/experiments/reviews/CAMPAIGN-006-CANDIDATE-REVIEW-I1R.json`
- Primary final matrix: `docs/results/rtx-5070-ti-2026-08-29-c6-integrated-final.json`
- Confirmation final matrix: `docs/results/rtx-5070-ti-2026-08-29-c6-integrated-final-confirmation.json`
- Organizer default: `docs/results/rtx-5070-ti-2026-08-29-c6-integrated-organizer-default.json`
- Held-out primary and confirmation: `docs/results/rtx-5070-ti-2026-08-29-c6-integrated-heldout-5seed.json` and `docs/results/rtx-5070-ti-2026-08-29-c6-integrated-heldout-5seed-confirmation.json`
- Source-derived matrix: `docs/results/rtx-5070-ti-2026-08-29-c6-integrated-source-derived.json`
- Integrated row-8 profile: `docs/results/rtx-5070-ti-2026-08-29-c6-integrated-row08-profile.json`
- Attempt records: `docs/experiments/attempts/C6-*.json`
