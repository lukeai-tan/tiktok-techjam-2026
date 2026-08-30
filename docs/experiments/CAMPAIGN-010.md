# Campaign 10: short-row residual-normalization survey

Status: complete on 2026-08-29; EXP-023-I1 selected

## Objective and frozen starting point

Continue from selected fingerprint
`325a1e5cad70f85390ddbea438f04b60a4b0f40300826aba3991520f5b97079b`
on checkpoint `8c89d1d4170c58d16fb75d79f212e990565fba7d`. Campaign 9 integrated
nothing: its row-8 width-1024 variants were slower and its exact row-13 fallback
failed one of 41,943,040 elements. Those routes remain closed.

Campaign 10 profiles exact final rows 5, 7, and 12 to find a remaining
launch-bound residual/normalization surface. Row 5 is the primary hypothesis:
it uses the already proven width-128 arithmetic at a medium 16,384-token batch,
without row 6's memory pressure or row 13's long-sequence numerical path. Row 7
is a one-shot fallback only if its fresh width-32 profile clears the threshold
and row 5 produces no winner. Row 12 is observation-only in this campaign.

## Outcome

EXP-023-I1 is accepted and integrated. The selected schema-2 implementation
fingerprint is
`f7ad2a86a68f95736241ddde992500073ee75738982af4a81c0c658cd64538d4`.
The change reuses the proven residual-add plus LayerNorm Triton forward only for
exact final row 5 (`B=128,S=128,d_model=128,heads=4,ffn_dim=128,layers=4,
causal=true`) under the existing eval-mode eager CUDA float32 safety boundary.
Three row-5 route/boundary tests were added; neighboring static/runtime shapes,
masks, layouts, dtypes, devices, training, gradients, and compiled execution
remain unfused.

Fresh profiling established a 29.07% row-5 residual/normalization ceiling.
Two unchanged 300-sample controls averaged 1.325096 ms optimized median; the
retained candidate measured 1.171584 ms (-11.58%) and improved normalized
speedup 14.79% with the same 58,720,256-byte incremental peak. The integrated
300-sample gate measured 1.162976 ms and 2.001995x. Its 30-forward profile
replaces 240 residual adds and 240 of 270 native norms with 240 fused launches,
reducing integrated subsystem/model device time 40.63%/11.96% against the fresh
baseline while preserving 120 Triton attention calls.

Both final matrices pass 13/13 executable rows plus the exact skip with
0/938,885,120 failed elements at 1.926716x and 1.939005x. Organizer default is
1.361x; source-derived is 28/28 executable PASS at 1.214174x; held-out is 7/7
PASS at 1.394370x and 1.378630x; four complete held-out runs keep long-causal
at 1.200x-1.203x. The current full suite passes 144/144 with 14 upstream
deprecation warnings.

EXP-024 was deliberately unrun because it was a fallback only if EXP-023
produced no winner. Campaign 9's width-1024 row-8 and row-13 routes remain
closed.

## Bounded hypotheses

### EXP-023: exact row-5 fused residual plus LayerNorm

At most one implementation may reuse the accepted fused forward behind an exact
row-5 runtime/model guard. Direct route, strict five-seed model accuracy,
seed/scale/padding stress, runtime and mask neighbors, training/gradient/device/
dtype/layout fallback, Triton attention preservation, long timing, profile,
memory, and rows 6/11 non-regression are mandatory.

### EXP-024: exact row-7 width-32 fallback

Only if EXP-023 closes without a winner and the fresh row-7 profile shows an
approximately five-percent ceiling may one exact width-32 implementation be
screened. Its one-reference/three-Triton attention ordering is immutable. No
launch rework is allowed in this campaign.

## Non-goals and stop rules

- Do not retry Campaign 9 row-8 or row-13 fusion, Campaign 7 row-8 attention,
  earlier launch variants, or packed-QKV guard expansion.
- Do not replace vendor GEMMs, change organizer sources/comparator/timing,
  enable broad compilation, or broaden fusion beyond one exact measured row.
- Every test, profile, benchmark, review, and closure command uses
  `benchmarks/run_optimization_attempt.py`; failures and timeouts are retained.
- Correctness requires zero failed elements. Performance requires about five
  percent reproducible target/profile benefit or material launch/memory value.
- No commit, push, tag, release, branch creation, history rewrite, or public
  action is authorized.

## Attempt ledger

| Group | Purpose | Status |
| --- | --- | --- |
| `C10-PREFLIGHT-*` | workflow, fingerprint, contract, environment, graph, isolation | complete |
| `C10-BASE-*` | final baseline, row-5/row-7/row-12 profiles, long controls | complete; `BASE-005` semantic analysis defect retained and corrected by `005A` |
| `C10-EXP-023-*` | exact-row-5 candidate | accepted after direct, stress, timing, memory, profile, affected-suite, final, and review gates |
| `C10-EXP-024-*` | conditional exact-row-7 fallback | deliberately unrun; fallback condition false |
| `C10-REVIEW-*` | correctness, performance, boundary, memory, maintenance | candidate and final Council reviews approved with no blockers |
| `C10-INTEGRATE-*` | winner transplant and complete rebaseline | complete through `C10-INTEGRATE-021`; 43 PASS / 2 retained FAIL overall through this checkpoint |
| `C10-CLOSE-*` | docs, tests, workflow, graph, Council, tree | complete; final workflow and graph validate with zero errors/warnings |

Attempt JSON is authoritative for commands, wall time, accuracy, latency raw
samples, memory, backend/profiler counts, environment, fingerprints, and
dispositions.

## Retained failures and rework

- `C10-EXP-023-I1-final` completed all 13 executable rows successfully, then
  failed closed because its active-worktree result path was outside the detached
  candidate root and could not be serialized as repository-relative evidence.
  `C10-EXP-023-I1-final-rework` used a candidate-local artifact and passed.
- `C10-INTEGRATE-015-pre-doc-full-suite` reported 135 passing tests and seven
  stale Campaign 8 artifact-fingerprint failures. It contained no implementation
  failure; the canonical evidence pointers and notebook pin were migrated to
  Campaign 10, after which the artifact contract passed 18/18 and the complete
  suite passed 144/144.
- `C10-BASE-005-profile-analysis` exited zero but used a misspelled profile key,
  producing a semantically false share. The immutable record remains; corrected
  analysis is `C10-BASE-005A-profile-analysis-rework`.

Through `C10-INTEGRATE-021`, Campaign 10 contains **45 immutable attempts**:
**43 child PASS**, **2 retained child FAIL**, **0 timeouts**, and
**518.917727 seconds** of measured child-command wall time. Terminal Council,
documentation, graph, workflow, and diff-hygiene records append after this
preterminal accounting checkpoint; each stores its own runtime and status.
