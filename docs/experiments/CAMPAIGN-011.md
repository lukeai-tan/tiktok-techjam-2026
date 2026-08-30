# Campaign 11: head-count residual-normalization survey

Status: complete and selected on 2026-08-30

## Objective and frozen starting point

Continue from selected Campaign 10 fingerprint
`f7ad2a86a68f95736241ddde992500073ee75738982af4a81c0c658cd64538d4`
on checkpoint `8c89d1d4170c58d16fb75d79f212e990565fba7d`. Campaign 10 integrated
only exact row-5 residual/LayerNorm fusion. Campaign 9 width-1024 row-8 and
exact row-13 arithmetic, Campaign 7 wide-head attention, prior long-row launch
variants, and Campaign 10's conditional row-7 fallback remain closed.

Campaign 11 profiles exact final rows 9, 10, and 1. Row 9 is the primary
hypothesis because it is the slowest remaining width-128/head-count final row by
speedup and uses a distinct `head_dim=128` attention path. Row 10 is a one-shot
fallback only if row 9 produces no winner and its profile clears the threshold.
Row 1 is observation-only to distinguish causal/head-count effects from a
non-causal width-128 control.

## Bounded hypotheses

### EXP-025: exact row-9 fused residual plus LayerNorm

At most one implementation may reuse the accepted fused forward behind an exact
row-9 runtime/model guard. Direct route, strict five-seed accuracy,
seed/scale/padding stress, runtime and mask neighbors, training/gradient/device/
dtype/layout fallback, Triton attention preservation, counterbalanced long
timing, profile, memory, and rows 5/6/11 non-regression are mandatory.

### EXP-026: exact row-10 fallback

Only if EXP-025 closes without a winner and row 10 has an approximately
five-percent actionable non-attention ceiling may one exact row-10 reuse be
screened. No attention launch, arithmetic, QKV policy, or broad predicate change
is allowed.

## Non-goals and stop rules

- Do not retry Campaign 9 row-8/row-13 work, Campaign 7 wide-head attention,
  Campaign 2 long-row tiles, Campaign 10 row-7 fallback, or broaden fusion.
- Do not replace vendor GEMMs, alter attention arithmetic, enable broad
  compilation, or change organizer sources/comparator/timing.
- Every test, profile, benchmark, review, and closure command uses
  `benchmarks/run_optimization_attempt.py`; failures and timeouts are retained.
- Correctness requires zero failed elements. Performance requires about five
  percent reproducible target/profile benefit with memory and drift disclosed.
- No commit, push, tag, release, branch creation, history rewrite, deployment,
  or public action is authorized.

## Outcome

EXP-025-I1 is selected. Its campaign-closing, pre-packaging fingerprint is
`9c326536ea27cfc619f01531152b2c82986d9dc3f4274691d3e8191bbb0804eb`.
Subsequent adapter packaging and canonical benchmark relocation produced the
Campaign 11 evidence fingerprint
`908a0d708cd8f70f44d5f14fda93d3cafb1cc18345f43914e715594cfa7b7ef9`;
the historical attempt records remain unchanged.
The source change adds only an exact row-9 predicate to the previously accepted
fused residual/LayerNorm route; three tests pin the selected route, unsupported
boundaries, and an 18-scenario numerical stress matrix. EXP-026 was deliberately
unrun because its row-10 fallback condition became false after EXP-025 won.

The final Council decision is
[`ACCEPT_EXP_025_ROW9_AND_SELECT_CAMPAIGN_11`](reviews/CAMPAIGN-011-FINAL-REVIEW.json).
It has no blocking findings. The decision explicitly relies on controlled
CUDA-event timing and the fused-subsystem mechanism rather than one noisy
top-level profiler snapshot.

## Baseline and target authorization

The fresh Campaign 10 final rebaseline passed all 13 executable rows plus the
exact authorized skip with zero of 938,885,120 failed elements at 1.926244x.
Thirty-forward profiles produced:

| Row | Model device time | Residual + native norm | Share | Attention route | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| 9 | 30,214.766 us | 5,769.588 us | 19.095% | 120 Triton | primary candidate authorized |
| 10 | 56,826.603 us | 6,025.865 us | 10.604% | 120 Triton | fallback only |
| 1 | 37,145.829 us | 5,763.501 us | 15.516% | 120 SDPA | observation only |

## Causal row-9 timing, accuracy, memory, and routing

All long target runs use five accuracy trials, ten warmups, six rounds of fifty
timed samples, and alternating baseline/optimized order.

| Run | Optimized median | Baseline-to-optimized speedup | Failed elements | Incremental peak | Attention calls |
| --- | ---: | ---: | ---: | ---: | --- |
| unchanged control A | 0.815776 ms | 1.051x | 0 | 29,360,128 B | 1,240 Triton |
| isolated candidate B | 0.717696 ms | 1.141609x | 0 | 29,360,128 B | 1,240 Triton |
| unchanged control C | 0.816160 ms | 1.000x | 0 | 29,360,128 B | 1,240 Triton |
| active integration | 0.717648 ms | 1.150046x | 0 | 29,360,128 B | 1,240 Triton |

The controls differ by 0.047%. The active result is 12.049% below their
0.815968 ms mean and 0.0067% below the isolated candidate. This clears the
approximately five-percent acceptance threshold without accuracy, memory, or
backend regression.

## Repeated profile mechanism

| Profile set | Mean model time | Mean residual/norm time | Fusion counts |
| --- | ---: | ---: | --- |
| two Campaign 10 controls | 32,578.855 us | 5,765.324 us | 0 fused; 270 native norms each |
| two active Campaign 11 profiles | 33,407.673 us | 3,357.389 us | 240 fused; 30 native norms each |

Mean subsystem time falls 41.766%. Mean top-level model time rises 2.544%
because one active snapshot is slow; this is retained as profiler variance and
does not override the tightly counterbalanced 300-sample timing gate. Both active
profiles preserve 120 Triton attention calls.

## Integrated evidence

| Gate | Result |
| --- | --- |
| direct + boundary | 2 passed |
| row-9 stress | all 18 seed/scale/padding scenarios passed |
| affected GPU suite | 40 passed; 14 upstream warnings |
| final primary / confirmation | 13/13 executable PASS + exact skip twice; zero failed; 1.977420x / 1.986499x |
| organizer default | 5/5 PASS; zero failed; 1.385x; 1,950 Triton calls |
| source-derived | 28/28 executable PASS + exact skip; zero of 459,776,000 failed; 1.206505x |
| held-out primary / confirmation | 7/7 PASS twice; 1.339847x / 1.386495x |
| held-out rechecks | 7/7 PASS twice; 1.384122x / 1.515376x; long-causal remains 1.198x-1.204x across all four runs |
| inherited row 5 long | 1.163168 ms optimized; 1.880066x; zero failed; 1,240 Triton |
| inherited row 6 long | 188.457397 ms optimized; 1.546330x; zero failed; 210 Triton + 210 reference |
| inherited row 11 long | 0.890672 ms optimized; 4.710116x; zero failed; 1,240 Triton |
| inherited mechanism profiles | exact row-5/row-6/row-8/row-11 fusion, projection, and backend counts preserved |
| artifact + notebook contracts | 19 passed |
| complete repository suite | 148 passed; 14 upstream warnings |

## Retained non-pass attempts

| Attempt | Status | Wall time | Cause and resolution |
| --- | --- | ---: | --- |
| `C11-REVIEW-001-exp025-council` | ERROR | 0.001202 s | child path resolved as `\\.venv`; corrected `001A` passed |
| `C11-INTEGRATE-002-source-parity` | FAIL | 0.042261 s | raw bytes differed only by LF/CRLF; exact fingerprint and normalized hashes matched in `002A` |
| `C11-INTEGRATE-007-row9-analysis` | FAIL | 0.044529 s | backend counts were read from the wrong JSON level; corrected `007A` passed |
| `C11-INTEGRATE-010-profile-analysis` | FAIL | 0.023557 s | Windows stripped double quotes from multiline `python -c`; single-quoted `010A` passed |
| `C11-INTEGRATE-025-pre-doc-full-suite` | FAIL | 20.757379 s | 139 passed and eight stale Campaign 10 artifact pointers failed; current evidence migrated and 148/148 passed |
| `C11-OPS-002-graph-wrapper-parser-failure` | FAIL | 0.022794 s | retrospectively mirrored the outer PowerShell argv parser failure; native argument-array graph wrappers passed |
| `C11-CLOSE-003-program-accounting` | FAIL | 0.025686 s | the compact cross-program counter had mismatched brackets; multiline rework followed |
| `C11-CLOSE-003A-program-accounting` | FAIL | 0.024082 s | Windows removed double quotes from the multiline child; single-quoted `003B` passed |

No timeout occurred. Two PowerShell parser errors also occurred before their
intended wrappers could start: one while constructing the first multiline
profile analysis and one while batching graph wrappers. They created no child
process or direct attempt JSON. The second is explicitly mirrored by the
retained `C11-OPS-002` failure; both are disclosed here rather than represented
as measured child commands.

## Accounting snapshot and attempt ledger

`C11-CLEANUP-001-program-accounting` computes the terminal Campaign 11
checkpoint through `C11-CLOSE-007`: **75 attempts, 67 PASS, 8 non-pass, 0
timeouts, and 527.126837 seconds** of child-command wall time. Cleanup and later
validation records append after that exact checkpoint and remain individually
measurable in `attempts/`.

| Group | Attempts in checkpoint | PASS | Non-pass | Wall time | Disposition |
| --- | ---: | ---: | ---: | ---: | --- |
| `C11-PREFLIGHT-*` | 4 | 4 | 0 | 7.855000 s | complete |
| `C11-GRAPH-*` | 6 | 6 | 0 | 1.177954 s | opening and closure complete; graph strict-green |
| `C11-BASE-*` | 8 | 8 | 0 | 70.835205 s | complete |
| `C11-EXP-025-*` | 9 | 9 | 0 | 86.760079 s | accepted |
| `C11-EXP-026-*` | 0 | 0 | 0 | 0 s | deliberately unrun |
| `C11-REVIEW-*` | 3 | 2 | 1 | 0.076423 s | accepted after retained launcher correction |
| `C11-INTEGRATE-*` | 35 | 31 | 4 | 339.797092 s | complete after retained corrections |
| `C11-OPS-*` | 1 | 0 | 1 | 0.022794 s | retained orchestration failure |
| `C11-CLOSE-*` through `007` | 9 | 7 | 2 | 20.602289 s | accounting, workflow, artifacts, full suite, and diff hygiene complete |

Attempt JSON is authoritative for command wall time, accuracy/error totals,
latency samples, memory, backend/profiler counts, environment, fingerprints,
and dispositions. Post-campaign cleanup removed the unexecuted Campaign 12
scaffold, twelve obsolete worktrees, generated caches, and three unreferenced
smoke outputs while preserving all immutable evidence and the one EXP-025
worktree still needed by raw-artifact paths. No commit, push, tag, branch
creation, release, deployment, or public action was performed or authorized.
