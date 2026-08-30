# CAMPAIGN-005: Extended residual-bottleneck optimization round

Status: independently approved for local acceptance; no commit, push, release, or public action authorized

Starting commit: `3be02a3ebe562a89ca360b196057a2762b425ec4`

Starting implementation SHA-256:
`de768f1ff9ddee54a9ad83a67f3e1f205044c0ad5c723fc3bb4881093c97f611`

Selected implementation SHA-256:
`9159177a21d039366ed4d3aef431b4b14d3bcef26d5eeaab0808efa739294029`

## Outcome

Campaign 5 accepted three exact routing changes and no new approximation:

1. final row 7 keeps layer 0 on exact reference attention and runs layers 1-3
   through the existing padded-width Triton kernel;
2. final row 6 keeps layers 0-1 on reference and runs layers 2-3 through the
   existing Triton kernel; and
3. the exact two-layer, 512-token held-out long-causal shape uses PyTorch SDPA,
   with and without prefix padding.

The fresh starting implementation measured 1.776534x final-matrix geomean. The
integrated primary measured **1.911947x** (+7.62%) and the complete confirmation
measured **1.995117x**. Each run passed all 13 executable final rows, plus the
exact authorized non-pass resource skip, with **0/938,885,120 failed elements**.
Backend counts changed from Triton 1,120 / reference 336 to Triton 1,260 /
reference 196, with no final-matrix SDPA calls.

Two complete five-seed held-out matrices passed 7/7 with zero failed elements
at **1.447477x** and **1.449715x**. The previous long-causal regressions were
removed: primary non-padded/padded speedups are **1.247x/1.280x**, and
confirmation is **1.216x/1.423x**.

## Frozen constraints

- Organizer sources, final row order, comparator, tolerances, timing policy,
  and exact resource-skip authorization were protected.
- Required correctness was zero failed elements under `atol=0.001 OR
  rtol=0.01`; a nonzero command, unexpected fallback, OOM, error, or
  unauthorized skip rejected the gate.
- Only three profile-authorized hypotheses were opened. Each was bounded to
  five implementation/review cycles.
- Every command was wrapped by `benchmarks/run_optimization_attempt.py`.
  Failed, superseded, and slower artifacts were retained without editing.
- Promotion required about 5% aggregate improvement or a clear profiler-backed
  kernel/held-out benefit, followed by complete revalidation.

## Fresh baseline and bottleneck analysis

| Evidence | Starting result | Interpretation |
| --- | --- | --- |
| Complete final matrix | 13/13 executable PASS; 0/938,885,120 failed; 1.776534x | Clean Campaign 4 comparison point |
| Final row 6 | 1.060x; 112 reference calls | Large causal batch was an accuracy-sensitive attention target |
| Final row 7 | 1.050x; 112 reference calls | Existing padded-width width-8 kernel had potential if layer drift could be bounded |
| Final row 8 | 1.007x; 112 reference calls | Profile showed `aten::addmm` at 100,828.802 us, about 71% of 142,135.631 us model time |
| Held-out long-causal | 0.798x | Existing Triton route was slower than original |
| Held-out long-causal-padding | 0.878x | Same residual with prefix padding |

The row-6 reference profile measured 2,790,718.259 us across ten model forwards;
row 7 measured 19,390.479 us. Both exposed materialized reference attention
operations. The long-causal profiles proved 20 Triton launches and about 5.0-5.2
ms of `_attention_fwd` device time across ten forwards.

## Backend screens and rejected full routes

| Attempt | Route | Accuracy | Useful measurement | Decision |
| --- | --- | --- | --- | --- |
| `C5-OBS-006` | row 7, full Triton | FAIL: 1/1,310,720 elements; max abs 0.00131607 | Existing kernel executed | Reject full route |
| `C5-OBS-007` | row 7, full SDPA | FAIL: 1/1,310,720 | Same strict boundary miss | Reject full route |
| `C5-OBS-008` | row 6, full Triton | FAIL: 21/819,200,000; max abs 0.00137609 | All five trials affected | Reject full route |
| `C5-OBS-009` | row 6, full SDPA | FAIL: 21/819,200,000 | Same accumulated model drift | Reject full route |
| `C5-OBS-010` | long-causal, SDPA | PASS; zero failures | 1.199x | Authorize exact held-out experiment |
| `C5-OBS-011` | long-causal-padding, SDPA | PASS; zero failures | 1.230x | Authorize padded exact route |
| `C5-OBS-012` | row 8, SDPA | FAIL: 1/41,943,040; max abs 0.00111229 | Vendor GEMM dominated profile | Reject and stop row-8 work |

This screen is why the final implementation is layer- and shape-specific. It
does not pretend that a backend which is usually close is exact enough for the
organizer's deep-stack comparator.

## EXP-010: final row 7

| Iteration | Layer route | Accuracy and stress | Target speed | Decision |
| --- | --- | --- | --- | --- |
| I1 | layers 0-2 Triton; layer 3 reference | FAIL: 1/1,310,720 | Not promoted | Reject |
| I2 | layers 0-1 Triton; layers 2-3 reference | PASS; 18 stress scenarios PASS | 1.276x screen; 1.280x/1.348x confirmations | Correct, superseded |
| I3 | layer 0 reference; layers 1-3 Triton | PASS; max abs 0.000942588; 18 stress scenarios PASS | 1.484x target; 1.492x/1.596x confirmations | Keep |

I3's profile proved 30 Triton and 10 reference attention calls. `_attention_fwd`
used 1,185.985 us and ten-forward model time fell from 19,390.479 us to
12,868.043 us (**-33.64%**). A counterbalanced unchanged-row-6 control showed
the apparent full-matrix movement was system timing noise rather than a row-6
regression. The first I3 focused-test command contained an invalid hyphenated
test selector; its nonzero record was retained, the selector was corrected, and
the rerun passed.

## EXP-011: final row 6

| Iteration | Layer route | Accuracy | Target speed | Decision |
| --- | --- | --- | --- | --- |
| I1 | layer 0 reference; layers 1-3 Triton | FAIL: 1/819,200,000 | Not promoted | Reject |
| I2 | layers 0-1 reference; layers 2-3 Triton | PASS; max abs 0.000898957 | 1.549x screen; 1.488x/1.495x confirmations | Keep |

I2's profile proved 20 Triton and 20 reference attention calls. `_attention_fwd`
used 171,113.185 us and ten-forward model time fell from 2,790,718.259 us to
2,239,829.181 us (**-19.74%**). Its isolated complete final matrix passed at
1.839x geomean before composite integration.

## EXP-012: held-out long-causal cases

The selected exact guard is `B=2, S=512, d_model=512, heads=8, layers=2,
causal=true`, with either no padding or the measured prefix padding. It chooses
SDPA; neighboring shapes retain their previous dispatcher policy.

- 19 focused GPU Transformer tests passed, including three seeds in both mask
  modes during candidate isolation.
- Candidate held-out matrices passed 7/7 at 1.453x and 1.396x geomean.
- Both profiles proved 20 SDPA calls and the efficient attention kernel path.
- The final five-seed matrices passed 7/7 at 1.447477x and 1.449715x.

## Composite verification

| Gate | Result |
| --- | --- |
| Focused GPU/dispatch/organizer suite | 92 passed; 14 upstream warnings |
| Final primary | 13/13 executable PASS + exact skip; 0/938,885,120 failed; 1.911947x |
| Final confirmation | same correctness/backend counts; 1.995117x |
| Organizer default | 5/5 PASS; 0/2,621,440 failed; 1.397x; 1,950 Triton calls |
| Held-out primary / confirmation, five seeds | 7/7 PASS twice; zero failures; 1.447477x / 1.449715x |
| Source-derived | 28/28 executable PASS + exact skip; 0/459,776,000 failed; 1.204815x |
| Integrated row 6 profile | 20 Triton / 20 reference; `_attention_fwd` 164,981.228 us |
| Integrated row 7 profile | 30 Triton / 10 reference; `_attention_fwd` 991.253 us |
| Integrated long-causal profiles | 20 SDPA calls in each mask mode |
| Integrated row 11 profile | 40 Triton calls; custom kernel proven |
| Complete repository suite | 121/121 PASS; 14 upstream deprecation warnings |

The first pre-documentation full suite reported 116 passed and five failed.
All five failures were stale curated-artifact/fingerprint assertions caused by
the intentional candidate change; they are retained in
`C5-INTEGRATE-013-full-tests-predoc.json`. No production or numerical test
failed. The curated assertions were then updated to the new immutable artifacts
before the final full-suite rerun.

## Attempt accounting

Every attempt is an immutable JSON record under [`attempts/`](attempts/) with:
UTC start/end, child wall time, return code/timeout state, stdout/stderr, exact
command, before/after environment, Git implementation fingerprint, result hash,
correctness totals, raw timing summaries, backend counts, memory, and profiler
metrics when applicable.

Through final workflow validation, Campaign 5 contains **74 immutable attempts:
65 PASS, 9 FAIL, 0 timeouts, and 736.085883 seconds** (12 minutes 16.086
seconds) of measured child-command wall time. The independent reviewer
recomputed the first 73 records; the passing workflow validator is the 74th.

| Phase | Attempts | PASS | FAIL | Child wall time |
| --- | ---: | ---: | ---: | ---: |
| Preflight | 4 | 4 | 0 | 4.657120 s |
| Fresh baseline | 2 | 2 | 0 | 54.834449 s |
| Counterbalanced control | 1 | 1 | 0 | 30.948358 s |
| Profiles and backend screens | 12 | 7 | 5 | 38.913709 s |
| EXP-010 row 7 | 17 | 15 | 2 | 187.499923 s |
| EXP-011 row 6 | 7 | 6 | 1 | 147.393545 s |
| EXP-012 held-out causal | 5 | 5 | 0 | 17.757621 s |
| Composite integration/rebaseline | 16 | 15 | 1 | 234.530664 s |
| Test/static/workflow closure | 4 | 4 | 0 | 18.836794 s |
| Repository graph closure | 6 | 6 | 0 | 0.713699 s |
| **Total** | **74** | **65** | **9** | **736.085883 s** |

The nine retained nonzero commands are the five full-backend accuracy screens,
EXP-010 I1, EXP-010's malformed focused selector, EXP-011 I1, and the stale
pre-documentation artifact suite. The post-documentation full suite is
`C5-CLOSE-002-full-suite.json`: **121 passed, 14 warnings**, zero failures,
11.0905135 seconds of child wall time. `C5-CLOSE-003-diff-check.json` then
passed; its stderr contains only expected AutoCRLF notices. Strict graph
validation passed with zero errors and zero warnings.

The completed workflow then validated with zero errors and zero warnings in
`C5-CLOSE-004-workflow-final.json`.

## Decision

**KEEP the composite Campaign 5 implementation.** It clears the approximate 5%
aggregate gate, improves the two targeted final rows, removes both held-out
regressions, and preserves zero failed elements across the complete final,
organizer-default, held-out, and source-derived gates. Row 8 remains exact
reference because both the accuracy screen and profile ceiling reject further
attention routing. No commit, push, tag, release, or public action is part of
this campaign turn.

The independent composite release review recomputed every Campaign 5 attempt,
metric, artifact link, route count, profile, test, diff, and graph gate and
returned `APPROVE_LOCAL_ACCEPTANCE` with no blockers. Its immutable decision
sidecar is [`reviews/CAMPAIGN-005-CANDIDATE-REVIEW.json`](reviews/CAMPAIGN-005-CANDIDATE-REVIEW.json).
