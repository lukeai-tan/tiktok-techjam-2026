# CAMPAIGN-002: Logged post-EXP-001 optimization loops

Status: complete; final release review pending
Parent checkpoint: `cdfada9df980dd471311bead2d378c3589e06320`  
Target: NVIDIA GeForce RTX 5070 Ti, driver 616.56, PyTorch 2.13.0+cu130,
Triton 3.7.1.post27, Python 3.12.10

## Objective

Continue bounded end-to-end optimization after EXP-001 while preserving the
organizer contract and making every attempt independently auditable. Failed,
slower, invalid, and inconclusive attempts remain part of the record.

## Attempt evidence contract

Every executed attempt must have a versioned JSON record under
`docs/experiments/attempts/` containing:

- identity, hypothesis, parent commit, candidate branch/commit, changed paths,
  decision state, and rollback;
- exact argv, UTC start/end, command wall time, return code, and captured
  stdout/stderr;
- Git dirty state and implementation fingerprint before and after execution;
- Python, framework, CUDA, GPU, driver, CPU, and disk metadata;
- result-artifact path and SHA-256;
- requested/executable/pass/fail/OOM/error/skip counts, accuracy trials,
  compared/failed elements, and maximum absolute/relative errors;
- baseline/optimized latency distributions, per-case and geometric-mean
  speedup, throughput where available, and command/per-case durations;
- peak allocation, backend counts, custom-kernel profiler events, and any
  compile metadata exposed by the measured artifact; and
- an explicit `keep`, `reject`, `rework`, `inconclusive`, or `observation`
  decision with rationale and reviewer status.

The logger must persist a record even when the command exits nonzero, crashes,
times out, or does not produce a metrics artifact.

## Acceptance and stopping gates

- Zero failed elements, NaN/Inf mismatch, unauthorized skip, or protected-source
  drift across required validation.
- Approximately 5% paired final-matrix geomean improvement, or a clearly
  measured memory/launch benefit, with no affected required case regressing
  more than 2% without an explicit independent waiver.
- Actual custom-kernel execution where claimed; fallback counts remain visible.
- At most three new candidate hypotheses in this campaign and at most three
  producer/reviewer cycles per hypothesis.
- Stop early when the remaining profile ceiling is immaterial, measurements are
  unstable, or candidates fall below the noise/acceptance threshold.

## Attempt ledger

| ID | Hypothesis | Scope | Wall time | Correctness | Performance | Decision |
| --- | --- | --- | ---: | --- | --- | --- |
| C2-BASE-001-FINAL-ROW13 | Fresh long-`head_dim=32` baseline | final row 13 | 5.820 s | PASS; 0/41,943,040 failed | 17.6598 ms; 4.790x | observation |
| C2-BASE-002-FINAL-ROW1 | Fresh short-`head_dim=32` baseline | final row 1 | 3.069 s | PASS; 0/5,242,880 failed | 1.2358 ms; 1.191x | observation |
| C2-OBS-001 | Row-1 short `head_dim=32` attention remains material | final row 1 profile | 2.666 s | profile-only; Triton 40/40 | `_attention_fwd` 6,150.907 us / 40; 33.55% of optimized range | observation |
| C2-OBS-002 | Accepted short `head_dim=64` still has residual ceiling | final row 10 profile | 2.541 s | profile-only; Triton 40/40 | `_attention_fwd` 2,690.243 us / 40; 28.44% of optimized range | observation |
| C2-OBS-003 | Long `head_dim=32` attention dominates row 13 | final row 13 profile | 2.824 s | profile-only; Triton 40/40 | `_attention_fwd` 192,575.551 us / 40; 82.00% of optimized range | observation |
| C2-OBS-004 | Sequence-512 `head_dim=64` attention explains held-out pressure | held-out causal-padding profile | 2.754 s | profile-only; Triton 20/20 | `_attention_fwd` 4,926.368 us / 20; 53.49% of optimized range | observation |
| C2-EXP-002-I1-FAST | Long `head_dim=32`: `BLOCK_N=128`, two stages | focused dispatch tests | 1.769 s | 16/16 tests passed | structural gate only | reject |
| C2-EXP-002-I1-ROW13 | Long `head_dim=32`: `BLOCK_N=128`, two stages | final row 13 | 7.899 s | PASS; 0/41,943,040 failed; max abs 0.00114191; Triton 112/112 | 41.3406 ms, 2.047x vs 17.6598 ms, 4.790x baseline | reject |
| C2-EXP-002-I2-FAST | Long `head_dim=32`: two stages only | focused dispatch tests | 1.744 s | 16/16 tests passed | structural gate only | reject |
| C2-EXP-002-I2-ROW13 | Long `head_dim=32`: two stages only | final row 13 | 6.595 s | PASS; 0/41,943,040 failed; max abs 0.00114191; Triton 112/112 | 17.9966 ms, 4.701x vs 17.6598 ms, 4.790x baseline | reject |
| C2-EXP-002-I3-FAST | Long `head_dim=32`: `BLOCK_M=32` | focused dispatch tests | 1.775 s | 16/16 tests passed | structural gate only | reject |
| C2-EXP-002-I3-ROW13 | Long `head_dim=32`: `BLOCK_M=32` | final row 13 | 6.471 s | PASS; 0/41,943,040 failed; max abs 0.00114191; Triton 112/112 | 21.5897 ms, 3.918x vs 17.6598 ms, 4.790x baseline | reject |
| C2-EXP-003-I1-FAST | Short `head_dim=32`: 32x64 tiles | focused dispatch tests | 2.720 s | 16/16 tests passed | structural gate only | superseded |
| C2-EXP-003-I1-ROW1 | Short `head_dim=32`: 32x64 tiles | final row 1 | 3.568 s | PASS; 0/5,242,880 failed; Triton 112/112 | 0.8579 ms; 1.559x | superseded |
| C2-EXP-003-I2-FAST | Short `head_dim=32`: 32x128 tiles | focused dispatch tests | 3.276 s | 16/16 tests passed | structural gate only | reject |
| C2-EXP-003-I2-ROW1 | Short `head_dim=32`: 32x128 tiles | final row 1 | 3.744 s | PASS; 0/5,242,880 failed; Triton 112/112 | 0.8805 ms; 1.516x | reject |
| C2-EXP-003-I3-FAST | Short `head_dim=32`: 64x64 tiles | focused dispatch tests | 3.190 s | 16/16 tests passed | structural gate only | keep |
| C2-EXP-003-I3-ROW1 | Short `head_dim=32`: 64x64 tiles | final row 1 | 3.766 s | PASS; 0/5,242,880 failed; Triton 112/112 | 0.8164 ms; 1.653x | keep |
| C2-EXP-003-CONF-A-BASE | Alternating confirmation control A | final row 1 | 3.071 s | PASS; 0/5,242,880 failed | 1.2425 ms | observation |
| C2-EXP-003-CONF-A-I1 | Alternating confirmation I1 A | final row 1 | 3.052 s | PASS; 0/5,242,880 failed | 0.8660 ms | observation |
| C2-EXP-003-CONF-A-I3 | Alternating confirmation I3 A | final row 1 | 2.995 s | PASS; 0/5,242,880 failed | 0.8203 ms | observation |
| C2-EXP-003-CONF-B-BASE | Alternating confirmation control B | final row 1 | 3.170 s | PASS; 0/5,242,880 failed | 1.2424 ms | observation |
| C2-EXP-003-CONF-B-I1 | Alternating confirmation I1 B | final row 1 | 3.014 s | PASS; 0/5,242,880 failed | 0.8565 ms | observation |
| C2-EXP-003-CONF-B-I3 | Alternating confirmation I3 B | final row 1 | 3.068 s | PASS; 0/5,242,880 failed | 0.8235 ms | observation |
| C2-EXP-003-POSTMERGE-TESTS | Integrated code preserves full suite | complete pytest | 11.443 s | 97 pass, 4 fail, 14 warnings | expected stale fingerprints | rework |
| C2-EXP-003-REBASE-FINAL | Integrated candidate clears final gate | 14 final rows | 48.853 s | 13/13 executable PASS; 0/938,885,120 failed; one authorized skip | 1.525823x; +6.948% vs prior | keep |
| C2-EXP-003-REBASE-DEFAULT | Organizer default remains correct | untouched default harness | 2.925 s | 5/5 PASS; 0/2,621,440 failed; Triton 1,950/1,950 | 1.3374 ms; 1.314x | keep |
| C2-EXP-003-REBASE-HELDOUT | Candidate does not overfit final rows | seven held-out cases | 2.939 s | 7/7 PASS; 0/13,117,440 failed | 1.228277x; all optimized medians lower | keep |
| C2-EXP-003-REBASE-SOURCE | Source-derived contracts remain green | 29 source-derived entries | 62.112 s | 28/28 executable PASS; 0/459,776,000 failed; one authorized skip | 1.193898x; all optimized medians lower | keep with timing-noise note |
| C2-EXP-003-REBASE-PROFILE-ROW1 | Target kernel time falls | row 1 profile | 3.281 s | Triton 40/40 | 2,103.978 us; -69.980% | keep |
| C2-EXP-003-REBASE-PROFILE-ROW10 | EXP-001 neighbor stays intact | row 10 profile | 2.574 s | Triton 40/40 | 2,694.679 us; -15.937% vs prior integrated | keep |
| C2-EXP-003-REBASE-PROFILE-HELDOUT | Held-out neighbor stays intact | long-causal-padding profile | 2.546 s | Triton 10/10 | 2,460.878 us; -32.835% vs prior | keep |
| C2-EXP-003-POSTREBASE-TESTS | Rebaselined suite is fully current | complete pytest | 7.452 s | one stale metric assertion; logger tee encoding error | diagnostic failure retained | rework |
| C2-LOG-HARDENING-004-FOCUSED | Unicode-safe tee and metric assertion | focused logger/result tests | 3.148 s | 16 pass, 1 Windows-newline assertion fail | tee itself preserved output | rework |
| C2-LOG-HARDENING-005-FOCUSED | Portable logger hardening | focused logger/result tests | 3.153 s | 17/17 passed | Unicode and CRLF covered | keep |
| C2-EXP-003-FINAL-TESTS | Integrated, rebaselined state is green | complete pytest | 7.571 s | 102 passed, 14 upstream warnings | final automated gate | keep |
| C2-EXP-004-I1-DIRECT | Existing kernel safely supports `head_dim=8` | direct GPU attention suite | 2.080 s | 11 passed; `head_dim=8` compile failed because dot K must be >=16 | no performance run authorized | reject |

Rows are added only from committed attempt JSON and experiment decisions; no
metric is copied from an unrecorded console run.

## Ranked candidate hypotheses

1. **EXP-002 — long `head_dim=32` launch geometry.** Row 13 is the largest
   measured remaining bottleneck at 82.00% of the optimized range. Test bounded
   `BLOCK_M`, `BLOCK_N`, warp, and stage variants without changing arithmetic,
   dispatch, or the organizer policy.
2. **EXP-003 — short `head_dim=32` launch geometry.** Row 1 still spends 33.55%
   of its range in attention, and the same policy covers final rows 1-5 and 12.
   Only proceed after EXP-002 is decided so launch-policy branches do not overlap.
3. **EXP-004 — guarded `head_dim=8` custom support.** Final rows 7 and 11 use
   exact reference fallback today. Attempt only if direct correctness and compile
   evidence show the existing kernel is structurally safe at this width; preserve
   immediate rollback to reference.

The held-out sequence-512 `head_dim=64` path remains a regression signal and a
required anti-overfitting check, but it is not promoted ahead of hypotheses that
affect published final rows.

### EXP-002 decision

Closed as **rejected** after all three allowed producer cycles. Every candidate
preserved the organizer accuracy contract and executed Triton, but none improved
the fresh baseline. The least harmful change, reducing pipeline depth only,
still regressed optimized median latency by 1.91%. No EXP-002 implementation
commit is integrated; only the immutable attempt and result evidence is retained.

### EXP-003 decision

Accepted and integrated after three producer cycles, two alternating
confirmation rounds, and independent pre-integration review. The 64x64 tile
was selected only for `head_dim == 32 and seq_len <= 128`; arithmetic,
dispatch, tolerances, timing policy, and protected organizer sources are
unchanged. Across the three screen/confirmation samples, optimized row-1
latency was 0.8164/0.8203/0.8235 ms (mean 0.8201 ms, sample SD 0.0036 ms)
versus 1.2358/1.2425/1.2424 ms for the unchanged policy (mean 1.2402 ms,
sample SD 0.0038 ms).

Post-integration final geomean rose from 1.426692x to 1.525823x (+6.948%).
All directly affected executable final rows improved, every correctness gate
remained zero-failure, and the largest unaffected final optimized-latency
regression was 1.59%, inside the 2% threshold. The current fingerprint is
`8eb7d21551ab69e83f532deaeefb2ce1999dc3e198f48a8d4be5753ad2c93a8a`.

### EXP-004 decision

Closed as **rejected** at the mandatory first compile gate. Adding width 8 to
the support set caused Triton's dot lowering to fail with `K >= 16`; the other
11 direct-kernel cases passed. Padding the dot dimension or introducing
alternate arithmetic would be a different, wider kernel-design hypothesis, so
no model-level or performance run was authorized and production keeps exact
reference fallback for final rows 7 and 11.

Campaign 2 contains 37 attempt records: 33 child commands passed and four
failed gates were retained as rework or rejection evidence. Recorded
child-command wall time totals 244.635 seconds; orchestration, review, commits,
and documentation time are intentionally excluded from that machine-measured
sum.

## Logging implementation findings

- `LOG-SELFTEST-001`: the initial direct-file CLI invocation failed before a
  record could be written with `ModuleNotFoundError: No module named 'tools'`.
  Pytest had masked the entrypoint-path difference. The logger now inserts the
  repository root before importing shared capture helpers; the failure and
  corrected nonzero-command self-test are retained here because logging
  failures must not be silently omitted.
- `LOG-SELFTEST-002`: the first organizer-validation summary test failed with
  `TypeError: unsupported operand type(s) for |: 'tuple' and 'dict'` because a
  dictionary-union expression had incorrect comprehension precedence. The
  extractor now unions two explicit dictionaries and the regression remains
  covered by `tests/test_optimization_attempt.py`.
- Initial logging gate: 9/9 focused logger tests passed, including direct-file
  invocation, nonzero exit persistence, timeout persistence, invalid/missing
  artifact handling, metric extraction, and exclusive writes. The complete
  repository suite passed 100 tests with 14 upstream PyTorch deprecation
  warnings.
- `LOG-HARDENING-003`: pre-measurement review found that exact raw argv would
  persist the external virtual-environment path. Execution still uses the real
  argv, but the durable record now renders absolute paths as repo-relative paths
  or basenames and captures the exact Python/runtime identity separately. A
  regression test verifies that the user's home path is absent.
- `LOG-HARDENING-004`: the post-rebaseline failure emitted a replacement
  character that Windows' cp1252 console could not encode. The logger persisted
  the attempt but its live tee thread raised `UnicodeEncodeError`. Tee output now
  replaces console-unsupported characters while retaining the original Unicode
  in the JSON record.
- `LOG-HARDENING-005`: the first hardening test itself assumed LF bytes and
  failed under Windows CRLF translation (16 pass, 1 fail). The portable
  assertion passed with the complete focused gate at 17/17, followed by the
  complete repository gate at 102 passed.

## Source-of-truth impact

`docs/REQUIREMENTS.md` remains authoritative. This campaign may update the
implementation support envelope, tests, kernel design, technical report, result
index, and optimization roadmap only when accepted behavior or evidence changes.
Organizer sources, comparator, tolerance, final-shape order, and benchmark timing
policy are protected and are not optimization targets.
