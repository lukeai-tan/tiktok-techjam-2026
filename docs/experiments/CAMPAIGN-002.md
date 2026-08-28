# CAMPAIGN-002: Logged post-EXP-001 optimization loops

Status: in progress  
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
- Final logging gate: 9/9 focused logger tests passed, including direct-file
  invocation, nonzero exit persistence, timeout persistence, invalid/missing
  artifact handling, metric extraction, and exclusive writes. The complete
  repository suite passed 100 tests with 14 upstream PyTorch deprecation
  warnings.
- `LOG-HARDENING-003`: pre-measurement review found that exact raw argv would
  persist the external virtual-environment path. Execution still uses the real
  argv, but the durable record now renders absolute paths as repo-relative paths
  or basenames and captures the exact Python/runtime identity separately. A
  regression test verifies that the user's home path is absent.

## Source-of-truth impact

`docs/REQUIREMENTS.md` remains authoritative. This campaign may update the
implementation support envelope, tests, kernel design, technical report, result
index, and optimization roadmap only when accepted behavior or evidence changes.
Organizer sources, comparator, tolerance, final-shape order, and benchmark timing
policy are protected and are not optimization targets.
