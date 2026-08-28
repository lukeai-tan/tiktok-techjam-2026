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

Rows are added only from committed attempt JSON and experiment decisions; no
metric is copied from an unrecorded console run.

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
- Final logging gate: 8/8 focused logger tests passed, including direct-file
  invocation, nonzero exit persistence, timeout persistence, invalid/missing
  artifact handling, metric extraction, and exclusive writes. The complete
  repository suite passed 99 tests with 14 upstream PyTorch deprecation
  warnings.

## Source-of-truth impact

`docs/REQUIREMENTS.md` remains authoritative. This campaign may update the
implementation support envelope, tests, kernel design, technical report, result
index, and optimization roadmap only when accepted behavior or evidence changes.
Organizer sources, comparator, tolerance, final-shape order, and benchmark timing
policy are protected and are not optimization targets.
