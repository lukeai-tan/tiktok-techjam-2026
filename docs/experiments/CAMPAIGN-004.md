# CAMPAIGN-004: Profile-led head-dimension-8 optimization round

Canonical overview: [Complete Track 3 Optimization History](OPTIMIZATION_HISTORY.md).
This file remains the detailed Campaign 4 attempt and decision ledger.

Status: independently approved for local acceptance; measured as a dirty,
uncommitted checkpoint
Parent checkpoint: `b41fdaf90f869a920346401b2b9fd93899fe805e`
Baseline tag: `track3-optimization-campaign3-accepted-20260828`
Target: NVIDIA GeForce RTX 5070 Ti, driver and runtime to be revalidated in the
immutable preflight record.

## Task envelope

Objective: continue the bounded Track 3 optimization loop while retaining every
executed attempt and its correctness, latency, memory, backend, profiler,
environment, provenance, wall-time, and decision evidence.

Observable outcome: either retain a reproducible candidate that clears all
correctness, performance, evidence, maintainability, and independent-review
gates, or close Campaign 4 with measured plateau evidence.

In scope:

- fresh measurement and profiling of final row 11, the only executable Campaign
  3 final row slower than its baseline;
- controlled `auto`, exact-reference, and SDPA backend comparison;
- a profile-authorized padded-width Triton design for `head_dim=8`, with at most
  three bounded candidate configurations and immediate rollback;
- affected row 7, boundary, held-out, source-derived, exact organizer, and full
  regression gates for any retained change.

Out of scope:

- organizer sources, final-shape order, comparator, tolerances, timing policy,
  or authorized resource-skip policy;
- forcing unsupported behavior without direct compilation and correctness proof;
- unrelated fusion, new dependencies, backward support, deployment, push, or
  public submission actions.

## Acceptance and stopping gates

- Zero failed elements, NaN/Inf mismatch, unauthorized fallback, unexpected
  error/OOM, or unauthorized skip across required gates.
- Approximately 5% final-matrix geomean improvement, or a clearly measured
  profiler-backed kernel, memory, or launch benefit, with no affected required
  case regressing more than 2% absent independent waiver.
- Actual backend counts and profiler events must prove the claimed path.
- At most three candidates for the head-dimension-8 subsystem; stop earlier if
  attention is immaterial, correctness fails structurally, or results are noise.

## Attempt evidence contract

Every test, profile, benchmark, candidate, confirmation, and closure gate runs
through `benchmarks/run_optimization_attempt.py`. Immutable JSON records live in
`docs/experiments/attempts/`; rejected, failed, rework, inconclusive, and plateau
outcomes remain present.

## Attempt ledger

| ID | Hypothesis | Scope | Wall time | Correctness / evidence | Decision |
| --- | --- | --- | ---: | --- | --- |
| C4-PREFLIGHT-000-DEFAULT-PYTHON | Default Python may be usable | import gate | 0.112 s | FAIL: Python 3.14 had no `torch` | rework |
| C4-PREFLIGHT-000B-ENVIRONMENT | Rebuilt pinned Windows environment is usable | environment capture | 1.220 s | PASS: Python 3.12.10, Torch 2.13.0+cu130, Triton 3.7.1, CUDA 13.0, RTX 5070 Ti | keep |
| C4-PREFLIGHT-001-WORKFLOW | Workflow contract is complete | workflow validation | 0.058 s | FAIL: two required final-report sections absent; one graph warning | rework |
| C4-PREFLIGHT-002-WORKFLOW-REWORK | Corrected workflow is complete | workflow validation | 0.051 s | PASS: 0 errors, 0 warnings | keep |
| C4-PREFLIGHT-003-BASELINE | Baseline evidence/tests remain green | 33 focused tests | 3.312 s | FAIL: 32 passed, logger persisted `.venv/Scripts/python.exe` | rework |
| C4-PREFLIGHT-004-LOGGER-PORTABILITY | Logger canonicalizes its active interpreter | focused logger test | 1.345 s | PASS: 1/1 | keep infrastructure fix |
| C4-PREFLIGHT-005-BASELINE-REWORK | Reworked baseline gate is green | 33 focused tests | 3.204 s | PASS: 33/33 | keep |
| C4-BASE-001-FINAL-ROW11 | Fresh row 11 reproduces the remaining slow path | exact final row 11 | 3.488 s | PASS, 0/5,242,880 failed; reference 112; 5.7495/5.6149 ms; 1.024x | observation |
| C4-OBS-001-FINAL-ROW11-PROFILE | Final matrix can serve directly as profile manifest | profile setup | 1.185 s | FAIL: manifest uses `explicit_cases`, profiler requires `cases` | rework |
| C4-OBS-002-PROFILE-MANIFEST | Exact row 11 has a profile manifest | manifest test | 1.389 s | PASS: 1/1 and exact source match | keep |
| C4-OBS-003-FINAL-ROW11-PROFILE-REWORK | Reference attention dominates row 11 | exact profile | 2.956 s | PASS: reference 40; optimized range 41,658.659 us | observation |
| C4-OBS-004-BACKEND-AUTO | Auto establishes a reference control | backend screen | 1.925 s | PASS; reference 92; 4.1410 ms; 1.007x | observation |
| C4-OBS-005-BACKEND-SDPA | SDPA removes the reference bottleneck | backend screen | 1.853 s | PASS; SDPA 92; 1.2399 ms; 3.354x | advance SDPA candidate |
| C4-OBS-006-BACKEND-REFERENCE | Forced reference reproduces auto | backend screen | 1.905 s | PASS; reference 92; 4.1516 ms; 1.003x | observation |
| C4-EXP-008-I1-FAST | Exact SDPA route preserves model correctness | focused model stress | 2.049 s | PASS: 4/4 including 18 seed/scale/padding scenarios | keep correctness evidence |
| C4-EXP-008-I1-ROW11 | Exact production SDPA improves row 11 | exact final row 11 | 3.200 s | PASS, 0 failed; SDPA 112; 1.8658 ms; 3.179x | superseded by Triton |
| C4-EXP-008-I1-AFFECTED | SDPA route leaves historical row 7 safe | exact rows 7 and 11 | 4.883 s | PASS 2/2; reference/SDPA split; 1.814x geomean | keep evidence; candidate superseded |
| C4-EXP-009-I1-DIRECT | Zero-padded width eight compiles and is correct | direct GPU attention | 7.182 s | PASS: 13/13 float16/float32 mask/boundary cases | keep enabling evidence |
| C4-EXP-009-I1-FAST | 64x128 padded Triton preserves model correctness | focused model stress | 2.908 s | PASS: 4/4 including 18 scenarios | keep enabling evidence |
| C4-EXP-009-I1-ROW11 | 64x128 padded Triton beats SDPA | exact final row 11 | 3.116 s | PASS, 0 failed; Triton 112; 1.2739 ms; 4.494x | superseded by I2 |
| C4-EXP-009-I1-AFFECTED | I1 preserves rows 7 and 11 | exact affected rows | 4.959 s | PASS 2/2; reference/Triton split; 2.165x geomean | keep evidence; candidate superseded |
| C4-EXP-009-I1-PROFILE | I1 executes the intended custom kernel | exact row-11 profile | 3.375 s | PASS: Triton 40; `_attention_fwd` 6,618.305 us | observation |
| C4-EXP-009-I2-FAST | 64x64 is correct at the target boundary | direct/dispatch tests | 2.859 s | FAIL: 13 passed, one stale duplicate test assertion failed | rework; retained |
| C4-EXP-009-I2-FAST-REWORK | Corrected I2 gate is green | direct/dispatch tests | 2.110 s | PASS: 15/15 | keep |
| C4-EXP-009-I2-ROW11 | 64x64 is faster than I1 | exact final row 11 | 3.677 s | PASS, 0 failed; Triton 112; 1.0595 ms; 5.409x | keep |
| C4-EXP-009-I2-CONF-A | I2 reproduces in confirmation A | exact final row 11 | 3.281 s | PASS, 0 failed; Triton 112; 1.0628 ms; 5.725x | keep |
| C4-EXP-009-I2-CONF-B | I2 reproduces counterbalanced | exact final row 11 | 3.159 s | PASS, 0 failed; Triton 112; 1.0624 ms; 5.478x | keep |
| C4-EXP-009-I2-AFFECTED | I2 preserves rows 7 and 11 | exact affected rows | 4.896 s | PASS 2/2; reference/Triton split; 2.372x geomean | keep |
| C4-EXP-009-I2-PROFILE | I2 lowers custom-kernel time | exact row-11 profile | 3.245 s | PASS: Triton 40; `_attention_fwd` 4,993.999 us | keep |
| C4-EXP-009-I3-FAST | 32x64 remains correct | direct/dispatch tests | 2.670 s | PASS: 15/15 | keep correctness evidence |
| C4-EXP-009-I3-ROW11 | 32x64 may beat I2 | exact final row 11 | 3.648 s | PASS, 0 failed; Triton 112; 1.3105 ms; 4.633x | reject: 18.93% slower than I2 median |
| C4-INTEGRATE-000-BOUNDARY-LABEL | Test wording can be clarified without code drift | dispatch test | 1.426 s | PASS: 1/1; implementation hash unchanged | keep |
| C4-INTEGRATE-001-FOCUSED | Reviewed candidate clears focused integration | four focused files | 12.401 s | PASS: 64/64; 14 deprecation warnings | keep |
| C4-INTEGRATE-002-FINAL | Candidate clears the exact final matrix | 14 published rows | 55.692 s | 13/13 executable PASS; authorized skip; 0/938,885,120 failed; 1.780075x | keep; final review approved |
| C4-INTEGRATE-002B-FINAL-CONFIRMATION | Full-matrix gain reproduces | 14 published rows | 52.365 s | same correctness/backends; 1.784920x | keep; final review approved |
| C4-INTEGRATE-003-DEFAULT | Untouched organizer default stays green | default harness | 4.524 s | 5/5 PASS; 0/2,621,440 failed; Triton 1,950; 1.881x | keep |
| C4-INTEGRATE-004-HELDOUT | Candidate generalizes to held-out shapes | seven cases | 7.881 s | 7/7 PASS; 0 failed; 1.434850x | keep |
| C4-INTEGRATE-005-SOURCE | Source-derived contract stays green | 29 entries | 67.956 s | 28/28 executable PASS; authorized skip; 0/459,776,000 failed; 1.215170x | keep |
| C4-INTEGRATE-006-AFFECTED | Final affected route split is exact | rows 7 and 11 | 5.794 s | 2/2 PASS; reference 112/Triton 112; 2.416x geomean | keep |
| C4-INTEGRATE-007-PROFILE | Integrated row 11 proves custom execution | exact profile | 2.876 s | Triton 40; `_attention_fwd` 4,767.023 us; model range 10,592.605 us | keep |
| C4-INTEGRATE-008-CURATED-ARTIFACTS | Curated claims bind to current evidence | artifact tests | 1.703 s | PASS: 8/8 | keep |
| C4-INTEGRATE-009-FULL-TESTS | Repository remains green | full pytest | 9.703 s | PASS: 112/112; 14 deprecation warnings | keep |
| C4-CLOSURE-001-WORKFLOW | Closure workflow remains valid | workflow validation | 0.049 s | PASS: 0 errors, 0 warnings | keep |
| C4-CLOSURE-002-FULL-TESTS | Post-doc/council repository remains green | full pytest | 7.667 s | PASS: 112/112; 14 deprecation warnings | keep |

IDs, hypotheses, scopes, wall times, execution outcomes, and measured evidence
above are transcribed from immutable attempt JSON. The Decision column is the
closure disposition owned by the candidate, council, and final review sidecars;
it intentionally does not overwrite the execution-time `decision` or
`review_status` retained in each attempt record. At this checkpoint there
are 44 records: 39 child commands passed, five deliberately retained failed
gates did not pass, and measured child wall time totals 315.261 seconds. The
records retain command, stdout/stderr, return code, wall time, environment,
implementation fingerprint, result hash/link, and parsed accuracy, latency,
memory, backend, profiler, or test metrics when applicable.

## Pre-logger startup incidents

Two launcher failures occurred before the logger itself could start and are not
misrepresented as immutable attempt records:

- default `python` reached Python 3.14 without Torch and failed after 0.117 s;
  the same failure was reproduced by `C4-PREFLIGHT-000-DEFAULT-PYTHON`;
- the checked-in ignored `.venv` was a Linux environment with no Windows
  `.venv\Scripts\python.exe`, so that direct startup failed after 0.005 s.

The exact ignored `.venv` target was rebuilt with documented Python 3.12 and
the pinned GPU packages. No tracked source or user data was removed.

## Selected design and correctness boundary

Triton dot rejects `K < 16`. The selected implementation separates the real
`HEAD_DIM=8` from compile-time `DOT_HEAD_DIM=16`. Padded Q/K/V lanes are masked
to zero, output stores are masked to eight lanes, and softmax scale uses the real
width. Existing 16/32/64/128 paths keep `DOT_HEAD_DIM == HEAD_DIM`.

Automatic multi-layer routing is exact and fail-closed: only
`B=64,S=128,d_model=128,heads=16,layers=4,causal=true` selects the padded Triton
path. Final row 7 remains reference, as do other width-eight shapes until they
receive separate performance evidence. Direct forced width-eight correctness is
covered across float16/float32, causal/all-valid/prefix masks, sequence 31/65,
and boundary dispatch; model stress covers three seeds, three input scales, and
two padding ratios.

## Candidate comparison

| Candidate | Backend / launch | Exact row-11 optimized median | Relative decision |
| --- | --- | ---: | --- |
| fresh baseline | exact reference | 5.6149 ms | control |
| EXP-008-I1 | SDPA | 1.8658 ms | correct, superseded |
| EXP-009-I1 | Triton 64x128 | 1.2739 ms | correct, 16.60% slower than I2 median |
| EXP-009-I2 | Triton 64x64 | 1.0624 ms three-run median | selected |
| EXP-009-I3 | Triton 32x64 | 1.3105 ms | correct, 18.93% slower than I2 median |

I2's three optimized medians were 1.0595/1.0628/1.0624 ms, a 0.0033 ms range
(0.311%). The median is 81.08% below fresh reference and 43.06% below the exact
production SDPA alternative. Its profile reduced `_attention_fwd` 24.54% versus
I1. The integrated profile records a 74.57% reduction in the ten-step optimized
model range versus the fresh reference profile.

## Integration regression review

The primary final geomean increased from Campaign 3's 1.555780x to 1.780075x
(+14.417%); the complete confirmation measured 1.784920x. Across all 13 rows,
the paired geometric mean of optimized latency improved 1.109159x and the paired
geometric mean of normalized speedup improved 1.144168x.

Only rows 7 and 11 are in the explicit width-eight risk envelope. Row 7 stayed
on reference; its raw optimized latency moved +8.008% while its same-run baseline
moved +7.747%, leaving normalized speedup -0.187% versus Campaign 3, inside the
2% gate. Row 11's baseline moved -0.092%, optimized latency fell 81.890%, and
normalized speedup improved 451.636%. Other row-to-row raw timing movement was
treated as environment noise because the corresponding math/dispatch was
unchanged and the second complete matrix reproduced the aggregate within 0.28%.

## Review decision and residual risk

The independent candidate review in
`reviews/CAMPAIGN-004-CANDIDATE-REVIEW.json` approved EXP-009-I2 for integration
rebaseline and verified the padding math, exact route, run hashes, candidate
fingerprint, three confirmations, affected rows, and profiler proof. After
documentation, council, graph, workflow, and final-test closure, the independent
gatekeeper in `reviews/CAMPAIGN-004-FINAL-REVIEW.json` returned
`APPROVE_LOCAL_ACCEPTANCE` with no blocker for this exact dirty fingerprint.
This is content-level local acceptance only and authorizes no Git or public
action.

Residual risk is explicit: timing is RTX-5070-Ti-specific; strict comparison
passes with a nonzero row-11 maximum absolute difference of 0.00106898; direct
forced width-eight correctness is broader than the deliberately narrow automatic
performance route; and no backward kernel is claimed.

## AI Council review

The post-integration council reviewed requirements, architecture,
implementation/numerics, security/operations, tests/performance, documentation,
and a devil's-advocate challenge. Its structured record is
`reviews/CAMPAIGN-004-AI-COUNCIL.json`. The council found no blocker and approved
the candidate for independent final review, conditional on preserving the exact
row-11 route, keeping unmeasured width-eight model shapes on reference, retaining
dirty-worktree provenance, and retesting on any different evaluator GPU. Those
controls were carried into the final approval sidecar.

## Graph closure

The repository graph received separate verified implementation and validation
events for Campaign 4. Generated notes were rebuilt from 45 total events and
the graph validator returned zero errors and zero warnings. The graph records
the implementation hash, exact routing boundary, final/confirmation metrics,
profile proof, attempt ledger, and residual evaluator-GPU risk without copying
source or secrets.
