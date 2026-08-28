# CAMPAIGN-003: Logged post-Campaign-2 optimization round

Status: integrated and rebaselined; final release review pending
Parent checkpoint: `b884cdd3805d51c1cd7cbd34ba19c37474135237`
Baseline tag: `track3-optimization-campaign3-baseline-20260828`
Target: NVIDIA GeForce RTX 5070 Ti, driver 616.56, PyTorch 2.13.0+cu130,
Triton 3.7.1.post27, Python 3.12.10

## Task envelope

Objective: improve the accepted Track 3 implementation through another bounded,
profile-driven round while preserving the organizer contract and recording every
executed attempt, including failed, slower, invalid, and inconclusive work.

Observable outcome: either integrate a reproducible candidate that clears the
existing correctness, performance, evidence, and maintainability gates, or close
the round with measured plateau evidence that prevents low-value repetition.

In scope:

- fresh profiling of final row 9, currently the only executable final row below
  baseline at 0.945x;
- short-`head_dim=128` launch geometry, shape-aware attention routing, and only
  profile-justified projection or launch-count changes;
- isolated candidate work, affected-neighbor checks, held-out regression gates,
  integration rebaseline, documentation, and graph closure.

Out of scope:

- organizer sources, final-shape order, comparator, tolerances, and timing policy;
- forcing Triton onto unsupported widths, changing arithmetic to rescue a failed
  candidate, speculative broad fusion, new dependencies, backward support, or
  public submission actions.

Assumptions and risks:

- the frozen organizer inputs remain authoritative until a verifiable revision
  is supplied; unstated dtype, padding, timing, and backward policy remain open;
- all performance claims are RTX 5070 Ti-specific;
- the GPU queue is serialized to avoid timing interference;
- row 9 may be dominated by vendor GEMMs or framework overhead rather than the
  custom kernel, in which case launch tuning must stop early.

## Acceptance and stopping gates

- Zero failed elements, NaN/Inf mismatch, unauthorized fallback, unexpected
  error/OOM, or unauthorized skip across required candidate gates.
- Approximately 5% paired final-matrix geomean improvement, or a clearly
  measured memory/launch benefit, with no affected required case regressing more
  than 2% without explicit independent waiver.
- Backend counts and profiler events must prove the implementation that ran.
- At most three launch-policy iterations and one profile-justified alternate
  surface; stop after three unsuccessful candidates for one subsystem.
- Stop when the target is immaterial, unstable, below the noise floor, or would
  require changing protected policy/arithmetic.

## Planned workflow

| Step | Owner role | Artifact | Validation / exit |
| --- | --- | --- | --- |
| Preflight | Task Intake + Planning | workflow, baseline tag, graph event | workflow validator; clean source/fingerprint checkpoint |
| Profile | Backend Performance Engineer | row-9 baseline, trace, backend comparison | correctness first; profiler/backend agreement |
| Candidates | Machine Learning Engineer | isolated launch/routing candidates | fast tests, direct GPU, row 9, affected neighbors |
| Validate | Performance Test Engineer | alternating statistics and regression review | correct, reproducible, within thresholds |
| Review | Independent Reviewer and Critic | read-only decision packet | provenance, noise, fallback, maintainability |
| Integrate | Workflow Orchestrator | accepted code and regenerated evidence | every curated gate and full suite green |
| Close | Release Gatekeeper | docs, ledger, graph, local checkpoint | independent review, AI Council, clean tree |

No workflow steps are parallelized because candidate policies overlap and GPU
measurements must not contend.

## Initial hypothesis ranking

1. **EXP-005: short `head_dim=128` launch geometry.** Current final row 9 uses
   32x64 tiles and is the only executable final row below parity. Profile first,
   then test at most three bounded tile/warp variants without arithmetic changes.
2. **EXP-006: shape-aware backend policy.** Compare Triton, SDPA, and exact
   reference on the same causal row before changing `auto`; proceed only if an
   alternate backend is correct and materially faster end to end.
3. **EXP-007: projection or launch-count envelope.** Reserved until profiling
   proves projection/packing overhead is material. Do not run speculatively.

## Attempt evidence contract

Every test, profile, benchmark, candidate, confirmation, and closure gate must
run through `benchmarks/run_optimization_attempt.py` and produce a versioned JSON
record under `docs/experiments/attempts/`. Records retain command wall time,
stdout/stderr, return state, Git and implementation identity, environment,
artifact hash, correctness counts, latency distributions, memory, backend and
profiler statistics, decision rationale, and review status. Nonzero commands and
missing or invalid artifacts remain in the ledger.

## Attempt ledger

| ID | Hypothesis | Scope | Wall time | Correctness / evidence | Decision |
| --- | --- | --- | ---: | --- | --- |
| C3-PREFLIGHT-001-WORKFLOW | Campaign workflow is mechanically complete | workflow validator | 0.049 s | 0 errors; 0 warnings | keep |
| C3-PREFLIGHT-002-BASELINE | Accepted baseline remains source/evidence current | focused integrity tests | 3.388 s | 33/33 passed; fingerprint unchanged | keep |
| C3-PREFLIGHT-003-PROFILE-MANIFEST | Row 9 is an exact versioned profile case | focused manifest test | 1.381 s | 1/1 passed; dimensions match final row 9 | keep |
| C3-BASE-001-FINAL-ROW9 | Fresh row-9 baseline reproduces the regression | final row 9 | 3.486 s | PASS; 0/5,242,880 failed; Triton 112/112; 1.1745/1.2341 ms; 0.952x | observation |
| C3-OBS-001-FINAL-ROW9-PROFILE | Row-9 attention is a material bottleneck | final row 9 profile | 2.880 s | Triton 40/40; `_attention_fwd` 6,775.468 us / 40, 55.23% of optimized range | observation |
| C3-OBS-002-BACKEND-AUTO | Same-runner auto/Triton control | final row 9 | 1.982 s | PASS; 0/5,242,880 failed; Triton 92/92; 1.15082 ms | observation |
| C3-OBS-003-BACKEND-SDPA | SDPA may win short causal `head_dim=128` | final row 9 | 1.765 s | PASS; 0/5,242,880 failed; SDPA 92/92; 0.69224 ms, -39.85% vs auto | observation |
| C3-OBS-004-BACKEND-REFERENCE | Exact attention bounds fallback cost | final row 9 | 1.732 s | PASS; 0/5,242,880 failed; reference 92/92; 1.00851 ms | observation |
| C3-EXP-005-I1-FAST | Short head-128 16x64 tiles preserve focused correctness | 28 focused/direct tests | 2.985 s | 28/28 passed | reject |
| C3-EXP-005-I1-ROW9 | 16x64 reduces row-9 tile pressure | final row 9 | 3.930 s | PASS; 0/5,242,880 failed; Triton 112/112; 0.9609 ms | reject; slower than I2 |
| C3-EXP-005-I2-FAST | Short head-128 32x32 tiles preserve focused correctness | 28 focused/direct tests | 2.351 s | 28/28 passed | keep; review approved |
| C3-EXP-005-I2-ROW9 | 32x32 reduces row-9 K/V tile pressure | final row 9 | 3.839 s | PASS; 0/5,242,880 failed; Triton 112/112; 0.9042 ms | keep; review approved |
| C3-EXP-005-I3-FAST | Short head-128 16x32 tiles preserve focused correctness | 28 focused/direct tests | 2.290 s | 28/28 passed | reject |
| C3-EXP-005-I3-ROW9 | 16x32 reduces both tile axes | final row 9 | 3.766 s | PASS; 0/5,242,880 failed; Triton 112/112; 1.1087 ms | reject; slower than I2 |
| C3-EXP-006-I1-FAST | Unmasked short causal head-128 SDPA route is bounded | 46 focused/direct/model tests | 16.681 s | 46/46 passed | rework |
| C3-EXP-006-I1-ROW9 | First SDPA route activates in production | final row 9 | 3.170 s | PASS correctness, but Triton 112/112; 1.2322 ms; 0.921x | rework; backend proof failed |
| C3-EXP-006-I2-FAST | Corrected masked SDPA route is numerically safe | 47 focused/direct/model tests | 4.761 s | 47/47 passed, including OR-tolerance masks | reject after timing |
| C3-EXP-006-I2-ROW9 | Corrected route activates SDPA in production | final row 9 | 2.948 s | PASS; 0/5,242,880 failed; SDPA 112/112; 0.9613 ms | reject after confirmation |
| C3-EXP-006-I2-CONF-A | SDPA paired confirmation A | final row 9 | 2.923 s | PASS; SDPA 112/112; 0.9716 ms | reject |
| C3-EXP-005-I2-CONF-A | 32x32 Triton paired confirmation A | final row 9 | 3.033 s | PASS; Triton 112/112; 0.9049 ms | keep; review approved |
| C3-EXP-005-I2-CONF-B | 32x32 Triton counterbalanced confirmation B | final row 9 | 3.006 s | PASS; Triton 112/112; 0.9034 ms | keep; review approved |
| C3-EXP-006-I2-CONF-B | SDPA counterbalanced confirmation B | final row 9 | 2.898 s | PASS; SDPA 112/112; 0.9636 ms | reject |
| C3-INTEGRATE-001-FAST | Approved candidate survives integration | 42 focused/direct/model tests | 4.664 s | 42/42 passed | keep |
| C3-INTEGRATE-002-FINAL | Integrated candidate clears the final matrix | 14 published rows | 50.395 s | 13/13 executable PASS; 0/938,885,120 failed; one authorized skip; 1.555780x | keep |
| C3-INTEGRATE-003-DEFAULT | Untouched organizer default remains correct | exact default harness | 3.023 s | 5/5 PASS; 0/2,621,440 failed; Triton 1,950/1,950; 1.367x | keep |
| C3-INTEGRATE-004-HELDOUT | Candidate does not overfit visible rows | seven held-out cases | 3.114 s | 7/7 PASS; 0/13,117,440 failed; 1.220024x | keep |
| C3-INTEGRATE-005-SOURCE | Candidate preserves supplied-contract breadth | 29 source-derived entries | 62.229 s | 28/28 executable PASS; 0/459,776,000 failed; one authorized skip; 1.200529x | keep |
| C3-INTEGRATE-006-PROFILE-ROW9 | Intended Triton kernel caused the gain | row-9 ten-step profile | 3.353 s | Triton 40/40; `_attention_fwd` 3,018.182 us, -55.454% | keep |
| C3-INTEGRATE-007-CURATED-TESTS | Promoted evidence is current and fail-closed | curated artifact tests | 1.616 s | 7/7 passed; shared fingerprint and source hashes | keep |
| C3-CLOSURE-001-FULL-TESTS | Complete repository contract remains green | full pytest suite | 7.344 s | 104/104 passed; 14 upstream warnings | keep |
| C3-CLOSURE-002-FINAL-TESTS | Final docs and ledger remain green | final full pytest suite | 7.233 s | 104/104 passed; 14 upstream warnings | keep |

Campaign 3 contains 31 immutable attempt records. All 31 child commands passed
and none failed; this does not hide optimization outcomes: eight candidate
records are rejected and two are marked rework. Recorded child-command wall
time totals 218.214 seconds; orchestration, documentation, commits, and review
time are excluded.

## Baseline observation

The fresh row-9 run confirms a real target rather than relying on Campaign 2's
single curated value. It passed the strict comparator with maximum absolute
error 0.00106275, but optimized median latency was 1.2341 ms against a 1.1745 ms
baseline (0.952x). The matching ten-step profile recorded 40 Triton launches and
6,775.468 us of `_attention_fwd` self device time inside a 12,266.268 us model
range. Attention therefore accounts for 55.23% of the optimized range and
authorizes bounded EXP-005 launch testing.

The controlled backend screen used the same project matrix runner for auto,
forced SDPA, and forced reference. All three passed zero-failure correctness.
Optimized medians were 1.15082 ms for auto/Triton, 0.69224 ms for SDPA, and
1.00851 ms for exact reference. SDPA was 39.85% faster than auto and 31.36%
faster than exact reference. Separate-run baseline medians varied from 0.82642
to 0.93888 ms, so these screens authorize EXP-006 but do not accept it; the
routing candidate requires alternating confirmation and final-matrix evidence.

## Candidate decision

EXP-005 exhausted the three permitted launch variants without changing kernel
arithmetic. I2's 32x32 tile produced optimized medians of 0.9042, 0.9049, and
0.9034 ms. Its 0.9042 ms three-run median has a 0.0015 ms range (0.166%) and is
26.73% below the fresh 1.2341 ms baseline.

The first EXP-006 route passed 46 tests but missed production because the
untouched organizer creates an all-valid mask at zero padding; the attempt was
retained as rework rather than relabeled as success. The corrected route proved
112/112 SDPA calls and exact correctness, then measured 0.9613, 0.9716, and
0.9636 ms in the same counterbalanced sequence. EXP-005-I2 was 6.16% faster by
three-run medians, so the broader SDPA policy was rejected and never integrated.

An independent read-only reviewer recomputed the candidate implementation
fingerprint, verified attempt-to-result hashes and protected organizer-source
hashes, confirmed zero failures in all three row-9 results, and approved the
exact `head_dim == 128 and seq_len <= 128` 32x32 branch for integration. The
reviewer's boundary-hardening recommendation was also applied: stages and an
adjacent head dimension are asserted without changing production behavior.

## Integrated result and stopping decision

The final matrix remained 13/13 executable PASS plus the single authorized
resource skip, with 0/938,885,120 failed elements. Geometric-mean speedup rose
from 1.525823x to 1.555780x (+1.963%). The only final row inside the changed
envelope, row 9, improved from 1.2055 ms to 0.9071 ms (-24.75%) and from 0.945x
to 1.281x while retaining 112/112 Triton calls. Timing movement on other rows is
unaffected by the exact launch guard and is retained in the raw artifact.

The integrated profile supplies the alternative acceptance gate requested by
the workflow: `_attention_fwd` fell from 6,775.468 us to 3,018.182 us across 40
calls (-55.454%), and the ten-step optimized range fell from 12,266.268 us to
8,959.045 us (-26.962%). Attention's share fell from 55.23% to 33.69%.

EXP-007 was not run. The reserved projection/launch-count surface required new
profile proof before execution; after the accepted attention change, the
remaining leaders are vendor GEMMs and native LayerNorm, so replacing them would
be speculative and outside the bounded stop rule. The campaign therefore stops
with one accepted implementation rather than continuing until a number moves.

## Source-of-truth impact

`docs/REQUIREMENTS.md` remains authoritative and now records the exact short
head-128 launch threshold; the support envelope, arithmetic, public API, and
persisted state are unchanged. Kernel design, tests, technical report, result
index, README, Devpost copy, compliance matrix, demo runbook, organizer-input
audit, and optimization-plan summary were refreshed to the current fingerprint
`9071e3c049a7a3bc2311fc9d33997202ce4bead93d9daced375340fe6308eb9e`.

Final release review, repo-graph closure, and the accepted local checkpoint tag
remain pending; no push, public submission, or external release action is part
of this campaign.
