# CAMPAIGN-003: Logged post-Campaign-2 optimization round

Status: in progress
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

Campaign 3 currently contains five passing attempt records and zero failed
commands. Recorded child-command wall time totals 11.184 seconds; orchestration,
documentation, commits, and review time are excluded.

## Baseline observation

The fresh row-9 run confirms a real target rather than relying on Campaign 2's
single curated value. It passed the strict comparator with maximum absolute
error 0.00106275, but optimized median latency was 1.2341 ms against a 1.1745 ms
baseline (0.952x). The matching ten-step profile recorded 40 Triton launches and
6,775.468 us of `_attention_fwd` self device time inside a 12,266.268 us model
range. Attention therefore accounts for 55.23% of the optimized range and
authorizes bounded EXP-005 launch testing.

## Source-of-truth impact

`docs/REQUIREMENTS.md` remains authoritative. Update it only if accepted behavior
or the support envelope changes; otherwise record an explicit no-change decision.
Kernel design, tests, technical report, result index, README, demo runbook, and
optimization-plan summaries must be refreshed if accepted code or evidence changes.
