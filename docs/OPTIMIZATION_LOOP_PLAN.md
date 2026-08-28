# Bounded Agentic GPU Optimization Loop

Status: executed on 2026-08-28; EXP-001, EXP-003, and EXP-005-I2 accepted and rebaselined
Applies to: the PyTorch/Triton Transformer implementation in this repository

## Objective

Improve end-to-end Transformer performance and memory use on the target GPU while
preserving the organizer contract, exact correctness behavior, truthful dispatch
accounting, and reproducible evidence.

The loop is a controlled experiment program, not an open-ended request for agents
to keep changing code until a benchmark number looks better.

## Campaign outcome

- Phase 0 merged the current `origin/main`, repaired native-Windows evidence
  portability, froze the organizer-published 14-row final matrix, and established
  a clean green checkpoint.
- Profiling identified final row 10 (`head_dim=64`, sequence 128) as the material
  bottleneck: `_attention_fwd` consumed 30,324.486 us across 40 launches.
- EXP-001 changed only the short-head-dimension-64 launch policy from
  `BLOCK_M=64` / `BLOCK_N=128` to `BLOCK_M=32` / `BLOCK_N=64`. Two paired full
  matrix trials improved geometric-mean speedup by 8.98% and 10.19%.
- An independent reviewer approved the bounded implementation, including an
  explicit noise waiver for unaffected `head_dim=32` timing variation.
- Campaign 2 retained all 39 executed attempts: 35 child commands passed and
  four failed gates remain as evidence. Their measured child wall time totals
  255.273 seconds, with stdout/stderr, correctness, latency distributions,
  memory/backend/profiler data, artifact hashes, environment, and decisions.
  EXP-002 rejected all three long-`head_dim=32` variants; each was slower.
- EXP-004 stopped at its first direct compile gate because Triton dot requires
  `K >= 16`; the production `head_dim=8` exact-reference fallback remains.
- EXP-003 tested three short-`head_dim=32` tile policies. Alternating row-1
  confirmation selected the 64x64 tile at a 0.8201 ms mean versus the 1.2402 ms
  unchanged mean. Independent review approved the exact guarded branch.
- The accepted candidates were merged and every curated artifact was regenerated
  from implementation SHA-256
  `8eb7d21551ab69e83f532deaeefb2ce1999dc3e198f48a8d4be5753ad2c93a8a`.
  The integrated final matrix is 13/13 executable PASS plus one authorized
  resource skip, zero failed elements across 938,885,120 comparisons, and
  1.526x geometric-mean speedup, 6.95% above the post-EXP-001 matrix.
- Campaign 3 profiled final row 9, screened three bounded short-`head_dim=128`
  launch geometries, and counterbalanced the best Triton result against a
  shape-aware SDPA alternative. Independent review approved only the 32x32
  Triton tile through sequence 128.
- The accepted Campaign 3 candidate reproduced 0.9042/0.9049/0.9034 ms optimized
  medians, 26.73% below the fresh 1.2341 ms baseline and 6.16% faster than the
  SDPA alternative's three-run median. Integrated `_attention_fwd` time fell
  55.45%, and final geomean rose from 1.525823x to 1.555780x while all required
  correctness gates remained green.
- Campaign 3 curated artifacts share implementation SHA-256
  `9071e3c049a7a3bc2311fc9d33997202ce4bead93d9daced375340fe6308eb9e`.

The durable experiment records are
`docs/experiments/EXP-001-head64-short-tiles.md`,
`docs/experiments/EXP-003-short-head32-kv-tiles.md`, and
`docs/experiments/CAMPAIGN-002.md`, with Campaign 3 recorded in
`docs/experiments/CAMPAIGN-003.md`; rejected alternatives, failed gates, and
pre-integration paired evidence remain versioned for auditability.

## Source of truth and scope

Use these sources in this order:

1. The untouched organizer PyTorch harness and
   `benchmarks/reference/organizer_downloads.json`.
2. The 14-row final organizer shape table frozen in
   `benchmarks/final_evaluator_shapes.json`.
3. `docs/REQUIREMENTS.md` and the checked-in executable comparator.
4. Measured behavior on the target GPU.
5. `docs/KERNEL_DESIGN.md`, `docs/TECH_REPORT.md`, and README claims.

The benchmark harness, comparator, tolerance, timing protocol, and frozen source
files are not optimization targets. They may only change through an explicit
requirements decision and a separately reviewed update.

Current implementation context:

- The repository already has a fused online-softmax Triton attention kernel,
  guarded dispatch, packed-QKV inference, and target-GPU evidence.
- The organizer now publishes 14 final shape rows. Framework, dtype, padding,
  timing, tolerance, and backward policy remain unstated, so the final-shape
  runner records the selected PyTorch defaults as explicit assumptions.
- The target measurements are specific to the RTX 5070 Ti environment recorded in
  the result artifacts. Other GPUs must be treated as a separate measurement.
- The native-Windows LF/CRLF evidence-hash portability failure and the
  `origin/main` Colab-notebook merge conflict are Phase 0 gates and must remain
  resolved before candidate measurements are accepted.

## Non-negotiable principles

- Correctness is a hard gate; performance never justifies a numerical regression.
- Profile before proposing an optimization. Every hypothesis must name the
  bottleneck or measurement it is intended to improve.
- Preserve an immutable reference harness and compare against the same baseline
  on every candidate.
- Count the backend and kernel that actually ran. A framework fallback must not be
  reported as custom-kernel execution.
- Keep candidate changes isolated. Only the integration owner merges a winner.
- Record failures, OOMs, unsupported cases, skipped cases, and inconclusive runs;
  do not delete or weaken tests to obtain a green result.
- Keep a held-out shape set so the implementation does not overfit the visible
  final organizer matrix.
- Do not read credentials or `.env` files, and do not place tokens in commands,
  URLs, logs, or evidence artifacts.

## Agent roles and boundaries

Use the minimum number of agents that reduces latency or risk. A recommended cap
is four concurrent agents plus one integrator.

| Role | Primary responsibility | Write boundary |
| --- | --- | --- |
| Experiment coordinator/integrator | Freeze the baseline, rank candidates, own the accepted branch, and publish decisions | The integration branch only; never edits the organizer reference |
| Kernel ideator/implementer | Develop one bounded kernel, layout, tiling, or fusion hypothesis | An isolated branch/worktree per candidate |
| Performance analyst | Inspect traces, timing distributions, memory, launch counts, and statistical significance | Read-only during measurement; proposes experiments, not threshold changes |
| Correctness/release reviewer | Run or inspect all positive and negative gates, review fallback truthfulness and evidence provenance | Read-only review; may request rejection or rework |

Ideation agents should initially return hypotheses, expected gains, risks,
affected files, and validation commands. They should not all edit the same working
tree. Parallel work is allowed only when branches have disjoint candidate changes
or the interface is frozen.

## Iteration workflow

### Phase 0: Preflight

1. Confirm the branch, clean working tree, parent commit, Python/PyTorch/CUDA/
   Triton versions, GPU identity, and available disk memory.
2. Verify organizer download hashes and the reference manifest.
3. Verify the Windows line-ending/hash issue and resolve it before using test
   results as a release signal.
4. Check whether a final organizer matrix or revised benchmark has appeared. If
   so, update the requirements and regenerate the case manifest before tuning.
5. Create a named baseline checkpoint. Do not overwrite it.

### Phase 1: Establish the baseline

Run the same fixed commands for every candidate and capture:

- CPU contract and state-dict compatibility;
- direct Triton attention correctness;
- end-to-end correctness across dtype, mask, causal, boundary, seed, and scale
  cases;
- untouched organizer default and source-derived validation;
- raw CUDA-event timings with warmup and alternating order;
- peak allocation, backend counts, profiler kernel names, and implementation
  fingerprint.

If the baseline cannot pass its own gates, stop optimization and repair the
measurement or implementation first.

### Phase 2: Generate and rank hypotheses

Ask each ideator for a distinct hypothesis. Useful candidate surfaces include:

- attention tile sizes, warps, stages, and launch geometry;
- Q/K/V layout, contiguity, and projection packing;
- launch-count or intermediate-buffer reduction;
- output projection or FFN changes only when profiling shows material impact;
- shape-aware dispatch thresholds based on measured end-to-end data;
- low-precision paths only if the final evaluator actually requires them.

Rank candidates by expected end-to-end gain divided by correctness, memory,
portability, and maintenance risk. Do not spend cycles on ideas that have no
profile-supported bottleneck.

### Phase 3: Implement one candidate

Each candidate must have:

- a unique experiment ID and parent commit;
- a single stated hypothesis;
- an isolated branch/worktree;
- no changes to organizer source, comparator, tolerance, or benchmark timing;
- a rollback path and a list of expected dispatch/evidence changes.

Do not combine unrelated optimizations in one candidate. A combined candidate may
be evaluated later after its components are understood independently.

### Phase 4: Fast rejection gate

Before an expensive GPU run, check:

- Python compilation and JSON validity;
- CPU correctness and strict state-dict loading;
- dispatch positive/negative tests;
- no accidental dense `[B,H,S,S]` or causal-mask allocation in the custom path;
- no changed organizer bytes or protected harness definitions;
- no new untracked evidence being mistaken for committed evidence.

Reject immediately on a hard failure.

### Phase 5: Full candidate measurement

Use the authoritative runners without changing their policy. For each candidate:

1. Run correctness before timing.
2. Run every current required case and the held-out cases.
3. Use multiple seeds/trials and the existing warmup, repeat, and alternating-order
   protocol.
4. Capture raw samples, median and distribution statistics, memory, backend
   counts, profiler events, environment, commit, and implementation hash.
5. Classify every case as `PASS`, `FAIL`, `OOM`, `ERROR`, or an explicitly
   authorized resource skip.

Microbenchmarks may explain a result but cannot accept a candidate by themselves.

### Phase 6: Independent review and decision

The reviewer compares the candidate with the frozen baseline and checks:

- zero correctness failures under the strict executable tolerance;
- no new fallback, unsupported-shape, compile, or memory failure;
- actual custom-kernel execution where claimed;
- statistically credible end-to-end improvement rather than timing noise;
- no important-case regression or unacceptable memory/compile-time cost;
- evidence and documentation that describe the measured candidate, not a stale
  checkout.

The integrator then records one decision: `keep`, `reject`, `rework`, or
`inconclusive`. A rejected candidate is not silently reused in later claims.

### Phase 7: Integrate and rebaseline

Only an accepted candidate is merged into the integration branch. Regenerate the
implementation fingerprint and curated evidence after integration, then make the
new commit the next baseline. If the implementation-fingerprinted code changes,
old performance evidence is no longer current until rerun.

## Acceptance gates

The following are release gates, not suggestions:

1. **Correctness:** zero failed elements, NaN/Inf mismatches, or unauthorized
   skips across the current matrix, held-out cases, and direct-kernel tests.
2. **Contract:** strict state-dict compatibility, unchanged organizer harness,
   truthful `auto`/`triton`/`sdpa`/`reference` dispatch, and fail-closed error
   accounting.
3. **Performance:** compare end-to-end medians and distributions. As a team
   optimization gate, require approximately 5% geometric-mean improvement or a
   clearly measured memory/launch benefit, with no required case regressing by
   more than 2% without an explicit review decision. These thresholds do not
   replace organizer rules.
4. **Evidence:** raw timings, environment, GPU, driver/runtime, backend counts,
   profiler kernel names, source hashes, and implementation fingerprint are
   captured from the candidate being evaluated.
5. **Maintainability:** the change has a documented support envelope, fallback,
   reason for its threshold/configuration, and no unnecessary dependency or API
   change.

## Loop limits and stopping rules

- Maximum five iterations for one hypothesis before escalation or abandonment.
- Stop a subsystem after three consecutive candidates fail to produce a
  meaningful end-to-end or memory improvement.
- Stop when profiler data shows the subsystem is no longer material.
- Stop and rebaseline if measurements become unstable or the environment changes.
- Stop current tuning if the organizer revises the final shape table or benchmark;
  reconcile the revision and rerun the complete gate first.
- Stop when remaining work is only speculative, benchmark-specific, or below the
  measurement noise floor.
- A human must approve any tolerance, harness, dependency, source-integrity, or
  public-submission change.

## Experiment record

Every candidate should produce a compact record containing:

```text
experiment_id
parent_commit and candidate_commit
hypothesis and targeted profiler bottleneck
changed paths and support-envelope impact
environment/GPU/runtime identifiers
commands and case manifest
correctness counts and failures
baseline/candidate raw timing summaries
geomean, per-case speedups, peak memory, and compile overhead
actual backend/kernel counts and profiler proof
regressions, OOMs, errors, and skips
reviewer decision and reason
```

All Campaign 2 and later commands must be executed through
`benchmarks/run_optimization_attempt.py`. The versioned record additionally
captures UTC start/end timestamps, command wall time, exit/timeout state,
stdout/stderr, accuracy/error totals, latency distributions and sample counts,
memory, backend/profiler counts, environment, artifact hash, Git state, and the
logger hash. A nonzero command, timeout, missing artifact, or invalid artifact
must still produce a record. A later review decision may be stored in a linked
immutable sidecar, but measured fields are never hand-edited.

Keep rejected and inconclusive records. They prevent the agents from repeating
the same low-value idea and make the final technical report honest.

## Recommended first campaign for this repository

1. Repair the native-Windows artifact hash portability problem and resolve the
   branch/main notebook conflict.
2. Reconfirm the clean baseline and its evidence fingerprint.
3. Profile the full Transformer to identify the largest remaining end-to-end
   contributor after packed QKV and fused attention.
4. Run at most three independent candidates against that bottleneck: one kernel
   configuration/layout candidate, one launch/intermediate-reduction candidate,
   and one dispatch or projection candidate.
5. Validate each candidate on the current and held-out shapes before considering
   any broader fusion.
6. Keep the best accepted candidate, reject the rest, and publish refreshed
   evidence only after integration.

Avoid forcing Triton everywhere, changing tolerances to save a candidate,
blanket-enabling `torch.compile`, or reviving standalone LayerNorm fusion without
new profile evidence. The existing measurements already indicate that some of
those directions are slower or less observable.

## External holds

This optimization loop cannot complete the following submission tasks by itself:

- obtaining clarification for the final table's unstated dtype, padding, timing,
  tolerance, framework, and backward policy;
- making the repository publicly reachable and verifying it signed out;
- recording/uploading the public demo and publishing the Devpost entry.

These remain explicit release holds even if every local optimization gate passes.

## Final decision rule

The campaign is complete when the accepted candidate has passed the full evidence
gate, no unresolved correctness or provenance finding remains, the branch is
mergeable, and further candidates are below the agreed improvement threshold.
“More iterations” is not evidence of progress; a measured, reproducible plateau
is a valid stopping result.
