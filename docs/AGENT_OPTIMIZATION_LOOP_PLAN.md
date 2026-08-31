# Bounded Agentic GPU Optimization Loop

Status: stopped and consolidated after Campaign 11 on 2026-08-30
Applies to: the PyTorch/Triton Transformer implementation in this repository

For the reader-facing summary, start with the [documentation hub](README.md).
This plan records the experiment controls and stopping rules; campaign outcomes
belong in the [campaign run-through](experiments/CAMPAIGN_RUN_THROUGH.md) and
the [optimization history](experiments/OPTIMIZATION_HISTORY.md).

## Objective

Improve end-to-end Transformer performance and memory use on the target GPU while
preserving the organizer contract, exact correctness behavior, truthful dispatch
accounting, and reproducible evidence.

The loop is a controlled experiment program, not an open-ended request for agents
to keep changing code until a benchmark number looks better.

## Campaign outcome

The loop completed Campaign 11's candidate, integration, review, documentation,
workflow, and graph gates. The current implementation passed all
13 executable final rows with zero failed elements across 938,885,120
comparisons; the exact 100,000-token resource row remains an authorized non-pass
skip. The current primary and confirmation geomeans are 1.977420x and 1.986499x.
Campaign 12 was drafted but stopped before preflight, profiling, benchmarking,
or source changes; its unused scaffold was removed and it has no campaign result.

| Stage | Accepted change | Final geomean after integration |
| --- | --- | ---: |
| EXP-001 | short `head_dim=64`: 32x64 tiles | 1.426692x |
| Campaign 2 / EXP-003 | short `head_dim=32`: 64x64 tiles | 1.525823x |
| Campaign 3 / EXP-005-I2 | short `head_dim=128`: 32x32 tiles | 1.555780x |
| Campaign 4 / EXP-009-I2 | exact row-11 `head_dim=8`: padded-width 64x64 Triton | 1.780075x; 1.784920x confirmation |
| Campaign 5 / EXP-010-I3 | exact row 7: layer 0 reference, layers 1-3 Triton | 1.911947x composite; row 7 1.524x |
| Campaign 5 / EXP-011-I2 | exact row 6: layers 0-1 reference, layers 2-3 Triton | same composite; row 6 1.503x |
| Campaign 5 / EXP-012-I1 | exact held-out long-causal SDPA route | 1.447477x held-out; prior 0.798x/0.878x targets became 1.247x/1.280x |
| Campaign 6 / EXP-015-I1R | exact-width-1024 packed QKV | 1.872916x current composite; row-8 profile time -7.91% versus same-window control |
| Campaign 7 / EXP-018-I2R | exact-row-6 fused residual plus LayerNorm | 1.880620x current composite; row-6 model profile -9.54%, 100-sample speedup 1.417x control to 1.547x candidate, no peak-memory increase |
| Campaign 8 / EXP-019-I1R | exact-row-11 fused residual plus LayerNorm | 1.876167x current composite; retained row-11 median -9.70%, integrated model profile -21.96%, no peak-memory increase |
| Campaign 10 / EXP-023-I1 | exact-row-5 fused residual plus LayerNorm | 1.926716x current composite; retained row-5 median -11.58%, integrated model profile -11.96%, 2.001995x over 300 samples |
| Campaign 11 / EXP-025-I1 | exact-row-9 fused residual plus LayerNorm | 1.977420x current composite; controlled optimized median -12.05%, mean residual/normalization profile time -41.77%, memory-neutral |

Campaigns 2-11 retain immutable attempt records for every passing and failed
child command; each campaign's exact count and wall time are recorded in its
ledger after closure.
The canonical [campaign run-through and flagship ranking](experiments/CAMPAIGN_RUN_THROUGH.md)
owns the executive narrative. The [complete optimization history](experiments/OPTIMIZATION_HISTORY.md)
owns the detailed chronology, metrics, route table, and evidence index. The
individual campaign records remain immutable-detail appendices.

## Source of truth and scope

Use these sources in this order:

1. The untouched organizer PyTorch harness and
   `benchmarks/reference/organizer_downloads.json`.
2. The 14-row final organizer shape table frozen in
   `benchmarks/final_evaluator_shapes.json`.
3. `docs/REQUIREMENTS.md`, `hackathon-docs/hackathon-details.md`, and the checked-in executable comparator.
4. Measured behavior on the target GPU.
5. `docs/KERNEL_DESIGN.md`, `docs/IMPLEMENTATION_EVIDENCE.md`, and README claims.

The benchmark harness, comparator, tolerance, timing protocol, and frozen source
files are not optimization targets. They may only change through an explicit
requirements decision and a separately reviewed update.

Current implementation context:

- The repository already has a fused online-softmax Triton attention kernel,
  exact-row-5, exact-row-6, exact-row-9, and exact-row-11 fused residual/LayerNorm routes, guarded dispatch, packed-QKV
  inference, and target-GPU evidence.
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
the same low-value idea and make the final submission report honest.

## Most recent campaign

1. Campaign 11 started from selected Campaign 10 fingerprint
   `f7ad2a86a68f95736241ddde992500073ee75738982af4a81c0c658cd64538d4`.
2. Fresh row-9 profiling exposed a 19.10% residual/normalization ceiling. A
   bounded exact-row-9 reuse of the accepted fused forward passed direct,
   neighbor, 18-scenario stress, long-run, memory, repeated-profile, final,
   held-out, source-derived, organizer-default, inherited-route, and suite gates.
3. Two unchanged controls averaged 0.815968 ms. The active 300-sample optimized
   median is 0.717648 ms (-12.05%), within 0.007% of the isolated candidate,
   while repeated profiles reduce mean subsystem time 41.77%. Top-level
   profiler time is noisy and is disclosed rather than used as causal proof.
4. The campaign-closing fingerprint is
   `9c326536ea27cfc619f01531152b2c82986d9dc3f4274691d3e8191bbb0804eb`;
   the packaged Campaign 11 evidence fingerprint is
   `908a0d708cd8f70f44d5f14fda93d3cafb1cc18345f43914e715594cfa7b7ef9`.
5. EXP-026 row-10 fusion was deliberately unrun because EXP-025 produced an
   accepted winner. The next loop must profile a materially different surface
   and must not retry Campaign 9's width-1024/row-13 arithmetic or Campaign
   11's conditional row-10 fallback without a new campaign and fresh evidence.

Avoid forcing Triton everywhere, changing tolerances to save a candidate,
blanket-enabling `torch.compile`, or reviving standalone LayerNorm fusion without
new profile evidence. The existing measurements already indicate that some of
those directions are slower or less observable.

## External holds

This optimization loop cannot complete the following submission tasks by itself:

- obtaining clarification for the final table's unstated dtype, padding, timing,
  tolerance, framework, and backward policy;
- recording/uploading the public demo and publishing the Devpost entry.

Anonymous access to the GitHub URL was verified on 2026-09-01. It must still be
rechecked after local changes are published and immediately before submission.
The organizer clarifications and public video/Devpost work remain explicit
release holds even if every local optimization gate passes.

## Final decision rule

The campaign is complete when the accepted candidate has passed the full evidence
gate, no unresolved correctness or provenance finding remains, the branch is
mergeable, and further candidates are below the agreed improvement threshold.
“More iterations” is not evidence of progress; a measured, reproducible plateau
is a valid stopping result.
