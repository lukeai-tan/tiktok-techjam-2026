# Track 3 documentation hub

Status: current on `feat/jared-attempt`, reconciled through Campaign 11 closure
on 2026-08-30.

This page is the short map for the repository. Read the executive documents for
the conclusion, then follow the evidence links when a number or decision needs
to be audited. The detailed campaign ledgers and JSON artifacts remain
authoritative; this hub does not replace them.

The canonical prose is organized below this directory by purpose: contract and
design documents stay at the stable `docs/` paths used by the campaign records,
campaign narratives live in `docs/experiments/`, the demo guide is in
`docs/guides/`, and the artifact index is next to the curated results. The root
`README.md` remains the repository landing page. The canonical demo guide is
[`docs/guides/DEMO_RUNBOOK.md`](guides/DEMO_RUNBOOK.md); there is no root-level
compatibility copy. Submission-facing files live under
[`deliverables/`](../deliverables/):
[`deliverables/03_TECHNICAL_REPORT.md`](../deliverables/03_TECHNICAL_REPORT.md)
is the submission technical report, while
[`docs/IMPLEMENTATION_EVIDENCE.md`](IMPLEMENTATION_EVIDENCE.md) is its detailed
repository evidence reference. Hackathon context, organizer inputs, and Devpost
prose live under [`hackathon-docs/`](../hackathon-docs/). Only
[`docs/TRACK3_COMPLIANCE.md`](TRACK3_COMPLIANCE.md) remains as a legacy pointer
because older artifacts still link to that path.

Campaign 11 performance evidence is frozen at measured fingerprint
`908a0d708cd8f70f44d5f14fda93d3cafb1cc18345f43914e715594cfa7b7ef9`.
The validation-hardened tree is
`a186b679885e9e787b3deba0ad710855ae4c2486ae491b53e4e64bfa13e7f9cf`;
optimized-model behavior did not change, and historical timings are not relabeled.

## Current answer at a glance

| Question | Current answer | Evidence |
| --- | --- | --- |
| What is the selected entry? | `transformer_opt/submission.py::UserOptimizedTransformer` | [requirements](REQUIREMENTS.md) |
| Which implementation is the flagship? | Campaign 11, fingerprint `908a0d708cd8f70f44d5f14fda93d3cafb1cc18345f43914e715594cfa7b7ef9` | [campaign ranking](experiments/CAMPAIGN_RUN_THROUGH.md#flagship-and-strongest-specialist-campaigns) |
| Does it pass the published final matrix? | 13/13 executable rows PASS; 0 failed elements across 938,885,120 comparisons; one authorized resource skip | [final artifact](results/rtx-5070-ti-2026-08-29-c11-integrated-final.json) |
| How fast is the final matrix? | 1.977420x primary geomean; 1.986499x confirmation | [result index](results/README.md#current-selected-submission-run) |
| Does it generalize? | Two held-out 7/7 PASS runs at 1.339847x and 1.386495x; long-causal stays in the 1.198x–1.204x band | [held-out artifacts](results/README.md#current-selected-submission-run) |
| Is the repository green? | 164/164 tests PASS; 14 upstream warnings | [submission validation](experiments/SUBMISSION_VALIDATION.md#current-campaign-11-measured-suite) plus current maintenance validation |

## Where to read next

| If you want to… | Read this first | Then use |
| --- | --- | --- |
| Understand the result and the best campaigns | [Campaign run-through](experiments/CAMPAIGN_RUN_THROUGH.md) | [Campaign ledgers](experiments/) |
| Inspect every metric, attempt, and disposition | [Optimization history](experiments/OPTIMIZATION_HISTORY.md) | [`attempts/`](experiments/attempts/) and [`results/`](results/) |
| Reproduce the checks | [Root README](../README.md#reproduce) | [Demo runbook](guides/DEMO_RUNBOOK.md) and [result commands](results/README.md) |
| Understand the complete code and harness flow | [Code flow](CODE_FLOW.md) | [Kernel design](KERNEL_DESIGN.md) and [implementation evidence](IMPLEMENTATION_EVIDENCE.md) |
| Understand the kernel implementation | [Kernel design](KERNEL_DESIGN.md) | [Code flow](CODE_FLOW.md) and [implementation evidence](IMPLEMENTATION_EVIDENCE.md) |
| Confirm the contract and assumptions | [Requirements](REQUIREMENTS.md) | [Organizer inputs](../hackathon-docs/ORGANIZER_INPUTS.md) and [hackathon details](../hackathon-docs/hackathon-details.md) |
| Compare optimized code with the original | [Current vs original snapshot](experiments/OPTIMIZATION_HISTORY.md#current-optimized-versus-original-snapshot) | [Current result artifacts](results/README.md) |
| Review the other branch/PR comparison | [Branch implementation comparison](experiments/BRANCH_IMPLEMENTATION_COMPARISON.md) | Its linked immutable attempt and result files |
| Check submission readiness and open holds | [Track 3 compliance](../hackathon-docs/TRACK3_COMPLIANCE.md) | [Selected submission validation](experiments/SUBMISSION_VALIDATION.md) |
| Understand how optimization was controlled | [Bounded optimization loop](AGENT_OPTIMIZATION_LOOP_PLAN.md) | Campaign ledgers and review sidecars |
| Prepare the public explanation | [Submission technical report](../deliverables/03_TECHNICAL_REPORT.md) | [Devpost draft](../hackathon-docs/DEVPOST_DESCRIPTION.md) and [demo runbook](guides/DEMO_RUNBOOK.md) |

## Flagship and strongest specialists

These labels separate the current cumulative implementation from historical or
shape-specific wins. They are not interchangeable claims.

| Label | Campaign | Why it matters | Boundary |
| --- | --- | --- | --- |
| Overall flagship | Campaign 11 | Selected measured fingerprint, latest complete zero-failure final pair, and the current 164-test gate | Timings are specific to the recorded RTX 5070 Ti and published assumptions |
| Broad architecture/generalization | Campaign 5 | Added the accuracy-safe row-6/row-7 hybrids and removed both held-out long-causal regressions | Historical snapshot, not the current submission fingerprint |
| High-volume specialist | Campaign 7 | Fused residual plus LayerNorm on exact row 6, the large-batch latency surface | Shape-specific result |
| Strongest single-row lineage | Campaign 4 + Campaign 8 | Made width-eight attention legal, then fused exact row-11 residual/normalization work | The final gain is cumulative; it is not attributable to one campaign alone |

For the full ranking, trade-offs, and rejected alternatives, use the
[campaign run-through](experiments/CAMPAIGN_RUN_THROUGH.md#flagship-and-strongest-specialist-campaigns).

## Campaign map

| Phase | Focus | Outcome | Detail |
| --- | --- | --- | --- |
| Foundation / short-head tiles | Short `head_dim=64` attention | Accepted 32x64 tiles | [Campaign run-through](experiments/CAMPAIGN_RUN_THROUGH.md#foundational-phase-from-prototype-to-a-defensible-baseline) |
| Campaign 2 | Short `head_dim=32` and the first width-eight probe | Accepted short-head tiles; width-eight remained a probe | [Campaign 2 ledger](experiments/CAMPAIGN-002.md) |
| Campaign 3 | Short `head_dim=128` | Accepted profiler-backed launch geometry | [Campaign 3 ledger](experiments/CAMPAIGN-003.md) |
| Campaign 4 | Exact row-11 `head_dim=8` | Accepted padded internal dot width | [Campaign 4 ledger](experiments/CAMPAIGN-004.md) |
| Campaign 5 | Exact row-6/row-7 hybrids and long-causal routing | Accepted accuracy-safe hybrids and SDPA recovery | [Campaign 5 ledger](experiments/CAMPAIGN-005.md) |
| Campaign 6 | Exact-width-1024 projections | Accepted packed QKV in the measured envelope | [Campaign 6 ledger](experiments/CAMPAIGN-006.md) |
| Campaign 7 | Row-6 residual/LayerNorm; wide-head probe | Accepted row-6 fusion; rejected width-256 attention | [Campaign 7 ledger](experiments/CAMPAIGN-007.md) |
| Campaign 8 | Row-11 residual/LayerNorm | Accepted exact-row reuse of the guarded fusion | [Campaign 8 ledger](experiments/CAMPAIGN-008.md) |
| Campaign 9 | Row-8/row-13 fusion alternatives | No winner; unsuitable variants were closed | [Campaign 9 ledger](experiments/CAMPAIGN-009.md) |
| Campaign 10 | Row-5 residual/LayerNorm | Accepted exact-row fusion | [Campaign 10 ledger](experiments/CAMPAIGN-010.md) |
| Campaign 11 | Row-9 residual/LayerNorm | Accepted current flagship integration | [Campaign 11 ledger](experiments/CAMPAIGN-011.md) |

Campaign 12 was drafted and stopped before preflight, profiling, benchmarking,
or source changes; it has no result or accepted implementation. The detailed
run-through preserves candidate-level decisions, while the optimization history
preserves the complete chronology and attempt accounting.

## How the evidence is organized

1. **Contract:** `benchmarks/reference/`, `benchmarks/final_evaluator_shapes.json`,
   [requirements](REQUIREMENTS.md), and the untouched organizer harness define
   what may be measured.
2. **Curated results:** [`results/`](results/) contains versioned JSON with raw
   timings, accuracy counts, backend counts, memory, environment, and source
   fingerprints. [Its README](results/README.md) explains which artifact answers
   each question.
3. **Attempt records:** [`experiments/attempts/`](experiments/attempts/) logs
   every wrapped command, including failures, skips, OOMs, inconclusive screens,
   duration, and provenance. Do not treat a passing command as an accepted
   optimization without its campaign disposition.
4. **Decision records:** [`experiments/CAMPAIGN-*.md`](experiments/) and
   [`experiments/reviews/`](experiments/reviews/) explain why candidates were
   accepted, rejected, reworked, or deliberately left unrun.
5. **Summaries:** the [campaign run-through](experiments/CAMPAIGN_RUN_THROUGH.md)
   is the readable outcome; the [optimization history](experiments/OPTIMIZATION_HISTORY.md)
   is the denser chronology and accounting layer.

## Validation path

Run the commands in the [root README](../README.md#reproduce) in this order:

1. CPU and GPU tests;
2. the untouched organizer default;
3. the published final matrix;
4. the source-derived matrix and held-out matrix;
5. the profiler cases that prove the expected backend and fused routes.

The [demo runbook](guides/DEMO_RUNBOOK.md) turns that sequence into a short narrated
walk-through. A fresh run must be tied to the current implementation fingerprint;
do not copy a prior number into a new claim.

## Open holds

- The published final table still omits dtype, padding, timing, tolerance,
  backward policy, and confirmation that live attachment bytes are unchanged.
- Public repository verification and the YouTube/Devpost publication steps are
  external human actions.
- Performance is measured on the recorded RTX 5070 Ti. A different evaluator
  GPU requires a fresh run and does not inherit these speedups.

See [Track 3 compliance](../hackathon-docs/TRACK3_COMPLIANCE.md) for the current PASS/HOLD matrix.

## Reading convention

Use the shortest document that answers the question. Do not duplicate a measured
value in a new summary unless the source artifact and implementation fingerprint
are linked. Keep failed and rejected evidence in the audit tree; consolidation
means clearer pointers and less repeated prose, not deletion of the experiment
record.
