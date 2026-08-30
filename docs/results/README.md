# Result artifacts

Use the [documentation hub](../README.md) to choose the right report first.
This page owns artifact purpose, provenance, and reproduction commands; it is
not a second campaign narrative.

Curated evidence in this directory is intentionally versioned; scratch traces
and exploratory runs stay under ignored `results/`.

Campaign 11 performance artifacts remain immutable at measured fingerprint
`908a0d708cd8f70f44d5f14fda93d3cafb1cc18345f43914e715594cfa7b7ef9`.
The current validation-hardened tree is
`a186b679885e9e787b3deba0ad710855ae4c2486ae491b53e4e64bfa13e7f9cf`; it changes
profiler/evidence enforcement and corrects one source comment, but does not
change optimized math or dispatch behavior. Fresh
maintenance outputs belong under ignored `results/` unless a later evidence
campaign deliberately curates them. Do not rewrite the Campaign 11 JSON.

Use [the campaign run-through](../experiments/CAMPAIGN_RUN_THROUGH.md) for the
executive flagship ranking and [the complete optimization history](../experiments/OPTIMIZATION_HISTORY.md)
for chronology, attempt totals, and candidate decisions. This index owns
artifact purpose and reproduction.

## Fast lookup

| Need | Start with |
| --- | --- |
| Published final correctness and speed | [`c11-integrated-final.json`](rtx-5070-ti-2026-08-29-c11-integrated-final.json) |
| Independent final confirmation | [`c11-integrated-final-confirmation.json`](rtx-5070-ti-2026-08-29-c11-integrated-final-confirmation.json) |
| Held-out generalization and long-causal stability | [held-out artifacts](#current-selected-submission-run) |
| Untouched organizer default | [`c11-integrated-organizer-default.json`](rtx-5070-ti-2026-08-29-c11-integrated-organizer-default.json) |
| Source-derived contract breadth | [`c11-integrated-source-derived.json`](rtx-5070-ti-2026-08-29-c11-integrated-source-derived.json) |
| Proof that fused routes actually ran | [row-5, row-6, row-9, and row-11 profiles](#current-selected-submission-run) |
| Every command, including failures and timing | [`experiments/attempts/`](../experiments/attempts/) |

## Current selected-submission run

**Flagship measured snapshot:** Campaign 11. Campaign 5 remains the highest
historical aggregate, but Campaign 11 is the selected cumulative performance
evidence. The maintenance fingerprint above has separate current validation.

All artifacts in this section independently record schema-2 implementation
SHA-256
`908a0d708cd8f70f44d5f14fda93d3cafb1cc18345f43914e715594cfa7b7ef9`.
The actual entry is `transformer_opt/submission.py::UserOptimizedTransformer`.

- `rtx-5070-ti-2026-08-29-c11-integrated-final.json`: primary organizer-published
  final matrix. All 13 executable rows passed with zero failed elements across
  938,885,120 comparisons; the exact 100,000-token resource case is a separate
  authorized non-pass skip. Geomean is 1.977420x; row 5 is 2.314x, row 8 is
  1.097x, newly fused row 9 is 1.780x, row 11 is 6.377x, and backend accounting
  is Triton 1,260 / SDPA 0 / reference 196. Dedicated row-5, row-6, row-9, and
  row-11 long runs supplement the complete-matrix snapshots.
- `rtx-5070-ti-2026-08-29-c11-integrated-final-confirmation.json`: complete
  confirmation with the same correctness and backend counts. Geomean is
  1.986499x; aggregate geomeans differ by 0.459%.
- `rtx-5070-ti-2026-08-29-c11-integrated-heldout-5seed.json` and
  `rtx-5070-ti-2026-08-29-c11-integrated-heldout-5seed-confirmation.json`: two complete
  seven-case held-out runs, five accuracy seeds per case, raw alternating-order
  CUDA-event samples, backend counts, memory, and environment. Both are 7/7
  PASS with zero failed elements across 13,117,440 comparisons per run;
  geomeans are 1.339847x and 1.386495x. Two additional measured-fingerprint
  rechecks measure 1.384122x and a noisy 1.515376x. Across all four matrices,
  long-causal stays in a narrow 1.198x-1.204x band while padded long-causal
  spans 1.213x-1.335x. `rtx-5070-ti-2026-08-29-c11-integrated-long-causal-long.json`
  adds 300 optimized samples at 1.198052x and 620 SDPA calls.
- `rtx-5070-ti-2026-08-29-c11-integrated-row09-profile.json` plus its active
  confirmation versus two Campaign 11 baseline profiles: each active profile
  records 240 fused launches, 30 remaining native norms, and 120 Triton calls.
  Mean residual/normalization device time falls from 5,765.324 us to 3,357.389
  us (-41.77%). Mean top-level model profiler time rises 2.54% because one
  snapshot is slow, so it is disclosed as variance rather than causal evidence.
  The counterbalanced 300-sample CUDA-event gate is the performance decision:
  two Campaign 10 controls average 0.815968 ms, while
  `rtx-5070-ti-2026-08-29-c11-integrated-row09-long.json` measures 0.717648 ms
  (-12.05%), 1.150046x baseline-to-optimized, zero failed elements, and the same
  29,360,128-byte incremental peak as both controls and the isolated candidate.
- `rtx-5070-ti-2026-08-29-c11-integrated-row05-profile.json` versus
  `rtx-5070-ti-2026-08-29-c10-baseline-row05-profile.json`: the new exact-row-5
  route replaces 240 residual adds and 240 of 270 native norms with 240 fused
  launches over 30 forwards. Current residual/normalization device time falls
  from 12,089.646 us to 7,016.653 us (-41.96%). The current top-level profiler
  snapshot is an outlier and is not used as a speed claim. The long gate
  `rtx-5070-ti-2026-08-29-c11-integrated-row05-long.json` keeps optimized median
  at 1.163168 ms, zero failures, 1.880066x, and a 58,720,256-byte peak; the lower
  speedup ratio versus Campaign 10 comes from baseline-window drift.
- `rtx-5070-ti-2026-08-29-c11-integrated-row06-profile.json` versus
  `rtx-5070-ti-2026-08-29-c7-baseline-row06-profile.json`: the accepted fused residual
  plus LayerNorm route replaces 80 residual-add and 80 of 90 native-LayerNorm
  calls with 80 `_residual_layer_norm_fwd` launches over ten forwards. Combined
  residual/normalization device time falls from 486,023.333 us to 317,375.671 us
  (-34.70%), while model device time falls from 2,026,089.666 us to 1,884,493.452
  us (-6.99%). `rtx-5070-ti-2026-08-29-c11-integrated-row06-long.json` records 100
  samples at 1.546330x with the same 11,802,787,840-byte peak.
- `rtx-5070-ti-2026-08-29-c11-integrated-row11-profile.json` versus
  `rtx-5070-ti-2026-08-29-c8-baseline-row11-profile-e.json`: over 30 forwards,
  240 residual adds and 240 of 270 native norms become 240 fused launches.
  Combined residual/normalization time falls from 5,978.920 us to 3,220.674 us
  (-46.13%), and model device time falls from 41,211.814 us to 32,109.116 us
  (-22.09%). `rtx-5070-ti-2026-08-29-c11-integrated-row11-long.json` records 300
  samples at 0.890672 ms, 4.710116x, zero failed elements, and unchanged
  29,360,128-byte optimized incremental peak.
- `rtx-5070-ti-2026-08-29-c11-integrated-row06-profile.json` proves the accepted
  row-6 hybrid with 20 Triton plus 20 reference calls over ten forwards.
- `rtx-5070-ti-2026-08-29-c11-integrated-row08-profile.json` proves the retained
  exact-row-8 reference-attention route on the measured fingerprint. The historical
  `rtx-5070-ti-2026-08-29-c6-integrated-row08-profile.json` versus
  `rtx-5070-ti-2026-08-29-c6-baseline-row08-profile-c.json`: same-window
  profiler proof for exact-width packed QKV. Across ten forwards, `aten::addmm`
  calls fell from 240 to 160, `addmm` device time fell from 106,065.035 us to
  94,048.315 us (-11.33%), and optimized-model device time fell from
  150,050.615 us to 138,182.163 us (-7.91%). Both profiles correctly record
  reference attention, so the generic custom-attention profile status is
  `INCONCLUSIVE`; the comparison proves the projection change, not Triton use.
  The accepted candidate also raises pre-forward allocated memory by 50,380,800
  bytes for four packed width-1024 layers. The measured optimized incremental
  activation peak remains 369,115,136 bytes in both control and candidate.
- `rtx-5070-ti-2026-08-28-final-10-profile.json`: historical Campaign 2
  profiler proof for the EXP-001 target. `_attention_fwd` used 2,694.679 us
  across 40 Triton calls, down 91.11% from the frozen pre-candidate profile.
- `rtx-5070-ti-2026-08-28-final-01-profile.json`: historical Campaign 2
  profiler proof for the EXP-003 target. `_attention_fwd` used 2,103.978 us
  across 40 Triton calls, down 69.98% from the pre-EXP-003 curated profile.
- `rtx-5070-ti-2026-08-29-c11-integrated-organizer-default.json`: the untouched downloaded
  PyTorch harness with only the submitted `UserOptimizedTransformer` injected.
  The organizer-default six-layer case passed 5/5 trials with zero failed
  elements and measured a 1.385x median speedup; all 1,950 optimized attention
  calls used Triton.
- `rtx-5070-ti-2026-08-29-c11-integrated-source-derived.json`: 29 source-derived entries
  executed through the untouched PyTorch harness in isolated processes. All 28
  feasible cases passed with 0/459,776,000 failed elements; the supplied
  TensorFlow benchmark's designated 100,000-token resource skip is recorded
  separately and is not counted as a pass. Overall geomean is 1.206505x and
  aggregate backend counts are Triton 672 / SDPA 1,344 / reference 2,688.

`rtx-5070-ti-2026-08-28-exp001-*.json` files are the immutable paired and
candidate evidence used for the acceptance decision. They are historical
pre-integration evidence, not substitutes for the curated post-merge artifacts.
The decision and noise waiver are recorded in
`docs/experiments/CAMPAIGN_RUN_THROUGH.md` and
`docs/experiments/OPTIMIZATION_HISTORY.md`.

`rtx-5070-ti-2026-08-28-c2-*.json` files and the matching records under
`docs/experiments/attempts/` retain every Campaign 2 observation, candidate,
rejection, confirmation, failed gate, and rebaseline. They are audit evidence,
not substitutes for the curated artifacts listed above.

`rtx-5070-ti-2026-08-28-c3-*.json` files and their Campaign 3 attempt records
retain the fresh baseline, profiles, rejected launch/SDPA alternatives,
counterbalanced confirmations, and integrated rebaseline. They remain audit
history for the earlier accepted fingerprint.

`rtx-5070-ti-2026-08-28-c4-*.json` files and all matching immutable attempt
records retain the environment repair, failed/reworked gates, fresh baseline,
backend screens, rejected SDPA/launch alternatives, counterbalanced row-11
confirmations, profiles, affected-row proof, and two complete final matrices.
The Campaign 4 record maps every run to its final disposition. They are now
historical optimization evidence for the same implementation fingerprint; the
`submission-*` artifacts above are the fresh selection-validation evidence.

`rtx-5070-ti-2026-08-28-c5-*.json` files and matching `C5-*` attempt records
retain every Campaign 5 baseline, profile, backend screen, candidate,
confirmation, stress test, integration gate, and failure. The integrated files
are historical; baseline and `exp010`/`exp011`/`exp012` files are immutable
decision evidence rather than current submission results.

`rtx-5070-ti-2026-08-29-c6-*.json` files and matching `C6-*` attempt records
retain the Campaign 6 preflight, fresh controls, four bounded candidate loops,
rejections, independent reviews, integrated matrices, profiles, tests, and
closure checks. They are historical evidence for the prior accepted fingerprint.

`rtx-5070-ti-2026-08-29-c7-*.json` files and matching `C7-*` attempt records
retain the Campaign 7 preflight, fresh controls, rejected head-width-256 route,
accepted residual-normalization fusion, stress tests, profiles, integration
matrices, review, tests, and closure checks. Only the Campaign 7 integrated
files are historical evidence for the prior accepted fingerprint.

`rtx-5070-ti-2026-08-29-c8-*.json` files and matching `C8-*` attempt records
retain Campaign 8's preflight, baselines, the superseded I1 observations, the
training-boundary I1R refinement, 36 stress scenarios, repeated timing and
profiles, the independent Council review, integration matrices, and every
failed fail-closed gate. Its integrated files are historical evidence for the
prior selected fingerprint.

`rtx-5070-ti-2026-08-29-c9-*.json` files and matching `C9-*` attempt records
retain Campaign 9's row-8/row-13 profiles, the correct but regressing width-1024
fusion, the regressing eight-warp variant, the inaccurate row-13 route, both
retained command-analysis failures, restoration proof, and no-winner closure.

`rtx-5070-ti-2026-08-29-c10-*.json` files and matching `C10-*` attempt records
retain Campaign 10's preflight, row-5/row-7/row-12 profiles, counterbalanced
row-5 controls, rejected variants, direct and stress gates, candidate review,
integration matrices, long runs, profiler proof, tests, and every failed
fail-closed gate. Its integrated files are the immutable prior selected
checkpoint and the controls for Campaign 11.

`rtx-5070-ti-2026-08-29-c11-*.json` files and matching `C11-*` attempt records
retain Campaign 11's preflight, three profile targets, counterbalanced row-9
controls, isolated candidate, review, transplant provenance, complete active
matrices, inherited-route rechecks, profiles, tests, launcher failures, and
closure gates. The integrated files listed first bind the Campaign 11 measured fingerprint.

`rtx-5070-ti-2026-08-28-branchfix-*.json` files and the matching `BC1-*`
attempt records retain the fresh comparison between the selected flagship and
`origin/fix/google-colab-accuracy-issue`. The candidate files use the current
frozen harness in an isolated synthetic worktree; the `candidate-adapted-*`
files add telemetry only and remain separately fingerprinted failed evidence.
They are not substitutes for the selected-submission artifacts. See the
[complete branch comparison](../experiments/BRANCH_IMPLEMENTATION_COMPARISON.md).

## Historical filename note

The legacy `rtx-5070-ti-2026-08-27*.json` filenames were regenerated during
later rebaselines and their current bytes carry later implementation
fingerprints. Earlier foundational values must be read from the exact Git
revisions cited by the complete optimization history, not inferred from the
date embedded in the current filename. The files remain in place to preserve
existing links and provenance; no measured artifact was renamed or rewritten
during consolidation.

These files prove the checked-in contract on the recorded RTX 5070 Ti
environment. The final table itself omits dtype, padding, timing, tolerance,
and backward policy, so the artifacts record the selected PyTorch assumptions
rather than claiming unstated organizer rules or performance on another GPU.

The Campaign 11 artifacts were generated directly with fingerprint schema 2, which
canonicalizes checkout line endings and redacts host-specific paths. They share
implementation SHA-256
`908a0d708cd8f70f44d5f14fda93d3cafb1cc18345f43914e715594cfa7b7ef9`.
They truthfully record a dirty local candidate based on commit
`8c89d1d4170c58d16fb75d79f212e990565fba7d`. At
capture time no commit, tag, branch rewrite, push, or public action had been
performed; later Git packaging does not change that recorded provenance. No
measured field was hand-edited.

Regenerate from Windows PowerShell. These commands deliberately write every
reproduction artifact under ignored `results/`; promote nothing into the
curated directories without a separate evidence-campaign decision. The wrapper
records command, runtime, environment, fingerprint, status, and parsed result
metrics for every run:

```powershell
$python = ".venv\Scripts\python.exe"

& $python benchmarks/run_optimization_attempt.py `
  --attempt-id REPRO-C11-FINAL `
  --hypothesis "Reproduce selected Campaign 11 final evidence" `
  --scope "Final evaluator matrix" --rollback "Discard reproduction outputs" `
  --decision observation --decision-rationale "Reproduction only" `
  --review-status not_required `
  --out results/REPRO-C11-FINAL-attempt.json `
  --result-artifact results/repro-c11-final.json -- `
  $python benchmarks/run_organizer_validation.py `
  --matrix benchmarks/final_evaluator_shapes.json `
  --out results/repro-c11-final.json

& $python benchmarks/run_optimization_attempt.py `
  --attempt-id REPRO-C11-HELDOUT `
  --hypothesis "Reproduce selected Campaign 11 held-out evidence" `
  --scope "Project held-out matrix" --rollback "Discard reproduction outputs" `
  --decision observation --decision-rationale "Reproduction only" `
  --review-status not_required `
  --out results/REPRO-C11-HELDOUT-attempt.json `
  --result-artifact results/repro-c11-heldout.json -- `
  $python benchmarks/run_matrix.py --device cuda --attention-backend auto `
  --accuracy-trials 5 --out results/repro-c11-heldout.json

& $python benchmarks/run_optimization_attempt.py `
  --attempt-id REPRO-C11-ROW9-PROFILE `
  --hypothesis "Reproduce selected Campaign 11 row-9 profiler proof" `
  --scope "Exact final row 9 profiler" --rollback "Discard reproduction outputs" `
  --decision observation --decision-rationale "Reproduction only" `
  --review-status not_required `
  --out results/REPRO-C11-ROW9-PROFILE-attempt.json `
  --result-artifact results/repro-c11-row9-profile.json -- `
  $python benchmarks/profile_cases.py `
  --manifest benchmarks/campaign11_profile_shapes.json `
  --case final-09-b64-d128-h1-s128 --dtype float32 `
  --attention-backend auto --expect-backend triton `
  --expect-fused-residual-layer-norm --steps 30 `
  --out results/repro-c11-row9-profile.json `
  --trace results/repro-c11-row9-profile-trace.json

& $python benchmarks/run_optimization_attempt.py `
  --attempt-id REPRO-C11-DEFAULT `
  --hypothesis "Reproduce selected Campaign 11 organizer-default evidence" `
  --scope "Untouched organizer default" --rollback "Discard reproduction outputs" `
  --decision observation --decision-rationale "Reproduction only" `
  --review-status not_required `
  --out results/REPRO-C11-DEFAULT-attempt.json `
  --result-artifact results/repro-c11-default.json -- `
  $python benchmarks/run_organizer_torch.py --device cuda `
  --evidence-out results/repro-c11-default.json

& $python benchmarks/run_optimization_attempt.py `
  --attempt-id REPRO-C11-SOURCE `
  --hypothesis "Reproduce selected Campaign 11 source-derived evidence" `
  --scope "Source-derived organizer matrix" --rollback "Discard reproduction outputs" `
  --decision observation --decision-rationale "Reproduction only" `
  --review-status not_required `
  --out results/REPRO-C11-SOURCE-attempt.json `
  --result-artifact results/repro-c11-source.json -- `
  $python benchmarks/run_organizer_validation.py `
  --out results/repro-c11-source.json
```

Do not hand-edit measured values. Rerun the command after implementation,
manifest, framework, driver, or hardware changes.

The `REPRO-C11-DEFAULT` command intentionally reproduces the untouched parser's
runtime defaults (`atol=0.002`, `rtol=0.02`) because that is what the historical
organizer-default artifact measured. It is not the repository-strict matrix
gate. For fresh evidence-grade validation, pass `--atol 0.001 --rtol 0.01`;
`--evidence-out` also rejects `--non-strict-weight-copy`.
