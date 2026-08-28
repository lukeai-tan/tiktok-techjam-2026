# Result artifacts

Curated evidence in this directory is intentionally versioned; scratch traces
and exploratory runs stay under ignored `results/`.

Use [the complete optimization history](../experiments/OPTIMIZATION_HISTORY.md)
for the chronological campaign narrative, attempt totals, candidate decisions,
and current-best verdict. This index owns artifact purpose and reproduction.

## Current selected-submission run

All artifacts in this section independently record schema-2 implementation
SHA-256
`de768f1ff9ddee54a9ad83a67f3e1f205044c0ad5c723fc3bb4881093c97f611`.
The actual entry is `torch_transformer_benchmark.py::UserOptimizedTransformer`.

- `rtx-5070-ti-2026-08-28-submission-final.json`: primary organizer-published
  final matrix. All 13 executable rows passed with zero failed elements across
  938,885,120 comparisons; the exact 100,000-token resource case is a separate
  authorized non-pass skip. Geomean is 1.775778x, row 11 is 5.456x, and backend
  accounting is Triton 1,120 / SDPA 0 / reference 336.
- `rtx-5070-ti-2026-08-28-submission-final-confirmation.json`: independent
  complete confirmation with the same correctness and backend counts. Geomean
  is 1.770185x and row 11 is 5.408x; aggregate geomeans differ by 0.315%.
- `rtx-5070-ti-2026-08-28-submission-heldout.json` and
  `rtx-5070-ti-2026-08-28-submission-heldout-confirmation.json`: two complete
  seven-case held-out runs, five accuracy seeds per case, raw alternating-order
  CUDA-event samples, backend counts, memory, and environment. Both are 7/7
  PASS with zero failed elements across 13,117,440 comparisons per run;
  geomeans are 1.210008x and 1.266010x. The non-padded long-causal case
  reproduces at 0.793x and 0.800x, so this residual slowdown is retained rather
  than hidden by the aggregate.
- `rtx-5070-ti-2026-08-28-submission-final-11-profile.json`: selected-submission
  profiler proof. `_attention_fwd` used 4,763.665 us across 40 Triton calls;
  every recorded attention call used the repository-owned kernel.
- `rtx-5070-ti-2026-08-28-final-10-profile.json`: historical Campaign 2
  profiler proof for the EXP-001 target. `_attention_fwd` used 2,694.679 us
  across 40 Triton calls, down 91.11% from the frozen pre-candidate profile.
- `rtx-5070-ti-2026-08-28-final-01-profile.json`: historical Campaign 2
  profiler proof for the EXP-003 target. `_attention_fwd` used 2,103.978 us
  across 40 Triton calls, down 69.98% from the pre-EXP-003 curated profile.
- `rtx-5070-ti-2026-08-28-submission-organizer-default.json`: the untouched downloaded
  PyTorch harness with only the submitted `UserOptimizedTransformer` injected.
  The organizer-default six-layer case passed 5/5 trials with zero failed
  elements and measured a 1.352x median speedup; all 1,950 optimized attention
  calls used Triton.
- `rtx-5070-ti-2026-08-28-submission-source-derived.json`: 29 source-derived entries
  executed through the untouched PyTorch harness in isolated processes. All 28
  feasible cases passed with 0/459,776,000 failed elements; the supplied
  TensorFlow benchmark's designated 100,000-token resource skip is recorded
  separately and is not counted as a pass. Overall geomean is 1.203466x and
  aggregate backend counts are Triton 672 / SDPA 1,344 / reference 2,688.

`rtx-5070-ti-2026-08-28-exp001-*.json` files are the immutable paired and
candidate evidence used for the acceptance decision. They are historical
pre-integration evidence, not substitutes for the curated post-merge artifacts.
The decision and noise waiver are recorded in
`docs/experiments/EXP-001-head64-short-tiles.md`.

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

The current artifacts were generated directly with fingerprint schema 2, which
canonicalizes checkout line endings and redacts host-specific paths. They share
implementation SHA-256
`de768f1ff9ddee54a9ad83a67f3e1f205044c0ad5c723fc3bb4881093c97f611`.
They truthfully record a dirty local candidate based on commit `b41fdaf`. At
capture time no commit, tag, branch rewrite, push, or public action had been
performed; later Git packaging does not change that recorded provenance. No
measured field was hand-edited.

Regenerate from Windows PowerShell:

```powershell
$python = ".venv\Scripts\python.exe"

& $python benchmarks/run_organizer_validation.py `
  --matrix benchmarks/final_evaluator_shapes.json `
  --out docs/results/rtx-5070-ti-2026-08-28-submission-final.json

& $python benchmarks/run_matrix.py --device cuda --attention-backend auto `
  --accuracy-trials 5 --out docs/results/rtx-5070-ti-2026-08-28-submission-heldout.json

& $python benchmarks/profile_cases.py `
  --manifest benchmarks/campaign4_profile_shapes.json `
  --case final-11-b64-d128-h16-s128 --dtype float32 `
  --attention-backend auto --steps 10 `
  --out docs/results/rtx-5070-ti-2026-08-28-submission-final-11-profile.json `
  --trace results/rtx-5070-ti-2026-08-28-submission-final-11-profile-trace.json

& $python benchmarks/run_organizer_torch.py --device cuda `
  --evidence-out docs/results/rtx-5070-ti-2026-08-28-submission-organizer-default.json

& $python benchmarks/run_organizer_validation.py `
  --out docs/results/rtx-5070-ti-2026-08-28-submission-source-derived.json
```

Do not hand-edit measured values. Rerun the command after implementation,
manifest, framework, driver, or hardware changes.
