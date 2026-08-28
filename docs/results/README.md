# Result artifacts

Curated evidence in this directory is intentionally versioned; scratch traces
and exploratory runs stay under ignored `results/`.

Use [the complete optimization history](../experiments/OPTIMIZATION_HISTORY.md)
for the chronological campaign narrative, attempt totals, candidate decisions,
and current-best verdict. This index owns artifact purpose and reproduction.

## Current selected-submission run

All artifacts in this section independently record schema-2 implementation
SHA-256
`9159177a21d039366ed4d3aef431b4b14d3bcef26d5eeaab0808efa739294029`.
The actual entry is `torch_transformer_benchmark.py::UserOptimizedTransformer`.

- `rtx-5070-ti-2026-08-28-c5-integrated-final.json`: primary organizer-published
  final matrix. All 13 executable rows passed with zero failed elements across
  938,885,120 comparisons; the exact 100,000-token resource case is a separate
  authorized non-pass skip. Geomean is 1.911947x, row 6 is 1.503x, row 7 is
  1.524x, row 11 is 5.948x, and backend accounting is Triton 1,260 / SDPA 0 /
  reference 196.
- `rtx-5070-ti-2026-08-28-c5-integrated-final-confirmation.json`: complete
  complete confirmation with the same correctness and backend counts. Geomean
  is 1.995117x and row 11 is 5.926x; aggregate geomeans differ by 4.35%.
- `rtx-5070-ti-2026-08-28-c5-integrated-heldout-5seed.json` and
  `rtx-5070-ti-2026-08-28-c5-integrated-heldout-5seed-confirmation.json`: two complete
  seven-case held-out runs, five accuracy seeds per case, raw alternating-order
  CUDA-event samples, backend counts, memory, and environment. Both are 7/7
  PASS with zero failed elements across 13,117,440 comparisons per run;
  geomeans are 1.447477x and 1.449715x. Exact-shape SDPA removed the former
  long-causal regressions: non-padded measured 1.247x/1.216x and padded
  measured 1.280x/1.423x.
- `rtx-5070-ti-2026-08-28-c5-integrated-row06-profile.json` and
  `rtx-5070-ti-2026-08-28-c5-integrated-row07-profile.json`: prove the accepted
  layer hybrids: row 6 used 20 Triton plus 20 reference calls, while row 7 used
  30 Triton plus 10 reference calls over ten profiled forwards.
- `rtx-5070-ti-2026-08-28-c5-integrated-long-causal-profile.json` and
  `rtx-5070-ti-2026-08-28-c5-integrated-long-causal-padding-profile.json`:
  each proves 20 SDPA calls and no Triton/reference attention calls.
- `rtx-5070-ti-2026-08-28-c5-integrated-row11-profile.json`: retained
  row-11 profiler proof with 40 `_attention_fwd` events and 40 Triton calls.
- `rtx-5070-ti-2026-08-28-final-10-profile.json`: historical Campaign 2
  profiler proof for the EXP-001 target. `_attention_fwd` used 2,694.679 us
  across 40 Triton calls, down 91.11% from the frozen pre-candidate profile.
- `rtx-5070-ti-2026-08-28-final-01-profile.json`: historical Campaign 2
  profiler proof for the EXP-003 target. `_attention_fwd` used 2,103.978 us
  across 40 Triton calls, down 69.98% from the pre-EXP-003 curated profile.
- `rtx-5070-ti-2026-08-28-c5-integrated-organizer-default.json`: the untouched downloaded
  PyTorch harness with only the submitted `UserOptimizedTransformer` injected.
  The organizer-default six-layer case passed 5/5 trials with zero failed
  elements and measured a 1.397x median speedup; all 1,950 optimized attention
  calls used Triton.
- `rtx-5070-ti-2026-08-28-c5-integrated-source-derived.json`: 29 source-derived entries
  executed through the untouched PyTorch harness in isolated processes. All 28
  feasible cases passed with 0/459,776,000 failed elements; the supplied
  TensorFlow benchmark's designated 100,000-token resource skip is recorded
  separately and is not counted as a pass. Overall geomean is 1.204815x and
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

`rtx-5070-ti-2026-08-28-c5-*.json` files and matching `C5-*` attempt records
retain every Campaign 5 baseline, profile, backend screen, candidate,
confirmation, stress test, integration gate, and failure. The integrated files
listed first are current; baseline and `exp010`/`exp011`/`exp012` files are
immutable decision evidence rather than alternative submission results.

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

The current artifacts were generated directly with fingerprint schema 2, which
canonicalizes checkout line endings and redacts host-specific paths. They share
implementation SHA-256
`9159177a21d039366ed4d3aef431b4b14d3bcef26d5eeaab0808efa739294029`.
They truthfully record a dirty local candidate based on commit
`3be02a3ebe562a89ca360b196057a2762b425ec4`. At
capture time no commit, tag, branch rewrite, push, or public action had been
performed; later Git packaging does not change that recorded provenance. No
measured field was hand-edited.

Regenerate from Windows PowerShell:

```powershell
$python = ".venv\Scripts\python.exe"

& $python benchmarks/run_organizer_validation.py `
  --matrix benchmarks/final_evaluator_shapes.json `
  --out docs/results/rtx-5070-ti-2026-08-28-c5-integrated-final.json

& $python benchmarks/run_matrix.py --device cuda --attention-backend auto `
  --accuracy-trials 5 --out docs/results/rtx-5070-ti-2026-08-28-c5-integrated-heldout-5seed.json

& $python benchmarks/profile_cases.py `
  --manifest benchmarks/campaign5_profile_shapes.json `
  --case final-11-b64-d128-h16-s128 --dtype float32 `
  --attention-backend auto --steps 10 `
  --out docs/results/rtx-5070-ti-2026-08-28-c5-integrated-row11-profile.json `
  --trace results/rtx-5070-ti-2026-08-28-c5-integrated-row11-profile-trace.json

& $python benchmarks/run_organizer_torch.py --device cuda `
  --evidence-out docs/results/rtx-5070-ti-2026-08-28-c5-integrated-organizer-default.json

& $python benchmarks/run_organizer_validation.py `
  --out docs/results/rtx-5070-ti-2026-08-28-c5-integrated-source-derived.json
```

Do not hand-edit measured values. Rerun the command after implementation,
manifest, framework, driver, or hardware changes.
