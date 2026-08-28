# Result artifacts

Curated evidence in this directory is intentionally versioned; scratch traces
and exploratory runs stay under ignored `results/`.

## Current target run

- `rtx-5070-ti-2026-08-28-c3-final.json`: the organizer-published
  14-row final shape table under the recorded PyTorch assumptions. All 13
  executable rows passed with zero failed elements across 938,885,120
  comparisons; the exact 100,000-token resource case is recorded separately
  and excluded from the pass count. Geometric-mean speedup is 1.556x; the
  accepted Campaign 3 row 9 measures 1.281x.
- `rtx-5070-ti-2026-08-28-c3-heldout.json`: seven-case project-owned held-out matrix, five
  accuracy seeds per case, raw alternating-order CUDA-event samples, backend
  counts, memory, and environment/revision fingerprint. It is 7/7 PASS with a
  1.220x geomean.
- `rtx-5070-ti-2026-08-28-c3-final-09-profile.json`: integrated Campaign 3
  profiler proof. `_attention_fwd` used 3,018.182 us across 40 Triton calls,
  down 55.45% from the fresh pre-candidate row-9 profile.
- `rtx-5070-ti-2026-08-28-final-10-profile.json`: historical Campaign 2
  profiler proof for the EXP-001 target. `_attention_fwd` used 2,694.679 us
  across 40 Triton calls, down 91.11% from the frozen pre-candidate profile.
- `rtx-5070-ti-2026-08-28-final-01-profile.json`: historical Campaign 2
  profiler proof for the EXP-003 target. `_attention_fwd` used 2,103.978 us
  across 40 Triton calls, down 69.98% from the pre-EXP-003 curated profile.
- `rtx-5070-ti-2026-08-28-c3-organizer-default.json`: the untouched downloaded
  PyTorch harness with only the submitted `UserOptimizedTransformer` injected.
  The organizer-default six-layer case passed 5/5 trials with zero failed
  elements and measured a 1.367x median speedup; all 1,950 optimized attention
  calls used Triton.
- `rtx-5070-ti-2026-08-28-c3-source-derived.json`: 29 source-derived entries
  executed through the untouched PyTorch harness in isolated processes. All 28
  feasible cases passed with 0/459,776,000 failed elements; the supplied
  TensorFlow benchmark's designated 100,000-token resource skip is recorded
  separately and is not counted as a pass. Overall geomean is 1.201x and the
  float32-only geomean is 1.385x.

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
counterbalanced confirmations, and integrated rebaseline. The five Campaign 3
artifacts identified above are current curated evidence; the others remain
audit history.

These files prove the checked-in contract on the recorded RTX 5070 Ti
environment. The final table itself omits dtype, padding, timing, tolerance,
and backward policy, so the artifacts record the selected PyTorch assumptions
rather than claiming unstated organizer rules or performance on another GPU.

The current artifacts were generated directly with fingerprint schema 2, which
canonicalizes checkout line endings and redacts host-specific paths. They share
implementation SHA-256
`9071e3c049a7a3bc2311fc9d33997202ce4bead93d9daced375340fe6308eb9e`.
No measured field was hand-edited.

Regenerate from Windows PowerShell:

```powershell
$python = ".venv\Scripts\python.exe"

& $python benchmarks/run_organizer_validation.py `
  --matrix benchmarks/final_evaluator_shapes.json `
  --out docs/results/rtx-5070-ti-2026-08-28-c3-final.json

& $python benchmarks/run_matrix.py --device cuda --attention-backend auto `
  --accuracy-trials 5 --out docs/results/rtx-5070-ti-2026-08-28-c3-heldout.json

& $python benchmarks/profile_cases.py `
  --manifest benchmarks/final_profile_shapes.json `
  --case final-09-b64-d128-h1-s128 --dtype float32 `
  --attention-backend auto --steps 10 `
  --out docs/results/rtx-5070-ti-2026-08-28-c3-final-09-profile.json `
  --trace results/rtx-5070-ti-c3-final-09-profile-trace.json

& $python benchmarks/run_organizer_torch.py --device cuda `
  --evidence-out docs/results/rtx-5070-ti-2026-08-28-c3-organizer-default.json

& $python benchmarks/run_organizer_validation.py `
  --out docs/results/rtx-5070-ti-2026-08-28-c3-source-derived.json
```

Do not hand-edit measured values. Rerun the command after implementation,
manifest, framework, driver, or hardware changes.
