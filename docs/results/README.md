# Result artifacts

Curated evidence in this directory is intentionally versioned; scratch traces
and exploratory runs stay under ignored `results/`.

## Current target run

- `rtx-5070-ti-2026-08-28-final-evaluator-baseline.json`: the organizer-published
  14-row final shape table under the recorded PyTorch assumptions. All 13
  executable rows passed with zero failed elements across 938,885,120
  comparisons; the exact 100,000-token resource case is recorded separately
  and excluded from the pass count. Geometric-mean speedup is 1.427x.
- `rtx-5070-ti-2026-08-27.json`: seven-case project-owned held-out matrix, five
  accuracy seeds per case, raw alternating-order CUDA-event samples, backend
  counts, memory, and environment/revision fingerprint. It is 7/7 PASS with a
  1.221x geomean.
- `rtx-5070-ti-2026-08-27-profile.json`: profiler summary for the
  `long-causal-padding` case, including `_attention_fwd` event proof and the top
  device-time events.
- `rtx-5070-ti-2026-08-28-final-10-profile.json`: integrated profiler proof for
  the EXP-001 target. `_attention_fwd` used 3,205.548 us across 40 Triton calls,
  down 89.43% from the frozen pre-candidate profile.
- `rtx-5070-ti-2026-08-28-final-01-profile.json`: representative neighboring
  final-shape profile proving the unchanged `head_dim=32` path still executes
  Triton.
- `rtx-5070-ti-2026-08-27-organizer-default.json`: the untouched downloaded
  PyTorch harness with only the submitted `UserOptimizedTransformer` injected.
  The organizer-default six-layer case passed 5/5 trials with zero failed
  elements and measured a 1.408x median speedup; all 1,950 optimized attention
  calls used Triton.
- `rtx-5070-ti-2026-08-27-organizer-validation.json`: 29 source-derived entries
  executed through the untouched PyTorch harness in isolated processes. All 28
  feasible cases passed with 0/459,776,000 failed elements; the supplied
  TensorFlow benchmark's designated 100,000-token resource skip is recorded
  separately and is not counted as a pass. Overall geomean is 1.233x.

`rtx-5070-ti-2026-08-28-exp001-*.json` files are the immutable paired and
candidate evidence used for the acceptance decision. They are historical
pre-integration evidence, not substitutes for the curated post-merge artifacts.
The decision and noise waiver are recorded in
`docs/experiments/EXP-001-head64-short-tiles.md`.

These files prove the checked-in contract on the recorded RTX 5070 Ti
environment. The final table itself omits dtype, padding, timing, tolerance,
and backward policy, so the artifacts record the selected PyTorch assumptions
rather than claiming unstated organizer rules or performance on another GPU.

The current artifacts were generated directly with fingerprint schema 2, which
canonicalizes checkout line endings and redacts host-specific paths. They share
implementation SHA-256
`83d952ab3268cffba2ac9b396c64f5733c6e46e58d37c03f92de04c7ff5a6e4f`.
No measured field was hand-edited.

Regenerate from Windows PowerShell:

```powershell
$python = ".venv\Scripts\python.exe"

& $python benchmarks/run_organizer_validation.py `
  --matrix benchmarks/final_evaluator_shapes.json `
  --out docs/results/rtx-5070-ti-2026-08-28-final-evaluator-baseline.json

& $python benchmarks/run_matrix.py --device cuda --attention-backend auto `
  --accuracy-trials 5 --out docs/results/rtx-5070-ti-2026-08-27.json

& $python benchmarks/profile_cases.py --case long-causal-padding --dtype float32 `
  --attention-backend auto --steps 5 `
  --out docs/results/rtx-5070-ti-2026-08-27-profile.json `
  --trace results/rtx-5070-ti-profile-trace.json

& $python benchmarks/run_organizer_torch.py --device cuda `
  --evidence-out docs/results/rtx-5070-ti-2026-08-27-organizer-default.json

& $python benchmarks/run_organizer_validation.py `
  --out docs/results/rtx-5070-ti-2026-08-27-organizer-validation.json

& $python benchmarks/profile_cases.py `
  --manifest benchmarks/final_profile_shapes.json `
  --case final-10-b64-d128-h2-s128 --dtype float32 `
  --attention-backend auto --steps 10 `
  --out docs/results/rtx-5070-ti-2026-08-28-final-10-profile.json `
  --trace results/rtx-5070-ti-final-10-profile-trace.json
```

Do not hand-edit measured values. Rerun the command after implementation,
manifest, framework, driver, or hardware changes.
