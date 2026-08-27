# Result artifacts

Curated evidence in this directory is intentionally versioned; scratch traces
and exploratory runs stay under ignored `results/`.

## Current target run

- `rtx-5070-ti-2026-08-27.json`: seven-case provisional matrix, five accuracy
  seeds per case, raw alternating-order CUDA-event samples, backend counts, and
  environment/revision fingerprint.
- `rtx-5070-ti-2026-08-27-profile.json`: profiler summary for the
  `long-causal-padding` case, including `_attention_fwd` event proof and the top
  device-time events.
- `rtx-5070-ti-2026-08-27-organizer-default.json`: the untouched downloaded
  PyTorch harness with only the submitted `UserOptimizedTransformer` injected.
  The organizer-default six-layer case passed 5/5 trials with zero failed
  elements and measured a 1.411x median speedup; all 1,950 optimized attention
  calls used Triton.
- `rtx-5070-ti-2026-08-27-organizer-validation.json`: 29 source-derived entries
  executed through the untouched PyTorch harness in isolated processes. All 28
  feasible cases passed with 0/459,776,000 failed elements; the supplied
  TensorFlow benchmark's designated 100,000-token resource skip is recorded
  separately and is not counted as a pass.

The matrix is labelled provisional because the organizer's final shape list is
not present in the repository. These files prove the checked-in contract on the
recorded RTX 5070 Ti environment; they do not claim coverage of an unpublished
matrix or a different GPU.

The current artifacts were generated directly with fingerprint schema 2, which
canonicalizes checkout line endings and redacts host-specific paths. They share
implementation SHA-256
`112124f9ca9811f5ed697339726b3c90c23b3847f5e3659ca7c8dfdd296e65d9`.
No measured field was hand-edited.

Regenerate from Windows PowerShell:

```powershell
$python = ".venv\Scripts\python.exe"

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
```

Do not hand-edit measured values. Rerun the command after implementation,
manifest, framework, driver, or hardware changes.
