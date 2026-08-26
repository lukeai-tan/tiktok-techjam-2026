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

The matrix is labelled provisional because the organizer's final shape list is
not present in the repository. These files prove the checked-in contract on the
recorded RTX 5070 Ti environment; they do not claim coverage of an unpublished
matrix or a different GPU.

Regenerate from Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-wsl.ps1 `
  benchmarks/run_matrix.py --device cuda --attention-backend auto `
  --accuracy-trials 5 --out docs/results/rtx-5070-ti-2026-08-27.json

powershell -ExecutionPolicy Bypass -File scripts/run-wsl.ps1 `
  benchmarks/profile_cases.py --case long-causal-padding --dtype float32 `
  --attention-backend auto --steps 5 `
  --out docs/results/rtx-5070-ti-2026-08-27-profile.json `
  --trace results/rtx-5070-ti-profile-trace.json
```

Do not hand-edit measured values. Rerun the command after implementation,
manifest, framework, driver, or hardware changes.
