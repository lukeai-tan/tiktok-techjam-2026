# Demo Video Runbook

Target length: three to five minutes. Record a terminal at readable scale; do
not show credentials, browser sessions, or unrelated files.
Use only repository content, your own narration, and challenge material you are
authorized to show. Do not add third-party music, logos, footage, or other
copyrighted/trademarked assets.

## Preflight

From Windows PowerShell at the repository root:

```powershell
git status --short --branch
nvidia-smi
$python = ".venv\Scripts\python.exe"
& $python -c "import torch,triton; print(torch.__version__, triton.__version__); print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
```

Confirm the branch/revision, RTX 5070 Ti, CUDA PyTorch, and Triton are visible.

## Recording sequence

1. Show `docs/ORGANIZER_INPUTS.md`, then the challenge and architecture in
   `docs/REQUIREMENTS.md` and `docs/KERNEL_DESIGN.md`: untouched download
   checksum, selected PyTorch contract, BSHD layout, tiled QK, online softmax,
   internal causal/padding masks, and no dense score tensor.
2. Show the kernel implementation and the strict `auto`/`triton`/fallback
   dispatcher.
3. Run the direct GPU and end-to-end tests:

   ```powershell
   & $python -m pytest tests/test_gpu_attention.py `
     tests/test_gpu_transformer.py tests/test_dispatch.py -q
   ```

4. Run the untouched organizer PyTorch default harness:

   ```powershell
   & $python benchmarks/run_organizer_torch.py --device cuda
   ```

   Point out the exact six-layer default configuration, 5/5 `PASS`, zero failed
   elements, and the live baseline/optimized median speedup.

5. Run the organizer-published final dimensions:

   ```powershell
   & $python benchmarks/run_organizer_validation.py `
     --matrix benchmarks/final_evaluator_shapes.json `
     --out results/demo-final-evaluator.json
   ```

   Point out 13/13 executable `PASS`, 0/938,885,120 failed elements, the
   authorized row-14 resource skip, 1.555780x geomean speedup, Campaign 3 row 9
   at 1.281x, and row 13 at 4.800x.

6. Run the full source-derived organizer validation:

   ```powershell
   & $python benchmarks/run_organizer_validation.py `
     --out results/demo-organizer-validation.json
   ```

   Point out 28/28 executable `PASS`, 0/459,776,000 failed elements, and the
   one explicitly authorized 100,000-token resource skip that is not counted
   as a pass.

7. Prove the accepted Campaign 3 row-9 kernel actually ran:

   ```powershell
   & $python benchmarks/profile_cases.py `
     --manifest benchmarks/final_profile_shapes.json `
     --case final-09-b64-d128-h1-s128 --dtype float32 `
     --attention-backend auto --steps 10 --out results/demo-profile.json
   ```

   Show 40 `_attention_fwd` events and 40 matching Triton dispatches. Compare
   the result with the recorded 55.45% profiler-time reduction in
   `docs/experiments/CAMPAIGN-003.md`.

8. Finish on `docs/TECH_REPORT.md`: the untouched organizer default is 1.367x,
   and the published final matrix is 13/13 executable PASS with zero failed
   elements and a 1.555780x geometric-mean speedup. Show the source-derived and
   held-out artifacts as broader correctness and anti-overfitting evidence.

## Failure handling

- If CUDA or Triton is unavailable, stop and fix the environment; do not present
  a CPU run as GPU evidence.
- If any case is `FAIL`, `OOM`, or `ERROR`, show it honestly and do not quote the
  previous green aggregate as the current run.
- If the profiler has no `_attention_fwd` event while Triton dispatch is
  expected, treat the demo as failed rather than claiming custom execution.
- If the organizer publishes revised dimensions, execution policy, or script,
  update the organizer checksum manifest and rerun all curated evidence before
  recording.
