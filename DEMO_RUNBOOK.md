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

1. Show the challenge and architecture in `docs/REQUIREMENTS.md` and
   `docs/KERNEL_DESIGN.md`: BSHD layout, tiled QK, online softmax, internal
   causal/padding masks, and no dense score tensor.
2. Show the kernel implementation and the strict `auto`/`triton`/fallback
   dispatcher.
3. Run the direct GPU and end-to-end tests:

   ```powershell
   & $python -m pytest tests/test_gpu_attention.py `
     tests/test_gpu_transformer.py tests/test_dispatch.py -q
   ```

4. Run one fast, visible benchmark case:

   ```powershell
   & $python benchmarks/run_matrix.py --device cuda --case long-causal-padding `
     --dtype float32 --attention-backend auto --quick --accuracy-trials 3 `
     --out results/demo.json
   ```

   Point out `PASS`, zero failed elements, speedup, and nonzero Triton backend
   counts.

5. Prove the GPU kernel actually ran:

   ```powershell
   & $python benchmarks/profile_cases.py --case long-causal-padding --dtype float32 `
     --attention-backend auto --steps 3 --out results/demo-profile.json
   ```

   Show the `_attention_fwd` event count and matching model dispatch count.

6. Finish on the curated seven-case table in `docs/TECH_REPORT.md`: 7/7 PASS,
   zero failed elements, and the measured speedup range/geomean.

## Failure handling

- If CUDA or Triton is unavailable, stop and fix the environment; do not present
  a CPU run as GPU evidence.
- If any case is `FAIL`, `OOM`, or `ERROR`, show it honestly and do not quote the
  previous green aggregate as the current run.
- If the profiler has no `_attention_fwd` event while Triton dispatch is
  expected, treat the demo as failed rather than claiming custom execution.
- If the organizer publishes a final matrix, update the manifest and rerun all
  curated evidence before recording.
