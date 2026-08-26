# Transformer GPU Kernel Implementation Handoff

Updated: 2026-08-27 (Asia/Singapore)

## Objective and user intent

Implement Track 3, **"Implement a GPU Kernel for a Transformer Layer,"** as a
real repository-owned GPU implementation, not merely a branch or a wrapper
around PyTorch SDPA. The user first requested a documentation/code audit and a
temporary plan, then explicitly requested that the plan be carried out
completely. They have now requested this handoff so work can continue in a new
chat.

The implementation is complete for the checked-in/provisional contract. A
compiled-mode review found and fixed a CUDA-graph output-lifetime defect, added
regression coverage, and regenerated the target-GPU evidence against the final
implementation fingerprint.

## Repository and Git state

- Repository: `C:\Users\jared\Downloads\GitHub\tiktok-techjam-2026`
- Remote: `https://github.com/lukeai-tan/tiktok-techjam-2026`
- Branch: `feat/transformer-gpu-kernel-implementation`
- Branch base commit: `3c7a05205f1090f7f074312ed82e3e6753d16b92`
- The implementation is organized as segmented commits on the feature branch;
  use `git log --oneline origin/main..HEAD` for the exact pushed chain.

Current tracked/untracked scope includes the Triton implementation, dispatcher,
benchmark tools, tests, docs, CI, notebook, scripts, curated results, and this
handoff. `transformer_opt/triton_impl.py` is intentionally deleted because its
standalone LayerNorm was measured slower than native PyTorch.

## Source-of-truth and evidence boundary

Read these first:

1. `docs/REQUIREMENTS.md` - current executable contract and acceptance criteria.
2. `docs/PROJECT_CONTEXT.md` - architecture, decisions, and verified status.
3. `benchmarks/reference/manifest.json` - frozen Git-blob provenance.
4. `benchmarks/official_shapes.json` - seven-case **provisional** matrix.
5. `docs/TEMP_TRANSFORMER_GPU_KERNEL_PLAN.md` - historical audit/plan; its old
   HOLD text describes the pre-implementation repository.
6. `docs/RELEASE_GATE.md` - evidence-backed current-contract verdict.
7. `docs/workflows/transformer-gpu-kernel.json` - completed workflow record.

The final organizer benchmark/shape list is not present in the repository.
Current work is complete only for the checked-in/provisional contract. Do not
claim final organizer-matrix completeness until new organizer material is
reconciled and rerun.

## Implemented architecture

### Custom kernel

- `transformer_opt/kernels/attention.py` contains `_attention_fwd`, a Triton
  forward attention kernel.
- Input/output layout is `[B, S, H, D]`.
- It fuses QK, online softmax, causal/key-padding masking, and P@V.
- It does not materialize a dense `[B,H,S,S]` score/probability tensor or dense
  causal mask.
- Softmax state/output accumulation is fp32; score rounding intentionally
  follows the checked-in reference's dtype boundaries.
- Float32 follows `torch.backends.cuda.matmul.allow_tf32`; IEEE mode is tested.
- All-masked rows return finite zeros.

### Support envelope and dispatch

- `transformer_opt/config.py`: CUDA compute capability >= 8.0; head dimensions
  16/32/64/128; sequence length <= 8192; fp32/fp16; final stride 1; forward
  inference only.
- `transformer_opt/dispatch.py`: auditable `auto`, `triton`, `sdpa`, and
  `reference` paths.
- Forced Triton fails clearly when unavailable/unsupported; no swallowed compile
  errors or silent custom claims.
- `UserOptimizedTransformer` records actual backend counts in eager execution.
- End-to-end automatic low-precision model runs use explicit reference math for
  strict numerical safety; direct fp16 Triton attention remains tested.
- QKV/output/FFN GEMMs and LayerNorm remain vendor PyTorch/cuBLAS operations.

### Benchmark/evidence tooling

- `benchmarks/run_matrix.py`: manifest-driven, correctness-before-timing,
  alternating model order, raw CUDA-event samples, explicit PASS/FAIL/OOM/ERROR,
  zero-case failure, peak memory, dispatch counts, environment and content
  fingerprint.
- `benchmarks/profile_cases.py`: profiler proof for `_attention_fwd`.
- `tools/capture_environment.py`: redacted environment/revision capture and
  implementation fingerprint.
- `sweep.py`: compatibility wrapper around the fail-closed matrix runner.
- `scripts/run-wsl.ps1`: authoritative PowerShell-to-WSL runner.
- `tools/triton-cc`: compiler shim preferring gcc/clang and falling back to Zig.

## Verified target environment

- GPU: NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, 16,303 MiB.
- Host path: Windows + WSL2 Ubuntu.
- WSL Python: 3.14.4.
- Venv: `/home/jared/.venvs/tiktok-techjam-2026`.
- PyTorch: 2.13.0+cu130; CUDA runtime 13.0.
- Triton: 3.7.1.
- Driver reported earlier: 610.47.
- CPU: AMD Ryzen 7 9850X3D.
- WSL kernel: 6.6.114.1.

The minimal WSL image has no system C compiler/Python headers. User-scoped
bootstrap already exists:

- Zig 0.16.0: `/home/jared/.local/opt/zig-x86_64-linux-0.16.0`
- Zig link: `/home/jared/.local/bin/zig`
- extracted Python headers: `/home/jared/.local/opt/python3.14-dev`

Use the wrapper; do not reinstall unless it actually fails:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-wsl.ps1 <python args>
```

## Final curated custom result

The post-fix target-GPU evidence is:

- 7 requested / 7 completed / 7 PASS; 0 FAIL/OOM/ERROR.
- 35 accuracy trials; 13,117,440 elements; zero failed elements.
- Maximum absolute error: `0.0009976625442504883`.
- Every timing call selected Triton; no SDPA/reference timing fallback.
- 90 raw CUDA samples per model/case (10 warmup, 30 repeats x 3 rounds).
- Median end-to-end speedup range: 1.138x to 1.566x.
- Geometric-mean speedup: 1.360x.
- Long-attention incremental peak allocation: 78 MiB to 22 MiB (71.8%).
- Profiler: `_attention_fwd` count 10 for five two-layer forwards; fallback
  counts zero.

Artifacts:

- `docs/results/rtx-5070-ti-2026-08-27.json`
- `docs/results/rtx-5070-ti-2026-08-27-profile.json`

Both artifacts and the current implementation carry fingerprint
`a36abc1d440e7d5318348854a673f832c5d9ae649e295ffeae01cd599d478eb5`.
`tests/test_result_artifacts.py` fails closed if this relationship becomes stale.

## Latest compile and backend-comparison investigation

### CUDA-graph accuracy bug found and fixed

Running both models with `torch.compile(..., mode="reduce-overhead")` initially
failed during accuracy comparison with:

```text
RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run
```

Cause: the compiled baseline returned a CUDA-graph-owned buffer that the next
compiled model invocation invalidated. Fix in `run_accuracy_tests()`:

```python
reference = baseline(x, valid_mask).clone()
```

This is the change that advanced the implementation fingerprint to `a36abc...`.

The same representative compiled command then passed 3/3 trials with zero
failed elements and measured 1.124x compiled-baseline-to-compiled-optimized
speedup (0.1115 ms vs 0.0992 ms median):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-wsl.ps1 torch_transformer_benchmark.py --device cuda --dtype float32 --batch-size 2 --seq-len 128 --d-model 256 --heads 8 --ffn-dim 1024 --layers 2 --attention-backend auto --accuracy-trials 3 --warmup 3 --repeats 10 --benchmark-rounds 2 --compile-baseline --compile-user --compile-mode reduce-overhead
```

Backend counters are zero under compilation because Python-side counter mutation
is deliberately suppressed while Dynamo is compiling. Do not cite those zero
counts as evidence of fallback or custom execution. Eager profiler evidence is
the authoritative custom-kernel proof.

The eager counterpart passed and measured 1.245x in that short representative
run (0.3427 ms vs 0.2752 ms). These short runs are evaluation notes, not curated
competition claims. Compilation is not enabled by default because it adds a
second optimization system, weakens dispatch observability, and did not improve
the custom-vs-baseline ratio in this sample.

### SDPA comparator

A full scratch SDPA matrix was run after the clone fix:

- path: ignored `results/rtx-5070-ti-sdpa-comparison.json`
- fingerprint: current `a36abc...`
- result: 7/7 PASS, all timing calls SDPA, 1.201x geomean versus eager baseline.
- custom Triton measured 1.360x geomean and was materially faster in
  masked/causal/long-attention regimes; SDPA was slightly faster for one short
  no-mask regime.

The scratch SDPA artifact remains ignored by design; headline claims use only
the curated custom artifacts.

## Validation and reproduction

The final GPU suite contains 59 tests. It covers direct Triton correctness,
causal/padding/tile boundaries, all-masked behavior, TF32-disabled IEEE math,
end-to-end strict state-dict compatibility, low-precision routing, dispatch
positive/negative paths, fail-closed matrix states, compiled CUDA-graph output
ownership, and result-fingerprint checks.

Run the final checks with:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-wsl.ps1 -m pytest tests -q
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-wsl.ps1 -m compileall -q torch_transformer_benchmark.py transformer_opt benchmarks tools sweep.py
   python -m json.tool benchmarks/official_shapes.json > $null
   python -m json.tool benchmarks/reference/manifest.json > $null
   python -m json.tool docs/workflows/transformer-gpu-kernel.json > $null
   python -m json.tool notebooks/colab_benchmark.ipynb > $null
   python C:\Users\jared\.codex\skills\agentic-workflow-orchestrator\scripts\validate_workflow.py docs/workflows/transformer-gpu-kernel.json
   git diff --check
   ```

Verify the frozen reference without a PowerShell text pipeline:

   ```powershell
   python -c "import hashlib, subprocess, json; m=json.load(open('benchmarks/reference/manifest.json')); b=subprocess.check_output(['git','cat-file','blob',m['git_blob_oid']]); assert hashlib.sha256(b).hexdigest()==m['sha256']; print('reference checksum: PASS')"
   ```

The repo-memory graph lives at
`C:\Users\jared\Documents\Codex-Graphs\repos\github-com--lukeai-tan--tiktok-techjam-2026`.

One nonblocking upstream warning may appear from Triton on Python 3.14:
`AnnAssign.__init__` deprecation that will become an error in Python 3.15.

## Remaining external submission work

- Reconcile any final organizer benchmark, shapes, dtypes, timing, backward, or
  source-modification rules with `docs/REQUIREMENTS.md`, then rerun the matrix.
- Record and upload the public video using `DEMO_RUNBOOK.md`.
- Do not claim remote GitHub Actions execution until the pushed workflow has
  actually completed on GitHub.

## Important decisions and rejected work

- Keep PyTorch SDPA as the safe fallback/comparator, but do not present it as
  the repository-owned kernel.
- Remove the inherited standalone Triton LayerNorm. Measurements:
  - 1024x512: native 0.00832 ms vs custom 0.01629 ms (0.511x).
  - 2048x512: native 0.01082 ms vs custom 0.01562 ms (0.693x).
  - 256x1024: native 0.00758 ms vs custom 0.01664 ms (0.456x).
- Do not implement residual+LayerNorm or custom FFN/GEMM work without a new
  profile showing enough end-to-end ceiling. Native LayerNorm is a small share;
  mature GEMMs dominate the wide model.
- Do not relax the executable correctness rule. It is per-element:
  `abs_error <= 0.001 OR abs_error <= 0.01 * abs(reference)`.
- Do not call fallback a custom success. Backend counts and profiler names are
  required evidence.
- Keep `results/` ignored for scratch data and `docs/results/` tracked for
  curated, provenance-rich evidence.

## Main created/changed paths

- Kernel/dispatch: `transformer_opt/kernels/attention.py`,
  `transformer_opt/config.py`, `transformer_opt/dispatch.py`,
  `transformer_opt/__init__.py`, `torch_transformer_benchmark.py`.
- Benchmark/provenance: `benchmarks/run_matrix.py`,
  `benchmarks/profile_cases.py`, `benchmarks/official_shapes.json`,
  `benchmarks/reference/manifest.json`, `tools/capture_environment.py`,
  `sweep.py`.
- Runtime: `scripts/run-wsl.ps1`, `tools/triton-cc`, `requirements.txt`.
- Tests: `tests/test_correctness.py`, `tests/test_dispatch.py`,
  `tests/test_gpu_attention.py`, `tests/test_gpu_transformer.py`,
  `tests/test_sweep_integrity.py`, `tests/test_result_artifacts.py`.
- Docs/delivery: `README.md`, `docs/REQUIREMENTS.md`,
  `docs/PROJECT_CONTEXT.md`, `docs/KERNEL_DESIGN.md`,
  `docs/TECH_REPORT.md`, `docs/results/`, `docs/DEVPOST_DESCRIPTION.md`,
  `DEMO_RUNBOOK.md`, `.github/workflows/test.yml`,
  `notebooks/colab_benchmark.ipynb`.

## Official references already consulted

- PyTorch installation: `https://docs.pytorch.org/get-started/locally/`
- PyTorch CUDA 13.0 wheels: `https://download.pytorch.org/whl/cu130/torch/`
- Triton installation: `https://triton-lang.org/main/getting-started/installation.html`
- Triton fused-attention tutorial:
  `https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html`
- Zig downloads/index: `https://ziglang.org/download/` and
  `https://ziglang.org/download/index.json`

No memory files were used for this repository task. No secrets were read or
written.
