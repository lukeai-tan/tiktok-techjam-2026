# Transformer GPU Kernel Handoff

Updated: 2026-08-27 (Asia/Singapore)

## Objective and current verdict

Implement TikTok TechJam 2026 Track 3 as a real repository-owned GPU kernel,
not a wrapper that relabels a framework fallback. The repository is **PASS for
the checked-in/provisional executable contract** and **HOLD for final external
submission**.

The remaining holds are explicit in `docs/TRACK3_COMPLIANCE.md`:

1. obtain and reconcile the final organizer benchmark/shape combinations;
2. make the GitHub repository public (the unauthenticated URL returned 404 on
   2026-08-27); and
3. record/upload the public YouTube demo and publish the Devpost entry.

Do not claim final organizer-matrix completeness or public-video completion
until those actions are independently verified.

## Repository and Git state

- Remote: `https://github.com/lukeai-tan/tiktok-techjam-2026`
- Branch: `feat/transformer-gpu-kernel-implementation`
- Change-set base commit: `d287d82e8796bc17ba08af63f3d6de44673e29c0`.
- The completed work is segmented into target-GPU implementation/evidence,
  secure Colab reproduction, and submission-audit documentation commits.
- The worktree should be clean after those commits; verify it before making
  further changes.
- The owner explicitly authorized committing and pushing this branch on
  2026-08-27. Publishing the Devpost/YouTube entries or changing repository
  visibility remains a separate external action.

## Read first

1. `docs/REQUIREMENTS.md` - executable contract and acceptance criteria.
2. `docs/TRACK3_COMPLIANCE.md` - every Track 3 clause, evidence, and external
   hold.
3. `docs/KERNEL_DESIGN.md` - algorithm, packed projection, support envelope,
   and rejected tuning.
4. `docs/TECH_REPORT.md` - environment, methods, measurements, and AI use.
5. `benchmarks/reference/manifest.json` - frozen checked-in benchmark provenance.
6. `benchmarks/official_shapes.json` - seven-case provisional matrix.
7. `docs/RELEASE_GATE.md` - current pass/hold decision.

The obsolete temporary implementation plan was removed after its requirements
and decisions were incorporated into permanent documentation.

## Implemented architecture

### Repository-owned Triton attention

- `transformer_opt/kernels/attention.py` defines `_attention_fwd`.
- Layout is `[B,S,H,D]`; arbitrary batch/sequence/head strides are supported
  with unit final stride.
- One tiled launch performs QK, fp32 online softmax, causal/prefix-padding
  masking, and P@V.
- The custom path never materializes a dense `[B,H,S,S]` score/probability
  tensor or combined causal mask.
- Reference score-rounding boundaries and the float32 TF32 toggle are preserved.
- All-masked rows return finite zeros.

### Packed QKV inference

- For eager CUDA float32 inference through `d_model=512`, three Q/K/V
  projections are replaced by one vendor GEMM using derived packed weights.
- The cache signature covers parameter data pointers, mutation versions,
  devices, and dtypes. Loading, mutation, or a device/dtype move rebuilds it.
- Packed tensors are non-persistent and do not alter baseline state-dict keys.
- Training, compilation, CPU, low precision, and wider shapes use the original
  separate projections.

### Dispatch and numerical safety

- `transformer_opt/config.py` declares compute capability >= 8.0, head widths
  16/32/64/128, sequence <= 8192, fp32/fp16, unit final stride, and inference.
- `transformer_opt/dispatch.py` exposes `auto`, `triton`, `sdpa`, and
  `reference`; forced Triton rejects unsupported inputs clearly.
- Measured auto policy uses SDPA for unmasked non-causal float32 sequences <=128
  with head dimension <=32 and Triton for the other validated fp32 regimes.
- End-to-end fp16/bf16 auto mode uses reference-style attention because fused
  differences compounded beyond the unusually strict executable tolerance.
- Backend counters and profiler events prevent silent fallback from being
  presented as custom execution.

### Benchmark and provenance

- `benchmarks/run_matrix.py`: correctness before timing, alternating order, raw
  CUDA-event samples, peak allocation, backend counts, and explicit
  PASS/FAIL/OOM/ERROR accounting.
- `benchmarks/profile_cases.py`: independent `_attention_fwd` profiler proof.
- `tools/capture_environment.py`: schema-2 cross-platform implementation hash,
  redacted paths, CPU/GPU/driver/runtime/disk capture.
- Curated artifacts fail tests if they do not match the current implementation.
- The Colab notebook uses anonymous Git access first and a temporary
  `GIT_ASKPASS` prompt for private access; it never embeds a token in a URL.

## Current target evidence

Environment:

- AMD Ryzen 9 9950X, 16 cores / 32 logical processors;
- NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, 16,303 MiB;
- NVIDIA driver 610.88;
- native Windows 11 build 26200;
- Python 3.12.10;
- PyTorch 2.13.0+cu130, CUDA runtime 13.0;
- Triton 3.7.1 (`triton-windows==3.7.1.post27`).

Curated matrix:

- 7 requested / 7 completed / 7 PASS; 0 FAIL/OOM/ERROR.
- 35 trials; 13,117,440 elements; zero failed elements.
- Maximum absolute error: `0.0009923577308654785`.
- Timing selected SDPA for two short unmasked cases and Triton for the other
  five masked, causal, long, or wider-head cases.
- 90 raw samples per model/case after 10 warmups.
- Median speedup range: 1.236x to 1.741x.
- Geometric-mean end-to-end speedup: 1.498x.
- Long-attention incremental allocation: 78 MiB to 22 MiB (71.8%).

Profiler:

- `_attention_fwd`: count 10 for five two-layer forwards.
- Backend counts: Triton 10, SDPA 0, reference 0.
- `addmm`: count 40, consistent with packed QKV (the separate-QKV design used
  60 across the same five forwards).

Artifacts and implementation fingerprint:

- `docs/results/rtx-5070-ti-2026-08-27.json`
- `docs/results/rtx-5070-ti-2026-08-27-profile.json`
- `314dfa1615fe17b610d4851dd2a55377561f34b5a409762bf7fe43a4e5c196de`

The prior WSL evidence was replaced because the implementation changed and the
Ubuntu distribution is no longer installed on this host. The current artifacts
were freshly generated under native Windows; they were not manually migrated.

## Rejected work

- Standalone Triton LayerNorm: only 0.46x-0.69x native CUDA performance.
- Residual/LayerNorm fusion: LayerNorm is a small measured share relative to
  its support and numerical risk.
- Custom output/FFN GEMMs: no profile evidence to justify replacing cuBLAS.
- Causal loop-frontier pruning: correct but neutral in controlled end-to-end
  measurements on this GPU.
- Alternate tile/stage policy: favorable microbenchmarks did not improve the
  full matrix, so the simpler existing launch policy was restored.
- Default `torch.compile`: useful as an optional comparison, but it weakens
  eager dispatch observability and did not improve the custom/baseline ratio in
  the prior controlled sample.

## Validation commands

The current native target run passes 66/66 tests.

With the native environment from README:

```powershell
$python = ".venv\Scripts\python.exe"
& $python -m pytest tests -q
& $python -m compileall -q torch_transformer_benchmark.py transformer_opt benchmarks tools sweep.py
& $python -m json.tool benchmarks/official_shapes.json > $null
& $python -m json.tool benchmarks/reference/manifest.json > $null
& $python -m json.tool docs/workflows/transformer-gpu-kernel.json > $null
& $python -m json.tool notebooks/colab_benchmark.ipynb > $null
git diff --check
```

Verify the frozen reference from Git bytes, not a PowerShell text pipeline:

```powershell
& $python -c "import hashlib,subprocess,json; m=json.load(open('benchmarks/reference/manifest.json')); b=subprocess.check_output(['git','cat-file','blob',m['git_blob_oid']]); assert hashlib.sha256(b).hexdigest()==m['sha256']; print('reference checksum: PASS')"
```

GPU tests skip on CPU and never count as target-GPU evidence. If an Ubuntu WSL
distribution exists, `scripts/run-wsl.ps1` remains an optional equivalent
runner; it is not the source of the current curated measurements.
