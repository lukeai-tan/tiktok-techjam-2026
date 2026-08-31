# SpeedROCm public code / GitHub repository deliverable

## Intended repository

**Repository:** <https://github.com/lukeai-tan/tiktok-techjam-2026>

We verified anonymous access to this URL on 2026-09-01: the page returned HTTP
200, and remote `main` matched the pre-edit local HEAD. Recheck it after these
local documentation edits are published and immediately before putting it on
Devpost. This handoff does not push code, publish a release, or change repository
visibility.

## Project overview

We replace explicit quadratic attention intermediates in the supplied PyTorch
Transformer with a guarded Triton online-softmax kernel. We also use measured
packed QKV projections, exact-shape residual/LayerNorm fusion, and shape-aware
Triton/SDPA/reference routing. Our optimized class inherits the organizer
baseline, so the reference parameter names and `strict=True` weight-copy
contract remain intact.

Our primary result on the recorded RTX 5070 Ti is a 1.977× geometric-mean
speedup across the 13 executable published final rows with zero failed output
elements. The [technical report](03_TECHNICAL_REPORT.md) explains why the
attention arithmetic remains quadratic while attention-specific intermediate
memory becomes linear in sequence length.

### Name and platform note

We use SpeedROCm as the public project title. Our current code and recorded
results use NVIDIA CUDA on an RTX 5070 Ti; they do not establish AMD ROCm
compatibility. The install commands below therefore install the CUDA build of
PyTorch and `triton-windows`. The Python package remains `transformer_opt`.

### Reviewer reading order

1. Read the root [`README.md`](../README.md) for the code-facing overview.
2. Inspect `transformer_opt/submission.py` for the submitted model entry point.
3. Run the test command below before running performance benchmarks.
4. Read [`03_TECHNICAL_REPORT.md`](03_TECHNICAL_REPORT.md) for the Big-O
   derivation, per-input results, plots, and optimization flow diagrams.
5. Compare any fresh run with the checked-in JSON evidence; do not assume a
   different GPU will reproduce identical latency.

### Short glossary

| Term or symbol | Meaning |
| --- | --- |
| QKV | Query, key, and value projections used by attention. |
| SDPA | PyTorch scaled dot-product attention, used as a measured safe backend for selected inputs. |
| Triton | The GPU-kernel language/compiler used for repository-owned kernels, not NVIDIA Triton Inference Server. |
| CUDA | NVIDIA's GPU runtime; this is the runtime used by the current project. |
| ROCm | AMD's GPU platform; the word is part of the project name, but current ROCm support is not claimed. |
| fp16 / fp32 | 16-bit and 32-bit floating-point formats. |
| `d_model` | Total model width. |
| `×` | Speedup multiplier; `2×` means twice as fast for the stated comparison. |
| `ms`, `MiB` | Milliseconds and mebibytes (1,048,576 bytes). |
| OOM | Out of memory. An OOM is reported as a non-pass, never converted to success. |
| Implementation fingerprint | A SHA-256 identity derived from the selected implementation so results can be tied to exact code. |

### PowerShell notation used below

| Notation | Meaning |
| --- | --- |
| `$python` | A local variable containing the virtual-environment Python path. It is not a secret or environment variable. |
| `& $python ...` | PowerShell's call operator (`&`) runs the executable path stored in `$python`. |
| A trailing `` ` `` | PowerShell line continuation; the command continues on the next displayed line. |
| `--name value` | A command-line option followed by its value. |
| `# text` | A comment for the reader; PowerShell does not execute the text after `#` on that line. |

## Repository layout

| Path | Contents |
| --- | --- |
| `transformer_opt/submission.py` | `UserOptimizedTransformer`, packed-QKV cache, model-level routing, and exact-shape fusion guards |
| `transformer_opt/dispatch.py` | Auditable Triton, SDPA, and explicit-reference attention dispatch |
| `transformer_opt/config.py` | Support envelope, launch geometry, and measured routing policy |
| `transformer_opt/kernels/attention.py` | Triton tiled attention with online fp32 softmax state |
| `transformer_opt/kernels/residual_layer_norm.py` | Triton residual-add plus LayerNorm inference kernel |
| `benchmarks/` | Untouched organizer harnesses, shape manifests, matrix runners, and profiler runner |
| `tests/` | CPU contracts, direct GPU kernel tests, fallback tests, end-to-end tests, and artifact checks |
| `docs/` | Requirements, kernel design, technical report, campaign history, runbook, and curated result JSON |
| `hackathon-docs/` | Challenge snapshot, organizer-input reconciliation, Devpost draft, and compliance matrix |
| `notebooks/colab_benchmark.ipynb` | Fingerprint-pinned notebook workflow |
| `deliverables/` | Submission-facing descriptions, technical report, repository handoff, and video script |

## Setup and installation

The measured target is native Windows 11 with Python 3.12.10, PyTorch
`2.13.0+cu130`, CUDA 13.0, and Triton 3.7.1. On Windows, install the supported
`triton-windows` package:

```powershell
py -3.12 -m venv .venv
$python = ".venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
& $python -m pip install triton-windows==3.7.1.post27 numpy==2.5.2 pytest==9.1.1
```

For CPU-only development, install the CPU PyTorch wheel instead of the CUDA
wheel. GPU tests skip when CUDA/Triton is unavailable; a CPU pass is not GPU
performance evidence.

## Reproduce the repository test contract

From the repository root:

```powershell
$python = ".venv\Scripts\python.exe"
& $python -m pytest tests -q
```

The measured Campaign 11 checkpoint recorded **148/148 tests passed** with 14
upstream PyTorch deprecation warnings. The validation-hardened maintenance tree
records **164/164 tests passed**; re-run locally because GPU availability,
driver, package versions, and timing can differ.

## Reproduce the benchmark evidence

The following commands write new, disposable artifacts under the ignored
`results/` folder. Do not hand-edit the curated JSON under `docs/results/`.

```powershell
$python = ".venv\Scripts\python.exe"

# Untouched organizer PyTorch default with only the submission injected.
& $python benchmarks/run_organizer_torch.py --device cuda

# Published final dimensions: 13 executable rows plus one authorized resource skip.
& $python benchmarks/run_organizer_validation.py `
  --matrix benchmarks/final_evaluator_shapes.json `
  --out results/reproduce-final-evaluator.json

# All feasible cases derived from the supplied organizer contracts.
& $python benchmarks/run_organizer_validation.py `
  --out results/reproduce-source-derived.json

# Project-owned held-out matrix.
& $python benchmarks/run_matrix.py `
  --device cuda --attention-backend auto --accuracy-trials 5 `
  --out results/reproduce-heldout.json

# Profiler proof for the newly retained exact row-9 fusion.
& $python benchmarks/profile_cases.py `
  --manifest benchmarks/campaign11_profile_shapes.json `
  --case final-09-b64-d128-h1-s128 --dtype float32 `
  --attention-backend auto --expect-backend triton `
  --expect-fused-residual-layer-norm --steps 30 `
  --out results/reproduce-row9-profile.json
```

The benchmark contract checks correctness before timing, uses synchronized CUDA
events, alternates baseline and optimized order, records backend counts, and
fails closed for numerical failures, OOM, unexpected exceptions, or zero
completed cases. First-use compilation and random input generation are excluded
from steady-state forward latency.

## README coverage

The root [`README.md`](../README.md) contains the required repository README
sections:

- project overview and selected result;
- Windows, WSL, and CPU setup instructions;
- reproduction commands for tests, organizer matrices, held-out cases, and
  profiler proof;
- limitations, benchmark-policy assumptions, memory trade-offs, and future
  work; and
- team-contribution guidance that does not invent unverified names.

The root README is the code-facing canonical README. This handoff explains how
it maps to the competition deliverable and should be kept synchronized if the
public repository changes.

## Limitations to disclose

- The final organizer table publishes dimensions but omits dtype, padding,
  timing, tolerance, and backward policy. The measured claims are explicitly
  PyTorch float32/no-padding claims using the stricter checked-in comparator.
- The custom kernels are forward/inference-only and tuned on one RTX 5070 Ti.
- Low-precision automatic model routing chooses reference math to protect the
  strict zero-failure contract. Direct fp16 kernel coverage is not a claim of
  automatic deep-stack fp16 speedup.
- Packed QKV uses bounded derived-weight memory and is enabled only for measured
  `d_model <= 512` and exact `d_model == 1024` eager CUDA float32 shapes.
- The 100,000-token published row is an authorized resource skip, not a
  successful execution.
- These local documentation edits must be published before the public copy
  reflects them. The YouTube/Devpost steps still require action outside this
  local repository.

## Contributions

We used OpenAI Codex during repository audit, implementation, testing,
profiling, evidence reconciliation, and documentation; the history also records
a pre-existing Claude Code prototype. We did not treat AI output as proof: the
acceptance basis is source review, tests, profiler events, and fingerprint-bound
result artifacts. Public-facing copy uses `we` and `our` as the project voice,
but Git history alone does not establish the final team list or contribution
split. Add only verified participant names and responsibilities before
submission.

## Public-release checklist

- [x] Repository URL opened anonymously on 2026-09-01.
- [ ] After these local edits are published, verify that the public repository
      contains the intended source, tests, benchmark manifests, README, and
      evidence/docs paths.
- [ ] No secrets, local `.env` values, machine-specific credentials, or
      unintended scratch artifacts are committed.
- [ ] The final public repository link is pasted into Devpost.
- [ ] The demo video is uploaded to YouTube as public, plays while signed out,
      contains no unauthorized copyrighted/trademarked material, and its URL is
      pasted into Devpost.
- [ ] The organizer has confirmed that the **SpeedROCm** project title is
      acceptable under the event's trademark rule; use plain text and no AMD,
      ROCm, NVIDIA, CUDA, PyTorch, Triton, or TikTok logos without permission.
- [ ] Add the verified participant names and responsibilities, if applicable.
