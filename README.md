# Transformer Layer GPU Kernel — TikTok TechJam 2026

A repository-owned Triton attention kernel for the Track 3 Transformer
benchmark. It fuses QK, online softmax, masking, and P@V without materializing
the quadratic attention matrix, then routes unsupported cases to explicit,
auditable fallbacks.

## Verified result

On an NVIDIA GeForce RTX 5070 Ti under WSL2, the current provisional float32
matrix completed **7/7 PASS**, with **0 failed elements across 35 accuracy
trials / 13,117,440 output elements**. Every optimized timing call used the
custom Triton kernel.

| case | baseline median | optimized median | speedup |
| --- | ---: | ---: | ---: |
| tiny overhead | 0.325 ms | 0.262 ms | 1.242x |
| medium throughput | 0.319 ms | 0.238 ms | 1.336x |
| medium + padding | 0.652 ms | 0.498 ms | 1.309x |
| long causal | 0.679 ms | 0.474 ms | 1.432x |
| long causal + padding | 0.830 ms | 0.530 ms | 1.566x |
| long attention | 0.814 ms | 0.525 ms | 1.550x |
| wide model | 0.236 ms | 0.208 ms | 1.138x |

Geometric-mean end-to-end speedup: **1.360x**. The long-attention incremental
peak allocation fell from 78 MiB to 22 MiB (71.8%). The largest observed absolute
error was **0.000997663**, within the executable atol=0.001 OR rtol=0.01
contract.

Raw samples, environment metadata, implementation fingerprint, and dispatch
counts are in
[the matrix artifact](docs/results/rtx-5070-ti-2026-08-27.json).
[Profiler evidence](docs/results/rtx-5070-ti-2026-08-27-profile.json)
records _attention_fwd ten times for five two-layer forwards.

The matrix is explicitly **provisional**: the final organizer shape list is not
present in this repository. See [the requirements](docs/REQUIREMENTS.md) for
the evidence boundary.

## What is implemented

The checked-in reference is a pre-LayerNorm Transformer:

~~~text
x = x + Attention(LayerNorm(x))
x = x + Linear(GELU(Linear(LayerNorm(x)), approximate="none"))
~~~

UserOptimizedTransformer preserves the exact parameter structure and strict
weight copy while replacing explicit attention with:

- projection-friendly [B,S,H,D] input/output layout;
- tiled QK and P@V in one Triton launch;
- fp32 online-softmax state and accumulator;
- causal and prefix-padding bounds inside the kernel;
- no [B,H,S,S] score, probability, or dense combined-mask allocation;
- a measured fixed launch policy for head dimensions 16/32/64/128;
- observable auto, triton, sdpa, and reference routing.

The primary end-to-end custom path uses the benchmark-default float32 and
follows its TF32 toggle. Direct fp16 kernel tests pass, but the model's auto
mode keeps fp16/bf16 on exact reference-style math because fused
low-precision differences compound beyond this unusually strict deep-stack
tolerance. Forced modes remain available for honest comparison.

Algorithm, bounds, numerical choices, dispatch rules, and rejected
optimizations are documented in [the kernel design](docs/KERNEL_DESIGN.md).

## Known-good target environment

| component | version |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, 16,303 MiB |
| OS | WSL2 Ubuntu |
| Python | 3.14.4 |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime | 13.0 |
| Triton | 3.7.1 |

PyTorch publishes separate CPU/CUDA wheels, so choose the matching index.

### WSL CUDA setup

~~~bash
# Inside Ubuntu. A compiler and matching Python headers are needed for Triton's
# small first-use driver shim.
sudo apt update
sudo apt install -y build-essential python3-dev python3-venv

python3 -m venv ~/.venvs/tiktok-techjam-2026
~/.venvs/tiktok-techjam-2026/bin/python -m pip install --upgrade pip
~/.venvs/tiktok-techjam-2026/bin/python -m pip install +  torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
~/.venvs/tiktok-techjam-2026/bin/python -m pip install +  numpy==2.5.2 pytest==9.1.1
~~~

The verified minimal WSL image had no sudo-capable compiler. It uses a
user-scoped Zig 0.16.0 fallback plus extracted libpython3.14-dev headers.
tools/triton-cc prefers gcc/clang and otherwise uses that fallback. Set
TRITON_PYTHON_DEV_ROOT if the extracted headers live elsewhere.

From Windows, scripts/run-wsl.ps1 finds the Ubuntu user and uses
~/.venvs/tiktok-techjam-2026. Override it when needed:

~~~powershell
$env:TIKTOK_TECHJAM_PYTHON = "/path/to/venv/bin/python"
~~~

### CPU-only development

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.5.2 pytest==9.1.1
python -m pytest tests -q
~~~

GPU tests skip on CPU; they do not become GPU evidence.

## Reproduce

From Windows PowerShell:

~~~powershell
# Entire CPU + GPU suite on the verified WSL environment
powershell -ExecutionPolicy Bypass -File scripts/run-wsl.ps1 -m pytest tests -q

# One direct benchmark using the competition integration point
powershell -ExecutionPolicy Bypass -File scripts/run-wsl.ps1 torch_transformer_benchmark.py --device cuda --dtype float32 --attention-backend auto --accuracy-trials 5

# Full manifest: raw samples and explicit PASS/FAIL/OOM/ERROR accounting
powershell -ExecutionPolicy Bypass -File scripts/run-wsl.ps1 benchmarks/run_matrix.py --device cuda --attention-backend auto --accuracy-trials 5 --out results/matrix.json

# Profiler proof
powershell -ExecutionPolicy Bypass -File scripts/run-wsl.ps1 benchmarks/profile_cases.py --case long-causal-padding --dtype float32 --attention-backend auto --steps 5 --out results/profile.json --trace results/profile-trace.json
~~~

The matrix runner fails closed:

- unexpected exceptions are ERROR;
- allocation failures are OOM;
- numerical failures are FAIL;
- zero executed cases fail;
- only a nonempty all-PASS result exits zero.

sweep.py remains as a compatibility entry point for the same runner.

## Repository layout

~~~text
torch_transformer_benchmark.py        reference + required optimized class
transformer_opt/
  config.py                           support envelope and launch policy
  dispatch.py                         custom/SDPA/reference routing
  kernels/attention.py                Triton fused attention
benchmarks/
  official_shapes.json                provisional machine-readable matrix
  run_matrix.py                       fail-closed correctness/performance runner
  profile_cases.py                    profiler proof
  reference/manifest.json             frozen benchmark fingerprint
tests/                                CPU contract + direct/end-to-end GPU tests
docs/
  REQUIREMENTS.md                     source-of-truth and acceptance criteria
  KERNEL_DESIGN.md                    kernel algorithm and trade-offs
  TECH_REPORT.md                      measured technical report
  results/                            curated raw evidence
DEMO_RUNBOOK.md                       public walkthrough sequence
~~~

## Limitations

- The final organizer matrix and any post-workshop benchmark revision remain
  external unknowns and must be reconciled before submission.
- The kernel is forward/inference only.
- Tuning evidence is specific to the RTX 5070 Ti.
- Float16/bfloat16 deep-stack auto runs prioritize exact benchmark correctness
  over fused speed.
- There is no production deployment because Track 3 explicitly excludes it.

For a short public demo, follow [the demo runbook](DEMO_RUNBOOK.md).
