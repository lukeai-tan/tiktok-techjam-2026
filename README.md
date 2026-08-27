# Transformer Layer GPU Kernel — TikTok TechJam 2026

A repository-owned Triton attention kernel for the Track 3 Transformer
benchmark. It fuses QK, online softmax, masking, and P@V without materializing
the quadratic attention matrix, then routes unsupported cases to explicit,
auditable fallbacks.

## Verified result

On an NVIDIA GeForce RTX 5070 Ti under native Windows 11, the current
provisional float32
matrix completed **7/7 PASS**, with **0 failed elements across 35 accuracy
trials / 13,117,440 output elements**. Measured auto-routing used SDPA for two
short unmasked cases and the custom Triton kernel for all five masked, causal,
long, or wide regimes.

| case | baseline median | optimized median | speedup |
| --- | ---: | ---: | ---: |
| tiny overhead | 0.478 ms | 0.312 ms | 1.532x |
| medium throughput | 0.502 ms | 0.311 ms | 1.612x |
| medium + padding | 0.654 ms | 0.488 ms | 1.340x |
| long causal | 0.694 ms | 0.462 ms | 1.500x |
| long causal + padding | 0.925 ms | 0.531 ms | 1.741x |
| long attention | 0.823 ms | 0.520 ms | 1.583x |
| wide model | 0.267 ms | 0.216 ms | 1.236x |

Geometric-mean end-to-end speedup: **1.498x**. The long-attention incremental
peak allocation fell from 78 MiB to 22 MiB (71.8%). The largest observed absolute
error was **0.000992358**, within the executable atol=0.001 OR rtol=0.01
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
- one cached QKV projection for measured eager-fp32 shapes up to d_model=512,
  with automatic invalidation and no state-dict changes;
- a measured fixed launch policy for head dimensions 16/32/64/128;
- measured auto-routing (short unmasked fp32 heads <=32 use SDPA; the other
  validated fp32 regimes use Triton); and
- observable forced triton, sdpa, and reference routing.

The primary optimized end-to-end path uses the benchmark-default float32 and
follows its TF32 toggle. Direct fp16 kernel tests pass, but the model's auto
mode keeps fp16/bf16 on exact reference-style math because fused
low-precision differences compound beyond this unusually strict deep-stack
tolerance. Forced modes remain available for honest comparison.

Algorithm, bounds, numerical choices, dispatch rules, and rejected
optimizations are documented in [the kernel design](docs/KERNEL_DESIGN.md).

## Known-good target environment

| component | version |
| --- | --- |
| CPU | AMD Ryzen 9 9950X, 16 cores / 32 logical processors |
| GPU | NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, 16,303 MiB |
| NVIDIA driver | 610.88 |
| OS | Windows 11, build 26200 |
| Python | 3.12.10 |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime | 13.0 |
| Triton | 3.7.1 |

PyTorch publishes separate CPU/CUDA wheels, so choose the matching index. On
Windows, the supported Triton package is `triton-windows`.

### Native Windows CUDA setup

~~~powershell
py -3.12 -m venv .venv
$python = ".venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
& $python -m pip install triton-windows==3.7.1.post27 numpy==2.5.2 pytest==9.1.1
~~~

A normal CPython installation supplies the headers and import library used by
Triton's first-use driver shim. The measured artifact used the same official
Python 3.12.10 runtime and package versions.

### Optional WSL CUDA setup

~~~bash
# Inside Ubuntu. A compiler and matching Python headers are needed for Triton's
# small first-use driver shim.
sudo apt update
sudo apt install -y build-essential python3-dev python3-venv

python3 -m venv ~/.venvs/tiktok-techjam-2026
~/.venvs/tiktok-techjam-2026/bin/python -m pip install --upgrade pip
~/.venvs/tiktok-techjam-2026/bin/python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
~/.venvs/tiktok-techjam-2026/bin/python -m pip install numpy==2.5.2 pytest==9.1.1
~~~

`tools/triton-cc` prefers gcc/clang and otherwise supports a user-scoped Zig
fallback. Set `TRITON_PYTHON_DEV_ROOT` if extracted Python headers live outside
the system include path. The current curated artifact is native Windows, not
WSL.

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
$python = ".venv\Scripts\python.exe"

# Entire CPU + GPU suite
& $python -m pytest tests -q

# One direct benchmark using the competition integration point
& $python torch_transformer_benchmark.py --device cuda --dtype float32 --attention-backend auto --accuracy-trials 5

# Full manifest: raw samples and explicit PASS/FAIL/OOM/ERROR accounting
& $python benchmarks/run_matrix.py --device cuda --attention-backend auto --accuracy-trials 5 --out results/matrix.json

# Profiler proof
& $python benchmarks/profile_cases.py --case long-causal-padding --dtype float32 --attention-backend auto --steps 5 --out results/profile.json --trace results/profile-trace.json
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
  TRACK3_COMPLIANCE.md                brief-to-evidence audit and external holds
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
- Packed QKV duplicates up to 6 MiB of derived float32 weights for a two-layer
  d_model=512 model; it is disabled where measurement did not justify the cost.
- There is no production deployment because Track 3 explicitly excludes it.

## Team contributions

Repository evidence does not establish additional human team members. If this
is a solo submission, the submitter owns the human contribution and the AI-tool
roles are documented in the technical report. If a team applies, add only
verified names and responsibilities here and on Devpost before submission.

## Submission status

The implementation is ready for the checked-in contract, but public-repository,
final-organizer-matrix, and YouTube/Devpost steps remain external holds. See the
[Track 3 compliance matrix](docs/TRACK3_COMPLIANCE.md) and follow the
[demo runbook](DEMO_RUNBOOK.md).
