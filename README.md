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
| tiny overhead | 0.484 ms | 0.312 ms | 1.552x |
| medium throughput | 0.519 ms | 0.318 ms | 1.630x |
| medium + padding | 0.653 ms | 0.490 ms | 1.332x |
| long causal | 0.696 ms | 0.462 ms | 1.507x |
| long causal + padding | 0.923 ms | 0.527 ms | 1.752x |
| long attention | 0.813 ms | 0.518 ms | 1.570x |
| wide model | 0.256 ms | 0.208 ms | 1.230x |

Geometric-mean end-to-end speedup: **1.501x**. The long-attention incremental
peak allocation fell from 78 MiB to 22 MiB (71.8%). The largest observed absolute
error was **0.000992358**, within the executable atol=0.001 OR rtol=0.01
contract.

Raw samples, environment metadata, implementation fingerprint, and dispatch
counts are in
[the matrix artifact](docs/results/rtx-5070-ti-2026-08-27.json).
[Profiler evidence](docs/results/rtx-5070-ti-2026-08-27-profile.json)
records _attention_fwd ten times for five two-layer forwards.

The newly supplied organizer PyTorch file is also preserved untouched. Running
that exact harness at its default six-layer configuration produced **5/5 PASS,
0 failed elements out of 2,621,440, and 1.411x median speedup**. All 1,950
optimized attention calls used Triton. See the
[exact-harness artifact](docs/results/rtx-5070-ti-2026-08-27-organizer-default.json)
and [organizer-input audit](docs/ORGANIZER_INPUTS.md).

The rigorous source-derived validation then ran the untouched PyTorch
parser/comparator/timer in a fresh process for every feasible dimension signal
from both supplied files: **28/28 executable cases passed**, with **0 failed
elements out of 459,776,000 across 140 accuracy trials**. The matrix covers
float32, float16, bfloat16, causal attention, prefix padding, batch sizes through
10,000, model widths through 1,024, head counts 1/2/4/16, and sequence length
1,024. Overall geomean speedup was **1.258x**; the float32 subset was **1.509x**.
The TensorFlow script's designated 100,000-token quadratic stress case is the
single source-authorized resource skip and is not counted as a pass. See the
[validation artifact](docs/results/rtx-5070-ti-2026-08-27-organizer-validation.json).

The project matrix remains explicitly **provisional**: both organizer benchmark
files are now present, but the supplied PyTorch file exposes one configurable
case rather than the promised final evaluator matrix. See
[the requirements](docs/REQUIREMENTS.md) for the evidence boundary.

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
- measured auto-routing (short unmasked fp32 heads <=32 use SDPA; deep six-layer
  causal or batch-above-8 cases use accuracy-safe SDPA; validated default and
  smaller masked fp32 regimes use Triton); and
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

# Strongest contract proof: untouched organizer parser/comparator/timing harness
& $python benchmarks/run_organizer_torch.py --device cuda

# Rigorous supplied-contract matrix: 28 executable cases plus one declared skip
& $python benchmarks/run_organizer_validation.py `
  --out results/organizer-validation.json

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

## Repository layout

~~~text
torch_transformer_benchmark.py        reference + required optimized class
transformer_opt/
  config.py                           support envelope and launch policy
  dispatch.py                         custom/SDPA/reference routing
  kernels/attention.py                Triton fused attention
benchmarks/
  torch_transformer_benchmark.py      untouched organizer PyTorch download
  tensorflow_transformer_benchmark.py untouched organizer TensorFlow download
  run_organizer_torch.py              inject submission into untouched harness
  organizer_validation_matrix.json   source-derived validation policy
  run_organizer_validation.py        isolated exact-harness matrix runner
  official_shapes.json                provisional machine-readable matrix
  run_matrix.py                       fail-closed correctness/performance runner
  profile_cases.py                    profiler proof
  reference/manifest.json             frozen benchmark fingerprint
  reference/organizer_downloads.json  exact supplied-file checksums/contracts
tests/                                CPU contract + direct/end-to-end GPU tests
docs/
  ORGANIZER_INPUTS.md                 received/missing organizer resource audit
  REQUIREMENTS.md                     source-of-truth and acceptance criteria
  KERNEL_DESIGN.md                    kernel algorithm and trade-offs
  TECH_REPORT.md                      measured technical report
  TRACK3_COMPLIANCE.md                brief-to-evidence audit and external holds
  results/                            curated raw evidence
notebooks/colab_benchmark.ipynb       fail-closed Colab reproduction workflow
DEMO_RUNBOOK.md                       public walkthrough sequence
~~~

## Limitations

- The two benchmark downloads are reconciled, but the final PyTorch evaluator
  matrix and any post-workshop revision remain external unknowns.
- The supplied TensorFlow benchmark's 100,000-token quadratic stress case is
  preflight-skipped exactly as its own resource policy allows.
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

The implementation passes the project matrix, untouched organizer default, and
all 28 feasible source-derived organizer validation cases. Public-repository,
final-evaluator-matrix, and YouTube/Devpost steps remain external holds. See the
[Track 3 compliance matrix](docs/TRACK3_COMPLIANCE.md) and follow the
[demo runbook](DEMO_RUNBOOK.md).
