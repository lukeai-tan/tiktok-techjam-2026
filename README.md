# Transformer Layer GPU Kernel — TikTok TechJam 2026

A repository-owned Triton attention kernel for the Track 3 Transformer
benchmark. It fuses QK, online softmax, masking, and P@V without materializing
the quadratic attention matrix, then routes unsupported cases to explicit,
auditable fallbacks.

## Verified result

On an NVIDIA GeForce RTX 5070 Ti under native Windows 11, the
organizer-published final shape table completed **13/13 executable PASS**, with
the one source-authorized 100,000-token resource case excluded from the pass
count. Under the recorded PyTorch assumptions (float32, no padding, and the
stricter executable comparator), all 65 accuracy trials passed with **0 failed
elements across 938,885,120 comparisons**.

The selected local submission is
`torch_transformer_benchmark.py::UserOptimizedTransformer`, with schema-2
implementation SHA-256
`de768f1ff9ddee54a9ad83a67f3e1f205044c0ad5c723fc3bb4881093c97f611`.
Fresh final-matrix geometric-mean end-to-end speedup is **1.776x**. Per-row
speedups range from 1.013x to 5.456x; the Campaign 4 target (row 11,
`head_dim=8`, sequence 128) measures **5.456x**. Backend accounting records
1,120 Triton calls, 336 explicit reference calls for unsupported or very-large
batch regimes, and zero SDPA calls. The complete table, stdout, environment,
source hashes, and implementation fingerprint are in
[the selected-submission artifact](docs/results/rtx-5070-ti-2026-08-28-submission-final.json).

A second complete run of the same implementation reproduced the contract and
backend counts at **1.770x**. Two project-owned held-out runs are **7/7 PASS at
1.210x and 1.266x**, the untouched organizer default is **5/5 PASS at 1.352x**,
and the source-derived matrix is **28/28 executable PASS at 1.203x** plus the
same authorized non-pass skip. The held-out non-padded long-causal case is a
reproduced local slowdown at about 0.80x; the published final-matrix claim does
not hide or average away that limitation. These are separate evidence gates,
not substitutes for the final matrix.

The complete optimization history now has one canonical entry point:
[what was tried, every campaign aggregate, rejected and failed work, score
evolution, and why Campaign 4 is best](docs/experiments/OPTIMIZATION_HISTORY.md).
Campaigns 2-4 retain all **114** immutable attempt records: 105 passing child
commands, 9 failed child commands, and 788.748 seconds of measured child wall
time. Submission selection adds **25** `S1-*` records: 24 passing commands, one
retained workflow-schema failure, and 223.238 seconds of child wall time. No
rejected or failed evidence was deleted during consolidation or selection.

The final dimensions are published, but dtype, padding, timing, tolerance, and
backward policy remain unstated. See [the requirements](docs/REQUIREMENTS.md)
for the exact assumptions and [the result index](docs/results/README.md) for all
current reproducibility commands.

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
- a measured fixed launch policy for head dimensions 8/16/32/64/128, with
  zero-masked 16-lane dot padding for width eight;
- measured auto-routing (short unmasked fp32 heads <=32 use SDPA; deep six-layer
  causal or batch-above-8 cases use accuracy-safe SDPA; low precision,
  unsupported head widths, and causal batches above 128 use exact reference
  math; validated custom fp32 regimes use Triton); and
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
| CPU | AMD Ryzen 7 9850X3D, 8 cores / 16 logical processors |
| GPU | NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, 16,303 MiB |
| NVIDIA driver | 616.56 |
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

# Organizer-published final rows: 13 executable plus one declared skip
& $python benchmarks/run_organizer_validation.py `
  --matrix benchmarks/final_evaluator_shapes.json `
  --out results/final-evaluator-validation.json

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
  final_evaluator_shapes.json         organizer-published final shape rows
  final_profile_shapes.json           representative final profiler cases
  official_shapes.json                project-owned held-out matrix
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
notebooks/colab_benchmark.ipynb       fingerprint-pinned full-suite Colab workflow
DEMO_RUNBOOK.md                       public walkthrough sequence
~~~

## Limitations

- The final shape dimensions are published, but their dtype, padding, timing,
  tolerance, backward policy, and any post-workshop revision remain external
  unknowns. Current live attachment bytes could not be freshly checksummed.
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

The repo-local submission entry is selected at the fingerprint above. It passes
all 13 executable final rows, both project-held-out confirmations, the untouched
organizer default, all 28 feasible source-derived organizer validation cases,
and the complete 117-test CPU/GPU suite. The immutable validation artifacts were
captured before Git packaging and therefore record a dirty local candidate;
committing or pushing this checkpoint does not relabel those measurements as a
clean run. Organizer policy clarification and YouTube/Devpost steps remain
external holds. See the
[Track 3 compliance matrix](docs/TRACK3_COMPLIANCE.md) and follow the
[demo runbook](DEMO_RUNBOOK.md).
