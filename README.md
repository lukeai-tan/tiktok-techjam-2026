# Transformer Layer GPU Kernel — TikTok TechJam 2026

A repository-owned Triton attention kernel for the Track 3 Transformer
benchmark. It fuses QK, online softmax, masking, and P@V without materializing
the quadratic attention matrix, then routes unsupported cases to explicit,
auditable fallbacks.

For the shortest path through the repository, start with the
[documentation hub](docs/README.md). It separates the executive campaign
outcome, implementation contract, reproducibility artifacts, and submission
holds.

## Verified result

On an NVIDIA GeForce RTX 5070 Ti under native Windows 11, the
organizer-published final shape table completed **13/13 executable PASS**, with
the one source-authorized 100,000-token resource case excluded from the pass
count. Under the recorded PyTorch assumptions (float32, no padding, and the
stricter executable comparator), all 65 accuracy trials passed with **0 failed
elements across 938,885,120 comparisons**.

The selected local submission is
`transformer_opt/submission.py::UserOptimizedTransformer`, with schema-2
implementation SHA-256
`908a0d708cd8f70f44d5f14fda93d3cafb1cc18345f43914e715594cfa7b7ef9`.
Campaign 11's primary final-matrix geometric-mean end-to-end speedup is
**1.977x**; a second complete run measured **1.986x** with identical
correctness and aggregate backend counts. Final rows 6 and 7 use
accuracy-bounded hybrid execution: row 6 keeps its first two layers exact and
uses Triton for the last two, while row 7 keeps layer zero exact and uses
Triton for layers one through three. Exact rows 5 and 9 also use the guarded
fused residual-plus-LayerNorm path. Row 9's dedicated 300-sample optimized
median is **0.718 ms**, **12.05% below** two counterbalanced Campaign 10
controls and within 0.007% of the isolated candidate. The inherited row-5,
row-6, and row-11 long gates measure **1.880x**, **1.546x**, and **4.710x**.
Backend accounting is Triton 1,260 / SDPA 0 / reference 196. The complete
table, stdout, environment, source hashes, and fingerprint are in
[the Campaign 11 final artifact](docs/results/rtx-5070-ti-2026-08-29-c11-integrated-final.json).

Two five-seed project-owned held-out runs are **7/7 PASS at 1.340x and 1.386x**.
Two additional current-fingerprint rechecks and a 300-sample confirmation put
the exact long-causal SDPA route in a stable **1.198x-1.204x** band; the four
padded runs span **1.213x-1.335x**. The untouched organizer default is **5/5
PASS at 1.385x**, and the source-derived matrix is **28/28 executable PASS at
1.207x** plus the same
authorized non-pass skip. These are separate evidence gates, not substitutes
for the published final matrix.

Campaign 7 introduced a guarded fused residual-add plus LayerNorm route for
exact row 6. Campaign 8 reused that kernel for exact row 11, and Campaign 10
extended it to exact row 5. Campaign 11 now extends it to exact row 9. The route explicitly
blocks training mode as well as gradients, compilation, unsupported layouts,
dtypes, masks, devices, and neighboring shapes. Over 30 row-11 forwards the
active profile preserves 240 fused launches plus 30 native norms. On row 9,
two active profiles reduce mean residual/normalization device time **41.77%**
versus two Campaign 10 controls, while top-level profiler time remains noisy;
the counterbalanced 300-sample CUDA-event gate measures the causal **12.05%**
optimized-latency reduction. Peak allocation remains 29,360,128 bytes.
Campaign 7's rejected head-width-256 route remains closed.

Campaign 11 is the selected successor. Its preflight, baselines,
candidate/integrated profiles, bounded screens, rejected variants,
confirmations, counterbalanced controls, integration matrices, and failed gates
remain under `docs/experiments/attempts/`; no evidence was deleted. Use the
[campaign run-through](docs/experiments/CAMPAIGN_RUN_THROUGH.md) for the readable
outcome, the [optimization history](docs/experiments/OPTIMIZATION_HISTORY.md)
for chronology and accounting, and the [result index](docs/results/README.md)
for artifact-level reproduction.

The final dimensions are published, but dtype, padding, timing, tolerance, and
backward policy remain unstated. The exact assumptions are in the
[requirements](docs/REQUIREMENTS.md).

## Which campaign is the flagship?

Use **Campaign 11** and fingerprint
`908a0d708cd8f70f44d5f14fda93d3cafb1cc18345f43914e715594cfa7b7ef9`.
It is the current cumulative implementation, owns the latest complete
zero-failure final pair, and passed the 148-test repository gate. Campaign 5
is the strongest historical broad-generalization snapshot, Campaign 7 is the
best high-volume row-6 specialist, and the Campaign 4 plus Campaign 8 lineage
is the strongest single-row result. The [documentation hub](docs/README.md) and
[ranked comparison](docs/experiments/CAMPAIGN_RUN_THROUGH.md#flagship-and-strongest-specialist-campaigns)
explain the evidence and limits behind each label.

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
- one cached QKV projection for measured eager-fp32 shapes up to d_model=512
  and exact d_model=1024, with automatic invalidation and no state-dict changes;
- a measured fixed launch policy for head dimensions 8/16/32/64/128, with
  zero-masked 16-lane dot padding for width eight;
- measured auto-routing (short unmasked fp32 heads <=32 use SDPA; deep six-layer
  causal or batch-above-8 cases use accuracy-safe SDPA; low precision,
  unsupported head widths, and unmeasured causal batches above 128 use exact
  reference math; exact final rows 6 and 7 use layer-bounded Triton/reference
  hybrids; the measured held-out B2/S512/head64 causal envelope uses SDPA; other
  validated custom fp32 regimes use Triton); and
- exact-row fused residual plus LayerNorm execution for final rows 5, 6, 9, and 11,
  guarded to eval-mode eager CUDA float32 inference; and
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

# One direct benchmark using the untouched organizer harness and submission adapter
& $python benchmarks/run_organizer_torch.py --device cuda

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

The Colab notebook is pinned to `feat/jared-attempt` and the Campaign 11
fingerprint. That pin is committed on this branch; a clean GitHub reproduction
still requires access to the branch and a matching CUDA environment. The
notebook test proves its structure and pinning locally, not remote availability.

The matrix runner fails closed:

- unexpected exceptions are ERROR;
- allocation failures are OOM;
- numerical failures are FAIL;
- zero executed cases fail;
- only a nonempty all-PASS result exits zero.

## Repository layout

~~~text
transformer_opt/submission.py         optimized submission adapter
transformer_opt/
  config.py                           support envelope and launch policy
  dispatch.py                         custom/SDPA/reference routing
  kernels/attention.py                Triton fused attention
  kernels/residual_layer_norm.py      guarded fused residual plus LayerNorm
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
  README.md                             documentation map and reading order
  REQUIREMENTS.md                     source-of-truth and acceptance criteria
  KERNEL_DESIGN.md                    kernel algorithm and trade-offs
  TECH_REPORT.md                      measured technical report
  experiments/                         campaign narratives, ledgers, and attempts
  guides/                              operational demo procedure
  results/                            curated raw evidence
hackathon-docs/                       competition context and submission material
notebooks/colab_benchmark.ipynb       fingerprint-pinned full-suite Colab workflow
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
- Packed QKV duplicates about 6 MiB of derived float32 weights for a two-layer
  d_model=512 model and about 48 MiB for the measured four-layer d_model=1024
  row. The row-8 cache raises pre-forward allocated memory by 50,380,800 bytes;
  widths 513-1023 and wider unmeasured widths remain disabled.
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
and the complete **148/148-test** CPU/GPU suite. The immutable validation artifacts were
captured before Git packaging and therefore record a dirty local candidate;
committing or pushing this checkpoint does not relabel those measurements as a
clean run. Organizer policy clarification and YouTube/Devpost steps remain
external holds. See the
[Track 3 compliance matrix](hackathon-docs/TRACK3_COMPLIANCE.md) and follow the
[demo runbook](docs/guides/DEMO_RUNBOOK.md).
