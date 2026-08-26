# TEMP: Transformer GPU Kernel Gap Assessment and Implementation Plan

> Planning artifact only. Delete or archive this file after the work is incorporated into the permanent requirements, design, benchmark, and technical-report documents.

> **Execution update (2026-08-27): COMPLETED for the checked-in/provisional
> contract.** Permanent requirements, project context, kernel design, fail-closed
> benchmark tooling, GPU tests, profiler proof, curated results, technical
> report, CI, notebook, and demo runbook now supersede this assessment. The RTX
> 5070 Ti matrix passed 7/7 cases with a 1.360x geomean speedup. Historical facts
> below describe the pre-implementation audit and are intentionally retained as
> the rationale for the changes. Final organizer shapes remain external/unknown.

## 1. Decision

**Current verdict: HOLD — useful prototype, not yet a strong or evidenced solution to Track 3.**

The repository has a sensible correctness-oriented starting point: the optimized PyTorch model preserves the reference module structure and replaces explicit attention with `torch.nn.functional.scaled_dot_product_attention` (SDPA). However, the submission is not ready because:

1. no current GPU correctness, kernel-dispatch, memory, or performance evidence exists;
2. the default optimized path relies on a PyTorch-provided kernel rather than a repository-owned GPU kernel;
3. the only hand-written Triton kernel is an optional, disabled, standalone LayerNorm that the repository says has never been run on GPU;
4. benchmark provenance and the final required shape matrix are not frozen;
5. the sweep can silently skip runtime errors and still report success; and
6. the technical report still contains GPU placeholders.

The current work should be kept as a fallback/baseline, not discarded. The recommended competitive implementation is a shape-aware PyTorch solution with a real Triton fused-attention kernel as the primary custom kernel, SDPA as a verified fallback, and additional residual/normalization or FFN fusion only when profiling shows an end-to-end benefit.

## 2. Scope and evidence boundary

### In scope

- Track 3: "Implement a GPU Kernel for a Transformer Layer."
- The PyTorch submission path only; the challenge says either PyTorch or TensorFlow is sufficient.
- Inference correctness, custom-kernel implementation, shape-aware dispatch, GPU profiling, benchmarking, reproducibility, and submission documentation.
- Targeting the currently available host GPU first: NVIDIA GeForce RTX 5070 Ti, 16,303 MiB VRAM.

### Out of scope

- Production deployment, which the challenge explicitly excludes.
- Training/backward kernels unless the final organizer benchmark requires them.
- Replacing vendor GEMM kernels merely to claim more custom code; QKV, output, and FFN GEMMs should remain on cuBLAS/PyTorch unless profiling proves a better design.
- TensorFlow optimization unless the team explicitly switches tracks before implementation starts.

### Facts, assumptions, and unknowns

- **Fact:** `docs/hackathon-details.md:674-780` asks for one or more GPU kernels, numerical agreement, multiple input shapes, local GPU optimization, and a technical report.
- **Fact:** the current default implementation calls PyTorch SDPA (`torch_transformer_benchmark.py:221-247`).
- **Fact:** the Triton LayerNorm is enabled only with `TRANSFORMER_OPT_TRITON_LN=1` and is described as GPU-only and unrun (`transformer_opt/triton_impl.py:1-9`).
- **Fact:** `docs/TECH_REPORT.md:96-107` still has empty GPU result sections.
- **Fact:** the active Windows and WSL Python environments do not currently contain PyTorch; the Windows environment also lacks pytest and Triton. Therefore no repository correctness test or GPU benchmark was reproduced during this audit.
- **Fact:** `nvidia-smi` currently detects an RTX 5070 Ti with driver 610.47 and CUDA UMD 13.3.
- **Unknown:** whether the checked-in PyTorch file is byte-for-byte the latest organizer harness outside the customized class. Git history begins with the already-modified version and records no upstream checksum.
- **Unknown:** the final organizer shape combinations and target-evaluation rules. The checked-in challenge document is labelled early-bird material and points to the 28 August workshop.
- **Assumption for planning:** optimize inference on the RTX 5070 Ti in WSL2, while keeping portable fallbacks for other CUDA GPUs.

## 3. Source-of-truth preflight

There is no formal `docs/PRD.md` or `docs/PROJECT_CONTEXT.md`. Until the organizer publishes final materials, use this authority order:

1. pristine organizer benchmark and its comparison logic;
2. final organizer shape matrix and workshop clarifications;
3. `docs/hackathon-details.md`, section 3;
4. measured behavior on the target GPU;
5. this plan, README, and technical report.

Before implementation, obtain a fresh organizer copy of the selected PyTorch benchmark, record its source URL, retrieval date, and SHA-256, and preserve it read-only. Keep submission code separate from the reference harness. Do not treat README claims as stronger than benchmark or runtime evidence.

After the official details stabilize, create `docs/REQUIREMENTS.md` containing the exact formula, supported dtypes, input-shape matrix, masks, tolerances, timing method, allowed dependencies, and deliverables. This temporary plan is not a substitute for that file.

## 4. Current implementation assessment

### What is worth retaining

- The pre-LayerNorm residual structure matches the checked-in reference model.
- Strict state-dict compatibility avoids unfair or accidental weight differences.
- SDPA is a good optimized fallback and a useful performance comparator.
- The boolean attention mask follows SDPA's `True == attend` contract.
- Correctness is checked before timing in the main benchmark.
- CUDA timing in the main benchmark uses CUDA events and alternates model order across rounds.
- The CPU tests cover several shapes, causal/non-causal behavior, padding/no-padding ratios, multiple layers, and strict weight copy.
- The Colab notebook provides a practical remote-GPU route.

### Blocking findings

| ID | Finding | Evidence | Why it blocks readiness | Required response |
| --- | --- | --- | --- | --- |
| B1 | No current GPU evidence | `docs/TECH_REPORT.md:96-107,130-134`; local Python lacks PyTorch/Triton | The challenge is specifically GPU-performance work. CPU equivalence cannot validate CUDA SDPA, Triton compilation, dtype error, memory, or speed. | Build a supported WSL CUDA environment and run the official matrix on the RTX 5070 Ti. |
| B2 | Default path does not implement a repository-owned GPU kernel | `torch_transformer_benchmark.py:238-243` calls PyTorch SDPA; README calls custom attention future work | It is a framework optimization, but weak evidence for the literal and technical intent of "implement a GPU kernel." | Implement and exercise at least one real Triton/CUDA kernel on mandatory cases. |
| B3 | Existing custom kernel is optional and unverified | `transformer_opt/triton_impl.py:6-9`; `torch_transformer_benchmark.py:202-219` | Disabled code that has not compiled or run on the target cannot count as a delivered kernel. | Add direct GPU tests, benchmarking, constraints, dispatch evidence, and a safe fallback. |
| B4 | Benchmark provenance and final shapes are not frozen | no upstream checksum; PyTorch sweep uses seven custom shapes while the TensorFlow starter exposes a different dimension matrix | Optimizing against an altered or incomplete harness risks solving the wrong workload. | Preserve a pristine organizer harness and machine-readable official shape manifest before tuning. |
| B5 | The sweep can produce a false-green result | `sweep.py:200-230` catches every `RuntimeError`, labels it skip, and `all([])` is true | Compile failures, invalid kernels, and total OOM can be hidden as a successful run. | Use explicit PASS/FAIL/OOM/ERROR/SKIPPED states; unexpected errors fail; zero executed cases fail. |

### High-priority findings

| ID | Finding | Evidence / interpretation | Required response |
| --- | --- | --- | --- |
| H1 | Advertised SDPA fast-path use is not proven | PyTorch selects an SDPA backend at runtime. The code neither forces nor records the chosen backend. | Add backend capability checks and profiler evidence per shape; compare forced Flash, efficient, cuDNN, math, and custom paths where available. |
| H2 | The no-padding benchmark still passes an all-`True` tensor mask | `generate_random_case()` returns an all-valid mask when `padding_ratio <= 0` (`torch_transformer_benchmark.py:358-362`), so `_build_attn_mask()` does not take its `valid_token_mask is None` branch (`:249-267`). | Add an explicit semantic no-mask path without introducing a device-to-host synchronization; verify actual backend selection. |
| H3 | Causal + padding constructs a dense `[B,1,S,S]` mask | `torch_transformer_benchmark.py:260-267` | It reintroduces quadratic mask memory and may change backend eligibility, undermining the long-sequence story. | Handle causal and valid-key bounds inside the custom online-softmax kernel without materializing the dense mask. |
| H4 | LayerNorm is not fused with neighboring work | `transformer_opt/triton_impl.py:31-56` replaces one LayerNorm operation with another standalone operation | It may simply replace a mature vendor kernel without reducing launches or memory traffic. | Benchmark it directly; prefer residual-add + LayerNorm fusion if numerically safe. Remove/disable it for shapes where it loses. |
| H5 | Triton kernel has no documented support envelope or tuning | one next-power-of-two block, no dtype/width/contiguity guard, no autotune or launch configuration | Unsupported widths may fail; supported widths may be slower than PyTorch. | Define validated dimensions/dtypes/strides, choose launch parameters by shape, and fall back outside the envelope. |
| H6 | Tests validate CPU SDPA, not GPU kernels | `tests/test_correctness.py` uses CPU fp32 only and does not enable Triton or compile | Passing tests cannot support GPU-readiness claims. | Add GPU fp16/bf16/float32 tests, direct kernel tests, compile tests, multiple seeds/input scales, mask boundaries, and dispatch assertions. |
| H7 | Bottleneck analysis is generic rather than measured | `docs/TECH_REPORT.md:35-50` has no profiler trace or kernel breakdown | Kernel work may target an insignificant portion of end-to-end runtime. | Establish eager/compiled profiles, kernel-launch counts, occupancy-relevant metrics, and peak memory before choosing secondary fusion. |
| H8 | Benchmark reporting is not competition-grade | `sweep.py` times baseline then optimized rather than alternating, records limited environment data, and ignores result JSON in Git | Results can be biased or unreproducible. | Centralize a truthful benchmark runner with environment capture, alternating order, error states, raw samples, and committed curated reports. |

### Medium-priority findings

- `requirements.txt` has broad minimum versions and does not identify a known-good Python/PyTorch/CUDA/Triton combination for the target GPU.
- README setup is primarily POSIX/Colab-oriented and does not document the intended Windows + WSL2 path.
- There is no CI for CPU contract tests or lint/static checks; GPU validation can remain a manual required gate if hosted GPU CI is unavailable.
- The technical report does not yet include target-hardware environment, raw commands, profiler screenshots/tables, peak memory, variance, or failure/skip accounting.
- Deliverable work remains for team contributions, a public demo-video runbook, and Devpost-ready result summaries.

## 5. Recommended target architecture

```text
Pristine organizer harness
        |
        v
UserOptimizedTransformer
        |
        +--> shape/backend dispatcher
               |
               +--> custom Triton fused attention (primary deliverable)
               |      - online softmax; no S x S score or mask materialization
               |      - causal and valid-token bounds inside the kernel
               |      - fp16/bf16 first; shape-specific launch configs
               |
               +--> PyTorch SDPA fallback/reference competitor
               |
               +--> explicit baseline fallback for unsupported/debug cases
        |
        +--> optional residual-add + LayerNorm Triton kernel
        |
        +--> optional FFN epilogue fusion, only if profiling justifies it
```

Design rules:

1. Keep the organizer harness immutable and keep all custom code under `transformer_opt/`.
2. Use custom attention as the main technical contribution because explicit attention is the clearest long-sequence bottleneck and provides a meaningful kernel story.
3. Preserve SDPA as a strong fallback. A custom kernel is not automatically faster than PyTorch; dispatch only when measured wins are repeatable.
4. Avoid allocating dense attention scores or a dense causal+padding mask in the custom path.
5. Keep accumulations needed for softmax and normalization in fp32, then cast output to the requested dtype.
6. Make the support envelope explicit: device capability, dtype, head dimension, sequence range, layout/strides, causal flag, and padding representation.
7. Treat launch parameters and shape thresholds as versioned benchmark data, not unexplained constants.
8. Preserve exact reference semantics and a safe fallback for every mandatory input.

Useful primary references for implementation and validation:

- [PyTorch SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention) for backend-selection behavior and numerical caveats.
- [PyTorch `sdpa_kernel` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html) for forcing backends during verification.
- [Triton fused-attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) as a starting reference, not code that may be copied without understanding and adaptation.
- [Triton LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) for support guards, launch configuration, and benchmark structure.

## 6. Files that need to be created or changed

Names are recommendations; preserve the organizer's required integration point.

### Create

| Path | Purpose |
| --- | --- |
| `docs/REQUIREMENTS.md` | Final organizer formula, shapes, dtypes, mask rules, tolerances, timing rules, dependencies, and deliverables. |
| `benchmarks/reference/torch_transformer_benchmark.py` | Pristine, checksum-recorded organizer harness; read-only comparison source. |
| `benchmarks/official_shapes.json` | Machine-readable mandatory cases and provenance. |
| `transformer_opt/kernels/attention.py` | Custom forward fused-attention Triton kernel and thin validated wrapper. |
| `transformer_opt/kernels/residual_layernorm.py` | Optional fused residual-add + LayerNorm kernel if profiling supports it. |
| `transformer_opt/dispatch.py` | Auditable support checks and shape/backend selection with explicit fallbacks. |
| `transformer_opt/config.py` | Kernel configuration/support-envelope data, without hidden global environment switches. |
| `tests/test_gpu_attention.py` | Direct custom-attention GPU correctness and boundary tests. |
| `tests/test_gpu_transformer.py` | End-to-end official-matrix correctness across dtype/mask/causal/compile paths. |
| `tests/test_dispatch.py` | Positive and negative dispatch tests proving custom vs fallback selection. |
| `tests/test_triton_layernorm.py` | Direct correctness and support-envelope tests for retained normalization kernels. |
| `tests/test_sweep_integrity.py` | Tests that FAIL/ERROR/OOM/SKIPPED and zero-case runs cannot become green. |
| `tools/capture_environment.py` | Redacted environment, GPU, driver, CUDA, framework, dependency, and commit capture. |
| `benchmarks/run_matrix.py` | Single truthful correctness/performance runner with raw samples and explicit statuses. |
| `benchmarks/profile_cases.py` | Reproducible `torch.profiler`/Nsight case launcher and NVTX ranges. |
| `docs/KERNEL_DESIGN.md` | Algorithm, tiling, memory layout, numerical strategy, support envelope, fallbacks, and trade-offs. |
| `docs/results/README.md` | Result provenance and interpretation. |
| `docs/results/<gpu>-<date>.json` | Curated raw benchmark evidence committed with environment and revision metadata. |
| `.github/workflows/test.yml` | CPU contract tests and static checks; GPU gate may remain documented/manual. |
| `DEMO_RUNBOOK.md` | Exact end-to-end commands and capture sequence for the public demo video. |

### Change

| Path | Required change |
| --- | --- |
| `torch_transformer_benchmark.py` | Restrict changes to the designated optimized implementation/import seam; preserve and document organizer-harness parity. |
| `transformer_opt/triton_impl.py` | Replace with/re-export validated kernels, or retire the standalone LayerNorm if it has no measured win. Do not silently catch all initialization errors. |
| `transformer_opt/__init__.py` | Export stable wrapper/dispatcher interfaces only. |
| `sweep.py` | Either retire in favor of `benchmarks/run_matrix.py` or make status/error/timing/environment handling truthful. |
| `tests/test_correctness.py` | Retain fast CPU semantic tests; add no-mask, all-valid mask, boundary, multiple-seed, and adverse input-scale cases. |
| `requirements.txt` | Pin or constrain a known-good environment; separate runtime and developer dependencies if helpful. |
| `notebooks/colab_benchmark.ipynb` | Run the same manifest/runner as local validation and fail visibly on missing custom-kernel coverage. |
| `README.md` | Remove unverified performance wording, document supported environment, explain custom/fallback paths, and show verified reproduction commands. |
| `docs/TECH_REPORT.md` | Replace placeholders and generic expectations with measured, revision-specific evidence and limitations. |
| `.gitignore` | Continue ignoring scratch output but allow curated, provenance-rich result artifacts under `docs/results/`. |

## 7. Dependency-aware work plan

| Step | Owner capability | Work and artifacts | Depends on | Acceptance / evidence | Rework trigger |
| --- | --- | --- | --- | --- | --- |
| P0 | Requirements/intake | Obtain latest organizer scripts, workshop clarifications, official shape list, source URL, and SHA-256; create `docs/REQUIREMENTS.md` and immutable reference harness | organizer release/workshop | Exact requirements and harness provenance are recorded; local modifications are diffable | Any unresolved formula, tolerance, shape, or timing ambiguity |
| P1 | GPU/ML engineer | Create a supported WSL2 Python environment for RTX 5070 Ti; capture versions and run smoke kernels | P0 | CUDA PyTorch sees the GPU; a trivial CUDA op and Triton vector kernel compile/run; no unsupported-architecture warning | Import, compile, driver, or device-code failure |
| P2 | QA engineer | Repair benchmark truthfulness and result schema; add integrity tests | P0 | Unexpected exceptions fail; OOM is distinct; zero cases fail; all requested cases are accounted for | Any missing or misclassified case |
| P3 | Performance engineer | Run untouched baseline and current SDPA implementation; capture correctness, latency distribution, memory, and profiles | P1, P2 | Every official case has explicit status; backend/kernel names and bottleneck contribution are evidenced | Missing cases, unstable timing, or unproven backend |
| P4 | GPU-kernel engineer | Implement direct custom Triton attention with online softmax and causal/padding bounds | P0, P3 | Direct kernel matches reference for supported shapes/dtypes/seeds and allocates no dense score/mask tensor | Any numerical failure, unsupported mandatory case, or slower result without fallback |
| P5 | GPU-kernel engineer | Implement shape-aware dispatch and SDPA/reference fallbacks | P4 | Tests prove custom execution for intended cases and fallback for every unsupported case; no silent fallback | Profiler contradicts selected path or regression gate fails |
| P6 | GPU-kernel engineer | Evaluate residual+LayerNorm fusion; retain only measured wins | P3, P5 | Direct and end-to-end correctness pass; retained cases show repeatable end-to-end benefit | No statistically meaningful gain or support risk |
| P7 | GPU-kernel engineer | Evaluate FFN epilogue/compile/CUDA-graph options for remaining bottlenecks | P3, P5 | Optimization is justified by profile contribution and wins on its target regime | Complexity exceeds measured benefit |
| P8 | QA + performance | Execute full matrix across fp16/bf16/required fp32, causal/padding, compile/eager, seeds/scales; collect raw samples and memory | P5-P7 | Official tolerance passes with zero failed elements; every mandatory case accounted for; performance gates met | Any correctness failure, hidden skip, regression, or unstable result |
| P9 | Independent reviewer | Review requirements fidelity, kernel math, masks, bounds, dispatch, benchmark fairness, docs, and reproducibility | P8 | No blocking/high findings; all fixes retested against the same revision | Reproducible defect or unsupported claim |
| P10 | Technical writer/demo owner | Finalize README, design doc, report, Devpost text, result tables, limitations, AI-tool methodology, and demo runbook/video | P8, P9 | A clean environment reproduces documented results; no placeholder or unverified claim remains | Procedure fails or claim lacks raw evidence |
| P11 | Release gatekeeper | Issue submit/hold decision | P9, P10 | Required artifacts complete; results tied to code/environment; residual risks explicit | Any missing gate or pending human/organizer clarification |

The critical path is `P0 -> P1/P2 -> P3 -> P4 -> P5 -> P8 -> P9 -> P10 -> P11`. P1 and P2 can proceed independently once P0 freezes the contract. Documentation scaffolding can proceed early, but performance claims must wait for P8 evidence.

## 8. Correctness and performance gates

### Mandatory correctness gates

- Use the organizer's exact comparison function and tolerance semantics. Do not substitute `torch.isclose` if the organizer uses a different OR/AND rule.
- Pass every official shape with zero failed output elements for every required dtype, mask mode, and causal mode.
- Run multiple seeds and at least three input scales, including a scale that stresses softmax stability.
- Cover sequence/head dimensions at tile boundaries and non-power-of-two model dimensions if allowed by the official matrix.
- Cover `None`, all-valid, partially padded, minimum-valid-prefix, and invalid/unsupported mask cases according to the contract.
- Verify eager and compiled modes separately; a pass in one does not imply a pass in the other.
- Compare the direct Triton kernels with independent PyTorch reference operations before end-to-end tests.
- Prove the selected implementation path in tests/profiles; output equality alone cannot distinguish custom execution from silent fallback.
- Treat NaN, Inf, compile error, runtime error, missing case, and unexpected fallback as failures unless the official manifest explicitly permits a skip.

### Mandatory benchmark-integrity gates

- Record repository revision, dirty status, GPU name, compute capability, driver, CUDA runtime, PyTorch, Triton, Python, OS/WSL, dtype, flags, clocks/power assumptions, warmups, repetitions, and raw samples.
- Use GPU events or an equivalently synchronized method; exclude data/model construction and compilation from steady-state latency while reporting compile cost separately.
- Warm both paths, alternate order, use the same input and weights, and report median plus dispersion (for example p10/p90 or interquartile range).
- Record peak allocated/reserved memory and profiler kernel names for representative regimes.
- Account for every requested case as PASS, FAIL, OOM, ERROR, or explicitly justified SKIPPED.
- Preserve raw result artifacts; generated summary tables must be reproducible from them.

### Recommended performance release gate

The challenge sets no explicit minimum speedup, so this is a team recommendation rather than an organizer requirement:

- required: geometric-mean end-to-end speedup is significantly above `1.0x` on the official matrix;
- required: no mandatory case is below `0.95x` unless the dispatcher selects a faster fallback and the result clears the gate;
- target: at least `1.25x` geometric-mean speedup over the untouched eager baseline;
- required: long-sequence custom-attention cases show lower peak memory than explicit baseline attention;
- required: the custom path beats or provides a clearly different supported regime from SDPA on at least one meaningful mandatory case. Otherwise the custom-kernel story is technically valid but not competitive.

## 9. AI Council review

- **Requirements:** The safest interpretation is that a framework SDPA call alone is not enough for a strong Track 3 submission. Preserve it as fallback but deliver and prove a real kernel.
- **Architecture:** Separate immutable benchmark, custom kernels, dispatcher, and evidence. This prevents harness drift and makes fallback behavior auditable.
- **Implementation:** Start with attention, because causal+padding can be handled without a dense mask and the baseline explicitly materializes attention scores. Only add LayerNorm/FFN work after profiling.
- **Security/privacy:** No material data-security surface is present. Environment capture must still avoid `.env`, usernames, tokens, and unrelated process details.
- **Testing/QA:** Current CPU tests are useful but insufficient. GPU dtype/kernel/dispatch coverage and truthful error accounting are release blockers.
- **Docs/user experience:** The public repo needs one supported setup path, one authoritative runner, real result artifacts, and a demo that visibly exercises the custom kernel.
- **Devil's advocate:** PyTorch SDPA may outperform a hand-written Triton kernel on common shapes. The answer is measured shape dispatch and honest fallback, not forcing custom code everywhere or claiming an unevidenced win.

## 10. Release checklist

- [ ] Latest organizer requirements and pristine harness are checksum-recorded.
- [ ] Final official shape matrix is machine-readable and fully accounted for.
- [ ] RTX 5070 Ti WSL2 environment is reproducible and supports the selected PyTorch/Triton versions.
- [ ] At least one repository-owned GPU kernel compiles, runs, and is proven active.
- [ ] Direct kernel and end-to-end correctness gates pass on GPU.
- [ ] No unexpected fallback, error, OOM, or missing case is reported as green.
- [ ] Baseline, SDPA, custom, and compiled variants are fairly benchmarked.
- [ ] Performance and peak-memory evidence are committed with provenance.
- [ ] Independent review has no unresolved blocking/high findings.
- [ ] README, kernel design, technical report, Devpost text, and demo have no placeholders or unsupported claims.
- [ ] Final submit/hold decision is recorded for the exact tested revision.

## 11. Immediate next action

Do **P0 first**, then **P1 and P2 in parallel**:

1. attend/check the 28 August Track 3 workshop materials;
2. download and checksum the final organizer PyTorch harness and shape list;
3. create the WSL2 CUDA Python environment for the detected RTX 5070 Ti; and
4. fix the sweep's false-green behavior before generating any performance table.

Only after those steps should kernel implementation begin. This prevents optimizing an unverified harness or producing benchmark numbers that cannot support a submission claim.
