# Track 3 Compliance Matrix

Use the [documentation hub](../docs/README.md) for the shortest route through the
evidence. This file is the PASS/HOLD release audit, not another campaign
narrative.

Audit date: 2026-08-29 (Asia/Singapore)

## Verdict

**PASS for every repository-owned requirement in the checked-in Track 3
brief. HOLD for final submission.** The code contains a repository-owned GPU
kernel, passes the stricter executable correctness rule, records target-GPU
performance and profiler proof, and supplies the required written materials.

Three external items cannot be completed truthfully from repository code alone:

1. the organizer's final dimensions are published, but dtype, padding, timing,
   tolerance, backward policy, and current attachment-byte identity remain
   unconfirmed;
2. the GitHub URL returned HTTP 404 to an unauthenticated check on the prior audit
   date, so it is not yet a verified public repository; and
3. a human must record/upload the public YouTube demo and publish the Devpost
   entry.

`PASS` below means the requirement has executable or documentary evidence.
`HOLD` means an external artifact or organizer input is still required.

## Problem-statement requirements

| Brief requirement | Status | Repository evidence | Remaining action |
| --- | --- | --- | --- |
| Submit one or more GPU kernels implementing the fixed Transformer layer | PASS | `transformer_opt/kernels/attention.py` owns the Triton `_attention_fwd` kernel; `transformer_opt/submission.py` integrates it in `UserOptimizedTransformer` while reusing the untouched organizer harness. | None for the checked-in contract. |
| Use either PyTorch or TensorFlow and choose which operations to fuse | PASS | PyTorch path selected; QK, online softmax, causal/padding masking, and P@V are fused in one launch. | None. |
| Keep output within relative error 0.02 and absolute error 0.002 | PASS | The project enforces the stricter executable rule: abs <= 0.001 **or** relative <= 0.01. The exact-harness matrix has 140/140 passing trials and zero failed elements across 459,776,000 comparisons. | Rerun if the organizer changes the executable rule. |
| Handle varied batch, sequence, and dimension shapes; shape checks are allowed | PASS / HOLD | All 13 executable published final rows pass with zero failures across 938,885,120 comparisons; all 28 feasible source-derived cases also pass. The exact 100,000-token resource row is an authorized skip, not a pass. | HOLD only for the final table's unstated execution policy. |
| Use AI-assisted development | PASS | `docs/TECH_REPORT.md` records OpenAI Codex and the inherited Claude Code prototype, the tasks performed, and the evidence policy. | Add any human/team details on Devpost. |
| Optimize and test on the participant's own GPU | PASS | Curated CUDA-event and profiler artifacts were produced locally on an RTX 5070 Ti. | Retest on the evaluation GPU if it differs. |
| Provide a clear technical report including AI skills/tools | PASS | `docs/TECH_REPORT.md` covers contract, environment, bottleneck analysis, kernel design, methods, results, AI use, rejected work, and limitations. | None. |
| Download the benchmark, customize the implementation, run locally, and report CPU/GPU/disk/optimizations/results | PASS | Both local Lark downloads are byte-preserved in `benchmarks/` and frozen by `organizer_downloads.json`. The fresh untouched PyTorch default passed 5/5 trials at 1.385x, the final dimensions passed 13/13 executable rows twice, and the source-derived matrix passed 28/28 executable cases on the recorded RTX 5070 Ti. | Recheck if the organizer publishes a later revision or confirms changed attachment bytes. |
| In scope: code generation, fusion, profiling; production deployment is not required | PASS | Custom Triton, packed-QKV inference, CUDA events, PyTorch profiler, tests, and fail-closed result tooling are included; no production deployment is claimed. | None. |

## Deliverables

| Deliverable | Status | Evidence | Remaining action |
| --- | --- | --- | --- |
| Devpost project description: problem, tools, APIs, libraries, data/assets | PASS as draft | `hackathon-docs/DEVPOST_DESCRIPTION.md` contains every requested section and measured claims. | Paste/publish it on Devpost and add the final video URL. |
| Public, structured, commented code repository | HOLD | Source, tests, CI, setup, reproduction, limitations, and contribution guidance are present. The configured GitHub URL was not publicly reachable during this audit. | Make the repository public and verify it while signed out. |
| README overview, setup, reproduction, limitations, team contributions | PASS | `README.md` contains all five topics; it does not invent contributor names not established by repository evidence. | Replace the solo/team statement only if verified team details apply. |
| Short public YouTube end-to-end demo linked from Devpost | HOLD | `docs/guides/DEMO_RUNBOOK.md` gives a verified 3-5 minute recording sequence with tests, benchmark, profiler proof, and content-safety guidance. | Human records, uploads publicly, checks signed-out playback, and adds the link. |
| No unauthorized trademarks or copyrighted content in the demo | READY | The runbook restricts the recording to owned code/terminal output and permitted challenge material, with no third-party music, logos, or unrelated assets. | Follow the runbook during recording. |

## Judging readiness

| Criterion | Weight | Evidence-backed position | Last-mile opportunity |
| --- | ---: | --- | --- |
| Technical execution | 35% | Custom online-softmax attention and fused residual/LayerNorm Triton kernels, measured dispatcher, strict correctness, packed-QKV optimization, full CPU/GPU suite, 13-row final proof, 28-case isolated validation, raw evidence, and profiler proof. | Retest unchanged on the evaluator GPU if it differs. |
| Innovation and problem insight | 20% | Eliminates quadratic score/mask intermediates, consumes projection-native BSHD strides, reproduces reference rounding boundaries, and uses measured shape-aware routing instead of forcing custom code everywhere. | Explain these three decisions visually in the demo. |
| Impact and relevance | 20% | The current final matrix measures 1.977x geomean speedup, row 9 measures 1.780x in the complete matrix and its controlled optimized median falls 12.05%, row 5 measures 2.314x, row 11 measures 6.377x, exact-width packed QKV reduces row-8 model profile time, exact-row-9 fusion reduces mean residual/normalization profile time 41.77%, and long-attention incremental allocation falls 71.8%. | Frame the result around lower latency, memory pressure, serving capacity, and measured shape-aware routing. |
| Feasibility and practicality | 15% | Explicit support envelope, exact fallbacks, state-dict compatibility, bounded packed-weight cache, reproducible environment, and honest forward-only limitation. | Demonstrate fallback behavior once in Q&A if asked. |
| Presentation and communication | 10% | README, technical report, kernel design, Devpost copy, compliance matrix, and demo runbook form one consistent story. | Record the public demo and rehearse the measured claims. |

## Current measured gate

- Organizer-published final dimensions: 13/13 executable PASS, zero failures
  across 938,885,120 comparisons over 65 trials, plus one authorized resource
  skip excluded from the pass count; 1.977420x geomean speedup and 1.986499x
  complete confirmation.
- Final attention dispatch: Triton 1,260 / SDPA 0 / reference 196. Rows 5, 9,
  and 11 are 2.314x, 1.780x, and 6.377x in the primary matrix. Dedicated
  300/100/300/300-sample runs put rows 5, 6, 9, and 11 at 1.880x, 1.546x,
  1.150x, and 4.710x. Campaign 11's row-9 fusion cuts controlled optimized
  latency 12.05% and mean subsystem profile time 41.77% while retaining the
  29,360,128-byte incremental peak allocation.
- Untouched organizer PyTorch default: 5/5 PASS, 0/2,621,440 failed elements,
  1.385x median speedup, and Triton 1,950 / SDPA 0 / reference 0 optimized
  attention calls.
- Source-derived exact-harness matrix: 28/28 executable PASS, 0/459,776,000
  failed elements over 140 trials, plus one authorized resource skip excluded
  from the pass count; 1.206505x overall geomean speedup.
- Project-held-out matrix: two 7/7 PASS runs, each with zero failures across
  13,117,440 comparisons, at 1.339847x and 1.386495x geomean. Four complete
  matrices keep long-causal in a 1.198x-1.204x band; four padded measurements
  span 1.213x-1.335x. Exact-shape SDPA removed both held-out long-causal regressions and remains the measured route.
- Complete CPU/GPU suite: 148/148 tests passed; 14 upstream PyTorch deprecation
  warnings were reported and no required coverage was removed.
- Long-attention incremental peak allocation: 78 MiB to 22 MiB (71.8%).
- Implementation fingerprint:
  `908a0d708cd8f70f44d5f14fda93d3cafb1cc18345f43914e715594cfa7b7ef9`.

These values apply only to the exact fingerprinted implementation, recorded
RTX 5070 Ti environment, published dimensions, and explicit PyTorch assumptions.
They are not a guarantee of placement or a claim about omitted organizer policy.
