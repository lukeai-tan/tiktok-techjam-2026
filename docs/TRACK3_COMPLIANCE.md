# Track 3 Compliance Matrix

Audit date: 2026-08-27 (Asia/Singapore)

## Verdict

**PASS for every repository-owned requirement in the checked-in Track 3
brief. HOLD for final submission.** The code contains a repository-owned GPU
kernel, passes the stricter executable correctness rule, records target-GPU
performance and profiler proof, and supplies the required written materials.

Three items cannot be completed truthfully from repository code alone:

1. the organizer's final shape combinations and latest pristine benchmark are
   not available in this checkout;
2. the GitHub URL returned HTTP 404 to an unauthenticated check on the audit
   date, so it is not yet a verified public repository; and
3. a human must record/upload the public YouTube demo and publish the Devpost
   entry.

`PASS` below means the requirement has executable or documentary evidence.
`HOLD` means an external artifact or organizer input is still required.

## Problem-statement requirements

| Brief requirement | Status | Repository evidence | Remaining action |
| --- | --- | --- | --- |
| Submit one or more GPU kernels implementing the fixed Transformer layer | PASS | `transformer_opt/kernels/attention.py` owns the Triton `_attention_fwd` kernel; `torch_transformer_benchmark.py` integrates it in `UserOptimizedTransformer`. | None for the checked-in contract. |
| Use either PyTorch or TensorFlow and choose which operations to fuse | PASS | PyTorch path selected; QK, online softmax, causal/padding masking, and P@V are fused in one launch. | None. |
| Keep output within relative error 0.02 and absolute error 0.002 | PASS | The project enforces the stricter executable rule: abs <= 0.001 **or** relative <= 0.01. The curated run has 35/35 passing trials and zero failed elements. | Rerun if the organizer changes the executable rule. |
| Handle varied batch, sequence, and dimension shapes; shape checks are allowed | PASS / HOLD | Seven provisional shapes cover small/large batch, sequence, width, causal, and padding regimes. Dispatch guards device, dtype, head width, sequence limit, stride, and gradients. | HOLD only for the unpublished final organizer combinations. |
| Use AI-assisted development | PASS | `docs/TECH_REPORT.md` records OpenAI Codex and the inherited Claude Code prototype, the tasks performed, and the evidence policy. | Add any human/team details on Devpost. |
| Optimize and test on the participant's own GPU | PASS | Curated CUDA-event and profiler artifacts were produced locally on an RTX 5070 Ti. | Retest on the evaluation GPU if it differs. |
| Provide a clear technical report including AI skills/tools | PASS | `docs/TECH_REPORT.md` covers contract, environment, bottleneck analysis, kernel design, methods, results, AI use, rejected work, and limitations. | None. |
| Download the benchmark, customize the implementation, run locally, and report CPU/GPU/disk/optimizations/results | PASS / HOLD | The exact checked-in base blob and hashes are frozen in `benchmarks/reference/manifest.json`; environment and raw results are embedded in `docs/results/`. | HOLD for confirming byte identity against any newer organizer download. |
| In scope: code generation, fusion, profiling; production deployment is not required | PASS | Custom Triton, packed-QKV inference, CUDA events, PyTorch profiler, tests, and fail-closed result tooling are included; no production deployment is claimed. | None. |

## Deliverables

| Deliverable | Status | Evidence | Remaining action |
| --- | --- | --- | --- |
| Devpost project description: problem, tools, APIs, libraries, data/assets | PASS as draft | `docs/DEVPOST_DESCRIPTION.md` contains every requested section and measured claims. | Paste/publish it on Devpost and add the final video URL. |
| Public, structured, commented code repository | HOLD | Source, tests, CI, setup, reproduction, limitations, and contribution guidance are present. The configured GitHub URL was not publicly reachable during this audit. | Make the repository public and verify it while signed out. |
| README overview, setup, reproduction, limitations, team contributions | PASS | `README.md` contains all five topics; it does not invent contributor names not established by repository evidence. | Replace the solo/team statement only if verified team details apply. |
| Short public YouTube end-to-end demo linked from Devpost | HOLD | `DEMO_RUNBOOK.md` gives a verified 3-5 minute recording sequence with tests, benchmark, profiler proof, and content-safety guidance. | Human records, uploads publicly, checks signed-out playback, and adds the link. |
| No unauthorized trademarks or copyrighted content in the demo | READY | The runbook restricts the recording to owned code/terminal output and permitted challenge material, with no third-party music, logos, or unrelated assets. | Follow the runbook during recording. |

## Judging readiness

| Criterion | Weight | Evidence-backed position | Last-mile opportunity |
| --- | ---: | --- | --- |
| Technical execution | 35% | Custom online-softmax Triton kernel, measured dispatcher, strict correctness, packed-QKV optimization, 66-test target suite, fail-closed benchmark, raw evidence, and profiler proof. | Rerun the final organizer matrix unchanged. |
| Innovation and problem insight | 20% | Eliminates quadratic score/mask intermediates, consumes projection-native BSHD strides, reproduces reference rounding boundaries, and uses measured shape-aware routing instead of forcing custom code everywhere. | Explain these three decisions visually in the demo. |
| Impact and relevance | 20% | The measured path lowers end-to-end latency and cuts long-attention incremental allocation by 71.8%, directly relevant to Transformer inference cost and sequence scalability. | Frame the result around lower latency, memory pressure, and serving capacity. |
| Feasibility and practicality | 15% | Explicit support envelope, exact fallbacks, state-dict compatibility, bounded packed-weight cache, reproducible environment, and honest forward-only limitation. | Demonstrate fallback behavior once in Q&A if asked. |
| Presentation and communication | 10% | README, technical report, kernel design, Devpost copy, compliance matrix, and demo runbook form one consistent story. | Record the public demo and rehearse the measured claims. |

## Current measured gate

- 7 requested / 7 completed / 7 PASS; 0 FAIL, OOM, or ERROR.
- 35 accuracy trials; 13,117,440 checked output elements; zero failures.
- Maximum absolute error: `0.0009923577308654785`.
- Median end-to-end speedup range: 1.236x to 1.741x.
- Geometric-mean speedup: 1.498x.
- Auto timing dispatch: SDPA for two short unmasked cases and Triton for the
  five masked, causal, long, or wider-head cases.
- Long-attention incremental peak allocation: 78 MiB to 22 MiB (71.8%).
- Profiler: `_attention_fwd` count 10 for five two-layer forwards; Triton
  dispatch 10, SDPA 0, reference 0.
- Implementation fingerprint:
  `314dfa1615fe17b610d4851dd2a55377561f34b5a409762bf7fe43a4e5c196de`.

These values apply only to the checked-in provisional matrix and the exact
fingerprinted implementation. They are not a guarantee of placement or a
claim about unpublished organizer cases.
