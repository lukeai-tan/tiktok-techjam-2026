# Selected Submission Validation

Start with the [documentation hub](../README.md) for the reading order. This
file is the release-gate record for the selected fingerprint, not a replacement
for the campaign history or raw result artifacts.

Status: complete through Campaign 11; approved for repo-local submission selection with external holds

Date: 2026-08-30 (Asia/Singapore)

Base commit: `8c89d1d4170c58d16fb75d79f212e990565fba7d`

Packaged Campaign 11 evidence SHA-256:
`908a0d708cd8f70f44d5f14fda93d3cafb1cc18345f43914e715594cfa7b7ef9`

The Campaign 11 pre-packaging candidate itself was selected under historical
fingerprint
`9c326536ea27cfc619f01531152b2c82986d9dc3f4274691d3e8191bbb0804eb`.
Adapter packaging and canonical benchmark relocation produced the measured
`908a0d...` identity above without changing the optimized Transformer math.

## Selection outcome

The actual submission entry is
`transformer_opt/submission.py::UserOptimizedTransformer`, which
`benchmarks/run_organizer_torch.py` injects into the byte-preserved organizer
PyTorch harness. Campaign 11 extends the accepted fused residual/LayerNorm
forward only to exact final row 9 while retaining the exact-shape and eval-mode boundary. The
schema-2 packaged fingerprint above binds the Campaign 11 source, final matrices, held-out
matrices, organizer-default run, source-derived run, profiles, and tests.

The untouched organizer downloads remain frozen at:

- PyTorch: `5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e`;
- TensorFlow: `00e99b6e1d19e961039b66eb3d3c055b36cc50f0436da2558f5f1fbe292ef798`.

## 2026-08-30 validation hardening

Documentation review found that the profiler had derived its expected Triton
state from the backend counts it was supposed to verify. The maintenance change
makes expectations independent command inputs, requires both a positive Triton
dispatch count and `_attention_fwd` profiler evidence, and gives fused
residual/LayerNorm its own required event gate. Wrapped attempt summaries now
honor that final validation status instead of treating any attention event as a
pass. The organizer runner rejects same-name modules loaded from another path
and refuses diagnostic non-strict weight copying for evidence output.

These tooling changes produce current schema-2 fingerprint
`a186b679885e9e787b3deba0ad710855ae4c2486ae491b53e4e64bfa13e7f9cf`;
optimized math and dispatch behavior did not change; one source comment now
states the packed-QKV gradient-state boundary accurately. Current validation
recorded:

- 164/164 repository tests passing with 14 upstream deprecation warnings;
- strict organizer execution at explicit `atol=0.001`, `rtol=0.01`: 5/5 PASS,
  zero failed elements; its maintenance timing is retained as diagnostic output
  and does not replace the fingerprinted Campaign 11 performance artifacts;
- positive row-9 proof: 120 Triton dispatches and `_attention_fwd` calls, 240
  `_residual_layer_norm_fwd` calls, and `validation_passed: true`;
- negative row-9 proof expecting reference: exit 1,
  `validation_passed: false`, and wrapped attempt `metrics.status: FAIL`;
- observational row-9 control with no declared expectation: exit 0,
  `validation_passed: null`, and wrapped attempt `metrics.status: INCONCLUSIVE`;
  and
- evidence plus `--non-strict-weight-copy`: rejected before benchmark execution.

The fresh files are disposable maintenance outputs under ignored `results/`.
They do not overwrite or relabel Campaign 11's immutable `908a0d...` timing
artifacts.

Fifty-one preflight contract tests independently checked those hashes,
protected baseline definitions, harness injection, final-row transcription,
logger behavior, and evidence policy before candidate integration.

## Current Campaign 11 measured suite

| Gate | Correctness | Performance / backend |
| --- | --- | --- |
| Complete final primary | 13/13 executable PASS + exact non-pass skip; 0/938,885,120 failed | 1.977420x; row 5 2.314x; row 9 1.780x; row 11 6.377x; Triton 1,260 / SDPA 0 / reference 196 |
| Complete final confirmation | same zero-failure contract and backend counts | 1.986499x |
| Untouched organizer default | 5/5 PASS; 0/2,621,440 failed | 1.385x; 1,950 Triton calls |
| Held-out primary / confirmation | 7/7 PASS twice; 0/13,117,440 failed each | 1.339847x / 1.386495x |
| Held-out stability | four measured-fingerprint matrices plus one 300-sample target run; zero failures | long-causal 1.198x-1.204x; SDPA-only; aggregate geomean 1.340x-1.515x |
| Source-derived | 28/28 executable PASS + exact non-pass skip; 0/459,776,000 failed | 1.206505x; Triton 672 / SDPA 1,344 / reference 2,688 |
| Exact row 9 long/profile | five trials; zero failures; 240 fused calls in each profile | 0.717648 ms; controlled optimized latency -12.05%; mean subsystem time -41.77%; memory-neutral |
| Exact row 5 long | five trials; zero failures | 1.163168 ms, 1.880066x, memory-neutral |
| Exact row 6 long | five trials; zero failures | 188.457397 ms, 1.546330x, memory-neutral |
| Exact row 11 long/profile | five trials; zero failures; 240 fused calls in profile | 0.890672 ms, 4.710116x; profile model time -22.09%; memory-neutral |
| Complete pytest | 148/148 PASS; 14 upstream warnings | CPU and CUDA coverage; no required test removed |

Campaign 11's immutable attempt ledger and wall-time aggregate are recorded in
[CAMPAIGN-011](CAMPAIGN-011.md). Campaign 10 remains the immutable before-state
and row-9 control. The detailed table below is retained as the
historical 2026-08-28 selection baseline; its older fingerprint and slowdown
finding are not current submission evidence.

## Historical 2026-08-28 measured suite

| Gate | Correctness | Performance / backend | Child wall time |
| --- | --- | --- | ---: |
| Complete pytest | 112/112 PASS; 14 upstream deprecation warnings | CPU and CUDA coverage; no required test removed | 8.096423 s |
| Direct root entry | 5/5 PASS; 0/2,621,440 failed | 1.195x; Triton 1,950 / SDPA 0 / reference 0 | 3.104835 s |
| Untouched organizer default | 5/5 PASS; 0/2,621,440 failed | 1.352x; Triton 1,950 / SDPA 0 / reference 0 | 3.011792 s |
| Final primary | 13/13 executable PASS + exact non-pass skip; 0/938,885,120 failed | 1.775778x; row 11 5.456x; Triton 1,120 / SDPA 0 / reference 336 | 51.468772 s |
| Final confirmation | same zero-failure contract and backend counts | 1.770185x; row 11 5.408x | 51.641385 s |
| Held-out primary | 7/7 PASS; 0/13,117,440 failed over 35 trials | 1.210009x; Triton 900 / SDPA 400 / reference 0 | 3.052503 s |
| Held-out confirmation | same zero-failure contract | 1.266010x; same backend counts | 3.025856 s |
| Source-derived | 28/28 executable PASS + exact non-pass skip; 0/459,776,000 failed over 140 trials | 1.203466x; Triton 672 / SDPA 1,344 / reference 2,688 | 64.045062 s |
| Integrated row-11 profile | profiler gate PASS | 40 Triton calls; `_attention_fwd` 40 calls / 4,763.665 us | 2.713192 s |

The two full final geomeans differ by 0.315%, while correctness and backend
counts are identical. Their selected-submission JSON is
[primary](../results/rtx-5070-ti-2026-08-28-submission-final.json) and
[confirmation](../results/rtx-5070-ti-2026-08-28-submission-final-confirmation.json).

## Historical residual performance finding

The held-out aggregate remains above baseline, but the non-padded
`long-causal` case reproduced below baseline at 0.793x and 0.800x. The padded
variant measured 0.941x then 0.992x. Both runs remained numerically exact under
the checked comparator, used the intended Triton route, and preserved memory
benefits; the confirmation's long-causal incremental allocation fell 50.27%,
and long-attention fell 71.79%. This is a real target-machine limitation, not a
correctness failure, and it is not present in the organizer-published final
matrix.

## Immutable attempt ledger

Every child command was run through `benchmarks/run_optimization_attempt.py`.
The first workflow validation failure is retained rather than erased.

| Attempt | Result | Wall time | What it established |
| --- | --- | ---: | --- |
| S1-PREFLIGHT-001-WORKFLOW | FAIL | 0.047139 s | Canonical closure sections and graph-title wording were missing |
| S1-PREFLIGHT-002-WORKFLOW-REWORK | PASS | 0.048078 s | Corrected workflow validates with zero errors/warnings |
| S1-PREFLIGHT-003-FINGERPRINT | PASS | 1.268184 s | Expected equals actual SHA-256 across 13 paths |
| S1-PREFLIGHT-004-ENTRY-CONTRACT | PASS | 1.563987 s | Five protected-source and integration tests pass |
| S1-SUITE-001-FULL-PYTEST | PASS | 8.096423 s | Complete 112-test suite passes |
| S1-SUITE-002-DIRECT-ENTRY | PASS | 3.104835 s | Direct root entry is correct and timed |
| S1-SUITE-003-ORGANIZER-DEFAULT | PASS | 3.011792 s | Untouched default harness is correct and timed |
| S1-SUITE-004-FINAL-PRIMARY | PASS | 51.468772 s | Complete published final matrix primary |
| S1-SUITE-005-FINAL-CONFIRMATION | PASS | 51.641385 s | Complete final-matrix reproduction |
| S1-SUITE-006-HELDOUT | PASS | 3.052503 s | Held-out correctness, timing, backend, and memory |
| S1-SUITE-007-SOURCE-DERIVED | PASS | 64.045062 s | Complete supplied-contract breadth gate |
| S1-SUITE-008-PROFILE-ROW11 | PASS | 2.713192 s | Custom-kernel profiler proof |
| S1-SUITE-009-HELDOUT-CONFIRMATION | PASS | 3.025856 s | Countercheck and residual slowdown confirmation |
| S1-CLOSE-001-CURATED-ARTIFACTS | PASS | 1.625872 s | Fresh curated pointers and invariants pass |
| S1-CLOSE-002-FULL-PYTEST | PASS | 7.647732 s | Post-integration 113-test suite passes |
| S1-CLOSE-003-PROVENANCE | PASS | 1.616577 s | Attempt/result SHA and fingerprint links pass |
| S1-CLOSE-004-FULL-PYTEST | PASS | 7.771741 s | Provenance-integrated 114-test suite passes |
| S1-CLOSE-005-DOC-EVIDENCE | PASS | 1.653092 s | Submission docs and slowdown disclosure pass |
| S1-CLOSE-006-FINAL-PYTEST | PASS | 7.698775 s | Final 115-test CPU/GPU suite passes |
| S1-CLOSE-007-GRAPH-REBUILD | PASS | 0.122727 s | Graph rebuilds to 55 events |
| S1-CLOSE-008-GRAPH-VALIDATE | PASS | 0.119298 s | Pre-decision graph has 0 errors/warnings |
| S1-CLOSE-009-WORKFLOW-FINAL | PASS | 0.049152 s | Completed workflow has 0 errors/warnings |
| S1-CLOSE-010-GRAPH-FINAL-REBUILD | PASS | 0.108309 s | Final decision graph rebuilds to 56 events |
| S1-CLOSE-011-GRAPH-FINAL-VALIDATE | PASS | 0.117969 s | Final strict graph has 0 errors/warnings |
| S1-CLOSE-012-FINAL-DOC-EVIDENCE | PASS | 1.619356 s | Final artifact, provenance, and documentation invariants pass |

Measured child-command total: **223.237808 seconds** across 25 attempts,
24 passing commands and one retained planning-schema failure. This includes
every preflight, test, benchmark, profile, provenance, documentation, workflow,
and graph command executed as part of selection validation.

## Scope and external holds

This is a repo-local selection and validation. At validation time the working
tree was dirty on the stated base commit and no commit, branch, tag, history
rewrite, push, public-repository mutation, or Devpost action had been
authorized. Later Git packaging does not alter the immutable run provenance.
Organizer dtype, padding, timing, backward, and post-workshop policy remain
external unknowns exactly as recorded in `docs/REQUIREMENTS.md`.

## Release decision

The independent review and AI Council approve the current exact fingerprint for the
repo-local submission entry. This is not approval for a public release or a
claim that every shape is faster. Required controls are to retain the held-out
stability evidence, regenerate evidence after any fingerprint change, retest
on a different evaluator GPU, and obtain separate authority for every Git or
public-submission action.
