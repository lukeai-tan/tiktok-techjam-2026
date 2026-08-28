# Selected Submission Validation

Status: complete; approved for repo-local submission selection with residuals

Date: 2026-08-28 (Asia/Singapore)

Base commit: `b41fdaf90f869a920346401b2b9fd93899fe805e`

Selected implementation SHA-256:
`de768f1ff9ddee54a9ad83a67f3e1f205044c0ad5c723fc3bb4881093c97f611`

## Selection outcome

The repository has no separate candidate registry. Its actual submission entry
is `torch_transformer_benchmark.py::UserOptimizedTransformer`, which
`benchmarks/run_organizer_torch.py` injects into the byte-preserved organizer
PyTorch harness. Independent schema-2 fingerprint recomputation showed that
this live root entry already equals the requested SHA-256 across all 13
fingerprinted implementation paths. No implementation byte was copied or
rewritten, because doing so would have changed the selected fingerprint or
created a false selector.

The untouched organizer downloads remain frozen at:

- PyTorch: `1bd12523657f338c09b53f0bb9052d9d16f728a71bd22bc8298567e1a4d78c22`;
- TensorFlow: `00e99b6e1d19e961039b66eb3d3c055b36cc50f0436da2558f5f1fbe292ef798`.

Five focused integration tests independently checked those hashes, protected
baseline definitions, CPU harness injection, final-row transcription, and the
row-11 profile manifest before GPU benchmarking.

## Full measured suite

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

## Residual performance finding

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

The independent review and AI Council approve the exact fingerprint for the
repo-local submission entry. This is not approval for a public release or a
claim that every shape is faster. Required controls are to retain the held-out
slowdown disclosure, regenerate evidence after any fingerprint change, retest
on a different evaluator GPU, and obtain separate authority for every Git or
public-submission action.
