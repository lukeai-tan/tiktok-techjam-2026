# Flagship vs `fix/google-colab-accuracy-issue`

Date: 2026-08-28  
Hardware: NVIDIA GeForce RTX 5070 Ti, compute capability 12.0  
Verdict: retain the flagship implementation. The alternate branch is not an
accuracy-valid or evidence-complete submission candidate.

## Outcome

The checked-out flagship passed every executable published-final and
source-derived case: 41/41 executable cases across those two matrices, with
zero failed elements out of 1,398,661,120 comparisons. The alternate branch
passed the one organizer-default FP32 case, but failed 4/13 published-final
rows and all 13 reduced-precision source-derived rows. Its two broad matrices
contained 1,008,950 failed elements in total.

On the accuracy-valid published-final rows, the flagship's optimized latency
was 14.2% lower by direct geometric mean in the unmodified comparison run and
it was faster on 8/9 rows. The evaluation-only telemetry repeat gave the same
conclusion: 14.5% lower latency and faster on 9/9 rows. The flagship was also
13.3% lower-latency on the organizer default. The broad source-derived FP32
subset was effectively near parity: the two candidate runs moved from 2.5%
ahead to 1.3% behind the flagship, a smaller difference than the observed 3.9%
candidate run-to-run shift.

The alternate branch therefore has no defensible aggregate performance win to
trade against its correctness failures. Under the fail-closed contract, timing
from an accuracy-failing row is never accepted.

## Frozen comparison subjects

| Subject | Git ref | Commit | Root implementation blob |
| --- | --- | --- | --- |
| Flagship | `feat/transformer-gpu-kernel-implementation` | `b529b8356118a71ecb961280c533c0f9f863324f` | `31bacc24e8f17e24e41378bd311e0d999a708862` |
| Alternate | `origin/fix/google-colab-accuracy-issue` | `651fe8ee9a19a36d8d36c4135c525161114f7cae` | `5083fb4ebf6f83acd2b89a0cf62ae067f927dbb4` |

Their merge base is
`f9ac6d4caf9c71138685637d3d8d5c5fd7fd8c68`. The alternate branch's unique
commit adds historical FP16/BF16 result files; its implementation itself is an
older SDPA prototype relative to the flagship.

The flagship evidence fingerprint is
`de768f1ff9ddee54a9ad83a67f3e1f205044c0ad5c723fc3bb4881093c97f611`.
For an apples-to-apples harness comparison, the exact alternate implementation
files were overlaid into a detached temporary worktree containing the current
frozen harness. That synthetic evaluation fingerprint is
`20c3c74144c0b1b6b095e82f2af51f53b51f232ffd6fc891abf2442bcd689354`;
it is deliberately not presented as a releasable branch fingerprint.

## Method and controls

- Runtime: Python 3.12.10, PyTorch 2.13.0+cu130, Triton 3.7.1, CUDA 13.0,
  NVIDIA driver 616.56.
- Comparator: an element passes when absolute error is at most 0.001 or
  relative error is at most 1%; a case requires zero failed elements.
- Published-final and source-derived cases use five independent accuracy
  trials. Timing runs only after accuracy passes.
- The exact source-authorized 100,000-token stress row remains a resource skip
  and is never counted as a pass.
- Each executable matrix case runs in an isolated subprocess. The same current
  organizer runner, manifests, seeds, warmups, repeats, and timing mechanism
  were used for both implementations.
- Run order was counterbalanced: alternate then flagship for organizer-default,
  flagship then alternate for published-final, and alternate then flagship for
  source-derived.
- Every benchmark attempt, including failed attempts, has an immutable JSON
  record with command, environment, fingerprint, wall time, output, metrics,
  and artifact hash.

The alternate implementation predates the harness's
`attention_backend_counts` interface. Its raw accuracy-valid rows therefore
finish as formal `ERROR`, although their parsed accuracy and timing remain in
the artifacts. To distinguish that telemetry defect from numerical failure,
two separate evaluation-only runs added a source-backed counter around the
existing unconditional `torch.nn.functional.scaled_dot_product_attention`
call. No tensor operation, mask, weight, tolerance, seed, or timing policy was
changed. The adapted fingerprint is
`3ffd8505e2b5ace13a854d4273fa3b7b7631d1c62f12dfbb9ee18569d939dd53`.
Those runs are retained separately and were rejected rather than integrated.

## Organizer-default result

| Implementation | Accuracy | Failed elements | Max abs | Baseline median | Optimized median | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Flagship | PASS | 0 / 2,621,440 | 0.00100136 | 1.9441 ms | 1.3587 ms | 1.431x |
| Alternate | PASS | 0 / 2,621,440 | 0.000710778 | 1.9724 ms | 1.5670 ms | 1.259x |

Both are accurate on this single FP32 case. The flagship is 1.153x faster
directly, equivalent to 13.3% lower optimized latency. The alternate result has
no backend-counter evidence; source inspection and the adapted runs establish
that it uses SDPA.

## Published-final matrix

| Metric | Flagship | Alternate, raw | Alternate, telemetry-adapted |
| --- | ---: | ---: | ---: |
| Executable cases passing | 13 / 13 | 0 / 13 formal | 9 / 13 |
| Telemetry-only errors | 0 | 9 | 0 |
| Accuracy-failing cases | 0 | 4 | 4 |
| Failed elements | 0 / 938,885,120 | 24 / 938,885,120 | 24 / 938,885,120 |
| Maximum absolute error | 0.00114846 | 0.00137609 | 0.00137609 |
| Valid-set speedup vs own baseline | 1.794x across all 13 | 1.858x across only 9 | 1.861x across only 9 |
| Observed backends | 1,120 Triton; 336 reference; 0 SDPA | missing | SDPA only |

The alternate valid-set speedup is not comparable to the flagship's complete
13-row number because all four invalid rows are excluded. Direct optimized
latency on the nine common accuracy-valid rows is the valid comparison:

| Case | Alternate accuracy | Flagship optimized ms | Alternate optimized ms | Alternate / flagship |
| --- | --- | ---: | ---: | ---: |
| `final-01-b64-d128-h4-s128` | PASS (0 failed) | 0.8520 | 1.0141 | 1.190x |
| `final-02-b1-d128-h4-s128` | PASS (0 failed) | 0.8518 | 0.9925 | 1.165x |
| `final-03-b4-d128-h4-s128` | PASS (0 failed) | 0.7918 | 0.8512 | 1.075x |
| `final-04-b16-d128-h4-s128` | PASS (0 failed) | 0.8088 | 0.8968 | 1.109x |
| `final-05-b128-d128-h4-s128` | PASS (0 failed) | 1.8333 | 1.9055 | 1.039x |
| `final-06-b10000-d128-h4-s128` | FAIL (21 failed) | 607.4203 | not timed | - |
| `final-07-b64-d32-h4-s128` | FAIL (1 failed) | 1.2945 | not timed | - |
| `final-08-b64-d1024-h4-s128` | FAIL (1 failed) | 15.7252 | not timed | - |
| `final-09-b64-d128-h1-s128` | PASS (0 failed) | 0.9101 | 0.8778 | 0.965x |
| `final-10-b64-d128-h2-s128` | FAIL (1 failed) | 0.8892 | not timed | - |
| `final-11-b64-d128-h16-s128` | PASS (0 failed) | 1.1426 | 1.9602 | 1.716x |
| `final-12-b64-d128-h4-s32` | PASS (0 failed) | 0.7892 | 0.8158 | 1.034x |
| `final-13-b64-d128-h4-s1024` | PASS (0 failed) | 19.6848 | 26.4592 | 1.344x |

The flagship is faster on 8/9 valid rows in the raw run. The alternate is 3.5%
lower-latency only on row 9. The largest flagship advantage is row 11: 1.716x
directly, or 41.7% lower latency. Its only complete-matrix slowdown versus its
own baseline is row 8 at 0.999x, which remains accuracy-valid; the alternate
fails that row and cannot supply a trusted timing.

## Source-derived matrix

| Dtype | Cases | Flagship | Alternate | Alternate failed elements | Alternate max abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Float32 | 15 | 15 PASS | 15 PASS after telemetry adapter | 0 / 232,509,440 | 0.00118431 |
| Float16 | 12 | 12 PASS | 12 FAIL | 404,366 / 224,645,120 | 0.0117188 |
| BFloat16 | 1 | 1 PASS | 1 FAIL | 604,560 / 2,621,440 | 0.0625 |
| Total | 28 | 28 PASS | 15 PASS, 13 FAIL | 1,008,926 / 459,776,000 | 0.0625 |

The flagship has zero failed elements in all 459,776,000 comparisons and a
1.2266x complete-matrix geometric-mean speedup. The alternate is correct only
for the FP32 subset; its 1.4686x speedup is computed over those 15 rows alone
and cannot be extended to the invalid reduced-precision rows.

Direct latency over the 15 common accuracy-valid FP32 rows is statistically
indecisive at the precision of these single runs. The unmodified candidate run
is 2.5% lower-latency by geometric mean and wins 8/15 rows. The counter-adapted
repeat is 3.9% slower than that raw candidate run and leaves the flagship 1.3%
lower-latency, winning 6/15 rows. This spread brackets parity and does not alter
the decisive correctness result.

## Backend and implementation interpretation

The alternate branch replaces explicit attention with PyTorch SDPA for every
shape. Its optional Triton LayerNorm is disabled by default, and its own source
does not expose backend counts. The flagship uses the repository Triton
attention kernel on eligible high-value shapes, SDPA where the dispatcher
selects it, and exact reference fallbacks where fusion is not accuracy-safe or
profitable. That policy explains the complete accuracy coverage and the large
published-row gains without claiming fused execution on fallback rows.

The alternate branch's own historical Tesla T4 artifacts corroborate the fresh
reduced-precision result but are not mixed into the RTX comparison. At commit
`651fe8ee9a19a36d8d36c4135c525161114f7cae`, `docs/fp16.json` records all 7/7
rows failing with maximum absolute error between 0.005859375 and 0.0078125;
`docs/bf16.json` records all 7/7 failing with maximum absolute error between
0.046875 and 0.0625.

## Immutable attempt ledger

| Attempt | Outcome | Wall time | Result |
| --- | --- | ---: | --- |
| [`BC1-CANDIDATE-001-ORGANIZER-DEFAULT`](attempts/BC1-CANDIDATE-001-organizer-default.json) | PASS | 3.236 s | [JSON](../results/rtx-5070-ti-2026-08-28-branchfix-candidate-organizer-default.json) |
| [`BC1-FLAGSHIP-001-ORGANIZER-DEFAULT`](attempts/BC1-FLAGSHIP-001-organizer-default.json) | PASS | 3.169 s | [JSON](../results/rtx-5070-ti-2026-08-28-branchfix-flagship-organizer-default.json) |
| [`BC1-FLAGSHIP-002-FINAL`](attempts/BC1-FLAGSHIP-002-final.json) | PASS | 64.754 s | [JSON](../results/rtx-5070-ti-2026-08-28-branchfix-flagship-final.json) |
| [`BC1-CANDIDATE-002-FINAL`](attempts/BC1-CANDIDATE-002-final.json) | FAIL: 9 telemetry errors, 4 accuracy failures | 31.996 s | [JSON](../results/rtx-5070-ti-2026-08-28-branchfix-candidate-final.json) |
| [`BC1-CANDIDATE-003-SOURCE-DERIVED`](attempts/BC1-CANDIDATE-003-source-derived.json) | FAIL: 15 telemetry errors, 13 accuracy failures | 61.146 s | [JSON](../results/rtx-5070-ti-2026-08-28-branchfix-candidate-source-derived.json) |
| [`BC1-FLAGSHIP-003-SOURCE-DERIVED`](attempts/BC1-FLAGSHIP-003-source-derived.json) | PASS | 67.451 s | [JSON](../results/rtx-5070-ti-2026-08-28-branchfix-flagship-source-derived.json) |
| [`BC1-CANDIDATE-004-ADAPTED-FINAL`](attempts/BC1-CANDIDATE-004-adapted-final.json) | FAIL: 9 PASS, 4 accuracy failures | 32.561 s | [JSON](../results/rtx-5070-ti-2026-08-28-branchfix-candidate-adapted-final.json) |
| [`BC1-CANDIDATE-005-ADAPTED-SOURCE-DERIVED`](attempts/BC1-CANDIDATE-005-adapted-source-derived.json) | FAIL: 15 PASS, 13 accuracy failures | 62.140 s | [JSON](../results/rtx-5070-ti-2026-08-28-branchfix-candidate-adapted-source-derived.json) |
| [`BC1-CLOSE-001-COLAB-NOTEBOOK`](attempts/BC1-CLOSE-001-colab-notebook.json) | PASS: focused notebook structure, syntax, fingerprint, suite, and token checks | 0.219 s | - |
| [`BC1-CLOSE-002-FULL-TESTS`](attempts/BC1-CLOSE-002-full-tests.json) | PASS: intermediate 116-test suite | 8.355 s | - |
| [`BC1-CLOSE-003-CANDIDATE-PROVENANCE`](attempts/BC1-CLOSE-003-candidate-provenance.json) | PASS: all three alternate blobs match commit | 0.138 s | - |
| [`BC1-CLOSE-004-GRAPH-REBUILD`](attempts/BC1-CLOSE-004-graph-rebuild.json) | PASS: 24 file notes and 69 events rebuilt | 0.122 s | - |
| [`BC1-CLOSE-005-GRAPH-VALIDATE`](attempts/BC1-CLOSE-005-graph-validate.json) | PASS: 0 errors, 0 warnings | 0.129 s | - |
| [`BC1-CLOSE-006-ARTIFACT-VALIDATION`](attempts/BC1-CLOSE-006-artifact-validation.json) | PASS: fingerprints, hashes, metrics, and report bindings | 0.239 s | - |
| [`BC1-CLOSE-007-FINAL-TESTS`](attempts/BC1-CLOSE-007-final-tests.json) | PASS: final 117-test suite, 14 deprecation warnings | 8.355 s | - |
| [`BC1-CLOSE-008-FINAL-GRAPH-REBUILD`](attempts/BC1-CLOSE-008-final-graph-rebuild.json) | PASS: final 24 file notes and 70 events rebuilt | 0.115 s | - |
| [`BC1-CLOSE-009-FINAL-GRAPH-VALIDATE`](attempts/BC1-CLOSE-009-final-graph-validate.json) | PASS: final graph has 0 errors and 0 warnings | 0.127 s | - |
| [`BC1-CLOSE-010-FINAL-ARTIFACT-LINKS`](attempts/BC1-CLOSE-010-final-artifact-links.json) | PASS: final hashes, metrics, closure records, and 25 relative links | 0.252 s | - |
| [`BC1-CLOSE-011-EXACT-FINAL-TESTS`](attempts/BC1-CLOSE-011-exact-final-tests.json) | PASS: exact final test code, 117 passed with 14 deprecation warnings | 8.465 s | - |

The one-line notebook syntax preflight initially failed before execution because
PowerShell mangled nested quoting. It made no repository change and was replaced
by the durable `tests/test_colab_notebook.py` check; benchmark evidence was not
affected.

## Colab update

`notebooks/colab_benchmark.ipynb` now targets the checked-out flagship branch
and fails before testing if its implementation fingerprint differs from
`de768f1ff9ddee54a9ad83a67f3e1f205044c0ad5c723fc3bb4881093c97f611`.
It also rejects a reused clone with local changes, preserves the temporary
`GIT_ASKPASS` token flow, runs the full pytest suite, organizer default,
published-final matrix, source-derived matrix, project held-out matrix, and two
profiler cases, then bundles every JSON result and trace for download. The
checked-in notebook and comparison-artifact regression tests bring the final
repository suite to 117 passing tests.

Colab measurements remain GPU-specific and must not be merged numerically with
the RTX 5070 Ti evidence above.

## Decision

Keep the flagship submission entry and implementation fingerprint. Do not merge
the alternate branch's transformer implementation or its older Colab workflow.
The flagship is the only compared implementation that satisfies strict
correctness, complete backend evidence, and the complete published performance
matrix.
