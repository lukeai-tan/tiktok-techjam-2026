# Current Optimized vs Original Evaluation

Status: complete fresh comparison; all executable correctness gates passed

Date: 2026-08-28 (Asia/Singapore)

Commit: `b833f7292bf15680d0add6007a53f9f7bf747690`

Implementation fingerprint:
`de768f1ff9ddee54a9ad83a67f3e1f205044c0ad5c723fc3bb4881093c97f611`

## Comparison definition

“Original” is the byte-preserved organizer PyTorch `BaselineTransformer`.
“Current optimized” is
`torch_transformer_benchmark.py::UserOptimizedTransformer`. The harness copies
the original model's weights with `strict=True`, supplies identical inputs, runs
correctness before timing, and alternates baseline/optimized timing order. No
organizer source, tolerance, timing rule, implementation, or dispatch policy was
changed for this evaluation.

The strict executable comparator passes an element when absolute error is at
most `0.001` **or** relative error is at most `1%`. This is stricter than the
Track 3 prose. Passing therefore means numerical equivalence under the checked
contract, not universal bitwise identity.

Fresh target environment: NVIDIA GeForce RTX 5070 Ti, driver 616.56, Windows 11,
Python 3.12.10, PyTorch 2.13.0+cu130, CUDA 13.0, and Triton 3.7.1.

## Aggregate outcome

| Evaluation | Accuracy | Original vs optimized | Speedup | Equivalent median/geomean latency reduction |
| --- | --- | --- | ---: | ---: |
| Untouched organizer default | 5/5 trials PASS; 0/2,621,440 failed | 1.8046 ms vs 1.3454 ms | 1.341x | 25.43% |
| Published final matrix | 13/13 executable PASS + one authorized non-pass skip; 0/938,885,120 failed | per-row comparison below | 1.793579x geomean | 44.25% |
| Project-held-out matrix | 7/7 PASS; 0/13,117,440 failed | five faster, two slower | 1.190136x geomean | 15.98% |
| Source-derived matrix | 28/28 executable PASS + one authorized non-pass skip; 0/459,776,000 failed | 26 non-slower, two slower | 1.204977x geomean | 17.01% |

Across the four fresh benchmark artifacts, 245 accuracy trials compared
1,414,400,000 output elements with zero failures. The total includes intentional
overlap between the organizer-default, final, held-out, and source-derived
matrices. The largest observed absolute difference was `0.00131607`; it passed
through the relative-error branch of the OR comparator. Exact-reference routes
often had zero difference.

The organizer-default throughput increased from 567,446 to 761,089 tokens/s
(+34.13%). Its mean latency fell 25.65% and p90 latency fell 26.12%.

## Published final rows

| Row | Original median | Optimized median | Speedup | Latency change | Optimized attention route |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1.4152 ms | 0.8129 ms | 1.741x | -42.56% | Triton |
| 2 | 1.5439 ms | 0.8772 ms | 1.760x | -43.18% | Triton |
| 3 | 1.3663 ms | 0.8064 ms | 1.694x | -40.98% | Triton |
| 4 | 1.3538 ms | 0.7799 ms | 1.736x | -42.39% | Triton |
| 5 | 2.7009 ms | 1.4835 ms | 1.821x | -45.07% | Triton |
| 6 | 394.9666 ms | 366.9252 ms | 1.076x | -7.10% | reference fallback |
| 7 | 1.4500 ms | 1.3023 ms | 1.113x | -10.19% | reference fallback |
| 8 | 14.2202 ms | 13.9654 ms | 1.018x | -1.79% | reference fallback |
| 9 | 1.2025 ms | 0.8994 ms | 1.337x | -25.21% | Triton |
| 10 | 1.3308 ms | 0.8587 ms | 1.550x | -35.47% | Triton |
| 11 | 5.8184 ms | 1.0613 ms | 5.482x | -81.76% | Triton |
| 12 | 1.4018 ms | 0.7666 ms | 1.829x | -45.31% | Triton |
| 13 | 87.1028 ms | 18.2129 ms | 4.782x | -79.09% | Triton |
| 14 | not executed | not executed | not counted | authorized resource skip | none |

All 13 executable rows improved. The largest gain is final row 11, the exact
Campaign 4 padded-width `head_dim=8` target. Final row 8 is effectively near
parity because both implementations retain correctness-first reference
attention and most work is vendor GEMM/normalization rather than the custom
kernel. Aggregate backend accounting recorded 1,120 Triton calls, 336 reference
calls, and no SDPA calls.

## Held-out performance and memory

| Case | Original median | Optimized median | Speedup | Latency change | Incremental peak-memory change |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny-overhead | 0.4724 ms | 0.3073 ms | 1.537x | -34.95% | unchanged |
| medium-throughput | 0.4180 ms | 0.3197 ms | 1.307x | -23.51% | -26.67% |
| medium-padding | 0.6480 ms | 0.4832 ms | 1.341x | -25.43% | -31.25% |
| long-causal | 0.6822 ms | 0.8566 ms | 0.796x | **+25.57% slower** | -50.27% |
| long-causal-padding | 0.8266 ms | 0.9440 ms | 0.876x | **+14.21% slower** | -54.41% |
| long-attention | 0.8079 ms | 0.5129 ms | 1.575x | -36.51% | -71.79% |
| wide-model | 0.2350 ms | 0.2057 ms | 1.142x | -12.47% | unchanged |

The held-out aggregate is faster, but the two long-causal regressions are real
and reproduced. They remain numerically correct and trade latency for a large
reduction in incremental peak allocation. They must not be described as wins.

## Source-derived breadth

- Float32: 15 cases, all non-slower, 1.392322x geomean.
- Float16: 12 cases, 1.018418x geomean; two small regressions at 0.963x and
  0.989x. Automatic low-precision routing uses exact reference math, so these
  near-parity differences are timing variation rather than custom-kernel gains.
- Bfloat16: one exact-reference case, 1.038x.
- Best broad-matrix result: the 1024-token float32 case at 2.919x
  (24.3439 ms to 8.3393 ms).

## Logged execution evidence

| Attempt | Status | Child wall time |
| --- | --- | ---: |
| `E2-PREFLIGHT-001-FULL-TESTS` | PASS; 115 tests, 14 upstream warnings | 7.732788 s |
| `E2-COMPARE-001-ORGANIZER-DEFAULT` | PASS | 2.962283 s |
| `E2-COMPARE-002-FINAL-MATRIX` | PASS | 50.999552 s |
| `E2-COMPARE-003-HELDOUT` | PASS | 2.991622 s |
| `E2-COMPARE-004-SOURCE-DERIVED` | PASS | 63.628299 s |
| `E2-CLOSE-001-ARTIFACT-VALIDATION` | PASS | 0.043384 s |

Benchmark and preflight child-command time: **128.314544 seconds**. Including
closure validation, all six records total **128.357928 seconds**. Every command
has an immutable record under `attempts/E2-*.json`, including timestamps, command,
stdout/stderr, return code, wall time, environment, Git state, implementation
fingerprint, artifact hash, and parsed metrics. Fresh result artifacts are:

- [organizer default](../results/rtx-5070-ti-2026-08-28-current-vs-original-organizer-default.json)
- [published final matrix](../results/rtx-5070-ti-2026-08-28-current-vs-original-final.json)
- [held-out matrix](../results/rtx-5070-ti-2026-08-28-current-vs-original-heldout.json)
- [source-derived matrix](../results/rtx-5070-ti-2026-08-28-current-vs-original-source-derived.json)

The evaluation started from a clean pushed commit. Later artifacts truthfully
record a dirty worktree because earlier E2 evidence files were already
untracked; the implementation fingerprint and commit remained unchanged. No
code, test, organizer source, tolerance, or benchmark policy changed.

## Conclusion

The current implementation is materially better than the original for the
published float32 workload: it is correct under the strict comparator and
delivers a fresh 1.794x final-matrix geomean, with the largest gains on the
custom-attention targets. It also improves the held-out and broad aggregates.
It is not universally faster: long-causal held-out latency and two low-precision
source-derived cases regress. The correct claim is therefore “substantially
faster on the published final matrix with zero observed comparator failures,”
not “faster for every possible Transformer shape.”

These conclusions are specific to the recorded RTX 5070 Ti environment and the
published/derived forward-inference workloads. They do not establish backward,
training, other-GPU, or unstated organizer-policy performance.
