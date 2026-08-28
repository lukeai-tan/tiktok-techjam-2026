# Current Optimized vs Original Evaluation

Status: Campaign 5 comparison complete; every executable correctness gate passed

Date: 2026-08-28 (Asia/Singapore)

Base commit: `3be02a3ebe562a89ca360b196057a2762b425ec4`

Implementation fingerprint:
`9159177a21d039366ed4d3aef431b4b14d3bcef26d5eeaab0808efa739294029`

## Comparison definition

“Original” is the byte-preserved organizer PyTorch `BaselineTransformer`.
“Current optimized” is
`torch_transformer_benchmark.py::UserOptimizedTransformer`. The harness copies
the original weights with `strict=True`, supplies identical inputs, runs
correctness before timing, and alternates baseline/optimized timing order. No
organizer source, comparator, tolerance, timing policy, or resource-skip rule
changed for Campaign 5.

The strict executable comparator accepts an element when absolute error is at
most `0.001` **or** relative error is at most `1%`. The measurements are from an
RTX 5070 Ti, driver 616.56, Windows 11, Python 3.12.10, PyTorch 2.13.0+cu130,
CUDA 13.0, and Triton 3.7.1.

## Aggregate outcome

| Evaluation | Accuracy | Original vs optimized | Speedup | Equivalent median/geomean latency reduction |
| --- | --- | --- | ---: | ---: |
| Untouched organizer default | 5/5 trials PASS; 0/2,621,440 failed | 1.8948 ms vs 1.3565 ms | 1.397x | 28.41% |
| Published final primary | 13/13 executable PASS + one authorized non-pass skip; 0/938,885,120 failed | per-row comparison below | 1.911947x geomean | 47.70% |
| Published final confirmation | same correctness and backend counts | complete independent timing run | 1.995117x geomean | 49.88% |
| Project-held-out primary | 7/7 PASS; 0/13,117,440 failed | every case faster | 1.447477x geomean | 30.91% |
| Project-held-out confirmation | same zero-failure contract | every case faster | 1.449715x geomean | 31.02% |
| Source-derived matrix | 28/28 executable PASS + one authorized non-pass skip; 0/459,776,000 failed | broad mixed-dtype matrix | 1.204815x geomean | 17.00% |

The primary final geomean is 7.62% above the fresh Campaign 5 starting
implementation (1.776534x). Relative to the earlier selected-submission primary
at 1.775778x, it is 7.67% higher. Accuracy did not trade away: the primary and
confirmation final runs each retained zero failures across 938,885,120 checked
elements.

The organizer-default optimized throughput increased from 540,426 to 754,895
tokens/s (+39.69%). Median latency fell 28.41%, mean latency fell 26.20%, and
p90 latency fell 27.03%.

## Published final rows

| Row | Original median | Optimized median | Speedup | Latency reduction | Optimized attention route |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1.5312 ms | 0.8595 ms | 1.781x | 43.87% | Triton |
| 2 | 1.5842 ms | 0.8898 ms | 1.780x | 43.83% | Triton |
| 3 | 1.4291 ms | 0.8133 ms | 1.757x | 43.09% | Triton |
| 4 | 1.3422 ms | 0.7950 ms | 1.688x | 40.77% | Triton |
| 5 | 2.9995 ms | 1.6647 ms | 1.802x | 44.50% | Triton |
| 6 | 449.1052 ms | 298.7559 ms | 1.503x | 33.48% | layers 0-1 reference; layers 2-3 Triton |
| 7 | 1.3750 ms | 0.9024 ms | 1.524x | 34.37% | layer 0 reference; layers 1-3 Triton |
| 8 | 15.7496 ms | 15.2777 ms | 1.031x | 3.00% | reference |
| 9 | 1.2159 ms | 0.9240 ms | 1.316x | 24.01% | Triton |
| 10 | 1.4777 ms | 0.8725 ms | 1.694x | 40.96% | Triton |
| 11 | 6.4429 ms | 1.0832 ms | 5.948x | 83.19% | padded-width Triton |
| 12 | 1.4170 ms | 0.7879 ms | 1.799x | 44.40% | Triton |
| 13 | 95.9056 ms | 20.0648 ms | 4.780x | 79.08% | Triton |
| 14 | not executed | not executed | not counted | authorized resource skip | none |

Every executable row improved in the primary timing run. Backend accounting was
1,260 Triton calls, 196 reference calls, and no SDPA calls. Row 8 remains the
residual near-parity case: its forced-SDPA accuracy screen failed 1 of
41,943,040 compared elements, and its profile attributed about 71% of runtime
to vendor `aten::addmm`, leaving too little safe attention-only leverage.

## Held-out performance and memory

| Case | Original median | Optimized median | Speedup | Latency reduction | Route |
| --- | ---: | ---: | ---: | ---: | --- |
| tiny-overhead | 0.5204 ms | 0.3216 ms | 1.618x | 38.21% | SDPA |
| medium-throughput | 0.5264 ms | 0.2632 ms | 2.000x | 50.00% | SDPA |
| medium-padding | 0.6889 ms | 0.5042 ms | 1.366x | 26.79% | Triton |
| long-causal | 0.7217 ms | 0.5788 ms | 1.247x | 19.80% | exact-shape SDPA |
| long-causal-padding | 0.8954 ms | 0.6993 ms | 1.280x | 21.90% | exact-shape SDPA |
| long-attention | 0.8400 ms | 0.5191 ms | 1.618x | 38.20% | Triton |
| wide-model | 0.2445 ms | 0.2098 ms | 1.165x | 14.19% | Triton |

The two old flagship regressions are removed. The earlier implementation's
fresh Campaign 5 baseline measured 0.798x for `long-causal` and 0.878x for
`long-causal-padding`; the selected route measures 1.247x/1.280x in the primary
run and 1.216x/1.423x in confirmation. Both complete five-seed runs are retained.

## Accuracy and rejected alternatives

- Row 7 full Triton and full SDPA each failed 1/1,310,720 elements. A two-Triton-
  layer hybrid passed, but the selected three-Triton-layer hybrid was faster and
  passed 18 seed/scale/padding stress scenarios.
- Row 6 full Triton and full SDPA each failed 21/819,200,000 elements. A
  three-Triton-layer hybrid still failed one element; the selected two-layer
  hybrid passed all 819,200,000 comparisons.
- Row 8 SDPA failed one element and was not implemented.
- Exact held-out long-causal SDPA passed the full padded/unpadded multi-seed
  stress set and both complete held-out matrices.

The largest reported relative errors occur around reference values near zero;
the executable OR comparator and absolute-error branch determine correctness.
No failed element, OOM, crash, or unauthorized skip was recast as a pass.

## Evidence

- [Campaign 5 ledger](CAMPAIGN-005.md)
- [Primary final matrix](../results/rtx-5070-ti-2026-08-28-c5-integrated-final.json)
- [Final confirmation](../results/rtx-5070-ti-2026-08-28-c5-integrated-final-confirmation.json)
- [Primary held-out matrix](../results/rtx-5070-ti-2026-08-28-c5-integrated-heldout-5seed.json)
- [Held-out confirmation](../results/rtx-5070-ti-2026-08-28-c5-integrated-heldout-5seed-confirmation.json)
- [Untouched organizer default](../results/rtx-5070-ti-2026-08-28-c5-integrated-organizer-default.json)
- [Source-derived matrix](../results/rtx-5070-ti-2026-08-28-c5-integrated-source-derived.json)
- [Immutable attempts](attempts/)

These conclusions are specific to the recorded target environment and the
published/derived forward-inference workloads. They do not establish backward,
training, other-GPU, or unstated organizer-policy performance.
