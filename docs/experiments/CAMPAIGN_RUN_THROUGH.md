# Full Track 3 Optimization Campaign Run-Through

Status: complete, evidence-reconciled campaign narrative

Last reconciled: 2026-08-28 (Asia/Singapore)

Evidence checkout before this document: `c4ff1f520901f268a6b76509b3a8c57dfdfea036`

Benchmarked implementation commit: `b833f7292bf15680d0add6007a53f9f7bf747690`

Selected implementation SHA-256:
`de768f1ff9ddee54a9ad83a67f3e1f205044c0ad5c723fc3bb4881093c97f611`

## Purpose and scope

This document tells the complete optimization story: what the team started
with, what each campaign targeted, every meaningful candidate that was tried,
which candidates failed or were superseded, what was integrated, how the score
changed, and what the final implementation can and cannot claim.

The repository did not formally name the pre-logger work “Campaign 1.” This
run-through calls it the **foundational phase / EXP-001** and then follows the
formal Campaign 2, Campaign 3, and Campaign 4 ledgers. It also covers the two
post-optimization rounds:

1. selected-submission validation (`S1-*`), which froze and revalidated the
   requested fingerprint without changing implementation bytes; and
2. the fresh current-versus-original evaluation (`E2-*`), which directly
   compared the final optimized implementation with the byte-preserved original
   PyTorch Transformer.

The immutable JSON and detailed campaign ledgers remain authoritative for an
individual run. This document is the comprehensive narrative and outcome layer;
it does not rewrite raw evidence.

## Executive outcome

The best implementation is the selected fingerprint above. On the freshest
complete published final matrix it achieved:

- **13/13 executable rows PASS**, plus the one exact authorized resource skip;
- **0 failed elements across 938,885,120 comparisons**;
- **1.793579x geometric-mean end-to-end speedup** versus the original;
- **5.482x** on the strongest row, final row 11; and
- 1,120 Triton attention calls, 336 explicit-reference calls, and no SDPA calls.

Across the complete fresh `E2` comparison—organizer default, published final,
held-out, and source-derived matrices—245 accuracy trials compared
1,414,400,000 output elements with zero failures. Those matrices intentionally
overlap, so this total is an evidence-volume count, not a count of unique tensor
elements or shapes.

The result is materially faster on the published workload, but it is not
universally faster. The fresh seven-case held-out aggregate was 1.190136x, while
the non-padded long-causal case was 0.796x and the padded long-causal case was
0.876x. Those regressions are real, correctness-green, and accompanied by
50.27% and 54.41% lower incremental peak allocation respectively.

## Campaign progression at a glance

The first two checkpoints use a seven-case provisional matrix. All later
checkpoints use the 14-row published matrix, of which 13 rows are executable.
Values measured on different matrices, baselines, or runs are not treated as a
single causal series.

| Phase | Main target | Accepted outcome | Measured aggregate after the phase | Disposition |
| --- | --- | --- | ---: | --- |
| Fused-attention foundation | Materialized attention scores and masks | One repository-owned Triton online-softmax attention launch per layer | 1.359647x on 7 provisional cases | keep |
| Packed-QKV foundation | Three separate Q/K/V projection launches | One cached packed vendor GEMM in the measured eager-fp32 envelope | 1.497835x on 7 provisional cases | keep |
| Organizer reconciliation | Final-matrix precision and unsupported-shape failures | IEEE-fp32 causal dots plus correctness-first fallbacks | 1.439957x on 13 executable final rows | keep |
| Foundational EXP-001 | Short causal `head_dim=64` spills | 32x64 Triton tiles for `head_dim=64, S<=128` | 1.426692x integration run; paired candidate gains 8.98% and 10.19% | keep |
| Campaign 2 / EXP-003 | Short `head_dim=32` attention | 64x64 Triton tiles for `head_dim=32, S<=128` | 1.525823x, +6.948% versus post-EXP-001 | keep |
| Campaign 3 / EXP-005-I2 | Short `head_dim=128` row 9 | 32x32 Triton tiles for `head_dim=128, S<=128` | 1.555780x, +1.963% | keep via profiler-backed gate |
| Campaign 4 / EXP-009-I2 | Exact row-11 `head_dim=8` reference bottleneck | Real width 8 with zero-padded internal dot width 16 and 64x64 tiles | 1.780075x primary; 1.784920x confirmation | keep |
| Submission selection | Freeze and validate the requested fingerprint | Existing root entry already matched; no implementation rewrite | 1.775778x / 1.770185x final reproductions | approve locally with residuals |
| Fresh original comparison | Direct final-versus-original measurement | Same selected fingerprint, freshly benchmarked | 1.793579x final; 1.341x organizer default | validated |

The small apparent drop from the green 1.439957x run to the post-EXP-001
1.426692x integration run is why the program did not accept candidates from a
single aggregate snapshot. EXP-001 was accepted on two alternating paired
comparisons, both of which showed the affected row and complete candidate
matrix improving.

## Fixed contract used by every accepted campaign

The accepted work preserved the byte-frozen organizer PyTorch harness, state
dict, comparator, shape order, timing policy, and the exact resource-skip rule.
Correctness was checked element by element using the stricter executable rule:

```text
absolute_error <= 0.001
OR
absolute_error <= 0.01 * abs(reference)
```

A case passed only when no output element failed that OR comparator. NaN and
infinity mismatches failed. Correctness ran before timing, CUDA events were used
after warm-up and synchronization, and baseline/optimized timing order
alternated. Backend counters and profiler events had to prove which attention
path actually ran.

The primary tuning environment was NVIDIA GeForce RTX 5070 Ti on native Windows
11 with driver 616.56, Python 3.12.10, PyTorch 2.13.0+cu130, CUDA 13.0, and
Triton 3.7.1. All performance conclusions are bounded to that environment.

## Evidence and runtime accounting

The foundational phase and EXP-001 predate the immutable attempt logger. They
have curated results, profiles, Git revisions, and review decisions, but no
complete per-command count or child-runtime total. Campaign 2 and every later
round used `benchmarks/run_optimization_attempt.py`.

| Record set | Immutable attempts | Passing child commands | Retained failed child commands | Logged child wall time |
| --- | ---: | ---: | ---: | ---: |
| Campaign 2 | 39 | 35 | 4 | 255.272605 s |
| Campaign 3 | 31 | 31 | 0 | 218.214354 s |
| Campaign 4 | 44 | 39 | 5 | 315.261251 s |
| Submission selection | 25 | 24 | 1 | 223.237808 s |
| Current-versus-original evaluation | 6 | 6 | 0 | 128.357928 s |
| **Logged total** | **145** | **135** | **10** | **1,140.343945 s** |

The logged total is 19 minutes and 0.344 seconds of child-command time. It
excludes orchestration, analysis, review, documentation, commits, and the
pre-ledger foundation. The current tree contains 88 result JSON files and 145
attempt JSON files. A passing command is not automatically an accepted
optimization: many correct timing screens were rejected because a different
candidate was faster or because the gain did not reproduce.

## Foundational phase: from prototype to a defensible baseline

### Initial prototype and audit

The initial implementation established the pre-LayerNorm Transformer contract,
strict reference-state loading, optional PyTorch SDPA, low-precision modes,
optional `torch.compile`, and a standalone Triton LayerNorm. The first audit
found that importing or compiling Triton was not enough to prove that a custom
kernel executed. The measurement system was therefore hardened to fail closed,
record environment and source fingerprints, retain non-pass outcomes, and count
the actual backend.

The standalone Triton LayerNorm was correct but slower than native CUDA:

| Shape | Native CUDA | Custom Triton | Native/custom | Outcome |
| --- | ---: | ---: | ---: | --- |
| 1024 x 512 | 0.00832 ms | 0.01629 ms | 0.511x | reject |
| 2048 x 512 | 0.01082 ms | 0.01562 ms | 0.693x | reject |
| 256 x 1024 | 0.00758 ms | 0.01664 ms | 0.456x | reject |

Residual-add/LayerNorm fusion, custom output or FFN GEMMs, and broad compilation
were not promoted. The measured launch savings did not justify the numerical,
support, and maintenance risk, while vendor GEMMs and native normalization were
already strong.

### Repository-owned fused attention

The first durable optimization replaced materialized attention scores and dense
causal masks with a Triton kernel that streams K/V tiles, maintains fp32 online
softmax state, applies causal and prefix masks inside each tile, and computes
P@V without allocating `[B,H,S,S]` intermediates.

The provisional seven-case matrix passed 35 accuracy trials with zero failures
and measured 1.359647x geomean, with per-case speedups from 1.138x to 1.566x.
Long-attention incremental allocation fell from 78 MiB to 22 MiB, a 71.8%
reduction. Causal and causal-plus-padding cases reduced incremental allocation
by 52.4% and 54.4%. Profiler evidence showed one custom attention launch per
layer.

Two related ideas were rejected: causal loop-frontier pruning did not improve
the complete end-to-end matrix, and alternative early tile/stage policies did
not produce a repeatable aggregate win. Direct low-precision attention remained
tested, but automatic deep-stack fp16/bfloat16 execution retained exact
reference math after small fused differences accumulated beyond the comparator.

### Packed QKV and bounded vendor routing

The next foundation cached a derived packed QKV weight and replaced three
projection GEMMs with one for eager CUDA float32 inference through
`d_model=512`. Across five two-layer forwards, profiler `addmm` calls fell from
60 to 40. A narrow, short, unmasked float32 corner was also allowed to select
PyTorch SDPA where it was measured to be preferable.

The provisional seven-case geomean rose from 1.359647x to 1.497835x. Packing
remained disabled for training, CPU, low precision, `torch.compile`, and wider
models. Output projection and FFN work stayed in vendor GEMMs.

### Organizer reconciliation and precision repair

The published final matrix exposed gaps that the provisional matrix did not.
The first final artifact correctly failed: only 1 of 13 executable rows passed,
11,869 of 938,885,120 elements failed, and its tempting 3.095x geomean was
discarded as invalid performance evidence.

The repair disabled TF32 inside causal Triton dot products, routed unsupported
head dimensions and causal batches above 128 to exact reference math, and added
boundary checks. The resulting green final baseline passed 13/13 executable
rows with zero failed elements at 1.439957x.

### EXP-001: short `head_dim=64`

Final row 10 spent 30,324.486 us across 40 attention launches, representing
79.6% of recorded GPU time. The existing 64x128 launch reported 2,468 spills
and 81,920 bytes of shared memory. EXP-001 tested a bounded 32x64 launch with
four warps and two stages for `head_dim=64, S<=128`; compiled evidence fell to
two spills and 49,152 bytes.

| Full-matrix pair | Control geomean | Candidate geomean | Relative gain | Control row 10 | Candidate row 10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.303394x | 1.420410x | 8.98% | 3.5660 ms | 0.9703 ms |
| 2 | 1.309126x | 1.442585x | 10.19% | 3.7258 ms | 0.9717 ms |

The row-10 attention range fell 89.41%, from 30,324.486 us to 3,210.441 us.
Exact-reference routing was rejected because its projected aggregate gain was
only about 4.7% and row 10 remained below parity. SDPA was rejected after one
element failed in five exact trials. The 32x64 Triton policy was independently
approved and became the accepted EXP-001 checkpoint.

## Campaign 2: long and short `head_dim=32`, plus the first width-eight probe

Campaign 2 introduced the immutable attempt logger and started from the
accepted EXP-001 checkpoint. It ranked three profile-backed hypotheses:

1. long `head_dim=32` launch geometry for final row 13;
2. short `head_dim=32` launch geometry for final row 1 and related rows; and
3. guarded direct `head_dim=8` support.

### EXP-002: long `head_dim=32`

Row 13's fresh optimized control was 17.6598 ms and attention represented
82.00% of the optimized profile range. All three candidates were correct and
executed Triton, but every one was slower than the fresh control:

| Candidate | Row-13 optimized median | Same-run speedup | Outcome |
| --- | ---: | ---: | --- |
| `BLOCK_N=128`, two stages | 41.3406 ms | 2.047x | reject |
| two stages only | 17.9966 ms | 4.701x | reject; 1.91% latency regression |
| `BLOCK_M=32` | 21.5897 ms | 3.918x | reject |

The subsystem stopped after its three allowed producer cycles. No long-row-13
launch change was integrated.

### EXP-003: short `head_dim=32`

Row 1's unchanged optimized median was 1.2358 ms, and attention represented
33.55% of its optimized profile range. Three launch policies were screened:

| Policy | Row-1 optimized median | Correctness | Outcome |
| --- | ---: | --- | --- |
| unchanged 64x128 | 1.2358 ms | 0/5,242,880 failed | control |
| 32x64 | 0.8579 ms | zero failed | superseded |
| 32x128 | 0.8805 ms | zero failed | reject |
| 64x64 | 0.8164 ms | zero failed | keep |

Two alternating confirmations selected 64x64. Its three candidate medians were
0.8164/0.8203/0.8235 ms, with a 0.8201 ms mean and 0.0036 ms sample standard
deviation. The corresponding control mean was 1.2402 ms with 0.0038 ms sample
standard deviation.

After integration, the published-final geomean rose from 1.426692x to
1.525823x (+6.948%). Final, organizer-default, held-out, and source-derived
correctness all remained zero-failure. Row-1 attention time fell 69.980%.

### EXP-004: direct `head_dim=8`

The existing kernel was extended only far enough to test whether an unpadded
width-eight dot was structurally legal. Eleven other direct cases passed, but
Triton rejected the width-eight dot because its lowering requires `K >= 16`.
The campaign stopped before model timing and kept exact reference fallback.
This rejection became the key design constraint for Campaign 4: pad only the
kernel's internal dot width while preserving the real public head dimension.

### Campaign 2 outcome

Campaign 2 retained 39 attempts: 35 passing child commands, four failed gates,
and 255.272605 seconds of child runtime. EXP-003 was accepted; EXP-002 and
EXP-004 were rejected. The campaign closed with 103 repository tests passing
and 14 upstream PyTorch deprecation warnings.

## Campaign 3: short `head_dim=128`

Campaign 3 started from the accepted Campaign 2 checkpoint. Fresh final row 9
was slower than the original at 1.2341 ms optimized versus 1.1745 ms baseline
(0.952x). Attention consumed 6,775.468 us across 40 calls, or 55.23% of the
optimized profile range.

### EXP-005: Triton launch geometry

| Tile | Optimized median | Outcome |
| --- | ---: | --- |
| 16x64 | 0.9609 ms | reject; slower than 32x32 |
| 32x32 | 0.9042 ms | keep |
| 16x32 | 1.1087 ms | reject |

The accepted 32x32 candidate reproduced at 0.9042/0.9049/0.9034 ms, a 0.0015
ms range. Its three-run median was 26.73% below the fresh 1.2341 ms optimized
baseline.

### EXP-006: production SDPA alternative

A forced-backend screen made SDPA look promising at 0.69224 ms, versus 1.15082
ms for auto/Triton and 1.00851 ms for exact reference. The first production
route passed 46 tests but missed the organizer's all-valid mask and still ran
Triton; the run was retained as rework rather than described as an SDPA win.

The corrected route proved 112 SDPA calls and exact correctness, then measured
0.9613/0.9716/0.9636 ms. The 32x32 Triton candidate was 6.16% faster by the
three-run medians, so production SDPA was rejected for this shape.

### Integration and stop decision

The final geomean rose from 1.525823x to 1.555780x (+1.963%). Although that was
below the nominal 5% aggregate gate, it cleared the explicit profiler-backed
alternative: `_attention_fwd` fell 55.454%, from 6,775.468 us to 3,018.182 us,
and the ten-step optimized model range fell 26.962%. Final row 9 improved from
1.2055 ms to 0.9071 ms and from 0.945x to 1.281x.

EXP-007 projection/launch-count work was deliberately not run. After EXP-005,
the profile leaders were vendor GEMMs and native LayerNorm, so further custom
work lacked profile authorization. Campaign 3 ended with 31/31 child commands
passing, 218.214354 seconds logged, a 104-test closure suite, and one accepted
change: the exact `head_dim=128, S<=128` 32x32 launch rule.

## Campaign 4: exact final-row-11 `head_dim=8`

Campaign 4 targeted the only executable Campaign 3 final row that remained
slower than its baseline. Fresh row 11 used reference attention and measured
5.6149 ms; reference attention dominated the 41,658.659 us ten-step profile.

### Preflight and backend screen

The campaign retained several setup failures instead of hiding them: default
Python 3.14 had no Torch, the first workflow schema was incomplete, the logger
persisted a nonportable active-interpreter path, and the first profile command
used an incompatible manifest shape. Each infrastructure defect was repaired
and rerun before candidate acceptance.

The backend screen measured 4.1410 ms for auto/reference, 4.1516 ms for forced
reference, and 1.2399 ms for forced SDPA. This authorized a bounded SDPA
candidate while the team designed a legal Triton width-eight path.

### EXP-008 and EXP-009 candidate sequence

| Candidate | Backend / launch | Exact row-11 optimized median | Outcome |
| --- | --- | ---: | --- |
| fresh production control | exact reference | 5.6149 ms | control |
| EXP-008-I1 | SDPA | 1.8658 ms | correct; superseded |
| EXP-009-I1 | padded Triton 64x128 | 1.2739 ms | correct; superseded |
| EXP-009-I2 | padded Triton 64x64 | 1.0595/1.0628/1.0624 ms | keep |
| EXP-009-I3 | padded Triton 32x64 | 1.3105 ms | reject; 18.93% slower than I2 median |

The accepted design separates real `HEAD_DIM=8` from compile-time
`DOT_HEAD_DIM=16`. Padded Q/K/V lanes load as zero, only the real eight output
lanes are stored, and the scale remains `8**-0.5`. Automatic model routing is
deliberately exact: only `B=64, S=128, d_model=128, heads=16, layers=4,
causal=true` selects this path. Final row 7 and every other unmeasured
model-level width-eight shape remain on reference attention.

I2's three optimized medians had a 0.0033 ms range. Its median was 81.08% below
fresh reference and 43.06% below the exact production SDPA alternative. The
integrated row-11 profile retained 40 Triton calls and reduced the ten-step
model range 74.57% versus fresh reference.

### Campaign 4 outcome

The published-final geomean increased from Campaign 3's 1.555780x to
1.780075x (+14.417%), and a second complete run measured 1.784920x with the
same correctness and backend counts. The primary final matrix was 13/13
executable PASS plus the authorized skip, with 0/938,885,120 failed elements.
The held-out and source-derived gates remained correctness-green at 1.434850x
and 1.215170x in that campaign's runs.

Campaign 4 retained 44 attempts: 39 passing commands, five failed gates, and
315.261251 seconds of child runtime. It closed with 112/112 repository tests
passing and independent candidate, AI Council, and release-gate review for the
exact local fingerprint.

## Selected-submission validation

The requested SHA-256 was not a file to copy into a registry. The live root
entry, `torch_transformer_benchmark.py::UserOptimizedTransformer`, already
recomputed to the requested schema-2 fingerprint across all 13 implementation
paths. Rewriting it would have changed the identity or created a false selector,
so the validation round froze the existing bytes and tested them directly.

| Gate | Correctness | Performance / proof |
| --- | --- | --- |
| Complete pytest | 112/112 PASS initially; final closure reached 115/115 | CPU and CUDA coverage; no required test removed |
| Direct root entry | 5/5 PASS; 0/2,621,440 failed | 1.195x; 1,950 Triton calls |
| Untouched organizer default | 5/5 PASS; 0/2,621,440 failed | 1.352x; 1,950 Triton calls |
| Final primary | 13/13 executable PASS + exact skip; 0/938,885,120 failed | 1.775778x; row 11 5.456x |
| Final confirmation | same correctness and backend counts | 1.770185x; row 11 5.408x |
| Held-out primary / confirmation | 7/7 PASS twice; zero failed elements | 1.210009x / 1.266010x |
| Source-derived | 28/28 executable PASS + exact skip; 0/459,776,000 failed | 1.203466x |
| Row-11 profile | profiler gate PASS | 40 Triton calls; 4,763.665 us attention range |

The two full final geomeans differed by only 0.315% and had identical
correctness and backend counts. The held-out non-padded long-causal case
reproduced below baseline at 0.793x and 0.800x. This became a required residual
disclosure, not a waived or hidden result.

Selection validation retained 25 attempts: 24 passing commands, one failed
workflow-schema preflight, and 223.237808 seconds. Its decision was approval for
repo-local selection with residuals, not a public-release or Devpost approval.

## Fresh current optimized versus original evaluation

The final evaluation compared the byte-preserved organizer PyTorch
`BaselineTransformer` with the selected root
`UserOptimizedTransformer`. It copied the original weights with `strict=True`,
used identical inputs, ran correctness before timing, and alternated timing
order. No implementation, source, tolerance, timing rule, or dispatch policy
changed during this round.

| Evaluation | Accuracy | Original versus optimized | Speedup | Equivalent latency reduction |
| --- | --- | --- | ---: | ---: |
| Untouched organizer default | 5/5 PASS; 0/2,621,440 failed | 1.8046 ms vs 1.3454 ms | 1.341x | 25.43% |
| Published final matrix | 13/13 executable PASS + exact skip; 0/938,885,120 failed | per-row table below | 1.793579x geomean | 44.25% |
| Project-held-out matrix | 7/7 PASS; 0/13,117,440 failed | five faster, two slower | 1.190136x geomean | 15.98% |
| Source-derived matrix | 28/28 executable PASS + exact skip; 0/459,776,000 failed | 26 non-slower, two slower | 1.204977x geomean | 17.01% |

The organizer-default throughput increased from 567,446 to 761,089 tokens/s,
an increase of 34.13%. Mean latency fell 25.65% and p90 latency fell 26.12%.

### Fresh published-final rows

| Row | Original median | Optimized median | Speedup | Optimized route |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1.4152 ms | 0.8129 ms | 1.741x | Triton |
| 2 | 1.5439 ms | 0.8772 ms | 1.760x | Triton |
| 3 | 1.3663 ms | 0.8064 ms | 1.694x | Triton |
| 4 | 1.3538 ms | 0.7799 ms | 1.736x | Triton |
| 5 | 2.7009 ms | 1.4835 ms | 1.821x | Triton |
| 6 | 394.9666 ms | 366.9252 ms | 1.076x | reference fallback |
| 7 | 1.4500 ms | 1.3023 ms | 1.113x | reference fallback |
| 8 | 14.2202 ms | 13.9654 ms | 1.018x | reference fallback |
| 9 | 1.2025 ms | 0.8994 ms | 1.337x | Triton |
| 10 | 1.3308 ms | 0.8587 ms | 1.550x | Triton |
| 11 | 5.8184 ms | 1.0613 ms | 5.482x | Triton |
| 12 | 1.4018 ms | 0.7666 ms | 1.829x | Triton |
| 13 | 87.1028 ms | 18.2129 ms | 4.782x | Triton |
| 14 | not executed | not executed | not counted | authorized resource skip |

Every executable final row improved in this fresh run. Rows 6–8 intentionally
used reference attention because their shape or correctness envelope did not
justify a custom route. Row 8 was near parity because most of its cost remained
vendor GEMM and normalization work.

### Fresh held-out trade-offs

| Case | Original median | Optimized median | Speedup | Incremental peak-memory change |
| --- | ---: | ---: | ---: | ---: |
| tiny-overhead | 0.4724 ms | 0.3073 ms | 1.537x | unchanged |
| medium-throughput | 0.4180 ms | 0.3197 ms | 1.307x | -26.67% |
| medium-padding | 0.6480 ms | 0.4832 ms | 1.341x | -31.25% |
| long-causal | 0.6822 ms | 0.8566 ms | 0.796x | -50.27% |
| long-causal-padding | 0.8266 ms | 0.9440 ms | 0.876x | -54.41% |
| long-attention | 0.8079 ms | 0.5129 ms | 1.575x | -71.79% |
| wide-model | 0.2350 ms | 0.2057 ms | 1.142x | unchanged |

The source-derived matrix was 1.392322x for 15 float32 cases, 1.018418x for 12
float16 cases, and 1.038x for one bfloat16 case. Two small low-precision cases
measured 0.963x and 0.989x. Automatic low-precision routing uses exact reference
math, so those near-parity movements are timing variation rather than claimed
custom-kernel wins.

The `E2` round retained six passing attempt records totaling 128.357928 seconds:
a full 115-test preflight, four complete comparison artifacts, and an artifact
closure check.

## Complete candidate disposition summary

| Candidate or direction | Final disposition | Reason |
| --- | --- | --- |
| Standalone Triton LayerNorm | reject | Correct but only 0.456x–0.693x as fast as native CUDA |
| Fused Triton online-softmax attention | keep | End-to-end speedup, one custom launch per layer, and large memory reduction |
| Packed QKV projection | keep | Removed two projection GEMMs in a bounded inference envelope |
| Broad custom output/FFN GEMMs | deliberately unrun | Profiles did not justify replacing vendor GEMMs |
| Broad reduced-precision automatic Triton | reject for model auto | Deep-stack numerical differences exceeded the strict comparator |
| Causal TF32 dot products | repair | Rare final-matrix misses; causal Triton now uses IEEE fp32 |
| Causal loop-frontier pruning | reject | No complete end-to-end matrix improvement |
| EXP-001 exact reference route | reject | About 4.7% projected aggregate gain; target row stayed below parity |
| EXP-001 SDPA route | reject | One failed element in five exact trials |
| EXP-001 short head64 32x64 | keep | Two paired aggregate wins and 89.41% lower target attention time |
| EXP-002 long head32 variants | reject all | Every candidate was slower than the fresh control |
| EXP-003 short head32 32x64 | superseded | Correct, but slower than 64x64 |
| EXP-003 short head32 32x128 | reject | Correct, but slower than 64x64 |
| EXP-003 short head32 64x64 | keep | Stable target gain, +6.948% final geomean, 69.98% lower attention time |
| EXP-004 direct unpadded width 8 | reject | Triton dot lowering requires `K >= 16` |
| EXP-005 short head128 16x64 | reject | Correct, but slower than 32x32 |
| EXP-005 short head128 32x32 | keep | Best repeated target timing and 55.454% lower attention range |
| EXP-005 short head128 16x32 | reject | Correct, but slower than 32x32 |
| EXP-006 production SDPA | rework, then reject | First route did not activate; corrected route was slower than 32x32 Triton |
| EXP-007 projection/launch work | deliberately unrun | Post-EXP-005 profile did not authorize it |
| EXP-008 exact row-11 SDPA | superseded | Correct and fast, but selected padded Triton was 43.06% faster |
| EXP-009 padded Triton 64x128 | superseded | Correct, but slower than 64x64 |
| EXP-009 padded Triton 64x64 | keep | Best stable row-11 timing and largest campaign-wide gain |
| EXP-009 padded Triton 32x64 | reject | 18.93% slower than selected I2 median |

## Retained failed and rework gates

Ten logged child commands exited nonzero. They are part of the evidence rather
than omitted noise:

| Round | Failed gate | Resolution or outcome |
| --- | --- | --- |
| Campaign 2 | post-merge full suite | Four stale fingerprint assertions; artifacts were rebaselined |
| Campaign 2 | post-rebaseline full suite | Stale held-out metric and Windows tee encoding; both repaired |
| Campaign 2 | logger portability gate | LF/CRLF-only assertion; made platform-portable |
| Campaign 2 | EXP-004 direct width-eight gate | `K >= 16` compile failure; candidate rejected |
| Campaign 4 | default Python import | Python 3.14 had no Torch; pinned Python 3.12 used |
| Campaign 4 | first workflow validation | Missing closure sections and graph label; workflow reworked |
| Campaign 4 | baseline portability gate | Active `.venv` interpreter path leaked into durable evidence; canonicalized |
| Campaign 4 | first row-11 profile setup | Final manifest schema differed from profiler manifest; dedicated case added |
| Campaign 4 | first I2 focused gate | One stale duplicate assertion after 13 passes; expectation corrected |
| Submission selection | first workflow validation | Missing canonical closure fields; workflow corrected and revalidated |

Campaign 2 also disclosed two logger self-test defects that happened before a
durable attempt could be written: a direct-entrypoint import-path error and a
dictionary-union precedence error. Campaign 4 disclosed two pre-logger startup
incidents: default Python without Torch and a stale Linux-style `.venv` without
a Windows interpreter. These incidents are documented but are not falsely
counted as immutable attempt JSON.

## What worked across the program

- **Profile-led, shape-specific policies.** Every accepted tile addressed a
  measured bottleneck and stopped at an exact dispatch guard.
- **Correctness-first fallback.** Unsupported or sensitive cases used explicit
  reference math rather than forcing Triton everywhere.
- **Paired and counterbalanced confirmation.** Sub-millisecond timing drift was
  separated from candidate effects before integration.
- **Backend and profiler proof.** A candidate could not claim a Triton or SDPA
  win unless counters and profiler events showed that path executed.
- **Packed vendor work rather than replacement GEMMs.** QKV packing removed
  launches while continuing to use optimized vendor multiplication.
- **Immutable failures.** Rejected and broken attempts prevented repeated dead
  ends and made infrastructure repairs auditable.
- **Bounded stopping rules.** Campaigns stopped when the next idea lacked a
  profile-supported ceiling, even if more speculative changes were possible.

## What did not work, and why

- “Custom” was not automatically faster: standalone Triton LayerNorm lost to
  native CUDA, and long-head32 tile changes lost to the existing launch.
- Broader routing was often worse than an exact shape guard. The successful
  width-eight kernel is deliberately limited to final row 11 at model level.
- A microbenchmark win was insufficient. Campaign 3's forced SDPA screen was
  faster, but the corrected production route lost to the selected Triton tile.
- Numerical shortcuts were not accepted when they threatened the zero-failure
  rule. Causal TF32 and broad low-precision auto routing were restricted.
- Replacing strong vendor GEMMs or native LayerNorm without a profile-backed
  ceiling would have increased risk without credible end-to-end payoff.

## Current best execution policy

| Surface | Retained policy |
| --- | --- |
| Attention algorithm | Tiled Triton QK, fp32 online softmax, masks, and P@V in one launch per layer |
| Short `head_dim=32, S<=128` | 64x64, four warps, two stages |
| Short `head_dim=64, S<=128` | 32x64, four warps, two stages |
| Short `head_dim=128, S<=128` | 32x32, four warps, two stages |
| Direct `head_dim=8, S<=128` | 64x64 with internal dot width 16 |
| Model-level `head_dim=8` | Triton only for exact final row 11; otherwise reference |
| Long supported shapes | Conservative measured policy; no Campaign 2 long-head32 variant retained |
| QKV projection | Cached packed vendor GEMM for eager CUDA float32 through `d_model=512` |
| Narrow short unmasked float32 | SDPA only where the measured dispatcher prefers it |
| Causal float32 Triton | IEEE-fp32 dot products; no TF32 |
| Low precision, unsupported layouts/widths, large causal batches, CPU, training | Explicit correctness-first fallback |
| LayerNorm, output projection, FFN | Native PyTorch/vendor kernels |

## Final conclusion

The optimization program succeeded on its declared target. It converted an
initial framework-heavy prototype into a measured, repository-owned Triton
attention implementation with packed QKV, four accepted shape-specific launch
policies, exact fallbacks, and a complete audit trail. The strongest defensible
claim is:

> On the recorded RTX 5070 Ti environment, the selected implementation passed
> every executable published final row under the stricter comparator with zero
> observed failed elements and achieved a fresh 1.793579x geometric-mean
> end-to-end speedup versus the original.

The best design is Campaign 4's selected fingerprint, not any one isolated
kernel candidate. Its performance comes from combining fused attention, packed
QKV, profile-selected launch geometry for head dimensions 32/64/128, padded
internal width for the exact head-dimension-8 target, and reference/SDPA
fallbacks where custom execution was unsupported or unjustified.

The result does not prove backward/training support, universal other-GPU
performance, or speedup on every Transformer shape. Row 14 remains an
authorized non-pass resource skip, two long-causal held-out cases regress in
latency, and organizer dtype, padding, timing, backward, and post-workshop
policy remain external unknowns. Any fingerprint or evaluator-environment
change requires a fresh full validation.

## Audit trail

- Contract and acceptance criteria: [requirements](../REQUIREMENTS.md)
- Bounded loop and stopping rules: [optimization loop plan](../OPTIMIZATION_LOOP_PLAN.md)
- Canonical cross-campaign evidence index: [optimization history](OPTIMIZATION_HISTORY.md)
- Foundational accepted experiment: [EXP-001](EXP-001-head64-short-tiles.md)
- Campaign 2 ledger: [CAMPAIGN-002](CAMPAIGN-002.md)
- Campaign 3 ledger: [CAMPAIGN-003](CAMPAIGN-003.md)
- Campaign 4 ledger: [CAMPAIGN-004](CAMPAIGN-004.md)
- Selection and full-suite validation: [submission validation](SUBMISSION_VALIDATION.md)
- Fresh original comparison: [current versus original evaluation](CURRENT_VS_ORIGINAL_EVALUATION.md)
- Immutable attempt records: [`attempts/`](attempts/)
- Result index: [result artifacts](../results/README.md)
- Fresh final result: [current-versus-original final JSON](../results/rtx-5070-ti-2026-08-28-current-vs-original-final.json)
- Fresh held-out result: [current-versus-original held-out JSON](../results/rtx-5070-ti-2026-08-28-current-vs-original-heldout.json)
- Fresh source-derived result: [current-versus-original source-derived JSON](../results/rtx-5070-ti-2026-08-28-current-vs-original-source-derived.json)

