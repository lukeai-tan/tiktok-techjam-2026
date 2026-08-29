# Full Track 3 Optimization Campaign Run-Through

Status: evidence-reconciled through Campaign 11 closure on 2026-08-30

Last reconciled: 2026-08-30 (Asia/Singapore)

Campaign 11 base checkpoint: `8c89d1d4170c58d16fb75d79f212e990565fba7d`

Campaign 6 base commit: `7f4fcba4ffb891cb876fa9ec27afa2395b99c77a`

Selected implementation SHA-256:
`9c326536ea27cfc619f01531152b2c82986d9dc3f4274691d3e8191bbb0804eb`

## Purpose and scope

This document tells the complete optimization story: what the team started
with, what each campaign targeted, every meaningful candidate that was tried,
which candidates failed or were superseded, what was integrated, how the score
changed, and what the final implementation can and cannot claim.

The repository did not formally name the pre-logger work “Campaign 1.” This
run-through calls it the **foundational phase / EXP-001** and then follows the
  formal Campaign 2 through Campaign 11 ledgers. It also
covers the two post-Campaign-4 evaluation rounds:

1. selected-submission validation (`S1-*`), which froze and revalidated the
   requested fingerprint without changing implementation bytes; and
2. the fresh current-versus-original evaluation (`E2-*`), which directly
   compared the final optimized implementation with the byte-preserved original
   PyTorch Transformer.

The immutable JSON and detailed campaign ledgers remain authoritative for an
individual run. This document is the canonical narrative, ranking, and outcome
layer. [OPTIMIZATION_HISTORY](OPTIMIZATION_HISTORY.md) is the denser metric and
evidence index; individual campaign files are immutable-detail ledgers. These
roles prevent the three layers from competing as duplicate summaries.

## Executive outcome

The selected implementation is the fingerprint above. On the freshest
complete published final matrix it achieved:

- **13/13 executable rows PASS**, plus the one exact authorized resource skip;
- **0 failed elements across 938,885,120 comparisons**;
- **1.977420x geometric-mean end-to-end speedup** versus the original, with a
  complete confirmation at **1.986499x**;
- **6.377x** on the strongest row, final row 11; and
- 1,260 Triton attention calls, 196 explicit-reference calls, and no final-matrix SDPA calls.

Across the current-fingerprint final pair, organizer default, held-out pair,
and source-derived matrix, 345 accuracy trials compared 2,366,402,560 output
elements with zero failures. Those matrices intentionally overlap, so this is
an evidence-volume count, not a count of unique tensor elements or shapes.

The result is materially faster on the published workload, but it is not
universally faster. Two five-seed seven-case held-out aggregates are 1.339847x
and 1.386495x. Campaign 5 removed the prior non-padded and padded long-causal
regressions with exact-shape SDPA; four current runs keep long-causal at
1.198x-1.204x. Row 8 remains on exact reference attention, while Campaign 6's
packed projection reduces its same-window model profile time 7.91%.

Campaign 5's 1.995117x run remains the highest historical aggregate. Campaign 11
is selected on isolated, contemporaneous row-9 evidence rather than cross-window
aggregate timing: fused residual/normalization reduces the repeated subsystem
mean 41.77%, and a 300-sample optimized median moves from 0.815968 ms across
unchanged controls to 0.717648 ms without increasing
incremental peak allocation.

### Flagship and strongest specialist campaigns

**Ship Campaign 11.** It is the only current cumulative implementation, carries
every accepted predecessor change, owns the selected fingerprint, passes the
latest 148-test repository suite, and has two complete zero-failure final runs.
Campaign 5's slightly higher 1.995117x historical observation is not a better
submission candidate because it came from an older timing window and excludes
later accepted, separately proven work.

| Rank or use | Campaign | Why it belongs in the best few | Important limit |
| --- | --- | --- | --- |
| **Overall flagship** | **Campaign 11 / EXP-025-I1** | Current cumulative fingerprint; 1.977420x / 1.986499x final pair; zero of 938,885,120 failed per run; row-9 optimized median -12.05%; 148/148 tests | Target-GPU and published-shape specific; held-out geomean is lower than Campaign 5's historical pair |
| **Best broad architecture/generalization campaign** | **Campaign 5** | Added the row-6/row-7 accuracy-safe hybrids and fixed both long-causal regressions; 1.911947x / 1.995117x final and 1.447477x / 1.449715x held-out | Historical snapshot, not the current implementation or evidence fingerprint |
| **Best high-volume latency specialist** | **Campaign 7 / EXP-018-I2R** | Row-6 fusion attacks the 10,000-batch case; current inherited long timing is 291.417252 -> 188.457397 ms (35.33% lower), with no peak-memory increase | Exact-row optimization; whole-matrix movement is much smaller than the target-row saving |
| **Best single-row lineage** | **Campaign 4 + Campaign 8 on row 11** | Campaign 4 made width-eight Triton legal; Campaign 8 added exact residual/normalization fusion. Current row 11 is 4.710116x over 300 samples and 6.377x in the final matrix | Most of the gain is shape-specific and cumulative, so it should not be attributed to Campaign 8 alone |

Campaign 10 is the strongest additional narrow win: its exact row-5 fusion
reduced the controlled candidate median 11.58%, and the inherited current long
gate is 1.880066x. Campaign 6 is retained for the exact-width-1024 packed-QKV
profile improvement; Campaign 9 is valuable primarily as a well-evidenced
no-winner campaign that prevented unsafe row-8/row-13 changes.

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
| Campaign 5 / EXP-010-I3 | Exact row-7 reference bottleneck | layer 0 reference, layers 1-3 padded-width Triton | row 7 1.524x; model profile -33.64% | keep |
| Campaign 5 / EXP-011-I2 | Exact row-6 reference bottleneck | layers 0-1 reference, layers 2-3 Triton | row 6 1.503x; model profile -19.74% | keep |
| Campaign 5 / EXP-012-I1 | Two held-out long-causal regressions | exact-shape SDPA for padded/unpadded cases | 1.447477x/1.449715x held-out geomeans | keep |
| Campaign 5 composite | Full integration and rebaseline | all three exact routes | 1.911947x / 1.995117x final | highest historical aggregate |
| Campaign 6 / EXP-015-I1R | Exact row-8 projection bottleneck | packed QKV only at exact width 1024, preserving widths 513-1023 | row-8 model profile -7.91%; Campaign 6 final 1.872916x / 1.863721x | selected on same-window causal evidence |
| Campaign 7 / EXP-018-I2R | Exact row-6 residual plus LayerNorm bottleneck | guarded fused residual/LayerNorm with released temporaries | row-6 model profile -9.54%; 1.547046x over 100 samples; Campaign 7 final 1.880620x / 1.927261x | keep; no peak-memory increase |
| Campaign 8 / EXP-019-I1R | Exact row-11 residual plus LayerNorm bottleneck | reuse the guarded fused forward under exact row-11 and eval-mode guards | row-11 candidate median -9.70%; final 1.876167x / 1.911052x | keep; no peak-memory increase |
| Campaign 9 | Width-1024 row-8 and exact row-13 fusion | no integration | row-8 variants slower; row 13 failed 1/41,943,040 | close no winner |
| Campaign 10 / EXP-023-I1 | Exact row-5 residual plus LayerNorm bottleneck | reuse the guarded fused forward under an exact row-5 guard | row-5 median -11.58%; 2.001995x over 300 samples; final 1.926716x / 1.939005x | keep; 144/144 suite |
| Campaign 11 / EXP-025-I1 | Exact row-9 residual plus LayerNorm bottleneck | reuse the guarded fused forward under an exact row-9 guard | controlled optimized median -12.05%; mean subsystem -41.77%; final 1.977420x / 1.986499x | keep; memory-neutral |

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
| Alternate-branch comparison | 19 | 15 | 4 | 352.965078 s |
| Campaign 5 | 74 | 65 | 9 | 736.085883 s |
| Campaign 6 | 121 | 118 | 3 | 874.998508 s |
| Campaign 7 | 75 | 70 | 5 | 1,022.152100 s |
| Campaign 8 | 68 | 61 | 7 | 532.609478 s |
| Campaign 9 | 35 | 32 | 3 | 168.347636 s |
| Campaign 10 | 56 | 53 | 3 | 540.442413 s |
| Campaign 11 through `C11-CLOSE-007` | 75 | 67 | 8 | 527.126837 s |
| **Logged total through `C11-CLOSE-007`** | **668** | **616** | **52** | **5,895.071878 s** |

The exact duration of each later terminal record is retained in its own attempt
rather than predicted in this prose checkpoint. Logged time excludes unwrapped
orchestration, analysis, review, documentation, commits, and the pre-ledger
foundation. At the checkpoint the tree contained 361 curated result JSON files
and 668 attempt JSON files. The cleanup accounting record and its later
validation records append after that exact checkpoint. A passing command is not automatically an accepted
optimization: many correct timing screens were rejected because a different
candidate was faster or because the gain did not reproduce.

## Cleanup and terminal retention

Optimization stopped after Campaign 11. Campaign 12 was scoped but never
profiled, benchmarked, or implemented, so its three unexecuted scaffold files
were removed and it is not counted as a campaign outcome. Cleanup also removed
twelve obsolete auxiliary worktrees, seven generated Python/pytest cache
directories, and three unreferenced smoke JSON files. Experiment branches and
all immutable attempt, result, review, trace, and campaign evidence remain.

One detached EXP-025 worktree is intentionally retained because four immutable
candidate attempts point to raw result artifacts stored there. Its selected
source is normalized-byte identical to the active implementation. Removing that
worktree before relocating those artifacts would make provenance worse, not
cleaner. No branch, commit, tag, or remote was changed during cleanup.

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

## Campaign 5: layer-aware hybrids and long-causal recovery

Campaign 5 deliberately reopened only three surfaces after a fresh baseline
and profile pass. The starting final matrix was 13/13 executable PASS at
1.776534x. Rows 6-8 were exact reference routes at 1.060x, 1.050x, and 1.007x;
the two long-causal held-out cases were 0.798x and 0.878x.

### Full-backend screens

| Screen | Accuracy outcome | Profile/performance interpretation | Disposition |
| --- | --- | --- | --- |
| row 7 full Triton | 1/1,310,720 failed | Kernel executes but deep-stack drift crosses one boundary | reject |
| row 7 full SDPA | 1/1,310,720 failed | Same strict-comparator limit | reject |
| row 6 full Triton | 21/819,200,000 failed | Large causal batch accumulates drift | reject |
| row 6 full SDPA | 21/819,200,000 failed | No accuracy advantage over full Triton | reject |
| row 8 SDPA | 1/41,943,040 failed | `aten::addmm` is about 71% of model time | reject and stop |
| long-causal SDPA | zero failed | 1.199x unpadded, 1.230x padded screen | continue |

### Candidate loops

Row 7 required three placements. First-three-layers Triton failed one element.
First-two-layers Triton passed and reproduced at 1.280x/1.348x, but the selected
first-layer-reference/final-three-Triton route was faster: 1.484x target,
1.492x/1.596x confirmations, 18/18 stress scenarios PASS, and a 33.64% reduction
in ten-forward model time. The profile proved 30 Triton and 10 reference calls.

Row 6's one-reference/three-Triton route still failed one of 819,200,000
elements. The selected two-reference/two-Triton route passed with max absolute
error 0.000898957, measured 1.549x with 1.488x/1.495x confirmations, and reduced
ten-forward model time 19.74%. The profile proved 20 Triton and 20 reference
calls.

The exact two-layer 512-token causal held-out shape moved to SDPA for both
padding modes. Candidate stress and profile gates passed, then two complete
five-seed matrices reproduced 7/7 PASS with zero failures at 1.447477x and
1.449715x. Non-padded/padded primary speedups are 1.247x/1.280x; confirmation
is 1.216x/1.423x.

### Composite outcome

| Gate | Result |
| --- | --- |
| Final primary / confirmation | 13/13 executable PASS + exact skip twice; zero failed; 1.911947x / 1.995117x |
| Organizer default | 5/5 PASS; zero failed; 1.397x; 1,950 Triton calls |
| Source-derived | 28/28 executable PASS + exact skip; zero failed; 1.204815x |
| Held-out primary / confirmation | 7/7 PASS twice over five seeds; zero failed; 1.447477x / 1.449715x |
| Profiles | exact row-6/row-7 hybrid counts, row-11 Triton, and long-causal SDPA all proven |
| Full repository suite | 121/121 PASS; 14 upstream deprecation warnings |

The composite primary is 7.62% above the fresh Campaign 5 baseline. All nine
nonzero commands remain logged: five backend screens, two unsafe layer
placements, one malformed pytest selector, and the expected stale-artifact
pre-documentation suite. The corrected focused and full-suite gates passed.

## Campaign 6: exact-width projection optimization and measured plateaus

Campaign 6 started from the Campaign 5 fingerprint and a fresh 1.838500x final
control. It bounded four remaining profile-authorized surfaces instead of
reopening previously failed full-backend routes.

### Launch loops

| Surface | Variants | Correctness | Performance outcome | Disposition |
| --- | ---: | --- | --- | --- |
| row 6 huge-batch head-32 launch | 5 | all relevant gates passed | raw timing or profiler evidence did not beat the retained geometry | plateau |
| row 7 padded-width head-8 launch | 5 | all relevant gates passed | long-sample/profile evidence plateaued or regressed | plateau |
| row 11 padded-width launch | 3 | all relevant gates passed | 6.42%-19.77% slower than the long control | reject all |

### Row-8 projection loop

The two-plus-one Q/K/V grouping measured 14.132976 ms optimized against a
13.975408 ms baseline (0.988851x) and was rejected. Three 300-sample packed-QKV
runs measured 1.022022x, 1.030071x, and 1.023827x. Two contemporaneous unchanged
controls measured 0.981690x and 0.993542x, establishing that the candidate—not
favorable baseline drift—removed projection work.

Independent review rejected the first `d_model <= 1024` guard because it would
have enabled unmeasured widths 513-1023. The accepted `EXP-015-I1R` rework uses
`d_model <= 512 or d_model == 1024`. A positive width-1024 test and negative
width-768 test make that boundary executable. The integrated row-8 profile
recorded 160 rather than 240 `aten::addmm` calls over ten forwards, 11.33% less
`addmm` device time, and 7.91% less model device time.

### Composite outcome

| Gate | Result |
| --- | --- |
| Final primary / confirmation | 13/13 executable PASS + exact skip twice; zero failed; 1.872916x / 1.863721x |
| Organizer default | 5/5 PASS; zero failed; 1.338x; 1,950 Triton calls |
| Source-derived | 28/28 executable PASS + exact skip; zero failed; 1.244108x |
| Held-out primary / confirmation | 7/7 PASS twice over five seeds; zero failed; 1.365499x / 1.380821x |
| Profiles | row-6/row-7 hybrids, exact-width row-8 QKV reduction, row-11 Triton, and long-causal SDPA all proven |
| Full repository suite | 125/125 PASS; 14 upstream PyTorch deprecation warnings |

The current final pair differs by 0.491% and the primary is 1.872% above the
fresh Campaign 6 control. Campaign 5's higher full-matrix values remain valid
historical measurements; the accepted decision is based on the same-window,
targeted projection comparison and profiler evidence.

## Campaign 7: wide-head rejection and exact row-6 fusion

Campaign 7 opened from the Campaign 6 checkpoint with a fresh 1.863463x final
control. Its row-6 profile showed 203,340.055 us of residual adds and 282,683.278
us of native LayerNorm inside 2,026,089.666 us of ten-forward model device time.
That 24.0% combined ceiling authorized a fusion experiment. Row 8's exact
`head_dim=256` reference attention was tested first because it remained the
near-parity final row.

### EXP-017: direct `head_dim=256` attention

The 16x16 and 16x32 primitives compiled and passed direct arithmetic, but both
four-layer row-8 variants missed two of 41,943,040 strict comparisons. The
16x64 variant requested 151,616 shared-memory bytes against the device's
101,376-byte limit and could not compile. A hybrid keeping the first layer exact
passed all accuracy checks and reduced copies from 200 to 50 and BMMs from 80 to
20 over the profile window, but its 30 wide Triton kernels cost 29,166.344 us;
model device time regressed from 130,180.675 us to 141,240.672 us (+8.50%). The
entire route was rejected and its code stayed isolated.

### EXP-018: fused residual plus LayerNorm

The first exact-row-6 fusion passed arithmetic and model accuracy but kept
intermediates alive, adding 1,967,128,576 peak bytes. I2 explicitly released
those lifetimes; I2R further narrowed the runtime shape and mask boundary. The
final candidate passed optional-bias and state-dict checks, noncontiguous-mask
and neighboring-shape fallbacks, and 18 seed/scale/padding stress scenarios
covering 2,949,120,000 outputs with 144 expected fused calls.

The candidate profile replaced 80 residual adds and 80 of 90 native norms with
80 fused launches over ten forwards. Candidate E/G 100-sample brackets averaged
1.554314x versus F/H unchanged controls at 1.419031x, a +9.53% normalized gain.
The integrated profile measured 1,832,789.301 us of model time (-9.54%), and the
final 100-sample run measured 293.910400 ms baseline, 189.981712 ms optimized,
and 1.547046x. Candidate and control both used an 11,802,787,840-byte optimized
incremental peak. Independent review approved local integration.

### Composite outcome

| Gate | Result |
| --- | --- |
| Final primary / confirmation | 13/13 executable PASS + exact skip twice; zero failed; 1.880620x / 1.927261x |
| Organizer default | 5/5 PASS; zero failed; 1.358x; 1,950 Triton attention calls |
| Source-derived | 28/28 executable PASS + exact skip; zero failed; 1.202688x |
| Held-out primary / confirmation | 7/7 PASS twice over five seeds; zero failed; 1.380355x / 1.377674x |
| Profiles | row-6 attention hybrid plus 80 fused residual/norm events; retained row-7/row-8/row-11 and long-causal routes proven |
| Functional suite before pointer migration | 122/122 PASS with only stale fingerprint-evidence tests excluded |
| Complete post-integration suite | 137/137 PASS; 14 upstream PyTorch deprecation warnings |

## Campaign 8: exact row-11 residual/normalization fusion

Campaign 8 opened from Campaign 7 fingerprint
`a994eb1c0a5a7053335adbb1a4ab13dcde1f0ea247e5f9c422c017d8b297be8b`.
The row-11 baseline profile measured 41,211.814 us of model device time over 30
forwards, including 5,978.920 us in 240 residual adds plus 270 native norms.
The campaign froze every attention, packed-QKV, and row-6 launch policy and
tested only reuse of the proven fused residual/normalization forward.

### EXP-019: exact row-11 fusion

I1 generalized the row-6 helper behind an exact row-6-or-row-11 predicate. It
passed direct, stress, and affected suites; two 300-sample runs measured
0.894080/0.893904 ms versus 0.993920/0.990384 ms unchanged controls. The first
10-step profile was contradictory at the top-level model range even though it
showed the expected 80 fused events, so it was retained and repeated rather
than selected. A paired 30-step profile then showed a 13.30% model reduction
and 42.16% subsystem reduction.

The AI Council boundary review found that I1 relied on disabled gradients but
did not explicitly exclude `model.train()` under inference mode. I1R added the
eval-mode guard and row-11 CPU, dtype, layout, runtime-shape, mask, gradient,
training, and head-neighbor checks. Ten boundary tests, both 18-scenario row-6
and row-11 stress matrices, and all 34 affected tests passed. The stress pair
covered 2,967,994,368 outputs with zero failures and 288 fused calls.

Two retained-fingerprint 300-sample runs averaged 0.897184 ms and 4.697149x;
three unchanged controls averaged 0.993525 ms and 4.262788x. That is a 9.70%
optimized-median reduction and 10.19% normalized speedup gain, with identical
29,360,128-byte optimized incremental peak allocation. The active profile
replaced all 240 residual adds plus 240 native norms with 240 fused launches,
reduced subsystem time 46.28%, and reduced model device time 21.96%.

EXP-020 was deliberately not run: its workflow condition allowed row-7 fusion
only if EXP-019 closed without a winner. Reopening it after I1R cleared every
gate would have violated the campaign stop rule.

### Composite outcome

| Gate | Result |
| --- | --- |
| Final primary / confirmation | 13/13 executable PASS + exact skip twice; zero failed; 1.876167x / 1.911052x |
| Organizer default | 5/5 PASS; zero failed; 1.351x; 1,950 Triton attention calls |
| Source-derived | 28/28 executable PASS + exact skip; zero failed; 1.208961x |
| Held-out primary / confirmation | 7/7 PASS twice over five seeds; zero failed; 1.398943x / 1.401668x |
| Held-out stability | four complete matrices keep long-causal at 1.195x-1.199x; dedicated 300-sample run is 1.195x |
| Row-6 non-regression | 100 samples, 187.837311 ms, 1.547529x, unchanged memory and 2/2 attention split |
| Row-11 integration | 300 samples, 0.896928 ms, 4.651860x, unchanged memory; 240 fused events in the 30-step profile |
| Pre-doc fail-closed suite | 132 PASS / 7 expected stale-pointer failures; no implementation failure |
| Final repository suite | 139/139 PASS; 14 upstream PyTorch deprecation warnings |

## Campaign 9: wide fusion and long-row plateau

Campaign 9 froze Campaign 8 fingerprint `325a1e5c...79b` and passed a fresh
13/13 executable final matrix plus the exact skip at 1.893x. Thirty-step
profiles put residual-add plus native-LayerNorm at 10.44% of row-8 model time
and 9.67% of row-13 model time.

EXP-021-I1 extended the exact fusion to width-1024 row 8. It passed direct,
boundary, and 18 stress scenarios over 150,994,944 outputs, but its 13.209216 ms
optimized median was 0.49% slower than the 13.144816 ms control. The fused
subsystem was 1.54% slower than separate operations. I2's eight-warps remained
correct but increased that deficit to 5.58%. Both were removed.

EXP-022-I1 was the authorized row-13 fallback. It also passed direct, boundary,
and 18 stress scenarios, then the expanded five-trial gate found one failed
element out of 41,943,040 at seed 1238. Timing and memory were not run after
the hard accuracy failure. The candidate was removed, 139/139 restored tests
passed, and Campaign 8 remained selected with no Campaign 9 source integration.

## Campaign 10: exact row-5 fusion

Campaign 10 profiled rows 5, 7, and 12 from the restored Campaign 8 checkpoint.
Row 5 showed the largest safe remaining launch ceiling: residual adds plus
native LayerNorm consumed 12,089.646 us, 29.07% of the 30-forward model profile.
The width-32 row-7 fallback was reserved for use only if row 5 produced no
winner, and row 12 remained observation-only.

EXP-023-I1 reused the accepted residual/LayerNorm kernel behind exact static and
runtime row-5 guards. Direct and 18-scenario stress gates compared 150,994,944
outputs with zero failures, the affected suite passed 37 tests, and row-6/row-11
routing remained unchanged. Two counterbalanced controls averaged 1.325096 ms;
the retained 300-sample candidate measured 1.171584 ms (-11.58%) with identical
58,720,256-byte incremental peak allocation. The integrated 300-sample gate
measured 1.162976 ms and 2.001995x.

The integrated profile contains 240 fused launches, 30 remaining native norms,
no residual-add event, and 120 Triton attention calls over 30 forwards. Against
the fresh baseline, subsystem/model device time fell 40.63%/11.96%. Both final
matrices, organizer default, two held-out matrices, source-derived matrix,
inherited profiles/long runs, and the 144-test suite passed. EXP-024 was
deliberately unrun because the accepted row-5 result satisfied its fallback
condition.

## Campaign 11: exact row-9 fusion

Campaign 11 started from the selected Campaign 10 fingerprint and profiled
final rows 9, 10, and 1. Row 9 exposed the actionable head-count-specific
surface: residual-add plus native-LayerNorm consumed 5,769.588 us, 19.10% of the
first 30-forward model profile. Row 10 was authorized only as a fallback, and
row 1 remained an observation-only non-causal control.

EXP-025-I1 added only one exact static/runtime predicate for row 9. The isolated
candidate passed route and boundary checks, all 18 seed/scale/padding scenarios,
40 affected tests, memory and profiler gates, and the complete final matrix.
Two unchanged controls measured 0.815776 and 0.816160 ms optimized median. The
isolated candidate measured 0.717696 ms and the active transplant measured
0.717648 ms, a 12.05% reduction from the control mean and a 0.007% difference
from the reviewed candidate. Peak allocation stayed 29,360,128 bytes and the
active long run kept all 1,240 attention calls on Triton.

Two active profiles each record 240 fused launches, 30 remaining native norms,
and 120 Triton attention calls. Mean subsystem time falls from 5,765.324 us to
3,357.389 us (-41.77%). One active top-level profile was slow, so the active
model-event mean is 2.54% above the two baselines; this is disclosed profiler
variance, not evidence against the tightly counterbalanced CUDA-event timing.
Both complete final matrices pass at 1.977420x/1.986499x, organizer-default is
1.385x, source-derived is 1.206505x, and inherited row-5/row-6/row-8/row-11
mechanism gates remain exact. EXP-026 was deliberately unrun because the row-9
winner made its conditional row-10 fallback false.

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
| Campaign 5 full row-7 Triton/SDPA | reject | Each failed one strict-comparator element |
| EXP-010-I2 two-layer row-7 hybrid | superseded | Correct, but slower than the selected three-layer hybrid |
| EXP-010-I3 three-layer row-7 hybrid | keep | Stress-green and reduced profiled model time 33.64% |
| Campaign 5 full row-6 Triton/SDPA | reject | Each failed 21 strict-comparator elements |
| EXP-011-I1 three-layer row-6 hybrid | reject | Still failed one of 819,200,000 elements |
| EXP-011-I2 two-layer row-6 hybrid | keep | Zero failures and reduced profiled model time 19.74% |
| Campaign 5 row-8 SDPA | reject | One failed element and low attention-only profile ceiling |
| EXP-012-I1 exact long-causal SDPA | keep | Two five-seed matrices removed both prior regressions |
| Campaign 6 row-6 launch variants | reject/plateau | Five correct variants did not improve the accepted timing/profile |
| Campaign 6 row-7 launch variants | reject/plateau | Five correct variants plateaued or regressed |
| Campaign 6 row-11 launch variants | reject all | Three new axes were 6.42%-19.77% slower than control |
| EXP-015-I2 two-plus-one projection | reject | 0.988851x; slower than separate projections |
| EXP-015-I1 broad packed-QKV guard | rework | Review found unmeasured widths 513-1023 |
| EXP-015-I1R exact-width packed QKV | keep | Correct boundary; -11.33% `addmm` time and -7.91% model profile time |
| EXP-017 16x16 / 16x32 wide-head attention | reject | Each full row-8 route missed two strict-comparator elements |
| EXP-017 16x64 wide-head attention | reject | Shared-memory request exceeded the target-device limit |
| EXP-017 exact-first-layer hybrid | reject | Correct, but model profile time regressed 8.50% |
| EXP-018 I1 residual/norm fusion | rework | Correct, but temporary lifetimes added 1.967 GB peak allocation |
| EXP-018 I2R exact-row-6 residual/norm fusion | keep | Stress-green, -9.54% model profile time, +9.53% normalized long-run speedup, memory-neutral |
| EXP-019 I1 exact-row-11 fusion | rework | Correct and faster, but Council found the training-mode boundary was implicit rather than explicit |
| EXP-019 I1R exact-row-11 fusion | keep | 36 stress scenarios, -9.70% retained median latency, -21.96% integrated profile time, memory-neutral |
| EXP-020 exact-row-7 fusion | deliberately unrun | Fallback-only loop was disabled once EXP-019 produced an accepted winner |
| EXP-021 row-8 width-1024 fusion | reject both | Four warps regressed median and subsystem; eight warps was slower still |
| EXP-022 row-13 fusion | reject | Expanded five-trial gate failed one of 41,943,040 elements |
| EXP-023 exact-row-5 fusion | keep | Controlled optimized median -11.58%; profile-backed and memory-neutral |
| EXP-024 exact-row-7 fusion | deliberately unrun | Campaign 10 row-5 winner made its fallback condition false |
| EXP-025 exact-row-9 fusion | keep | Controlled optimized median -12.05%; mean subsystem -41.77%; memory-neutral |
| EXP-026 exact-row-10 fusion | deliberately unrun | Campaign 11 row-9 winner made its fallback condition false |

## Retained failed and rework gates

Thirty-three logged child commands exited nonzero through Campaign 8, excluding
the separately reported alternate-branch comparison. They are part of the evidence rather
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
| Campaign 5 | row-7 Triton and SDPA screens | One failed element in each; full-backend routes rejected |
| Campaign 5 | row-6 Triton and SDPA screens | 21 failed elements in each; full-backend routes rejected |
| Campaign 5 | row-8 SDPA screen | One failed element and low profile ceiling; row 8 unchanged |
| Campaign 5 | EXP-010-I1 | One failed element; different layer placement tested |
| Campaign 5 | EXP-010-I3 focused selector | Invalid hyphenated pytest name; corrected selector passed |
| Campaign 5 | EXP-011-I1 | One failed element; two-reference-layer route tested |
| Campaign 5 | pre-doc full suite | Five stale artifact assertions; artifacts updated and 121-test suite passed |
| Campaign 6 | first workflow validation | Invalid conditional PRD-impact labels; workflow corrected and revalidated |
| Campaign 6 | pre-doc full suite | 118 passed and five stale Campaign 5 evidence-pointer tests failed; pointers migrated |
| Campaign 6 | first curated-evidence closure | Reference-attention row-8 profile was incorrectly classified as custom-kernel proof; semantic assertion corrected |
| Campaign 7 | EXP-017 16x16 row-8 screen | Two of 41,943,040 elements failed; variant rejected |
| Campaign 7 | EXP-017 16x32 row-8 screen | Same two elements failed; variant rejected |
| Campaign 7 | EXP-017 16x64 direct compile | 151,616 requested shared-memory bytes exceeded the 101,376-byte limit; variant rejected |
| Campaign 7 | pre-integration candidate suite | 128 passed and five stale current-fingerprint artifact tests failed; functional suite passed 122/122 and pointers migrated after integration |
| Campaign 7 | first fail-closed council-validator rework | Inner quote loss caused a PowerShell parse failure; the next validator protected outer and inner quoting and passed |
| Campaign 8 | wrapper-argument typo | `--id` was rejected before benchmark execution; the failure was reconstructed under `C8-OPS-001` and rerun correctly |
| Campaign 8 | pre-doc full suite | 132 passed and seven stale Campaign 7 fingerprint-pointer tests failed; current artifacts and pointers were regenerated |
| Campaign 8 | pre-doc artifact contract | 14 passed and only the intentionally stale documentation-selection assertion failed; docs were reconciled |
| Campaign 8 | disclosure-literal reworks | Two successive 14-pass/1-fail gates exposed a missing and then line-split compliance invariant; the exact disclosure was repaired and 15/15 passed |
| Campaign 8 | post-doc full suite | 138 passed and one stale Colab test fingerprint failed; the test pin was synchronized with the already-correct notebook and 139/139 passed |
| Campaign 8 | graph-rebuild wrapper label | Unsupported decision label `retain` was rejected before payload execution; `C8-GRAPH-005A` retains the failure and the corrected `keep` run passed |
| Campaign 9 | candidate-analysis quoting | Two parser/command analysis attempts failed and were rerun with Windows-safe quoting |
| Campaign 9 | expanded row-13 accuracy | One of 41,943,040 elements failed; timing was not run and the candidate was removed |
| Campaign 10 | stale artifact and candidate-path serialization | Both fail-closed gates were retained; active artifacts were regenerated with repo-relative paths |
| Campaign 10 | first final-review validator | PowerShell stripped inner quotes; the corrected immutable validator passed |
| Campaign 11 | first candidate-review validator | The child path resolved as `\.venv`; the corrected `C11-REVIEW-001A` passed |
| Campaign 11 | raw-byte transplant parity | LF/CRLF bytes differed; exact fingerprint and normalized source hashes matched |
| Campaign 11 | first integrated analysis scripts | One used the wrong backend-count nesting and one lost double quotes; corrected attempts passed |
| Campaign 11 | pre-doc full suite | 139 passed and eight selected-artifact fingerprint pointers failed; current artifacts and pointers were regenerated |
| Campaign 11 | graph wrapper orchestration | PowerShell rejected the batched argv before either wrapper launched; `C11-OPS-002` mirrors the incident and native argument-array graph gates passed |
| Campaign 11 | program accounting counters | The compact counter had mismatched brackets and its first multiline rework lost double quotes on Windows; both failures remain, and single-quoted `C11-CLOSE-003B` passed |

`C7-CLOSE-003` is not in the nonzero count because its child returned zero, but
it is still a disclosed semantic false green: the outer PowerShell process
expanded away the inner variable, emitted non-terminating command errors, and
then printed the success marker. `C7-CLOSE-004` made errors terminating but
exposed a second quote-loss defect; `C7-CLOSE-005` preserved both variables and
comparison strings and is the authoritative review-field validation.

Campaign 11 additionally disclosed two outer PowerShell parser incidents before
the intended wrappers could launch. One is mirrored by `C11-OPS-002`; neither is
falsely counted as direct measured execution. Campaign 2 also disclosed two logger self-test defects that happened before a
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
- **Fusion only when it removes a neighboring operation.** Standalone LayerNorm
  lost, but exact-row residual/norm fusion cleared a fresh end-to-end ceiling.
- **Immutable failures.** Rejected and broken attempts prevented repeated dead
  ends and made infrastructure repairs auditable.
- **Bounded stopping rules.** Campaigns stopped when the next idea lacked a
  profile-supported ceiling, even if more speculative changes were possible.

## What did not work, and why

- “Custom” was not automatically faster: standalone Triton LayerNorm lost to
  native CUDA, wide-head attention regressed even after removing copies/BMMs,
  and long-head32 tile changes lost to the existing launch.
- Broader routing was often worse than an exact shape or layer guard. The
  successful width-eight kernel is all-layer only on row 11 and layer-limited
  on row 7.
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
| Exact final row 7 | layer 0 reference; layers 1-3 padded-width Triton |
| Exact final row 5 | all four attention layers use Triton; residual adds fuse with downstream LayerNorms under the exact eval-mode eager CUDA float32 guard |
| Exact final row 6 | layers 0-1 reference; layers 2-3 Triton; residual adds fused with downstream LayerNorms under the exact eager CUDA float32 guard |
| Exact final row 9 | all four attention layers use Triton; residual adds fuse with downstream LayerNorms under the exact eval-mode eager CUDA float32 guard |
| Exact final row 11 | all four attention layers use padded-width Triton; residual adds fuse with downstream LayerNorms under the exact eval-mode eager CUDA float32 guard |
| Other model-level `head_dim=8` | reference |
| Exact held-out `B=2,S=512,d=512,h=8,layers=2,causal` | SDPA with or without measured prefix padding |
| Long supported shapes | Conservative measured policy; no Campaign 2 long-head32 variant retained |
| QKV projection | Cached packed vendor GEMM for eager CUDA float32 through `d_model=512` and exact `d_model=1024`; widths 513-1023 stay separate |
| Narrow short unmasked float32 | SDPA only where the measured dispatcher prefers it |
| Causal float32 Triton | IEEE-fp32 dot products; no TF32 |
| Low precision, unsupported layouts/widths, other large causal batches, CPU, training | Explicit correctness-first fallback |
| Other LayerNorm, output projection, FFN | Native PyTorch/vendor kernels |

## Final conclusion

The optimization program succeeded on its declared target. It converted an
initial framework-heavy prototype into a measured, repository-owned Triton
attention implementation with bounded packed QKV, an exact-row residual/norm
kernel, four accepted attention launch policies, exact fallbacks, and a complete
audit trail. The strongest defensible
claim is:

> On the recorded RTX 5070 Ti environment, the selected implementation passed
> every executable published final row under the stricter comparator with zero
> observed failed elements and achieved a current 1.977420x geometric-mean
> end-to-end speedup versus the original.

The selected design is Campaign 11's fingerprint, not any one isolated
kernel candidate. Its performance comes from combining fused attention, fused
row-5, row-6, row-9, and row-11 residual/normalization, packed
QKV, profile-selected launch geometry for head dimensions 32/64/128, padded
internal width for head dimension 8, exact layer-aware hybrids, and SDPA/reference
fallbacks where custom execution was unsupported or unjustified.

The result does not prove backward/training support, universal other-GPU
performance, or speedup on every Transformer shape. Row 14 remains an
authorized non-pass resource skip, row 8 remains near parity despite its
same-window projection improvement, and organizer
dtype, padding, timing, backward, and post-workshop
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
- Campaign 5 ledger: [CAMPAIGN-005](CAMPAIGN-005.md)
- Campaign 6 ledger: [CAMPAIGN-006](CAMPAIGN-006.md)
- Campaign 7 ledger: [CAMPAIGN-007](CAMPAIGN-007.md)
- Campaign 8 ledger: [CAMPAIGN-008](CAMPAIGN-008.md)
- Campaign 9 ledger: [CAMPAIGN-009](CAMPAIGN-009.md)
- Campaign 10 ledger: [CAMPAIGN-010](CAMPAIGN-010.md)
- Campaign 11 ledger: [CAMPAIGN-011](CAMPAIGN-011.md)
- Consolidation Council decision: [final consolidation review](reviews/CAMPAIGN-CONSOLIDATION-FINAL-REVIEW.json)
- Selection and full-suite validation: [submission validation](SUBMISSION_VALIDATION.md)
- Fresh original comparison: [current versus original evaluation](CURRENT_VS_ORIGINAL_EVALUATION.md)
- Current final result: [Campaign 11 final JSON](../results/rtx-5070-ti-2026-08-29-c11-integrated-final.json)
- Current held-out result: [Campaign 11 five-seed held-out JSON](../results/rtx-5070-ti-2026-08-29-c11-integrated-heldout-5seed.json)
- Immutable attempt records: [`attempts/`](attempts/)
- Result index: [result artifacts](../results/README.md)
- Fresh final result: [current-versus-original final JSON](../results/rtx-5070-ti-2026-08-28-current-vs-original-final.json)
- Fresh held-out result: [current-versus-original held-out JSON](../results/rtx-5070-ti-2026-08-28-current-vs-original-heldout.json)
- Fresh source-derived result: [current-versus-original source-derived JSON](../results/rtx-5070-ti-2026-08-28-current-vs-original-source-derived.json)
