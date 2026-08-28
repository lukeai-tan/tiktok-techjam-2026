# Complete Track 3 Optimization History

Status: canonical cross-campaign history, reconciled 2026-08-28

Current accepted implementation: Campaign 4 local candidate based on `b41fdaf`

Implementation SHA-256: `de768f1ff9ddee54a9ad83a67f3e1f205044c0ad5c723fc3bb4881093c97f611`

This is the single navigation and interpretation layer for the complete
optimization program. It consolidates what was tried, why each candidate was
kept or rejected, how the score evolved, and which implementation is best. It
does not replace or rewrite the immutable attempt records, measured result JSON,
campaign ledgers, or review sidecars.

## Verdict

The best accepted implementation is the Campaign 4 fingerprint above. Its
primary organizer-final run is **13/13 executable PASS** plus the one exact
authorized resource skip, with **0 failed elements across 938,885,120
comparisons** and **1.780074850625043x** geometric-mean end-to-end speedup. A
second complete run of the same implementation reproduced the contract and
backend counts at **1.7849198559254662x**. The strongest single final case is
row 11 at **5.395x**.

After selecting that exact fingerprint as the repo-local submission, a fresh
full-suite revalidation measured **1.775778x** and **1.770185x** across two
complete final matrices, again with identical zero-failure and backend-count
evidence. The fresh organizer default is **1.352x**, source-derived geomean is
**1.203466x**, and two correctness-green held-out runs are **1.210008x** and
**1.266010x**. The latter disclose a repeatable approximately 0.80x slowdown
for the non-padded long-causal held-out case. These selection runs validate the
winner; they do not retroactively replace historical campaign measurements.

The winning design is not one universal launch. It combines fused Triton online
softmax attention, packed QKV projection in its measured envelope, exact
shape-aware launch policies for short head dimensions, and correctness-first
fallbacks. The largest final campaign gain came from making `head_dim=8` legal
inside Triton's 16-lane dot-product minimum by zero-padding only the internal
dot width and enabling that model path only for exact final row 11.

## Evidence authority and inventory

Evidence is read in this order:

1. The untouched organizer downloads, final shape manifest, requirements, and
   checked-in comparator.
2. Machine-generated result JSON and its recorded implementation fingerprint,
   source hashes, environment, command, accuracy counts, timings, and backends.
3. Immutable Campaign 2-4 and selected-submission attempt JSON, including
   nonzero commands.
4. Campaign and experiment decision records plus independent review sidecars.
5. This curated history, the technical report, and README summaries.

The working tree now contains **84 curated result JSON files** and **139
immutable attempt JSON files**. Campaigns 2-4 still own 114 of those attempts:
105 passing child commands, 9 failed child commands, and 788.748209 seconds.
Selected-submission validation adds 25 attempts: 24 passing child commands, one
retained workflow-schema failure, and 223.237808 seconds. Across both ledgers,
that is 129 passing commands, 10 failed commands, and 1,011.986017 seconds.
These times exclude orchestration and unlogged pre-ledger work.

| Record set | Attempts | Child PASS | Child FAIL | Measured child wall time | Detailed ledger |
| --- | ---: | ---: | ---: | ---: | --- |
| Campaign 2 | 39 | 35 | 4 | 255.272605 s | [CAMPAIGN-002](CAMPAIGN-002.md) |
| Campaign 3 | 31 | 31 | 0 | 218.214354 s | [CAMPAIGN-003](CAMPAIGN-003.md) |
| Campaign 4 | 44 | 39 | 5 | 315.261251 s | [CAMPAIGN-004](CAMPAIGN-004.md) |
| Submission selection | 25 | 24 | 1 | 223.237808 s | [SUBMISSION_VALIDATION](SUBMISSION_VALIDATION.md) |
| **Logged total** | **139** | **129** | **10** | **1,011.986017 s** | [`attempts/`](attempts/) |

The total is summed from all raw attempt values before rounding. Adding the four
independently rounded record-set display values differs by one microsecond.

A passing child command is not the same as an accepted optimization. Campaign 3
has no execution failures, for example, but still contains eight rejected
candidate records and two rework records. Final dispositions live in the
campaign ledgers because the immutable attempt JSON preserves the state known at
execution time.

The foundational phase and EXP-001 predate the Campaign 2 logger. They retain
curated measurements, profiler artifacts, Git revisions, and review decisions,
but there is no complete per-command count or wall-time ledger for that period.
This history does not invent those missing values.

## Score evolution

The first two rows use the seven-case provisional matrix. Later rows use the
14-row organizer-final matrix, of which 13 rows are executable. Values across
different matrices are not treated as a direct causal series. Even within the
same matrix, fresh baseline movement is why candidates were accepted using
paired measurements, affected-case checks, profiles, and confirmations rather
than a single headline number.

| Checkpoint | Matrix | Correctness | Geomean | What changed |
| --- | --- | --- | ---: | --- |
| Fused Triton foundation | 7 provisional cases | 7/7 PASS; 0/13,117,440 failed | 1.359647x | Replaced materialized attention with fused Triton online softmax |
| Packed-QKV foundation | 7 provisional cases | 7/7 PASS | 1.497835x | Packed three QKV projections into one measured eager-fp32 GEMM and kept bounded dispatch |
| Green final-matrix baseline | 14 final rows | 13/13 PASS + one authorized skip; 0/938,885,120 failed | 1.439957x | Repaired causal TF32 and unsupported-shape precision failures |
| Post-EXP-001 integration | 14 final rows | same zero-failure contract | 1.426692x | Accepted short `head_dim=64` 32x64 tiles after paired gains of 8.98% and 10.19% |
| Post-EXP-003 / Campaign 2 | 14 final rows | same zero-failure contract | 1.525823x | Accepted short `head_dim=32` 64x64 tiles; +6.948% versus post-EXP-001 |
| Campaign 3 | 14 final rows | same zero-failure contract | 1.555780x | Accepted short `head_dim=128` 32x32 tiles; +1.963%, with a 55.454% profile reduction |
| Campaign 4 primary | 14 final rows | same zero-failure contract | 1.780075x | Accepted exact-row-11 padded-width `head_dim=8` Triton; +14.417% versus Campaign 3 |
| Campaign 4 confirmation | 14 final rows | same contract and backend counts | 1.784920x | Independent complete-run reproduction of the current fingerprint |

The earliest two snapshots are recoverable at their original Git revisions:

```powershell
git show d287d82e:docs/results/rtx-5070-ti-2026-08-27.json
git show ab4ac52f:docs/results/rtx-5070-ti-2026-08-27.json
```

The legacy working-tree filename was regenerated during later rebaselines, so
its current bytes must not be used as proof of the earlier 1.359647x or
1.497835x snapshots.

## Chronological run-through

### 1. Prototype and contract audit

The initial prototype used PyTorch scaled-dot-product attention, optional
standalone Triton LayerNorm, low-precision modes, and an optional
`torch.compile` path. It established strict state-dict compatibility and the
pre-LayerNorm residual structure, but its initial GPU claims were incomplete.
The subsequent audit froze the executable comparator, added fail-closed error
accounting, captured environment and fingerprints, and required proof of the
actual backend rather than assuming that importing Triton meant a custom kernel
ran.

The optional standalone LayerNorm was measured and retired:

| rows x width | Native CUDA | Custom Triton | Native/custom | Decision |
| --- | ---: | ---: | ---: | --- |
| 1024 x 512 | 0.00832 ms | 0.01629 ms | 0.511x | reject |
| 2048 x 512 | 0.01082 ms | 0.01562 ms | 0.693x | reject |
| 256 x 1024 | 0.00758 ms | 0.01664 ms | 0.456x | reject |

It was accurate but slower and did not eliminate an adjacent launch. A residual
add plus LayerNorm fusion, custom output/FFN GEMMs, and broader compilation work
were not promoted because profiling did not justify their numerical and
maintenance risk.

### 2. Repository-owned fused attention

The first durable performance win was the repository-owned Triton attention
kernel. It streams K/V tiles, maintains fp32 online-softmax state, applies
causal and prefix masks inside the tile, and computes P@V without materializing
`[B,H,S,S]` scores or a dense causal mask. The provisional seven-case run
passed 35 accuracy trials with zero failed elements and measured 1.359647x
geomean, 1.138x to 1.566x per case.

The long-attention incremental allocation fell from 78 MiB to 22 MiB (71.8%).
Causal and causal-plus-padding cases reduced incremental allocation by 52.4%
and 54.4%. Profiler events proved one custom attention launch per layer.

A causal loop-frontier pruning idea and alternate early tile/stage policies did
not improve the end-to-end matrix and were discarded. Direct fp16 attention
remains tested, but automatic deep-stack fp16/bfloat16 execution uses exact
reference math because small fused differences compounded beyond the stricter
comparator.

### 3. Packed QKV and bounded vendor routing

The next accepted foundation cached a derived packed QKV weight and replaced
three projection GEMMs with one for measured eager CUDA float32 inference up to
`d_model=512`. Profiler `addmm` counts fell from 60 to 40 across five two-layer
forwards. The same change set routed a narrow, short, unmasked float32 corner to
PyTorch SDPA. The provisional geomean rose from 1.359647x to 1.497835x while all
seven cases remained correct.

Packing is deliberately disabled for training, CPU, low precision,
`torch.compile`, and wider models. QKV/output/FFN arithmetic otherwise remains
in vendor GEMMs; no custom GEMM was accepted.

### 4. Organizer reconciliation and precision repair

The organizer downloads and published final matrix changed the proof surface.
The first final-matrix artifact correctly failed: only 1 of 13 executable rows
passed and 11,869 of 938,885,120 elements failed. The tempting 3.095x geomean
was therefore invalid and was retained as
[`final-evaluator-pre-repair-failed.json`](../results/rtx-5070-ti-2026-08-28-final-evaluator-pre-repair-failed.json),
not reported as a performance result.

The repair disabled TF32 inside causal Triton dots, routed unsupported head
dimensions and causal batches above 128 to explicit reference math, and added
boundary tests. The green baseline then passed all 13 executable rows with zero
failed elements at 1.439957x. The exact organizer-default case and the broader
source-derived matrix were also made first-class gates.

### 5. EXP-001: short `head_dim=64`

The row-10 profile showed 30,324.486 us across 40 `_attention_fwd` launches,
79.6% of recorded GPU time, and 2,468 spills with 81,920 bytes of shared
memory. The accepted 32x64 tile reduced compiled evidence to two spills and
49,152 bytes.

| Full-matrix pair | Control geomean | Candidate geomean | Relative gain | Control row 10 | Candidate row 10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.303394x | 1.420410x | 8.98% | 3.5660 ms | 0.9703 ms |
| 2 | 1.309126x | 1.442585x | 10.19% | 3.7258 ms | 0.9717 ms |

Exact-reference routing projected only about a 4.7% aggregate gain and left row
10 below parity. SDPA failed one element in five exact row-10 trials. Both were
rejected. The bounded `head_dim == 64 and seq_len <= 128` Triton tile was
independently approved. See [EXP-001](EXP-001-head64-short-tiles.md).

### 6. Campaign 2: long and short `head_dim=32`, then the first width-eight probe

Campaign 2 introduced the immutable attempt logger and ran 39 child commands.

EXP-002 tested three long-row-13 policies against a 17.6598 ms control:

| Candidate | Row-13 optimized median | Speedup | Decision |
| --- | ---: | ---: | --- |
| `BLOCK_N=128`, two stages | 41.3406 ms | 2.047x | reject; much slower |
| two stages only | 17.9966 ms | 4.701x | reject; 1.91% latency regression |
| `BLOCK_M=32` | 21.5897 ms | 3.918x | reject; slower |

EXP-003 then tested the short row-1 path:

| Policy | Row-1 optimized median | Correctness | Decision |
| --- | ---: | --- | --- |
| unchanged 64x128 | 1.2358 ms | 0/5,242,880 failed | control |
| 32x64 | 0.8579 ms | zero failed | superseded |
| 32x128 | 0.8805 ms | zero failed | reject |
| 64x64 | 0.8164 ms | zero failed | keep |

Two alternating confirmations selected 64x64 at a 0.8201 ms mean and 0.0036
ms sample standard deviation, versus 1.2402 ms for the unchanged policy. After
integration, the final geomean rose from 1.426692x to 1.525823x (+6.948%), and
row-1 attention time fell 69.98%. See [EXP-003](EXP-003-short-head32-kv-tiles.md).

EXP-004 tried to add direct `head_dim=8` support without changing the kernel's
dot width. Eleven other direct tests passed, but Triton compilation rejected
the width-eight dot because `K >= 16` is required. The attempt stopped before
model timing and production kept exact reference fallback. Campaign 4 later
used this failure as the design constraint for a distinct padded-width solution.

### 7. Campaign 3: short `head_dim=128`

Fresh row-9 timing was 1.2341 ms optimized versus 1.1745 ms reference (0.952x),
and the profile attributed 6,775.468 us across 40 calls to attention. Three
Triton candidates were screened:

| Tile | Optimized median | Decision |
| --- | ---: | --- |
| 16x64 | 0.9609 ms | reject; slower than 32x32 |
| 32x32 | 0.9042 ms | keep |
| 16x32 | 1.1087 ms | reject |

A backend screen made SDPA appear promising at 0.69224 ms, but the first
production route missed the organizer's all-valid mask and still executed
Triton. That run was retained as rework. The corrected SDPA route proved 112
SDPA calls and exact correctness, then measured 0.9613/0.9716/0.9636 ms. The
32x32 Triton candidate reproduced 0.9042/0.9049/0.9034 ms and was 6.16% faster
by the three-run medians, so SDPA was rejected for this production route.

The integrated final geomean rose to 1.555780x (+1.963%). The nominal aggregate
gain was below 5%, but the explicit alternative acceptance gate was met:
`_attention_fwd` fell 55.454% and the ten-step model range fell 26.962%, with
zero correctness failures. Reserved EXP-007 projection/launch-count work was
not run because the new profile left vendor GEMMs and native LayerNorm as the
leaders; continuing would have violated the stop rule.

### 8. Campaign 4: exact final-row-11 `head_dim=8`

Fresh row 11 used reference attention and measured 5.6149 ms. Reference
attention dominated the 41,658.659 us ten-step model range. Exact SDPA was
correct and advanced at 1.8658 ms, then the padded-width Triton design compiled
and passed direct float16/float32 mask and boundary tests.

| Candidate | Backend / tile | Exact row-11 optimized median | Decision |
| --- | --- | ---: | --- |
| fresh control | exact reference | 5.6149 ms | control |
| EXP-008-I1 | SDPA | 1.8658 ms | correct, superseded |
| EXP-009-I1 | padded Triton 64x128 | 1.2739 ms | correct, superseded |
| EXP-009-I2 | padded Triton 64x64 | 1.0595/1.0628/1.0624 ms | keep |
| EXP-009-I3 | padded Triton 32x64 | 1.3105 ms | reject; 18.93% slower than I2 median |

The accepted kernel separates the real `HEAD_DIM=8` from compile-time
`DOT_HEAD_DIM=16`. Padded Q/K/V lanes load as zero, only eight output lanes are
stored, and the scale remains `8**-0.5`. Automatic model routing enables this
path only for `B=64,S=128,d_model=128,heads=16,layers=4,causal=true`. Final row
7 and every other unmeasured model-level width-eight shape remain on exact
reference math.

The primary final geomean improved 14.417% from Campaign 3 to 1.780075x; the
complete confirmation measured 1.784920x. Integrated row-11 attention used 40
Triton calls, and the ten-step model range fell 74.57% versus the fresh
reference profile. Candidate and final reviews approved local acceptance with
no blocker for the exact fingerprint.

## Rejected, failed, reworked, and deliberately unrun work

These outcomes are part of the result, not missing successes:

| Area | Outcome | Why it was not selected |
| --- | --- | --- |
| Standalone Triton LayerNorm | rejected | Accurate but only 0.456x-0.693x as fast as native CUDA |
| Residual + LayerNorm fusion | deferred | Profile ceiling too small for the numerical/support risk |
| Custom output/FFN GEMMs | deliberately unrun | Vendor GEMMs dominated and no evidence justified replacing them |
| Broad reduced-precision auto route | rejected for model auto | Deep-stack differences exceeded the strict comparator |
| Causal TF32 attention | repaired | Rare final-matrix misses; causal dots now use IEEE fp32 |
| Causal loop-frontier prune | rejected | No full end-to-end matrix improvement |
| EXP-001 exact reference route | rejected | Only about 4.7% projected aggregate gain; row 10 remained below parity |
| EXP-001 SDPA route | rejected | One failed element in five exact trials |
| EXP-002 long `head_dim=32` tiles/stages | rejected | All three were slower than the fresh control |
| EXP-003 32x64 and 32x128 | superseded/rejected | Correct but slower than 64x64 |
| EXP-004 direct unpadded `head_dim=8` | rejected | Triton dot requires `K >= 16` |
| EXP-005 16x64 and 16x32 | rejected | Correct but slower than 32x32 |
| EXP-006 production SDPA | reworked then rejected | First route did not activate; corrected route was slower than 32x32 Triton |
| EXP-007 projection/launch work | deliberately unrun | No post-EXP-005 profile authorization |
| EXP-008 exact row-11 SDPA | superseded | Correct, but padded Triton was 43.06% faster by selected medians |
| EXP-009 64x128 and 32x64 | superseded/rejected | Correct, but slower than 64x64 |

The nine logged nonzero child commands were retained verbatim:

| Campaign | Failed gate | Evidence and resolution |
| --- | --- | --- |
| C2 | post-merge full suite | 97 passed, 4 stale-fingerprint assertions failed; artifacts were rebaselined |
| C2 | post-rebaseline full suite | stale held-out metric plus Windows console tee encoding failure; both were repaired |
| C2 | EXP-004 direct width-eight gate | 11 tests passed before the `K >= 16` compile failure; candidate rejected |
| C2 | first logger portability hardening | 16 passed, 1 LF/CRLF test assumption failed; assertion made portable |
| C4 | default Python import | Python 3.14 lacked Torch; pinned Python 3.12 environment used |
| C4 | first workflow validation | two final-report sections and one graph closure label were missing; workflow reworked |
| C4 | baseline portability gate | logger persisted an active `.venv` interpreter path; canonicalization fixed |
| C4 | first row-11 profile setup | final matrix used `explicit_cases`, profiler required `cases`; dedicated manifest added |
| C4 | first I2 focused gate | 13 tests passed and one stale duplicate assertion failed; duplicate expectation corrected |

Campaign 2 also documents two logger self-test failures that happened before a
durable attempt could be written: a direct-entrypoint import-path error and a
dictionary-union precedence error. Campaign 4 documents two pre-logger startup
incidents totaling 0.122 seconds: default Python without Torch and a stale
Linux-style `.venv` without a Windows interpreter. They are disclosed without
being misrepresented as immutable attempt JSON.

## Current best implementation

The retained execution policy is:

| Surface | Current policy |
| --- | --- |
| Attention algorithm | One Triton launch per layer for tiled QK, online softmax, masks, and P@V; fp32 softmax accumulation |
| Short `head_dim=32`, `S<=128` | 64x64, 4 warps, 2 stages |
| Short `head_dim=64`, `S<=128` | 32x64, 4 warps, 2 stages |
| Short `head_dim=128`, `S<=128` | 32x32, 4 warps, 2 stages |
| Direct `head_dim=8`, `S<=128` | 64x64 with internal dot width 16 |
| Model-level `head_dim=8` | Triton only for exact final row 11; otherwise reference |
| Long supported shapes | Conservative measured tile/stage policy; no Campaign 2 long-head32 change retained |
| QKV projection | One cached packed vendor GEMM for eager CUDA float32 inference through `d_model=512` |
| Short unmasked float32, `head_dim<=32` | PyTorch SDPA where the measured dispatcher prefers it |
| Causal float32 Triton | IEEE fp32 dot products; no TF32 |
| Low precision, unsupported widths/layouts, large causal batches, CPU, training | Explicit correctness-first fallback |
| LayerNorm, output projection, FFN | Native PyTorch/vendor kernels |

This is the primary Campaign 4 final table:

| Row | B | S | d / heads | Head dim | Baseline | Optimized | Speedup | Backend |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 64 | 128 | 128 / 4 | 32 | 1.5453 ms | 0.9318 ms | 1.658x | Triton |
| 2 | 1 | 128 | 128 / 4 | 32 | 1.7926 ms | 0.9858 ms | 1.818x | Triton |
| 3 | 4 | 128 | 128 / 4 | 32 | 1.4219 ms | 0.8233 ms | 1.727x | Triton |
| 4 | 16 | 128 | 128 / 4 | 32 | 1.3349 ms | 0.7683 ms | 1.738x | Triton |
| 5 | 128 | 128 | 128 / 4 | 32 | 2.7514 ms | 1.4872 ms | 1.850x | Triton |
| 6 | 10,000 | 128 | 128 / 4 | 32 | 368.8780 ms | 345.3306 ms | 1.068x | reference |
| 7 | 64 | 128 | 32 / 4 | 8 | 1.4298 ms | 1.3406 ms | 1.067x | reference |
| 8 | 64 | 128 | 1024 / 4 | 256 | 13.8622 ms | 13.6390 ms | 1.016x | reference |
| 9 | 64 | 128 | 128 / 1 | 128 | 1.2179 ms | 0.9005 ms | 1.352x | Triton |
| 10 | 64 | 128 | 128 / 2 | 64 | 1.3677 ms | 0.8965 ms | 1.526x | Triton |
| 11 | 64 | 128 | 128 / 16 | 8 | 5.7669 ms | 1.0690 ms | 5.395x | Triton |
| 12 | 64 | 32 | 128 / 4 | 32 | 1.3897 ms | 0.8094 ms | 1.717x | Triton |
| 13 | 64 | 1024 | 128 / 4 | 32 | 86.3632 ms | 17.7528 ms | 4.865x | Triton |
| 14 | 32 | 100,000 | 1024 / 16 | 64 | - | - | not counted | authorized resource skip |

Fresh selected-submission gates:

| Gate | Current result |
| --- | --- |
| Complete final primary / confirmation | 13/13 executable PASS + exact skip twice; 0/938,885,120 failed each; 1.775778x / 1.770185x |
| Untouched organizer default | 5/5 PASS; 0/2,621,440 failed; 1.352x; 1,950 Triton calls |
| Project-held-out primary / confirmation | 7/7 PASS twice; 0/13,117,440 failed each; 1.210008x / 1.266010x; long-causal 0.793x / 0.800x |
| Source-derived matrix | 28/28 executable PASS + exact skip; 0/459,776,000 failed; 1.203466x overall |
| Integrated row-11 profile | 40 Triton calls; `_attention_fwd` 4,763.665 us |
| Complete repository suite | 115/115 PASS; 14 upstream PyTorch deprecation warnings |

## What worked and why

- Profile-led, shape-specific launch policy produced the repeatable wins. Each
  accepted tile addressed a measured target and stopped at an exact guard.
- Correctness-first fallbacks preserved the zero-failure contract. Reference is
  intentionally the best choice for rows 6-8 even when it limits speedup.
- Paired and counterbalanced confirmations separated candidate effects from
  substantial sub-millisecond and baseline timing drift.
- Backend counts plus profiler events prevented false custom-kernel claims.
- The failed EXP-004 compile gate supplied the key Campaign 4 design constraint:
  pad the internal dot width, not the public tensor or scale.
- Packed QKV removed two projection launches without competing with cuBLAS.
- Immutable failed attempts prevented repeated dead ends and made infrastructure
  repairs auditable rather than invisible.

## Why this is the stopping point

The remaining slow or near-parity final rows deliberately use reference math
because their shapes are unsupported, accuracy-sensitive, or dominated by
vendor GEMMs. The implemented changes already cover every profile-authorized
attention target. Further broad fusion, custom GEMMs, or routing expansion would
need a new hardware-specific profile and a new logged campaign; it is not
justified by the current evidence.

The result is locally accepted, not a Devpost release. At selection-validation
time it was dirty and uncommitted on base commit `b41fdaf`; later Git packaging
preserves that historical benchmark provenance rather than relabeling the
artifacts as clean. No tag, branch rewrite, Devpost action, or public submission
is implied by the optimization approval.

## Limitations

- Performance is specific to the recorded RTX 5070 Ti, driver 616.56, native
  Windows, PyTorch 2.13.0+cu130, CUDA 13.0, and Triton 3.7.1 environment.
- The organizer final table omits dtype, padding, timing, tolerance, and backward
  policy. Current final-shape evidence uses documented PyTorch assumptions.
- Row 14 is an authorized non-pass resource skip, not a successful execution.
- The implementation is forward inference only; no backward kernel is claimed.
- Row-11 maximum absolute difference is nonzero but passes the executable OR
  comparator. The route must remain exact until broader model evidence exists.
- Pre-Campaign-2 work lacks a complete immutable attempt/wall-time ledger.

## Audit index

- Contract: [requirements](../REQUIREMENTS.md)
- Loop controls: [optimization loop plan](../OPTIMIZATION_LOOP_PLAN.md)
- Public technical narrative: [technical report](../TECH_REPORT.md)
- EXP-001 decision: [short head-dimension-64 tiles](EXP-001-head64-short-tiles.md)
- Campaign 2 and EXP-003: [CAMPAIGN-002](CAMPAIGN-002.md) and
  [EXP-003](EXP-003-short-head32-kv-tiles.md)
- Campaign 3: [CAMPAIGN-003](CAMPAIGN-003.md)
- Campaign 4: [CAMPAIGN-004](CAMPAIGN-004.md)
- Current result index: [result artifacts](../results/README.md)
- Current primary result: [selected-submission final JSON](../results/rtx-5070-ti-2026-08-28-submission-final.json)
- Current confirmation: [selected-submission final confirmation JSON](../results/rtx-5070-ti-2026-08-28-submission-final-confirmation.json)
- Current held-out results: [selected-submission held-out JSON](../results/rtx-5070-ti-2026-08-28-submission-heldout.json) and [confirmation](../results/rtx-5070-ti-2026-08-28-submission-heldout-confirmation.json)
- Current source-derived result: [selected-submission source-derived JSON](../results/rtx-5070-ti-2026-08-28-submission-source-derived.json)
- Current profiler proof: [selected-submission row-11 profile JSON](../results/rtx-5070-ti-2026-08-28-submission-final-11-profile.json)
- Selection-validation ledger: [submission validation](SUBMISSION_VALIDATION.md)
- Campaign 4 AI Council and reviews: [`reviews/`](reviews/)

No raw attempt or result was deleted, renamed, or hand-edited during either the
historical consolidation or selected-submission revalidation. Fresh selection
artifacts use distinct `submission-*` names and immutable `S1-*` attempt
records; historical Campaign 2-4 evidence remains unchanged.
