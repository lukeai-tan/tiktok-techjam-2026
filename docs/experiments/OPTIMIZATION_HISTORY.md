# Complete Track 3 Optimization History

Status: canonical cross-campaign history, reconciled through Campaign 11 closure on 2026-08-30

Current measured implementation snapshot: Campaign 11 is integrated on
`feat/jared-attempt` with the packaged fingerprint below. Its immutable evidence
was captured from the pre-packaging checkpoint `8c89d1d`; the artifacts retain
that historical dirty-state provenance even though the implementation is now
checked in.

Packaged evidence SHA-256: `908a0d708cd8f70f44d5f14fda93d3cafb1cc18345f43914e715594cfa7b7ef9`

Historical Campaign 11 candidate SHA-256:
`9c326536ea27cfc619f01531152b2c82986d9dc3f4274691d3e8191bbb0804eb`.
Packaging the adapter and canonical benchmark layout produced `908a0d...`;
the immutable attempt records retain whichever identity they actually ran.

Start with the [documentation hub](../README.md) or the
[campaign run-through](CAMPAIGN_RUN_THROUGH.md). This file is the detailed
chronology, metric ledger, and evidence index; it is intentionally not the
first-read narrative.

This is the canonical metric, chronology, and evidence index for the complete
optimization program. The shorter executive ranking and narrative live in
[CAMPAIGN_RUN_THROUGH](CAMPAIGN_RUN_THROUGH.md); individual campaign files
remain immutable-detail ledgers. This document does not replace or rewrite the
attempt records, measured result JSON, campaign ledgers, or review sidecars.

## Verdict

The selected measured implementation is the packaged Campaign 11 fingerprint
above. Its
primary organizer-final run is **13/13 executable PASS** plus the one exact
authorized resource skip, with **0 failed elements across 938,885,120
comparisons** and **1.977420208192665x** geometric-mean end-to-end speedup. A
second complete run reproduced the contract and backend counts at
**1.9864990073497601x**. The strongest single final case is row 11 at **6.377x**;
dedicated long runs put row 5 at **1.880x**, row 6 at **1.546x**, row 9 at
**1.150x**, and row 11 at **4.710x**.

The fresh organizer default is **1.385x**, source-derived geomean is
**1.206505x**, and two five-seed held-out runs are **1.339847x** and
**1.386495x**. Four measured-fingerprint held-out matrices plus a dedicated long
run place non-padded long-causal at **1.198x-1.204x**.

Campaign 5 retains the highest historical full-matrix observation at 1.995117x,
but unrelated whole-matrix baseline movement makes cross-campaign headline
geomeans unsuitable as a causal decision metric. Campaign 11 is selected because
same-window row-9 evidence isolates its benefit: 240 residual adds and 240
native norms become 240 fused launches, repeated combined device time falls
41.77%, and the counterbalanced 300-sample optimized median falls 12.05%
without increasing the measured peak. Top-level profiler time is noisy and is
not used as causal proof. No historical result
is silently rewritten.

## Flagship decision

The submission/flagship choice is **Campaign 11**, not the single run with the
largest historical geomean. Campaign 11 is the current cumulative fingerprint,
has the latest complete zero-failure final pair, the measured 148-test gate,
the current 164-test maintenance gate, and isolates
its new row-9 benefit against contemporaneous unchanged controls. Campaign 5 is
the best broad architectural and historical-generalization campaign; Campaign
7 is the best high-volume row-6 latency specialist; and the Campaign 4 plus
Campaign 8 row-11 lineage is the strongest single-row result. The full ranked
rationale and limits are kept in the
[campaign run-through](CAMPAIGN_RUN_THROUGH.md#flagship-and-strongest-specialist-campaigns).

The winning design is not one universal launch. It combines fused Triton online
softmax attention, packed QKV projection in its measured envelope, exact
shape-aware launch policies for short head dimensions, and correctness-first
fallbacks. The largest final campaign gain came from making `head_dim=8` legal
inside Triton's 16-lane dot-product minimum by zero-padding only the internal
dot width. Campaign 5 then applies that existing kernel only to later layers of
the two accuracy-sensitive reference rows and uses exact-shape SDPA only where
five-seed held-out evidence proves it. Campaign 6 adds exact-width-1024 packed
QKV while keeping widths 513-1023 on separate projections. Campaign 7 adds an
exact-row-6 residual/normalization fusion after rejecting wide-head attention;
Campaign 8 extends that same forward only to exact row 11. Campaign 9 closes
two unsuitable surfaces, Campaign 10 extends the proven forward only to exact
row 5, and Campaign 11 extends it only to exact row 9.

## Current optimized versus original snapshot

The current comparison is consolidated here rather than kept as a separate
report. “Original” is the byte-preserved organizer `BaselineTransformer`; the
optimized side is the Campaign 11 fingerprint above. Both use strict-copied
weights and identical inputs, correctness runs before timing, and baseline and
optimized timing order alternates. The raw JSON remains authoritative for every
sample and backend count.

| Gate | Correctness | Current result versus original |
| --- | --- | ---: |
| Untouched organizer default | 5/5 PASS; 0/2,621,440 failed | 1.385x median speedup |
| Published final primary | 13/13 executable PASS + authorized resource skip; 0/938,885,120 failed | 1.977420x geomean |
| Published final confirmation | same zero-failure contract and backend counts | 1.986499x geomean |
| Project-held-out primary / confirmation | 7/7 PASS twice; 0/13,117,440 failed per run | 1.339847x / 1.386495x geomean |
| Source-derived matrix | 28/28 executable PASS + authorized resource skip; 0/459,776,000 failed | 1.206505x geomean |

The dedicated long gates show where the cumulative implementation earns its
latency: row 5 is `2.186832 -> 1.163168 ms` (1.880066x), row 6 is
`291.417252 -> 188.457397 ms` (1.546330x), row 9 is
`0.825328 -> 0.717648 ms` (1.150046x), and row 11 is
`4.195168 -> 0.890672 ms` (4.710116x). Campaign 11's isolated row-9 change is
the controlled decision: two unchanged controls average 0.815968 ms, the
optimized median is 0.717648 ms (-12.05%), and incremental peak allocation is
unchanged at 29,360,128 bytes. See the [curated result index](../results/README.md)
for the exact artifact links.

## Evidence authority and inventory

Evidence is read in this order:

1. The untouched organizer downloads, final shape manifest, requirements, and
   checked-in comparator.
2. Machine-generated result JSON and its recorded implementation fingerprint,
   source hashes, environment, command, accuracy counts, timings, and backends.
3. Immutable Campaign 2-11 and selected-submission attempt JSON, including
   nonzero commands.
4. Campaign and experiment decision records plus independent review sidecars.
5. This curated history, the implementation evidence reference, and README summaries.

At the terminal Campaign 11 checkpoint the working tree contains **361 curated
result JSON files** and **668 immutable attempt JSON files** with **616 passing
child commands, 52 retained non-pass commands, zero timeouts, and 5,895.071878
seconds** of measured child-command wall time. Cleanup accounting and later
validation records append after this exact checkpoint. These times exclude
unwrapped orchestration and pre-ledger work.

| Record set | Attempts | Child PASS | Child FAIL | Measured child wall time | Detailed ledger |
| --- | ---: | ---: | ---: | ---: | --- |
| Campaign 2 | 39 | 35 | 4 | 255.272605 s | [CAMPAIGN-002](CAMPAIGN-002.md) |
| Campaign 3 | 31 | 31 | 0 | 218.214354 s | [CAMPAIGN-003](CAMPAIGN-003.md) |
| Campaign 4 | 44 | 39 | 5 | 315.261251 s | [CAMPAIGN-004](CAMPAIGN-004.md) |
| Submission selection | 25 | 24 | 1 | 223.237808 s | [SUBMISSION_VALIDATION](SUBMISSION_VALIDATION.md) |
| Current vs original | 6 | 6 | 0 | 128.357928 s | [consolidated snapshot](OPTIMIZATION_HISTORY.md#current-optimized-versus-original-snapshot) |
| Alternate-branch comparison | 19 | 15 | 4 | 352.965078 s | [BRANCH_IMPLEMENTATION_COMPARISON](BRANCH_IMPLEMENTATION_COMPARISON.md) |
| Campaign 5 | 74 | 65 | 9 | 736.085883 s | [CAMPAIGN-005](CAMPAIGN-005.md) |
| Campaign 6 | 121 | 118 | 3 | 874.998508 s | [CAMPAIGN-006](CAMPAIGN-006.md) |
| Campaign 7 | 75 | 70 | 5 | 1,022.152100 s | [CAMPAIGN-007](CAMPAIGN-007.md) |
| Campaign 8 | 68 | 61 | 7 | 532.609478 s | [CAMPAIGN-008](CAMPAIGN-008.md) |
| Campaign 9 | 35 | 32 | 3 | 168.347636 s | [CAMPAIGN-009](CAMPAIGN-009.md) |
| Campaign 10 | 56 | 53 | 3 | 540.442413 s | [CAMPAIGN-010](CAMPAIGN-010.md) |
| Campaign 11 through `C11-CLOSE-007` | 75 | 67 | 8 | 527.126837 s | [CAMPAIGN-011](CAMPAIGN-011.md) |
| **Logged total through `C11-CLOSE-007`** | **668** | **616** | **52** | **5,895.071878 s** | [`attempts/`](attempts/) |

Numeric subtotals are summed from raw attempt values before rounding. Each
terminal attempt retains its own machine-readable duration.

A passing child command is not the same as an accepted optimization. Campaign 3
has no execution failures, for example, but still contains eight rejected
candidate records and two rework records. Final dispositions live in the
campaign ledgers because the immutable attempt JSON preserves the state known at
execution time.

The foundational phase and EXP-001 predate the Campaign 2 logger. They retain
curated measurements, profiler artifacts, Git revisions, and review decisions,
but there is no complete per-command count or wall-time ledger for that period.
This history does not invent those missing values.

Campaign 11 is the terminal executed campaign. Campaign 12 was only drafted
after closure and was stopped before any preflight, profile, benchmark, source
change, or immutable attempt; its unused scaffold was removed during cleanup.
Twelve obsolete worktrees, seven generated cache directories, and three
unreferenced smoke outputs were also removed. One EXP-025 evidence worktree is
retained solely because raw artifacts referenced by four immutable attempts
still live there. No immutable campaign evidence or Git ref was deleted.

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
| Campaign 5 fresh baseline | 14 final rows | same contract and backend counts | 1.776534x | Rebased the earlier flagship before new candidates |
| Campaign 5 primary | 14 final rows | same zero-failure contract | 1.911947x | Accepted row-6/row-7 layer hybrids; +7.62% versus fresh Campaign 5 baseline |
| Campaign 5 confirmation | 14 final rows | same contract and backend counts | 1.995117x | Complete reproduction of the selected fingerprint |
| Campaign 6 fresh control | 14 final rows | same contract and backend counts | 1.838500x | Rebased the Campaign 5 implementation in the Campaign 6 window |
| Campaign 6 primary | 14 final rows | same zero-failure contract | 1.872916x | Retained exact-width-1024 packed QKV; +1.872% versus the fresh control |
| Campaign 6 confirmation | 14 final rows | same contract and backend counts | 1.863721x | Complete reproduction within 0.491% of the primary |
| Campaign 7 fresh control | 14 final rows | same contract and backend counts | 1.863463x | Rebased the Campaign 6 implementation before new candidates |
| Campaign 7 primary | 14 final rows | same zero-failure contract | 1.880620x | Retained exact-row-6 residual/LayerNorm fusion; +0.921% versus the fresh full-matrix control |
| Campaign 7 confirmation | 14 final rows | same contract and backend counts | 1.927261x | Complete reproduction within 2.480% of the primary |
| Campaign 8 primary | 14 final rows | same zero-failure contract | 1.876167x | Extended exact residual/LayerNorm fusion to row 11; dedicated retained timing is 9.70% lower |
| Campaign 8 confirmation | 14 final rows | same contract and backend counts | 1.911052x | Complete reproduction within 1.859% of the primary |
| Campaign 9 fresh control | 14 final rows | same zero-failure contract | 1.893x | Reprofiled row 8 and row 13; no candidate survived |
| Campaign 10 primary | 14 final rows | same zero-failure contract | 1.926716x | Extended exact residual/LayerNorm fusion to row 5; dedicated optimized median -11.58% |
| Campaign 10 confirmation | 14 final rows | same contract and backend counts | 1.939005x | Complete reproduction within 0.638% of the primary |
| Campaign 11 primary | 14 final rows | same zero-failure contract | 1.977420x | Extended exact residual/LayerNorm fusion to row 9; controlled optimized median -12.05% |
| Campaign 11 confirmation | 14 final rows | same contract and backend counts | 1.986499x | Complete reproduction within 0.459% of the primary |

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

Packing is deliberately disabled for gradient-enabled execution, CPU, low
precision, `torch.compile`, and wider models. `.train()` under
`torch.inference_mode()` can still use the derived cache because this guard
tracks gradient state. QKV/output/FFN arithmetic otherwise remains in vendor
GEMMs; no custom GEMM was accepted.

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
independently approved. See the [foundational phase](CAMPAIGN_RUN_THROUGH.md#foundational-phase-from-prototype-to-a-defensible-baseline).

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
row-1 attention time fell 69.98%. See the [Campaign 2 ledger](CAMPAIGN-002.md).

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

### 9. Campaign 5: layer-aware residual routes and long-causal recovery

Campaign 5 started from a fresh 1.776534x final matrix and profiled the three
remaining reference rows plus both long-causal held-out cases. It retained five
important negative backend screens: full Triton/SDPA each failed row 7 by one
element; full Triton/SDPA each failed row 6 by 21 elements; and row-8 SDPA
failed one element while vendor `aten::addmm` consumed about 71% of its profile.

| Candidate | Exact route | Accuracy | Performance | Decision |
| --- | --- | --- | --- | --- |
| EXP-010-I1 | row 7: first 3 Triton, final reference | 1/1,310,720 failed | rejected before promotion | reject |
| EXP-010-I2 | row 7: first 2 Triton, final 2 reference | zero failed; 18 stress PASS | 1.276x target; 1.280x/1.348x confirmation | superseded |
| EXP-010-I3 | row 7: first reference, final 3 Triton | zero failed; 18 stress PASS | 1.484x target; model profile -33.64% | keep |
| EXP-011-I1 | row 6: first reference, final 3 Triton | 1/819,200,000 failed | rejected before promotion | reject |
| EXP-011-I2 | row 6: first 2 reference, final 2 Triton | 0/819,200,000 failed | 1.549x target; model profile -19.74% | keep |
| EXP-012-I1 | exact held-out long-causal padded/unpadded SDPA | zero failed across stress and two complete five-seed matrices | 1.247x/1.280x primary targets | keep |

The composite primary final geomean rose 7.62% to 1.911947x, with a complete
1.995117x confirmation and identical correctness/backend counts. Five-seed
held-out geomeans were 1.447477x and 1.449715x. The complete source-derived
matrix remained 28/28 executable PASS at 1.204815x, and the untouched organizer
default remained 5/5 PASS at 1.397x. Row 8 was deliberately left on exact
reference rather than spend the hypothesis cap on a failed, low-ceiling route.

### 10. Campaign 6: exact-width packed QKV and launch plateaus

Campaign 6 reopened four surfaces only after fresh profiles. Five row-6 launch
variants and five row-7 padded-width variants were accurate but slower or
profiler-neutral. Three new row-11 query-tile/warp variants were 6.42%-19.77%
slower than the long control. Those loops closed as measured plateaus without
changing the accepted attention policy.

The row-8 loop compared projection strategies over 300 samples per run. A
two-plus-one grouping measured 0.988851x and was rejected. The original packed
candidate measured 1.022022x, 1.030071x, and 1.023827x while two same-window
unchanged controls measured 0.981690x and 0.993542x. Independent review rejected
its broad `d_model <= 1024` boundary, so the rework narrowed it to exact 1024
while preserving the prior `<=512` envelope and leaving widths 513-1023 alone.

The accepted rework passed exact row 8, a width-1024 held-out neighbor, strict
cache/state-dict tests, two complete final matrices, two five-seed held-out
matrices, the untouched organizer default, and all 28 feasible source-derived
cases. Against the contemporaneous Campaign 5 row-8 profile, integrated
`aten::addmm` calls fell 240 to 160, `addmm` device time fell 11.33%, and
ten-forward model device time fell 7.91%. The primary final matrix is 1.872916x
and confirmation 1.863721x; these are Campaign-6-fingerprint measurements, while
Campaign 5's higher historical geomeans remain historical observations.

### 11. Campaign 7: reject wide-head attention, fuse row-6 residual/norm

Campaign 7 opened from checkpoint `8c89d1d` and fingerprint `54df021e...d449ff4`.
The fresh full matrix was 13/13 executable PASS plus the authorized skip at
1.863463x. Row-6 profiling measured 2,026,089.666 us of model device time over
ten forwards; residual adds consumed 203,340.055 us and native LayerNorm
282,683.278 us, a combined 24.0% ceiling. Row 8 remained projection dominated,
but its exact `head_dim=256` attention was still an untested kernel width.

EXP-017 tested that width in a detached checkpoint worktree. A 16x16 primitive
compiled and passed direct arithmetic, but the four-layer row failed two of
41,943,040 elements. A 16x32 variant reproduced the same two misses. A 16x64
variant could not compile because it required 151,616 bytes of shared memory on
a 101,376-byte-limit device. Keeping the first layer on exact attention repaired
accuracy and removed 150 of 200 copies plus 60 of 80 BMMs over the profiled
window, but 30 wide Triton kernels cost 29,166.344 us and increased model time
from 130,180.675 us to 141,240.672 us (+8.50%). EXP-017 was rejected.

EXP-018 fused each residual result with the LayerNorm that immediately consumes
it on exact row 6. The first implementation passed arithmetic and model accuracy
but kept normalized intermediates alive and raised peak allocation by
1,967,128,576 bytes. The rework released those lifetimes and narrowed the
runtime/mask guards. It passed 18 seed/scale/padding scenarios spanning
2,949,120,000 outputs, with 144 expected fused calls, as well as optional-bias,
state-dict, noncontiguous-mask, neighboring-shape, focused-suite, final-matrix,
held-out, organizer-default, and source-derived gates.

The integrated profile replaces 80 residual adds and 80 of 90 native norms with
80 `_residual_layer_norm_fwd` launches across ten forwards. The fused subsystem
uses 309,611.219 us versus 486,023.333 us (-36.30%), while model device time falls
from 2,026,089.666 us to 1,832,789.301 us (-9.54%). In the counterbalanced long
bracket, candidate speedups averaged 1.554314x and controls 1.419031x (+9.53%
normalized). The final 100-sample integrated run measured 293.910400 ms baseline,
189.981712 ms optimized, and 1.547046x, with the same 11,802,787,840-byte
optimized incremental peak as the unchanged control. Independent review approved
local integration, and the implementation fingerprint became
`a994eb1c...297be8b`.

### 12. Campaign 8: extend exact residual/norm fusion to row 11

The fresh row-11 baseline profile measured 41,211.814 us of model device time
over 30 forwards, with 240 residual adds and 270 native norms consuming
5,978.920 us. EXP-019-I1 reused the accepted Campaign 7 fused forward behind an
exact row-6-or-row-11 guard. It passed correctness and delivered a repeatable
latency gain, but its first 10-step profiler total contradicted wall-clock
timing. That observation was retained; paired 30-step profiles resolved it in
favor of the candidate.

Council review then found that training fallback was implicit through gradient
state. I1R added an explicit eval-mode guard plus CPU, dtype, layout,
runtime-shape, mask, gradient, training, and model-neighbor tests. It passed ten
boundary tests, all 34 affected tests, and 36 row-6/row-11 seed/scale/padding
stress scenarios covering 2,967,994,368 outputs with zero failures. Two
retained-fingerprint 300-sample runs averaged 0.897184 ms versus 0.993525 ms
across three unchanged controls (-9.70%); normalized speedup rose 10.19%, and
peak allocation stayed exactly 29,360,128 bytes.

The integrated 30-forward profile records 240 fused residual/norm launches,
30 remaining native norms, no residual-add event, and 120 Triton attention
calls. Model device time falls 21.96% and subsystem time 46.28%. EXP-020 row-7
fusion was deliberately not run because its workflow condition was fallback
only and EXP-019 had produced an accepted winner.

### 13. Campaign 9: wide fusion and long-row plateau

Campaign 9 kept fingerprint `325a1e5c...79b` frozen and reprofiled exact rows 8
and 13. Row 8 exposed a 10.44% residual/norm ceiling, but the width-1024 fused
kernel was not competitive: I1 regressed the 300-sample optimized median 0.49%
and its subsystem 1.54%; I2's eight-warp rework increased the subsystem deficit
to 5.58%. Both remained correctness-green and were removed as performance
failures.

Row 13 exposed a 9.67% ceiling and passed direct, boundary, and 18-scenario
stress gates, then failed one strict-comparator element out of 41,943,040 in
the expanded five-trial gate. Timing correctly did not run after that failure.
The candidate was removed. Campaign 9 therefore closed with no winner, no
source integration, and the complete restored suite at 139/139 PASS.

### 14. Campaign 10: exact row-5 residual/normalization fusion

Campaign 10 started from Campaign 8 fingerprint `325a1e5c...79b` and first
reprofiled exact final rows 5, 7, and 12. Row 5 had the actionable ceiling:
residual adds plus native LayerNorm consumed 12,089.646 us, or 29.07% of its
41,581.112 us 30-forward model profile. Row 7 was below the conditional
authorization threshold and row 12 remained observation-only.

EXP-023-I1 reused the already accepted fused forward behind an exact row-5
configuration and runtime-shape guard. It passed direct route/boundary checks,
18 seed/scale/padding/neighbor scenarios, 37 affected tests, and 150,994,944
stress outputs with zero failures. Against two counterbalanced unchanged
controls averaging 1.325096 ms, the retained 300-sample candidate measured
1.171584 ms (-11.58%) and raised normalized speedup 14.79%, with the same
58,720,256-byte incremental peak. The integrated long gate measured 1.162976
ms and 2.001995x.

The active 30-forward profile records 240 fused residual/norm launches, 30
remaining native norms, no residual-add event, and 120 Triton attention calls.
Relative to the fresh baseline, subsystem device time falls 40.63% and model
device time falls 11.96%. The final/confirmation pair is 1.926716x/1.939005x,
every broader matrix is zero-failure, and the complete suite is 144/144 PASS.
EXP-024 row-7 fusion was deliberately unrun because its fallback condition was
false after EXP-023 produced a winner.

### 15. Campaign 11: exact row-9 residual/normalization fusion

Campaign 11 started from Campaign 10 fingerprint `f7ad2a86...38d4` and profiled
final rows 9, 10, and 1. Row 9 was the primary target because it was the slowest
remaining width-128/head-count row by fresh final speedup and its residual-add
plus native-LayerNorm subsystem consumed 5,769.588 us, 19.10% of the first
30-forward model profile. Row 10 was a conditional fallback and row 1 was an
observation-only non-causal control.

EXP-025-I1 reused the accepted fused forward behind the exact runtime/model
shape `(64,128,128)`, one head, FFN 128, four layers, and causal inference. It
passed two route/boundary tests, all 18 seed/scale/padding scenarios, 40 affected
tests, and the isolated final matrix with zero failures. Two unchanged controls
measured 0.815776 and 0.816160 ms optimized median. The isolated candidate was
0.717696 ms and the active transplant was 0.717648 ms, 12.05% below the control
mean and within 0.007% of the reviewed candidate. All four runs used the same
29,360,128-byte incremental peak; the active long run recorded 1,240 Triton
attention calls and zero failed elements.

Two active profiles each record 240 fused residual/norm launches, 30 remaining
native norms, and 120 Triton calls. Their mean subsystem time is 3,357.389 us
versus 5,765.324 us across two baselines (-41.77%). One active top-level profile
was slow, leaving the active pair 2.54% above the baseline pair; that variance is
disclosed and does not override the tightly counterbalanced CUDA-event timing.
The primary/confirmation final pair is 1.977420x/1.986499x with identical
correctness and backend totals. Organizer-default is 1.385x, source-derived is
1.206505x, and all inherited row-5/row-6/row-8/row-11 mechanism gates pass.
EXP-026 row-10 fusion was deliberately unrun because EXP-025 satisfied its
fallback condition.

## Rejected, failed, reworked, and deliberately unrun work

These outcomes are part of the result, not missing successes:

| Area | Outcome | Why it was not selected |
| --- | --- | --- |
| Standalone Triton LayerNorm | rejected | Accurate but only 0.456x-0.693x as fast as native CUDA |
| Broad residual + LayerNorm fusion | narrowed | Earlier profiles were too small; Campaign 7 accepted only exact row 6 after its fresh 24% ceiling and boundary proof |
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
| Campaign 5 full row-7 Triton/SDPA | rejected | Each missed one strict-comparator element |
| Campaign 5 full row-6 Triton/SDPA | rejected | Each missed 21 strict-comparator elements |
| EXP-010-I1 / EXP-011-I1 | rejected | Layer placement still accumulated one failed element |
| Campaign 5 row-8 SDPA | rejected | One failed element and a vendor-GEMM-dominated profile ceiling |
| Campaign 6 row-6 launch variants | rejected/plateau | Five accurate variants did not beat the accepted end-to-end/profile evidence |
| Campaign 6 row-7 launch variants | rejected/plateau | Five accurate variants plateaued or regressed |
| Campaign 6 row-11 launch variants | rejected/plateau | Three new axes were 6.42%-19.77% slower than the long control |
| EXP-015-I2 two-plus-one QKV | rejected | 0.988851x; slower than separate projections |
| EXP-015-I1 broad `<=1024` guard | reworked | Independent review found unmeasured widths 513-1023; I1R narrowed the route |
| EXP-017 16x16 / 16x32 wide-head attention | rejected | Each exact row-8 four-layer route missed two strict-comparator elements |
| EXP-017 16x64 wide-head attention | rejected | Required 151,616 shared-memory bytes against the 101,376-byte device limit |
| EXP-017 exact-first-layer hybrid | rejected | Correct, but row-8 model profile time regressed 8.50% |
| EXP-018 I1 fused residual/norm | reworked | Accurate, but temporary lifetimes added 1,967,128,576 peak bytes; I2 released them and I2R narrowed routing |
| EXP-019 I1 exact row-11 fusion | reworked | Correct and fast; Council required an explicit training-mode boundary |
| EXP-019 I1R exact row-11 fusion | accepted | Zero failures, -9.70% retained target median, -21.96% integrated model profile time, memory-neutral |
| EXP-020 exact row-7 fusion | deliberately unrun | Fallback condition was false after EXP-019 succeeded |
| EXP-021-I1 exact row-8 width-1024 fusion | rejected | Zero-failure stress, but optimized median regressed 0.49% and fused subsystem regressed 1.54% |
| EXP-021-I2 eight-warp width-1024 fusion | rejected | Fast correctness gates passed, but subsystem regressed 5.58% versus separate native operations |
| EXP-022-I1 exact row-13 fusion | rejected | Stress passed, but expanded seed 1238 failed 1/41,943,040 overall; timing was not run |
| EXP-024 exact row-7 fusion | deliberately unrun | Campaign 10 row-5 winner made its fallback condition false |
| EXP-025-I1 exact row-9 fusion | accepted | Zero failures; controlled optimized median -12.05%; mean subsystem profile time -41.77%; memory-neutral |
| EXP-026 exact row-10 fusion | deliberately unrun | Campaign 11 row-9 winner made its fallback condition false |

The following early infrastructure failures and Campaign 7 candidate failures
need specific context; the comprehensive run-through retains the complete
cross-campaign nonzero list, and every command remains verbatim in `attempts/`:

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
| C7 | EXP-017 16x16 row-8 model screen | 2/41,943,040 elements failed; variant rejected |
| C7 | EXP-017 16x32 row-8 model screen | Same 2/41,943,040 elements failed; variant rejected |
| C7 | EXP-017 16x64 direct compile | 151,616-byte shared-memory request exceeded the 101,376-byte limit; variant rejected |
| C7 | EXP-018 candidate full suite | 128 tests passed and five stale fingerprint-evidence tests failed; functional suite excluding only that evidence file passed 122/122, then pointers were migrated after integration |

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
| Exact final row 7 | layer 0 reference; layers 1-3 padded-width Triton |
| Exact final row 5 | four Triton attention layers plus exact eval-mode fused residual/LayerNorm |
| Exact final row 6 | layers 0-1 reference; layers 2-3 Triton; residual adds fused with downstream LayerNorms on guarded eager CUDA float32 execution |
| Exact final row 9 | four Triton attention layers plus exact eval-mode fused residual/LayerNorm |
| Exact final row 11 | four padded-width Triton attention layers plus exact eval-mode fused residual/LayerNorm |
| Other model-level `head_dim=8` | reference |
| Exact two-layer 512-token held-out causal shape | SDPA, with or without measured prefix padding |
| Long supported shapes | Conservative measured tile/stage policy; no Campaign 2 long-head32 change retained |
| QKV projection | One cached packed vendor GEMM for eager CUDA float32 inference through `d_model=512` and exact `d_model=1024`; widths 513-1023 stay separate |
| Short unmasked float32, `head_dim<=32` | PyTorch SDPA where the measured dispatcher prefers it |
| Causal float32 Triton | IEEE fp32 dot products; no TF32 |
| Low precision, unsupported widths/layouts, other large causal batches, CPU, training | Explicit correctness-first fallback |
| Other LayerNorm, output projection, FFN | Native PyTorch/vendor kernels |

This is the primary Campaign 11 final table:

| Row | B | S | d / heads | Head dim | Baseline | Optimized | Speedup | Backend |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 64 | 128 | 128 / 4 | 32 | 1.3778 ms | 0.8152 ms | 1.690x | Triton |
| 2 | 1 | 128 | 128 / 4 | 32 | 1.4685 ms | 0.8804 ms | 1.668x | Triton |
| 3 | 4 | 128 | 128 / 4 | 32 | 1.4305 ms | 0.7830 ms | 1.827x | Triton |
| 4 | 16 | 128 | 128 / 4 | 32 | 1.3367 ms | 0.7560 ms | 1.768x | Triton |
| 5 | 128 | 128 | 128 / 4 | 32 | 2.9495 ms | 1.2745 ms | 2.314x | Triton + fused residual/norm |
| 6 | 10,000 | 128 | 128 / 4 | 32 | 445.1712 ms | 332.4715 ms | 1.339x | 2 reference + 2 Triton layers; fused residual/norm |
| 7 | 64 | 128 | 32 / 4 | 8 | 1.4340 ms | 0.9723 ms | 1.475x | 1 reference + 3 Triton layers |
| 8 | 64 | 128 | 1024 / 4 | 256 | 15.0661 ms | 13.7354 ms | 1.097x | reference |
| 9 | 64 | 128 | 128 / 1 | 128 | 1.3186 ms | 0.7409 ms | 1.780x | Triton + fused residual/norm |
| 10 | 64 | 128 | 128 / 2 | 64 | 1.4622 ms | 0.9257 ms | 1.579x | Triton |
| 11 | 64 | 128 | 128 / 16 | 8 | 5.7496 ms | 0.9017 ms | 6.377x | Triton + fused residual/norm |
| 12 | 64 | 32 | 128 / 4 | 32 | 1.4426 ms | 0.8002 ms | 1.803x | Triton |
| 13 | 64 | 1024 | 128 / 4 | 32 | 88.8280 ms | 18.5412 ms | 4.791x | Triton |
| 14 | 32 | 100,000 | 1024 / 16 | 64 | - | - | not counted | authorized resource skip |

Fresh selected-submission gates:

| Gate | Current result |
| --- | --- |
| Complete final primary / confirmation | 13/13 executable PASS + exact skip twice; 0/938,885,120 failed each; 1.977420x / 1.986499x |
| Untouched organizer default | 5/5 PASS; 0/2,621,440 failed; 1.385x; 1,950 Triton calls |
| Project-held-out primary / confirmation | 7/7 PASS twice with five seeds; 0/13,117,440 failed each; 1.339847x / 1.386495x; four-run long-causal 1.198x-1.204x |
| Source-derived matrix | 28/28 executable PASS + exact skip; 0/459,776,000 failed; 1.206505x overall |
| Integrated profiles | row 5: 240 fused events, -41.96% subsystem time; row 6: 20 attention Triton/20 reference plus 80 fused events; row 8 retains packed QKV/reference attention; row 9: 240 fused events, -41.77% mean subsystem time; row 11: 240 fused events, -22.09% model time |
| Complete repository suite | 148/148 PASS; 14 upstream PyTorch deprecation warnings |

## What worked and why

- Profile-led, shape-specific launch policy produced the repeatable wins. Each
  accepted tile addressed a measured target and stopped at an exact guard.
- Correctness-first layer hybrids preserved the zero-failure contract for rows
  6-7. Reference attention remains intentional for row 8, while exact-width
  packed QKV reduces its projection work.
- Paired and counterbalanced confirmations separated candidate effects from
  substantial sub-millisecond and baseline timing drift.
- Backend counts plus profiler events prevented false custom-kernel claims.
- The failed EXP-004 compile gate supplied the key Campaign 4 design constraint:
  pad the internal dot width, not the public tensor or scale.
- Packed QKV removed two projection launches without competing with cuBLAS.
- Row-6 residual/normalization fusion succeeded only after the profiler exposed
  a 24% ceiling and the rework removed temporary lifetimes; the strict route
  guard keeps that measured win from becoming an unsupported broad policy.
- The wide-head experiment showed why launch-count reductions are insufficient:
  it removed copies/BMMs but the new kernel cost more and a wider tile exceeded
  shared-memory capacity.
- Immutable failed attempts prevented repeated dead ends and made infrastructure
  repairs auditable rather than invisible.

## Why Campaigns 8 through 10 stopped on these surfaces

Campaign 6 addressed only its measured projection bottleneck. Campaign 7 then
closed row-8 wide-head attention after two numerical failures, one shared-memory
compile failure, and one correct regression. It accepted exact-row-6 fusion only
after correctness, boundary, stress, profile, long-run, memory, and full-matrix
proof. Further broad fusion, custom GEMMs, or routing expansion needs a fresh
hardware-specific profile and a new logged campaign; the rejected variants do
not justify repeating those exact axes.

Campaign 8 closed the separately profiled row-11 residual/normalization ceiling
after one implementation and one boundary refinement. Conditional row-7 fusion
was not run because the accepted row-11 result satisfied the stop rule. Any
further fusion surface needs a new profile and campaign rather than broadening
the two exact guards.

Campaign 9 then measured the two next residual/normalization surfaces instead
of broadening blindly. Width-1024 row 8 was slower in both allowed launches,
and exact row 13 failed the zero-element accuracy contract. Those arithmetic
routes are closed; a later loop must target a different measured bottleneck.

Campaign 10 moved to the materially different medium-batch row-5 surface. Its
exact reuse of the accepted fusion cleared the complete evidence gate, so the
conditional row-7 candidate was not run. The guard remains exact to row 5;
broader width-128 fusion is not implied. A later campaign must start from a
fresh profile and cannot infer that neighboring rows share this result.

Campaign 11 then measured the head-count-specific row-9 surface rather than
broadening the existing guard. Its exact reuse cleared the complete evidence
gate, so the conditional row-10 candidate was not run. The guard remains exact
to row 9; the noisy top-level profiler pair reinforces that any later surface
needs its own counterbalanced timing and mechanism proof.

The result is locally accepted, not a Devpost release. At selection-validation
time it was dirty and uncommitted on base commit `8c89d1d`; later Git packaging
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
- Loop controls: [optimization loop plan](../AGENT_OPTIMIZATION_LOOP_PLAN.md)
- Submission-facing technical report: [SpeedROCm technical report](../../deliverables/03_TECHNICAL_REPORT.md)
- Repository evidence reference: [implementation and benchmark evidence](../IMPLEMENTATION_EVIDENCE.md)
- Foundational short-head decisions: [campaign run-through](CAMPAIGN_RUN_THROUGH.md#foundational-phase-from-prototype-to-a-defensible-baseline)
- Campaign 2 short-head decisions: [CAMPAIGN-002](CAMPAIGN-002.md)
- Campaign 3: [CAMPAIGN-003](CAMPAIGN-003.md)
- Campaign 4: [CAMPAIGN-004](CAMPAIGN-004.md)
- Campaign 5: [CAMPAIGN-005](CAMPAIGN-005.md)
- Campaign 6: [CAMPAIGN-006](CAMPAIGN-006.md)
- Campaign 7: [CAMPAIGN-007](CAMPAIGN-007.md)
- Campaign 8: [CAMPAIGN-008](CAMPAIGN-008.md)
- Campaign 9: [CAMPAIGN-009](CAMPAIGN-009.md)
- Campaign 10: [CAMPAIGN-010](CAMPAIGN-010.md)
- Campaign 11: [CAMPAIGN-011](CAMPAIGN-011.md)
- Current result index: [result artifacts](../results/README.md)
- Current primary result: [Campaign 11 final JSON](../results/rtx-5070-ti-2026-08-29-c11-integrated-final.json)
- Current confirmation: [Campaign 11 final confirmation JSON](../results/rtx-5070-ti-2026-08-29-c11-integrated-final-confirmation.json)
- Current held-out results: [Campaign 11 five-seed held-out JSON](../results/rtx-5070-ti-2026-08-29-c11-integrated-heldout-5seed.json) and [confirmation](../results/rtx-5070-ti-2026-08-29-c11-integrated-heldout-5seed-confirmation.json)
- Current source-derived result: [Campaign 11 source-derived JSON](../results/rtx-5070-ti-2026-08-29-c11-integrated-source-derived.json)
- Current profiler proof: [Campaign 11 row-9 profile JSON](../results/rtx-5070-ti-2026-08-29-c11-integrated-row09-profile.json)
- Selection-validation ledger: [submission validation](SUBMISSION_VALIDATION.md)
- Campaign 4 through Campaign 7 reviews: [`reviews/`](reviews/)

No raw attempt or result was deleted, renamed, or hand-edited during either the
historical consolidation or selected-submission revalidation. Fresh selection
artifacts use distinct `submission-*` names and immutable `S1-*` attempt
records; historical Campaign 2-4 evidence remains unchanged. Campaigns 5-11 use
their distinct `cN-*` result and `CN-*` attempt namespaces.
