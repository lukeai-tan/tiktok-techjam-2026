# Campaign 9: wide residual-normalization feasibility

Status: complete with no winner on 2026-08-29; Campaign 8 remains selected

## Objective and frozen starting point

Continue from Campaign 8's locally accepted fingerprint
`325a1e5cad70f85390ddbea438f04b60a4b0f40300826aba3991520f5b97079b`
on checkpoint `8c89d1d4170c58d16fb75d79f212e990565fba7d`. Campaign 8 closed
with two zero-failure final matrices at 1.876167x and 1.911052x, 139/139
repository tests, a blocker-free Council review, and strict graph/workflow/diff
seals.

The current exact-row-8 profile records 132,179.565 us of model device time
over ten forwards. Its 80 residual adds use 8,109.098 us and 90 native
LayerNorm calls use 6,244.311 us, a combined 10.86% ceiling. The accepted
residual-normalization primitive has been measured only at width 128; row 8 is
width 1024 and therefore requires a new compilation, correctness, launch,
memory, and timing decision rather than a broadened predicate by assumption.

Campaign 9 also profiles exact row 13 as a ranked fallback surface. It does not
retry Campaign 7's rejected row-8 attention kernels or alter Campaign 6's
accepted exact-width packed-QKV path.

## Bounded hypotheses

### EXP-021: exact row-8 width-1024 fused residual plus LayerNorm

At most two implementations may be screened. I1 reuses the proven fused
arithmetic behind an exact row-8 runtime/model guard. I2 may change only the
width-1024 launch configuration if I1 is numerically correct and memory-safe
but misses the performance threshold. Direct width-1024 arithmetic, strict
model comparison, seed/scale/padding stress, runtime and mask neighbors,
training/gradient/device/dtype/layout fallback, packed-QKV preservation, long
timing, profiler mechanism, peak allocation, and rows 6/11 non-regression are
mandatory.

### EXP-022: exact row-13 fallback

Only if EXP-021 closes without an accepted candidate and the fresh row-13
profile shows at least an approximately five-percent residual/normalization
ceiling may one exact width-128 implementation be screened. Its long-sequence
Triton attention route and all timing/comparator policy remain frozen.

## Non-goals and stop rules

- Do not retry row-8 custom attention, shared-memory-overflowing tiles, row-6,
  row-7, or row-11 attention launch variants, or packed-QKV guard expansion.
- Do not replace vendor GEMMs, change the organizer harness/comparator/timing,
  enable broad compilation, or broaden fusion beyond an exact measured row.
- Every test, profile, benchmark, review, and closure command uses
  `benchmarks/run_optimization_attempt.py`; failures and timeouts are retained.
- A retained candidate needs zero failed elements and approximately five-percent
  reproducible target/profile benefit or a material launch/memory improvement,
  with every regression and allocation cost reported.
- No commit, push, tag, release, branch creation, history rewrite, or public
  action is authorized.

## Attempt ledger

| Group | Purpose | Status |
| --- | --- | --- |
| `C9-PREFLIGHT-*` | workflow, fingerprint, contract, environment, and candidate isolation | complete |
| `C9-BASE-*` | current final matrix, exact row-8/row-13 profiles, and long controls | complete |
| `C9-EXP-021-*` | exact-row-8 width-1024 fused residual/normalization candidates | two variants rejected |
| `C9-EXP-022-*` | conditional exact-row-13 fallback | one variant rejected |
| `C9-REVIEW-*` | provenance, correctness, timing, memory, boundary, and maintenance | plateau review approved |
| `C9-INTEGRATE-*` | winner transplant and complete rebaseline | deliberately unrun; no winner |
| `C9-CLOSE-*` | docs, tests, workflow, graph, Council, and tree closure | complete |

The immutable attempt JSON is authoritative for commands, wall time, accuracy,
latency samples, memory, backend/profiler counts, environment, fingerprints,
and dispositions.

## Baseline and profile findings

The fresh complete baseline passed 13/13 executable final rows plus the exact
authorized resource skip with 0/938,885,120 failed elements and a 1.893x
geomean. Row 8 remained the weakest row at 1.046x.

Thirty-step profiles measured:

| Target | Model device time | Add + native norm | Share | Attention route |
| --- | ---: | ---: | ---: | --- |
| Exact row 8 | 395,759.491 us | 41,319.075 us | 10.44% | 120 reference calls |
| Exact row 13 | 733,988.788 us | 71,000.035 us | 9.67% | 120 Triton calls; 81.93% of model time |

The retained 300-sample row-8 control passed five accuracy trials with
0/41,943,040 failed elements, measured 13.428256 ms baseline and 13.144816 ms
optimized (1.021563x), used 1,240 reference-attention calls, and recorded a
369,115,136-byte optimized incremental peak.

## Candidate outcomes

### EXP-021 row 8

I1 compiled and passed direct width-1024 arithmetic, exact routing, common
boundaries, and all 18 seed-scale-padding scenarios. The stress matrix covered
150,994,944 outputs with zero failures, 144 expected fused calls, 72 reference
attention calls, and all four packed-QKV caches.

It did not improve performance. The 300-sample candidate median was 13.209216
ms, 0.49% slower than the 13.144816 ms control. The paired profile explained
the result: 240 fused plus 30 remaining native-norm calls used 41,954.629 us,
1.54% more than the separate 41,319.075 us baseline subsystem, while model
device time rose 1.04%.

I2 was the one permitted launch-only rework. Eight warps preserved the three
fast correctness/boundary gates but made the fused subsystem slower still at
43,624.204 us, 5.58% above the separate baseline. Both row-8 variants were
rejected and removed.

### EXP-022 row 13

The fallback was profile-authorized after row 8 closed. It passed exact routing,
common boundaries, and all 18 seed-scale-padding scenarios over 150,994,944
outputs with zero failures, 144 fused calls, and 72 Triton-attention calls.

The harder five-trial long gate then found one failed element in seed 1238:
one failure out of 8,388,608 for that trial and one out of 41,943,040 overall.
The other four trials passed. Timing and peak-memory measurement were correctly
not run after accuracy failed. The sole permitted row-13 candidate was rejected
and removed.

## Decision

Campaign 9 has no accepted candidate. Fingerprint
`325a1e5cad70f85390ddbea438f04b60a4b0f40300826aba3991520f5b97079b`
remains selected. The two failed profile-analysis commands are retained: the
first had a quoting syntax error; the second let a zero-device-time duplicate
profiler label overwrite the GPU event. `C9-BASE-004B` is the corrected
positive-device-time analysis. No commit, push, branch, tag, release, or public
action was performed or authorized.

Through `C9-CLOSE-002`, the ledger contains 32 immutable attempts: 29 PASS,
three FAIL, zero timeouts, and 166.619647 seconds of measured child-command wall
time. The three failures are the quoting parser, duplicate-label parser, and
strict row-13 accuracy gate described above. Terminal workflow and diff seals
append after this explicit accounting checkpoint rather than being predicted.
