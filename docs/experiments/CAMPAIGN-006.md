# Campaign 6: exact-shape launch and projection optimization

Status: locally complete on 2026-08-29; final Git packaging is not authorized

## Objective and frozen baseline

Continue the optimizer for a long, evidence-rich round without repeating prior
failed routes or weakening the organizer contract. The starting point is the
accepted Campaign 5 implementation on `feat/jared-attempt`, Git parent
`7f4fcba4ffb891cb876fa9ec27afa2395b99c77a`, with schema-2 implementation
fingerprint
`9159177a21d039366ed4d3aef431b4b14d3bcef26d5eeaab0808efa739294029`.
The starting tree is intentionally dirty because Campaign 5 was moved to this
branch without a commit; every Campaign 6 artifact must therefore bind claims to
the fingerprint and captured Git state rather than imply clean-commit provenance.

Campaign 5's primary and confirmation final geomeans were 1.911947x and
1.995117x. Both passed all 13 executable final rows with zero of 938,885,120
failed elements and retained the exact authorized row-14 resource skip.

## Scope and non-goals

This campaign opens four profile-authorized surfaces:

1. **EXP-013:** exact row-6 huge-batch `head_dim=32` launch geometry. The
   accepted hybrid uses reference attention in layers 0-1 and Triton in layers
   2-3; its integrated profile records 20 Triton launches and 164,981.228 us of
   `_attention_fwd` time over ten model forwards.
2. **EXP-014:** exact row-7 padded-width `head_dim=8` launch geometry, while
   protecting exact row 11. Row 7 uses reference layer 0 and Triton layers 1-3;
   its profile records 30 Triton launches and 991.253 us. Row 11 remains a
   mandatory neighbor because it shares the 64x64 padded-width kernel policy.
3. **EXP-015:** one fresh recheck of the exact row-8 wide-model projection
   surface. Its profile is dominated by vendor `aten::addmm` work, while earlier
   microbenchmarks found packed QKV neutral at width 1024. This loop stops after
   at most three variants and does not attempt custom GEMMs or broad compilation.
4. **EXP-016:** three previously untried exact row-11 padded-width launch axes:
   a 128-row query tile and two/four/eight-warp occupancy comparisons. Campaign
   4 already rejected 64x128 and 32x64 tiles; those candidates are not repeated.

The organizer sources, protected baseline class, comparator, tolerances, matrix
order, timing policy, skip policy, dependency set, state-dict contract, and
training/backward exclusion are not optimization targets. Full Triton or SDPA
routes for rows 6-8 are not retried: Campaign 5 already recorded 21, one, and one
failed elements respectively. Long `head_dim=32` tiles, standalone LayerNorm,
broad `torch.compile`, and speculative output/FFN GEMMs remain closed without
new contradictory profile evidence.

## Evidence and loop rules

- Every test, profile, benchmark, candidate screen, confirmation, review check,
  and closure gate runs through `benchmarks/run_optimization_attempt.py`.
- Attempt JSON is immutable and records UTC time, wall time, command outcome,
  stdout/stderr, environment, Git state, implementation fingerprint, result
  hash, accuracy/error totals, timing samples, memory, backend counts, profiler
  events, and the initial decision.
- Candidate code is developed in an isolated worktree copied from the exact
  Campaign 5 fingerprint. Only an independently reviewed winner is applied to
  the integration branch.
- Correctness runs before performance. Any failed element, unauthorized
  fallback/skip, error, OOM, missing artifact, or provenance mismatch rejects
  the relevant gate.
- A retained candidate needs about a five-percent reproducible target-latency,
  kernel-time, memory, or launch benefit (or about five-percent complete-matrix
  gain), with no required case regressing more than two percent without a
  documented independent decision.
- Each launch loop is capped at five variants; the projection loop is capped at
  three. Three substantially unchanged failures stop the subsystem.

## Fresh baseline and profile authorization

The opening final matrix passed all 13 executable rows plus the exact authorized
skip at **1.838500x**. Fresh target profiles and long controls established the
comparison windows used below:

| Surface | Relevant fresh control |
| --- | --- |
| row 6 | 290.6584 ms and 379.3366 ms optimized medians in two exact controls |
| row 7 | 0.733536 ms and 0.720944 ms optimized medians over 600 samples |
| row 8 | 13.966592 ms and 14.814192 ms optimized medians in the two contemporaneous controls used for the final decision |
| row 11 | 0.996448 ms optimized median over 600 samples |

The baseline profiles confirmed the campaign premise: rows 6, 7, and 11 were
attention-launch candidates, while row 8 was dominated by vendor projection and
FFN GEMMs.

## EXP-013: exact row-6 launch geometry

Every candidate preserved zero failed elements across 819,200,000 comparisons
and the required 56 Triton / 56 reference attention split. None beat the fresh
control with credible end-to-end or profiler evidence.

| Variant | Geometry change | Baseline / optimized median | Speedup | Decision |
| --- | --- | ---: | ---: | --- |
| I1 | generic short head-32 128x64 | 399.8567 / 563.6865 ms | 0.709x | reject |
| I2 | exact row-6 64x32 | 476.6485 / 315.9318 ms | 1.509x | reject; optimized latency above 290.6584 ms control |
| I3 | exact row-6 64x128 | 419.0229 / 2,238.6833 ms | 0.187x | reject |
| I4 | exact row-6 64x64, eight warps | 458.9541 / 301.7720 ms | 1.521x | reject; below threshold versus control |
| I5 | exact row-6 64x64, two warps | 465.3501 / 617.7467 ms | 0.753x | reject |

The five-variant cap was reached, so the existing 64x64 four-warp route remains.

## EXP-014: exact row-7 padded-width geometry

All five variants remained accuracy-green over 1,310,720 comparisons. I1 also
protected exact row 11 and reproduced row-7 correctness in two confirmations.
Long-sample results did not clear the approximately five-percent gate versus the
0.733536/0.720944 ms controls.

| Variant | Geometry change | Optimized median | Speedup within run | Decision |
| --- | --- | ---: | ---: | --- |
| I1 | 128x64 | 0.737984 ms | 1.363x | reject; 0.61% slower than matched control |
| I2 | 128x32 | 0.769872 ms | 1.348x | reject |
| I3 | 64x32 | 0.715296 ms | 1.446x | reject; sub-threshold and profile did not improve |
| I4 | 64x64, two warps | 0.935184 ms | 1.275x | reject |
| I5 | 64x64, eight warps | 0.714752 ms | 1.415x | reject; sub-threshold and profile did not improve |

The existing 64x64 four-warp policy remains for rows 7 and 11.

## EXP-016: exact row-11 new launch axes

Each candidate passed zero-failure correctness across 5,242,880 comparisons but
lost to the 0.996448 ms long control.

| Variant | Geometry change | Optimized median | Change versus control | Decision |
| --- | --- | ---: | ---: | --- |
| I1 | 128x64 | 1.060448 ms | +6.42% slower | reject |
| I2 | 64x64, two warps | 1.115920 ms | +11.99% slower | reject |
| I3 | 64x64, eight warps | 1.193408 ms | +19.77% slower | reject |

## EXP-015: exact row-8 projection work

I2 grouped Q/K together and left V separate. It passed correctness but measured
13.975408 / 14.132976 ms, or **0.988851x**, and was rejected.

The packed-QKV candidate produced three 300-sample rework runs at
**1.022022x**, **1.030071x**, and **1.023827x**. Two same-window unchanged
controls were **0.981690x** and **0.993542x**. Exact row 8 had zero failed
elements and zero maximum absolute error; the width-1024 held-out neighbor had
zero failures, maximum absolute error 0.000293195, and its expected Triton path.

Independent review rejected I1's first `d_model <= 1024` guard because it would
have enabled unmeasured widths 513-1023. I1R uses the surgical condition
`d_model <= 512 or d_model == 1024`. Tests exercise both the positive width-1024
path and a negative width-768 boundary. The integrated ten-forward profile
versus the same-window unchanged profile measured:

| Metric | Control | Integrated | Change |
| --- | ---: | ---: | ---: |
| `aten::addmm` calls | 240 | 160 | -33.33% |
| `aten::addmm` device time | 106,065.035 us | 94,048.315 us | -11.33% |
| optimized-model device time | 150,050.615 us | 138,182.163 us | -7.91% |

The trade-off is explicit: four packed width-1024 layers add 50,380,800 bytes
(about 48 MiB) to allocated memory before timing. The optimized incremental
activation peak is 369,115,136 bytes in both the unchanged control and accepted
candidate; the cache is persistent derived-weight storage outside that
incremental measurement.

The profile artifacts correctly report reference attention and therefore an
`INCONCLUSIVE` generic custom-kernel status. Their purpose is projection-event
comparison; they do not falsely claim that row 8 executes Triton attention.

## Integrated current-fingerprint gates

All artifacts below bind to implementation fingerprint
`54df021e77cfed86011bae0b41e07c3f42842b54e19c139aa925eb2f0d449ff4`.

| Gate | Accuracy | Performance / backend proof |
| --- | --- | --- |
| Final primary | 13/13 executable PASS + exact skip; 0/938,885,120 failed | 1.872916x; Triton 1,260 / SDPA 0 / reference 196 |
| Final confirmation | same correctness and backend totals | 1.863721x; 0.491% from primary |
| Organizer default | 5/5 PASS; 0/2,621,440 failed | 1.338x; 1,950 Triton calls |
| Held-out primary | 7/7 PASS; 0/13,117,440 failed | 1.365499x |
| Held-out confirmation | 7/7 PASS; 0/13,117,440 failed | 1.380821x |
| Source-derived | 28/28 executable PASS + exact skip; 0/459,776,000 failed | 1.244108x; required mixed routing |
| Focused regression suite | 94 passed; 14 upstream warnings | cache, boundary, dispatch, artifacts, and organizer contracts |
| Complete repository suite | 125/125 PASS; 14 upstream warnings | all CPU/GPU, notebook, artifact, documentation, and harness tests |

The integrated primary is 1.872% above the fresh Campaign 6 control. Campaign
5's 1.911947x/1.995117x full-matrix observations remain the highest historical
aggregates, but they are not contemporaneous causal controls. The accepted
decision rests on the targeted row-8 measurements and profiler reduction.

## Attempt accounting and retained failures

At closure Campaign 6 has **121 immutable attempt records: 118 command-level
PASS, 3 retained FAIL, and zero timeouts**. The 116 records through the final
125-test suite total **874.542363 seconds** of child-command wall time. The five
non-mutating workflow/graph/diff seal records each retain their own duration;
the exact closed aggregate is the sum of all raw `execution.wall_time_seconds`
values rather than a hand-edited benchmark field.

| Group | Attempts | Child PASS | Child FAIL | Wall time |
| --- | ---: | ---: | ---: | ---: |
| `C6-PREFLIGHT-*` | 5 | 4 | 1 | 4.914448 s |
| `C6-GRAPH-*` | 11 | 11 | 0 | 1.549910 s + three final seal durations |
| `C6-BASE-*` | 19 | 19 | 0 | 190.235670 s |
| `C6-EXP-*` | 58 | 58 | 0 | 414.170251 s |
| `C6-REVIEW-*` | 3 | 3 | 0 | 0.148020 s |
| `C6-INTEGRATE-*` | 16 | 15 | 1 | 229.535320 s |
| `C6-CLOSE-*` | 9 | 8 | 1 | 33.988744 s + workflow and final-diff seal durations |

The retained failures are evidence, not deleted noise:

1. `C6-PREFLIGHT-001-workflow` rejected invalid conditional PRD-impact labels;
   the corrected workflow passed.
2. `C6-INTEGRATE-016-full-tests-predoc` passed 118 tests and failed five stale
   Campaign 5 artifact-pointer assertions; implementation tests were green and
   the pointers were then migrated.
3. `C6-CLOSE-001-curated-evidence-pre-doc` passed 11 tests and failed one
   assertion that incorrectly treated a reference-attention profile as custom-
   kernel proof; the reworked semantic assertion and full doc contract passed.

## Current decision

**KEEP `EXP-015-I1R` and reject every Campaign 6 launch candidate.** The code is
small, boundary-exact, independently approved, state-dict compatible, and tied
to a reproducible target/profile benefit. No rejected candidate source remains
in the integration implementation.

The independent candidate review, final AI Council/release-gate review, and
their exact evidence conditions are under `reviews/`. The final seal is 125/125
repository tests, completed workflow validation, strict graph validation, and
diff hygiene, all recorded as the terminal attempts named by the final review.

The exact pause state and pre-closure metrics are preserved in
`CAMPAIGN-006-CHECKPOINT-2026-08-29.md`. No commit, push, tag, release, or public
action is authorized for the post-checkpoint closure changes.
