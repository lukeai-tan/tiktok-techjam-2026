# EXP-003: Short head-dimension-32 K/V tiles

## Experiment identity

- Status: keep - independently approved, integrated, and rebaselined
- Campaign: `docs/experiments/CAMPAIGN-002.md`
- Screening baseline commit: `b90069a`
- Winning isolated implementation commit: `c6a0d99`
- Integrated implementation commit: `085d6bc`
- Integrated implementation fingerprint:
  `8eb7d21551ab69e83f532deaeefb2ce1999dc3e198f48a8d4be5753ad2c93a8a`
- Target: NVIDIA GeForce RTX 5070 Ti, PyTorch 2.13.0+cu130,
  Triton 3.7.1.post27, Python 3.12.10

## Hypothesis and scope

The prior short `head_dim=32` launch used `BLOCK_M=64`, `BLOCK_N=128`, four
warps, and two stages. The row-1 profile attributed 7,008.677 us across 40
launches to `_attention_fwd`, leaving enough end-to-end ceiling to test smaller
tiles without changing arithmetic.

The accepted rule changes only `head_dim == 32 and seq_len <= 128` to a 64x64
tile with four warps and two stages. Head dimension 16 retains 64x128;
head dimension 64 retains EXP-001's 32x64 policy; longer sequences retain their
existing policy. The organizer sources, comparator, tolerances, dispatch rules,
timing policy, model API, state dict, and persisted formats are unchanged.

## Bounded alternatives

| Variant | Tile | Row-1 optimized median | Correctness | Decision |
| --- | --- | ---: | --- | --- |
| unchanged | 64x128 | 1.2358 ms | 0/5,242,880 failed | control |
| I1 | 32x64 | 0.8579 ms | 0/5,242,880 failed | superseded |
| I2 | 32x128 | 0.8805 ms | 0/5,242,880 failed | reject |
| I3 | 64x64 | 0.8164 ms | 0/5,242,880 failed | keep |

Two alternating confirmations compared the unchanged policy, I1, and I3:

| Policy | Samples | Mean | Median | Sample SD |
| --- | --- | ---: | ---: | ---: |
| unchanged 64x128 | 1.2358 / 1.2425 / 1.2424 ms | 1.2402 ms | 1.2424 ms | 0.0038 ms |
| I1 32x64 | 0.8579 / 0.8660 / 0.8565 ms | 0.8601 ms | 0.8579 ms | 0.0051 ms |
| I3 64x64 | 0.8164 / 0.8203 / 0.8235 ms | 0.8201 ms | 0.8203 ms | 0.0036 ms |

Every run used five accuracy trials, compared 5,242,880 elements with zero
failures, and recorded Triton 112 / SDPA 0 / reference 0. I3 reduced median
latency 33.98% versus the unchanged policy and 4.38% versus I1.

## Independent review

The independent reviewer approved integration after verifying:

- the exact `head_dim == 32 and seq_len <= 128` guard and boundary tests;
- no protected harness, manifest, matrix, comparator, or source changes;
- matching attempt-to-result SHA-256 links and implementation fingerprints;
- three zero-failure candidate measurements with 0.87% candidate median drift;
  and
- affected final rows 1-6 and 12 plus organizer-default were explicitly routed
  into mandatory post-integration validation.

## Integrated validation

- Final shapes: 13/13 executable PASS, one authorized resource skip,
  0/938,885,120 failed elements, and 1.525823x geomean. This is 6.948% above
  the post-EXP-001 1.426692x artifact.
- Directly affected executable final rows 1-5 and 12 all improved optimized
  median latency by 15.05% to 42.70%. Row 6 uses exact reference at its very
  large batch and therefore does not select the custom kernel.
- Largest unaffected final optimized-latency regression: 1.59%, inside the 2%
  gate.
- Untouched organizer default: 5/5 PASS, 0/2,621,440 failed, 1,950/1,950 Triton
  launches, and optimized median 1.3374 ms versus 1.3533 ms previously.
- Held-out matrix: 7/7 PASS, 0/13,117,440 failed, 1.228277x geomean, and every
  optimized median lower than the prior artifact.
- Source-derived matrix: 28/28 executable PASS plus one authorized skip and
  0/459,776,000 failed. Reported geomean moved from 1.232813x to 1.193898x as
  the fresh baselines moved more, while every optimized median improved by
  8.98% to 36.97%; this timing-noise caveat is retained rather than waived
  silently.
- Profiler: row-1 `_attention_fwd` fell from 7,008.677 us to 2,103.978 us
  across 40 launches (69.98%). Row 10 and held-out long-causal-padding retained
  exact Triton launch counts and improved in the fresh profiles.
- Complete repository suite: 102 passed with 14 upstream PyTorch deprecation
  warnings.

## Logging and failed gates

All attempt JSON records contain exact portable argv, UTC timestamps, child
wall time, return code, stdout/stderr, Git/fingerprint state, environment,
artifact path and SHA-256, correctness/error counts, latency distributions,
backend counts, memory/profiler data when available, and an explicit decision.

Three failures remain first-class evidence:

1. The first post-merge suite correctly failed four stale-fingerprint artifact
   assertions before rebaseline (97 passed, 4 failed).
2. The post-rebaseline suite found one stale held-out metric assertion and a
   Windows cp1252 live-tee error.
3. The first logger-hardening regression exposed a test-only LF/CRLF assumption
   (16 passed, 1 failed).

The tee now replaces characters unsupported by the active console encoding
while preserving original Unicode in JSON. The corrected focused gate passed
17/17 before the 102-test complete gate.

## Evidence index

- Full ledger and measured wall times: `docs/experiments/CAMPAIGN-002.md`
- Attempt records: `docs/experiments/attempts/C2-EXP-003-*.json` and
  `docs/experiments/attempts/C2-LOG-HARDENING-*.json`
- Screening/confirmation results:
  `docs/results/rtx-5070-ti-2026-08-28-c2-exp003-*.json`
- Curated final matrix:
  `docs/results/rtx-5070-ti-2026-08-28-final-evaluator-baseline.json`
- Curated default, held-out, source-derived, and profiler artifacts:
  `docs/results/README.md`

## Decision

Keep the exact 64x64 short-`head_dim=32` launch policy. Rollback is the single
guarded branch in `transformer_opt/config.py` plus its dispatch assertions.
Do not generalize the policy to smaller head dimensions or longer sequences
without a new isolated, logged campaign.
