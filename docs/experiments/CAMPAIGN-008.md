# Campaign 8: exact-row residual-normalization extension

Status: complete and locally accepted on 2026-08-29; Git publication remains unauthorized

## Objective and frozen starting point

Continue from Campaign 7's locally accepted implementation without reopening
rejected attention widths or launch geometries. The starting checkpoint remains
`8c89d1d4170c58d16fb75d79f212e990565fba7d`; the uncommitted schema-2 source
fingerprint is
`a994eb1c0a5a7053335adbb1a4ab13dcde1f0ea247e5f9c422c017d8b297be8b`.
Campaign 7 closed at 13/13 executable PASS plus the exact authorized resource
skip, 1.880620x/1.927261x final geomeans, and 137/137 repository tests.

The fresh exact-row-11 profile records 10,082.392 us of model device time over
ten forwards. Native LayerNorm uses 1,577.287 us and residual adds 343.305 us,
a combined 19.05% ceiling. Row 11 already uses the accepted padded-width Triton
attention kernel in all four layers, so this campaign tests whether Campaign 7's
separate residual/normalization primitive can remove launches around that route.

## Bounded hypotheses

### EXP-019: exact row-11 fused residual plus LayerNorm

At most two implementations may be screened. I1 reuses the proven arithmetic
and launch policy under an exact runtime/model guard. I2 may tune only the fused
normalization launch for the smaller row count if I1 is correct but below the
performance threshold. The attention launch, packed-QKV boundary, comparator,
and existing exact-row-6 route remain frozen.

Required gates are direct arithmetic, exact row-11 state-dict/model accuracy,
multiple seeds/scales and both padding modes, runtime and mask boundaries,
attention and fusion backend counts, long timing, profiler events, incremental
peak allocation, row-6 non-regression, and broader matrices.

### EXP-020: exact row-7 fusion fallback

Only if EXP-019 closes without an accepted candidate may the same primitive be
tested on exact row 7. Its current residual-plus-native-norm share is 13.94% of
the ten-forward model profile. The existing one-reference/three-Triton attention
ordering is immutable. At most one implementation may be screened.

## Non-goals and stop rules

- Do not retry Campaign 7 `head_dim=256` attention or Campaign 6 row-6, row-7,
  or row-11 attention launch variants.
- Do not broaden fusion beyond exact measured rows, replace vendor GEMMs, change
  packed-QKV boundaries, enable broad compilation, or weaken correctness.
- Every command-level test, profile, benchmark, review, and closure gate uses
  `benchmarks/run_optimization_attempt.py`; failures and timeouts are retained.
- A retained candidate needs zero failed elements and approximately five-percent
  reproducible target/profile benefit or a material launch/memory improvement,
  with every regression and allocation cost reported.
- No commit, push, tag, release, branch creation, history rewrite, or public
  action is authorized.

## Attempt ledger

| Group | Purpose | Status |
| --- | --- | --- |
| `C8-PREFLIGHT-*` | workflow, fingerprint, contract, environment, candidate isolation | complete |
| `C8-BASE-*` | fresh final matrix, row-11/row-7 profiles, and long controls | complete |
| `C8-EXP-019-*` | exact-row-11 fused residual/normalization candidates | I1 superseded; I1R accepted |
| `C8-EXP-020-*` | conditional exact-row-7 fallback candidate | deliberately unrun; fallback condition false |
| `C8-REVIEW-*` | correctness, provenance, timing, memory, boundary, maintenance | approved |
| `C8-INTEGRATE-*` | accepted winner transplant and complete rebaseline | complete |
| `C8-CLOSE-*` | docs, tests, workflow, graph, Council, and tree closure | complete |

The immutable attempt JSON is authoritative for commands, wall time, accuracy,
latency samples, memory, backend/profiler counts, environment, fingerprints,
and dispositions.

## Candidate sequence and decision

I1 generalized the exact-row-6 fused forward to an exact row-6-or-row-11
predicate. Direct and route gates passed, two 300-sample target runs measured
0.894080/0.893904 ms, and unchanged controls measured 0.993920/0.990384 ms.
The first 10-step profile showed the correct 80 fused events but a contradictory
top-level model total. That result remains retained. A paired 30-step baseline
and candidate profile resolved the anomaly: model time fell 13.30% and the
residual/normalization subsystem fell 42.16%.

The AI Council boundary pass found that I1 relied on disabled gradients but did
not explicitly reject `model.train()` under inference mode. I1R added
`not self.training` and row-11-specific CPU, dtype, layout, runtime-shape,
noncontiguous-mask, gradient, training, and head-neighbor checks. The retained
implementation fingerprint is
`325a1e5cad70f85390ddbea438f04b60a4b0f40300826aba3991520f5b97079b`.

I1R passed ten boundary/direct tests, both 18-scenario row-6 and row-11 stress
matrices, and all 34 affected tests. The stress pair covered 2,967,994,368
outputs with zero failures and 288 expected fused calls. Two retained 300-sample
runs averaged 0.897184 ms and 4.697149x; three unchanged controls averaged
0.993525 ms and 4.262788x. That is a 9.6969% optimized-median reduction and
10.1896% normalized speedup gain. Candidate and controls all used a
29,360,128-byte optimized incremental peak.

The Council review in
`docs/experiments/reviews/CAMPAIGN-008-EXP-019-REVIEW.json` approved local
acceptance pending integration rebaseline. EXP-020 remained unrun because its
fallback-only condition was false once EXP-019-I1R cleared every gate.

## Integrated outcome

| Gate | Result |
| --- | --- |
| Final primary | 13/13 executable PASS + exact authorized skip; 0/938,885,120 failed; 1.876167x |
| Final confirmation | same correctness and backend counts; 1.911052x |
| Organizer default | 5/5 PASS; 0/2,621,440 failed; 1.351x; 1,950 Triton calls |
| Source-derived | 28/28 executable PASS + exact skip; 0/459,776,000 failed; 1.208961x |
| Held-out primary / confirmation | 7/7 PASS twice; 0/13,117,440 failed per run; 1.398943x / 1.401668x |
| Held-out stability | four complete matrices put long-causal at 1.194515x-1.198512x; 300-sample run 1.194853x |
| Row 6 long/profile | 187.837311 ms, 1.547529x over 100 samples; 80 fused calls; unchanged 2-reference/2-Triton route and peak allocation |
| Row 11 long/profile | 0.896928 ms, 4.651860x over 300 samples; 240 fused and 120 Triton-attention calls; unchanged peak allocation |
| Row 11 profile delta | model 41,211.814 -> 32,159.723 us (-21.96%); residual/norm 5,978.920 -> 3,211.676 us (-46.28%) |
| Complete repository suite | 139/139 PASS; 14 upstream PyTorch deprecation warnings |

## Retained failures and reworks

- `C8-OPS-001-wrapper-argument-error` retains the rejected `--id` wrapper typo;
  the intended control was rerun with `--attempt-id`.
- `C8-INTEGRATE-012-pre-doc-full-suite` recorded 132 passes and seven expected
  stale Campaign 7 artifact-pointer failures, with no implementation failure.
- `C8-INTEGRATE-017-artifact-contract-pre-doc` recorded 14 passes and one stale
  documentation-selection failure before reconciliation.
- `C8-INTEGRATE-018` and `019` retain two exact documentation-string failures:
  first the disclosure was absent, then a Markdown line break split the literal
  invariant. `C8-INTEGRATE-020` is the authoritative 15/15 green contract.
- `C8-INTEGRATE-021-full-suite` recorded 138 passes and one stale Colab test
  fingerprint. `C8-INTEGRATE-022` repaired the exact pin, and `023` passed
  139/139.
- The first `C8-GRAPH-005` command used the unsupported decision label
  `retain` and was rejected before the graph payload ran. The failure was
  reproduced and retained as `C8-GRAPH-005A-invalid-decision`; the corrected
  `keep` invocation rebuilt the graph successfully.

Through `C8-CLOSE-007B`, Campaign 8 has 65 immutable attempts: 58 PASS, seven
FAIL, zero timeouts, and 530.881889 seconds of measured child-command wall time.
The terminal workflow and diff seals follow that explicit accounting checkpoint
as separate immutable records rather than being predicted into the subtotal.
