# Campaign 7: wide-head attention and residual-normalization feasibility

Status: complete on 2026-08-29

## Objective and frozen starting point

Continue optimization after Campaign 6 without repeating its closed row-6,
row-7, or row-11 launch axes. The implementation starts from checkpoint
`8c89d1d4170c58d16fb75d79f212e990565fba7d` on `feat/jared-attempt`, with
schema-2 implementation fingerprint
`54df021e77cfed86011bae0b41e07c3f42842b54e19c139aa925eb2f0d449ff4`.
Campaign 6 closure documentation and attempts are intentionally uncommitted;
candidate source work must therefore use a detached isolated worktree from the
checkpoint and must not overwrite the active working tree.

The current complete final pair is 13/13 executable PASS plus the exact
authorized resource skip at 1.872916x and 1.863721x. Exact row 8 remains the
near-parity case at 1.052x primary speedup. Its ten-forward profile records
138,182.163 us of model device time: 94,048.315 us in `aten::addmm`, 13,466.867
us in copies, 9,273.619 us in reference-attention BMMs, 8,522.438 us in residual
adds, and 6,358.582 us in native LayerNorm. Campaign 6 already optimized the QKV
projection, so this campaign first tests the still-untried attention width.

## Bounded hypotheses

### EXP-017: exact row-8 `head_dim=256` Triton attention

The repository kernel's arithmetic naturally supports a compile-time dot width
of 256, but the declared support set stops at 128. A repository-owned fused
route could remove reference attention's BHSD copies, score tensor, causal mask,
softmax, and two BMMs. The route is eligible only for exact organizer row 8 in
model `auto`; forced direct-kernel testing may expose width 256 for controlled
validation. The launch loop is capped at four previously untried small-tile
geometries, beginning with 16x16 to limit accumulator/register pressure.

Required gates are successful compilation, zero failed elements in direct and
four-layer exact-row tests, exact backend counts, row-8 memory and latency,
multiple seeds/scales/padding boundaries, and a reference-attention neighbor.
Compilation failure, any failed element, or three non-improving valid variants
closes the route.

### EXP-018: fused residual plus LayerNorm on exact row 6

This is conditional on a fresh Campaign 7 profile confirming that residual adds
and native LayerNorm still provide at least a ten-percent combined end-to-end
ceiling after excluding duplicated profiler events. At most two implementations
may be screened. The candidate must preserve PyTorch LayerNorm epsilon,
parameter/bias semantics, invalid-row zeroing, strict state-dict compatibility,
and exact fallback outside a measured shape. The previously rejected standalone
Triton LayerNorm is not retried.

## Non-goals and stop rules

- Do not retry Campaign 6 row-6, row-7, or row-11 attention launch variants.
- Do not broaden packed QKV beyond `d_model <= 512` and exact 1024.
- Do not replace vendor output/FFN GEMMs, enable broad `torch.compile`, weaken
  the comparator, change organizer inputs, or reinterpret the resource skip.
- Every command-level test, profile, benchmark, review, and closure gate must use
  `benchmarks/run_optimization_attempt.py` and retain failures/timeouts.
- A retained candidate needs zero failed elements and approximately five-percent
  reproducible target/model/profile improvement or a material memory/launch
  benefit, with every cost reported.
- No commit, push, tag, release, branch creation, or public action is authorized.

## Attempt ledger

| Group | Purpose | Status |
| --- | --- | --- |
| `C7-PREFLIGHT-*` | workflow, fingerprint, hashes, environment, detached worktrees | pass |
| `C7-BASE-*` | fresh final matrix and exact row-6/row-8 profiles/controls | pass |
| `C7-EXP-017-*` | head-dimension-256 compile, correctness, geometry, timing, memory | reject |
| `C7-EXP-018-*` | conditional fused residual-normalization candidates | keep I2R |
| `C7-REVIEW-*` | candidate provenance, boundaries, trade-offs, maintainability | approved for local integration |
| `C7-INTEGRATE-*` | accepted winner transplant and full rebaseline | pass |
| `C7-CLOSE-*` | documentation, tests, workflow, graph, and tree closure | pass after two validator reworks |

The immutable attempt JSON is authoritative for exact commands, runtime,
accuracy, latency, memory, backend counts, profiler events, and dispositions.

## EXP-017 outcome: rejected wide-head attention

- 16x16 and 16x32 direct kernels compiled and passed primitive arithmetic, but
  each four-layer row-8 screen failed 2/41,943,040 elements with maximum absolute
  error 0.00109979.
- 16x64 did not compile: 151,616 bytes of shared memory were required against a
  101,376-byte device limit.
- Keeping the first layer exact repaired accuracy and reduced copies from 200 to
  50 plus BMMs from 80 to 20 over the profile window, but 30 wide Triton kernels
  cost 29,166.344 us. Ten-forward model device time regressed from 130,180.675
  us to 141,240.672 us (+8.50%).
- No EXP-017 source was integrated.

## EXP-018 outcome: accepted exact-row-6 fusion

The initial fused residual/LayerNorm implementation passed arithmetic and model
accuracy but extended intermediate lifetimes, adding 1,967,128,576 peak bytes.
I2 released normalized/attention/FFN temporaries; I2R narrowed the exact runtime
shape and mask boundary. The final route preserves optional bias, epsilon,
state-dict compatibility, padded-row zeroing, and fallbacks for noncontiguous
masks, neighboring shapes, compiled execution, gradients, CPU, and other dtypes.

- Direct and boundary gate: 10/10 PASS.
- Stress: 18 seed/scale/padding scenarios, 2,949,120,000 compared outputs, zero
  failures, and all 144 expected fused calls.
- Candidate screen: 0/819,200,000 failed; maximum absolute error 0.000944704.
- Candidate profile: model device time 2,026,089.666 us to 1,799,039.344 us
  (-11.21%); residual-plus-normalization subsystem 486,023.333 us to 302,200.048
  us (-37.82%).
- Counterbalanced 100-sample bracket: candidates averaged 1.554314x, unchanged
  controls 1.419031x, a +9.53% normalized improvement; candidate optimized
  latency averaged 181.707161 ms versus 197.826908 ms (-8.15%).
- Integrated profile: 80 residual adds and 80 of 90 native norms become 80 fused
  launches over ten forwards; subsystem time -36.30%, model time -9.54%.
- Integrated 100-sample run: 293.910400 ms baseline, 189.981712 ms optimized,
  1.547046x; optimized incremental peak is unchanged at 11,802,787,840 bytes.

## Integrated evidence

The selected schema-2 implementation fingerprint is
`a994eb1c0a5a7053335adbb1a4ab13dcde1f0ea247e5f9c422c017d8b297be8b`.

| Gate | Result |
| --- | --- |
| Final primary / confirmation | 13/13 executable PASS plus exact skip twice; 0/938,885,120 failed each; 1.880620x / 1.927261x |
| Organizer default | 5/5 PASS; 0/2,621,440 failed; 1.358x; 1,950 Triton attention calls |
| Held-out primary / confirmation | 7/7 PASS twice; zero failed; 1.380355x / 1.377674x |
| Source-derived | 28/28 executable PASS plus exact skip; 0/459,776,000 failed; 1.202688x |
| Profiles | row 6, row 7, row 8, row 11, long causal, and long causal with padding all preserve expected routes |
| Complete suite | 137/137 PASS; 14 upstream PyTorch deprecation warnings |
| Review | `CAMPAIGN-007-CANDIDATE-REVIEW.json`: approve local acceptance pending integration rebaseline; rebaseline passed |

## Decision

**KEEP EXP-018-I2R and reject EXP-017.** The accepted fusion has direct,
boundary, stress, profile, long-run, memory, full-matrix, organizer-default,
held-out, and source-derived proof. The guard remains exact because Campaign 7
did not establish a benefit or numerical contract on adjacent shapes. No commit,
push, tag, release, branch creation, or public action is implied by this local
acceptance.

## Attempt and runtime accounting

Campaign 7 closes with **75 immutable attempt records: 70 child-process PASS,
5 retained child-process FAIL, and 0 timeouts**. The 73 records through the final
137-test suite total **1,022.059239 seconds** of child-command wall time; the two
terminal workflow and diff seal durations remain exact in their own records.
There are 37 Campaign 7 result JSON files.

The five nonzero commands are the two strict EXP-017 model misses, its shared-
memory compile failure, the expected five-stale-pointer candidate suite, and the
first fail-closed council-validator rework. `C7-CLOSE-003` is separately
disclosed as a semantic false green: its child exited zero after PowerShell
expanded away the inner variable and emitted non-terminating errors. The next
attempt failed closed on a different quoting defect; `C7-CLOSE-005` protected
the inner program and proved the review fields. Neither misleading attempt was
deleted or treated as substantive review evidence.
