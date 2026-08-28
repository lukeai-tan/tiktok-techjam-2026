# Campaign 6: exact-shape launch and projection optimization

Status: checkpointed on 2026-08-29; final documentation and closure pending

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

## Planned attempt ledger

The immutable attempt files are authoritative. This table is populated from
those records at closure rather than hand-entered during measurement.

| Group | Purpose | Status |
| --- | --- | --- |
| `C6-PREFLIGHT-*` | workflow, graph, hashes, environment, baseline gates | pending |
| `C6-BASE-*` | fresh complete matrix and target profiles | pending |
| `C6-EXP-013-*` | row-6 launch candidates and confirmations | pending |
| `C6-EXP-014-*` | row-7/row-11 padded-width launch candidates | pending |
| `C6-EXP-015-*` | row-8 wide-model projection candidates | pending |
| `C6-EXP-016-*` | row-11 untried query-tile and warp candidates | pending |
| `C6-INTEGRATE-*` | composite rebaseline and complete validation | pending |
| `C6-CLOSE-*` | documentation, independent review, workflow, graph, and tree closure | pending |

## Current decision

`EXP-015-I1R` is independently approved and integrated at implementation
fingerprint
`54df021e77cfed86011bae0b41e07c3f42842b54e19c139aa925eb2f0d449ff4`.
The other three candidate surfaces are closed as measured plateaus. Complete
current-fingerprint benchmark gates pass, while final documentation curation,
result-pointer tests, workflow closure, and graph closure remain pending.

The exact pause state, metrics, retained failures, and resume sequence are in
`docs/experiments/CAMPAIGN-006-CHECKPOINT-2026-08-29.md`. No commit, push, tag,
release, or public action has been performed or authorized by this campaign.
