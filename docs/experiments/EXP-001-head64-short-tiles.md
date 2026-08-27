# EXP-001: Short head-dimension-64 tiles

## Experiment identity

- Status: keep — independently approved for integration and mandatory rebaseline
- Parent commit: `8656e923114938eb122d78d2fcf45e42d34496f9`
- Candidate implementation commit: `07dbc5d162b47fdd9cdcda3240c22c6c34cf40e7`
- Branch: `exp/head64-short-tiles`
- Target: NVIDIA GeForce RTX 5070 Ti, PyTorch 2.13.0+cu130,
  Triton 3.7.1.post27, Python 3.12.10

## Hypothesis

The existing `BLOCK_M=64`, `BLOCK_N=128` launch spills heavily for causal
IEEE-fp32 attention at `head_dim=64`, making final evaluator row 10 slower than
the explicit baseline. Use `BLOCK_M=32`, `BLOCK_N=64`, four warps, and two
stages only when `head_dim == 64` and `seq_len <= 128`; preserve every other
launch configuration.

Compiled target metadata reported 2,468 spills and 81,920 bytes of shared
memory for the prior tile versus two spills and 49,152 bytes for the candidate.
The rollback is the single launch-policy branch in `transformer_opt/config.py`.

## Changed paths

- `transformer_opt/config.py`: bounded launch-policy branch.
- `tests/test_dispatch.py`: selected and neighboring-shape policy checks.
- `docs/KERNEL_DESIGN.md`: updated launch table and rationale.

The organizer sources, comparator, tolerances, timing code, model API, state
dict, and persisted formats are unchanged.

## Correctness and dispatch

- Final organizer shapes: 13/13 executable `PASS`, zero failed elements across
  938,885,120 comparisons; row 14 remains the one authorized resource skip and
  is excluded from the pass count.
- Source-derived matrix: 28/28 executable `PASS`, zero failed elements across
  459,776,000 comparisons; one authorized skip.
- Seven-case held-out matrix: 7/7 `PASS`, five trials per case.
- Untouched organizer default: 5/5 trials `PASS`, zero failures across
  2,621,440 comparisons.
- Candidate fast/full non-artifact tests: 84 passed; the curated-artifact tests
  intentionally remain stale on the isolated branch until integration and a
  clean rebaseline.
- Final row 10 continued to report `triton=112`, `sdpa=0`, `reference=0`.

## Paired performance

The complete final matrix was run in alternating clean worktrees. Each case
used the frozen final policy: five accuracy trials, warmup 3, repeats 10, and
two alternating timing rounds.

| Pair | Baseline geomean | Candidate geomean | Relative gain | Baseline row 10 | Candidate row 10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.303394x | 1.420410x | 8.98% | 0.416x / 3.5660 ms | 1.469x / 0.9703 ms |
| 2 | 1.309126x | 1.442585x | 10.19% | 0.389x / 3.7258 ms | 1.392x / 0.9717 ms |

Unchanged cases showed substantial run-to-run noise, but the only affected final
row improved in both pairs. On the other affected protected shape, the untouched
organizer default, optimized latency was 1.5158 ms before and 1.5207 ms after
(0.32% slower, inside the 2% gate). The affected held-out wide-model row
improved from the committed 0.982x baseline to 1.114x in the candidate run.

## Profiler evidence

For ten final-row-10 forwards (40 attention launches), `_attention_fwd` fell
from 30,324.486 us to 3,210.441 us, an 89.41% reduction. Backend accounting
remained exactly 40 Triton calls in both profiles.

## Evidence artifacts

- `docs/results/rtx-5070-ti-2026-08-28-exp001-paired-baseline-1.json`
- `docs/results/rtx-5070-ti-2026-08-28-exp001-head64-short-tiles.json`
- `docs/results/rtx-5070-ti-2026-08-28-exp001-paired-baseline-2.json`
- `docs/results/rtx-5070-ti-2026-08-28-exp001-paired-candidate-2.json`
- `docs/results/rtx-5070-ti-2026-08-28-exp001-final-10-profile.json`
- `docs/results/rtx-5070-ti-2026-08-28-exp001-held-out.json`
- `docs/results/rtx-5070-ti-2026-08-28-exp001-source-derived.json`
- `docs/results/rtx-5070-ti-2026-08-28-exp001-default-baseline-1.json`
- `docs/results/rtx-5070-ti-2026-08-28-exp001-organizer-default.json`

The organizer runner records timing distribution summaries rather than raw
samples; the held-out runner artifact retains its raw CUDA-event samples.

## Alternatives rejected

- Reference routing passed correctness but projected only about 4.7% final
  geomean improvement and left row 10 below baseline performance.
- SDPA routing had one failed element in five exact row-10 trials.
- Residual-add/LayerNorm fusion could not address attention's measured 79.6%
  share of row-10 GPU time and contradicted earlier slow standalone LayerNorm
  evidence.

## Decision gate

Integrate only if independent review confirms evidence provenance, the bounded
support envelope, greater-than-5% paired final-matrix improvement, no affected
required-case regression over 2%, and truthful custom-kernel execution. After
integration, regenerate every implementation-fingerprinted curated artifact and
require the complete test suite to pass.

## Independent review decision

The independent release gatekeeper approved the candidate implementation for
integration. The paired gains, bounded dispatch rule, correctness matrices,
backend accounting, and profiler causality met the gate. Final rows 2 and 5 had
aggregate timing regressions above 2%, but both are `head_dim=32`, cannot select
the new rule, and moved in conflicting directions between paired trials; the
reviewer therefore recorded an explicit measurement-noise waiver. No affected
`head_dim=64` case had a material regression. Release-final evidence remains
conditional on a clean post-merge rebaseline and full-suite pass.
