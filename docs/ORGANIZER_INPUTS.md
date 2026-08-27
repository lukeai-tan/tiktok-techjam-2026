# Track 3 Organizer Inputs

Audit date: 2026-08-27 (Asia/Singapore)

## Received and frozen

The following participant-supplied downloads came from the ByteDance Lark
workspace and are preserved byte-for-byte under `benchmarks/`:

| Resource | SHA-256 | Use |
| --- | --- | --- |
| `benchmarks/torch_transformer_benchmark.py` | `1bd12523657f338c09b53f0bb9052d9d16f728a71bd22bc8298567e1a4d78c22` | Selected organizer contract and untouched test harness. |
| `benchmarks/tensorflow_transformer_benchmark.py` | `00e99b6e1d19e961039b66eb3d3c055b36cc50f0436da2558f5f1fbe292ef798` | Alternative-framework contract and dimension-scope cross-check. |

The signed Lark transport query strings are deliberately not stored. Sanitized
origin evidence and exact sizes/checksums are in
`benchmarks/reference/organizer_downloads.json`.

## How the two files are used

PyTorch is the selected framework, as permitted by the brief. The root
`torch_transformer_benchmark.py` keeps the organizer's baseline Transformer,
weight-copy contract, and correctness comparator, and replaces the documented
`UserOptimizedTransformer` extension point. The protected definitions are
AST-compared in tests.

For the strongest end-to-end proof, `benchmarks/run_organizer_torch.py` loads
the untouched download, injects the submitted class at that extension point,
and delegates argument parsing, accuracy checks, and timing to the organizer's
code:

```powershell
& $python benchmarks/run_organizer_torch.py --device cuda
```

The TensorFlow file is not a second implementation requirement. It is retained
untouched as the single canonical copy under `benchmarks/` and used to audit
the organizer's broader shape signals; no root TensorFlow submission copy is
needed because PyTorch is the selected framework. Its default
one-factor-at-a-time axes are:

- batch sizes: 1, 4, 16, 128, 10000;
- QKV/model widths: 32, 128, 1024;
- head counts: 1, 2, 4, 16; and
- sequence lengths: 32, 1024, 100000.

Its 100000-token stress case is explicitly allowed to be preflight-skipped when
the quadratic baseline exceeds available memory. These TensorFlow defaults are
not silently relabeled as the final PyTorch evaluation matrix because the two
provided scripts differ in framework, defaults, tolerance, dtype, and case
generation.

`benchmarks/run_organizer_validation.py` nevertheless turns every published
dimension signal into auditable coverage for the selected implementation. It
runs six PyTorch-contract variants, then each of the 11 feasible TensorFlow
compact shapes in both PyTorch float32 and the TensorFlow-default float16. Each
case executes in a fresh process through the untouched PyTorch parser,
comparator, and timer. On the recorded RTX 5070 Ti, all 28 executable cases
passed five trials each with zero failures across 459,776,000 elements. The
single 100000-token stress entry is recorded as `SKIPPED_RESOURCE`, explicitly
authorized by the source and excluded from the pass count. Reproduce with:

```powershell
& $python benchmarks/run_organizer_validation.py `
  --out results/organizer-validation.json
```

The versioned policy is `benchmarks/organizer_validation_matrix.json`; the raw
evidence is
`docs/results/rtx-5070-ti-2026-08-27-organizer-validation.json`.

## Still needed from the organizer

One material input remains outstanding:

1. **The final evaluator matrix or test harness for the selected PyTorch path.**
   The brief promises that all shape combinations will be provided, but the
   supplied PyTorch script exposes a configurable single case rather than a
   frozen matrix. The TensorFlow defaults cannot be assumed to be that matrix.

The following clarifications should accompany that file if they are not encoded
in it:

- required dtypes, padding ratios, and causal/non-causal cases;
- whether backward/gradient execution is evaluated;
- whether compilation time, model construction, and first-run autotuning count
  toward timing; and
- whether evaluator rules differ from the supplied scripts after the workshop.

No dataset, pretrained weights, tokenizer, or external model asset is needed:
both supplied scripts generate model weights and inputs locally. No TensorFlow
installation is needed for the selected PyTorch submission.
