# Reference benchmark provenance

There are now two provenance layers:

- `organizer_downloads.json` fingerprints the untouched PyTorch and TensorFlow
  files supplied from the ByteDance Lark workspace on 2026-08-27. The signed
  transport URLs are intentionally not persisted.
- `manifest.json` preserves the older Git snapshot used by the existing
  revision-linked RTX 5070 Ti result artifacts. It remains immutable so those
  results retain their original implementation fingerprint.

PyTorch is the selected submission framework. The root
`torch_transformer_benchmark.py` is its optimized copy, while
`benchmarks/run_organizer_torch.py` injects the same submitted class into the
untouched organizer harness. Automated tests compare protected baseline
definitions and strict state-dict compatibility. The TensorFlow download is
retained as an alternative-framework and shape-scope cross-check; Track 3 says
only one framework implementation is required.

`benchmarks/organizer_validation_matrix.json` records how the two contracts are
combined without altering either download. `benchmarks/run_organizer_validation.py`
executes every feasible translated case in an isolated process and permits only
the TensorFlow source's exact designated 100000-token resource skip.

Verify the saved SHA-256 from PowerShell without sending Git's binary blob
through PowerShell's text pipeline:

```powershell
python -c "import hashlib, subprocess; blob = subprocess.check_output(['git', 'cat-file', 'blob', '5083fb4ebf6f83acd2b89a0cf62ae067f927dbb4']); print(hashlib.sha256(blob).hexdigest())"
```

Verify both supplied downloads:

```powershell
$python = ".venv\Scripts\python.exe"
& $python -m pytest tests/test_organizer_benchmarks.py -q
```

When the organizer publishes a newer benchmark, store its retrieval time,
checksum, and a semantic diff before changing optimized code. Do not commit a
temporary signed download URL or Windows `Zone.Identifier` stream.
