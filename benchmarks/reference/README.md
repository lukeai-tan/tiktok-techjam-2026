# Reference benchmark provenance

`manifest.json` fingerprints the exact benchmark blob used to freeze the
current implementation contract. It deliberately does not call the file a
pristine organizer download because that provenance is not available in the
repository history.

Verify the saved SHA-256 from PowerShell without sending Git's binary blob
through PowerShell's text pipeline:

```powershell
python -c "import hashlib, subprocess; blob = subprocess.check_output(['git', 'cat-file', 'blob', '5083fb4ebf6f83acd2b89a0cf62ae067f927dbb4']); print(hashlib.sha256(blob).hexdigest())"
```

When the organizer publishes a newer benchmark, store its source URL,
retrieval time, checksum, and a semantic diff before changing optimized code.
