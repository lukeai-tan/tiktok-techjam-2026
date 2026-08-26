# Transformer GPU Kernel Release Gate

Date: 2026-08-27

## Verdict

**PASS for the repository's current executable contract.** The implementation,
tests, target-GPU matrix, profiler proof, and reproduction documentation are
complete and mutually consistent.

**HOLD for a final organizer-submission claim.** The organizer's final shape
matrix and any superseding benchmark have not been published in this
repository, and the public demo video still requires a human recording/upload.
Those are external submission tasks, not missing kernel implementation work.

## Review record

This was a role-based local council review; it was not represented as an
independent human or multi-agent audit.

### Correctness and numerical review: PASS

- The repository-owned `_attention_fwd` Triton kernel fuses QK, online softmax,
  causal/key masking, and P@V without a dense `[B,H,S,S]` intermediate.
- Forced custom dispatch fails clearly outside the declared envelope; automatic
  dispatch records the backend actually used.
- Float32 follows PyTorch's TF32 setting, direct float16 kernel behavior is
  tested, and correctness-sensitive deep float16/bfloat16 model execution uses
  the explicit reference path.
- The full local suite passed: 59 tests, including tile boundaries, causal and
  padding masks, all-masked robustness, TF32-disabled math, unsupported-input
  fallback, state-dict compatibility, fail-closed result accounting, and
  fingerprint-linked artifact checks.

### Performance and evidence review: PASS for provisional matrix

- Target: NVIDIA GeForce RTX 5070 Ti under WSL2, PyTorch 2.13.0+cu130,
  Triton 3.7.1, CUDA runtime 13.0.
- Matrix: 7 requested / 7 completed / 7 PASS; 0 FAIL, OOM, or ERROR.
- Accuracy: 35 trials, 13,117,440 output elements, zero failures; maximum
  absolute error 0.000997663 under the executable `atol=0.001 OR rtol=0.01`
  rule.
- Timing: 90 raw CUDA-event samples per model/case after warm-up, alternating
  baseline/optimized order; median end-to-end speedup 1.138x-1.566x and 1.360x
  geometric mean.
- Profiler: `_attention_fwd` appeared 10 times for five two-layer forwards;
  dispatch counts were Triton 10, SDPA 0, reference 0.
- Matrix and profile share implementation fingerprint
  `a36abc1d440e7d5318348854a673f832c5d9ae649e295ffeae01cd599d478eb5`.
- The inherited standalone Triton LayerNorm was removed after target-device
  measurements showed it was slower than native PyTorch LayerNorm.

### Release, documentation, and safety review: PASS with external holds

- Requirements, project context, kernel design, technical report, README,
  Devpost copy, demo runbook, benchmark provenance, and result provenance agree.
- The matrix runner is fail-closed for exceptions, OOM-only runs, and empty
  selections. Curated artifacts are rejected by tests when their implementation
  fingerprint is stale.
- The local workflow syntax, JSON, compile, and test checks pass. The GitHub
  Actions workflow is present but has not been claimed as remotely executed.
- No new runtime secrets or external mutation are required.

## External actions before submission

1. Reconcile any final organizer benchmark, shape list, dtype, timing, backward,
   and source-modification rules with `docs/REQUIREMENTS.md` and the reference
   manifest.
2. Rerun the complete matrix and profiler if any implementation-fingerprinted
   path changes.
3. Record and upload the public demo using `DEMO_RUNBOOK.md`, then add its public
   URL to Devpost.
4. Commit and push only when the repository owner requests those Git actions.
