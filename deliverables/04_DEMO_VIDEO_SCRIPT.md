# SpeedROCm demo video script

## Recording goal

Create a concise **3–5 minute** public YouTube walkthrough showing the
solution working end-to-end: the baseline problem, the optimized kernel flow,
live or recorded benchmark evidence, profiler proof, and the correctness-first
fallback policy.

The final video must be uploaded to YouTube with public visibility and linked in
Devpost. This file is a recording script, not proof that the upload has happened.

## Name, pronunciation, and platform disclosure

- Display the project name as **SpeedROCm** and pronounce it “Speed Rock-em.”
- State in the opening that the current prototype was measured on NVIDIA CUDA,
  not AMD ROCm. The name is branding, not a runtime-compatibility claim.
- Before public upload, ask the organizer to confirm that the SpeedROCm title
  itself is acceptable under the event's trademark rule. This script does not
  constitute legal or organizer approval.
- Use a plain-text title card. Do not use AMD, ROCm, NVIDIA, CUDA, PyTorch,
  Triton, TikTok, or other third-party logos or brand artwork unless the team
  has confirmed permission. Necessary product names may be spoken or shown as
  plain text when describing the actual environment.

## On-screen terminology legend

Use these meanings consistently in captions, diagrams, and narration:

| Term or symbol | Say or explain it as |
| --- | --- |
| Baseline | The supplied reference Transformer. |
| Optimized | The SpeedROCm implementation being compared with the baseline. |
| Q, K, V / QKV | Query, key, and value tensors. |
| P@V | Attention probabilities multiplied by value vectors. |
| `[B, H, S, S]` | Batch by heads by sequence by sequence; the two sequence axes make this storage grow quadratically. |
| fp32 | 32-bit floating point. |
| SDPA | Scaled dot-product attention, a PyTorch backend. |
| `×` | “times faster” when attached to a speedup result. |
| `ms` | Milliseconds. |
| `MiB` | Mebibytes; one MiB is 1,048,576 bytes. |
| Geometric mean | A multiplicative average of per-input speedups. |
| Resource skip | A permitted case that was not run because of resource requirements; it is not a pass. |

## Claims to use

Use the output produced during the recording if it differs from the curated
artifact. The current evidence-backed claims are:

- 13/13 executable published final rows passed, with zero failed elements across
  938,885,120 comparisons;
- primary final-matrix geometric-mean speedup 1.977× and complete confirmation
  1.986×;
- untouched organizer default: 5/5 accuracy passes and 1.385× median speedup;
- row 11 final matrix: 5.7496 ms → 0.9017 ms, 6.377×;
- row 13 final matrix: 88.8280 ms → 18.5412 ms, 4.791×;
- project-held-out long-attention incremental peak allocation: 78 MiB → 22 MiB,
  a 71.79% reduction; and
- row-9 fusion profile: 240 fused residual/LayerNorm launches and 120 Triton
  attention launches over 30 forwards, with a controlled 12.05% latency
  reduction against unchanged controls.

Do not describe the authorized 100,000-token row as a successful run. Say that it
was a source-authorized resource skip and was excluded from the pass count.

## Pre-recording checklist

Complete these checks before pressing record:

- [ ] Close unrelated applications and browser tabs.
- [ ] Use a readable terminal font and a clean repository-root PowerShell prompt.
- [ ] Confirm no credentials, API keys, `.env` values, private URLs, or personal
      paths beyond the necessary repository prompt are visible.
- [ ] Confirm the GPU, CUDA, PyTorch, and Triton versions.
- [ ] Decide whether the video will show live benchmark output or previously
      captured JSON opened from the repository. If a live command fails, show
      the failure honestly; do not silently quote an older green run.
- [ ] Keep the repository's own diagrams, code, terminal output, and authorized
      challenge material only. Do not add third-party music, logos, stock
      footage, unrelated screenshots, or copyrighted clips.
- [ ] Recheck the public repository URL and leave the YouTube URL blank until
      the upload and signed-out playback check are complete.

## Shot list and narration

The table is written as a teleprompter. The “screen” column describes what to
show; the “voice-over” column can be read nearly verbatim.

| Time | Screen | Voice-over |
| --- | --- | --- |
| 0:00–0:20 | Plain-text title card: “SpeedROCm — shape-aware Triton acceleration for Transformer inference”. Show the repository name and Track 3 label; use no third-party logos. | “This is Speed Rock-em, our project name for a PyTorch and Triton Transformer optimization. To be precise, this prototype was built and measured on NVIDIA CUDA, not AMD ROCm. The goal is lower latency and memory use while preserving the supplied model’s strict output contract.” |
| 0:20–0:40 | Open the baseline flow in [`03_TECHNICAL_REPORT.md`](03_TECHNICAL_REPORT.md), or show a simple diagram: QK → scores → softmax → P@V. Add a small caption defining Q, K, P, and V. | “Q and K are the query and key tensors. Their product creates attention scores, softmax turns those scores into probabilities, and P times V combines those probabilities with the value vectors. The reference stores a batch-by-head-by-sequence-by-sequence matrix, so this temporary storage grows with the square of sequence length.” |
| 0:40–1:00 | Show the repository tree: `transformer_opt/`, `benchmarks/`, `tests/`, `docs/`. Highlight `attention.py`, `dispatch.py`, and `submission.py`. | “The submission keeps the organizer model and parameter names, then replaces only measured bottlenecks. The main code is a fused attention kernel, a guarded dispatcher, packed QKV projection logic, and exact-shape residual/LayerNorm fusion.” |
| 1:00–1:25 | Show the Mermaid attention diagram or a code view around `_attention_fwd`. Scroll through the key/value tile loop, mask predicate, running maximum/sum, and accumulator. | “Each Triton program owns a small query tile and streams key and value tiles. It applies causal and prefix-padding bounds inside the tile, updates 32-bit floating-point online-softmax state, accumulates probability times value, and writes the context. No full score or probability matrix is allocated.” |
| 1:25–1:45 | Show the QKV cache diagram and the exact routing guards. | “For measured eager CUDA 32-bit shapes, the query, key, and value projections are packed into one vendor linear launch. The derived cache is invalidated after parameter changes and does not alter the saved parameter dictionary. Shape-aware routing sends sensitive cases to PyTorch scaled dot-product attention, or SDPA, or to exact reference math.” |
| 1:45–2:05 | Run the environment probe from PowerShell. | “The recorded target is an RTX 5070 Ti with Python 3.12.10, PyTorch 2.13 with CUDA 13, and Triton 3.7.1. The benchmark uses synchronized CUDA events and excludes first-use compilation from steady-state timing.” |
| 2:05–2:30 | Run the focused test command, or show its completed terminal output: `pytest tests/test_gpu_attention.py tests/test_gpu_transformer.py tests/test_dispatch.py -q`. | “These tests cover direct kernel arithmetic, masks, causal behavior, backend selection, strict weight copying, packed-cache invalidation, exact-row hybrids, and fusion boundaries. A CPU pass is treated as semantic coverage; GPU claims require CUDA evidence.” |
| 2:30–2:55 | Run the untouched organizer default harness. Keep the final summary visible. | “First, the untouched organizer PyTorch harness is run with only the submission injected. The recorded result is 5 out of 5 accuracy passes, zero failed elements, and a 1.385 times median speedup. All 1,950 optimized attention calls in this run used Triton.” |
| 2:55–3:25 | Run the published final matrix command, or open the curated final JSON summary. Show rows, PASS counts, and the skip accounting. | “Next is the published final shape table. Thirteen executable rows pass with zero failed elements across 938,885,120 comparisons. The multiplicative average across those per-row speedups is 1.977 times. The 100,000-token row is an authorized resource skip, which means it was not executed and is not counted as a pass.” |
| 3:25–3:50 | Show the row-9 profiler command/result. Highlight `_attention_fwd`, `_residual_layer_norm_fwd`, and backend counts. | “The profiler proves that the custom code really executed. On exact row 9, thirty forwards contain 120 Triton attention launches and 240 fused residual/LayerNorm launches, with 30 initial native norms remaining. The counterbalanced 300-sample gate reduced optimized latency from a 0.815968 millisecond control mean to 0.717648 milliseconds.” |
| 3:50–4:10 | Show the held-out memory table/chart and long-attention row. | “The held-out cases test different batch sizes, sequence lengths, padding, causality, and model widths. Long attention drops incremental peak allocation from 78 to 22 mebibytes, a 71.79 percent reduction, because the dense score and probability intermediates are gone. One mebibyte is 1,048,576 bytes.” |
| 4:10–4:30 | Show the limitations section and the repository evidence links. | “The design is forward-only and tuned on one GPU. Automatic low-precision and unsupported cases prioritize exact reference behavior, packed QKV is limited to measured widths, and the organizer’s omitted dtype, padding, timing, tolerance, and backward policies remain explicit assumptions.” |
| 4:30–4:45 | End card with repository URL, “Technical report in `deliverables/03_TECHNICAL_REPORT.md`”, and a placeholder for the final YouTube link. | “The code, tests, raw result artifacts, technical report, and reproduction commands are in the public repository. After this video is uploaded publicly and verified while signed out, its YouTube URL will be added to Devpost.” |

## Exact commands for the recording

Run from the repository root in Windows PowerShell. Use disposable output names
under the ignored `results/` folder for live reruns.

In the commands below, `$python` stores the virtual-environment Python path,
`&` runs that path, a trailing grave accent continues the same command on the
next line, and `#` begins a non-executed comment.

```powershell
git status --short --branch
nvidia-smi

$python = ".venv\Scripts\python.exe"
& $python -c "import torch,triton; print(torch.__version__, triton.__version__); print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"

# Focused GPU, model, and dispatch proof.
& $python -m pytest tests/test_gpu_attention.py `
  tests/test_gpu_transformer.py tests/test_dispatch.py -q

# Untouched organizer PyTorch default.
& $python benchmarks/run_organizer_torch.py --device cuda

# Published final dimensions.
& $python benchmarks/run_organizer_validation.py `
  --matrix benchmarks/final_evaluator_shapes.json `
  --out results/demo-final-evaluator.json

# Source-derived contract breadth.
& $python benchmarks/run_organizer_validation.py `
  --out results/demo-source-derived.json

# Row-9 profiler proof.
& $python benchmarks/profile_cases.py `
  --manifest benchmarks/campaign11_profile_shapes.json `
  --case final-09-b64-d128-h1-s128 --dtype float32 `
  --attention-backend auto --expect-backend triton `
  --expect-fused-residual-layer-norm --steps 30 `
  --out results/demo-row9-profile.json
```

If recording from a machine without a CUDA/Triton environment, show the
repository artifacts and explain that the live GPU rerun is unavailable. Do not
present CPU output as a GPU performance result.

## Evidence to display on screen

Use the following files for short, readable close-ups:

| Evidence | File | What to highlight |
| --- | --- | --- |
| Current final matrix | [`../docs/results/rtx-5070-ti-2026-08-29-c11-integrated-final.json`](../docs/results/rtx-5070-ti-2026-08-29-c11-integrated-final.json) | 13 PASS, one `SKIPPED_RESOURCE`, zero failed elements, per-row median timings, backend counts. |
| Final confirmation | [`../docs/results/rtx-5070-ti-2026-08-29-c11-integrated-final-confirmation.json`](../docs/results/rtx-5070-ti-2026-08-29-c11-integrated-final-confirmation.json) | Independent 1.986× geometric mean and matching correctness/backend totals. |
| Held-out matrix | [`../docs/results/rtx-5070-ti-2026-08-29-c11-integrated-heldout-5seed.json`](../docs/results/rtx-5070-ti-2026-08-29-c11-integrated-heldout-5seed.json) | Seven different cases, five accuracy seeds per case, latency and memory. |
| Row-9 profiler | [`../docs/results/rtx-5070-ti-2026-08-29-c11-integrated-row09-profile.json`](../docs/results/rtx-5070-ti-2026-08-29-c11-integrated-row09-profile.json) | `_attention_fwd`, `_residual_layer_norm_fwd`, 240 fused calls, 120 Triton calls. |
| Technical explanation | [`03_TECHNICAL_REPORT.md`](03_TECHNICAL_REPORT.md) | Big-O derivation, before/after tables, Mermaid diagrams, rejected candidates, limitations. |

## Recording integrity rules

- Keep terminal output large enough to read but crop out usernames, home paths,
  tokens, keys, and unrelated files.
- If a live run reports `FAIL`, `OOM`, or `ERROR`, leave the result visible and
  explain it. Do not replace it with a previous artifact without saying which
  artifact is being shown.
- If the profiler does not contain `_attention_fwd` when Triton is expected,
  stop the take; dispatch counters alone are not profiler proof.
- If `validation_passed` is false or the fused residual/LayerNorm call count is
  zero, stop the take; the proof command is intentionally fail-closed.
- Do not show a private organizer document or an authenticated browser session
  unless its contents are authorized for public release.
- Use only original narration and repository-owned visuals, plus challenge
  material that the team is authorized to publish. No third-party soundtrack,
  logos, footage, memes, or stock imagery.

## Upload and Devpost handoff

After recording:

1. Review the complete video for readable output, secrets, private tabs, and
   unauthorized third-party content.
2. Upload to YouTube with **Public** visibility.
3. Open the video in a signed-out/private browser and confirm playback, audio,
   and the selected visibility.
4. Paste the public YouTube URL into the Devpost project description.
5. Open the GitHub repository while signed out and verify that the repository URL
   in Devpost is public and resolves.
6. Record the final URLs in the submission system; do not put a fabricated URL
   in this repository.
