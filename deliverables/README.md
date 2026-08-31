# SpeedROCm submission deliverables

We prepared this submission-facing Markdown package for TikTok TechJam 2026
Track 3. The repository root and `docs/` remain the technical source of truth;
these files are organized for copy/paste, review, and recording. The
submission-facing technical report is `03_TECHNICAL_REPORT.md`.
`docs/IMPLEMENTATION_EVIDENCE.md` is its supporting repository reference, not a
second submission report.

## Start here

- For a quick non-technical overview, read
  [`01_PROJECT_DESCRIPTION_DEVPOST.md`](01_PROJECT_DESCRIPTION_DEVPOST.md).
- To install and reproduce the project, read
  [`02_PUBLIC_REPOSITORY.md`](02_PUBLIC_REPOSITORY.md).
- For the algorithms, Big-O derivations, before/after measurements, plots, and
  code-flow diagrams, read
  [`03_TECHNICAL_REPORT.md`](03_TECHNICAL_REPORT.md).
- To record the public walkthrough, follow
  [`04_DEMO_VIDEO_SCRIPT.md`](04_DEMO_VIDEO_SCRIPT.md).

## Project name and platform scope

We use **SpeedROCm** as the public project name. Our current implementation is
not an AMD ROCm build: it uses PyTorch and Triton on NVIDIA CUDA, and we recorded
the performance evidence on an NVIDIA GeForce RTX 5070 Ti. The name is branding
only and does not claim AMD affiliation, NVIDIA affiliation, or current AMD
ROCm runtime compatibility.

The Python package and import paths remain `transformer_opt`; they are code
identifiers, not a second project name.

## Voice and authorship

We use `we` and `our` for decisions, results, and limitations in the public
submission copy. We keep equations, code behavior, and artifact descriptions in
neutral technical language where that is clearer. References to Codex and
Claude Code are required provenance disclosures, not a substitute for measured
evidence or human ownership. Add the verified participant names and individual
responsibilities before submitting.

## Quick reading legend

| Term or symbol | Meaning in these deliverables |
| --- | --- |
| Baseline | The supplied reference Transformer used for correctness and timing comparisons. |
| Optimized | `UserOptimizedTransformer`, the submitted implementation. |
| Recorded evidence | A checked-in result from the stated implementation fingerprint and hardware environment; it is not a promise for every GPU. |
| Executable row | A benchmark shape that completed correctness and timing. |
| Authorized resource skip | A source-permitted case that was not executed because its dense baseline exceeds practical resources; it is not counted as a pass. |
| `×` after a number | Speedup multiplier. For example, `2×` means twice as fast, or half the latency, for the stated comparison. |
| `ms`, `µs` | Milliseconds and microseconds. `1 ms = 1,000 µs`. |
| `MiB`, `TB` | Mebibytes (1,048,576 bytes) and decimal terabytes (1,000,000,000,000 bytes). |

The technical report contains the complete notation, acronym, unit, and
diagram legend used by the submission.

## Files

| Deliverable | File | Purpose |
| --- | --- | --- |
| Written project description | [`01_PROJECT_DESCRIPTION_DEVPOST.md`](01_PROJECT_DESCRIPTION_DEVPOST.md) | Devpost-ready project narrative covering the problem, solution, tools, APIs, libraries, datasets, assets, results, and limitations. |
| Public code repository handoff | [`02_PUBLIC_REPOSITORY.md`](02_PUBLIC_REPOSITORY.md) | Repository overview, setup, reproduction commands, README coverage, limitations, contribution notes, and public-visibility checklist. |
| Technical report | [`03_TECHNICAL_REPORT.md`](03_TECHNICAL_REPORT.md) | Before/after complexity derivations, measured timings, input matrices, optimization evidence, plots, Mermaid code-flow diagrams, and rejected alternatives. |
| Demo video | [`04_DEMO_VIDEO_SCRIPT.md`](04_DEMO_VIDEO_SCRIPT.md) | A detailed 3–5 minute recording script with narration, terminal actions, evidence to show, and copyright/privacy checks. |

## Submission state

- We have prepared the written description, repository handoff, technical
  report, and video script.
- The code URL is <https://github.com/lukeai-tan/tiktok-techjam-2026>. It
  returned HTTP 200 to an anonymous request on 2026-09-01, and remote `main`
  matched the pre-edit local HEAD. Recheck after these local documentation
  changes are published and immediately before submission.
- We still need to record the YouTube video, upload it with public visibility,
  verify signed-out playback, and add it to Devpost.
- Because `ROCm` is also a third-party platform name and the video rules mention
  trademarks, the team should confirm that the **SpeedROCm** title itself is
  acceptable to the organizer before public submission. The script uses plain
  text and no third-party logos, but this local documentation review is not
  legal or organizer approval.
- Add only verified participant names and responsibilities to the contribution
  sections.

## Evidence anchors

- Current implementation: [`../transformer_opt/submission.py`](../transformer_opt/submission.py),
  [`../transformer_opt/dispatch.py`](../transformer_opt/dispatch.py),
  [`../transformer_opt/kernels/attention.py`](../transformer_opt/kernels/attention.py),
  and [`../transformer_opt/kernels/residual_layer_norm.py`](../transformer_opt/kernels/residual_layer_norm.py).
- Implementation contract: [`../docs/REQUIREMENTS.md`](../docs/REQUIREMENTS.md).
- Supporting implementation and benchmark evidence:
  [`../docs/IMPLEMENTATION_EVIDENCE.md`](../docs/IMPLEMENTATION_EVIDENCE.md).
- Current selected-submission result:
  [`../docs/results/rtx-5070-ti-2026-08-29-c11-integrated-final.json`](../docs/results/rtx-5070-ti-2026-08-29-c11-integrated-final.json).
- Independent final confirmation:
  [`../docs/results/rtx-5070-ti-2026-08-29-c11-integrated-final-confirmation.json`](../docs/results/rtx-5070-ti-2026-08-29-c11-integrated-final-confirmation.json).
- Human-readable campaign chronology:
  [`../docs/experiments/CAMPAIGN_RUN_THROUGH.md`](../docs/experiments/CAMPAIGN_RUN_THROUGH.md)
  and [`../docs/experiments/OPTIMIZATION_HISTORY.md`](../docs/experiments/OPTIMIZATION_HISTORY.md).
- Existing compliance audit:
  [`../hackathon-docs/TRACK3_COMPLIANCE.md`](../hackathon-docs/TRACK3_COMPLIANCE.md).
