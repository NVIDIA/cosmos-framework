# Sim-Auto distillation notes

> **Skill note (not in the source):** these blocks are from the body of the Sim-Auto post-training skill that this skill replaces (cosmos-framework `main` 45fecab), moved here word for word because SKILL.md has a size budget; only link targets were re-pointed. "Data instructions" is a new heading over the data paragraphs of [What you need](../SKILL.md#what-you-need), which step 2 of the Quickstart calls "the data instructions above"; "Three stages" is the source's heading and section; "Training budget, stage weights, monitoring and records" is a new heading over the last four paragraphs of [Continue training and export](../SKILL.md#continue-training-and-export). The rest of the body is in [SKILL.md](../SKILL.md#distill-sim-auto).

## Data instructions

The shared [reference scene](../../cosmos3-inference-simulation-auto/references/reference-scene.md)
provides an HF download of `daytime_scene` and its native training index.
It is a format example, not a training corpus. Use
[Transfer-Auto data preparation](../SKILL.md#prepare-a-local-training-bundle)
to create the same local format. Then apply the
[Sim-Auto data settings](sim-auto-data.md), including the camera-map
overrides required for the generic bundle builder. For this 720p/300-frame scene,
apply the [short training settings](training-bundle-and-short-check.md#short-training-check)
as well; the full TF window does not fit.
An inference request alone is not training data.

Keep caption files as scene descriptions only. The native training transforms
add the camera-rig/current-view direction and FOV description and the joint
WSM/HD-map task instruction automatically during formatting and tokenization.
Do not duplicate either in your labels; see
[automatic prefixes](../../cosmos3-inference-simulation-auto/references/caption-policy.md#caption-text-and-automatic-prefixes).

## Three stages

| Stage                                             | What it learns                                       | Student starts from | Teacher                              |
| ------------------------------------------------- | ---------------------------------------------------- | ------------------- | ------------------------------------ |
| 1: teacher forcing (TF)                           | Generate causally using clean preceding observations | Transfer-Auto       | None                                 |
| 2: teacher-forcing diffusion consistency (TF-dCM) | Reduce generation to four denoising steps            | Accepted Stage 1    | Frozen Stage 1                       |
| 3: self-forcing distribution matching (SF-DMD)    | Generate using its own preceding predictions         | Accepted Stage 2    | Original bidirectional Transfer-Auto |

Stage 3 also trains a fake-score network initialized from the original Transfer-Auto
teacher. That network is not part of the inference artifact.

## Training budget, stage weights, monitoring and records

The reference budget is 4000 **total trainer iterations per stage**. Stage 3 alternates
one student update with five critic updates: 4000 iterations give 667 student and
3333 critic updates. Evaluate the trained student on held-out scenes.

Use regular-network weights between stages; do not substitute EMA. Each new
stage starts new optimizer, scheduler and iteration state. Same-stage resume
restores that stage's complete state instead. See the
[checkpoint guide](sim-auto-checkpoints.md).

Training uses paired RGB/LiDAR dropout and per-camera captions. Inspect both
present- and absent-LiDAR batches. Monitor actual losses and Stage 3
`phase_metrics.jsonl`, not just whether a process is alive.

Preserve accepted checkpoints, resolved configs, runtime/source versions, data
manifest and test results. Do not upload confidential data or credentials in an
MR, logs or model export.
