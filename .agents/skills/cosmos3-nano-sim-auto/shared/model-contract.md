# Sim-Auto model and checkpoint contract

Sim-Auto generates driving camera views and optional LiDAR in small temporal
chunks, using cached history and four denoising steps. It follows per-camera WSM
controls and captions, with HD-map range controls for joint output.

Cosmos3-Nano-Sim-Auto is the causal, four-step distilled descendant of
Cosmos3-Nano-Transfer-Auto. Training constructs teachers and a fake-score network;
inference loads only the exported student.

> **Skill note (not in the source):** the section "Preserve these model settings" moved to the Sim-Auto section of the [Auto model contract](../../cosmos3-inference-simulation-auto/references/model-contract.md#sim-auto).

## Checkpoint handoffs

| Producer → consumer | Artifact                                               | Check                                                                                |
| ------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| Transfer-Auto → TF  | Weights-only DCP iteration root                        | Complete regular network, compatible geometry/tokenizers; new optimizer/counters     |
| TF → TF-dCM         | Accepted iteration root; its `model/` is the teacher   | Teacher and student use the same accepted regular Stage 1 weights                    |
| TF-dCM → SF-DMD     | Stage 2 `model/`, plus original Transfer-Auto `model/` | Stage 2 student; original bidirectional teacher and fake-score initialization        |
| SF-DMD → inference  | Regular-student HF export                              | Student architecture, tokenizers, per-view text and rolling metadata travel together |

See [preparation](../post-training/references/checkpoint-preparation.md)
and [save/resume/export](../post-training/references/checkpoints.md).
Do not load a raw training checkpoint through the HF inference entry point.

## Trainable parameters

The supplied stages optimize `moe_gen`, `time_embedder`, `vae2llm`,
`llm2vae`, `lidar2llm`, `llm2lidar` and `rig_view_embed`.
Tokenizers and other groups remain frozen. Keep teacher/student and regular/EMA
identities explicit; do not weaken strict loading to conceal an incompatible artifact.

Use [post-training](../post-training/SKILL.md),
[inference](../../cosmos3-inference-simulation-auto/SKILL.md#sim-auto), or
[troubleshooting](../troubleshooting/SKILL.md) for the selected task.
Keep both Auto bundles together so the shared guides remain available.
