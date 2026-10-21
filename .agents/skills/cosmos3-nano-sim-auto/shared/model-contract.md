# Sim-Auto model and checkpoint contract

Sim-Auto generates driving camera views and optional LiDAR in small temporal
chunks, using cached history and four denoising steps. It follows per-camera WSM
controls and captions, with HD-map range controls for joint output.

Cosmos3-Nano-Sim-Auto is the causal, four-step distilled descendant of
Cosmos3-Nano-Transfer-Auto. Training constructs teachers and a fake-score network;
inference loads only the exported student.

## Preserve these model settings

- Joint camera/LiDAR architecture: 11 camera identities plus one LiDAR identity,
  LiDAR patch size 1, and camera-local text.
- Causal history: three RGB latent frames per chunk, strict-past clean RGB for
  controls, chunk-causal clean passes and a 0.4-second cross-view history window
  with overlapping keys counted once.
- RGB compression: temporal 4, spatial 16×16. LiDAR: temporal 1, spatial 16×16,
  128 latent channels. Use the tokenizer config and weights delivered together.
- Camera/LiDAR clocks: 30/10 Hz, synchronized by source-frame IDs. Camera-major
  storage is not a spatial mosaic. See the [data contract](data-contract.md).
- Fixed SDE levels: `[1.0, 0.9375, 0.8333333333333334, 0.625]`, then zero.
  Generic RF step-count/shift options do not replace this schedule.
- The documented streaming export stores `rolling_kv_cache_chunks=2`,
  `rolling_kv_sink_chunks=1` and `rolling_prompt_policy=segmented`.
  Preserve these in the inference artifact.

The model uses learned physical sensor IDs; it does not infer camera identity
from a filename or caption. Follow the shared
[input-preparation](../../cosmos3-nano-transfer-auto/inference/references/input-preparation.md)
and [caption guide](../../cosmos3-nano-transfer-auto/inference/references/caption-guide.md).

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
[inference](../inference/SKILL.md), or
[troubleshooting](../troubleshooting/SKILL.md) for the selected task.
Keep both Auto bundles together so the shared guides remain available.
