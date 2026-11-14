# Auto: synchronized camera and LiDAR generation

> **Skill note (not in the source):** this file is the Auto cookbook README `README.md` (retired into this skill from `.agents/skills/cookbooks/cosmos3/simulation/auto/README.md`, cosmos-framework `main` 45fecab) copied verbatim. Edits are limited to three kinds: link targets that pointed at files this migration moved were re-pointed; path anchors in code blocks that the move invalidated were replaced; and location text that named the retired cookbook folder now names this skill's folder. Each code or location change is marked in place with a skill note. The changes are recorded in `docs/cookbook_to_skill_migration.md`.

Run Cosmos3-Nano-Transfer-Auto for bidirectional generation, or Cosmos3-Nano-Sim-Auto
for few-step causal generation. Both use camera-local captions and synchronized WSM
controls. Joint generation also needs numerical HD-map LiDAR controls.

## Choose a notebook

| Workflow                  | Notebook                                                                                   | Example output                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| Native Transfer-Auto      | [Framework](transfer-auto-cosmos-framework.md)                                             | Seven cameras, 297 frames, 720p; optional LiDAR                                                                    |
| Native Sim-Auto           | [Framework](sim-auto-cosmos-framework.md)                                                  | Seven cameras, 297 frames, 720p; optional LiDAR                                                                    |
| Served Transfer-Auto      | [vLLM-Omni](transfer-auto-vllm-omni.md)                                                    | Seven cameras, 201 frames, 480p; WSM-only and joint examples                                                       |
| Served Transfer-Auto      | [SGLang](transfer-auto-sglang.md)                                                          | Seven cameras, 201 frames, 480p; WSM-only and joint examples from the published `lmsysorg/sglang:dev` image        |
| Served Sim-Auto           | [vLLM-Omni](sim-auto-vllm-omni.md)                                                         | Seven cameras, 297 frames, 480p; RGB and joint examples                                                            |
| Served Sim-Auto           | [SGLang](sim-auto-sglang.md)                                                               | Seven cameras, 297 frames, 480p; RGB example from the published `lmsysorg/sglang:dev` image                        |
| Render your controls      | [WSM renderer](generate-wsm.md)                                                            | Controls from your own RDS-HQ scene and calibration                                                                |
| Sim-Auto with FlashDreams | [FlashDreams](https://github.com/NVIDIA/flashdreams/integrations_v2/cosmos3_nano_sim_auto) | Integrate with [FlashDreams](https://flashdreams.org/main/models/cosmos3_nano_sim_auto/) for accelerated inference |

The shorter 480p serving examples use different output settings from the native
notebooks; these are example choices, not model limits. All examples use 30 FPS.
Native notebooks offer 480p or 720p and RGB-only, joint RGB/LiDAR, or WSM-only T2V.

## Setup and sample

Follow the [Framework setup guide](../../../../docs/setup.md) before opening a
native notebook with that environment's Python kernel. Start from the repository
root or this Auto folder. Serving notebooks document their separate server and
client environments. Authenticate with Hugging Face and accept any required model
terms before downloading weights.

All inference notebooks download the same
[daytime sample](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto/tree/main/assets/sample_data),
originally from [PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).
It has seven named cameras, 300 source RGB/WSM frames at 30 FPS, and 100 LiDAR
control sweeps at 10 Hz. It is a reference example, not a complete training dataset.
The helper verifies downloaded inputs against their published checksums. No media
or proprietary raw scene is duplicated in this cookbook.

The original `captions/` contains dense paragraphs. The alternative
`captions_with_timestamp/` better matches the training caption style and empirically
improves inference. Notebooks select it by default; set `CAPTION_FOLDER="captions"`
to compare. Describe each view locally and keep shared objects' appearance and
motion consistent across cameras. The runtime adds the camera/FOV and WSM task
prefixes; do not copy another camera's scene description into your caption.

Native Framework consumes the published JSONL request and `control.tar` directly.
vLLM-Omni uses the small request templates in [assets/daytime_scene](../assets/daytime_scene/README.md)
and converts that same archive into a local numerical tensor. A range-map MP4 is
only a visualization and cannot replace LiDAR controls. The native 297-frame example
produces 99 LiDAR sweeps; the 201-frame serving example produces 67.

## Sampling and custom data

Transfer-Auto starts with guidance **3**, shift **10**, and **35 steps**. Guidance
controls text-conditioning strength; shift redistributes denoising time steps.
The native notebook uses the skill's recommended per-view negative prompt.
Sim-Auto uses guidance **1** and the released checkpoint's fixed **four-step**
schedule; Transfer-Auto CFG and negative-prompt tuning do not carry over.
See the [Transfer-Auto inference guide](../SKILL.md#transfer-auto)
and [Sim-Auto skill](../SKILL.md#sim-auto) for model-specific guidance.

Use explicit `camera_key` values that match the physical views. Keep calibration,
RGB, WSM, captions and LiDAR synchronized, using the model's expected timing and
WSM 4.0.1 rendering. For data and rig adaptation, follow the
[Transfer-Auto post-training skill](../../cosmos3-nano-transfer-auto/post-training/SKILL.md).

## Rendering WSM from your own scene

The published sample contains **already-rendered** controls, not the raw RDS-HQ
inputs needed by the renderer. [generate_wsm.ipynb](generate-wsm.md) requires your
own scene archives, per-camera calibration and pose trajectories. It validates
the selected camera inputs and runs the [WSM renderer](../scripts/wsm/README.md)
at 30 FPS for 297 frames. **Render first, then caption:** use the linked
[per-view captioning guide](caption-guide.md)
on the final rendered views, or review your existing captions. The notebook
then writes a native T2V request that either native inference notebook can load.
Enable `GENERATE_LIDAR_CONTROL` in the notebook to render same-scene geometry controls
and export `request-joint.json`. This requires the scene's LiDAR extrinsic, ring
calibration and sweep-end poses; measured LiDAR is optional. See the
[LiDAR renderer guide](../scripts/wsm/docs/lidar-rangemap.md).
Ensure you have rights to use and distribute your input scene.
