# Auto: synchronized camera and LiDAR generation

Run Cosmos3-Nano-Transfer-Auto for bidirectional generation, or Cosmos3-Nano-Sim-Auto
for few-step causal generation. Both use camera-local captions and synchronized WSM
controls. Joint generation also needs numerical HD-map LiDAR controls.

## Choose a notebook

| Workflow | Notebook | Example output |
| --- | --- | --- |
| Native Transfer-Auto | [Framework](run_auto_with_cosmos_framework.ipynb) | Seven cameras, 297 frames, 720p; optional LiDAR |
| Native Sim-Auto | [Framework](run_sim-auto_with_cosmos_framework.ipynb) | Seven cameras, 297 frames, 720p; optional LiDAR |
| Served Transfer-Auto | [vLLM-Omni](run_auto_with_vllm_omni.ipynb) | Seven cameras, 201 frames, 480p; WSM-only and joint examples |
| Served Sim-Auto | [vLLM-Omni](run_sim_auto_with_vllm_omni.ipynb) | Seven cameras, 297 frames, 480p; RGB and joint examples |
| Render your controls | [WSM renderer](generate_wsm.ipynb) | Controls from your own RDS-HQ scene and calibration |
| Sim-Auto with FlashDreams | [FlashDreams](https://github.com/NVIDIA/flashdreams/integrations_v2/cosmos3_nano_sim_auto) | Integrate with [FlashDreams](https://flashdreams.org/main/models/cosmos3_nano_sim_auto/) for accelerated inference |

The shorter 480p serving examples use different output settings from the native
notebooks; these are example choices, not model limits. All examples use 30 FPS.
Native notebooks offer 480p or 720p and RGB-only, joint RGB/LiDAR, or WSM-only T2V.

## Setup and sample

Follow the [Framework setup guide](../../../../../../docs/setup.md) before opening a
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
vLLM-Omni uses the small request templates in [assets/daytime_scene](assets/daytime_scene/README.md)
and converts that same archive into a local numerical tensor. A range-map MP4 is
only a visualization and cannot replace LiDAR controls. The native 297-frame example
produces 99 LiDAR sweeps; the 201-frame serving example produces 67.

## Sampling and custom data

Transfer-Auto starts with guidance **3**, shift **10**, and **35 steps**. Guidance
controls text-conditioning strength; shift redistributes denoising time steps.
The native notebook uses the skill's recommended per-view negative prompt.
Sim-Auto uses guidance **1** and the released checkpoint's fixed **four-step**
schedule; Transfer-Auto CFG and negative-prompt tuning do not carry over.
See the [Transfer-Auto inference guide](../../../../cosmos3-nano-transfer-auto/inference/SKILL.md)
and [Sim-Auto skill](../../../../cosmos3-nano-sim-auto/inference/SKILL.md) for model-specific guidance.

Use explicit `camera_key` values that match the physical views. Keep calibration,
RGB, WSM, captions and LiDAR synchronized, using the model's expected timing and
WSM 4.0.1 rendering. For data and rig adaptation, follow the
[Transfer-Auto post-training skill](../../../../cosmos3-nano-transfer-auto/post-training/SKILL.md).

## Rendering WSM from your own scene

The published sample contains **already-rendered** controls, not the raw RDS-HQ
inputs needed by the renderer. [generate_wsm.ipynb](generate_wsm.ipynb) requires your
own scene archives, per-camera calibration and pose trajectories. It validates
the selected camera inputs and runs the [WSM renderer](wsm/README.md)
at 30 FPS for 297 frames. **Render first, then caption:** use the linked
[per-view captioning guide](../../../../cosmos3-nano-transfer-auto/inference/references/caption-guide.md)
on the final rendered views, or review your existing captions. The notebook
then writes a native T2V request that either native inference notebook can load.
Enable `GENERATE_LIDAR_CONTROL` in the notebook to render same-scene geometry controls
and export `request-joint.json`. This requires the scene's LiDAR extrinsic, ring
calibration and sweep-end poses; measured LiDAR is optional. See the
[LiDAR renderer guide](wsm/docs/lidar-rangemap.md).
Ensure you have rights to use and distribute your input scene.
