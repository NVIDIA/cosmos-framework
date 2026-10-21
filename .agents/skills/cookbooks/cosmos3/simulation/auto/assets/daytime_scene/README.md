# Shared daytime inference example

The four JSON files here are **vLLM-Omni request templates**, not bundled media.
All Auto notebooks download inputs from the same pinned
[Hugging Face sample](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto/tree/b8dcc2f70421e31db86d1b81d194c791d65eaad2/assets/sample_data).
The scene and original paragraph captions come from the released
[PhysicalAI-Autonomous-Vehicles dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).

## Contents and caption styles

The sample contains seven named cameras, 300 RGB and WSM frames at 30 FPS, and
100 HD-map LiDAR control sweeps at 10 Hz. It also provides native inference requests,
training metadata and recorded targets. Inference notebooks download only their
needed inputs, not training LiDAR targets or model weights.

- `daytime_scene/rgb/<camera_key>.mp4`: recorded RGB; inference uses its first frame.
- `daytime_scene/wsm/<camera_key>.mp4`: WSM 4.0.1 controls for the same camera/timeline.
- `daytime_scene/captions/<camera_key>.json`: original dense paragraph captions.
- `daytime_scene/captions_with_timestamp/<camera_key>.json`: timestamped captions that
  better match training and empirically improve inference. These are the default.
- `daytime_scene/lidar/control.tar`: numerical HD-map range controls.

Set `CAPTION_FOLDER` in the notebook to choose either caption style. The native
notebooks preserve caption JSON files. Serving helpers extract caption text for
the requested window, clipping timestamp labels for shorter examples. Keep each
view's captions consistent with visible geometry/motion and with overlapping views.

## Request templates

| Template | RGB conditioning | LiDAR output | Frames per camera |
| --- | --- | --- | --- |
| `transfer_auto_wsm.json` | None | No | 201 |
| `transfer_auto_joint.json` | First frame | Yes | 201 |
| `sim_auto.json` | First frame | No | 297 |
| `sim_auto_joint.json` | First frame | Yes | 297 |

These serving templates use 480p, 30 FPS and seed 0. Transfer-Auto uses guidance 3,
shift 10 and 35 steps; Sim-Auto uses guidance 1 and the fixed four-step student
schedule. Native notebooks use the **published native JSONLs** instead, defaulting
to 720p and 297 frames for both models. They use the same inputs and caption choices.

## LiDAR formats

Native Framework reads `control.tar` directly. vLLM-Omni preparation converts it
locally into `prepared/lidar_control.safetensors` under the notebook output folder.
The converter reads sparse 128×3600 range maps, pools each pair of horizontal
columns using the nearest valid return, and writes `[3,100,128,1800]` FP32 frames:
range in metres, unit intensity at valid returns, and validity. Source archive
checksums and sweep IDs are stored in the tensor metadata.

The 201-frame serving request consumes 67 sweeps; the 297-frame request consumes
99. Native output stores `range_m`, `intensity` and `frame_ids`; serving output uses
its packed `frames` representation. Each notebook checks its own output contract.
No converted tensor is checked in; rerun joint preparation to recreate it.
