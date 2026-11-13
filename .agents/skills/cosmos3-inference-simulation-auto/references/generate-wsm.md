<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Render WSM v4.0.1 and prepare Auto inference

**Your scene → calibrated WSM controls → per-view captions → native Auto request.**

Supply scene annotations, camera intrinsics/extrinsics and poses. Captions are prepared
**after rendering** so they describe the final views. The published HF daytime sample
contains already-rendered WSM videos, not raw RDS-HQ inputs; use it directly in the
inference notebooks if you only want to try the models.

Convert your own source with the [RDS-HQ schema](../scripts/wsm/docs/rds-hq.md) and
[conversion skill](../scripts/wsm/skills/scenario-to-rds-hq/SKILL.md).
Use data you have permission to process and keep confidential inputs outside the checkout.
No raw customer scene or calibration is bundled with this cookbook.

> **Skill note (not in the source):** this file is the cookbook notebook `generate_wsm.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/simulation/auto/generate_wsm.ipynb`, cosmos-framework `main` 45fecab) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced as `python`, outputs omitted (the committed notebook has none). Edits are limited to three kinds: link targets that pointed at files this migration moved were re-pointed; path anchors in code that the move invalidated were replaced, each marked in place with a `Skill note (not in the source)` comment; and location text that named the retired cookbook folder (default output folders, where to run from) now names this skill's folder, each marked by a skill note. The changes are recorded in `docs/cookbook_to_skill_migration.md`.

## 1. Environment and inputs

Install the standalone renderer following [WSM setup](../scripts/wsm/README.md#install)
and its NVIDIA EGL instructions. The renderer needs a graphics-capable NVIDIA driver; the
notebook kernel needs NumPy, IPython and FFmpeg/FFprobe. Model inference uses the separate
Framework environment described in the native inference notebooks.

Set these environment variables before running the configuration cell:

```bash
export AUTO_RDS_HQ_ROOT=/path/to/your/rds_hq
export AUTO_RDS_HQ_CLIP=your_clip_id
export WSM_PYTHON=/path/to/renderer/venv/bin/python
```

`AUTO_RDS_HQ_ROOT` contains `pose/`, `ftheta_intrinsic/`, `all_object_info/`, map layers
and traffic-light states. The clip ID is shared by their archives. Prepare a synchronized
30 Hz pose timeline covering at least 297 frames. The renderer samples existing poses;
it does not upsample 10 Hz poses to 30 Hz. Convert/resample the scene consistently first
rather than changing FPS metadata alone.

Optionally set `AUTO_RDS_HQ_CAPTIONS` to existing camera-local JSONs for later review/export.
Otherwise the notebook prints a new `view_captions/` location for caption preparation.
Set `AUTO_WSM_OUTPUT_ROOT` to choose your output directory. To reuse controls rendered
with the same settings, point it to their output root, match `CAMERAS`, `NUM_FRAMES`,
`FPS` and `RESOLUTION_SPEC` below, then skip the render cell. The default 1280×720
profile matches the renderer README; choose `source` for unresized calibrated views.
For the README's one-camera example, select only `camera_front_wide_120fov`.

This notebook supports **F-theta scenes, supported Auto camera keys, and 30 Hz poses**.
For pinhole calibration or custom camera names, use the standalone renderer and the
[custom-rig handoff](../scripts/wsm/docs/scenario-to-rds-hq.md#custom-rigs-and-already-rendered-controls).
Do not relabel the projection or rename files to satisfy this notebook.

For best released-model quality, use WSM v4.0.1 and camera intrinsics/extrinsics matching
the reference rig. Inspect geometry with the [Camera Rig Explorer](../scripts/wsm/docs/camera-rig/guide.md)
and use [retargeting](../scripts/wsm/docs/scenario-to-rds-hq.md#camera-retargeting) when appropriate.
For paired post-training with your recorded RGB, retain your own rig. Renaming cameras
alone does not adapt the model, and retargeted controls cannot be paired with unchanged source RGB.

```python
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys

from IPython.display import Video, display


def find_auto_root() -> Path:
    configured = os.environ.get("COSMOS_COOKBOOK_ROOT")
    start = Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()
    for root in [start] if configured else [start, *start.parents]:
        # Skill note (not in the source): the Auto cookbook folder is now this skill's folder; the candidates were root / "auto", root / "cookbooks/cosmos3/simulation/auto", root / ".agents/skills/cookbooks/cosmos3/simulation/auto".
        for candidate in (root, root / "cosmos3-inference-simulation-auto",
                          root / ".agents/skills/cosmos3-inference-simulation-auto"):
            # Skill note (not in the source): the helpers are now in this skill's scripts/helpers folder; the marker was candidate / "helpers/wsm_helpers.py".
            if (candidate / "scripts/helpers/wsm_helpers.py").is_file():
                return candidate
    raise RuntimeError("Run inside the Framework checkout or set COSMOS_COOKBOOK_ROOT.")


AUTO_ROOT = find_auto_root()
# Skill note (not in the source): the skill folder is three levels below the repository root; this was AUTO_ROOT.parents[5], counted from the cookbook's Auto folder.
FRAMEWORK_DIR = AUTO_ROOT.parents[2]
# Skill note (not in the source): the helpers package is now under this skill's scripts folder; this was sys.path.insert(0, str(AUTO_ROOT)).
sys.path.insert(0, str(AUTO_ROOT / "scripts"))
from helpers.auto_helpers import CAMERA_ORDER, probe_video
from helpers.wsm_helpers import export_wsm_request, inspect_scene

INPUT_ROOT = Path(os.environ["AUTO_RDS_HQ_ROOT"]).expanduser().resolve()
CLIP_ID = os.environ["AUTO_RDS_HQ_CLIP"]
# Preserve venv/bin/python symlinks so the selected virtual environment stays active.
WSM_PYTHON = Path(os.environ["WSM_PYTHON"]).expanduser().absolute()
CAMERAS = list(CAMERA_ORDER[:7])  # Select the physical cameras present in your calibrated scene.
NUM_FRAMES, FPS = 297, 30
GENERATE_LIDAR_CONTROL = False  # Requires this scene's LiDAR extrinsic/calibration and an end pose.
RESOLUTION_SPEC = "resize_1280x720"  # Or "source"; must match reused controls.
OUTPUT_ROOT = Path(os.environ.get(
    "AUTO_WSM_OUTPUT_ROOT",
    str(FRAMEWORK_DIR / "outputs/wsm" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")),
)).expanduser().resolve()
CAPTION_ROOT = Path(os.environ.get("AUTO_RDS_HQ_CAPTIONS", str(OUTPUT_ROOT / "view_captions"))).expanduser().resolve()
scene = inspect_scene(INPUT_ROOT, CLIP_ID, CAMERAS, NUM_FRAMES, FPS, RESOLUTION_SPEC)
print("Expected rendered sizes:", scene["resolutions"])
print("Caption source directory (needed after rendering):", CAPTION_ROOT)
```

## 2. Render controls

WSM 4.0.1 is selected by `version=v4`, not `v4.1`. Use the supplied calibration and
poses; there is no fallback to a bundled scene or calibration. Review scene components
against the schema; the renderer reads the remaining map/object inputs.
Output stays local, with upload disabled. The explicit options below select **30 FPS,
297 frames (9.9 seconds)** instead of the renderer's general-purpose 10 FPS defaults.

Skip this cell only when `OUTPUT_ROOT` already contains the requested render; continue
with the video checks. A new render requires a fresh output directory.

```python
OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
command = [
    str(WSM_PYTHON), "-m", "cosmos_auto.wm.cli",
    "input_root=" + json.dumps(str(INPUT_ROOT)),
    "output_root=" + json.dumps(str(OUTPUT_ROOT)),
    "clip_id_json=" + json.dumps(CLIP_ID),
    "version=v4", "camera_type=ftheta",
    "camera_spec.camera_list=" + json.dumps(CAMERAS),
    f"resolution_spec={RESOLUTION_SPEC}", "s3_spec=no_upload",
    f"input_fps_spec.input_pose_fps={FPS}",
    f"output_frame_spec.target_render_fps={FPS}",
    f"output_frame_spec.target_chunk_frame={NUM_FRAMES}",
    "output_frame_spec.overlap_frame=0", "output_frame_spec.max_chunk=1",
]
subprocess.run(command, check=True)
```

## 3. Check the rendered views

Verify frame counts, FPS and the selected render sizes for every camera. Inspect geometry,
motion and signals throughout the clips. Compare source-camera WSM against synchronized
RGB when available. Retargeted views have different geometry; source RGB is not an overlay
reference for them.

```python
for camera in CAMERAS:
    video = OUTPUT_ROOT / "world_scenario" / f"ftheta_{camera}" / f"{CLIP_ID}_0.mp4"
    info = probe_video(video)
    if (info["frames"], info["fps"], [info["width"], info["height"]]) != (
        NUM_FRAMES, FPS, scene["resolutions"][camera],
    ):
        raise ValueError(f"Unexpected rendered video: {video}: {info}")
    print(camera, video)
    display(Video(filename=str(video), embed=True, width=480))
print("Selected interval:", NUM_FRAMES, "frames at", FPS, "FPS")
print("Caption source directory:", CAPTION_ROOT)
```

## 4. Prepare or review per-view captions

Use the existing [AV per-view captioning guide](caption-guide.md)
through the [Transfer-Auto inference skill](../SKILL.md#transfer-auto).
Provide the rendered controls, physical camera keys and the selected interval. Also provide
**time of day, weather, road condition and geographic setting**; WSM does not determine these.
A suitable request to your agent is:

> Inspect these synchronized rendered WSM views with the per-view captioning guide.
> Generate timestamped captions for the selected interval and my supplied scenario properties.
> Save one `<camera_key>.json` sidecar per camera in the printed caption source directory,
> using the `caption_structured` format below. Keep shared objects' appearance and motion
> consistent across views. Do not run model inference yet.

For existing captions, request a review against the **final** rendered views and corrections
as feedback; do not silently rewrite the originals. Retargeting can change object visibility,
so source-view captions need a fresh check. Timestamped text is preferred because it better
matches training; plain paragraphs also work. Framework adds camera/FOV and task prefixes.

Each selected camera needs a file named with its actual camera key, for example
`camera_front_wide_120fov.json`. For the default 297-frame interval:

```json
{
  "caption_structured": {
    "chunk_0_297": {
      "caption": "[0.0s-3.0s]: ...\n[3.0s-9.9s]: ...",
      "start_frame": 0,
      "end_frame": 297
    }
  }
}
```

Replace the ellipses with grounded camera-local descriptions. A `captions.json` mapping alone
is not the expected sidecar format. The example exporter uses one outer chunk covering the
first window (end frame exclusive), with timestamped text inside. It writes interval-specific
copies under `OUTPUT_ROOT/captions`; keep the originals in `CAPTION_ROOT` separately.
For training, captions must describe your actual recorded RGB targets; see the
[training caption policy](../../cosmos3-post-train-simulation-auto/references/data-contract.md#training-captions).

```python
sampling = {
    "num_frames": NUM_FRAMES, "fps": FPS, "num_steps": 35, "guidance": 3, "shift": 10,
    "seed": 0, "control_guidance": 1,
}
REQUEST_PATH = export_wsm_request(OUTPUT_ROOT, scene, sampling, CAPTION_ROOT)
print("Native Framework request:", REQUEST_PATH)
```

## 5. Optional: render same-scene LiDAR controls

Set `GENERATE_LIDAR_CONTROL = True` above to export a joint request. Install the updated
standalone WSM package in `WSM_PYTHON`. Supply this scene's rig/sensor collection and
128-ring calibration as described in the [LiDAR guide](../scripts/wsm/docs/lidar-rangemap.md).
Measured LiDAR is not required. Camera WSM MP4s do not replace the 3D scene or calibration.
The renderer uses RDS-HQ ego poses and map/object geometry; it fails on missing extrinsics.
The bundled lattice is a Pandar128 default: verify it matches your sensor/checkpoint.

At 30 FPS, 201 camera frames need 67 complete 10 Hz sweeps and at least 202 source
poses, including the last sweep's end. The cell checks sweep count and start-frame IDs
before writing the request. It never borrows another scene's controls or RGB. The output
`lidar_control.safetensors` is physical control input, distinct from generated `lidar.safetensors`.

```python
if GENERATE_LIDAR_CONTROL:
    import math
    from safetensors import safe_open

    if FPS != 30:
        raise ValueError("This joint handoff requires the 30 Hz camera / 10 Hz LiDAR timeline.")
    sweep_count = math.ceil(NUM_FRAMES * 10 / FPS)
    lidar_output = OUTPUT_ROOT / "lidar"
    subprocess.run([
        str(WSM_PYTHON), "-m", "cosmos_auto.lidar.cli",
        "--input-root", str(INPUT_ROOT), "--output-root", str(lidar_output),
        "--clip-id", CLIP_ID, "--camera-name", CAMERAS[0],
        "--max-frames", str(sweep_count), "--export-control", "--no-vis",
    ], check=True)
    control_path = lidar_output / "controls" / CLIP_ID / "lidar_control.safetensors"
    with safe_open(str(control_path), framework="np") as control:
        frames = control.get_slice("frames")
        source = json.loads(control.metadata()["source"])
        if frames.get_shape() != [3, sweep_count, 128, 1800] or frames.get_dtype() != "F32":
            raise ValueError("Missing complete physical LiDAR controls; check source sweep end poses.")
        if source["clip"] != CLIP_ID or source["camera_frame_ids"] != list(range(0, sweep_count * 3, 3)):
            raise ValueError("LiDAR controls must start at camera frame 0 and cover the same timeline.")
    joint_request = json.loads(REQUEST_PATH.read_text())
    joint_request["lidar"] = {"control_path": str(control_path)}
    joint_request_path = OUTPUT_ROOT / "request-joint.json"
    joint_request_path.write_text(json.dumps(joint_request, indent=2) + "\n")
    print("Joint request:", joint_request_path)
```

## 6. Generate RGB or joint RGB/LiDAR

Open the [native Transfer-Auto notebook](transfer-auto-cosmos-framework.md), set
`CUSTOM_REQUEST` to the exported `request.json`, and choose `MODE = "text2video"`.
The generated request needs no sample-specific converter.

For RGB image conditioning, supply actual same-view first frames matched to the rendered geometry.
For joint LiDAR generation, enable the preceding control-rendering cell and use its
`request-joint.json` and set `MODE = "joint"`. This retains LiDAR and uses text-to-video
when no RGB first frames are provided. `MODE = "text2video"` is camera-only and drops LiDAR.
Camera rendering and LiDAR rendering use the same source scene. Do not reuse the unrelated HF daytime sample's RGB or LiDAR.

For Sim-Auto T2V, the [native Sim-Auto notebook](sim-auto-cosmos-framework.md)
uses the same controls/captions and applies guidance 1 with the checkpoint's distilled sampler:
its preparation cell removes Transfer-Auto's `num_steps`, `shift` and negative-prompt fields.
Inspect all views for geometry, motion and cross-view appearance consistency.
