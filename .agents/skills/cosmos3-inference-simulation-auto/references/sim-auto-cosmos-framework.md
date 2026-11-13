<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3-Nano-Sim-Auto with Cosmos Framework

Generate synchronized camera videos, optionally with LiDAR, from the shared seven-camera
[daytime sample](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto/tree/main/assets/sample_data).
The real RGB and original paragraph captions come from
[PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).
This notebook uses the published native request and Framework's existing readers;
it does not require a training run or a sample-specific dataloader.

Run setup once, select `MODE`, then run preparation, generation and output inspection.
The default is seven-camera RGB + LiDAR, **297 frames at 30 FPS, 720p**, with first-frame RGB conditioning.
[All Auto workflows](auto-readme.md) · [Inference skill](../SKILL.md#sim-auto)

> **Skill note (not in the source):** this file is the cookbook notebook `run_sim-auto_with_cosmos_framework.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/simulation/auto/run_sim-auto_with_cosmos_framework.ipynb`, cosmos-framework `main` 45fecab) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced as `python`, outputs omitted (the committed notebook has none). Edits are limited to three kinds: link targets that pointed at files this migration moved were re-pointed; path anchors in code that the move invalidated were replaced, each marked in place with a `Skill note (not in the source)` comment; and location text that named the retired cookbook folder (default output folders, where to run from) now names this skill's folder, each marked by a skill note. The changes are recorded in `docs/cookbook_to_skill_migration.md`.

## 1. Environment

Install Framework using [setup](../../../../docs/setup.md) and the
[Sim-Auto inference guide](../../../../docs/inference_sim_auto.md).
Select that environment as the Jupyter kernel (or set `COSMOS_AUTO_PYTHON` to its Python).
The kernel needs `huggingface_hub`, `numpy`, `safetensors`, `IPython`, and `matplotlib`;
install FFmpeg/FFprobe on PATH. Use Linux and sufficient NVIDIA GPU memory.
Sim-Auto additionally requires FlashAttention-4, included in the documented CUDA training environment.

The default launch uses four GPUs with CP=4 and CFGP=1. Change `NUM_GPUS` only for an
allocation with enough memory. This notebook does not install packages or clone another checkout.
Accept the HF model access terms and run `hf auth login` if required. Keep credentials out of notebooks.

Both 480p (832×480) and 720p (1280×720) are supported. The reference sample uses 720p.
For sampling, duration, FPS and custom-rig guidance, see the
[shared parameter guide](input-and-execution.md).

```python
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shlex
import subprocess
import sys
import tarfile

from IPython.display import HTML, display


def find_auto_root() -> Path:
    configured = os.environ.get("COSMOS_COOKBOOK_ROOT")
    start = Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()
    for root in [start] if configured else [start, *start.parents]:
        # Skill note (not in the source): the Auto cookbook folder is now this skill's folder; the candidates were root / "auto", root / "cookbooks/cosmos3/simulation/auto", root / ".agents/skills/cookbooks/cosmos3/simulation/auto".
        for candidate in (root, root / "cosmos3-inference-simulation-auto",
                          root / ".agents/skills/cosmos3-inference-simulation-auto"):
            # Skill note (not in the source): the helpers are now in this skill's scripts/helpers folder; the marker was candidate / "helpers/daytime_sample.py".
            if (candidate / "scripts/helpers/daytime_sample.py").is_file():
                return candidate
    raise RuntimeError("Run inside the Framework checkout or set COSMOS_COOKBOOK_ROOT to its Auto cookbook.")


AUTO_ROOT = find_auto_root()
# Skill note (not in the source): the skill folder is three levels below the repository root; this was AUTO_ROOT.parents[5], counted from the cookbook's Auto folder.
FRAMEWORK_DIR = AUTO_ROOT.parents[2]
# Preserve venv/bin/python symlinks so the selected virtual environment stays active.
FRAMEWORK_PYTHON = Path(os.environ.get("COSMOS_AUTO_PYTHON", sys.executable)).expanduser().absolute()
# Skill note (not in the source): the helpers package is now under this skill's scripts folder; this was sys.path.insert(0, str(AUTO_ROOT)).
sys.path.insert(0, str(AUTO_ROOT / "scripts"))
from helpers.auto_helpers import camera_grid, probe_video, validate_framework_outputs, validate_rgb_conditions
from helpers.daytime_sample import SAMPLE_REPO, SAMPLE_REVISION, fetch_sample, native_request
from helpers.lidar_rangemap import index_sweeps
from helpers.sim_auto_helpers import camera_previews, validate_sim_auto_outputs

MODEL_ID = "nvidia/Cosmos3-Nano-Sim-Auto"
MODEL_REVISION = os.environ.get("COSMOS_AUTO_MODEL_REVISION", "main")
LOCAL_CHECKPOINT = os.environ.get("COSMOS_AUTO_CHECKPOINT")
NUM_GPUS = 4
MODE = "joint"  # joint: RGB + LiDAR; camera: RGB with first frames; text2video: WSM + captions only
CAPTION_FOLDER = "captions_with_timestamp"  # Alternative: "captions"
CUSTOM_REQUEST = None  # Path to one native JSON/JSONL request; its captions are kept unchanged.
RESOLUTION = "720"  # "480" or "720", using aspect_ratio "16,9"
run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
OUTPUT_ROOT = FRAMEWORK_DIR / "outputs/sim_auto" / run_name
print("Framework:", FRAMEWORK_DIR)
print("Sample:", SAMPLE_REPO, "@", SAMPLE_REVISION)
print("Output:", OUTPUT_ROOT)
```

## 2. Download the sample and model

`AUTO_SAMPLE_ROOT` can point to an existing complete download. Otherwise, only the required
sample inputs are downloaded into the HF cache, with checksums verified. Model weights
are downloaded separately from one resolved model revision.

Two caption styles are available: `captions/` contains the original dense paragraphs;
`captions_with_timestamp/` better matches the model's training style and has empirically
given better inference results. Both work. Keep object appearance, motion and environmental
conditions consistent across views. Timing labels are text inside the same `[0,300)` caption
interval, not extra inference chunks. Framework adds the camera/task prefixes; do not duplicate them.

```python
from huggingface_hub import HfApi, snapshot_download

if CUSTOM_REQUEST is None:
    SAMPLE_ROOT = fetch_sample(os.environ.get("AUTO_SAMPLE_ROOT"))
    REQUEST_PATH = SAMPLE_ROOT / "input-sim-auto.jsonl"
else:
    REQUEST_PATH = Path(CUSTOM_REQUEST).expanduser().absolute()

if LOCAL_CHECKPOINT:
    CHECKPOINT_DIR = Path(LOCAL_CHECKPOINT).expanduser().resolve()
else:
    checkpoint_revision = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION).sha
    CHECKPOINT_DIR = Path(snapshot_download(
        MODEL_ID, revision=checkpoint_revision, ignore_patterns=["assets/**"],
    ))
    print("Model revision:", checkpoint_revision)
if not (CHECKPOINT_DIR / "config.json").is_file():
    raise FileNotFoundError(f"Expected a complete model snapshot at {CHECKPOINT_DIR}")
print("Input:", REQUEST_PATH)
```

## 3. Prepare a native request

Choose `MODE` above. Preparation keeps the published camera keys/order, uses existing caption files,
and resolves paths before writing a request beside the outputs. Native joint inference reads
`daytime_scene/lidar/control.tar` directly; it is not an MP4 and needs no preconversion.

Sim-Auto uses **guidance 1, control guidance 1** and the checkpoint's **four-step distilled schedule**.
Do not copy Transfer-Auto's CFG/shift/35-step recipe. At guidance 1, a negative prompt adds no CFG effect.

The sample contains 300 RGB/WSM frames and 100 LiDAR sweeps; output intentionally uses
297 camera frames and 99 sweeps. Source FPS, model `fps`, captions and sensor timing must agree.

```python
request = native_request(
    REQUEST_PATH, mode=MODE,
    caption_folder=CAPTION_FOLDER if CUSTOM_REQUEST is None else None,
)
request["resolution"] = RESOLUTION
request["aspect_ratio"] = "16,9"
request.update(guidance=1, control_guidance=1, normalize_cfg=False, autoregressive=True)
# The distilled checkpoint supplies its four-step schedule.
for field in ("num_steps", "shift", "per_view_negative_prompt", "negative_prompt"):
    request.pop(field, None)
views = request["multiview"]["views"]
for view in views:
    info = probe_video(Path(view["control_path"]))
    if info["fps"] != request["fps"] or info["frames"] < request["num_frames"]:
        raise ValueError(f"Control timeline does not cover the request: {view['camera_key']}")
if request["multiview"]["condition_video_as_image"]:
    validate_rgb_conditions(views, REQUEST_PATH.parent)
if "lidar" in request and request["lidar"]["control_path"].endswith(".tar"):
    with tarfile.open(request["lidar"]["control_path"]) as archive:
        _, sweeps = index_sweeps(archive)
    expected_sweeps = round(request["num_frames"] * 10 / request["fps"])
    ids = list(sweeps)[:expected_sweeps]
    if ids != [index * round(request["fps"] / 10) for index in range(expected_sweeps)]:
        raise ValueError("LiDAR controls do not cover the requested camera timeline.")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
prepared_request = OUTPUT_ROOT / "request.json"
prepared_request.write_text(json.dumps(request, indent=2) + "\n")
print(f"{len(views)} cameras; {request['num_frames']} frames; {RESOLUTION}p; mode={MODE}")
print("Prepared request:", prepared_request)
```

## 4. Generate

Run this cell to start GPU inference. A fresh output directory prevents an earlier result from
being mistaken for a new run. Compilation and diffusion caching are disabled for this explicit
baseline. Guardrails remain enabled; do not bypass input rejections without reviewing the input.
Monitor the process output for model-loading, sampling and export errors.

```python
generated_dir = OUTPUT_ROOT / "generated"
command = [
    str(FRAMEWORK_PYTHON), "-m", "torch.distributed.run", "--standalone",
    f"--nproc-per-node={NUM_GPUS}", "--module", "cosmos_framework.scripts.inference",
    "-i", str(prepared_request), "-o", str(generated_dir),
    "--checkpoint-path", str(CHECKPOINT_DIR),
    "--cp-size", str(NUM_GPUS), "--cfgp-size", "1", "--dp-shard-size", "1", "--dp-replicate-size", "1",
    "--no-use-torch-compile", "--no-use-cuda-graphs", "--no-diffusion-cache",
]
run_env = dict(os.environ, COSMOS_INTERNAL="0", COSMOS_TRAINING="0", TOKENIZERS_PARALLELISM="false")
print(shlex.join(command))
subprocess.run(command, cwd=FRAMEWORK_DIR, env=run_env, check=True)
```

## 5. Inspect the result

Require `sample_outputs.json` success and one video per selected camera. At the default settings,
expect **seven 1280×720 videos, 297 frames at 30 FPS**, and, for joint generation,
`lidar.safetensors` containing finite `range_m` and `intensity` arrays of shape
`[99,128,1800]` and `frame_ids = 0,3,...,294`. At 480p, videos are 832×480.

Inspect every view for control alignment, cross-view appearance and motion consistency,
morphing, exposure differences and temporal artifacts. Successful decoding is not a quality score.

```python
outputs = validate_sim_auto_outputs(generated_dir, request)
print("Generated cameras:", len(outputs["cameras"]))
display(HTML(camera_grid(camera_previews(outputs["cameras"], OUTPUT_ROOT / "preview"))))
if "lidar" in outputs:
    print("LiDAR:", outputs["lidar"], "sweeps:", outputs["lidar_sweeps"])
    import matplotlib.pyplot as plt
    from safetensors import safe_open

    with safe_open(outputs["lidar"], framework="numpy") as handle:
        ranges = handle.get_slice("range_m")[0]  # [128,1800]
        intensity = handle.get_slice("intensity")[0]  # [128,1800]
    figure, axes = plt.subplots(2, 1, figsize=(14, 5))
    axes[0].imshow(ranges, aspect="auto", vmin=5, vmax=100)
    axes[0].set_title("First generated LiDAR sweep — range (m)")
    axes[1].imshow(intensity, aspect="auto", cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Intensity")
    plt.show()
```

## Use your own inputs

Set `CUSTOM_REQUEST` to one native JSON/JSONL request and rerun configuration with a fresh output path.
Use the published sample as the format reference. Keep synchronized WSM/RGB, camera-local caption files,
and numerical LiDAR controls together. Preparation preserves custom caption paths; do not apply the
sample's caption-folder selector to unrelated data.

The learned camera identity comes from `camera_key`, not list position or filename.
Use the [Camera Rig Explorer](../scripts/wsm/docs/camera-rig/guide.md) and
[input-preparation guide](input-preparation.md).
A custom rig may perform worse; matching the nearest reference camera is only an initial approximation.
For adaptation, post-train Transfer-Auto first, then distill into Sim-Auto.

To render your own controls, use [Generate WSM](generate-wsm.md) with your RDS-HQ scene.
Do not pair retargeted controls with unchanged source RGB from a different camera geometry.
For troubleshooting and longer runs, follow the model's inference skill linked above.
