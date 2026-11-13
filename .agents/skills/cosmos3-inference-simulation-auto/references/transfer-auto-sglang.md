<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3-Nano-Transfer-Auto with SGLang

Cosmos3-Nano-Transfer-Auto generates a **synchronized multi-camera driving clip** from one World Scenario Model (WSM) control video per camera. The WSM controls render the scene's lane lines, road surface, and vehicle boxes from each camera's viewpoint, and a caption per camera describes its appearance. Joint requests also condition every camera on a first RGB frame and on a numeric LiDAR range map, and **generate numeric LiDAR** alongside the cameras.

This notebook runs the two examples of the [Transfer-Auto guide](transfer-auto-guide.md) on the shared **daytime sample**, one scene recorded by seven cameras, through one SGLang server. Each runs at **201 frames per camera, 30 FPS, and 832 × 480**:

1. **WSM-only:** seven WSM controls produce seven camera videos.
2. **Joint RGB and LiDAR:** seven WSM controls, seven first-frame RGB conditions, and a numeric HD-map LiDAR control produce seven camera videos and 67 numeric LiDAR sweeps.

Both run as asynchronous Videos API jobs. SGLang opens the inputs from paths the server can read, so the notebook writes every prepared input under its output folder inside the mounted cookbook root and uploads nothing. The examples are independent: run **Shared setup**, then either one.

| Component                           | Version                                                                                                                                                                                                                                                                             |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SGLang                              | The published [`lmsysorg/sglang:dev`](https://hub.docker.com/r/lmsysorg/sglang/tags) image, which carries the Cosmos3 Multiview-AV pipeline from [sgl-project/sglang#39475](https://github.com/sgl-project/sglang/pull/39475); the tag moves, so pull it before starting the server |
| `nvidia/Cosmos3-Nano-Transfer-Auto` | [`main`](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto/tree/main)                                                                                                                                                                                                        |
| Daytime sample                      | [`main`](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto/tree/main/assets/sample_data) of the same repository, under `assets/sample_data`                                                                                                                                  |

> **Skill note (not in the source):** this file is the cookbook notebook `run_auto_with_sglang.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/simulation/auto/run_auto_with_sglang.ipynb`, cosmos-framework `main` 45fecab) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced as `python`, outputs omitted (the committed notebook has none). Edits are limited to three kinds: link targets that pointed at files this migration moved were re-pointed; path anchors in code that the move invalidated were replaced, each marked in place with a `Skill note (not in the source)` comment; and location text that named the retired cookbook folder (default output folders, where to run from) now names this skill's folder, each marked by a skill note. The changes are recorded in `docs/cookbook_to_skill_migration.md`.

## 1. Prerequisites

**Server host:** Linux with Docker, the NVIDIA Container Toolkit, and at least one NVIDIA GPU. `HF_TOKEN` needs read access to `nvidia/Cosmos3-Nano-Transfer-Auto`. The server opens the inputs by path, so the cookbook checkout, including the output folder this notebook writes, must be on the server host or on a filesystem it shares with this notebook, at one absolute path.

**Notebook client:** Python 3.10 or newer, and a Hugging Face token with read access to `nvidia/Cosmos3-Nano-Transfer-Auto`, which also publishes the sample; run `hf auth login` or set `HF_TOKEN`. In the notebook kernel environment, install:

```bash
pip install httpx pillow numpy safetensors matplotlib ipython huggingface_hub
```

Install FFmpeg with the `libx264` encoder on the client host; the package also includes FFprobe. On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

On macOS with Homebrew, use `brew install ffmpeg`. Verify that both `ffmpeg -version` and `ffprobe -version` work.

**Inputs:** The examples use the shared daytime sample published with the model under `assets/sample_data`: one scene with seven synchronized cameras, each with an RGB recording, a WSM control, and a caption, plus a sparse HD-map LiDAR control archive. Step 3 downloads the files these examples need, about 100 MiB, into the Hugging Face cache and checks them against the sample's `SHA256SUMS`. To use a copy you already have, set `AUTO_SAMPLE_ROOT` to its `sample_data` folder. The joint preparation cell converts `daytime_scene/lidar/control.tar` into a local `prepared/lidar_control.safetensors` under the output folder; see the [format notes](../assets/daytime_scene/README.md).

## 2. Start the server

The server runs in the published SGLang image, `lmsysorg/sglang:dev`, which includes the Cosmos3 Multiview-AV pipeline, the joint LiDAR encoder and decoder, and the asynchronous numeric-output API. On the server host, set `COSMOS_ROOT` to the absolute cookbook root (`.agents/skills`), which step 3 prints as `Cookbook root`. `SGLANG_HF_CACHE` is a host directory for the Hugging Face cache. The container runs as root by default, so the setup lines create that directory world-writable; alternatively, start the container with your host user ID. Pull the image, then run the server in a terminal and leave it running while using the notebook; both examples use it:

```bash
docker pull lmsysorg/sglang:dev
export COSMOS_ROOT=/absolute/path/to/cosmos-framework/.agents/skills
export SGLANG_HF_CACHE="${SGLANG_HF_CACHE:-$HOME/.cache/sglang-huggingface}"
mkdir -p "$SGLANG_HF_CACHE"
chmod 777 "$SGLANG_HF_CACHE"
docker run --init --runtime nvidia --gpus all \
  -v $SGLANG_HF_CACHE:/root/.cache/huggingface \
  -v "${COSMOS_ROOT}:${COSMOS_ROOT}" \
  -e HF_TOKEN=$HF_TOKEN \
  -p 30000:30000 \
  --ipc=host \
  lmsysorg/sglang:dev \
  sglang serve \
  --model-path nvidia/Cosmos3-Nano-Transfer-Auto \
  --revision main \
  --host 0.0.0.0 --port 30000
```

| Setting                                        | Effect                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-v "${COSMOS_ROOT}:${COSMOS_ROOT}"`           | Mounts the cookbook root at its host path. The requests name every prepared WSM control, staged RGB condition, and converted LiDAR control by absolute path under the output folder, which must be visible at the same path.                                                                                                                                                                                                                                                                                                                |
| `-v $SGLANG_HF_CACHE:/root/.cache/huggingface` | Keeps the downloaded checkpoint across container restarts. The setup lines default the directory to `~/.cache/sglang-huggingface`, create it, and make it world-writable.                                                                                                                                                                                                                                                                                                                                                                   |
| `-e HF_TOKEN=$HF_TOKEN`                        | Read access to the gated checkpoint.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `--revision`                                   | Loads the checkpoint from `main`. The first start downloads it into the mounted cache.                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `docker pull`                                  | The `dev` tag moves with SGLang's main branch; pulling before each start picks up the current image. This notebook was validated against the image built from SGLang commit [`ab580b2e`](https://github.com/sgl-project/sglang/commit/ab580b2e060059c0f8387b598a9f648dcec13f93), published as the fixed tag `lmsysorg/sglang:nightly-dev-20261020-ab580b2e`; use that tag in place of `dev` to reproduce the validated run.                                                                                                                 |
| `--gpus all`                                   | Makes every GPU visible; select one with `--gpus '"device=0"'`. With N GPUs, add `--num-gpus N --ulysses-degree N --sp-degree N` to `sglang serve`: Ulysses sequence parallelism splits the camera-major sequence across the GPUs and speeds up generation. The Multiview pipeline supports no tensor or ring parallelism, so pass both degrees explicitly, and N must divide its 8 key/value heads (2, 4, or 8). `--enable-cfg-parallel` additionally runs the two guidance branches on separate GPU groups and doubles the GPUs required. |
| `--host 0.0.0.0`                               | Accepts connections from outside the container. For a remote notebook, set `SGLANG_BASE_URL`, or forward the port with `ssh -L 30000:127.0.0.1:30000 <server-host>`.                                                                                                                                                                                                                                                                                                                                                                        |

The camera attention backend is selected automatically: FlashAttention-4 block-sparse kernels on Hopper and Blackwell, FlexAttention's Triton kernels elsewhere; `SGLANG_DIFFUSION_COSMOS3_MULTIVIEW_ATTENTION_BACKEND` (`triton` or `fa4`) overrides the choice. Joint LiDAR generation requires the checkpoint's `lidar_vae/` component; the LiDAR VAE runs on SGLang's built-in FlexAttention kernel, and `SGLANG_DIFFUSION_COSMOS3_LIDAR_USE_NATTEN=1` switches to `natten` when that package is installed in the image. The HTTP port opens only after the weights are loaded and the warmup request finishes, so wait for the `Uvicorn running` line.

## 3. Shared setup

Run the notebook from anywhere inside `.agents/skills`, or set `COSMOS_COOKBOOK_ROOT` to it. These optional environment variables override the defaults:

| Variable             | Default                                                                | Purpose                                                                                                                                           |
| -------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SGLANG_BASE_URL`    | `http://127.0.0.1:30000`                                               | Server address                                                                                                                                    |
| `SGLANG_API_KEY`     | unset                                                                  | Bearer token, if the server was started with `--api-key`                                                                                          |
| `COSMOS3_OUTPUT_DIR` | `.agents/skills/cosmos3-inference-simulation-auto/outputs/sglang/auto` | Output folder; each example writes to its own subfolder. Keep it inside the mounted cookbook root: the server reads every prepared input from it. |
| `AUTO_SAMPLE_ROOT`   | unset                                                                  | A local `sample_data` folder to use instead of downloading the sample                                                                             |

> **Skill note (not in the source):** the `COSMOS3_OUTPUT_DIR` default in the table above was `.agents/skills/cookbooks/cosmos3/simulation/outputs/sglang/auto` in the source; the default output folder moved into this skill with the cookbook.

The next cell sets the configuration and imports the CPU-only request, media, and HTTP helpers from [`helpers/auto_helpers.py`](../scripts/helpers/auto_helpers.py) and the sample helpers from [`helpers/daytime_sample.py`](../scripts/helpers/daytime_sample.py); none needs GPU libraries. Then fetch the sample and check the server.

```python
from pathlib import Path
import copy
import json
import os
import sys
import time

import httpx
from IPython.display import HTML, display


def find_auto_root() -> Path:
    configured = os.environ.get("COSMOS_COOKBOOK_ROOT")
    start = Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()
    for root in [start] if configured else [start, *start.parents]:
        # Skill note (not in the source): the Auto cookbook folder is now this skill's folder; the candidates were root / "auto", root / "cookbooks" / "cosmos3" / "simulation" / "auto", root / ".agents" / "skills" / "cookbooks" / "cosmos3" / "simulation" / "auto".
        for candidate in (root, root / "cosmos3-inference-simulation-auto", root / ".agents" / "skills" / "cosmos3-inference-simulation-auto"):
            # Skill note (not in the source): the helpers are now in this skill's scripts/helpers folder; the marker was candidate / "helpers" / "auto_helpers.py".
            if (candidate / "scripts" / "helpers" / "auto_helpers.py").is_file():
                return candidate
    raise FileNotFoundError("Run inside .agents/skills or set COSMOS_COOKBOOK_ROOT to it.")


AUTO_ROOT = find_auto_root()
# Skill note (not in the source): the Simulation cookbook folder is retired and this skill's folder takes its place, so the default outputs below land in AUTO_ROOT / "outputs"; this was SIMULATION_ROOT = AUTO_ROOT.parent.
SIMULATION_ROOT = AUTO_ROOT
# Skill note (not in the source): .agents/skills is now the parent of the skill folder; this was SIMULATION_ROOT.parents[2].
COOKBOOK_ROOT = SIMULATION_ROOT.parent  # The .agents/skills folder the server mounts.
# Skill note (not in the source): the helpers package is now under this skill's scripts folder; this line and the next used str(AUTO_ROOT).
if str(AUTO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(AUTO_ROOT / "scripts"))

from helpers.auto_helpers import (
    camera_grid,
    check_model,
    prepare_controls,
    prepare_request,
    probe_video,
    save_outputs,
    split_cameras,
    stage_conditions,
    submit_joint,
    submit_wsm,
    validate_lidar,
    validate_rgb_conditions,
    validate_sglang_lidar,
)
from helpers.daytime_sample import SAMPLE_CAMERAS, SAMPLE_REPO, SAMPLE_REVISION, fetch_sample, resolve_request

MODEL_ID = "nvidia/Cosmos3-Nano-Transfer-Auto"
MODEL_REVISION = "main"
SGLANG_BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000").rstrip("/")
SGLANG_API_KEY = os.environ.get("SGLANG_API_KEY")
HEADERS = {"Authorization": f"Bearer {SGLANG_API_KEY}"} if SGLANG_API_KEY else {}
HTTP_TIMEOUT = 600  # Per HTTP request: submission, each poll, and each download.
GENERATION_TIMEOUT = 3600  # Job wait, from submission to completion.
OUTPUT_DIR = Path(
    os.environ.get("COSMOS3_OUTPUT_DIR", SIMULATION_ROOT / "outputs" / "sglang" / "auto")
).expanduser().resolve()
AUTO_SAMPLE_ROOT = os.environ.get("AUTO_SAMPLE_ROOT")  # Unset: download the sample from Hugging Face
CAPTION_FOLDER = "captions_with_timestamp"  # Alternative: "captions"
# Skill note (not in the source): the Transfer-Auto negative prompt is now this skill's assets/negative-prompt.txt; this was COOKBOOK_ROOT / "cosmos3-nano-transfer-auto" / "inference" / "assets" / "negative-prompt.txt".
NEGATIVE_PROMPT_PATH = AUTO_ROOT / "assets" / "negative-prompt.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Cookbook root:", COOKBOOK_ROOT)
print("Server:", SGLANG_BASE_URL, "(with API key)" if SGLANG_API_KEY else "")
print("Model:", MODEL_ID, "@", MODEL_REVISION)
print("Sample:", AUTO_SAMPLE_ROOT or f"{SAMPLE_REPO} @ {SAMPLE_REVISION}")
print("Outputs:", OUTPUT_DIR)
if not OUTPUT_DIR.is_relative_to(COOKBOOK_ROOT):
    print(f"Note: {OUTPUT_DIR} is outside the cookbook root; the server must see it at the same path.")
```

### Download and verify the sample

Download the seven cameras' RGB recordings, WSM controls, both caption styles, and the HD-map LiDAR control archive from `assets/sample_data` on Hugging Face, and verify them against `SHA256SUMS`; later runs reuse the Hugging Face cache. No model weights are downloaded here.

`CAPTION_FOLDER="captions_with_timestamp"` selects timestamped captions, which better match training and empirically improve inference. Set it to `"captions"` for the original dense paragraphs.

```python
SAMPLE_ROOT = fetch_sample(AUTO_SAMPLE_ROOT)
print("Verified sample:", SAMPLE_ROOT)
print("Cameras:", ", ".join(SAMPLE_CAMERAS))
```

### Verify the server

Wait for the server to finish starting, then confirm that it serves the Transfer-Auto model. The cell retries connection errors and server errors for up to 30 minutes, so you can run it while the server is still loading. `/v1/models` reports the model name, not its revision; the `--revision` in the server command pins that.

```python
deadline = time.monotonic() + 1800
while True:
    try:
        with httpx.Client(base_url=SGLANG_BASE_URL, headers=HEADERS, timeout=30) as client:
            served = check_model(client, MODEL_ID)
        break
    except httpx.TransportError as error:
        reason = type(error).__name__
    except httpx.HTTPStatusError as error:
        if error.response.status_code < 500:
            raise
        reason = f"HTTP {error.response.status_code}"
    if time.monotonic() > deadline:
        raise TimeoutError(f"No ready server at {SGLANG_BASE_URL} after 1800s ({reason}).")
    time.sleep(5)
print("Server ready; serving", served)
```

## 4. WSM-only example

### Prepare the inputs

The [WSM request](../assets/daytime_scene/transfer_auto_wsm.json) asks for **201 frames per camera at 30 FPS** with `resolution: "480"` and `aspect_ratio: "16,9"`, seed 0, 35 steps, guidance 3, and flow shift 10, matching the recommended Transfer-Auto baseline. It lists the sample's seven cameras in the checkpoint's camera order, each with its WSM control and caption file. The server identifies each camera by its `camera_key`, so the seven keep their trained camera IDs 0 to 6; the generated video follows the request order.

`resolve_request` resolves the paths under the sample and replaces each caption file with its text, which covers all 300 source frames. The Videos API also takes a top-level caption; the request copies the front wide camera's. The notebook adds the inference skill's recommended `per_view_negative_prompt`. The runtime adds camera labels and control emphasis.

Every control is 1280 × 720 with 300 frames at 30 FPS. The next cell decodes every control, checks that their timelines are synchronized, and trims their first 201 frames without resizing into the output folder; controls that already have 201 frames are copied there unchanged. `prepare_request(..., backend="sglang")` keeps each prepared control's absolute path in the JSON-encoded `extra_params` and returns no uploads; SGLang does not take `control_reference_index` fields.

```python
WSM_REQUEST = AUTO_ROOT / "assets" / "daytime_scene" / "transfer_auto_wsm.json"
WSM_OUTPUT_DIR = OUTPUT_DIR / "wsm"

wsm_manifest = resolve_request(WSM_REQUEST, SAMPLE_ROOT, caption_folder=CAPTION_FOLDER)
wsm_manifest["per_view_negative_prompt"] = NEGATIVE_PROMPT_PATH.read_text().strip()
wsm_views = wsm_manifest["multiview"]["views"]
assert tuple(view["camera_key"] for view in wsm_views) == SAMPLE_CAMERAS
assert wsm_manifest["model_mode"] == "text2video" and not any("vision_path" in view for view in wsm_views)
wsm_controls = prepare_controls(
    wsm_views, SAMPLE_ROOT, WSM_OUTPUT_DIR / "prepared_controls",
    wsm_manifest["num_frames"], wsm_manifest["fps"], materialize=True,
)
wsm_request = copy.deepcopy(wsm_manifest)
for view, control in zip(wsm_request["multiview"]["views"], wsm_controls, strict=True):
    view["control_path"] = str(control)
wsm_form, wsm_uploads = prepare_request(wsm_request, SAMPLE_ROOT, MODEL_ID, backend="sglang")
assert not wsm_uploads  # SGLang opens each control_path itself; nothing is uploaded.
assert all(Path(view["control_path"]).is_absolute() for view in json.loads(wsm_form["extra_params"])["multiview"]["views"])
WSM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(WSM_OUTPUT_DIR / "request.json").write_text(json.dumps(wsm_form, indent=2, ensure_ascii=False) + "\n")
for camera_key, control in zip(SAMPLE_CAMERAS, wsm_controls):
    info = probe_video(control)
    print(f"{camera_key:28}  {info['frames']} frames, {info['width']}×{info['height']}, {float(info['fps']):g} FPS")
```

### Generate

`submit_wsm(..., async_mode=True)` sends the form fields to `POST /v1/videos`, polls `GET /v1/videos/{id}` every ten seconds, and downloads the finished MP4 from `/content`. It saves the latest job status as `job.json`; a rejected submission or a failed job leaves `response_error.txt`, and a polling timeout leaves the job running on the server at the printed URL. The request carries the model, top-level caption, canvas `size`, `num_frames`, `fps`, `seed`, `num_inference_steps`, `guidance_scale`, and `flow_shift` from the request file, and in `extra_params` the `wsm: {}` hint, the `multiview.views` list with each camera's key, caption, and absolute `control_path`, the `per_view_negative_prompt`, and the `resolution` and `aspect_ratio`. The first job also compiles kernels, so it takes longer than later ones.

```python
with httpx.Client(base_url=SGLANG_BASE_URL, headers=HEADERS, timeout=HTTP_TIMEOUT) as client:
    started = time.monotonic()
    wsm_video = submit_wsm(client, wsm_form, wsm_uploads, WSM_OUTPUT_DIR, async_mode=True, timeout=GENERATION_TIMEOUT)
    wsm_elapsed = round(time.monotonic() - started, 1)
print(f"{wsm_elapsed:.1f}s  {wsm_video}")
```

### Split and view the camera videos

The result is one camera-major MP4: 201 frames of the first camera, then 201 of the second, through the seventh. `split_cameras` checks its `7 × 201 = 1407` frames, FPS, and canvas, then writes one labeled video per camera; `save_outputs` records them in `sample_outputs.json`. The folder then holds:

```text
outputs/sglang/auto/wsm/
  request.json                    # The exact form fields that were sent
  job.json                        # The last job response
  prepared_controls/              # The seven trimmed controls the server read
  multiview_camera_major.mp4      # The downloaded result
  vision_viewNN_<camera>.mp4      # One video per camera
  sample_outputs.json             # Camera order, frame count, FPS, and the per-camera files
```

Inspect the grid for control adherence, cross-camera consistency, and visual quality, and see [Validate the rollout](transfer-auto-guide.md#validate-the-rollout).

```python
wsm_cameras = split_cameras(wsm_video, WSM_OUTPUT_DIR, SAMPLE_CAMERAS, wsm_manifest["num_frames"], wsm_manifest["fps"])
save_outputs(WSM_OUTPUT_DIR, wsm_video, wsm_cameras, wsm_manifest)
display(HTML(camera_grid(wsm_cameras)))
```

## 5. Joint RGB and LiDAR example

### Prepare the inputs

The [joint request](../assets/daytime_scene/transfer_auto_joint.json) uses the same server, cameras, and captions and asks for the same **201 frames per camera at 30 FPS**, 832 × 480, seed 0, 35 steps, guidance 3, flow shift 10, and control guidance 1. Each camera adds the sample's RGB recording under `vision_path`; with `condition_video_as_image: true`, only its first decoded frame conditions generation. The request also adds a numeric LiDAR control with `return_output: true`, which the SGLang request carries as `lidar.decode: true`.

`resolve_request` converts the published `daytime_scene/lidar/control.tar` to a local `prepared/lidar_control.safetensors` under the output folder: FP32 `frames[3,100,128,1800]` with range in metres, unit intensity at valid returns, and validity. The 201-frame request consumes its first 67 sweeps at 10 Hz; the next cell checks them one at a time, and the server truncates the trailing sweeps and applies its own padding and normalization.

The cell then decodes each RGB condition's first frame, trims the WSM controls to 201 frames, or copies them unchanged when they already have 201 frames, and copies each RGB condition into the output folder, because SGLang opens inputs by path and the sample lives in this notebook's Hugging Face cache. `prepare_request(..., backend="sglang")` keeps every path absolute and returns no uploads.

```python
LIDAR_REQUEST = AUTO_ROOT / "assets" / "daytime_scene" / "transfer_auto_joint.json"
LIDAR_OUTPUT_DIR = OUTPUT_DIR / "lidar"
LIDAR_FPS = 10.0

lidar_manifest = resolve_request(
    LIDAR_REQUEST, SAMPLE_ROOT, caption_folder=CAPTION_FOLDER,
    lidar_output=LIDAR_OUTPUT_DIR / "prepared" / "lidar_control.safetensors",
)
lidar_manifest["per_view_negative_prompt"] = NEGATIVE_PROMPT_PATH.read_text().strip()
lidar_views = lidar_manifest["multiview"]["views"]
assert tuple(view["camera_key"] for view in lidar_views) == SAMPLE_CAMERAS
assert lidar_manifest["multiview"]["condition_video_as_image"] is True
assert lidar_manifest["lidar"]["return_output"] is True
assert lidar_manifest["resolution"] == "480" and lidar_manifest["aspect_ratio"] == "16,9"
lidar_sweeps = round(lidar_manifest["num_frames"] * LIDAR_FPS / lidar_manifest["fps"])

lidar_input = Path(lidar_manifest["lidar"]["control_path"])
lidar_input_info = validate_lidar(lidar_input, lidar_sweeps)
validate_rgb_conditions(lidar_views, SAMPLE_ROOT)
lidar_controls = prepare_controls(
    lidar_views, SAMPLE_ROOT, LIDAR_OUTPUT_DIR / "prepared_controls",
    lidar_manifest["num_frames"], lidar_manifest["fps"], materialize=True,
)
lidar_conditions = stage_conditions(lidar_views, LIDAR_OUTPUT_DIR / "staged_conditions")
lidar_request = copy.deepcopy(lidar_manifest)
for view, control, condition in zip(lidar_request["multiview"]["views"], lidar_controls, lidar_conditions, strict=True):
    view["control_path"] = str(control)
    view["vision_path"] = str(condition)
lidar_form, lidar_uploads = prepare_request(lidar_request, SAMPLE_ROOT, MODEL_ID, backend="sglang")
lidar_extra = json.loads(lidar_form["extra_params"])
assert not lidar_uploads
assert lidar_extra["lidar"] == {"control_path": str(lidar_input.resolve()), "decode": True}
assert all(
    "control_path" in view and "vision_path" in view and not any(key.endswith("_reference_index") for key in view)
    for view in lidar_extra["multiview"]["views"]
)
LIDAR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(LIDAR_OUTPUT_DIR / "request.json").write_text(json.dumps(lidar_form, indent=2, ensure_ascii=False) + "\n")
rgb_conditions = dict(zip(SAMPLE_CAMERAS, lidar_conditions))
print(f"LiDAR input: {lidar_input_info['shape']} {lidar_input_info['dtype']}; using {lidar_sweeps} sweeps at {LIDAR_FPS:g} Hz")
print(f"Prepared {len(lidar_controls)} WSM controls, {len(lidar_conditions)} RGB conditions, and the LiDAR control under {LIDAR_OUTPUT_DIR}")
```

### Submit the job and download both artifacts

`submit_joint(..., backend="sglang")` sends the form fields to `POST /v1/videos`, polls `GET /v1/videos/{id}` until the job completes, then downloads the camera-major MP4 from `/content` and the numeric LiDAR from `/content?variant=rangemap`, saving the job's LiDAR descriptor beside it as `multiview_camera_major.lidar.json`. The latest job status is saved as `job.json`; a rejected submission or a failed job leaves `response_error.txt`, and a polling timeout leaves the job on the server at the printed URL. Rerunning the cell submits a new job.

SGLang also accepts `lidar.decode: false` for LiDAR-conditioned camera generation without numeric output; this walkthrough keeps decoding enabled so the next cell can inspect the generated tensor.

```python
with httpx.Client(base_url=SGLANG_BASE_URL, headers=HEADERS, timeout=HTTP_TIMEOUT) as client:
    started = time.monotonic()
    lidar_result = submit_joint(client, lidar_form, lidar_uploads, LIDAR_OUTPUT_DIR, timeout=GENERATION_TIMEOUT, backend="sglang")
    lidar_elapsed = round(time.monotonic() - started, 1)
print(f"Job {lidar_result['job_id']} completed in {lidar_elapsed:.1f}s")
print("Camera video:", lidar_result["video"])
print("Generated LiDAR:", lidar_result["lidar"])
```

### Validate generated outputs and display cameras

The camera-major MP4 must contain `7 × 201 = 1407` frames. LiDAR uses its own 10 Hz timeline: 67 sweeps. SGLang writes three tensors of shape `[67,128,1800]`: FP32 `range_m`, FP16 `intensity`, and uint8 `validity`, with the layout, FPS, and range limits in the file's metadata. This differs from the input's single `frames` tensor.

`validate_sglang_lidar` checks the tensor shapes, dtypes, physical values, and metadata; the cell then compares them with the job descriptor, splits the camera videos, writes `sample_outputs.json`, and displays the camera grid. Valid shapes and values do not establish that the LiDAR agrees with the HD map or the cameras; inspect the generated sweeps before relying on them. The folder then holds:

```text
outputs/sglang/auto/lidar/
  request.json                                # The exact form fields that were sent
  job.json                                    # The last job response, including the LiDAR descriptor
  prepared/lidar_control.safetensors          # The converted LiDAR control the server read
  prepared_controls/                          # The seven trimmed WSM controls
  staged_conditions/                          # The seven RGB conditions, copied from the sample
  multiview_camera_major.mp4                  # The camera-major video from /content
  vision_viewNN_<camera>.mp4                  # One video per camera
  multiview_camera_major.lidar.safetensors    # The numeric LiDAR from /content?variant=rangemap
  multiview_camera_major.lidar.json           # Its descriptor
  sample_outputs.json                         # Camera order, files, job ID, and the LiDAR descriptor
```

```python
lidar_output_info = validate_sglang_lidar(lidar_result["lidar"], lidar_sweeps, fps=LIDAR_FPS)
lidar_descriptor = lidar_result["descriptor"]
assert lidar_descriptor["sweeps"] == lidar_sweeps
assert lidar_descriptor["fps"] == LIDAR_FPS
assert [lidar_descriptor["sweeps"], lidar_descriptor["height"], lidar_descriptor["width"]] == lidar_output_info["shape"]
for field in ("fps", "min_range_m", "max_range_m"):
    assert float(lidar_output_info["metadata"][field]) == float(lidar_descriptor[field]), field

lidar_cameras = split_cameras(lidar_result["video"], LIDAR_OUTPUT_DIR, SAMPLE_CAMERAS, lidar_manifest["num_frames"], lidar_manifest["fps"])
save_outputs(LIDAR_OUTPUT_DIR, lidar_result["video"], lidar_cameras, lidar_manifest, lidar=lidar_result)
print(f"LiDAR: {lidar_output_info['shape']} at {lidar_descriptor['fps']:g} Hz; split into {len(lidar_cameras)} cameras")
display(HTML(camera_grid(lidar_cameras)))
```

## 6. Next steps

- **Use your own scene:** copy a request file from [`assets/daytime_scene/`](../assets/daytime_scene), point its paths at your data, then set `WSM_REQUEST` or `LIDAR_REQUEST` to the copy and `SAMPLE_ROOT` to the folder its paths are relative to. List any of the checkpoint's eleven cameras in its camera order, `CAMERA_ORDER` in [`helpers/auto_helpers.py`](../scripts/helpers/auto_helpers.py), each with a WSM control and a caption file or inline `prompt`. For the joint example, add a `vision_path` per camera and a numeric LiDAR control; [`helpers/lidar_rangemap.py`](../scripts/helpers/lidar_rangemap.py) converts a sparse HD-map range-map archive. Every prepared input lands under the output folder, so keep it inside the mounted cookbook root.
- **Validate further:** apply the [shared Simulation checklist](../../cosmos3-inference-simulation/references/simulation-readme.md#validate-a-rollout) and the [Transfer-Auto checks](transfer-auto-guide.md#validate-the-rollout).
- **Troubleshooting:**
  - *Missing inputs:* rerun the sample download in step 3 and the example's **Prepare the inputs** cell; `AUTO_SAMPLE_ROOT` must point at a complete `sample_data` folder.
  - *HTTP error on submission:* inspect `response_error.txt` and the server log. A path the server cannot open means the checkout or the output folder is not mounted at the same absolute path inside the container.
  - *Job `failed`:* `job.json` and `response_error.txt` carry the server's error message; the server log has the traceback.
  - *Timeout:* the job keeps running on the server at the printed URL; `GET /v1/videos/{id}` reports its status. Raise `GENERATION_TIMEOUT` before resubmitting.
- **Other runtimes:** the [vLLM-Omni notebook](transfer-auto-vllm-omni.md) runs the same two examples with uploads, and the [Cosmos Framework notebook](transfer-auto-cosmos-framework.md) runs them natively from the published JSONL. For the autoregressive student of this model, see the [Sim-Auto SGLang notebook](sim-auto-sglang.md).
