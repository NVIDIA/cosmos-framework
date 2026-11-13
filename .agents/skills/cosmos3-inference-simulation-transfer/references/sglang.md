<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Sim-Transfer with SGLang

Cosmos3-Nano-Sim-Transfer generates a **480p video from a caption and a control video**. Each sample uses one of four control modalities:

- **Depth** guides scene geometry and depth structure.
- **Edge** guides object boundaries and outlines.
- **Blur** guides coarse appearance, layout, and motion.
- **Segmentation** (`seg`) guides the layout of regions in the scene.

The caption describes the desired scene and appearance, while the control video guides its spatial structure and motion. The distilled causal model generates latents autoregressively with **four denoising steps**, retaining control and generated-latent K/V history across the rollout.

This notebook runs the four published HF `example2` controls of the [Transfer cookbook](transfer-guide.md) through one SGLang server at **121 frames, 832 × 480, and 30 FPS**, with the saved `example2` prompt. It checks each output's frame count, dimensions, and FPS, then plays the generated videos.

| Component                          | Version                                                                                                                                                                                                                                                                               |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SGLang                             | The published [`lmsysorg/sglang:dev`](https://hub.docker.com/r/lmsysorg/sglang/tags) image, which carries the Cosmos Dreams Transfer pipeline from [sgl-project/sglang#39475](https://github.com/sgl-project/sglang/pull/39475); the tag moves, so pull it before starting the server |
| `nvidia/Cosmos3-Nano-Sim-Transfer` | [`main`](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer/tree/main)                                                                                                                                                                                                           |
| `example2` sample                  | [`main`](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer/tree/main/assets/sample_data/example2) of the same repository, under `assets/sample_data/example2`                                                                                                                   |

> **Skill note (not in the source):** this file is the cookbook notebook `run_transfer_with_sglang.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/simulation/transfer/run_transfer_with_sglang.ipynb`, cosmos-framework `main` 45fecab) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced as `python`, outputs omitted (the committed notebook has none). Edits are limited to three kinds: link targets that pointed at files this migration moved were re-pointed; path anchors in code that the move invalidated were replaced, each marked in place with a `Skill note (not in the source)` comment; and location text that named the retired cookbook folder (default output folders, where to run from) now names this skill's folder, each marked by a skill note. The changes are recorded in `docs/cookbook_to_skill_migration.md`.

## Shared setup

Install the notebook dependencies, and authenticate to Hugging Face with read access to `nvidia/Cosmos3-Nano-Sim-Transfer`, which publishes both the checkpoint and the sample; run `hf auth login` or set `HF_TOKEN`:

```bash
pip install requests huggingface_hub matplotlib pillow ipython
```

Install FFmpeg on the client host; the package also includes FFprobe.

**Inputs:** the notebook downloads the saved `example2` prompt and its four precomputed controls from the same Hugging Face repository as the model. It writes them under `cosmos3-inference-simulation-transfer/outputs/sglang/transfer/inputs`, inside the cookbook root, because the server does not receive uploads: a Sim-Transfer request names each control by absolute `control_path`, and the Docker command below mounts the cookbook root at its host path. Run the notebook from that checkout on the server host, or use the same absolute checkout path on both hosts. Later runs reuse the downloaded files.

> **Skill note (not in the source):** the folder named above was `simulation/outputs/sglang/transfer/inputs` in the source; the default output folder moved into this skill with the cookbook.

```python
from fractions import Fraction
from pathlib import Path
import json
import os
import shutil
import subprocess
import time

import matplotlib.pyplot as plt
import requests
from IPython.display import Video, display

MODEL_ID = "nvidia/Cosmos3-Nano-Sim-Transfer"
MODEL_REVISION = "main"
ASSET_REVISION = "main"
BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000").rstrip("/")
assert shutil.which("ffmpeg") and shutil.which("ffprobe"), "Install FFmpeg and ffprobe."
root_override = os.environ.get("COSMOS_COOKBOOK_ROOT")
candidates = [Path(root_override)] if root_override else [Path.cwd(), *Path.cwd().parents]
# Skill note (not in the source): the guide transfer.md is now this skill's references/transfer-guide.md; the marker was "cookbooks/cosmos3/simulation/transfer/transfer.md".
COOKBOOK_ROOT = next((p for p in candidates if (p / "cosmos3-inference-simulation-transfer/references/transfer-guide.md").is_file()), None)
assert COOKBOOK_ROOT, "Run inside .agents/skills or set COSMOS_COOKBOOK_ROOT to it."
COOKBOOK_ROOT = COOKBOOK_ROOT.resolve()
# Skill note (not in the source): the outputs folder is now inside this skill's folder; the source path was COOKBOOK_ROOT / "cookbooks/cosmos3/simulation".
SIM_ROOT = COOKBOOK_ROOT / "cosmos3-inference-simulation-transfer"
OUTPUT_DIR = Path(os.environ.get("TRANSFER_OUTPUT_DIR", SIM_ROOT / "outputs" / "sglang" / "transfer")).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Cookbook root:", COOKBOOK_ROOT)
print("Checkpoint:", MODEL_ID, "@", MODEL_REVISION)
print("Inputs:", MODEL_ID, "@", ASSET_REVISION)
print("Outputs:", OUTPUT_DIR)
if not OUTPUT_DIR.is_relative_to(COOKBOOK_ROOT):
    print(f"Note: {OUTPUT_DIR} is outside the cookbook root; the server must see it at the same path.")
```

### Start the server

The server runs in the published SGLang image, `lmsysorg/sglang:dev`, and loads the exported `nvidia/Cosmos3-Nano-Sim-Transfer` artifact from `main`, matching `MODEL_REVISION` above. One server runs all four examples. `SGLANG_HF_CACHE` is a host directory for the Hugging Face cache, and `HF_TOKEN` needs read access to the checkpoint. The container runs as root by default, so the setup lines create that directory world-writable; alternatively, start the container with your host user ID. Run this in a terminal and leave it running while using the notebook:

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
  --model-path nvidia/Cosmos3-Nano-Sim-Transfer \
  --revision main \
  --host 0.0.0.0
```

The `dev` tag moves with SGLang's main branch, so pull it before each start. This notebook was validated against the image built from SGLang commit [`ab580b2e`](https://github.com/sgl-project/sglang/commit/ab580b2e060059c0f8387b598a9f648dcec13f93), published as the fixed tag `lmsysorg/sglang:nightly-dev-20261020-ab580b2e`; use that tag in place of `dev` to reproduce the validated run.

The first start downloads the checkpoint into the mounted cache. The HTTP port opens only after the weights are loaded and the warmup requests finish, so wait for the `Uvicorn running` line. The container mounts the cookbook root (`.agents/skills`) at its host path because the notebook sends each control video as a server-visible `control_path` rather than uploading it; the downloaded `example2` controls live under that root in the output folder. The next cell waits until `/v1/models` lists the model.

```python
deadline = time.monotonic() + 1800
while True:
    try:
        response = requests.get(f"{BASE_URL}/v1/models", timeout=10)
        response.raise_for_status()
        models = response.json()
        break
    except requests.RequestException:
        if time.monotonic() >= deadline:
            raise TimeoutError("Model server did not become ready; inspect the SGLang container log.")
        time.sleep(3)
assert MODEL_ID in [m["id"] for m in models["data"]], models
print("Server ready:", BASE_URL)
```

## Prepare the input

The four precomputed `example2` controls and the saved appearance prompt come from `nvidia/Cosmos3-Nano-Sim-Transfer`, under `assets/sample_data/example2`. Each request uses **121 frames, 832 × 480, 30 FPS, and seed 2025**.

| Control      | Precomputed HF asset        | What it preserves                     |
| ------------ | --------------------------- | ------------------------------------- |
| Depth        | `example2_depth.mp4`        | Scene geometry and depth structure    |
| Edge         | `example2_edge.mp4`         | Object boundaries and outlines        |
| Blur         | `example2_blur.mp4`         | Coarse appearance, layout, and motion |
| Segmentation | `example2_segmentation.mp4` | Semantic region layout                |

The next cell downloads the files under the output folder, compacts the saved prompt into one JSON string, and probes every control for its frame count and rate. To derive edge or blur controls from your own RGB video instead, follow the [Cosmos Framework workflow](transfer-guide.md#prepare-the-input).

```python
from huggingface_hub import snapshot_download

HINTS = ("depth", "edge", "blur", "seg")
NUM_FRAMES, FPS, WIDTH, HEIGHT, SEED = 121, 30, 832, 480, 2025
FILES = {"depth": "example2_depth.mp4", "edge": "example2_edge.mp4", "blur": "example2_blur.mp4", "seg": "example2_segmentation.mp4"}


def require_file(path: Path) -> Path:
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty input: {path}")
    return path


def probe_video(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(path)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)["streams"][0]


sample_root = Path(snapshot_download(
    MODEL_ID, revision=ASSET_REVISION, allow_patterns=["assets/sample_data/example2/*"], local_dir=OUTPUT_DIR / "inputs",
)) / "assets/sample_data/example2"
prompt = json.dumps(json.loads(require_file(sample_root / "example2_upsampled_prompt.json").read_text()))
cases = {}
for hint in HINTS:
    control = require_file(sample_root / FILES[hint])
    metadata = probe_video(control)
    assert int(metadata["nb_read_frames"]) >= NUM_FRAMES, metadata
    assert abs(float(Fraction(metadata["avg_frame_rate"])) - FPS) < 0.1, "Use controls at the recipe FPS (29.97/30 accepted)."
    cases[hint] = dict(name=f"example2_{hint}", hint=hint, control=control, prompt=prompt, num_frames=NUM_FRAMES,
                       width=WIDTH, height=HEIGHT, fps=FPS, seed=SEED, input_metadata=metadata)
    print(f"{hint}: {NUM_FRAMES} frames, {WIDTH}×{HEIGHT}, {FPS} FPS, seed {SEED}; control {metadata['width']}×{metadata['height']}, "
          f"{metadata['nb_read_frames']} source frames; prompt {len(prompt):,} characters")
```

### Preview controls

First, middle, and final requested frames of each precomputed control.

```python
def read_frame(path, frame_index):
    from io import BytesIO
    from PIL import Image
    result = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-vf", rf"select=eq(n\,{frame_index})", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
    ], check=True, capture_output=True)
    return Image.open(BytesIO(result.stdout)).convert("RGB")

fig, axes = plt.subplots(len(cases), 3, figsize=(12, 11), layout="constrained")
for row, case in zip(axes, cases.values()):
    for ax, frame_index in zip(row, (0, case["num_frames"] // 2, case["num_frames"] - 1)):
        ax.imshow(read_frame(case["control"], frame_index))
        ax.set_title(f"{case['hint']} — frame {frame_index}")
        ax.axis("off")
display(fig)
plt.close(fig)
```

## Generate the four examples

Generate a video from each caption and control pair using four denoising steps, with prompt emphasis on. Each example is one asynchronous job that saves an MP4 in `OUTPUT_DIR`.

The notebook submits each job to `POST /v1/videos`, polls `GET /v1/videos/{id}` every five seconds, and downloads `GET /v1/videos/{id}/content` once the job completes. `GENERATION_TIMEOUT` limits the wait to 30 minutes per example. The latest job response is saved beside each video as `<control>.job.json`; failed or cancelled jobs and HTTP errors show their error details.

```python
GENERATION_TIMEOUT = 1800  # Maximum job wait per example, in seconds.
POLL_INTERVAL = 10

outputs = {}
elapsed_seconds = {}
for name, case in cases.items():
    extra_params = dict(emphasize_control_in_prompt=True)
    form = {
        "model": MODEL_ID, "prompt": case["prompt"],
        "size": f"{case['width']}x{case['height']}",
        "num_frames": str(case["num_frames"]), "fps": str(case["fps"]),
        "seed": str(case["seed"]), "num_inference_steps": "4",
        "guidance_scale": "1.0",
        "control_hint": case["hint"],
        "control_path": str(case["control"]),
        "extra_params": json.dumps(extra_params),
    }
    started = time.monotonic()
    response = requests.post(
        f"{BASE_URL}/v1/videos",
        files=[(key, (None, value)) for key, value in form.items()],
        headers={"Accept": "application/json"}, timeout=60,
    )
    if not response.ok:
        raise RuntimeError(f"{name}: submission HTTP {response.status_code}\n{response.text[:4000]}")
    job = response.json()
    job_id = job.get("id")
    if not job_id:
        raise RuntimeError(f"{name}: no video job ID in response: {job}")
    job_path = OUTPUT_DIR / f"{name}.job.json"
    deadline = time.monotonic() + GENERATION_TIMEOUT
    while True:
        job_path.write_text(json.dumps(job, indent=2))
        status = job.get("status")
        print(f"{name}: {job_id} — {status} ({job.get('progress', 0)}%)")
        if status == "completed":
            break
        if status in {"failed", "cancelled", "canceled"}:
            raise RuntimeError(f"{name}: generation {status}\n{json.dumps(job, indent=2)}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"{name}: timed out waiting for job {job_id}; see {job_path}")
        time.sleep(min(POLL_INTERVAL, remaining))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"{name}: timed out waiting for job {job_id}; see {job_path}")
        response = requests.get(
            f"{BASE_URL}/v1/videos/{job_id}", timeout=min(30, remaining),
        )
        if not response.ok:
            raise RuntimeError(f"{name}: polling HTTP {response.status_code}\n{response.text[:4000]}")
        job = response.json()

    response = requests.get(
        f"{BASE_URL}/v1/videos/{job_id}/content",
        headers={"Accept": "video/mp4"}, timeout=300,
    )
    if not response.ok:
        raise RuntimeError(f"{name}: download HTTP {response.status_code}\n{response.text[:4000]}")
    if not response.headers.get("content-type", "").startswith("video/mp4") or not response.content:
        raise RuntimeError(f"{name}: expected a nonempty MP4 response, got {response.headers}")
    output = OUTPUT_DIR / f"{name}.mp4"
    temporary_output = output.with_suffix(".mp4.part")
    try:
        temporary_output.write_bytes(response.content)
        temporary_output.replace(output)
    finally:
        temporary_output.unlink(missing_ok=True)
    outputs[name] = output
    elapsed_seconds[name] = time.monotonic() - started
    print(f"{name}: {elapsed_seconds[name]:.1f}s — {output}")
```

## View generated videos

Play the depth, edge, blur, and segmentation results below.

```python
for name, output in outputs.items():
    print(name)
    display(Video(str(output), embed=True, width=640, html_attributes="controls loop"))
```
