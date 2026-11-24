<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Action: Policy with vLLM-Omni

## Prerequisites

Generator requires the Guardrail. Request access to the gated [nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail) HF repository before running guarded examples. When running with guardrails enabled, also set and pass `HF_TOKEN` to the container; it must be a Hugging Face token authorized for that repository. To disable guardrails server-wide, pass `--no-guardrails` to `vllm serve`; alternatively, set `guardrails: false` in vLLM-Omni `extra_params` or `extra_args` to disable them per request.

Running `nvidia/Cosmos3-Nano-Policy-DROID` also requires an up-to-date NumPy installation. Upgrade it in the environment that starts the model with `pip install --upgrade numpy`.

## Overview

This notebook demonstrates one-shot rollout generation through the asynchronous `/v1/videos` API for Cosmos3 Nano or Cosmos3-Edge using the checked-in LeRobot sample under `assets/droid_lerobot_example`. It returns a 17-frame rollout and a `[16, 8]` DROID joint-position action chunk.

> **Skill note (not in the source):** this file is the cookbook notebook `run_policy_with_vllm_omni.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/generator/action/run_policy_with_vllm_omni.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

## Start vLLM-Omni for One-Shot Policy Rollout

Choose either the Cosmos3-Nano or Cosmos3-Edge DROID checkpoint and start it in a terminal from the cookbook root (`.agents/skills`). Before running it, set `COSMOS3_REPO` to the root of a local checkout of the **cosmos-framework** repository:

```bash
export COSMOS3_REPO=/path/to/cosmos-framework
```

This must be the framework checkout that contains the `cosmos_framework/` Python package—not this `cosmos` cookbook repository, a model directory, or the `packages/cosmos3` model sources. The server container mounts this checkout at `/workspace/cosmos-framework`.

Both commands expose the generic image/video generation APIs, including the asynchronous `/v1/videos` policy-rollout path used first in this notebook. They publish the container's port `8000` to host port `8001`, matching the default `http://localhost:8001` endpoint.

### Cosmos3-Nano-Policy-DROID

```bash
docker rm -f cosmos3-vllm-omni-policy-notebook 2>/dev/null || true

docker run -d --name cosmos3-vllm-omni-policy-notebook \
  --runtime nvidia --gpus '"device=0"' \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e PYTHONPATH=/workspace/cosmos-framework \
  -v "$COSMOS3_REPO:/workspace/cosmos-framework" \
  -p 8001:8000 --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano-Policy-DROID \
    --no-guardrails \
    --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / \
    --port 8000 \
    --init-timeout 1800

# Wait until this returns model metadata before running the inference cells.
curl http://localhost:8001/v1/models
```

### Cosmos3-Edge-Policy-DROID

```bash
docker rm -f cosmos3-vllm-omni-policy-notebook 2>/dev/null || true

docker run -d --name cosmos3-vllm-omni-policy-notebook \
  --runtime nvidia --gpus '"device=0"' \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e PYTHONPATH=/workspace/cosmos-framework \
  -v "$COSMOS3_REPO:/workspace/cosmos-framework" \
  -p 8001:8000 --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Edge-Policy-DROID \
    --no-guardrails \
    --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / \
    --port 8000 \
    --init-timeout 1800

# Wait until this returns model metadata before running the inference cells.
curl http://localhost:8001/v1/models
```

The request cells below detect the active policy checkpoint. The Edge checkpoint receives the structured JSON prompt format it was trained on.

To inspect startup logs:

```bash
docker logs -f cosmos3-vllm-omni-policy-notebook
```

```python
from pathlib import Path
import os


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "cookbooks" / "cosmos3").is_dir():
            return path
    return start
COSMOS_ROOT = find_repo_root(Path.cwd().resolve())
COSMOS3_OUTPUT_ROOT = Path(
    os.environ.get("COSMOS3_VLLM_OUTPUT_ROOT", COSMOS_ROOT / "outputs" / "cosmos3_action_vllm")
).resolve()
COSMOS3_INPUT_DIR = COSMOS3_OUTPUT_ROOT / "inputs"
COSMOS3_POLICY_OUTPUT_DIR = COSMOS3_OUTPUT_ROOT / "action_policy_droid"
COSMOS3_ACTION_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3" / "generator" / "action"
DROID_ASSET_ROOT = COSMOS3_ACTION_ROOT / "assets" / "droid_lerobot_example"
VLLM_BASE_URL = os.environ.get("COSMOS3_VLLM_BASE_URL", "http://localhost:8001").rstrip("/")

COSMOS3_INPUT_DIR.mkdir(parents=True, exist_ok=True)
COSMOS3_POLICY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("COSMOS_ROOT:", COSMOS_ROOT)
print("DROID_ASSET_ROOT:", DROID_ASSET_ROOT)
print("COSMOS3_INPUT_DIR:", COSMOS3_INPUT_DIR)
print("COSMOS3_POLICY_OUTPUT_DIR:", COSMOS3_POLICY_OUTPUT_DIR)
print("COSMOS3_VLLM_BASE_URL:", VLLM_BASE_URL)
```

## Prepare a DROID Policy Input

This cell extracts the first frame from each checked-in DROID camera video and creates a 640x540 multiview conditioning image for the video API.

```python
import subprocess

import numpy as np
from PIL import Image
from IPython.display import display

try:
    import imageio_ffmpeg
except ImportError as exc:
    raise RuntimeError("Install imageio-ffmpeg in this notebook kernel: pip install imageio-ffmpeg") from exc

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
CAMERA_VIDEO_PATHS = {
    "observation/wrist_image_left": DROID_ASSET_ROOT / "videos" / "observation.image.wrist_image_left" / "chunk-000" / "file-000.mp4",
    "observation/exterior_image_1_left": DROID_ASSET_ROOT / "videos" / "observation.image.exterior_image_1_left" / "chunk-000" / "file-000.mp4",
    "observation/exterior_image_2_left": DROID_ASSET_ROOT / "videos" / "observation.image.exterior_image_2_left" / "chunk-000" / "file-000.mp4",
}
for key, video_path in CAMERA_VIDEO_PATHS.items():
    assert video_path.exists(), f"missing {key}: {video_path}"


def extract_first_frame(video_path: Path, out_path: Path) -> Path:
    if not out_path.exists() or out_path.stat().st_mtime < video_path.stat().st_mtime:
        subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-i", str(video_path), "-frames:v", "1", str(out_path)],
            check=True,
        )
    return out_path


frame_paths = {
    key: extract_first_frame(video_path, COSMOS3_INPUT_DIR / f"policy_{key.split('/')[-1]}.png")
    for key, video_path in CAMERA_VIDEO_PATHS.items()
}
frames = {key: Image.open(path).convert("RGB") for key, path in frame_paths.items()}

# Match vLLM-Omni's compose_robolab_views geometry: preserve the 640x360
# wrist view on top and resize each exterior view to 320x180 below it.
wrist = frames["observation/wrist_image_left"]
source_size = wrist.size
assert all(frame.size == source_size for frame in frames.values()), (
    "DROID camera frames must have matching dimensions",
    {key: frame.size for key, frame in frames.items()},
)
target_w = wrist.width
bottom_h, half_w = wrist.height // 2, wrist.width // 2
target_h = wrist.height + bottom_h
left = frames["observation/exterior_image_1_left"].resize((half_w, bottom_h), Image.Resampling.BILINEAR)
right = frames["observation/exterior_image_2_left"].resize((half_w, bottom_h), Image.Resampling.BILINEAR)
policy_image = Image.new("RGB", (target_w, target_h))
policy_image.paste(wrist, (0, 0))
policy_image.paste(left, (0, wrist.height))
policy_image.paste(right, (half_w, wrist.height))
policy_image_path = COSMOS3_INPUT_DIR / "droid_policy_first_frame.png"
policy_image.save(policy_image_path)

policy_prompt = os.environ.get(
    "COSMOS3_POLICY_PROMPT",
    "Pick up the object and place it in the target container.",
)
print("policy prompt:", policy_prompt)
print("video API conditioning image:", policy_image_path, policy_image.size)
display(policy_image)
```

## Run One-Shot Policy Inference Through `/v1/videos`

This path sends the correctly composed multiview image and instruction to the asynchronous video API, polls the job, writes the generated rollout video, and saves the raw `[16, 8]` DROID joint-position action chunk from the response metadata.

```python
import json
import time
from pathlib import Path

try:
    import requests
except ImportError as exc:
    raise RuntimeError("Install requests in this notebook kernel: pip install requests") from exc


def check_vllm_server(timeout_s: int = 600, interval_s: int = 10) -> str:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=10)
            response.raise_for_status()
            model_response = response.json()
            model_ids = [entry.get("id") for entry in model_response.get("data", []) if entry.get("id")]
            if not model_ids:
                raise RuntimeError(f"vLLM server returned no model IDs: {model_response}")
            print(model_response)
            return model_ids[0]
        except requests.RequestException as exc:
            last_error = exc
            print(f"Waiting for vLLM server at {VLLM_BASE_URL}: {exc}")
            time.sleep(interval_s)
    raise RuntimeError(
        f"vLLM server did not become ready at {VLLM_BASE_URL} within {timeout_s}s. "
        "Check `docker logs -f cosmos3-vllm-omni-policy-notebook`."
    ) from last_error


ACTION_VIDEO_RES_SIZE_INFO = {
    "480": {
        "1,1": (640, 640),
        "4,3": (736, 544),
        "3,4": (544, 736),
        "16,9": (832, 480),
        "9,16": (480, 832),
    }
}


def closest_action_size(height: int, width: int, resolution: str = "480") -> tuple[int, int]:
    input_ratio = height / width
    candidates = ACTION_VIDEO_RES_SIZE_INFO[resolution].values()
    return min(candidates, key=lambda size: abs(input_ratio - size[1] / size[0]))


def make_edge_policy_prompt(instruction: str, width: int, height: int, num_frames: int, fps: int) -> str:
    aspect_ratio = next(
        ratio
        for ratio, size in ACTION_VIDEO_RES_SIZE_INFO["480"].items()
        if size == (width, height)
    )
    duration_seconds = num_frames / fps
    prompt = {
        "cinematography": {
            "framing": (
                "This video contains concatenated views from multiple camera perspectives. "
                "The top row is the wrist camera and the bottom row contains two external cameras."
            )
        },
        "actions": [
            {
                "time": f"0:00-0:{round(duration_seconds):02d}",
                "description": instruction.rstrip(".!?") + ".",
            }
        ],
        "duration": f"{int(duration_seconds)}s",
        "fps": float(fps),
        "resolution": {"H": height, "W": width},
        "aspect_ratio": aspect_ratio,
    }
    return json.dumps(prompt, separators=(",", ":"))


def submit_policy_video(active_model: str) -> dict:
    run_dir = COSMOS3_POLICY_OUTPUT_DIR / "video_api"
    run_dir.mkdir(parents=True, exist_ok=True)
    request_image_path = run_dir / "policy_input.png"
    policy_image.save(request_image_path)
    print("saved", request_image_path)

    input_width, input_height = Image.open(request_image_path).size
    target_width, target_height = closest_action_size(input_height, input_width)
    extra_params = {
        "action_mode": "policy",
        "domain_name": "droid_lerobot",
        # Released DROID policy checkpoints predict 7 joints + 1 gripper.
        "raw_action_dim": 8,
        "action_chunk_size": 16,
        "image_size": 480,
        "guardrails": False,
    }
    is_edge_policy = "Cosmos3-Edge-Policy-DROID" in active_model
    request_prompt = (
        make_edge_policy_prompt(policy_prompt, target_width, target_height, num_frames=17, fps=15)
        if is_edge_policy
        else policy_prompt
    )
    form = {
        "prompt": request_prompt,
        "num_frames": 17,
        "fps": 15,
        "size": f"{target_width}x{target_height}",
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 5.0,
        "seed": 0,
        "extra_params": json.dumps(extra_params),
    }

    with request_image_path.open("rb") as image_file:
        response = requests.post(
            f"{VLLM_BASE_URL}/v1/videos",
            data={key: str(value) for key, value in form.items()},
            files={"input_reference": (request_image_path.name, image_file, "image/png")},
            timeout=120,
        )
    if not response.ok:
        (run_dir / "error_response.txt").write_text(response.text)
        print("vLLM request failed:", response.status_code)
        print(response.text)
        print("form:", json.dumps(form, indent=2))
        response.raise_for_status()

    initial = response.json()
    (run_dir / "response.json").write_text(json.dumps(initial, indent=2))

    while True:
        response = requests.get(f"{VLLM_BASE_URL}/v1/videos/{initial['id']}", timeout=30)
        response.raise_for_status()
        final = response.json()
        (run_dir / "final.json").write_text(json.dumps(final, indent=2))
        print(initial["id"], final.get("status"), f"{final.get('progress', 0)}%")
        if final.get("status") == "completed":
            break
        if final.get("status") in {"failed", "cancelled"}:
            raise RuntimeError(json.dumps(final, indent=2))
        time.sleep(2)

    action = final.get("action")
    if not action or "data" not in action:
        raise RuntimeError(f"vLLM response did not include action data: {json.dumps(final, indent=2)}")
    action_array = np.asarray(action["data"], dtype=np.float32)
    if action_array.shape != (16, 8):
        raise RuntimeError(f"Expected a [16, 8] DROID action chunk, got {action_array.shape}")
    if not np.isfinite(action_array).all():
        raise RuntimeError("DROID action response contains non-finite values")
    (run_dir / "action.json").write_text(json.dumps(action, indent=2))
    sample_outputs = {"outputs": [{"content": {"action": action["data"]}}]}
    (run_dir / "sample_outputs.json").write_text(json.dumps(sample_outputs, indent=2))

    content_response = requests.get(f"{VLLM_BASE_URL}/v1/videos/{initial['id']}/content", timeout=300)
    content_response.raise_for_status()
    video_path = run_dir / "policy_rollout.mp4"
    if content_response.content:
        video_path.write_bytes(content_response.content)
        print("saved", video_path)
    else:
        video_path = None
        print("video content endpoint returned an empty body")

    print("saved", run_dir / "action.json")
    print("action shape:", action.get("shape"), "dtype:", action.get("dtype"), "domain_id:", action.get("domain_id"))
    return {"initial": initial, "final": final, "run_dir": run_dir, "input_image_path": request_image_path, "video_path": video_path, "action": action}


active_vllm_model = check_vllm_server()
print("active vLLM model:", active_vllm_model)
policy_video_result = submit_policy_video(active_vllm_model)
```

## Inspect Video API Outputs

Preview the rollout video if the server returned one, and print the first few predicted action rows.

```python
import subprocess

import imageio_ffmpeg
from IPython.display import Video, display

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def make_preview(src: Path, crf: int = 28) -> Path:
    preview = src.with_name(f"{src.stem}_preview.mp4")
    if not preview.exists() or preview.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                "veryfast",
                "-an",
                "-pix_fmt",
                "yuv420p",
                str(preview),
            ],
            check=True,
        )
    return preview


action = policy_video_result["action"]
action_array = np.asarray(action["data"], dtype=np.float32)
print("action array:", action_array.shape, action_array.dtype)
print(action_array[: min(5, len(action_array))])

video_path = policy_video_result.get("video_path")
if video_path is not None:
    preview = make_preview(video_path)
    print(f"preview: {preview}")
    display(Video(str(preview), embed=True))
```
