<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Nano Action: Policy with SGLang

## Overview

This notebook runs Cosmos3 Nano **action policy** inference through SGLang using the checked-in DROID LeRobot sample under `assets/droid_lerobot_example`.

It first sends a `POST /v1/actions/generations` request with a first frame and instruction to predict a DROID policy action vector. It can then send that action vector to the forward-dynamics `POST /v1/videos` endpoint to render a rollout video.

> **Skill note (not in the source):** this file is the cookbook notebook `run_policy_with_sglang.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/generator/action/run_policy_with_sglang.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

<!-- rumdl-disable MD026 -->
<!-- Skill note (not in the source): the directive above keeps the notebook's "### Set up HF Cache:" heading of this cell as written (trailing colon); the rule is re-enabled right after the cell. -->

### Set up HF Cache:

SGLang container is run under root by default. User can create a writable cache directory on host, or start the container with host user id.

```bash
export SGLANG_HF_CACHE="${SGLANG_HF_CACHE:-$HOME/.cache/sglang-huggingface}"
mkdir -p "$SGLANG_HF_CACHE"
chmod 777 "$SGLANG_HF_CACHE"
```

### Start SGLang Server

Start the server in a terminal from the cookbook root (`.agents/skills`). Before running it, set `COSMOS3_REPO` to the root of a local checkout of the **cosmos-framework** repository:

```bash
export COSMOS3_REPO=/path/to/cosmos-framework
```

This must be the framework checkout that contains the `cosmos_framework/` Python package—not this `cosmos` cookbook repository, a model directory, or the `packages/cosmos3` model sources. The server container mounts this checkout at `/workspace/cosmos-framework`.

> **Skill note (not in the source):** for a guardrail-free run (no access to the gated `nvidia/Cosmos-1.0-Guardrail`) add `-e SGLANG_DISABLE_COSMOS3_GUARDRAILS=1 \` after the `PYTHONPATH` line (SKILL.md's Variant 3 block carries it, plus `-e HF_TOKEN \`); without it the server exits at startup with `GatedRepoError 401` and `/v1/models` never answers (eval 2026-11-08, Cosmos3-Nano-Policy-DROID).

```bash
docker rm -f cosmos3-sglang-policy-notebook 2>/dev/null || true

docker run -d --init --name cosmos3-sglang-policy-notebook \
  --runtime nvidia --gpus '"device=0"' \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e PYTHONPATH=/workspace/cosmos-framework \
  -v $SGLANG_HF_CACHE:/root/.cache/huggingface \
  -v "$PWD:/workspace" \
  -p 30000:30000 --ipc=host \
  lmsysorg/sglang:dev \
  sglang serve \
    --model-path nvidia/Cosmos3-Nano-Policy-DROID \
    --host 0.0.0.0

# Wait until this returns model metadata before running the inference cells.
curl http://localhost:30000/v1/models
```

When serving Cosmos3 Edge, use `nvidia/Cosmos3-Edge-Policy-DROID` as `model-path`.

To inspect startup logs:

```bash
docker logs -f cosmos3-sglang-policy-notebook
```

The notebook uses this same server for both the policy action request and the rollout-video request by default.

<!-- rumdl-enable MD026 -->

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
    os.environ.get("COSMOS3_SGLANG_OUTPUT_ROOT", COSMOS_ROOT / "outputs" / "cosmos3_action_sglang")
).resolve()
COSMOS3_INPUT_DIR = COSMOS3_OUTPUT_ROOT / "inputs"
COSMOS3_POLICY_OUTPUT_DIR = COSMOS3_OUTPUT_ROOT / "action_policy_droid"
COSMOS3_ACTION_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3" / "generator" / "action"
DROID_ASSET_ROOT = COSMOS3_ACTION_ROOT / "assets" / "droid_lerobot_example"
SGLANG_BASE_URL = os.environ.get("COSMOS3_SGLANG_BASE_URL", "http://localhost:30000").rstrip("/")
SGLANG_FD_BASE_URL = os.environ.get("COSMOS3_SGLANG_FD_BASE_URL", SGLANG_BASE_URL).rstrip("/")
SGLANG_MODEL = os.environ.get("COSMOS3_SGLANG_MODEL", "nvidia/Cosmos3-Nano-Policy-DROID")

COSMOS3_INPUT_DIR.mkdir(parents=True, exist_ok=True)
COSMOS3_POLICY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("COSMOS_ROOT:", COSMOS_ROOT)
print("DROID_ASSET_ROOT:", DROID_ASSET_ROOT)
print("COSMOS3_INPUT_DIR:", COSMOS3_INPUT_DIR)
print("COSMOS3_POLICY_OUTPUT_DIR:", COSMOS3_POLICY_OUTPUT_DIR)
print("COSMOS3_SGLANG_BASE_URL:", SGLANG_BASE_URL)
print("COSMOS3_SGLANG_FD_BASE_URL:", SGLANG_FD_BASE_URL)
print("COSMOS3_SGLANG_MODEL:", SGLANG_MODEL)
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

# DROID policy uses the training-time concat_view layout (see
# DROIDLeRobotDataset._compose_multi_view): the wrist camera at its native
# resolution on top, and the two exterior cameras at half width/height in the
# bottom row. For 640x360 DROID inputs this yields a 640x540 composite with the
# wrist filling the top two-thirds.
wrist = frames["observation/wrist_image_left"]
assert all(frame.size == wrist.size for frame in frames.values()), (
    "DROID camera frames must have matching dimensions",
    {key: frame.size for key, frame in frames.items()},
)
target_w, wrist_h = wrist.size
half_w, half_h = target_w // 2, wrist_h // 2
target_h = wrist_h + half_h
left = frames["observation/exterior_image_1_left"].resize((half_w, half_h), Image.Resampling.BILINEAR)
right = frames["observation/exterior_image_2_left"].resize((half_w, half_h), Image.Resampling.BILINEAR)
policy_image = Image.new("RGB", (target_w, target_h))
policy_image.paste(wrist, (0, 0))
policy_image.paste(left, (0, wrist_h))
policy_image.paste(right, (half_w, wrist_h))
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

## Run Policy Inference and Render Rollout

First, this sends a multipart request to `/v1/actions/generations` to get a DROID policy action vector. Then it sends a second request to the forward-dynamics `/v1/videos` endpoint, using that policy vector as the action condition, and saves the generated rollout video.

```python
import json
import mimetypes
import time
from pathlib import Path

from PIL import Image

try:
    import requests
except ImportError as exc:
    raise RuntimeError("Install requests in this notebook kernel: pip install requests") from exc


def check_sglang_server(timeout_s: int = 600, interval_s: int = 10) -> str:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{SGLANG_BASE_URL}/v1/models", timeout=10)
            response.raise_for_status()
            model_response = response.json()
            model_ids = [entry.get("id") for entry in model_response.get("data", []) if entry.get("id")]
            if not model_ids:
                raise RuntimeError(f"SGLang server returned no model IDs: {model_response}")
            print(model_response)
            return model_ids[0]
        except requests.RequestException as exc:
            last_error = exc
            print(f"Waiting for SGLang server at {SGLANG_BASE_URL}: {exc}")
            time.sleep(interval_s)
    raise RuntimeError(
        f"SGLang server did not become ready at {SGLANG_BASE_URL} within {timeout_s}s. "
        "Check `docker logs -f cosmos3-sglang-policy-notebook`."
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


def action_values_from_response(action: dict) -> list:
    values = action.get("values") or action.get("data")
    if values is None:
        raise RuntimeError(f"Policy response did not include action values: {json.dumps(action, indent=2)}")
    if values and isinstance(values[0], list) and values[0] and isinstance(values[0][0], list):
        values = values[0]
    return values


# Exact DROID multi-view caption used as cinematography.framing. SGLang keeps
# assembling the structured action JSON and accepts this request-level override.
CONCAT_VIEW_DESCRIPTION = (
    "The top row is from the wrist-mounted camera. "
    "The bottom row contains two horizontally concatenated third-person perspective views "
    "of the scene from opposite sides, with the robot visible."
)


def submit_policy_action(active_model: str) -> dict:
    run_dir = COSMOS3_POLICY_OUTPUT_DIR / "video_api"
    run_dir.mkdir(parents=True, exist_ok=True)

    input_width, input_height = Image.open(policy_image_path).size
    target_width, target_height = closest_action_size(input_height, input_width)
    extra_params = {
        "action_mode": "policy",
        "domain_name": "droid_lerobot",
        # Policy-DROID checkpoints default to 7 joints + 1 gripper (8D).
        "raw_action_dim": 8,
        "action_view_point": "concat_view",
        "action_cinematography_framing": CONCAT_VIEW_DESCRIPTION,
        "guardrails": False,
    }
    # The SGLang server applies ActionPromptJsonFormatter to the incoming
    # prompt itself, so send the plain task instruction (a client-side JSON
    # prompt would get double-wrapped into the "description" field).
    request_prompt = policy_prompt
    form = {
        "prompt": request_prompt,
        "num_frames": 17,
        "fps": 15,
        # /v1/actions/generations reads height/width options; it does not parse
        # a "size" string and would fall back to its 832x480 default, whose
        # 16:9 center-crop cuts into the 640x540 multi-view composite.
        "height": target_height,
        "width": target_width,
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 5.0,
        "seed": 0,
        "extra_params": json.dumps(extra_params),
    }

    with policy_image_path.open("rb") as image_file:
        response = requests.post(
            f"{SGLANG_BASE_URL}/v1/actions/generations",
            data={key: str(value) for key, value in form.items()},
            files={"input_reference": (policy_image_path.name, image_file, "image/png")},
            timeout=120,
        )
    if not response.ok:
        (run_dir / "error_response.txt").write_text(response.text)
        print("SGLang request failed:", response.status_code)
        print(response.text)
        print("form:", json.dumps(form, indent=2))
        response.raise_for_status()

    raw = response.json()
    (run_dir / "policy_response.json").write_text(json.dumps(raw, indent=2))
    action = raw["data"][0]["action"]
    action_values = action_values_from_response(action)
    action_array = np.asarray(action_values, dtype=np.float32)
    if action_array.shape != (16, 8):
        raise RuntimeError(f"Expected a [16, 8] DROID action chunk, got {action_array.shape}")
    if not np.isfinite(action_array).all():
        raise RuntimeError("DROID action response contains non-finite values")
    (run_dir / "action.json").write_text(json.dumps(action, indent=2))
    (run_dir / "policy_action_for_fd.json").write_text(json.dumps(action_values, indent=2))
    sample_outputs = {"outputs": [{"content": {"action": action_values}}]}
    (run_dir / "sample_outputs.json").write_text(json.dumps(sample_outputs, indent=2))

    print("saved", run_dir / "action.json")
    print("action shape:", action.get("shape"), "dtype:", action.get("dtype"), "domain_id:", action.get("domain_id"))
    return {"run_dir": run_dir, "raw": raw, "video_path": None, "action": action, "action_values": action_values}


def submit_forward_dynamics_rollout(policy_result: dict, active_model: str) -> Path:
    run_dir = policy_result["run_dir"]
    action_values = policy_result["action_values"]
    action_path = run_dir / "policy_action_for_fd.json"

    input_width, input_height = Image.open(policy_image_path).size
    target_width = (input_width // 16) * 16
    target_height = (input_height // 16) * 16
    mime_type = mimetypes.guess_type(policy_image_path.name)[0] or "image/png"
    extra_params = {
        "action_mode": "forward_dynamics",
        "domain_name": "droid_lerobot",
        "raw_action_dim": 8,
        "action_view_point": "concat_view",
        "action_cinematography_framing": CONCAT_VIEW_DESCRIPTION,
        "action": action_values,
        "guardrails": False,
    }
    request_prompt = policy_prompt
    form = {
        "prompt": request_prompt,
        "num_frames": len(action_values) + 1,
        "fps": 15,
        "size": f"{target_width}x{target_height}",
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 10.0,
        "seed": 0,
        "extra_params": json.dumps(extra_params),
    }

    print("rendering rollout from policy action:", action_path)
    print("forward-dynamics server:", SGLANG_FD_BASE_URL)
    with policy_image_path.open("rb") as image_file:
        response = requests.post(
            f"{SGLANG_FD_BASE_URL}/v1/videos",
            data={key: str(value) for key, value in form.items()},
            files={"input_reference": (policy_image_path.name, image_file, mime_type)},
            timeout=120,
        )
    if not response.ok:
        (run_dir / "rollout_error_response.txt").write_text(response.text)
        print("SGLang rollout request failed:", response.status_code)
        print(response.text)
        print("form:", json.dumps(form, indent=2))
        print("extra_params keys:", sorted(extra_params))
        print("action shape:", [len(action_values), len(action_values[0]) if action_values else 0])
        response.raise_for_status()

    initial = response.json()
    (run_dir / "rollout_response.json").write_text(json.dumps(initial, indent=2))
    while True:
        response = requests.get(f"{SGLANG_FD_BASE_URL}/v1/videos/{initial['id']}", timeout=30)
        if not response.ok:
            (run_dir / "rollout_poll_error_response.txt").write_text(response.text)
            print("SGLang rollout poll failed:", response.status_code)
            print(response.text)
            response.raise_for_status()
        final = response.json()
        (run_dir / "rollout_final.json").write_text(json.dumps(final, indent=2))
        print(initial["id"], final.get("status"), f"{final.get('progress', 0)}%")
        if final.get("status") == "completed":
            break
        if final.get("status") in {"failed", "cancelled"}:
            raise RuntimeError(
                f"SGLang rollout job {final.get('status')} for {initial['id']}. "
                f"Full response written to {run_dir / 'rollout_final.json'}:\n{json.dumps(final, indent=2)}"
            )
        time.sleep(2)

    response = requests.get(f"{SGLANG_FD_BASE_URL}/v1/videos/{initial['id']}/content", timeout=300)
    if not response.ok:
        (run_dir / "rollout_content_error_response.txt").write_text(response.text)
        print("SGLang rollout content request failed:", response.status_code)
        print(response.text)
        response.raise_for_status()
    video_path = run_dir / "policy_rollout.mp4"
    video_path.write_bytes(response.content)
    print("saved", video_path)
    return video_path


active_sglang_model = check_sglang_server()
print("active SGLang model:", active_sglang_model)
policy_video_result = submit_policy_action(active_sglang_model)
policy_video_result["video_path"] = submit_forward_dynamics_rollout(policy_video_result, active_sglang_model)
```

## Inspect Policy and Rollout Outputs

Print the first few predicted action rows and preview the rollout video generated by the forward-dynamics request.

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
action_array = np.asarray(policy_video_result["action_values"], dtype=np.float32)
print("action array:", action_array.shape, action_array.dtype)
print(action_array[: min(5, len(action_array))])

video_path = policy_video_result.get("video_path")
if video_path is not None:
    assert Path(video_path).exists(), f"missing rollout video: {video_path}"
    preview = make_preview(video_path)
    print(f"preview: {preview}")
    display(Video(str(preview), embed=True))
```
