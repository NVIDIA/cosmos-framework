<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Video Transfer with TensorRT-LLM

Run the five established controls—Canny/edge, blur, depth, segmentation, and WSM—against an
already-running TensorRT-LLM VisualGen server. The same notebook works with Cosmos3-Nano or
Cosmos3-Super; select the checkpoint when launching the server.

> **Skill note (not in the source):** this file is the cookbook notebook `run_video_transfer_with_trt_llm.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/generator/transfer/run_video_transfer_with_trt_llm.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

## Start the TensorRT-LLM server

Follow the shared [TensorRT-LLM setup](../../cookbooks/cosmos3/README.md#tensorrt-llm-generator), then launch one
server from the TensorRT-LLM checkout.

```bash
export TRTLLM_ROOT="${TRTLLM_ROOT:-$PWD}"

# Nano: one GPU
trtllm-serve nvidia/Cosmos3-Nano \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-nano-1gpu.yaml" \
  --port 8000

# Super: four GPUs (run instead of Nano)
torchrun --nproc_per_node=4 -m tensorrt_llm.commands.serve \
  nvidia/Cosmos3-Super \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-super-4gpu.yaml" \
  --port 8000
```

Install `requests` in this notebook kernel if needed, and install `ffmpeg` in the server
environment. The response is synchronous MP4 video
bytes. Every checked-in asset is already a precomputed control, so the notebook sends it once as
base64 media inside JSON `extra_params`; no top-level reference upload is needed. TensorRT-LLM
decodes the media back to bytes at the HTTP boundary.

```python
from pathlib import Path
import os


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "cookbooks" / "cosmos3").is_dir():
            return path
    return start


COSMOS_ROOT = find_repo_root(Path.cwd().resolve())
TRANSFER_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3" / "generator" / "transfer"
OUTPUT_ROOT = Path(
    os.environ.get("COSMOS3_TRTLLM_TRANSFER_OUTPUT_ROOT", TRANSFER_ROOT / "outputs" / "notebooks" / "trt_llm")
).resolve()
TRTLLM_BASE_URL = os.environ.get("COSMOS3_TRTLLM_BASE_URL", "http://localhost:8000").rstrip("/")
TRTLLM_API_KEY = os.environ.get("COSMOS3_TRTLLM_API_KEY", "tensorrt_llm")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("COSMOS_ROOT:", COSMOS_ROOT)
print("TRANSFER_ROOT:", TRANSFER_ROOT)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("TRTLLM_BASE_URL:", TRTLLM_BASE_URL)
```

## Request matrix

| Control      | Transport                  | Frames / fps | Text / control guidance |
| ------------ | -------------------------- | -----------: | ----------------------: |
| Edge         | Base64 precomputed control |     121 / 30 |               3.0 / 1.5 |
| Blur         | Base64 precomputed control |     121 / 30 |               3.0 / 1.5 |
| Depth        | Base64 precomputed control |     121 / 30 |               3.0 / 1.5 |
| Segmentation | Base64 precomputed control |     121 / 30 |               3.0 / 2.0 |
| WSM          | Base64 precomputed control |     100 / 10 |               1.0 / 3.0 |

The blur control is 4:3 and explicitly selects `1104x832`; the other controls are 16:9 and
explicitly select `1280x720`. TensorRT-LLM can
also derive the nearest supported bucket from the first precomputed control when neither dimension
is set. To compute edge or blur on the server instead, upload a raw source as multipart
`video_reference` and set the corresponding hint to `true`.

```python
import base64
import json
import time

import requests
from IPython.display import Video, display


def server_root_url() -> str:
    return TRTLLM_BASE_URL[:-3] if TRTLLM_BASE_URL.endswith("/v1") else TRTLLM_BASE_URL


def video_api_url() -> str:
    root = TRTLLM_BASE_URL if TRTLLM_BASE_URL.endswith("/v1") else f"{TRTLLM_BASE_URL}/v1"
    return f"{root}/videos/sync"


def wait_for_server(timeout_s: int = 1800, interval_s: int = 10) -> None:
    deadline = time.time() + timeout_s
    url = f"{server_root_url()}/health"
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=10)
            if response.ok:
                print("TensorRT-LLM server is ready:", url)
                return
        except requests.RequestException as exc:
            print("waiting for TensorRT-LLM:", exc)
        time.sleep(interval_s)
    raise TimeoutError(f"TensorRT-LLM did not become ready at {url}")


CONTROL_CASES = {
    "edge": {
        "control": "assets/edge/control_edge.mp4",
        "prompt": "assets/edge/prompt.json",
        "num_frames": 121,
        "fps": 30,
        "guidance_scale": 3.0,
        "control_guidance": 1.5,
        "size": "1280x720",
        "hint": {"preset_edge_threshold": "medium"},
    },
    "blur": {
        "control": "assets/blur/control_blur.mp4",
        "prompt": "assets/blur/prompt.json",
        "num_frames": 121,
        "fps": 30,
        "guidance_scale": 3.0,
        "control_guidance": 1.5,
        "size": "1104x832",
        "hint": {"preset_blur_strength": "medium"},
    },
    "depth": {
        "control": "assets/depth/control_depth.mp4",
        "prompt": "assets/depth/prompt.json",
        "num_frames": 121,
        "fps": 30,
        "guidance_scale": 3.0,
        "control_guidance": 1.5,
        "size": "1280x720",
    },
    "seg": {
        "control": "assets/seg/control_seg.mp4",
        "prompt": "assets/seg/prompt.json",
        "num_frames": 121,
        "fps": 30,
        "guidance_scale": 3.0,
        "control_guidance": 2.0,
        "size": "1280x720",
    },
    "wsm": {
        "control": "assets/wsm/control_wsm.mp4",
        "prompt": "assets/wsm/prompt.json",
        "num_frames": 100,
        "fps": 10,
        "guidance_scale": 1.0,
        "control_guidance": 3.0,
        "size": "1280x720",
    },
}


def compact_json(path: Path) -> str:
    return json.dumps(json.loads(path.read_text()), ensure_ascii=True, separators=(",", ":"))


def run_transfer(control_name: str) -> Path:
    """Send one precomputed control through the JSON/base64 input contract."""
    spec = CONTROL_CASES[control_name]
    control_path = (TRANSFER_ROOT / spec["control"]).resolve()
    prompt_path = (TRANSFER_ROOT / spec["prompt"]).resolve()
    negative_path = TRANSFER_ROOT / "assets" / "negative_prompt.json"
    for path in (control_path, prompt_path, negative_path):
        if not path.exists():
            raise FileNotFoundError(path)

    hint = dict(spec.get("hint", {}))
    # These files are already edge/blur/depth/seg/WSM control representations.
    # Send each one directly instead of also treating it as a raw input video.
    hint["control"] = base64.b64encode(control_path.read_bytes()).decode("ascii")

    extra_params = {
        "use_duration_template": False,
        "use_resolution_template": False,
        "use_system_prompt": False,
        "use_guardrails": True,
        "flow_shift": 10.0,
        control_name: hint,
        "control_guidance": spec["control_guidance"],
        "num_video_frames_per_chunk": spec["num_frames"],
        "num_conditional_frames": 1,
        "num_first_chunk_conditional_frames": 0,
        "share_vision_temporal_positions": True,
        "emphasize_control_in_prompt": False,
        "max_frames": spec["num_frames"],
    }
    request_body = {
        "prompt": compact_json(prompt_path),
        "negative_prompt": compact_json(negative_path),
        "size": spec["size"],
        "num_frames": spec["num_frames"],
        "fps": spec["fps"],
        "num_inference_steps": 50,
        "guidance_scale": spec["guidance_scale"],
        "max_sequence_length": 4096,
        "seed": 2026,
        "format": "mp4",
        "response_format": "file",
        "extra_params": extra_params,
    }
    headers = {"Accept": "video/mp4"}
    if TRTLLM_API_KEY:
        headers["Authorization"] = f"Bearer {TRTLLM_API_KEY}"
    response = requests.post(
        video_api_url(),
        json=request_body,
        headers=headers,
        timeout=3600,
    )
    if not response.ok:
        raise RuntimeError(f"TensorRT-LLM request failed ({response.status_code}): {response.text}")
    content_type = response.headers.get("content-type", "").lower()
    if not response.content:
        raise RuntimeError("TensorRT-LLM returned an empty media response")
    if "video/mp4" not in content_type or response.content[4:8] != b"ftyp":
        raise RuntimeError(f"Expected browser-compatible MP4, got {content_type!r}")
    output_path = OUTPUT_ROOT / f"transfer_{control_name}.mp4"
    output_path.write_bytes(response.content)
    print("control:", control_path)
    print("extra_params:", json.dumps({**extra_params, control_name: "<control media>"}, indent=2))
    print("saved:", output_path, content_type)
    return output_path


def view_transfer(path: Path) -> None:
    display(Video(str(path), embed=True))
```

## Canny / Edge transfer

```python
wait_for_server()
edge_output = run_transfer("edge")
```

```python
view_transfer(edge_output)
```

## Blur transfer

```python
wait_for_server()
blur_output = run_transfer("blur")
```

```python
view_transfer(blur_output)
```

## Depth transfer

```python
wait_for_server()
depth_output = run_transfer("depth")
```

```python
view_transfer(depth_output)
```

## Segmentation transfer

```python
wait_for_server()
seg_output = run_transfer("seg")
```

```python
view_transfer(seg_output)
```

## World Scenario Model (WSM) transfer

```python
wait_for_server()
wsm_output = run_transfer("wsm")
```

```python
view_transfer(wsm_output)
```
