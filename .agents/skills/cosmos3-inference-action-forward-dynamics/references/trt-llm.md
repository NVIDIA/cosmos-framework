<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Action Forward Dynamics with TensorRT-LLM

Run the checked-in autonomous-driving example: a first frame plus a 60-step, 9D action
trajectory conditions a 61-frame rollout. The `av` domain preset supplies the trained 480p,
10 fps recipe, so the request does not duplicate those mode defaults.

> **Skill note (not in the source):** this file is the cookbook notebook `run_fd_with_trt_llm.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/generator/action/run_fd_with_trt_llm.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

## Start the TensorRT-LLM server

Follow the shared [TensorRT-LLM setup](../../cookbooks/cosmos3/README.md#tensorrt-llm-generator), then run this
from the TensorRT-LLM checkout. This notebook is a client; keep the server in a separate terminal.

```bash
export TRTLLM_ROOT="${TRTLLM_ROOT:-$PWD}"
trtllm-serve nvidia/Cosmos3-Nano \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-nano-1gpu.yaml" \
  --port 8000
```

Install client-side decoding dependencies in this notebook kernel if needed:

```bash
pip install requests safetensors torch imageio imageio-ffmpeg pillow
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
ACTION_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3" / "generator" / "action"
OUTPUT_ROOT = Path(
    os.environ.get("COSMOS3_TRTLLM_ACTION_OUTPUT_ROOT", ACTION_ROOT / "outputs" / "notebooks" / "trt_llm")
).resolve()
TRTLLM_BASE_URL = os.environ.get("COSMOS3_TRTLLM_BASE_URL", "http://localhost:8000").rstrip("/")
TRTLLM_API_KEY = os.environ.get("COSMOS3_TRTLLM_API_KEY", "tensorrt_llm")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("COSMOS_ROOT:", COSMOS_ROOT)
print("ACTION_ROOT:", ACTION_ROOT)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("TRTLLM_BASE_URL:", TRTLLM_BASE_URL)
```

## TensorRT-LLM response contract

The notebooks use the blocking `POST /v1/videos/sync` route. Every Action request sets
`format=safetensors` and `response_format=file`; the response is `application/octet-stream` with named `video`, `action`,
and `frame_rate` tensors. This differs from vLLM-Omni's top-level action metadata.

TensorRT-LLM also implements the asynchronous `POST /v1/videos` route. It returns HTTP 202,
can be polled through `GET /v1/videos/{id}`, and serves the same tensor payload from
`GET /v1/videos/{id}/content`.

```python
import json
import mimetypes
import time

import imageio.v3 as iio
import requests
from IPython.display import Video, display
from safetensors.torch import load as load_safetensors


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


def submit_action(
    prompt: str, reference_path: Path, extra_params: dict, output_path: Path, *, seed: int = 0
) -> dict:
    """Call the synchronous API and decode its action/video tensor payload."""
    reference_path = reference_path.resolve()
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    media_type = mimetypes.guess_type(reference_path.name)[0] or "application/octet-stream"
    form = {
        "prompt": prompt,
        "format": "safetensors",
        "response_format": "file",
        "seed": str(seed),
        "extra_params": json.dumps(extra_params, separators=(",", ":")),
    }
    headers = {"Accept": "application/octet-stream"}
    if TRTLLM_API_KEY:
        headers["Authorization"] = f"Bearer {TRTLLM_API_KEY}"
    with reference_path.open("rb") as reference_file:
        response = requests.post(
            video_api_url(),
            data=form,
            files={"image_reference": (reference_path.name, reference_file, media_type)},
            headers=headers,
            timeout=3600,
        )
    if not response.ok:
        raise RuntimeError(f"TensorRT-LLM request failed ({response.status_code}): {response.text}")
    content_type = response.headers.get("content-type", "")
    if "application/octet-stream" not in content_type:
        raise RuntimeError(f"Expected a tensor payload, got {content_type!r}")
    output_path.write_bytes(response.content)
    payload = load_safetensors(response.content)
    missing = {"video", "action"} - payload.keys()
    if missing:
        raise RuntimeError(f"TensorRT-LLM action payload is missing {sorted(missing)}; got {sorted(payload)}")
    print("saved tensor payload:", output_path)
    print("video:", tuple(payload["video"].shape), payload["video"].dtype)
    print("action:", tuple(payload["action"].shape), payload["action"].dtype)
    return payload


def save_and_view_video(payload: dict, output_path: Path) -> Path:
    frames = payload["video"].detach().cpu().numpy()
    if frames.ndim == 5 and frames.shape[0] == 1:
        frames = frames[0]
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected video [T,H,W,3], got {frames.shape}")
    fps_tensor = payload.get("frame_rate")
    fps = float(fps_tensor.item()) if fps_tensor is not None else 24.0
    iio.imwrite(output_path, frames, fps=fps)
    print("saved rollout:", output_path, "fps:", fps)
    display(Video(str(output_path), embed=True))
    return output_path
```

## Prepare the checked-in AV inputs

```python
image_path = ACTION_ROOT / "assets" / "images" / "av_0.jpg"
action_path = ACTION_ROOT / "assets" / "actions" / "av_traj_forward.json"
actions = json.loads(action_path.read_text())
assert len(actions) == 60 and all(len(row) == 9 for row in actions)

prompt = "You are an autonomous vehicle planning system."
extra_params = {
    "action_mode": "forward_dynamics",
    "domain_name": "av",
    "action": actions,
    "use_guardrails": True,
}
print("image:", image_path)
print("action trajectory:", action_path, len(actions), "x", len(actions[0]))
print(json.dumps({**extra_params, "action": "<60x9 checked-in trajectory>"}, indent=2))
```

## Run forward dynamics

```python
wait_for_server()
fd_payload = submit_action(
    prompt,
    image_path,
    extra_params,
    OUTPUT_ROOT / "forward_dynamics_av.safetensors",
    seed=0,
)
```

## Inspect the rollout and action tensor

```python
save_and_view_video(fd_payload, OUTPUT_ROOT / "forward_dynamics_av.mp4")
fd_action = fd_payload["action"].detach().cpu().numpy()
print("first five action rows:")
print(fd_action[:5])
```
