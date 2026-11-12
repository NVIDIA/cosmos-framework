<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Action: Forward Dynamics with vLLM-Omni

## Prerequisites

Generator requires the Guardrail. Request access to the gated [nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail) HF repository before running these examples. When running with guardrails enabled, also set and pass `HF_TOKEN` to the container; it must be a Hugging Face token authorized for that repository. To disable guardrails server-wide, pass `--no-guardrails` to `vllm serve`; alternatively, set `guardrails: false` in vLLM-Omni `extra_params` or `extra_args` to disable them per request.

## Overview

This notebook runs Cosmos3 **action forward-dynamics** inference through the vLLM-Omni OpenAI-compatible video API:

```text
POST /v1/videos
```

Forward dynamics predicts future visual observations from an initial image and an action trajectory. This notebook contains separate AV, robotics, UMI, and human hand-pose sections that each build their own input spec, run inference, and visualize generated videos.

> **Skill note (not in the source):** this file is the cookbook notebook `run_fd_with_vllm_omni.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/generator/action/run_fd_with_vllm_omni.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

## Install Notebook Dependencies

Run this cell in the notebook kernel before the data-loading and visualization cells.

```python
%pip install matplotlib pandas pyarrow "lerobot[dataset]"
%pip install av==18.0.0
```

## Start vLLM-Omni Server

Choose either Cosmos3-Nano or Cosmos3-Edge and start that server in a terminal from the cookbook root (`.agents/skills`). The commands are mutually exclusive: both use the same container name and publish the container's port `8000` to host port `8001`, which matches the notebook's default `http://localhost:8001` endpoint.

### Cosmos3-Nano

```bash
docker rm -f cosmos3-vllm-omni-notebook 2>/dev/null || true

docker run -d --name cosmos3-vllm-omni-notebook \
  --runtime nvidia --gpus '"device=0"' \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v "$PWD:/workspace" \
  -v "$PWD:$PWD" \
  -w /workspace \
  -p 8001:8000 --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano \
    --no-guardrails \
    --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / \
    --port 8000 \
    --init-timeout 1800

# Wait until this returns model metadata before running the inference cells.
curl http://localhost:8001/v1/models
```

### Cosmos3-Edge

```bash
docker rm -f cosmos3-vllm-omni-notebook 2>/dev/null || true

docker run -d --name cosmos3-vllm-omni-notebook \
  --runtime nvidia --gpus '"device=0"' \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v "$PWD:/workspace" \
  -v "$PWD:$PWD" \
  -w /workspace \
  -p 8001:8000 --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Edge \
    --no-guardrails \
    --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / \
    --port 8000 \
    --init-timeout 1800

# Wait until this returns model metadata before running the inference cells.
curl http://localhost:8001/v1/models
```

The request cells below use whichever Nano or Edge server is active at the configured endpoint. To keep outputs from separate model runs, set `COSMOS3_VLLM_OUTPUT_ROOT` to a different directory before running the configuration cell.

### Quantized Checkpoints

Published [Edge FP8](https://huggingface.co/nvidia/Cosmos3-Edge-FP8),
[Nano FP8](https://huggingface.co/nvidia/Cosmos3-Nano-FP8) and
[Super FP8](https://huggingface.co/nvidia/Cosmos3-Super-FP8) checkpoints can be
downloaded to a local directory. Or use an output from the
[quantization walkthrough](../../cosmos3-quantization/SKILL.md).
Serve that directory with the original `--served-model-name` so the request
cells keep their existing model IDs. Do not select FP8 with `--revision` alone:
the Cosmos3 vLLM-Omni pipeline must receive a materialized local checkpoint.

These are static checkpoint FP8 weights and calibrated scales, exported with
`calibration_behavior="framework"`; this is separate from quantizing BF16 weights
at server load time. The exporter records the activation policy in
`transformer/config.json` under `quantization_config.runtime.diffusion_step_policy`.
The default policy requests A16 activations for the first/last three denoising
iterations and the generator reasoner, and native FP8 in the middle. Overlapping
first/last windows use A16, so a four-step student uses A16 throughout. Weights
remain FP8. Standalone vLLM reasoner inference uses ModelOpt FP8 independently of
this diffusion policy. Backend support for the exported policy must be checked
with the chosen runtime image.

Run these commands from the **Framework checkout root**. Choose one model at a
time. For a generated output set `FP8_MODEL_DIR` to its absolute directory and
`FP8_SOURCE_DIR` to the absolute source checkpoint directory, or a common parent
containing its snapshot and backing blobs. The source mount preserves exported
symlinks. If another cache holds symlink targets, mount that cache at its original
absolute path too. Keep `HF_HOME` consistent with the quantization run.

Choose one download block, then use the shared launch command below.

#### Cosmos3-Edge FP8

```bash
# Download published Edge FP8 instead of Nano; the launch below serves either tier.
export FP8_MODEL_DIR="$PWD/models/cosmos3-edge-fp8"
export SERVED_MODEL="nvidia/Cosmos3-Edge"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("nvidia/Cosmos3-Edge-FP8", local_dir=os.environ["FP8_MODEL_DIR"])
PY
```

#### Cosmos3-Nano FP8

```bash
# Download published Nano FP8; use Cosmos3-Super-FP8 and its served ID for Super.
export FP8_MODEL_DIR="${FP8_MODEL_DIR:-$PWD/models/cosmos3-nano-fp8}"
export SERVED_MODEL="${SERVED_MODEL:-nvidia/Cosmos3-Nano}"
# Skip this download when FP8_MODEL_DIR is a locally generated checkpoint.
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("nvidia/Cosmos3-Nano-FP8", local_dir=os.environ["FP8_MODEL_DIR"])
PY
```

#### Serve the selected checkpoint

```bash
export HF_HOME="$(realpath -m "${HF_HOME:-$HOME/.cache/huggingface}")"
FP8_MODEL_DIR="$(realpath "$FP8_MODEL_DIR")"
test -f "$FP8_MODEL_DIR/hf_quant_config.json"
# Mount each destination once, including on hosts whose HOME is /root.
MOUNTS=()
declare -A MOUNT_DESTINATIONS=()
add_mount() {
  local source_path="$1" destination="$2" mode="${3:-rw}"
  if [[ -z "${MOUNT_DESTINATIONS[$destination]+present}" ]]; then
    MOUNTS+=(-v "$source_path:$destination:$mode")
    MOUNT_DESTINATIONS[$destination]=1
  fi
}
add_mount "$HF_HOME" /root/.cache/huggingface
add_mount "$HF_HOME" "$HF_HOME" ro
add_mount "$PWD/.agents/skills" /workspace
add_mount "$PWD" "$PWD"
add_mount "$FP8_MODEL_DIR" "$FP8_MODEL_DIR" ro
if [ -n "${FP8_SOURCE_DIR:-}" ]; then
  FP8_SOURCE_DIR="$(realpath "$FP8_SOURCE_DIR")"
  add_mount "$FP8_SOURCE_DIR" "$FP8_SOURCE_DIR" ro
fi
docker run --runtime nvidia --gpus all \
  -e HF_TOKEN \
  "${MOUNTS[@]}" \
  -w /workspace \
  -p 8001:8000 --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve "$FP8_MODEL_DIR" \
    --served-model-name "$SERVED_MODEL" \
    --no-guardrails \
    --omni --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / --port 8000 --init-timeout 1800
```

For Super, add the upstream `--tensor-parallel-size 4` and
`--enable-layerwise-offload` options if needed. Poll
`curl http://localhost:8001/v1/models` until ready and verify the ID equals
`SERVED_MODEL`, then run the matching request and visualization cells below.
The image and activation policy require runtime validation on your GPU setup.

## Configure Notebook Variables

Run this cell after the vLLM-Omni server is available. It resolves local input/output paths and stores generated outputs under `outputs/cosmos3_action_vllm/` by default.

The configuration supports these environment variables:

- `COSMOS3_VLLM_OUTPUT_ROOT`: optional override for the output directory. The notebook resolves this into the `COSMOS3_OUTPUT_ROOT` Python variable, which contains generated inputs, chunk outputs, and stitched videos.
- `COSMOS3_DROID_ROOT`: optional override for the DROID LeRobot dataset. Set it to a version-named dataset root such as `/path/to/droid_plus_lerobot_640x360_20260412`, containing the `success/` split. If unset, the notebook creates a versioned alias for the bundled `assets/droid_lerobot_example` sample.

Optionally set `COSMOS3_REPO` to the root of a local checkout of the **cosmos-framework** repository:

```bash
export COSMOS3_REPO=/path/to/cosmos-framework
```

This must be the framework checkout that contains the `cosmos_framework/` Python package—a model directory cannot be used as a source checkout. The notebook adds this path to Python's import path to load `cosmos_framework`. Set it in the shell that launches Jupyter before running this cell.

```python
from pathlib import Path
import os
import json


def find_framework_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").is_file() and (path / "cosmos_framework").is_dir():
            return path
    raise FileNotFoundError("Launch from the Framework checkout or a notebook directory inside it.")


FRAMEWORK_ROOT = find_framework_root(Path.cwd().resolve())
COSMOS_ROOT = FRAMEWORK_ROOT / ".agents" / "skills"
COSMOS3_REPO = Path(os.environ.get("COSMOS3_REPO", FRAMEWORK_ROOT)).resolve()
COSMOS3_OUTPUT_ROOT = Path(
    os.environ.get("COSMOS3_VLLM_OUTPUT_ROOT", COSMOS_ROOT / "outputs" / "cosmos3_action_vllm")
).resolve()
COSMOS3_INPUT_DIR = COSMOS3_OUTPUT_ROOT / "inputs"
VLLM_BASE_URL = os.environ.get("COSMOS3_VLLM_BASE_URL", "http://localhost:8001").rstrip("/")


def resolve_input(rel_path: str) -> str:
    path = (COSMOS_ROOT / rel_path).resolve()
    assert path.exists(), f"missing input: {path}"
    return str(path)


COSMOS3_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
COSMOS3_INPUT_DIR.mkdir(parents=True, exist_ok=True)

print("COSMOS_ROOT:", COSMOS_ROOT)
print("COSMOS3_REPO:", COSMOS3_REPO)
print("COSMOS3_INPUT_DIR:", COSMOS3_INPUT_DIR)
print("COSMOS3_OUTPUT_ROOT:", COSMOS3_OUTPUT_ROOT)
print("COSMOS3_VLLM_BASE_URL:", VLLM_BASE_URL)


_ACTION_VIEWPOINT_FRAMING = {
    "ego_view": "This video is captured from a first-person perspective looking at the scene.",
    "third_person_view": "This video is captured from a third-person perspective looking towards the agent from the front.",
    "wrist_view": "This video is captured from a wrist-mounted camera.",
    "concat_view": "This video contains concatenated views from multiple camera perspectives.",
}
_ACTION_ASPECT_RATIOS = {"1,1": 1.0, "4,3": 4 / 3, "3,4": 3 / 4, "16,9": 16 / 9, "9,16": 9 / 16}


def build_action_prompt(record: dict, *, width: int, height: int, default_prompt: str) -> str:
    description = str(record.get("prompt") or "").strip() or default_prompt
    if not description.endswith((".", "!", "?")):
        description += "."

    num_frames = int(record["action_chunk_size"]) + 1
    fps = int(record["fps"])
    duration_seconds = num_frames / fps
    aspect_ratio = min(_ACTION_ASPECT_RATIOS, key=lambda key: abs(width / height - _ACTION_ASPECT_RATIOS[key]))
    minutes, seconds = divmod(round(duration_seconds), 60)
    action_prompt = {}
    framing = _ACTION_VIEWPOINT_FRAMING.get(record.get("view_point"))
    if framing is not None:
        action_prompt["cinematography"] = {"framing": framing}
    action_prompt.update(
        {
            "actions": [{"time": f"0:00-{minutes}:{seconds:02d}", "description": description}],
            "duration": f"{int(duration_seconds)}s",
            "fps": float(fps),
            "resolution": {"H": height, "W": width},
            "aspect_ratio": aspect_ratio,
        }
    )
    return json.dumps(action_prompt)
```

## AV

In this example, we show how to provide a set of ego poses of a autonomous vehicle and an image to generate driving videos using the model served by the active vLLM-Omni endpoint.

### Create the AV Forward-Dynamics Input Spec

AV forward-dynamics inference is driven by a JSONL spec, one line per run. Each line shares the same start frame (`vision_path`) but uses a different ego trajectory (`action_path`), so we get one generated video per trajectory.

The action input is prepared in a JSON file, which can be converted from camera poses (camera-to-world transformation, OpenCV convention, unit in meter) via `pose_abs_to_rel`:

```python
if str(COSMOS3_REPO) not in sys.path:
    sys.path.insert(0, str(COSMOS3_REPO))
from cosmos_framework.data.generator.action.utils.pose_utils import pose_abs_to_rel

poses_abs = np.array([...]) # [T, 4, 4], camera-to-world transformation in opencv convention, unit in meter
poses_rel = pose_abs_to_rel(
    poses_abs,
    rotation_format="rot6d",
    pose_convention="backward_framewise",
    translation_scale=1.35,
) # [T-1, 9], translation(3), rot6d(6), framewise relative transformation

with open("custom_traj.json", "w") as f:
    json.dump(poses_rel, f)
```

```python
# `resolve_input` and the COSMOS3_* paths come from the variables cell.
import json

# Local AV inputs, relative to the cookbook root.
av_input_image = "cookbooks/cosmos3/generator/action/assets/images/av_0.jpg"
av_input_actions = {
    "av_forward": "cookbooks/cosmos3/generator/action/assets/actions/av_traj_forward.json",
    "av_left": "cookbooks/cosmos3/generator/action/assets/actions/av_traj_left.json",
    "av_right": "cookbooks/cosmos3/generator/action/assets/actions/av_traj_right.json",
}

av_vision_path = resolve_input(av_input_image)
av_records = [
    {
        "action_chunk_size": 60,
        "action_path": resolve_input(action_rel),
        "domain_name": "av",
        "fps": 10,
        "image_size": 480,
        "view_point": "ego_view",
        "model_mode": "forward_dynamics",
        "name": name,
        "prompt": "You are an autonomous vehicle planning system.",
        "seed": 0,
        "vision_path": av_vision_path,
    }
    for name, action_rel in av_input_actions.items()
]

COSMOS3_INPUT_DIR.mkdir(parents=True, exist_ok=True)
av_fd_input_path = COSMOS3_INPUT_DIR / "action_forward_dynamics_av_custom.jsonl"
av_fd_input_path.write_text("".join(json.dumps(r) + "\n" for r in av_records))
av_fd_output_dir = COSMOS3_OUTPUT_ROOT / "action_forward_dynamics_av_custom"

os.environ["COSMOS3_AV_FD_INPUT"] = str(av_fd_input_path)
os.environ["COSMOS3_AV_FD_OUTPUT"] = str(av_fd_output_dir)

print("wrote AV spec:", av_fd_input_path)
print("AV runs:", list(av_input_actions))
print(av_fd_input_path.read_text())
```

### Visualize AV Input Trajectories

Before generating any video, plot each input ego trajectory as a 3D camera path with frustums and a top-down bird's-eye view.

```python
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# The notebook kernel may differ from the framework venv, so put the repo on the
# path before importing `cosmos_framework`.
if str(COSMOS3_REPO) not in sys.path:
    sys.path.insert(0, str(COSMOS3_REPO))
from cosmos_framework.data.generator.action.utils.pose_utils import pose_rel_to_abs

# frustum: apex + image-rectangle corners (camera +Z forward), and their edges
_FRUSTUM = np.array([[0, 0, 0], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
_EDGES = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]


def visualize_pose(poses_abs, *, n_frustums=20, scale_frac=0.03, aspect=16 / 9,
                   fov_deg=60.0, vertical_exaggeration=1.0, cmap="turbo",
                   title=None, save_path=None, show=True):
    """3D camera trajectory (with frustums) + a top-down bird's-eye view."""
    poses_abs = np.asarray(poses_abs)
    pos = poses_abs[:, :3, 3]
    fwd = poses_abs[:, :3, 2]
    T = len(pos)
    colors = plt.get_cmap(cmap)(np.arange(T) / max(T - 1, 1))
    scale = max(np.ptp(pos, axis=0).max() * scale_frac, 1e-3)
    step = max(1, T // max(n_frustums, 1))
    xzy = [0, 2, 1]

    fig = plt.figure(figsize=(14, 6))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    path = pos[:, xzy]
    ax.plot(*path.T, color="0.6", lw=1.0, alpha=0.7)
    lines, lcolors, allpts = [], [], [path]
    for i in range(0, T, step):
        cw = ((_FRUSTUM * [aspect, 1, 1] * scale * np.tan(np.radians(fov_deg) / 2))
              @ poses_abs[i, :3, :3].T + poses_abs[i, :3, 3])[:, xzy]
        allpts.append(cw)
        lines += [[cw[a], cw[b]] for a, b in _EDGES]
        lcolors += [colors[i]] * len(_EDGES)
    ax.add_collection3d(Line3DCollection(lines, colors=lcolors, linewidths=1.2))
    ax.scatter(*path[0], color="lime", s=80, edgecolor="k", label="first frame", zorder=5)
    ax.scatter(*path[-1], color="red", s=80, edgecolor="k", label="last frame", zorder=5)
    rng = np.clip(np.ptp(np.concatenate(allpts), axis=0), 1e-9, None)
    ax.set_box_aspect((rng[0], rng[1], rng[2] * vertical_exaggeration))
    ax.set_xlabel("X (m)", labelpad=12)
    ax.set_ylabel("Z forward (m)", labelpad=12)
    ax.set_zlabel("Y up (m)", labelpad=10)
    ax.set_zticks([])
    ax.set_title(title or f"Camera trajectory + frustums ({T} frames)")
    ax.legend(loc="upper left")
    ax.view_init(elev=22, azim=-70)

    ax2 = fig.add_subplot(1, 2, 2)
    seg = np.stack([pos[:-1, [0, 2]], pos[1:, [0, 2]]], axis=1)
    lc = LineCollection(seg, cmap=cmap, norm=plt.Normalize(0, T - 1), linewidth=2.5)
    lc.set_array(np.arange(T - 1))
    ax2.add_collection(lc)
    ax2.quiver(pos[::step, 0], pos[::step, 2], fwd[::step, 0], fwd[::step, 2],
               color=colors[::step], angles="xy", width=0.005, scale=22, zorder=3)
    ax2.scatter(*pos[0, [0, 2]], color="lime", s=80, edgecolor="k", label="first frame", zorder=5)
    ax2.scatter(*pos[-1, [0, 2]], color="red", s=80, edgecolor="k", label="last frame", zorder=5)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z forward (m)")
    ax2.set_title("Top-down (bird's-eye view)")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.autoscale_view()
    ax2.legend()
    fig.colorbar(lc, ax=ax2, label="frame index")

    plt.tight_layout(w_pad=6)
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print("saved", save_path)
    if show:
        plt.show()


for record in av_records:
    name = record["name"]
    with open(record["action_path"]) as f:
        poses_rel = np.array(json.load(f))

    # AV action convention: rot6d rotation, backward_framewise, translation_scale = 1.35.
    poses_abs = pose_rel_to_abs(
        poses_rel,
        rotation_format="rot6d",
        pose_convention="backward_framewise",
        translation_scale=1.35,
    )
    print(name, poses_rel.shape, poses_abs.shape)
    visualize_pose(poses_abs, title=f"{name}: camera trajectory + frustums ({len(poses_abs)} frames)", show=True)
```

### Run AV Forward-Dynamics Inference

Runs the active Cosmos3 model on every line of the AV spec through vLLM-Omni. Each run writes its video to:

```text
<output>/action_forward_dynamics_av_custom/<name>/vision.mp4
```

The request sets top-level `size` to the conditioning image resolution so vLLM-Omni returns output at the input resolution without reflection padding.

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


def check_vllm_server(timeout_s: int = 600, interval_s: int = 10) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=10)
            response.raise_for_status()
            print(response.json())
            return
        except requests.RequestException as exc:
            last_error = exc
            print(f"Waiting for vLLM server at {VLLM_BASE_URL}: {exc}")
            time.sleep(interval_s)
    raise RuntimeError(
        f"vLLM server did not become ready at {VLLM_BASE_URL} within {timeout_s}s. "
        "Check `docker logs -f cosmos3-vllm-omni-notebook`."
    ) from last_error


def submit_forward_dynamics(record: dict, fd_output_dir: Path) -> dict:
    run_dir = fd_output_dir / record["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    vision_path = Path(record["vision_path"])
    input_width, input_height = Image.open(vision_path).size
    mime_type = mimetypes.guess_type(vision_path.name)[0] or "application/octet-stream"
    extra_params = {
        "action_mode": "forward_dynamics",
        "domain_name": record["domain_name"],
        "action_chunk_size": record["action_chunk_size"],
        "image_size": record["image_size"],
        "action": json.loads(Path(record["action_path"]).read_text()),
        "guardrails": False,
    }
    prompt = build_action_prompt(
        record, width=input_width, height=input_height, default_prompt="A robot manipulates an object."
    )
    form = {
        "prompt": prompt,
        "num_frames": record["action_chunk_size"] + 1,
        "fps": record["fps"],
        "size": f"{input_width}x{input_height}",
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 10.0,
        "seed": record["seed"],
        "extra_params": json.dumps(extra_params),
    }

    with vision_path.open("rb") as image_file:
        response = requests.post(
            f"{VLLM_BASE_URL}/v1/videos",
            data={key: str(value) for key, value in form.items()},
            files={"input_reference": (vision_path.name, image_file, mime_type)},
            timeout=120,
        )
    if not response.ok:
        (run_dir / "error_response.txt").write_text(response.text)
        print("vLLM request failed:", response.status_code)
        print(response.text)
        print("form:", json.dumps(form, indent=2))
        print("extra_params keys:", sorted(extra_params))
        print("action shape:", [len(extra_params["action"]), len(extra_params["action"][0]) if extra_params["action"] else 0])
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

    response = requests.get(f"{VLLM_BASE_URL}/v1/videos/{initial['id']}/content", timeout=300)
    response.raise_for_status()
    video_path = run_dir / "vision.mp4"
    video_path.write_bytes(response.content)

    action = final.get("action")
    if action is not None:
        (run_dir / "action.json").write_text(json.dumps(action, indent=2))

    print("saved", video_path)
    if action is not None:
        print("action shape:", action.get("shape"), "dtype:", action.get("dtype"))
    return {"record": record, "initial": initial, "final": final, "run_dir": run_dir, "video_path": video_path, "action": action}


check_vllm_server()
av_results = []
for record in av_records:
    print(f"\nSubmitting {record['name']}")
    av_results.append(submit_forward_dynamics(record, av_fd_output_dir))
```

### Visualize AV Generated Videos

`Video(..., embed=True)` base64-inlines a file into the notebook, and embedding full-resolution runs can freeze the front-end. This cell first transcodes each video to a small preview using the ffmpeg binary bundled with `imageio-ffmpeg`, then embeds the previews. The full-resolution `vision.mp4` files are left untouched.

```python
import subprocess
import imageio_ffmpeg
from IPython.display import Video, display

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def make_preview(src: Path, crf: int = 28) -> Path:
    """Re-encode `src` to a compact, browser-friendly mp4 (cached)."""
    preview = src.with_name(f"{src.stem}_preview.mp4")
    if not preview.exists() or preview.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
             "-c:v", "libx264", "-crf", str(crf),
             "-preset", "veryfast", "-an", "-pix_fmt", "yuv420p", str(preview)],
            check=True,
        )
    return preview


for record in av_records:
    name = record["name"]
    src = av_fd_output_dir / name / "vision.mp4"
    assert src.exists(), f"missing: {src}"
    preview = make_preview(src)
    print(f"{name}  ({src.stat().st_size // 1024} KB -> {preview.stat().st_size // 1024} KB preview)")
    display(Video(str(preview), embed=True))
```

## Robotics

In this example, we show how to start from a LeRobot dataset of DROID and run **multiview** generation for robotics manipulation **autoregressively**.

> Note: Before running this LeRobot section, PyAV may need to be reinstalled as `av==15.1.0`.

### Create the Robotics Autoregressive Forward-Dynamics Plan

Robotics forward-dynamics runs autoregressively over five contiguous 16-action DROID chunks. This cell writes the GT first conditioning image for chunk 0 and one action JSON per chunk. Later chunks receive their conditioning image from the previous chunk's generated last frame during the inference loop.

```python
# `resolve_input` and the COSMOS3_* paths come from the variables cell.
import json
import os
import sys


from PIL import Image

# The notebook kernel may differ from the framework venv, so put the repo on the
# path before importing `cosmos_framework`.
if str(COSMOS3_REPO) not in sys.path:
    sys.path.insert(0, str(COSMOS3_REPO))

from cosmos_framework.data.generator.action.datasets import DROIDMergedLeRobotDataset

robotics_dataset_root = resolve_input("cookbooks/cosmos3/generator/action/assets/droid_lerobot_example")
# DROIDLeRobotDataset selects its feature layout from the directory basename and
# rejects the bundled asset (unregistered name -> ValueError: Unknown version).
# Cosmos3-Nano was trained on quantile-normalized DROID actions.
robotics_dataset = DROIDMergedLeRobotDataset(
    root=robotics_dataset_root,
    action_normalization="quantile_rot",
    apply_forward_clamp=True,
)
robotics_num_chunks = 5
robotics_chunk_length = 16
robotics_chunk_starts = [chunk_idx * robotics_chunk_length for chunk_idx in range(robotics_num_chunks)]
assert robotics_chunk_starts[-1] < len(robotics_dataset)

COSMOS3_INPUT_DIR.mkdir(parents=True, exist_ok=True)
robotics_initial_vision_path = COSMOS3_INPUT_DIR / "robotics_droid_autoregressive_input_chunk_00.png"
robotics_records = []

for chunk_idx, sample_idx in enumerate(robotics_chunk_starts):
    robotics_sample = robotics_dataset[sample_idx]
    assert int(robotics_sample["action"].shape[0]) == robotics_chunk_length

    chunk_name = f"robotics_action_cond_chunk_{chunk_idx:02d}"
    robotics_action_path = COSMOS3_INPUT_DIR / f"robotics_droid_action_chunk_{chunk_idx:02d}.json"
    robotics_action_path.write_text(json.dumps(robotics_sample["action"].cpu().tolist()))

    if chunk_idx == 0:
        first_frame = robotics_sample["video"][:, 0].permute(1, 2, 0).cpu().numpy()
        Image.fromarray(first_frame).save(robotics_initial_vision_path)
        vision_path = robotics_initial_vision_path
    else:
        vision_path = COSMOS3_INPUT_DIR / f"robotics_droid_autoregressive_input_chunk_{chunk_idx:02d}.png"

    robotics_records.append(
        {
            "action_chunk_size": robotics_chunk_length,
            "action_path": str(robotics_action_path),
            "domain_name": "droid_lerobot",
            "fps": int(robotics_sample["conditioning_fps"]),
            "image_size": 480,
            "view_point": robotics_sample["viewpoint"],
            "model_mode": "forward_dynamics",
            "name": chunk_name,
            "prompt": robotics_sample["ai_caption"],
            "seed": 0,
            "vision_path": str(vision_path),
        }
    )

robotics_fd_input_path = COSMOS3_INPUT_DIR / "action_forward_dynamics_robotics_custom.jsonl"
robotics_fd_input_path.write_text("".join(json.dumps(r) + "\n" for r in robotics_records))
robotics_fd_output_dir = COSMOS3_OUTPUT_ROOT / "action_forward_dynamics_robotics_custom"

os.environ["COSMOS3_ROBOTICS_FD_INPUT"] = str(robotics_fd_input_path)
os.environ["COSMOS3_ROBOTICS_FD_OUTPUT"] = str(robotics_fd_output_dir)

print("loaded DROID samples from:", robotics_dataset_root)
print("chunk starts:", robotics_chunk_starts)
print("total action frames:", robotics_num_chunks * robotics_chunk_length)
print("wrote GT initial frame:", robotics_initial_vision_path)
print("wrote robotics autoregressive plan:", robotics_fd_input_path)
print(robotics_fd_input_path.read_text())
```

### Run Robotics Autoregressive Forward-Dynamics Inference

Runs the active Cosmos3 model once per robotics chunk through vLLM-Omni. Chunk 0 uses the DROID GT first frame. After each chunk finishes, the cell extracts that chunk's last generated frame and uses it as the conditioning image for the next chunk. Guardrails are disabled for this robotics run via `extra_params={"guardrails": false}`.

Each request sets top-level `size` to the current conditioning image resolution so vLLM-Omni returns each autoregressive chunk without reflection padding.

```python
import json
import mimetypes
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

try:
    import requests
except ImportError as exc:
    raise RuntimeError("Install requests in this notebook kernel: pip install requests") from exc

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def check_vllm_server(timeout_s: int = 600, interval_s: int = 10) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=10)
            response.raise_for_status()
            print(response.json())
            return
        except requests.RequestException as exc:
            last_error = exc
            print(f"Waiting for vLLM server at {VLLM_BASE_URL}: {exc}")
            time.sleep(interval_s)
    raise RuntimeError(
        f"vLLM server did not become ready at {VLLM_BASE_URL} within {timeout_s}s. "
        "Check `docker logs -f cosmos3-vllm-omni-notebook`."
    ) from last_error


def submit_forward_dynamics(record: dict, fd_output_dir: Path, *, disable_guardrails: bool = False) -> dict:
    run_dir = fd_output_dir / record["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    vision_path = Path(record["vision_path"])
    input_width, input_height = Image.open(vision_path).size
    mime_type = mimetypes.guess_type(vision_path.name)[0] or "application/octet-stream"
    extra_params = {
        "action_mode": "forward_dynamics",
        "domain_name": record["domain_name"],
        "action_chunk_size": record["action_chunk_size"],
        "image_size": record["image_size"],
        "action": json.loads(Path(record["action_path"]).read_text()),
        "guardrails": False,
    }
    if disable_guardrails:
        extra_params["guardrails"] = False

    prompt = build_action_prompt(
        record, width=input_width, height=input_height, default_prompt="A robot manipulates an object."
    )
    form = {
        "prompt": prompt,
        "num_frames": record["action_chunk_size"] + 1,
        "fps": record["fps"],
        "size": f"{input_width}x{input_height}",
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 10.0,
        "seed": record["seed"],
        "extra_params": json.dumps(extra_params),
    }

    with vision_path.open("rb") as image_file:
        response = requests.post(
            f"{VLLM_BASE_URL}/v1/videos",
            data={key: str(value) for key, value in form.items()},
            files={"input_reference": (vision_path.name, image_file, mime_type)},
            timeout=120,
        )
    if not response.ok:
        (run_dir / "error_response.txt").write_text(response.text)
        print("vLLM request failed:", response.status_code)
        print(response.text)
        print("form:", json.dumps(form, indent=2))
        print("extra_params keys:", sorted(extra_params))
        print("action shape:", [len(extra_params["action"]), len(extra_params["action"][0]) if extra_params["action"] else 0])
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

    response = requests.get(f"{VLLM_BASE_URL}/v1/videos/{initial['id']}/content", timeout=300)
    response.raise_for_status()
    video_path = run_dir / "vision.mp4"
    video_path.write_bytes(response.content)

    action = final.get("action")
    if action is not None:
        (run_dir / "action.json").write_text(json.dumps(action, indent=2))

    print("saved", video_path)
    if action is not None:
        print("action shape:", action.get("shape"), "dtype:", action.get("dtype"))
    return {"record": record, "initial": initial, "final": final, "run_dir": run_dir, "video_path": video_path, "action": action}


check_vllm_server()
robotics_results = []
robotics_actual_records = []
current_vision_path = Path(robotics_records[0]["vision_path"])
assert current_vision_path.exists(), f"missing initial conditioning image: {current_vision_path}"

for chunk_idx, base_record in enumerate(robotics_records):
    record = dict(base_record)
    record["vision_path"] = str(current_vision_path)
    robotics_records[chunk_idx]["vision_path"] = str(current_vision_path)
    robotics_actual_records.append(record)

    print(f"\nSubmitting {record['name']}")
    print("conditioning image:", current_vision_path)
    result = submit_forward_dynamics(record, robotics_fd_output_dir, disable_guardrails=True)
    robotics_results.append(result)

    if chunk_idx + 1 < len(robotics_records):
        next_vision_path = COSMOS3_INPUT_DIR / f"robotics_droid_autoregressive_input_chunk_{chunk_idx + 1:02d}.png"
        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(result["video_path"]),
                "-vf",
                fr"select=eq(n\,{record['action_chunk_size']})",
                "-frames:v",
                "1",
                str(next_vision_path),
            ],
            check=True,
        )
        assert next_vision_path.exists(), f"failed to extract next conditioning image: {next_vision_path}"
        current_vision_path = next_vision_path

robotics_fd_input_path.write_text("".join(json.dumps(r) + "\n" for r in robotics_actual_records))
print("wrote autoregressive run spec:", robotics_fd_input_path)
print("completed chunks:", [record["name"] for record in robotics_actual_records])
```

### Stitch Robotics Generated Chunks

Each autoregressive chunk video includes its conditioning frame at frame 0. This cell drops that first frame from every chunk and concatenates the remaining 16 generated frames per chunk into one 80-frame video.

```python
import subprocess
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
robotics_stitch_dir = robotics_fd_output_dir / "_stitched_segments"
robotics_stitch_dir.mkdir(parents=True, exist_ok=True)

segment_paths = []
for record in robotics_records:
    src = robotics_fd_output_dir / record["name"] / "vision.mp4"
    assert src.exists(), f"missing: {src}"

    segment = robotics_stitch_dir / f"{record['name']}_generated.mp4"
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vf",
            r"select=gte(n\,1),setpts=N/FRAME_RATE/TB",
            "-an",
            "-r",
            str(record["fps"]),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            str(segment),
        ],
        check=True,
    )
    segment_paths.append(segment)

concat_file = robotics_stitch_dir / "concat.txt"
concat_file.write_text("".join(f"file '{path.as_posix()}'\n" for path in segment_paths))

robotics_stitched_video_path = robotics_fd_output_dir / "robotics_action_cond_stitched.mp4"
subprocess.run(
    [
        FFMPEG,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(robotics_stitched_video_path),
    ],
    check=True,
)

print("stitched robotics video:", robotics_stitched_video_path)
print("expected generated frames:", len(robotics_records) * robotics_chunk_length)
print("size KB:", robotics_stitched_video_path.stat().st_size // 1024)
```

### Visualize Robotics Generated Videos

`Video(..., embed=True)` base64-inlines a file into the notebook, and embedding full-resolution runs can freeze the front-end. This cell first displays a compact preview of the stitched 80-frame video when available, then previews each per-chunk video. The full-resolution `vision.mp4` files and stitched mp4 are left untouched.

```python
import subprocess
import imageio_ffmpeg
from IPython.display import Video, display

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def make_preview(src: Path, crf: int = 28) -> Path:
    """Re-encode `src` to a compact, browser-friendly mp4 (cached)."""
    preview = src.with_name(f"{src.stem}_preview.mp4")
    if not preview.exists() or preview.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
             "-c:v", "libx264", "-crf", str(crf),
             "-preset", "veryfast", "-an", "-pix_fmt", "yuv420p", str(preview)],
            check=True,
        )
    return preview


if "robotics_stitched_video_path" in globals():
    assert robotics_stitched_video_path.exists(), f"missing: {robotics_stitched_video_path}"
    stitched_preview = make_preview(robotics_stitched_video_path)
    print(
        f"stitched  ({robotics_stitched_video_path.stat().st_size // 1024} KB -> "
        f"{stitched_preview.stat().st_size // 1024} KB preview)"
    )
    display(Video(str(stitched_preview), embed=True))

for record in robotics_records:
    name = record["name"]
    src = robotics_fd_output_dir / name / "vision.mp4"
    assert src.exists(), f"missing: {src}"
    preview = make_preview(src)
    print(f"{name}  ({src.stat().st_size // 1024} KB -> {preview.stat().st_size // 1024} KB preview)")
    display(Video(str(preview), embed=True))
```

## UMI

This example runs UMI forward dynamics through vLLM-Omni autoregressively over all 16-action chunks in `assets/actions/umi.json`. The action file stores the raw UMI 10D action representation, so the setup cell validates the row dimension, writes one action JSON per chunk, and prepares a run plan.

### Create the UMI Autoregressive Forward-Dynamics Plan

The UMI action file is stored as one JSON array with `16 * n` action rows. vLLM-Omni receives one request per 16-action chunk. Chunk 0 uses the checked-in UMI conditioning image; later chunks use conditioning images extracted from the previous generated video.

```python
import json
from pathlib import Path

umi_input_image = "cookbooks/cosmos3/generator/action/assets/images/umi.png"
umi_input_action = "cookbooks/cosmos3/generator/action/assets/actions/umi.json"
umi_prompt = "mouse arrangement"
umi_fps = 20
umi_action_chunk_size = 16
umi_raw_action_dim = 10

umi_initial_vision_path = Path(resolve_input(umi_input_image))
umi_source_action_path = Path(resolve_input(umi_input_action))
umi_action = json.loads(umi_source_action_path.read_text())
assert len(umi_action) % umi_action_chunk_size == 0, (
    f"expected action count to be divisible by {umi_action_chunk_size}, got {len(umi_action)}"
)
assert all(len(row) == umi_raw_action_dim for row in umi_action), "UMI action rows must be 10D"

umi_num_chunks = len(umi_action) // umi_action_chunk_size
COSMOS3_INPUT_DIR.mkdir(parents=True, exist_ok=True)
umi_records = []

for chunk_idx in range(umi_num_chunks):
    chunk_name = f"umi_action_cond_chunk_{chunk_idx:02d}"
    start = chunk_idx * umi_action_chunk_size
    end = start + umi_action_chunk_size
    action_chunk_10d = umi_action[start:end]
    umi_action_path = COSMOS3_INPUT_DIR / f"umi_action_chunk_{chunk_idx:02d}_10d.json"
    umi_action_path.write_text(json.dumps(action_chunk_10d, indent=2) + "\n")

    if chunk_idx == 0:
        vision_path = umi_initial_vision_path
    else:
        vision_path = COSMOS3_INPUT_DIR / f"umi_autoregressive_input_chunk_{chunk_idx:02d}.png"

    umi_records.append(
        {
            "action_chunk_size": umi_action_chunk_size,
            "action_path": str(umi_action_path),
            "domain_name": "umi",
            "fps": umi_fps,
            "image_size": 256,
            "view_point": "ego_view",
            "model_mode": "forward_dynamics",
            "name": chunk_name,
            "prompt": umi_prompt,
            "seed": chunk_idx,
            "vision_path": str(vision_path),
        }
    )

umi_fd_input_path = COSMOS3_INPUT_DIR / "action_forward_dynamics_umi_custom.jsonl"
umi_fd_input_path.write_text("".join(json.dumps(r) + "\n" for r in umi_records))
umi_fd_output_dir = COSMOS3_OUTPUT_ROOT / "action_forward_dynamics_umi_custom"

print("UMI chunks:", umi_num_chunks)
print("wrote UMI spec:", umi_fd_input_path)
print(umi_fd_input_path.read_text())
```

### Run UMI Autoregressive Forward-Dynamics Inference

Runs one vLLM-Omni video request per UMI action chunk. After each chunk completes, the cell extracts that chunk's last generated frame and uses it as the conditioning image for the next chunk. Each request sets top-level `size` to the current conditioning image resolution to avoid reflection padding.

```python
import json
import mimetypes
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

try:
    import requests
except ImportError as exc:
    raise RuntimeError("Install requests in this notebook kernel: pip install requests") from exc

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def check_vllm_server_for_umi(timeout_s: int = 600, interval_s: int = 10) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=10)
            response.raise_for_status()
            print(response.json())
            return
        except requests.RequestException as exc:
            last_error = exc
            print(f"Waiting for vLLM server at {VLLM_BASE_URL}: {exc}")
            time.sleep(interval_s)
    raise RuntimeError(
        f"vLLM server did not become ready at {VLLM_BASE_URL} within {timeout_s}s. "
        "Check `docker logs -f cosmos3-vllm-omni-notebook`."
    ) from last_error


def submit_umi_forward_dynamics(record: dict, fd_output_dir: Path) -> dict:
    run_dir = fd_output_dir / record["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    vision_path = Path(record["vision_path"])
    input_width, input_height = Image.open(vision_path).size
    mime_type = mimetypes.guess_type(vision_path.name)[0] or "application/octet-stream"
    extra_params = {
        "action_mode": "forward_dynamics",
        "domain_name": record["domain_name"],
        "action_chunk_size": record["action_chunk_size"],
        "image_size": record["image_size"],
        "action": json.loads(Path(record["action_path"]).read_text()),
        "guardrails": False,
    }
    prompt = build_action_prompt(
        record, width=input_width, height=input_height, default_prompt="A robot manipulates an object."
    )
    form = {
        "prompt": prompt,
        "num_frames": record["action_chunk_size"] + 1,
        "fps": record["fps"],
        "size": f"{input_width}x{input_height}",
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 10.0,
        "seed": record["seed"],
        "extra_params": json.dumps(extra_params),
    }

    with vision_path.open("rb") as image_file:
        response = requests.post(
            f"{VLLM_BASE_URL}/v1/videos",
            data={key: str(value) for key, value in form.items()},
            files={"input_reference": (vision_path.name, image_file, mime_type)},
            timeout=120,
        )
    if not response.ok:
        (run_dir / "error_response.txt").write_text(response.text)
        print("vLLM request failed:", response.status_code)
        print(response.text)
        print("form:", json.dumps(form, indent=2))
        print("extra_params keys:", sorted(extra_params))
        print("action shape:", [len(extra_params["action"]), len(extra_params["action"][0]) if extra_params["action"] else 0])
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

    response = requests.get(f"{VLLM_BASE_URL}/v1/videos/{initial['id']}/content", timeout=300)
    response.raise_for_status()
    video_path = run_dir / "vision.mp4"
    video_path.write_bytes(response.content)
    print("saved", video_path)
    return {"record": record, "initial": initial, "final": final, "run_dir": run_dir, "video_path": video_path}


check_vllm_server_for_umi()
umi_results = []
umi_actual_records = []
current_vision_path = Path(umi_records[0]["vision_path"])
assert current_vision_path.exists(), f"missing initial conditioning image: {current_vision_path}"

for chunk_idx, base_record in enumerate(umi_records):
    record = dict(base_record)
    record["vision_path"] = str(current_vision_path)
    umi_records[chunk_idx]["vision_path"] = str(current_vision_path)
    umi_actual_records.append(record)

    print(f"\nSubmitting {record['name']}")
    print("conditioning image:", current_vision_path)
    result = submit_umi_forward_dynamics(record, umi_fd_output_dir)
    umi_results.append(result)

    if chunk_idx + 1 < len(umi_records):
        next_vision_path = COSMOS3_INPUT_DIR / f"umi_autoregressive_input_chunk_{chunk_idx + 1:02d}.png"
        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(result["video_path"]),
                "-vf",
                fr"select=eq(n\,{record['action_chunk_size']})",
                "-frames:v",
                "1",
                str(next_vision_path),
            ],
            check=True,
        )
        assert next_vision_path.exists(), f"failed to extract next conditioning image: {next_vision_path}"
        current_vision_path = next_vision_path

umi_fd_input_path.write_text("".join(json.dumps(r) + "\n" for r in umi_actual_records))
print("wrote autoregressive UMI run spec:", umi_fd_input_path)
print("completed UMI chunks:", [record["name"] for record in umi_actual_records])
```

### Stitch and Visualize UMI Generated Chunks

Each autoregressive chunk video includes its conditioning frame at frame 0. This cell drops that first frame from every chunk, concatenates the generated frames into one rollout video, transcodes a compact preview, and embeds it in the notebook.

```python
import subprocess

import imageio_ffmpeg
from IPython.display import Video, display

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
umi_video_paths = [umi_fd_output_dir / record["name"] / "vision.mp4" for record in umi_records]
for path in umi_video_paths:
    assert path.exists(), f"missing UMI chunk video: {path}"

umi_stitch_dir = umi_fd_output_dir / "_stitched_segments"
umi_stitch_dir.mkdir(parents=True, exist_ok=True)
segment_paths = []
for record, src in zip(umi_records, umi_video_paths, strict=True):
    segment = umi_stitch_dir / f"{record['name']}_generated.mp4"
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vf",
            r"select=gte(n\,1),setpts=N/FRAME_RATE/TB",
            "-an",
            "-r",
            str(record["fps"]),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            str(segment),
        ],
        check=True,
    )
    segment_paths.append(segment)

concat_file = umi_stitch_dir / "umi_concat.txt"
concat_file.write_text("".join(f"file '{path.as_posix()}'\n" for path in segment_paths))
umi_stitched_video_path = umi_fd_output_dir / "umi_action_cond_stitched.mp4"
subprocess.run(
    [
        FFMPEG,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(umi_stitched_video_path),
    ],
    check=True,
)


def make_umi_preview(src: Path, crf: int = 28) -> Path:
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

umi_preview_path = make_umi_preview(umi_stitched_video_path)
print("stitched UMI video:", umi_stitched_video_path)
print("expected generated frames:", len(umi_records) * umi_action_chunk_size)
print(f"UMI preview: {umi_preview_path}")
display(Video(str(umi_preview_path), embed=True))
```

## Human Hand Pose

This section runs a 57D bimanual human hand-pose forward-dynamics example from a single validation episode. The source video and hand annotations are synchronized at 30 FPS and sampled at 15 FPS for Cosmos3-Nano.

### Create the Hand-Pose Forward-Dynamics Input Spec

```python
import json
import sys

from PIL import Image

if str(COSMOS3_REPO) not in sys.path:
    sys.path.insert(0, str(COSMOS3_REPO))

from cosmos_framework.data.generator.action.datasets import HumanHandPoseLeRobotDataset

hand_pose_dataset_root = resolve_input(
    "cookbooks/cosmos3/generator/action/assets/human_hand_pose_lerobot_example"
)
hand_pose_dataset = HumanHandPoseLeRobotDataset(root=hand_pose_dataset_root)
hand_pose_sample = hand_pose_dataset[0]
hand_pose_chunk_length = 16
assert tuple(hand_pose_sample["action"].shape) == (hand_pose_chunk_length, 57)

hand_pose_action_path = COSMOS3_INPUT_DIR / "human_hand_pose_action_chunk_00.json"
hand_pose_action_path.write_text(json.dumps(hand_pose_sample["action"].cpu().tolist()))
hand_pose_vision_path = COSMOS3_INPUT_DIR / "human_hand_pose_input_chunk_00.png"
first_frame = hand_pose_sample["video"][:, 0].permute(1, 2, 0).cpu().numpy()
Image.fromarray(first_frame).save(hand_pose_vision_path)

hand_pose_record = {
    "action_chunk_size": hand_pose_chunk_length,
    "action_path": str(hand_pose_action_path),
    "domain_name": "hand_pose",
    "fps": int(hand_pose_sample["conditioning_fps"]),
    "image_size": 480,
    "view_point": hand_pose_sample["viewpoint"],
    "model_mode": "forward_dynamics",
    "name": "human_hand_pose_action_cond_chunk_00",
    "prompt": hand_pose_sample["ai_caption"],
    "seed": 0,
    "vision_path": str(hand_pose_vision_path),
}
hand_pose_fd_input_path = COSMOS3_INPUT_DIR / "action_forward_dynamics_hand_pose.jsonl"
hand_pose_fd_input_path.write_text(json.dumps(hand_pose_record) + "\n")
hand_pose_fd_output_dir = COSMOS3_OUTPUT_ROOT / "action_forward_dynamics_hand_pose"

print("loaded hand-pose sample from:", hand_pose_dataset_root)
print("caption:", hand_pose_record["prompt"])
print("action shape:", tuple(hand_pose_sample["action"].shape))
print(hand_pose_fd_input_path.read_text())
```

### Run Hand-Pose Forward-Dynamics Inference

```python
hand_pose_result = submit_forward_dynamics(
    hand_pose_record,
    hand_pose_fd_output_dir,
    disable_guardrails=True,
)
hand_pose_generated_video = hand_pose_result["video_path"]
assert hand_pose_generated_video.exists(), f"missing generated video: {hand_pose_generated_video}"
print("generated:", hand_pose_generated_video)
```

### Visualize the Hand-Pose Generated Video

```python
from IPython.display import Video, display

display(Video(str(hand_pose_generated_video), embed=True))
```
