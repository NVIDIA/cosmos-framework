<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Action-Conditioned Inverse Dynamics with Diffusers

Inverse dynamics runs the other direction from forward dynamics: the input is a clip and no trajectory, and the model recovers the motion that connects the frames. It runs `Cosmos3OmniPipeline` with the clip grouped into a `CosmosActionCondition`.

It runs the two AV clips plus the Bridge, AgiBotWorld-Beta, RoboMIND Franka, RoboMIND Franka dual-arm, RoboMIND UR, UMI, and Fractal episodes under [`assets/`](../../cosmos3-inference-action/assets). For forward dynamics, see [`run_fd_with_diffusers.ipynb`](../../cosmos3-inference-action-forward-dynamics/references/diffusers.md).

FP8 checkpoints: set `COSMOS3_DIFFUSERS_PRECISION=fp8` before loading. Nano and Super use the dedicated Hub repos `nvidia/Cosmos3-Nano-FP8` and `nvidia/Cosmos3-Super-FP8`. Use-case cells stay the same.

> **Skill note (not in the source):** this file is the cookbook notebook `run_id_with_diffusers.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/generator/action/run_id_with_diffusers.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

<!-- rumdl-disable MD034 -->
<!-- Skill note (not in the source): the directive above keeps the notebook's bare URL in this section's "uv version" note as written; the rule is re-enabled right after the section. -->

## 1. Prerequisites

Use a Linux machine with NVIDIA GPU access, model access on Hugging Face, and either `uvx hf@latest auth login` or `HF_TOKEN` set.

Predicted actions are written as JSON in model-normalized action space and plotted as camera trajectories.

The robot episodes are read with the LeRobot readers from the Cosmos framework, and predicted actions are plotted with its pose helpers. Point `COSMOS3_REPO` at your framework checkout (it defaults to `packages/cosmos3` beside this repo).

Generator requires the Guardrail. Request access to the gated [nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail) HF repository before running these examples. To disable the guardrail, set `COSMOS3_DIFFUSERS_GUARDRAILS=false` before running the helper cell.

> **Headless servers:** if you see an error like `libxcb.so.1: cannot open shared object file` (a missing system graphics library) when importing or running the pipeline, install the required system libraries:
>
> ```bash
> apt-get install -y libxcb1 libgl1 libglib2.0-0
> ```

> **uv version:** use `uv 0.11.29`, matching the version pinned by Cosmos Framework. Install it from https://astral.sh/uv/0.11.29/install.sh or update an existing self-managed installation with `uv self update 0.11.29`.

<!-- rumdl-enable MD034 -->

## 2. Configure Paths and Environment

The defaults are relative to this `cosmos` checkout and use the CUDA 13 or 12.8 Torch backend depending on the CUDA version installed on your system (`cu130` or `cu128`):

```bash
export COSMOS3_DIFFUSERS_ACTION_VENV=/path/to/.venv-cosmos3-diffusers-action
export COSMOS3_REPO=/path/to/packages/cosmos3
export COSMOS3_TORCH_BACKEND=cu130
export HF_HOME=/path/to/large/huggingface/cache
export UV_LINK_MODE=copy
export CUDA_VISIBLE_DEVICES=0
export COSMOS3_DIFFUSERS_PRECISION=bf16  # or fp8
```

```python
from pathlib import Path
import os


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "cookbooks" / "cosmos3").is_dir():
            return path
    return start


def configure_diffusers_environment() -> None:
    global COSMOS_ROOT
    global COSMOS3_ACTION_ROOT
    global COSMOS3_DIFFUSERS_ACTION_VENV
    global COSMOS3_TORCH_BACKEND
    global COSMOS3_ACTION_OUTPUT_ROOT
    global COSMOS3_REPO

    COSMOS_ROOT = find_repo_root(Path.cwd().resolve())
    COSMOS3_ACTION_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3" / "generator" / "action"
    COSMOS3_DIFFUSERS_ACTION_VENV = Path(
        os.environ.get("COSMOS3_DIFFUSERS_ACTION_VENV", COSMOS_ROOT / ".venv-cosmos3-diffusers-action")
    ).resolve()
    COSMOS3_TORCH_BACKEND = os.environ.get("COSMOS3_TORCH_BACKEND", "cu130")
    COSMOS3_REPO = Path(os.environ.get("COSMOS3_REPO", COSMOS_ROOT / "packages" / "cosmos3")).resolve()
    COSMOS3_ACTION_OUTPUT_ROOT = Path(
        os.environ.get(
            "COSMOS3_ACTION_OUTPUT_ROOT", COSMOS3_ACTION_ROOT / "outputs" / "notebooks" / "diffusers"
        )
    ).resolve()

    os.environ["COSMOS3_DIFFUSERS_ACTION_VENV"] = str(COSMOS3_DIFFUSERS_ACTION_VENV)
    os.environ["COSMOS3_TORCH_BACKEND"] = COSMOS3_TORCH_BACKEND
    os.environ["COSMOS3_ACTION_OUTPUT_ROOT"] = str(COSMOS3_ACTION_OUTPUT_ROOT)
    os.environ["COSMOS3_REPO"] = str(COSMOS3_REPO)
    os.environ.setdefault("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv"))
    os.environ.setdefault("UV_LINK_MODE", "copy")
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("COSMOS3_DIFFUSERS_PRECISION", "bf16")

    print(f"COSMOS_ROOT: {COSMOS_ROOT}")
    for key in [
        "COSMOS3_DIFFUSERS_ACTION_VENV",
        "COSMOS3_TORCH_BACKEND",
        "COSMOS3_ACTION_OUTPUT_ROOT",
        "COSMOS3_REPO",
        "UV_CACHE_DIR",
        "UV_LINK_MODE",
        "HF_HOME",
        "HF_HUB_DISABLE_XET",
        "CUDA_VISIBLE_DEVICES",
        "COSMOS3_DIFFUSERS_PRECISION",
    ]:
        print(f"{key}: {os.environ[key]}")
    print("HF_TOKEN:", "<set>" if os.environ.get("HF_TOKEN") else "<unset>")
    if not (COSMOS3_REPO / "cosmos_framework").is_dir():
        print(f"note: no cosmos_framework package under {COSMOS3_REPO}; point COSMOS3_REPO at your framework checkout")


configure_diffusers_environment()
```

## 3. Install Diffusers Dependencies

```bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
uv venv "$COSMOS3_DIFFUSERS_ACTION_VENV" --python 3.13 --seed --managed-python --allow-existing
source "$COSMOS3_DIFFUSERS_ACTION_VENV/bin/activate"

# The LeRobot readers, pose helpers, and trajectory plots add parquet support and plotting on top
# of the diffusers stack.
uv pip install --torch-backend="$COSMOS3_TORCH_BACKEND" \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  "lerobot @ https://github.com/huggingface/lerobot/archive/1a4316c6845330bc552fb982dbc44bdb4f66f2f1.tar.gz" \
  accelerate \
  av \
  cosmos_guardrail \
  datasets \
  draccus \
  huggingface_hub \
  imageio \
  imageio-ffmpeg \
  ipykernel \
  loguru \
  matplotlib \
  mujoco \
  pandas \
  pyarrow \
  scipy \
  torch \
  torchvision \
  transformers \
  "nvidia-modelopt==0.44.0"

# Video decoding uses the PyAV fallback defined in the helpers cell, avoiding TorchCodec ABI
# coupling with the selected PyTorch backend.

"$COSMOS3_DIFFUSERS_ACTION_VENV/bin/python" -m ipykernel install --user \
  --name cosmos3-diffusers-action \
  --display-name "Cosmos3 Diffusers Action (Python 3.13)"

echo
echo "Installed dependencies into: $COSMOS3_DIFFUSERS_ACTION_VENV"
echo "Next: switch this notebook kernel to: Cosmos3 Diffusers Action (Python 3.13)"
echo "After switching kernels, run the Restore Environment cell below, then continue with Verify."
```

## 4. Select the Diffusers Action Kernel

The install cell creates and registers the `Cosmos3 Diffusers Action (Python 3.13)` Jupyter kernel.

**Note**: Switch this notebook to that kernel before running the remaining Python cells, then run the restore cell immediately below. It can take some time for the new Jupyter kernel to show up in the notebook interface.

```python
# Run this cell immediately after switching to the Cosmos3 Diffusers Action kernel.
# It restores the same paths and cache settings as the setup cell above.
from pathlib import Path
import os


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "cookbooks" / "cosmos3").is_dir():
            return path
    return start


def configure_diffusers_environment() -> None:
    global COSMOS_ROOT
    global COSMOS3_ACTION_ROOT
    global COSMOS3_DIFFUSERS_ACTION_VENV
    global COSMOS3_TORCH_BACKEND
    global COSMOS3_ACTION_OUTPUT_ROOT
    global COSMOS3_REPO

    COSMOS_ROOT = find_repo_root(Path.cwd().resolve())
    COSMOS3_ACTION_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3" / "generator" / "action"
    COSMOS3_DIFFUSERS_ACTION_VENV = Path(
        os.environ.get("COSMOS3_DIFFUSERS_ACTION_VENV", COSMOS_ROOT / ".venv-cosmos3-diffusers-action")
    ).resolve()
    COSMOS3_TORCH_BACKEND = os.environ.get("COSMOS3_TORCH_BACKEND", "cu130")
    COSMOS3_REPO = Path(os.environ.get("COSMOS3_REPO", COSMOS_ROOT / "packages" / "cosmos3")).resolve()
    COSMOS3_ACTION_OUTPUT_ROOT = Path(
        os.environ.get(
            "COSMOS3_ACTION_OUTPUT_ROOT", COSMOS3_ACTION_ROOT / "outputs" / "notebooks" / "diffusers"
        )
    ).resolve()

    os.environ["COSMOS3_DIFFUSERS_ACTION_VENV"] = str(COSMOS3_DIFFUSERS_ACTION_VENV)
    os.environ["COSMOS3_TORCH_BACKEND"] = COSMOS3_TORCH_BACKEND
    os.environ["COSMOS3_ACTION_OUTPUT_ROOT"] = str(COSMOS3_ACTION_OUTPUT_ROOT)
    os.environ["COSMOS3_REPO"] = str(COSMOS3_REPO)
    os.environ.setdefault("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv"))
    os.environ.setdefault("UV_LINK_MODE", "copy")
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("COSMOS3_DIFFUSERS_PRECISION", "bf16")

    print(f"COSMOS_ROOT: {COSMOS_ROOT}")
    for key in [
        "COSMOS3_DIFFUSERS_ACTION_VENV",
        "COSMOS3_TORCH_BACKEND",
        "COSMOS3_ACTION_OUTPUT_ROOT",
        "COSMOS3_REPO",
        "UV_CACHE_DIR",
        "UV_LINK_MODE",
        "HF_HOME",
        "HF_HUB_DISABLE_XET",
        "CUDA_VISIBLE_DEVICES",
        "COSMOS3_DIFFUSERS_PRECISION",
    ]:
        print(f"{key}: {os.environ[key]}")
    print("HF_TOKEN:", "<set>" if os.environ.get("HF_TOKEN") else "<unset>")
    if not (COSMOS3_REPO / "cosmos_framework").is_dir():
        print(f"note: no cosmos_framework package under {COSMOS3_REPO}; point COSMOS3_REPO at your framework checkout")


configure_diffusers_environment()
```

## 5. Verify GPU and Python Environment

```python
import os
import sys
from pathlib import Path

if "COSMOS3_DIFFUSERS_ACTION_VENV" not in os.environ:
    raise RuntimeError("Run the Restore Environment cell after switching to the Diffusers kernel.")

expected_venv = Path(os.environ["COSMOS3_DIFFUSERS_ACTION_VENV"]).resolve()
current_venv = Path(sys.prefix).resolve()
print("kernel executable:", sys.executable)
print("kernel venv:", current_venv)
print("expected venv:", expected_venv)
if current_venv != expected_venv:
    raise RuntimeError(
        "This notebook is not running inside the Diffusers Action venv. "
        "Switch the notebook kernel to 'Cosmos3 Diffusers Action (Python 3.13)', then run the Restore Environment cell above."
    )

import torch
import diffusers

print("diffusers:", diffusers.__version__)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
```

## 6. Preview the Input Clips

```python
import json
from pathlib import Path

for video_name in ["av_0.mp4", "av_1.mp4"]:
    video_path = COSMOS3_ACTION_ROOT / "assets" / "videos" / video_name
    print(f"{video_path.relative_to(COSMOS_ROOT)} ({video_path.stat().st_size // 1024} KB)")

for episode in [
    "assets/bridge_lerobot_example",
    "assets/agibotworld_beta_lerobot_example",
    "assets/robomind_lerobot_example/franka",
    "assets/robomind_lerobot_example/franka_dual",
    "assets/robomind_lerobot_example/ur",
    "assets/umi_lerobot_example",
    "assets/fractal_lerobot_example",
]:
    root = COSMOS3_ACTION_ROOT / episode
    info = json.loads((root / "meta" / "info.json").read_text())
    cameras = sorted(key for key, feature in info["features"].items() if feature["dtype"] == "video")
    print(f"{root.relative_to(COSMOS_ROOT)}: {info.get('robot_type') or 'unknown'}, "
          f"{info['total_frames']} frames at {info['fps']} fps, {len(cameras)} camera(s)")
```

## 7. Define Action Cases, Runner, and Viewer Helpers

Inverse dynamics reads an existing clip and predicts the action chunk that connects its frames, so it conditions on `action.video` and passes no `raw_actions`. The only output is the predicted action array; nothing is generated to view as a video.

Predicted AV actions are 9D framewise deltas — a 3D translation plus a 6D rotation — which integrate into absolute camera poses for plotting.

Robot episodes work the same way with two differences: their actions come back normalized, so they are denormalized with the episode's own statistics first, and a single action vector can carry several end-effectors, so each case names the 9D block to integrate. Longer episodes are covered by predicting one chunk per sampled window and concatenating the results.

```python
import base64
import gc
import html
import json
import os
import sys
import time
from pathlib import Path
from IPython.display import HTML, display

if "COSMOS3_DIFFUSERS_ACTION_VENV" not in os.environ:
    raise RuntimeError("Run the Restore Environment cell after switching to the Diffusers kernel.")
expected_python = (Path(os.environ["COSMOS3_DIFFUSERS_ACTION_VENV"]) / "bin" / "python").resolve()
if Path(sys.executable).resolve() != expected_python:
    raise RuntimeError("Switch the notebook kernel to 'Cosmos3 Diffusers Action (Python 3.13)' before running Diffusers cells.")

import av
import torch
import lerobot.datasets.video_utils as lerobot_video_utils

from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
from diffusers import logging as diffusers_logging
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image


def decode_video_frames_av(video_path, timestamps, tolerance_s, backend=None):
    """Decode the nearest requested RGB frames with PyAV, returning [T, C, H, W] in [0, 1]."""
    loaded_timestamps = []
    loaded_frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            loaded_timestamps.append(float(frame.pts * frame.time_base))
            loaded_frames.append(frame.to_ndarray(format="rgb24"))

    if not loaded_frames:
        raise ValueError(f"No video frames decoded from {video_path}")
    loaded_timestamps = torch.tensor(loaded_timestamps, dtype=torch.float64)
    query_timestamps = torch.tensor([float(value) for value in timestamps], dtype=torch.float64)
    distances = torch.cdist(query_timestamps[:, None], loaded_timestamps[:, None], p=1)
    min_distances, frame_indexes = distances.min(dim=1)
    if not bool((min_distances < tolerance_s).all()):
        raise ValueError(
            f"No frame within tolerance {tolerance_s}: nearest distances {min_distances.tolist()}"
        )

    frames = np.stack([loaded_frames[int(index)] for index in frame_indexes], axis=0)
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0


# Framework readers import this function lazily, so patching the shared LeRobot module here
# covers every checked-in robot episode without importing TorchCodec.
lerobot_video_utils.decode_video_frames = decode_video_frames_av

# The notebook kernel may differ from the framework venv, so put the repo on the
# path before importing `cosmos_framework`.
if str(COSMOS3_REPO) not in sys.path:
    sys.path.insert(0, str(COSMOS3_REPO))
from cosmos_framework.data.generator.action.action_normalization import denormalize_action
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
    plt.close(fig)

MODEL_IDS = {
    "Cosmos3-Nano": "nvidia/Cosmos3-Nano",
    "Cosmos3-Super": "nvidia/Cosmos3-Super",
}

# Guardrails are on by default. Set COSMOS3_DIFFUSERS_GUARDRAILS=false to skip the safety checker.
GUARDRAILS = os.environ.get("COSMOS3_DIFFUSERS_GUARDRAILS", "true").strip().lower() not in {"0", "false", "no", "off"}

# Diffusion defaults shared by every action example.
FIXED_SAMPLING = {
    "num_steps": 30,
    "guidance": 1.0,
    "shift": 10.0,
    "seed": 0,
}

AV_PROMPT = "You are an autonomous vehicle planning system."

# All asset paths are repo-relative under cookbooks/cosmos3/generator/action.
ACTION_SETS = {
    "av_inverse_0": {
        "mode": "inverse_dynamics",
        "domain_name": "av",
        "chunk_size": 60,
        "resolution_tier": 480,
        "view_point": "ego_view",
        "fps": 10,
        "prompt": AV_PROMPT,
        "video": "assets/videos/av_0.mp4",
    },
    "av_inverse_1": {
        "mode": "inverse_dynamics",
        "domain_name": "av",
        "chunk_size": 60,
        "resolution_tier": 480,
        "view_point": "ego_view",
        "fps": 10,
        "prompt": AV_PROMPT,
        "video": "assets/videos/av_1.mp4",
    },
}

# Robot episodes checked in under `assets/`, one entry per case: the reader to use, how many chunks
# to predict, and which 9D pose block of the predicted action to integrate for the plot.
LEROBOT_ID_SETS = {
    "bridge_inverse": {
        "reader": "BridgeOrigLeRobotDataset",
        "root": "assets/bridge_lerobot_example",
        "num_chunks": 1,
        "chunk_length": 16,
        "resolution_tier": 480,
        "pose_index": 0,
        "pose_label": "end-effector",
    },
    "agibotworld_inverse": {
        "reader": "AgiBotWorldBetaLeRobotDataset",
        "root": "assets/agibotworld_beta_lerobot_example",
        "reader_kwargs": {"fps": 10, "sample_stride": 3},
        "num_chunks": 5,
        "chunk_length": 16,
        "resolution_tier": 480,
        "pose_index": 9,
        "pose_label": "right wrist",
    },
    "robomind_franka_inverse": {
        "reader": "RoboMINDFrankaDataset",
        "root": "assets/robomind_lerobot_example/franka",
        "reader_kwargs": {"embodiment_type": "robomind-franka", "fps": 10},
        "num_chunks": 4,
        "chunk_length": 16,
        "resolution_tier": 480,
        "pose_index": 0,
        "pose_label": "left arm",
    },
    "robomind_franka_dual_inverse": {
        "reader": "RoboMINDFrankaDataset",
        "root": "assets/robomind_lerobot_example/franka_dual",
        "reader_kwargs": {"embodiment_type": "robomind-franka-dual", "fps": 10},
        "num_chunks": 5,
        "chunk_length": 16,
        "resolution_tier": 480,
        "pose_index": 10,
        "pose_label": "right arm",
    },
    "robomind_ur_inverse": {
        "reader": "RoboMINDURDataset",
        "root": "assets/robomind_lerobot_example/ur",
        "reader_kwargs": {"fps": 10},
        "num_chunks": 5,
        "chunk_length": 16,
        "resolution_tier": 480,
        "pose_index": 0,
        "pose_label": "left arm",
    },
    "umi_inverse": {
        "reader": "UMILeRobotDataset",
        "root": "assets/umi_lerobot_example",
        "num_chunks": 5,
        "chunk_length": 16,
        "resolution_tier": 480,
        "pose_index": 0,
        "pose_label": "end-effector",
    },
    "fractal_inverse": {
        "reader": "FractalLeRobotDataset",
        "root": "assets/fractal_lerobot_example",
        "num_chunks": 2,
        "chunk_length": 16,
        "resolution_tier": 480,
        "pose_index": 0,
        "pose_label": "end-effector",
    },
}

_pipe = None
_pipe_model = None


def asset_path(relative_path: str) -> Path:
    path = COSMOS3_ACTION_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    return path.resolve()


def case_prompt(spec: dict) -> str:
    if "prompt_path" in spec:
        return asset_path(spec["prompt_path"]).read_text().strip()
    return spec["prompt"]


def case_output_dir(case: str) -> Path:
    output_dir = Path(os.environ["COSMOS3_ACTION_OUTPUT_ROOT"]) / case
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def cuda_allocated_gib() -> float:
    return torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0


def release_pipe() -> None:
    global _pipe, _pipe_model
    if _pipe is None:
        return
    _pipe, _pipe_model = None, None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"released previous pipeline; cuda allocated {cuda_allocated_gib():.1f} GiB")



FP8_SEPARATE_REPOS = {
    "Cosmos3-Nano": "nvidia/Cosmos3-Nano-FP8",
    "Cosmos3-Super": "nvidia/Cosmos3-Super-FP8",
}


def precision_mode() -> str:
    return os.environ.get("COSMOS3_DIFFUSERS_PRECISION", "bf16").strip().lower()


def is_fp8_precision() -> bool:
    return precision_mode() == "fp8"


def resolve_pretrained_source(model: str):
    """Hub repo id or local path for the requested precision."""
    override = os.environ.get("COSMOS3_FP8_MODEL_PATH")
    if is_fp8_precision() and override:
        return Path(override).expanduser().resolve()
    if is_fp8_precision():
        if model not in FP8_SEPARATE_REPOS:
            raise RuntimeError(
                f"FP8 is only supported for {sorted(FP8_SEPARATE_REPOS)}; got {model}. "
                "Unset COSMOS3_DIFFUSERS_PRECISION or set it to bf16."
            )
        return FP8_SEPARATE_REPOS[model]
    return MODEL_IDS.get(model, model)


def ensure_modelopt_state(pretrained_source, *, token=None, revision=None) -> Path:
    """Ensure ModelOpt state is present before Diffusers resolves its Hub allowlist."""
    checkpoint_root = Path(str(pretrained_source))
    if isinstance(pretrained_source, Path) or checkpoint_root.is_absolute():
        state_path = checkpoint_root / "transformer" / "modelopt_state.pth"
    else:
        from huggingface_hub import hf_hub_download

        state_path = Path(
            hf_hub_download(
                repo_id=str(pretrained_source),
                filename="transformer/modelopt_state.pth",
                revision=revision,
                token=token,
            )
        )
    if not state_path.is_file():
        raise RuntimeError(
            f"FP8 requested but no ModelOpt state at {state_path}. "
            "Use a ModelOpt FP8 checkpoint or set COSMOS3_FP8_MODEL_PATH to one."
        )
    return state_path


def validate_fp8_transformer(transformer) -> tuple[int, int, int]:
    """Smoke-check that ModelOpt restored FP8 weights and the quantizer graph.

    Counts float8 weights, quantizer modules, and meta tensors. It does not check
    that the W8A8 GEMM path is active.
    """
    tensors = list(transformer.named_parameters()) + list(transformer.named_buffers())
    fp8 = sum(parameter.dtype == torch.float8_e4m3fn for parameter in transformer.parameters())
    quantizers = sum("quantizer" in name.lower() for name, _ in transformer.named_modules())
    meta = sum(tensor.is_meta for _, tensor in tensors)
    if not fp8 or not quantizers or meta:
        raise RuntimeError(f"Invalid FP8 load: weights={fp8}, quantizers={quantizers}, meta={meta}")
    return fp8, quantizers, meta


def get_pipe(model: str) -> Cosmos3OmniPipeline:
    global _pipe, _pipe_model
    prec = precision_mode()
    pretrained_source = resolve_pretrained_source(model)
    cache_key = f"{pretrained_source}@{prec}"
    if _pipe is not None and _pipe_model == cache_key:
        return _pipe
    release_pipe()
    diffusers_logging.set_verbosity_info()
    t0 = time.time()
    fp8 = is_fp8_precision()
    token = os.environ.get("HF_TOKEN") or None
    if fp8:
        # ModelOpt FP8 graph restore + kernels (required before from_pretrained).
        import modelopt.torch.quantization.backends.fp8_per_tensor_gemm  # noqa: F401
        from modelopt.torch.opt import enable_huggingface_checkpointing

        enable_huggingface_checkpointing()
        state_path = ensure_modelopt_state(pretrained_source, token=token)
        print(f"ModelOpt state: {state_path}")
        print(f"loading FP8 {pretrained_source}...")
    else:
        print(f"loading {pretrained_source}...")
    pipe = Cosmos3OmniPipeline.from_pretrained(
        pretrained_source,
        torch_dtype=torch.bfloat16,
        safety_checker=None,
        enable_safety_checker=GUARDRAILS,
        token=token,
    )
    if fp8:
        fp8_weights, quantizers, meta = validate_fp8_transformer(pipe.transformer)
        print(f"FP8 state restored: weights={fp8_weights}, quantizers={quantizers}, meta={meta}")
    pipe.to("cuda")
    _pipe, _pipe_model = pipe, cache_key
    print(f"loaded pipeline in {time.time() - t0:.1f}s (precision={prec}); cuda allocated {cuda_allocated_gib():.1f} GiB")
    return _pipe


# AV poses are 9D framewise rot6d deltas; the AV convention scales translations by 1.35.
AV_TRANSLATION_SCALE = 1.35


def run_action(case: str, *, model: str = "Cosmos3-Nano") -> Path:
    """Infer the actions connecting an input clip's frames. Writes `<case>_action.json`."""
    spec = ACTION_SETS[case]
    output_dir = case_output_dir(case)
    action_path = output_dir / f"{case}_action.json"
    video = load_video(str(asset_path(spec["video"])))

    pipe = get_pipe(model)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=FIXED_SAMPLING["shift"], use_karras_sigmas=False
    )
    generator = torch.Generator(device="cuda").manual_seed(FIXED_SAMPLING["seed"])

    print(f"case:   {case} ({spec['mode']}, domain {spec['domain_name']}) with {model}")
    print(f"input:  {asset_path(spec['video']).relative_to(COSMOS_ROOT)} ({len(video)} frames)")
    print(f"chunk:  {spec['chunk_size']} transitions at {spec['fps']} fps, resolution tier {spec['resolution_tier']}")
    print(f"output: {action_path}")

    t0 = time.time()
    result = pipe(
        prompt=case_prompt(spec),
        action=CosmosActionCondition(
            mode=spec["mode"],
            chunk_size=spec["chunk_size"],
            domain_name=spec["domain_name"],
            resolution_tier=spec["resolution_tier"],
            video=video,
            view_point=spec["view_point"],
        ),
        fps=spec["fps"],
        num_inference_steps=FIXED_SAMPLING["num_steps"],
        guidance_scale=FIXED_SAMPLING["guidance"],
        use_system_prompt=False,
        generator=generator,
    )
    print(f"inferred in {time.time() - t0:.1f}s")

    if result.action is None:
        raise RuntimeError("the pipeline returned no actions for an inverse-dynamics run")
    actions = result.action[0]
    action_path.write_text(json.dumps(actions.tolist()) + "\n")
    print(f"wrote {action_path} ({tuple(actions.shape)} predicted actions)")
    return action_path


def display_video(path: Path, *, width: int = 720) -> None:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    label = html.escape(str(path))
    markup = f"""
<video controls playsinline preload="metadata" width="{width}" style="max-width: 100%; background: #000;">
  <source src="data:video/mp4;base64,{data}" type="video/mp4">
</video>
<div style="font-family: monospace; font-size: 12px; margin-top: 4px;">{label}</div>
"""
    display(HTML(markup))


def view_action(case: str) -> None:
    """Show the input clip, then plot the predicted actions as a camera trajectory."""
    spec = ACTION_SETS[case]
    output_dir = case_output_dir(case)
    print(f"input video: {asset_path(spec['video']).relative_to(COSMOS_ROOT)}")
    display_video(asset_path(spec["video"]), width=420)

    action_path = output_dir / f"{case}_action.json"
    if not action_path.is_file():
        print(f"No predicted actions at {action_path}; run the case first.")
        return
    poses_rel = np.asarray(json.loads(action_path.read_text()), dtype=np.float64)
    print(f"predicted actions: {poses_rel.shape[0]} steps x {poses_rel.shape[1]}D -> {action_path}")
    print("first step:", [round(value, 4) for value in poses_rel[0]])

    poses_abs = pose_rel_to_abs(
        poses_rel,
        rotation_format="rot6d",
        pose_convention="backward_framewise",
        translation_scale=AV_TRANSLATION_SCALE,
    )
    visualize_pose(
        poses_abs,
        title=f"{case}: predicted camera trajectory ({len(poses_abs)} frames)",
        save_path=output_dir / f"{case}_trajectory.png",
    )


def lerobot_dataset(spec: dict):
    """Open the episode named by a `LEROBOT_ID_SETS` entry with its framework reader."""
    import cosmos_framework.data.generator.action.datasets as action_datasets

    reader = getattr(action_datasets, spec["reader"])
    if spec["reader"] == "FractalLeRobotDataset":
        # Some framework revisions filter Fractal rows with `self._rows = ...` after the
        # base class changed `_rows` into a read-only, lazily materialized property. Redirect
        # that assignment to the cache backing the property. Fixed revisions need no patch.
        rows_property = getattr(reader, "_rows", None)
        if isinstance(rows_property, property) and rows_property.fset is None:
            def set_rows_cache(dataset, rows):
                dataset._rows_cache = rows

            reader._rows = rows_property.setter(set_rows_cache)
    return reader(
        root=asset_path(spec["root"]),
        chunk_length=spec["chunk_length"],
        **spec.get("reader_kwargs", {}),
    )


def sample_frames(sample: dict) -> list:
    video = sample["video"]

    if video.dtype == torch.uint8:
        frames = video.permute(1, 2, 3, 0).cpu().numpy()
    else:
        frames = (video.clamp(0, 1) * 255).round().to(torch.uint8)
        frames = frames.permute(1, 2, 3, 0).cpu().numpy()

    return [Image.fromarray(frame) for frame in frames]


def run_lerobot_action(case: str, *, model: str = "Cosmos3-Nano") -> Path:
    """Predict an action chunk per sampled window of the episode.

    Writes the concatenated predictions to `<case>_action.json` and the windows it conditioned on
    to `<case>_input.mp4`.
    """
    spec = LEROBOT_ID_SETS[case]
    dataset = lerobot_dataset(spec)
    output_dir = case_output_dir(case)
    chunk_length = spec["chunk_length"]
    samples = [dataset[index * chunk_length] for index in range(spec["num_chunks"])]

    pipe = get_pipe(model)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=FIXED_SAMPLING["shift"], use_karras_sigmas=False
    )

    print(f"case:    {case} (inverse_dynamics, domain {dataset.domain_name}) with {model}")
    print(f"episode: {asset_path(spec['root']).relative_to(COSMOS_ROOT)} ({len(dataset)} windows)")
    print(f"chunks:  {spec['num_chunks']} x {chunk_length} transitions, resolution tier {spec['resolution_tier']}")

    clip_frames, predictions = [], []
    for index, sample in enumerate(samples):
        frames = sample_frames(sample)
        fps = int(sample["conditioning_fps"])
        generator = torch.Generator(device="cuda").manual_seed(FIXED_SAMPLING["seed"])
        print(f"chunk {index}: {len(frames)} frames at {fps} fps")
        t0 = time.time()
        result = pipe(
            prompt=sample["ai_caption"],
            action=CosmosActionCondition(
                mode="inverse_dynamics",
                chunk_size=chunk_length,
                domain_name=dataset.domain_name,
                resolution_tier=spec["resolution_tier"],
                video=frames,
                view_point=dataset.viewpoint,
            ),
            fps=fps,
            num_inference_steps=FIXED_SAMPLING["num_steps"],
            guidance_scale=FIXED_SAMPLING["guidance"],
            use_system_prompt=False,
            generator=generator,
        )
        print(f"  inferred in {time.time() - t0:.1f}s")
        if result.action is None:
            raise RuntimeError("the pipeline returned no actions for an inverse-dynamics run")
        predictions.append(result.action[0])
        # The first frame of each window repeats the previous window's last frame.
        clip_frames.extend(frames[1:] if index else frames)

    actions = torch.cat(predictions, dim=0)
    action_path = output_dir / f"{case}_action.json"
    action_path.write_text(json.dumps(actions.tolist()) + "\n")
    clip_path = output_dir / f"{case}_input.mp4"
    export_to_video(
        clip_frames, str(clip_path), fps=int(samples[0]["conditioning_fps"]), macro_block_size=1
    )
    print(f"wrote {action_path} ({tuple(actions.shape)} predicted actions)")
    print(f"wrote {clip_path} ({len(clip_frames)} conditioning frames)")
    return action_path


def view_lerobot_action(case: str) -> None:
    """Show the windows the model saw, then plot the predicted pose block as a trajectory."""
    spec = LEROBOT_ID_SETS[case]
    output_dir = case_output_dir(case)
    action_path = output_dir / f"{case}_action.json"
    if not action_path.is_file():
        print(f"No predicted actions at {action_path}; run the case first.")
        return
    clip_path = output_dir / f"{case}_input.mp4"
    if clip_path.is_file():
        print(f"conditioning windows: {clip_path}")
        display_video(clip_path, width=420)

    dataset = lerobot_dataset(spec)
    actions = torch.as_tensor(json.loads(action_path.read_text()), dtype=torch.float32)
    actions = denormalize_action(actions, method="quantile", stats=dataset.load_action_stats())
    start = spec["pose_index"]
    poses_rel = np.asarray(actions[:, start : start + 9], dtype=np.float64)
    print(f"predicted actions: {tuple(actions.shape)}, plotting dims {start}:{start + 9} ({spec['pose_label']})")
    poses_abs = pose_rel_to_abs(
        poses_rel,
        rotation_format="rot6d",
        pose_convention="backward_framewise",
    )
    visualize_pose(
        poses_abs,
        title=f"{case}: predicted {spec['pose_label']} trajectory ({len(poses_abs)} frames)",
        save_path=output_dir / f"{case}_trajectory.png",
    )
```

## Quantized Checkpoints (FP8)

Default loads are BF16 (`COSMOS3_DIFFUSERS_PRECISION=bf16`). Set the env var to `fp8` before running the helper cell's `get_pipe` (or re-run that cell after changing it). Use-case request cells stay the same.

Nano and Super load `nvidia/Cosmos3-Nano-FP8` and `nvidia/Cosmos3-Super-FP8`. Do not pass `revision=fp8` on the BF16 cards.

Diffusers follows the checkpoint schedule (W8A16 on the first and last few steps, W8A8 in between). See the [Cosmos 3 Diffusers docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cosmos3).

Optional: point `COSMOS3_FP8_MODEL_PATH` at a local Diffusers checkpoint directory.

```python
# Default is bf16. Uncomment to load FP8 checkpoints in the cells below.
# os.environ["COSMOS3_DIFFUSERS_PRECISION"] = "fp8"
print("COSMOS3_DIFFUSERS_PRECISION:", os.environ.get("COSMOS3_DIFFUSERS_PRECISION", "bf16"))
```

## Use Cases

Run each case top-to-bottom: generate, then view the input clip alongside the result and the predicted trajectory.

## Inverse Dynamics: AV Clip 0

Predict the ego trajectory for the first clip.

### Run

```python
av_inverse_0_output = run_action("av_inverse_0")
```

### View Results

```python
view_action("av_inverse_0")
```

## Inverse Dynamics: AV Clip 1

The same settings on the second clip.

### Run

```python
av_inverse_1_output = run_action("av_inverse_1")
```

### View Results

```python
view_action("av_inverse_1")
```

## Inverse Dynamics: Bridge

Predict the end-effector motion of a Bridge tabletop episode from a single window.

### Run

```python
bridge_inverse_output = run_lerobot_action("bridge_inverse")
```

### View Results

```python
view_lerobot_action("bridge_inverse")
```

## Inverse Dynamics: AgiBotWorld-Beta

A humanoid episode whose action vector carries a head camera and both wrists; the right wrist block is plotted. The reader samples the episode down to 10 fps, and five windows cover a longer stretch of the demonstration.

### Run

```python
agibotworld_inverse_output = run_lerobot_action("agibotworld_inverse")
```

### View Results

```python
view_lerobot_action("agibotworld_inverse")
```

## Inverse Dynamics: RoboMIND Franka

A single-arm Franka episode over four windows.

### Run

```python
robomind_franka_inverse_output = run_lerobot_action("robomind_franka_inverse")
```

### View Results

```python
view_lerobot_action("robomind_franka_inverse")
```

## Inverse Dynamics: RoboMIND Franka Dual-Arm

A bimanual FR3 episode whose action vector carries both arms; the right arm block is plotted.

### Run

```python
robomind_franka_dual_inverse_output = run_lerobot_action("robomind_franka_dual_inverse")
```

### View Results

```python
view_lerobot_action("robomind_franka_dual_inverse")
```

## Inverse Dynamics: RoboMIND UR

A Universal Robots episode over five windows. Its reader recovers end-effector poses from joint angles with forward kinematics.

### Run

```python
robomind_ur_inverse_output = run_lerobot_action("robomind_ur_inverse")
```

### View Results

```python
view_lerobot_action("robomind_ur_inverse")
```

## Inverse Dynamics: UMI

A handheld gripper episode recorded with a wrist camera, over five windows.

### Run

```python
umi_inverse_output = run_lerobot_action("umi_inverse")
```

### View Results

```python
view_lerobot_action("umi_inverse")
```

## Inverse Dynamics: Fractal

A Google RT-1 episode over two windows.

### Run

```python
fractal_inverse_output = run_lerobot_action("fractal_inverse")
```

### View Results

```python
view_lerobot_action("fractal_inverse")
```
