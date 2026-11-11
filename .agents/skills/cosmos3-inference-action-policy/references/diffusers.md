<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Action-Conditioned Policy with Diffusers

Policy jointly predicts future video and an action chunk from a single conditioning frame plus a task instruction, running `Cosmos3OmniPipeline` with the inputs grouped into a `CosmosActionCondition`.

It runs the DROID sample under [`assets/`](../../cookbooks/cosmos3/generator/action/assets) on the `Cosmos3-Nano-Policy-DROID` checkpoint. For forward and inverse dynamics, see [`run_fd_with_diffusers.ipynb`](../../cosmos3-inference-action-forward-dynamics/references/diffusers.md) and [`run_id_with_diffusers.ipynb`](../../cosmos3-inference-action-inverse-dynamics/references/diffusers.md).

> **Skill note (not in the source):** this file is the cookbook notebook `run_policy_with_diffusers.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/generator/action/run_policy_with_diffusers.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

<!-- rumdl-disable MD034 -->
<!-- Skill note (not in the source): the directive above keeps the notebook's bare URL in this section's "uv version" note as written; the rule is re-enabled right after the section. -->

## 1. Prerequisites

Use a Linux machine with NVIDIA GPU access, model access on Hugging Face, and either `uvx hf@latest auth login` or `HF_TOKEN` set.

Policy runs on the post-trained [nvidia/Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID) checkpoint, conditioned on the DROID sample under `assets/droid_lerobot_example`.

Predicted actions are plotted as camera trajectories using the pose helpers from the Cosmos framework. Point `COSMOS3_REPO` at your framework checkout (it defaults to `packages/cosmos3` beside this repo).

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
  transformers

# The LeRobot video decoder needs a torchcodec build matching the Torch backend, which is
# published only on the PyTorch index. Keep that index scoped to this requirement so the rest of
# the tree still resolves against PyPI.
uv pip install --torch-backend="$COSMOS3_TORCH_BACKEND" \
  --extra-index-url "https://download.pytorch.org/whl/$COSMOS3_TORCH_BACKEND" \
  torchcodec

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

## 6. Preview the Conditioning Clips

```python
camera_root = COSMOS3_ACTION_ROOT / "assets" / "droid_lerobot_example" / "videos"

for camera_dir in sorted(camera_root.iterdir()):
    for video_path in sorted(camera_dir.rglob("*.mp4")):
        print(f"{video_path.relative_to(COSMOS_ROOT)} ({video_path.stat().st_size // 1024} KB)")
```

## 7. Define the Policy Case, Runner, and Viewer Helpers

Policy rolls out future video and the action chunk together from one conditioning frame, a task instruction, and the domain metadata, so it passes no `raw_actions`. `height`, `width`, and `num_frames` stay unset: the pipeline derives the frame count from `chunk_size + 1` and the conditioning canvas from `resolution_tier`.

DROID conditions on a single frame that concatenates three camera views, which is what `view_point="concat_view"` describes to the model. `build_concat_frame` composes it from the first frame of each clip.

The task instruction is the prompt; override it with `COSMOS3_POLICY_PROMPT`. Predicted actions come back as `result.action` and are written next to the video as JSON.

```python
import base64
import gc
import html
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML, Image, display
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image as PILImage, ImageOps

if "COSMOS3_DIFFUSERS_ACTION_VENV" not in os.environ:
    raise RuntimeError("Run the Restore Environment cell after switching to the Diffusers kernel.")

expected_python = (Path(os.environ["COSMOS3_DIFFUSERS_ACTION_VENV"]) / "bin" / "python").resolve()
if Path(sys.executable).resolve() != expected_python:
    raise RuntimeError(
        "Switch the notebook kernel to 'Cosmos3 Diffusers Action (Python 3.13)' "
        "before running Diffusers cells."
    )

from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
from diffusers import logging as diffusers_logging
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

if str(COSMOS3_REPO) not in sys.path:
    sys.path.insert(0, str(COSMOS3_REPO))

from cosmos_framework.data.generator.action.action_normalization import (
    denormalize_action,
    load_action_stats,
)
from cosmos_framework.data.generator.action.utils.pose_utils import pose_rel_to_abs


_FRUSTUM = np.array(
    [[0, 0, 0], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    dtype=float,
)
_EDGES = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]


def visualize_pose(
    poses_abs,
    *,
    n_frustums=20,
    scale_frac=0.03,
    aspect=16 / 9,
    fov_deg=60.0,
    vertical_exaggeration=1.0,
    cmap="turbo",
    title=None,
    save_path=None,
    show=True,
):
    """Show absolute poses as a 3D frustum path and top-down trajectory."""
    poses_abs = np.asarray(poses_abs)
    positions = poses_abs[:, :3, 3]
    forward = poses_abs[:, :3, 2]
    num_frames = len(positions)
    colors = plt.get_cmap(cmap)(np.arange(num_frames) / max(num_frames - 1, 1))
    scale = max(np.ptp(positions, axis=0).max() * scale_frac, 1e-3)
    step = max(1, num_frames // max(n_frustums, 1))
    xzy = [0, 2, 1]

    figure = plt.figure(figsize=(14, 6))

    axis_3d = figure.add_subplot(1, 2, 1, projection="3d")
    path = positions[:, xzy]
    axis_3d.plot(*path.T, color="0.6", lw=1.0, alpha=0.7)

    lines, line_colors, all_points = [], [], [path]
    for index in range(0, num_frames, step):
        frustum = (
            (_FRUSTUM * [aspect, 1, 1] * scale * np.tan(np.radians(fov_deg) / 2))
            @ poses_abs[index, :3, :3].T
            + poses_abs[index, :3, 3]
        )[:, xzy]
        all_points.append(frustum)
        lines.extend([[frustum[start], frustum[end]] for start, end in _EDGES])
        line_colors.extend([colors[index]] * len(_EDGES))

    axis_3d.add_collection3d(Line3DCollection(lines, colors=line_colors, linewidths=1.2))
    axis_3d.scatter(*path[0], color="lime", s=80, edgecolor="k", label="first frame", zorder=5)
    axis_3d.scatter(*path[-1], color="red", s=80, edgecolor="k", label="last frame", zorder=5)

    value_range = np.clip(np.ptp(np.concatenate(all_points), axis=0), 1e-9, None)
    axis_3d.set_box_aspect((value_range[0], value_range[1], value_range[2] * vertical_exaggeration))
    axis_3d.set_xlabel("X (m)", labelpad=12)
    axis_3d.set_ylabel("Z forward (m)", labelpad=12)
    axis_3d.set_zlabel("Y up (m)", labelpad=10)
    axis_3d.set_zticks([])
    axis_3d.set_title(title or f"End-effector trajectory ({num_frames} frames)")
    axis_3d.legend(loc="upper left")
    axis_3d.view_init(elev=22, azim=-70)

    axis_top_down = figure.add_subplot(1, 2, 2)
    segments = np.stack([positions[:-1, [0, 2]], positions[1:, [0, 2]]], axis=1)
    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=plt.Normalize(0, num_frames - 1),
        linewidth=2.5,
    )
    collection.set_array(np.arange(num_frames - 1))
    axis_top_down.add_collection(collection)
    axis_top_down.quiver(
        positions[::step, 0],
        positions[::step, 2],
        forward[::step, 0],
        forward[::step, 2],
        color=colors[::step],
        angles="xy",
        width=0.005,
        scale=22,
        zorder=3,
    )
    axis_top_down.scatter(
        *positions[0, [0, 2]],
        color="lime",
        s=80,
        edgecolor="k",
        label="first frame",
        zorder=5,
    )
    axis_top_down.scatter(
        *positions[-1, [0, 2]],
        color="red",
        s=80,
        edgecolor="k",
        label="last frame",
        zorder=5,
    )
    axis_top_down.set_xlabel("X (m)")
    axis_top_down.set_ylabel("Z forward (m)")
    axis_top_down.set_title("Top-down (bird's-eye view)")
    axis_top_down.set_aspect("equal", adjustable="datalim")
    axis_top_down.autoscale_view()
    axis_top_down.legend()
    figure.colorbar(collection, ax=axis_top_down, label="frame index")

    plt.tight_layout(w_pad=6)
    if save_path:
        figure.savefig(save_path, dpi=120, bbox_inches="tight")
        print("saved", save_path)
    if show:
        plt.show()
    plt.close(figure)


MODEL_IDS = {
    "Cosmos3-Nano-Policy-DROID": "nvidia/Cosmos3-Nano-Policy-DROID",
}

# The local `droid_lerobot_example` asset is not a full versioned DROID dataset,
# so load the checkpoint-compatible normalization stats directly.
DROID_ACTION_STATS_PATH = (
    COSMOS3_REPO
    / "cosmos_framework"
    / "data"
    / "generator"
    / "action"
    / "normalizer_stats"
    / "droid_lerobot_stats.json"
)
DROID_ACTION_STATS = {
    name: torch.from_numpy(values)
    for name, values in load_action_stats(str(DROID_ACTION_STATS_PATH)).items()
}

GUARDRAILS = os.environ.get("COSMOS3_DIFFUSERS_GUARDRAILS", "true").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

FIXED_SAMPLING = {
    "num_steps": 30,
    "guidance": 1.0,
    "shift": 5.0,
    "seed": 0,
}

CAMERA_VIDEOS = {
    "wrist": "assets/droid_lerobot_example/videos/observation.image.wrist_image_left/chunk-000/file-000.mp4",
    "exterior_1": "assets/droid_lerobot_example/videos/observation.image.exterior_image_1_left/chunk-000/file-000.mp4",
    "exterior_2": "assets/droid_lerobot_example/videos/observation.image.exterior_image_2_left/chunk-000/file-000.mp4",
}
CONCAT_SIZE = (640, 540)

ACTION_SETS = {
    "droid_policy": {
        "mode": "policy",
        "domain_name": "droid_lerobot",
        "chunk_size": 16,
        "resolution_tier": 480,
        "view_point": "concat_view",
        "fps": 15,
        "prompt": os.environ.get(
            "COSMOS3_POLICY_PROMPT",
            "Pick up the object and place it in the target container.",
        ),
    },
}

_pipe = None
_pipe_model = None


def asset_path(relative_path: str) -> Path:
    path = COSMOS3_ACTION_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    return path.resolve()


def case_output_dir(case: str) -> Path:
    output_dir = Path(os.environ["COSMOS3_ACTION_OUTPUT_ROOT"]) / case
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_concat_frame(case: str) -> Path:
    """Compose the first frame of each camera clip into the policy conditioning canvas."""
    width, height = CONCAT_SIZE
    top_height = height // 2
    bottom_height = height - top_height
    half_width = width // 2

    frames = {
        name: load_video(str(asset_path(relative_path)))[0]
        for name, relative_path in CAMERA_VIDEOS.items()
    }

    canvas = PILImage.new("RGB", CONCAT_SIZE)
    tiles = [
        ("wrist", (width, top_height), (0, 0)),
        ("exterior_1", (half_width, bottom_height), (0, top_height)),
        ("exterior_2", (half_width, bottom_height), (half_width, top_height)),
    ]

    for name, size, position in tiles:
        canvas.paste(
            ImageOps.fit(frames[name], size, method=PILImage.Resampling.BICUBIC),
            position,
        )

    frame_path = case_output_dir(case) / f"{case}_input.png"
    canvas.save(frame_path)
    return frame_path


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


def get_pipe(model: str) -> Cosmos3OmniPipeline:
    global _pipe, _pipe_model

    model_id = MODEL_IDS.get(model, model)
    if _pipe is not None and _pipe_model == model_id:
        return _pipe

    release_pipe()
    diffusers_logging.set_verbosity_info()
    print(f"loading {model_id}...")
    start_time = time.time()

    pipe = Cosmos3OmniPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        safety_checker=None,
        enable_safety_checker=GUARDRAILS,
        token=os.environ.get("HF_TOKEN") or None,
    )
    pipe.to("cuda")

    _pipe, _pipe_model = pipe, model_id
    print(
        f"loaded pipeline in {time.time() - start_time:.1f}s; "
        f"cuda allocated {cuda_allocated_gib():.1f} GiB"
    )
    return _pipe


def run_policy(case: str, *, model: str = "Cosmos3-Nano-Policy-DROID") -> Path:
    """Run policy and write the rollout video plus normalized predicted actions."""
    spec = ACTION_SETS[case]
    output_dir = case_output_dir(case)
    output_path = output_dir / f"{case}.mp4"

    frame_path = build_concat_frame(case)
    image = PILImage.open(frame_path).convert("RGB")

    pipe = get_pipe(model)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=FIXED_SAMPLING["shift"],
        use_karras_sigmas=False,
    )
    generator = torch.Generator(device="cuda").manual_seed(FIXED_SAMPLING["seed"])

    print(f"case:   {case} ({spec['mode']}, domain {spec['domain_name']}) with {model}")
    print(f"input:  {frame_path} {image.size}")
    print(f"prompt: {spec['prompt']}")
    print(
        f"frames: {spec['chunk_size'] + 1} at {spec['fps']} fps, "
        f"resolution tier {spec['resolution_tier']}"
    )
    print(f"output: {output_path}")

    start_time = time.time()
    result = pipe(
        prompt=spec["prompt"],
        action=CosmosActionCondition(
            mode=spec["mode"],
            chunk_size=spec["chunk_size"],
            domain_name=spec["domain_name"],
            resolution_tier=spec["resolution_tier"],
            image=image,
            view_point=spec["view_point"],
        ),
        fps=spec["fps"],
        num_inference_steps=FIXED_SAMPLING["num_steps"],
        guidance_scale=FIXED_SAMPLING["guidance"],
        use_system_prompt=False,
        generator=generator,
    )
    print(f"generated in {time.time() - start_time:.1f}s")

    export_to_video(result.video, str(output_path), fps=spec["fps"], macro_block_size=1)
    print(f"wrote {output_path}")

    if result.action is not None:
        action_path = output_dir / f"{case}_action.json"
        action_path.write_text(json.dumps(result.action[0].tolist()) + "\n")
        print(f"wrote {action_path} (predicted actions, model-normalized space)")

    return output_path


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


def view_policy(case: str) -> None:
    """Show the rollout and plot denormalized DROID end-effector and gripper predictions."""
    output_dir = case_output_dir(case)

    frame_path = output_dir / f"{case}_input.png"
    if frame_path.is_file():
        print(f"conditioning frame: {frame_path}")
        display(Image(filename=str(frame_path), width=420))

    output_path = output_dir / f"{case}.mp4"
    if not output_path.is_file():
        print(f"No generated video at {output_path}; run the case first.")
        return

    print(f"generated: {output_path} ({output_path.stat().st_size // 1024} KB)")
    display_video(output_path)

    action_path = output_dir / f"{case}_action.json"
    if not action_path.is_file():
        return

    normalized_actions = torch.as_tensor(
        json.loads(action_path.read_text()),
        dtype=torch.float32,
    )
    actions = denormalize_action(
        normalized_actions,
        method="quantile",
        stats=DROID_ACTION_STATS,
    )

    print(f"predicted actions: {actions.shape[0]} steps x {actions.shape[1]}D -> {action_path}")
    print("first denormalized step:", [round(value, 4) for value in actions[0].tolist()])

    poses_abs = pose_rel_to_abs(
        actions[:, :9].cpu().numpy(),
        rotation_format="rot6d",
        pose_convention="backward_framewise",
    )
    visualize_pose(
        poses_abs,
        title=f"{case}: predicted end-effector trajectory ({len(poses_abs)} frames)",
        save_path=output_dir / f"{case}_trajectory.png",
    )

    figure, axis = plt.subplots(figsize=(7, 2.4))
    axis.plot(actions[:, 9].cpu().numpy(), marker="o", color="#0b7285")
    axis.set_xlabel("action step")
    axis.set_ylabel("gripper")
    axis.set_title("Predicted gripper channel")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / f"{case}_gripper.png", dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(figure)
```

## Policy: DROID Pick and Place

Roll out the future observations and the matching action chunk for the DROID sample.

### Run

```python
droid_policy_output = run_policy("droid_policy")
```

### View Results

The predicted chunk carries a pose delta per step, so it integrates into an end-effector trajectory.

```python
view_policy("droid_policy")
```
