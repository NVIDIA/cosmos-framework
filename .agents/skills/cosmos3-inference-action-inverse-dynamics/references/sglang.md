<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Nano Action: Inverse Dynamics with SGLang

This notebook runs Cosmos3 Nano **action inverse-dynamics** inference through the SGLang OpenAI-compatible action generation API:

```text
POST /v1/actions/generations
```

Inverse dynamics is the reverse of forward dynamics: given a video, it predicts the ego-motion (action) trajectory that produced it. This notebook builds the same custom input spec as [`run_id_with_cosmos_framework.ipynb`](./cosmos-framework.md), keeps the same input-video preview and predicted-trajectory visualization, and only changes the environment setup plus the inference call.

> **Skill note (not in the source):** this file is the cookbook notebook `run_id_with_sglang.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/generator/action/run_id_with_sglang.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

<!-- rumdl-disable MD001 MD026 -->
<!-- Skill note (not in the source): the directive above keeps the notebook's "### Set up HF Cache:" heading of this cell as written (heading level and trailing colon); both rules are re-enabled right after the cell. -->

### Set up HF Cache:

SGLang container is run under root by default. User can create a writable cache directory on host, or start the container with host user id.

```bash
export SGLANG_HF_CACHE="${SGLANG_HF_CACHE:-$HOME/.cache/sglang-huggingface}"
mkdir -p "$SGLANG_HF_CACHE"
chmod 777 "$SGLANG_HF_CACHE"
```

### Start SGLang Server

Start the server in a terminal from the cookbook root (`.agents/skills`).

```bash
docker rm -f cosmos3-sglang-notebook 2>/dev/null || true

docker run -d --init --name cosmos3-sglang-notebook \
  --runtime nvidia --gpus '"device=0"' \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v $SGLANG_HF_CACHE:/root/.cache/huggingface \
  -v "$PWD:/workspace" \
  -p 30000:30000 --ipc=host \
  lmsysorg/sglang:dev \
  sglang serve \
  --model-path nvidia/Cosmos3-Nano \
  --host 0.0.0.0

# Wait until this returns model metadata before running the inference cell.
curl http://localhost:30000/v1/models
```

### FP8

To serve Cosmos3-Nano in FP8, you can either:

- Download the prebuilt FP8 checkpoint from Hugging Face: `nvidia/Cosmos3-Nano-FP8`.
- Quantize a checkpoint yourself by following the [quantization guide](../../cosmos3-quantization/SKILL.md).

To serve the prebuilt checkpoint, pass its Hugging Face ID to `--model-path`:

```bash
sglang serve \
  --model-path nvidia/Cosmos3-Nano-FP8 \
  --host 0.0.0.0
```

To serve a checkpoint you quantized yourself, pass its local directory to `--model-path`. If the server runs in Docker, mount that directory and use its path inside the container.

```bash
sglang serve \
  --model-path /path/to/fp8/checkpoint \
  --host 0.0.0.0
```

<!-- rumdl-enable MD001 MD026 -->

## Configure Notebook Variables

Before running the variables cell, set `COSMOS3_REPO` to the root of a local checkout of the **cosmos-framework** repository:

```bash
export COSMOS3_REPO=/path/to/cosmos-framework
```

This must be the framework checkout that contains the `cosmos_framework/` Python package—not this `cosmos` cookbook repository, a model directory, or the `packages/cosmos3` model sources. The notebook adds this path to Python's import path to load `cosmos_framework`. Set it in the shell that launches Jupyter before running the variables cell.

```python
from pathlib import Path
import os


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "cookbooks" / "cosmos3").is_dir():
            return path
    return start


COSMOS_ROOT = find_repo_root(Path.cwd().resolve())
COSMOS3_FRAMEWORK_PATH = Path(
    os.environ.get("COSMOS3_FRAMEWORK_PATH", COSMOS_ROOT.parent.parent)
).resolve()
COSMOS3_OUTPUT_ROOT = Path(
    os.environ.get("COSMOS3_SGLANG_OUTPUT_ROOT", COSMOS_ROOT / "generator" / "action" / "outputs" / "cosmos3_action_sglang")
).resolve()
COSMOS3_INPUT_DIR = COSMOS3_OUTPUT_ROOT / "inputs"
SGLANG_BASE_URL = os.environ.get("COSMOS3_SGLANG_BASE_URL", "http://localhost:30000").rstrip("/")

COSMOS3_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
COSMOS3_INPUT_DIR.mkdir(parents=True, exist_ok=True)

print("COSMOS_ROOT:", COSMOS_ROOT)
print("COSMOS3_FRAMEWORK_PATH:", COSMOS3_FRAMEWORK_PATH)
print("COSMOS3_INPUT_DIR:", COSMOS3_INPUT_DIR)
print("COSMOS3_OUTPUT_ROOT:", COSMOS3_OUTPUT_ROOT)
print("COSMOS3_SGLANG_BASE_URL:", SGLANG_BASE_URL)
```

## Create the Inverse-Dynamics Input Spec

Inverse-dynamics inference is driven by a JSONL spec, one line per run. Unlike forward dynamics, each line provides only an input video (`vision_path`) and **no** `action_path` — the action is what the model predicts.

This cell builds that spec from local AV videos, writing it under:

```text
outputs/cosmos3_action_sglang/inputs/action_inverse_dynamics_av_custom.jsonl
```

It mirrors the native PyTorch notebook's spec format. The `vision_path` is written as an absolute path, because the SGLang request cell reads the spec records directly.

```python
# `os` and the COSMOS3_* paths come from the configuration cell.
import json

# Local inputs, relative to the cookbook root.
input_videos = {
    "av_inverse_0": "cookbooks/cosmos3/generator/action/assets/videos/av_0.mp4",
    "av_inverse_1": "cookbooks/cosmos3/generator/action/assets/videos/av_1.mp4",
}

def resolve_input(rel_path: str) -> str:
    path = (COSMOS_ROOT / rel_path).resolve()
    assert path.exists(), f"missing input: {path}"
    return str(path)

records = [
    {
        "action_chunk_size": 60,
        "domain_name": "av",
        "fps": 10,
        "image_size": 480,
        "view_point": "ego_view",
        "model_mode": "inverse_dynamics",
        "name": name,
        "prompt": "You are an autonomous vehicle planning system.",
        "seed": 0,
        "vision_path": resolve_input(video_rel),
    }
    for name, video_rel in input_videos.items()
]

COSMOS3_INPUT_DIR.mkdir(parents=True, exist_ok=True)
id_input_path = COSMOS3_INPUT_DIR / "action_inverse_dynamics_av_custom.jsonl"
id_input_path.write_text("".join(json.dumps(r) + "\n" for r in records))
id_output_dir = COSMOS3_OUTPUT_ROOT / "action_inverse_dynamics_av_custom"

# The bash inference cell can only see the environment, so export the paths it needs.
os.environ["COSMOS3_ID_INPUT"] = str(id_input_path)
os.environ["COSMOS3_ID_OUTPUT"] = str(id_output_dir)

print("wrote spec:", id_input_path)
print("runs:", list(input_videos))
print(id_input_path.read_text())
```

## Preview the Input Video(s)

Preview each input video before running inference. `Video(..., embed=True)` base64-inlines the file, and these AV clips are several MB each, so we first transcode a small preview (~150 KB) with the ffmpeg binary bundled in `imageio-ffmpeg` (installed by `uv sync`), then embed it. The original input videos are left untouched.

```python
import subprocess
import imageio_ffmpeg
from IPython.display import Video, display

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

def make_preview(src: Path, dst: Path, crf: int = 28) -> Path:
    """Re-encode `src` to a compact, browser-friendly mp4 (cached)."""
    if not dst.exists():
        subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
             "-c:v", "libx264", "-crf", str(crf),
             "-preset", "veryfast", "-an", "-pix_fmt", "yuv420p", str(dst)],
            check=True,
        )
    return dst

# `records` comes from the prepare cell; preview each input video.
for record in records:
    name = record["name"]
    src = Path(record["vision_path"])
    preview = make_preview(src, COSMOS3_INPUT_DIR / f"{name}_input_preview.mp4")
    print(f"{name}  ({src.stat().st_size // 1024} KB -> {preview.stat().st_size // 1024} KB preview)")
    display(Video(str(preview), embed=True))
```

## Run Inverse-Dynamics Inference

Runs `Cosmos3-Nano` on every line of the spec through SGLang. Inverse dynamics predicts an action, and this cell writes a PyTorch-compatible result file for each run:

```text
<output>/<name>/sample_outputs.json
```

The predicted action trajectory is stored under `outputs[0].content["action"]`, matching the native notebook's visualization cell. SGLang also returns `response.json`, `final.json`, and `action.json` for API debugging.

```python
import json
import time
from pathlib import Path

try:
    import requests
except ImportError as exc:
    raise RuntimeError("Install requests in this notebook kernel: pip install requests") from exc


def check_sglang_server() -> None:
    response = requests.get(f"{SGLANG_BASE_URL}/v1/models", timeout=10)
    response.raise_for_status()
    print(response.json())


def log_sglang_error(response: requests.Response, run_dir: Path, label: str, context: dict | None = None) -> None:
    error_path = run_dir / f"{label}_error_response.txt"
    error_path.write_text(response.text)
    print(f"SGLang {label} request failed:", response.status_code)
    print(response.text)
    print("saved error response:", error_path)
    if context is not None:
        print("context:", json.dumps(context, indent=2))


def submit_inverse_dynamics(record: dict) -> dict:
    run_dir = id_output_dir / record["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(record["vision_path"])
    extra_params = {
        "action_mode": "inverse_dynamics",
        "domain_name": record["domain_name"],
        "action_view_point": record["view_point"],
        "raw_action_dim": 9,
        "guardrails": False,
    }
    form = {
        "prompt": record["prompt"],
        "num_frames": record["action_chunk_size"] + 1,
        "fps": record["fps"],
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 10.0,
        "seed": record["seed"],
        "extra_params": json.dumps(extra_params),
    }

    with video_path.open("rb") as video_file:
        response = requests.post(
            f"{SGLANG_BASE_URL}/v1/actions/generations",
            data={key: str(value) for key, value in form.items()},
            files={"input_reference": (video_path.name, video_file, "video/mp4")},
            timeout=120,
        )
    if not response.ok:
        log_sglang_error(
            response,
            run_dir,
            "submit",
            {"form": form, "extra_params": extra_params, "video_path": str(video_path)},
        )
        response.raise_for_status()
    response.raise_for_status()
    action = response.json()['data'][0]['action']
    (run_dir / "action.json").write_text(json.dumps(action, indent=2))

    sample_outputs = {"outputs": [{"content": {"action": action.get("values")}}]}
    (run_dir / "sample_outputs.json").write_text(json.dumps(sample_outputs, indent=2))

    print("saved", run_dir / "sample_outputs.json")
    print("action shape:", action.get("shape"), "dtype:", action.get("dtype"))
    return {"record": record, "run_dir": run_dir, "action": action}


check_sglang_server()
results = []
for record in records:
    print(f"\nSubmitting {record['name']}")
    results.append(submit_inverse_dynamics(record))
```

## Visualize the Predicted Action

Plot the action the model predicted from each input video, as a 3D camera path (with frustums) and a top-down bird's-eye view. The action is read from each run's `sample_outputs.json` and interpreted with the AV pose convention.

```python
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# The notebook kernel may differ from the framework venv, so put the repo on the
# path before importing `cosmos_framework`.
if str(COSMOS3_FRAMEWORK_PATH) not in sys.path:
    sys.path.insert(0, str(COSMOS3_FRAMEWORK_PATH))
from cosmos_framework.data.generator.action.utils.pose_utils import pose_rel_to_abs

# frustum: apex + image-rectangle corners (camera +Z forward), and their edges
_FRUSTUM = np.array([[0, 0, 0], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
_EDGES = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]

def visualize_pose(poses_abs, *, n_frustums=20, scale_frac=0.03, aspect=16 / 9,
                   fov_deg=60.0, vertical_exaggeration=1.0, cmap="turbo",
                   title=None, save_path=None, show=True):
    """3D camera trajectory (with frustums) + a top-down bird's-eye view.

    AV convention: world Y is up, world +Z is the heading. `vertical_exaggeration`
    stretches only the up-axis box (uniform world scaling, so frustums never skew);
    1.0 = geometrically faithful. The 3D plot reorders world (X, Y, Z) -> (X, Z, Y)
    so Y points up on screen.
    """
    poses_abs = np.asarray(poses_abs)
    pos = poses_abs[:, :3, 3]                     # camera centers [T, 3]
    fwd = poses_abs[:, :3, 2]                     # heading (+Z) [T, 3]
    T = len(pos)
    colors = plt.get_cmap(cmap)(np.arange(T) / max(T - 1, 1))
    scale = max(np.ptp(pos, axis=0).max() * scale_frac, 1e-3)
    step = max(1, T // max(n_frustums, 1))
    xzy = [0, 2, 1]                               # world (X,Y,Z) -> plot (X, Z, Y-up)

    fig = plt.figure(figsize=(14, 6))

    # (1) 3D perspective with frustums
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    path = pos[:, xzy]
    ax.plot(*path.T, color="0.6", lw=1.0, alpha=0.7)
    lines, lcolors, allpts = [], [], [path]
    for i in range(0, T, step):
        cw = ((_FRUSTUM * [aspect, 1, 1] * scale * np.tan(np.radians(fov_deg) / 2))
              @ poses_abs[i, :3, :3].T + poses_abs[i, :3, 3])[:, xzy]  # frustum in plot coords
        allpts.append(cw)
        lines += [[cw[a], cw[b]] for a, b in _EDGES]
        lcolors += [colors[i]] * len(_EDGES)
    ax.add_collection3d(Line3DCollection(lines, colors=lcolors, linewidths=1.2))
    ax.scatter(*path[0], color="lime", s=80, edgecolor="k", label="first frame", zorder=5)
    ax.scatter(*path[-1], color="red", s=80, edgecolor="k", label="last frame", zorder=5)
    rng = np.clip(np.ptp(np.concatenate(allpts), axis=0), 1e-9, None)
    ax.set_box_aspect((rng[0], rng[1], rng[2] * vertical_exaggeration))
    ax.set_xlabel("X (m)", labelpad=12); ax.set_ylabel("Z forward (m)", labelpad=12)
    ax.set_zlabel("Y up (m)", labelpad=10); ax.set_zticks([])
    ax.set_title(title or f"Camera trajectory + frustums ({T} frames)")
    ax.legend(loc="upper left"); ax.view_init(elev=22, azim=-70)

    # (2) top-down bird's-eye view (X-Z ground plane)
    ax2 = fig.add_subplot(1, 2, 2)
    seg = np.stack([pos[:-1, [0, 2]], pos[1:, [0, 2]]], axis=1)
    lc = LineCollection(seg, cmap=cmap, norm=plt.Normalize(0, T - 1), linewidth=2.5)
    lc.set_array(np.arange(T - 1)); ax2.add_collection(lc)
    ax2.quiver(pos[::step, 0], pos[::step, 2], fwd[::step, 0], fwd[::step, 2],
               color=colors[::step], angles="xy", width=0.005, scale=22, zorder=3)
    ax2.scatter(*pos[0, [0, 2]], color="lime", s=80, edgecolor="k", label="first frame", zorder=5)
    ax2.scatter(*pos[-1, [0, 2]], color="red", s=80, edgecolor="k", label="last frame", zorder=5)
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Z forward (m)")
    ax2.set_title("Top-down (bird's-eye view)")
    ax2.set_aspect("equal", adjustable="datalim"); ax2.autoscale_view(); ax2.legend()
    fig.colorbar(lc, ax=ax2, label="frame index")

    plt.tight_layout(w_pad=6)
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight"); print("saved", save_path)
    if show:
        plt.show()

# `records` and `id_output_dir` come from the prepare cell; read each run's
# predicted action from its sample_outputs.json.
for record in records:
    name = record["name"]
    outputs = json.loads((id_output_dir / name / "sample_outputs.json").read_text())
    poses_rel = np.array(outputs["outputs"][0]["content"]["action"])  # [T-1, 9] = [translation(3), rot6d(6)]

    # AV action convention (see cosmos_framework/data/generator/action/utils/pose_utils.py):
    # rot6d rotation, backward_framewise, translation_scale = 1.35.
    poses_abs = pose_rel_to_abs(
        poses_rel,
        rotation_format="rot6d",
        pose_convention="backward_framewise",
        translation_scale=1.35,
    )  # [T, 4, 4] camera-to-world
    print(name, poses_rel.shape, poses_abs.shape)
    visualize_pose(poses_abs, title=f"{name}: predicted camera trajectory + frustums ({len(poses_abs)} frames)", show=True)
```
