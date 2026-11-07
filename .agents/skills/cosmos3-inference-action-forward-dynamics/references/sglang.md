<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->
<!-- rumdl-disable MD001 MD026 -->
<!-- Skill note (not in the source): the directive above keeps the notebook's "### Set up HF Cache:" and "### Start the SGLang server:" headings of the title cell as written (heading level and trailing colon); both rules are re-enabled right after that cell. -->

# Cosmos3 Nano Action: Forward Dynamics with SGLang

This notebook runs Cosmos3 Nano **action forward-dynamics** inference through the SGlang OpenAI-compatible video API:

```text
POST /v1/videos
```

Forward dynamics predicts future visual observations from an initial image and an action trajectory. This notebook contains separate AV and robotics sections that each build their own input spec, run inference, and visualize generated videos.

### Set up HF Cache:

SGLang container is run under root by default. User can create a writable cache directory on host, or start the container with host user id.

```bash
export SGLANG_HF_CACHE="${SGLANG_HF_CACHE:-$HOME/.cache/sglang-huggingface}"
mkdir -p "$SGLANG_HF_CACHE"
chmod 777 "$SGLANG_HF_CACHE"
```

### Start the SGLang server:

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
- Quantize a checkpoint yourself by following the [quantization guide](../../cookbooks/cosmos3/quantization/README.md).

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

> **Skill note (not in the source):** this file is the cookbook notebook `run_fd_with_sglang.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/generator/action/run_fd_with_sglang.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

## Install Notebook Dependencies

Run this cell in the notebook kernel before the data-loading and visualization cells.

```python
%pip install matplotlib pandas pyarrow "lerobot[dataset]"
%pip install av==18.0.0
```

## Configure Notebook Variables

Run this cell after the SGLang server is available. It resolves local input/output paths and stores generated outputs under `outputs/cosmos3_action_sglang/` by default.

Before running it, set `COSMOS3_FRAMEWORK_PATH` to the root of a local checkout of the **cosmos-framework** repository:

```bash
export COSMOS3_FRAMEWORK_PATH=/path/to/cosmos-framework
```

This must be the framework checkout that contains the `cosmos_framework/` Python package. The notebook adds this path to Python's import path to load `cosmos_framework`. Set it in the shell that launches Jupyter before running this cell.

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


def resolve_input(rel_path: str) -> str:
    path = (COSMOS_ROOT / rel_path).resolve()
    assert path.exists(), f"missing input: {path}"
    return str(path)


COSMOS3_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
COSMOS3_INPUT_DIR.mkdir(parents=True, exist_ok=True)

print("COSMOS_ROOT:", COSMOS_ROOT)
print("COSMOS3_FRAMEWORK_PATH:", COSMOS3_FRAMEWORK_PATH)
print("COSMOS3_INPUT_DIR:", COSMOS3_INPUT_DIR)
print("COSMOS3_OUTPUT_ROOT:", COSMOS3_OUTPUT_ROOT)
print("COSMOS3_SGLANG_BASE_URL:", SGLANG_BASE_URL)
```

## AV

In this example, we show how to provide a set of ego poses of a autonomous vehicle and an image to generate driving videos using Cosmos3-Nano.

### Create the AV Forward-Dynamics Input Spec

AV forward-dynamics inference is driven by a JSONL spec, one line per run. Each line shares the same start frame (`vision_path`) but uses a different ego trajectory (`action_path`), so we get one generated video per trajectory.

The action input is prepared in a JSON file, which can be converted from camera poses (camera-to-world transformation, OpenCV convention, unit in meter) via `pose_abs_to_rel`:

```python
if str(COSMOS3_FRAMEWORK_PATH) not in sys.path:
    sys.path.insert(0, str(COSMOS3_FRAMEWORK_PATH))
from cosmos_framework.data.generator.action.utils.pose_utils import pose_rel_to_abs

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
import os

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

Runs `Cosmos3-Nano` on every line of the AV spec through SGLang. Each run writes its video to:

```text
<output>/action_forward_dynamics_av_custom/<name>/vision.mp4
```

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


def check_sglang_server(timeout_s: int = 600, interval_s: int = 10) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{SGLANG_BASE_URL}/v1/models", timeout=10)
            response.raise_for_status()
            print(response.json())
            return
        except requests.RequestException as exc:
            last_error = exc
            print(f"Waiting for SGLang server at {SGLANG_BASE_URL}: {exc}")
            time.sleep(interval_s)
    raise RuntimeError(
        f"SGLang server did not become ready at {SGLANG_BASE_URL} within {timeout_s}s. "
        "Check `docker logs -f cosmos3-sglang-notebook`."
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
        "action_view_point": record["view_point"],
        "action": json.loads(Path(record["action_path"]).read_text()),
        "guardrails": False,
    }
    prompt = str(record.get("prompt") or "").strip() or "A robot manipulates an object."
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
            f"{SGLANG_BASE_URL}/v1/videos",
            data={key: str(value) for key, value in form.items()},
            files={"input_reference": (vision_path.name, image_file, mime_type)},
            timeout=120,
        )
    if not response.ok:
        (run_dir / "error_response.txt").write_text(response.text)
        print("SGLang request failed:", response.status_code)
        print(response.text)
        print("form:", json.dumps(form, indent=2))
        print("extra_params keys:", sorted(extra_params))
        print("action shape:", [len(extra_params["action"]), len(extra_params["action"][0]) if extra_params["action"] else 0])
        response.raise_for_status()
    initial = response.json()
    (run_dir / "response.json").write_text(json.dumps(initial, indent=2))

    while True:
        response = requests.get(f"{SGLANG_BASE_URL}/v1/videos/{initial['id']}", timeout=30)
        response.raise_for_status()
        final = response.json()
        (run_dir / "final.json").write_text(json.dumps(final, indent=2))
        print(initial["id"], final.get("status"), f"{final.get('progress', 0)}%")
        if final.get("status") == "completed":
            break
        if final.get("status") in {"failed", "cancelled"}:
            raise RuntimeError(json.dumps(final, indent=2))
        time.sleep(2)

    response = requests.get(f"{SGLANG_BASE_URL}/v1/videos/{initial['id']}/content", timeout=300)
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


check_sglang_server()
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
if str(COSMOS3_FRAMEWORK_PATH) not in sys.path:
    sys.path.insert(0, str(COSMOS3_FRAMEWORK_PATH))

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

Runs `Cosmos3-Nano` once per robotics chunk through SGLang. Chunk 0 uses the DROID GT first frame. After each chunk finishes, the cell extracts that chunk's last generated frame and uses it as the conditioning image for the next chunk. Guardrails are disabled for this robotics run via `extra_params={"guardrails": false}`.

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


def check_sglang_server(timeout_s: int = 600, interval_s: int = 10) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{SGLANG_BASE_URL}/v1/models", timeout=10)
            response.raise_for_status()
            print(response.json())
            return
        except requests.RequestException as exc:
            last_error = exc
            print(f"Waiting for SGLang server at {SGLANG_BASE_URL}: {exc}")
            time.sleep(interval_s)
    raise RuntimeError(
        f"SGLang server did not become ready at {SGLANG_BASE_URL} within {timeout_s}s. "
        "Check `docker logs -f cosmos3-sglang-notebook`."
    ) from last_error


def submit_forward_dynamics(record: dict, fd_output_dir: Path, *, disable_guardrails: bool = False) -> dict:
    run_dir = fd_output_dir / record["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    vision_path = Path(record["vision_path"])
    input_width, input_height = Image.open(vision_path).size
    target_width = (input_width // 16) * 16
    target_height = (input_height // 16) * 16
    mime_type = mimetypes.guess_type(vision_path.name)[0] or "application/octet-stream"
    extra_params = {
        "action_mode": "forward_dynamics",
        "domain_name": record["domain_name"],
        "action_view_point": record["view_point"],
        "action": json.loads(Path(record["action_path"]).read_text()),
        "guardrails": False,
    }
    if disable_guardrails:
        extra_params["guardrails"] = False

    prompt = str(record.get("prompt") or "").strip() or "A robot manipulates an object."
    form = {
        "prompt": prompt,
        "num_frames": record["action_chunk_size"] + 1,
        "fps": record["fps"],
        "size": f"{target_width}x{target_height}",
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 10.0,
        "seed": record["seed"],
        "extra_params": json.dumps(extra_params),
    }

    with vision_path.open("rb") as image_file:
        response = requests.post(
            f"{SGLANG_BASE_URL}/v1/videos",
            data={key: str(value) for key, value in form.items()},
            files={"input_reference": (vision_path.name, image_file, mime_type)},
            timeout=120,
        )
    if not response.ok:
        (run_dir / "error_response.txt").write_text(response.text)
        print("SGLang request failed:", response.status_code)
        print(response.text)
        print("form:", json.dumps(form, indent=2))
        print("extra_params keys:", sorted(extra_params))
        print("action shape:", [len(extra_params["action"]), len(extra_params["action"][0]) if extra_params["action"] else 0])
        response.raise_for_status()
    initial = response.json()
    (run_dir / "response.json").write_text(json.dumps(initial, indent=2))

    while True:
        response = requests.get(f"{SGLANG_BASE_URL}/v1/videos/{initial['id']}", timeout=30)
        response.raise_for_status()
        final = response.json()
        (run_dir / "final.json").write_text(json.dumps(final, indent=2))
        print(initial["id"], final.get("status"), f"{final.get('progress', 0)}%")
        if final.get("status") == "completed":
            break
        if final.get("status") in {"failed", "cancelled"}:
            raise RuntimeError(json.dumps(final, indent=2))
        time.sleep(2)

    response = requests.get(f"{SGLANG_BASE_URL}/v1/videos/{initial['id']}/content", timeout=300)
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


check_sglang_server()
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

This example runs UMI forward dynamics through SGLang autoregressively over all 16-action chunks in `assets/actions/umi.json`. The action file stores the raw UMI 10D action representation, so the setup cell validates the row dimension, writes one action JSON per chunk, and prepares a run plan.

### Create the UMI Autoregressive Forward-Dynamics Plan

The UMI action file is stored as one JSON array with `16 * n` action rows. SGLang receives one request per 16-action chunk. Chunk 0 uses the checked-in UMI conditioning image; later chunks use conditioning images extracted from the previous generated video.

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

Runs one SGLang video request per UMI action chunk. After each chunk completes, the cell extracts that chunk's last generated frame and uses it as the conditioning image for the next chunk.

```python
import json
import mimetypes
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg

try:
    import requests
except ImportError as exc:
    raise RuntimeError("Install requests in this notebook kernel: pip install requests") from exc

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def check_sglang_server_for_umi(timeout_s: int = 600, interval_s: int = 10) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{SGLANG_BASE_URL}/v1/models", timeout=10)
            response.raise_for_status()
            print(response.json())
            return
        except requests.RequestException as exc:
            last_error = exc
            print(f"Waiting for SGLang server at {SGLANG_BASE_URL}: {exc}")
            time.sleep(interval_s)
    raise RuntimeError(
        f"SGLang server did not become ready at {SGLANG_BASE_URL} within {timeout_s}s. "
        "Check `docker logs -f cosmos3-sglang-notebook`."
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
        "action_view_point": record["view_point"],
        "action": json.loads(Path(record["action_path"]).read_text()),
        "guardrails": False,
    }
    prompt = str(record.get("prompt") or "").strip() or "A robot manipulates an object."
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
            f"{SGLANG_BASE_URL}/v1/videos",
            data={key: str(value) for key, value in form.items()},
            files={"input_reference": (vision_path.name, image_file, mime_type)},
            timeout=120,
        )
    if not response.ok:
        (run_dir / "error_response.txt").write_text(response.text)
        print("SGLang request failed:", response.status_code)
        print(response.text)
        print("form:", json.dumps(form, indent=2))
        print("extra_params keys:", sorted(extra_params))
        print("action shape:", [len(extra_params["action"]), len(extra_params["action"][0]) if extra_params["action"] else 0])
        response.raise_for_status()

    initial = response.json()
    (run_dir / "response.json").write_text(json.dumps(initial, indent=2))

    while True:
        response = requests.get(f"{SGLANG_BASE_URL}/v1/videos/{initial['id']}", timeout=30)
        response.raise_for_status()
        final = response.json()
        (run_dir / "final.json").write_text(json.dumps(final, indent=2))
        print(initial["id"], final.get("status"), f"{final.get('progress', 0)}%")
        if final.get("status") == "completed":
            break
        if final.get("status") in {"failed", "cancelled"}:
            raise RuntimeError(json.dumps(final, indent=2))
        time.sleep(2)

    response = requests.get(f"{SGLANG_BASE_URL}/v1/videos/{initial['id']}/content", timeout=300)
    response.raise_for_status()
    video_path = run_dir / "vision.mp4"
    video_path.write_bytes(response.content)
    print("saved", video_path)
    return {"record": record, "initial": initial, "final": final, "run_dir": run_dir, "video_path": video_path}


check_sglang_server_for_umi()
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
