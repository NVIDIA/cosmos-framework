<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Action: Inverse Dynamics with vLLM-Omni

## Prerequisites

Generator requires the Guardrail. Request access to the gated [nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail) HF repository before running these examples. When running with guardrails enabled, also set and pass `HF_TOKEN` to the container; it must be a Hugging Face token authorized for that repository. To disable guardrails server-wide, pass `--no-guardrails` to `vllm serve`; alternatively, set `guardrails: false` in vLLM-Omni `extra_params` or `extra_args` to disable them per request.

## Overview

This notebook runs Cosmos3 **action inverse-dynamics** inference through the vLLM-Omni OpenAI-compatible video API:

```text
POST /v1/videos
```

Inverse dynamics is the reverse of forward dynamics: given a video, it predicts the ego-motion (action) trajectory that produced it. This notebook builds the same custom input spec as [`run_id_with_cosmos_framework.ipynb`](./cosmos-framework.md), keeps the same input-video preview and predicted-trajectory visualization, and only changes the environment setup plus the inference call.

> **Skill note (not in the source):** this file is the cookbook notebook `run_id_with_vllm_omni.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/generator/action/run_id_with_vllm_omni.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

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
[quantization walkthrough](../../cookbooks/cosmos3/quantization/README.md).
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

#### Cosmos3-Edge FP8

```bash
# Download published Edge FP8 instead of Nano; the launch below serves either tier.
export FP8_MODEL_DIR="$PWD/models/cosmos3-edge-fp8"
export SERVED_MODEL="nvidia/Cosmos3-Edge"
python - <<'DOWNLOAD'
import os
from huggingface_hub import snapshot_download
snapshot_download("nvidia/Cosmos3-Edge-FP8", local_dir=os.environ["FP8_MODEL_DIR"])
DOWNLOAD
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

The variables cell defaults to this Framework checkout. Optionally set `COSMOS3_REPO` to the root of a local checkout of the **cosmos-framework** repository:

```bash
export COSMOS3_REPO=/path/to/cosmos-framework
```

This must be the framework checkout that contains the `cosmos_framework/` Python package—a model directory cannot be used as a source checkout. The notebook adds this path to Python's import path to load `cosmos_framework`. Set it in the shell that launches Jupyter before running the variables cell.

```python
from pathlib import Path
import os


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

COSMOS3_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
COSMOS3_INPUT_DIR.mkdir(parents=True, exist_ok=True)

print("COSMOS_ROOT:", COSMOS_ROOT)
print("COSMOS3_REPO:", COSMOS3_REPO)
print("COSMOS3_INPUT_DIR:", COSMOS3_INPUT_DIR)
print("COSMOS3_OUTPUT_ROOT:", COSMOS3_OUTPUT_ROOT)
print("COSMOS3_VLLM_BASE_URL:", VLLM_BASE_URL)
```

## Create the Inverse-Dynamics Input Spec

Inverse-dynamics inference is driven by a JSONL spec, one line per run. Unlike forward dynamics, each line provides only an input video (`vision_path`) and **no** `action_path` — the action is what the model predicts.

This cell builds that spec from local AV videos, writing it under:

```text
outputs/cosmos3_action_vllm/inputs/action_inverse_dynamics_av_custom.jsonl
```

It mirrors the native PyTorch notebook's spec format. The `vision_path` is written as an absolute path, because the vLLM request cell reads the spec records directly.

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

Runs the active Cosmos3 model on every line of the spec through vLLM-Omni. Inverse dynamics predicts an action, and this cell writes a PyTorch-compatible result file for each run:

```text
<output>/<name>/sample_outputs.json
```

The predicted action trajectory is stored under `outputs[0].content["action"]`, matching the native notebook's visualization cell. vLLM also returns `response.json`, `final.json`, and `action.json` for API debugging.

```python
import json
import time
from pathlib import Path

try:
    import requests
except ImportError as exc:
    raise RuntimeError("Install requests in this notebook kernel: pip install requests") from exc


def check_vllm_server() -> None:
    response = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=10)
    response.raise_for_status()
    print(response.json())


def submit_inverse_dynamics(record: dict) -> dict:
    run_dir = id_output_dir / record["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(record["vision_path"])
    extra_params = {
        "action_mode": "inverse_dynamics",
        "domain_name": record["domain_name"],
        "action_chunk_size": record["action_chunk_size"],
        "image_size": record["image_size"],
        "view_point": record["view_point"],
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
            f"{VLLM_BASE_URL}/v1/videos",
            data={key: str(value) for key, value in form.items()},
            files={"input_reference": (video_path.name, video_file, "video/mp4")},
            timeout=120,
        )
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
    (run_dir / "action.json").write_text(json.dumps(action, indent=2))

    sample_outputs = {"outputs": [{"content": {"action": action["data"]}}]}
    (run_dir / "sample_outputs.json").write_text(json.dumps(sample_outputs, indent=2))

    print("saved", run_dir / "sample_outputs.json")
    print("action shape:", action.get("shape"), "dtype:", action.get("dtype"))
    return {"record": record, "initial": initial, "final": final, "run_dir": run_dir, "action": action}


check_vllm_server()
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
if str(COSMOS3_REPO) not in sys.path:
    sys.path.insert(0, str(COSMOS3_REPO))
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
