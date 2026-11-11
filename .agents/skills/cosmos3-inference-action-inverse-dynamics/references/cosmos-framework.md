<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Action Inverse Dynamics with Cosmos Framework

This notebook runs Cosmos3 **action inverse-dynamics** inference through the native Cosmos Framework PyTorch entrypoint:

```bash
python -m cosmos_framework.scripts.inference
```

Inverse dynamics is the reverse of forward dynamics: given a video, it predicts the ego-motion action trajectory that produced it. This notebook is written as a first-run cookbook: clone or locate the framework source, install dependencies from scratch, verify the GPU environment, build a custom AV input spec, run inference, and visualize the predicted trajectory.

See `run_fd_with_cosmos_framework.ipynb` for the forward-dynamics counterpart (image + trajectory -> video).

Tested path from the audit:

- Framework checkout: `packages/cosmos3`
- Install command: `uv sync --all-extras --group=cu130-train`
- Backend: Cosmos Framework / `cosmos_framework.scripts.inference`
- Models: `Cosmos3-Nano` (default), `Cosmos3-Edge`, and `Cosmos3-Super`

Every inference cell below reads the checkpoint from `COSMOS3_CHECKPOINT_PATH` (set in step 2, default `Cosmos3-Nano`). To run the same inverse-dynamics examples on Edge instead, set `COSMOS3_CHECKPOINT_PATH=Cosmos3-Edge` or `Cosmos3-Super` before the configuration cell — nothing else changes.

> **Skill note (not in the source):** this file is the cookbook notebook `run_id_with_cosmos_framework.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/generator/action/run_id_with_cosmos_framework.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

## 1. Prerequisites

Before running the notebook:

1. Use a Linux machine with NVIDIA GPU access.
2. Make sure your Hugging Face account can access the Cosmos3 model repos.
3. Authenticate with Hugging Face:

```bash
uvx hf@latest auth login
```

or set:

```bash
export HF_TOKEN=<your_token>
```

4. Use a disk/cache location with enough free space. Nano downloads and CUDA dependencies can use tens of GiB.

## 2. Configure Paths

The defaults are intentionally relative to this `cosmos` checkout:

```text
<cosmos>/packages/cosmos3
```

You can override the important knobs before running the next cell:

```bash
export COSMOS3_REPO=/path/to/cosmos-framework
export COSMOS3_GIT_URL=https://github.com/NVIDIA/cosmos-framework.git
export COSMOS3_UV_GROUP=cu130-train  # CUDA 13 driver; use cu128-train for a CUDA 12.x driver
export COSMOS3_OUTPUT_ROOT=/path/to/action/outputs
export COSMOS3_CHECKPOINT_PATH=Cosmos3-Edge  # optional: Cosmos3-Edge or Cosmos3-Super; defaults to Cosmos3-Nano
export HF_HOME=/path/to/large/huggingface/cache
export CUDA_VISIBLE_DEVICES=0
```

`COSMOS3_CHECKPOINT_PATH` selects the model for every inference cell below; leave it unset to use `Cosmos3-Nano`, or set it to `Cosmos3-Edge` or `Cosmos3-Super` to run the same examples on those models.

For SSH access, set `COSMOS3_GIT_URL=git@github.com:NVIDIA/cosmos-framework.git`.

```python
%load_ext autoreload
%autoreload 2

from pathlib import Path
import os
import socket
import subprocess
import sys


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "cookbooks" / "cosmos3").is_dir():
            return path
    return start


def free_local_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def ensure_free_local_port_env(name: str) -> None:
    if name not in os.environ:
        os.environ[name] = free_local_port()


def framework_site_packages(python_bin: Path) -> Path | None:
    venv_root = python_bin.parent.parent
    for site_packages in sorted((venv_root / "lib").glob("python*/site-packages")):
        if (site_packages / "nvidia").is_dir():
            return site_packages
    return None


def nvidia_cuda_library_dirs(python_bin: Path) -> list[Path]:
    site_packages = framework_site_packages(python_bin)
    if site_packages is None:
        return []

    nvidia_root = site_packages / "nvidia"
    lib_dirs = []
    for lib_dir in sorted(nvidia_root.glob("**/lib")):
        if any(lib_dir.glob("lib*.so*")):
            lib_dirs.append(lib_dir)
    return lib_dirs


def torchcodec_ffmpeg_library_dirs(python_bin: Path) -> list[Path]:
    site_packages = framework_site_packages(python_bin)
    if site_packages is None:
        return []

    av_libs = site_packages / "av.libs"
    if not av_libs.is_dir():
        return []

    link_dir = COSMOS3_OUTPUT_ROOT / "torchcodec_ffmpeg_links"
    soname_patterns = {
        "libavcodec.so.62": "libavcodec-*.so.62*",
        "libavdevice.so.62": "libavdevice-*.so.62*",
        "libavfilter.so.11": "libavfilter-*.so.11*",
        "libavformat.so.62": "libavformat-*.so.62*",
        "libavutil.so.60": "libavutil-*.so.60*",
        "libswresample.so.6": "libswresample-*.so.6*",
        "libswscale.so.9": "libswscale-*.so.9*",
    }

    linked_any = False
    for soname, pattern in soname_patterns.items():
        matches = sorted(av_libs.glob(pattern))
        if not matches:
            continue
        link_dir.mkdir(parents=True, exist_ok=True)
        link = link_dir / soname
        target = matches[-1].resolve()
        if link.is_symlink() and link.resolve() == target:
            linked_any = True
            continue
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)
        linked_any = True

    return [link_dir, av_libs] if linked_any else []


def set_nvidia_package_home(env: dict[str, str], site_packages: Path | None, env_name: str, package_name: str) -> None:
    if site_packages is None:
        return
    package_dir = site_packages / "nvidia" / package_name
    if package_dir.is_dir():
        env.setdefault(env_name, str(package_dir))


def ensure_nvidia_package_alias(site_packages: Path | None, alias_name: str, package_name: str) -> None:
    if site_packages is None:
        return
    package_dir = site_packages / "nvidia" / package_name
    alias_dir = site_packages / "nvidia" / alias_name
    if not package_dir.is_dir():
        return
    if alias_dir.is_symlink() and not alias_dir.exists():
        alias_dir.unlink()
    if not alias_dir.exists():
        alias_dir.symlink_to(package_dir, target_is_directory=True)


def prepend_env_paths(env: dict[str, str], name: str, paths: list[Path]) -> None:
    new_paths = [str(path) for path in paths if path.exists()]
    old_paths = [path for path in env.get(name, "").split(":") if path]
    merged = []
    for path in [*new_paths, *old_paths]:
        if path not in merged:
            merged.append(path)
    if merged:
        env[name] = ":".join(merged)


def configure_cosmos_framework_runtime_env() -> None:
    python_bin = COSMOS3_REPO / ".venv" / "bin" / "python"
    assert python_bin.exists(), f"missing python executable: {python_bin}. Run the install cell first."

    old_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    cuda_lib_dirs = nvidia_cuda_library_dirs(python_bin)
    ffmpeg_lib_dirs = torchcodec_ffmpeg_library_dirs(python_bin)
    prepend_env_paths(os.environ, "PYTHONPATH", [COSMOS3_REPO])
    prepend_env_paths(os.environ, "LD_LIBRARY_PATH", [*ffmpeg_lib_dirs, *cuda_lib_dirs])

    site_packages = framework_site_packages(python_bin)
    ensure_nvidia_package_alias(site_packages, "cudart", "cuda_runtime")
    set_nvidia_package_home(os.environ, site_packages, "CUDNN_HOME", "cudnn")
    set_nvidia_package_home(os.environ, site_packages, "CUDART_HOME", "cuda_runtime")
    set_nvidia_package_home(os.environ, site_packages, "NVRTC_HOME", "cuda_nvrtc")
    set_nvidia_package_home(os.environ, site_packages, "CURAND_HOME", "curand")

    cuda_include_dir = site_packages / "nvidia" / "cuda_runtime" / "include" if site_packages else None
    if cuda_include_dir and cuda_include_dir.exists():
        os.environ.setdefault("NVTE_CUDA_INCLUDE_DIR", str(cuda_include_dir))

    if os.environ.get("LD_LIBRARY_PATH", "") != old_ld_library_path and os.environ.get("COSMOS3_LD_LIBRARY_PATH_READY") != "1":
        os.environ["COSMOS3_LD_LIBRARY_PATH_READY"] = "1"
        print("Updated LD_LIBRARY_PATH for the Cosmos Framework venv; restarting the notebook kernel once.")
        print("After the kernel reconnects, rerun Step 2 and this verify cell, then continue.")
        os.execv(sys.executable, [sys.executable, *sys.argv])

    if cuda_lib_dirs:
        print(f"prepended {len(cuda_lib_dirs)} NVIDIA CUDA library dirs to LD_LIBRARY_PATH")
    else:
        print("warning: no NVIDIA CUDA library dirs found in the Cosmos Framework venv")
    if ffmpeg_lib_dirs:
        print("prepended PyAV FFmpeg shared-library dirs to LD_LIBRARY_PATH")


def resolve_input(rel_path: str) -> str:
    path = (COSMOS_ROOT / rel_path).resolve()
    assert path.exists(), f"missing input: {path}"
    return str(path)


COSMOS_ROOT = find_repo_root(Path.cwd().resolve())
COSMOS3_ACTION_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3" / "generator" / "action"
COSMOS3_REPO = Path(os.environ.get("COSMOS3_REPO", COSMOS_ROOT / "packages" / "cosmos3")).resolve()
COSMOS3_GIT_URL = os.environ.get(
    "COSMOS3_GIT_URL",
    "https://github.com/NVIDIA/cosmos-framework.git",
)
COSMOS3_UV_GROUP = os.environ.get("COSMOS3_UV_GROUP", "cu130-train")
COSMOS3_OUTPUT_ROOT = Path(
    os.environ.get("COSMOS3_OUTPUT_ROOT", COSMOS3_REPO / "outputs" / "cookbooks" / "cosmos3" / "generator" / "action")
).resolve()
COSMOS3_INPUT_DIR = COSMOS3_OUTPUT_ROOT / "inputs"

os.environ["COSMOS_ROOT"] = str(COSMOS_ROOT)
os.environ["COSMOS3_ACTION_ROOT"] = str(COSMOS3_ACTION_ROOT)
os.environ["COSMOS3_REPO"] = str(COSMOS3_REPO)
os.environ["COSMOS3_GIT_URL"] = COSMOS3_GIT_URL
os.environ["COSMOS3_UV_GROUP"] = COSMOS3_UV_GROUP
os.environ["COSMOS3_OUTPUT_ROOT"] = str(COSMOS3_OUTPUT_ROOT)
os.environ["COSMOS3_INPUT_DIR"] = str(COSMOS3_INPUT_DIR)
os.environ.setdefault("COSMOS3_CHECKPOINT_PATH", "Cosmos3-Nano")
os.environ.setdefault("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv"))
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("COSMOS3_MASTER_ADDR", "127.0.0.1")
ensure_free_local_port_env("COSMOS3_NANO_TEXT_MASTER_PORT")
ensure_free_local_port_env("COSMOS3_NANO_IMAGE_MASTER_PORT")

print("cosmos root:", COSMOS_ROOT)
print("Action assets:", COSMOS3_ACTION_ROOT / "assets")
print("Cosmos Framework path:", COSMOS3_REPO)
print("Framework git URL:", COSMOS3_GIT_URL)
print("uv dependency group:", COSMOS3_UV_GROUP)
print("output root:", COSMOS3_OUTPUT_ROOT)
print("UV_CACHE_DIR:", os.environ["UV_CACHE_DIR"])
print("HF_HOME:", os.environ["HF_HOME"])
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
print("checkpoint path:", os.environ["COSMOS3_CHECKPOINT_PATH"])
```

## 3. Clone or Reuse Cosmos Framework

This cell creates `packages/` and clones the framework into `packages/cosmos3` if it is not already there. The default clone URL uses HTTPS so users do not need to configure an SSH key unless their access requires it.

```bash
set -euo pipefail

mkdir -p "$(dirname "$COSMOS3_REPO")"

if [ -d "$COSMOS3_REPO/.git" ]; then
  echo "Using existing framework checkout: $COSMOS3_REPO"
else
  echo "Cloning $COSMOS3_GIT_URL into $COSMOS3_REPO"
  git clone "$COSMOS3_GIT_URL" "$COSMOS3_REPO"
fi

cd "$COSMOS3_REPO"
git status --short --branch
git remote -v
```

## 4. Install Cosmos Framework Dependencies

This is the full install path used for the Cosmos Framework audit. It is heavier than an inference-only install, but it avoids missing training-extra dependencies that are currently imported by the framework inference path.

The dependency group selects the CUDA build of `torch`, and it must match your NVIDIA driver:

| Driver CUDA                | `COSMOS3_UV_GROUP`      |
| -------------------------- | ----------------------- |
| 13.x                       | `cu130-train` (default) |
| 12.x (most machines today) | `cu128-train`           |

The default `cu130-train` group installs CUDA 13 wheels, which need a CUDA 13 driver. On a CUDA 12.x driver, set `COSMOS3_UV_GROUP=cu128-train` before the configuration cell, otherwise the verify cell below reports `cuda available: False`. (These groups are defined in the framework's `pyproject.toml`; only `cu130-train` and `cu128-train` are provided.)

Expected behavior:

- Creates `.venv` inside `packages/cosmos3`.
- Downloads CUDA/Torch dependencies.
- May take several minutes.
- Sets `GIT_LFS_SKIP_SMUDGE=1` during install so optional git-LFS test artifacts do not block the local dependency mirror.
- May print a uv cache hardlink warning if your cache and repo are on different filesystems; this is usually harmless.

```bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

export GIT_LFS_SKIP_SMUDGE=1
cd "$COSMOS3_REPO"
uv sync --all-extras --group="$COSMOS3_UV_GROUP"
```

## 5. Verify GPU and Python Environment

The Cosmos Framework commands below use `CUDA_VISIBLE_DEVICES=0` by default. Adjust this if you want a different GPU.

This cell also prepares the notebook kernel for action-data helper imports by adding CUDA wheel libraries and PyAV FFmpeg libraries from the framework venv to the runtime environment. If the kernel restarts once after this cell, rerun Step 2 and this verify cell, then continue.

```python
configure_cosmos_framework_runtime_env()

verify_code = r'''
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
'''
subprocess.run(
    [str(COSMOS3_REPO / ".venv" / "bin" / "python"), "-c", verify_code],
    cwd=str(COSMOS3_REPO),
    env=os.environ.copy(),
    check=True,
)
```

## Import dependencies and define helper functions

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
import mediapy as media
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# The notebook kernel may differ from the framework venv, so put the repo on the
# path before importing `cosmos_framework`.
if str(COSMOS3_REPO) not in sys.path:
    sys.path.insert(0, str(COSMOS3_REPO))

from cosmos_framework.data.generator.action.action_normalization import denormalize_action
from cosmos_framework.data.generator.action.utils.pose_utils import pose_rel_to_abs
from cosmos_framework.tools.visualize.video import save_img_or_video

# frustum: apex + image-rectangle corners (camera +Z forward), and their edges
_FRUSTUM = np.array([[0, 0, 0], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
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
    """3D camera trajectory (with frustums) + a top-down bird's-eye view.

    AV convention: world Y is up, world +Z is the heading. `vertical_exaggeration`
    stretches only the up-axis box (uniform world scaling, so frustums never skew);
    1.0 = geometrically faithful. The 3D plot reorders world (X, Y, Z) -> (X, Z, Y)
    so Y points up on screen.
    """
    poses_abs = np.asarray(poses_abs)
    pos = poses_abs[:, :3, 3]  # camera centers [T, 3]
    fwd = poses_abs[:, :3, 2]  # heading (+Z) [T, 3]
    T = len(pos)
    colors = plt.get_cmap(cmap)(np.arange(T) / max(T - 1, 1))
    scale = max(np.ptp(pos, axis=0).max() * scale_frac, 1e-3)
    step = max(1, T // max(n_frustums, 1))
    xzy = [0, 2, 1]  # world (X,Y,Z) -> plot (X, Z, Y-up)

    fig = plt.figure(figsize=(14, 6))

    # (1) 3D perspective with frustums
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    path = pos[:, xzy]
    ax.plot(*path.T, color="0.6", lw=1.0, alpha=0.7)
    lines, lcolors, allpts = [], [], [path]
    for i in range(0, T, step):
        cw = (
            (_FRUSTUM * [aspect, 1, 1] * scale * np.tan(np.radians(fov_deg) / 2)) @ poses_abs[i, :3, :3].T
            + poses_abs[i, :3, 3]
        )[:, xzy]  # frustum in plot coords
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

    # (2) top-down bird's-eye view (X-Z ground plane)
    ax2 = fig.add_subplot(1, 2, 2)
    seg = np.stack([pos[:-1, [0, 2]], pos[1:, [0, 2]]], axis=1)
    lc = LineCollection(seg, cmap=cmap, norm=plt.Normalize(0, T - 1), linewidth=2.5)
    lc.set_array(np.arange(T - 1))
    ax2.add_collection(lc)
    ax2.quiver(
        pos[::step, 0],
        pos[::step, 2],
        fwd[::step, 0],
        fwd[::step, 2],
        color=colors[::step],
        angles="xy",
        width=0.005,
        scale=22,
        zorder=3,
    )
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


def create_record_from_dataset(dataset, num_chunks, chunk_length):
    chunk_starts = [chunk_idx * chunk_length for chunk_idx in range(num_chunks)]
    assert chunk_starts[-1] < len(dataset)

    COSMOS3_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for chunk_idx, sample_idx in enumerate(chunk_starts):
        data_sample = dataset[sample_idx]
        domain_name = dataset.domain_name
        chunk_name = f"{domain_name}_id_chunk_{chunk_idx:02d}"
        vision_path = COSMOS3_INPUT_DIR / f"{chunk_name}.mp4"
        save_img_or_video(data_sample["video"], str(vision_path.with_suffix("")), quality=10)
        records.append(
            {
                "action_chunk_size": chunk_length,
                "domain_name": domain_name,
                "fps": int(data_sample["conditioning_fps"]),
                "image_size": 480,
                "view_point": dataset.viewpoint,
                "model_mode": "inverse_dynamics",
                "name": chunk_name,
                "prompt": data_sample["ai_caption"],
                "seed": 0,
                "vision_path": str(vision_path),
            }
        )
    return records
```

## 6. AV Inverse Dynamics

In this example, we show how to generate ego poses from a set of autonomous vehicle ego videos using Cosmos3-Nano.

## Create the AV Inverse-Dynamics Input Spec

Inverse-dynamics inference is driven by a JSONL spec, one line per run. Unlike forward dynamics, each line provides only an input video (`vision_path`) and **no** `action_path` - the action is what the model predicts.

This cell builds that spec from local AV videos, writing it under:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/generator/action/inputs/action_inverse_dynamics_av_custom.jsonl
```

It does not modify the shipped framework examples. The `vision_path` is written as an absolute path, because the loader resolves relative paths against the spec file's own directory.

```python
# `resolve_input` and the COSMOS3_* paths come from the configuration cell.

# Local inputs, relative to the cookbook root.
input_videos = {
    "av_inverse_0": "cookbooks/cosmos3/generator/action/assets/videos/av_0.mp4",
    "av_inverse_1": "cookbooks/cosmos3/generator/action/assets/videos/av_1.mp4",
}

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

## Preview the AV Input Video(s)

Preview each input video before running inference. `Video(..., embed=True)` base64-inlines the file, and these AV clips are several MB each, so we first transcode a small preview (~150 KB) with the ffmpeg binary provided by `imageio-ffmpeg`, then embed it. The original input videos are left untouched.

```python
# `records` comes from the prepare cell; preview each input video.
for record in records:
    name = record["name"]
    src = Path(record["vision_path"])
    preview = media.read_video(src)
    fps = record["fps"]
    print(f"{name} fps={fps}")
    media.show_video(preview, fps=fps)
```

## Run AV Inverse-Dynamics Inference

Runs `Cosmos3-Nano` on every line of the spec. Inverse dynamics predicts an action rather than a video, so each run writes its result to:

```text
<output>/<name>/sample_outputs.json
```

The predicted action trajectory is stored under `outputs[0].content["action"]`.

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_NANO_TEXT_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_ID_INPUT" \
  -o "$COSMOS3_ID_OUTPUT" \
  --checkpoint-path "$COSMOS3_CHECKPOINT_PATH" \
  --seed 0 \
  --benchmark
```

## Visualize the AV Predicted Action

The output action can be fed into `pose_rel_to_abs` to convert back to camera poses (camera-to-world transformation, opencv convention, unit in meter).

We plot the camera poses predicted from each input video, as a 3D camera path (with frustums) and a top-down bird's-eye view.

```python
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

## 7. Bridge Inverse Dynamics

In this example, we show how to start from a LeRobot dataset of Bridge and generate end-effector poses for robotics manipulation.

### Create the Bridge Inverse-Dynamics Input Spec

This cell builds input spec from Bridge dataset, writing it under:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/generator/action/inputs/action_inverse_dynamics_bridge_orig_lerobot_custom.jsonl
```

```python
from cosmos_framework.data.generator.action.datasets import BridgeOrigLeRobotDataset

num_chunks = 1
chunk_length = 16
dataset_root = resolve_input("cookbooks/cosmos3/generator/action/assets/bridge_lerobot_example")
bridge_dataset = BridgeOrigLeRobotDataset(root=dataset_root,chunk_length=chunk_length)
bridge_records = create_record_from_dataset(bridge_dataset, num_chunks, chunk_length)

norm_stats = bridge_dataset.load_action_stats()
domain_name = bridge_dataset.domain_name
bridge_id_input_path = COSMOS3_INPUT_DIR / f"action_inverse_dynamics_{domain_name}_custom.jsonl"
bridge_id_input_path.write_text("".join(json.dumps(r) + "\n" for r in bridge_records))
bridge_id_output_dir = COSMOS3_OUTPUT_ROOT / f"action_inverse_dynamics_{domain_name}_custom"

os.environ["COSMOS3_BRIDGE_ID_INPUT"] = str(bridge_id_input_path)
os.environ["COSMOS3_BRIDGE_ID_OUTPUT"] = str(bridge_id_output_dir)
```

## Run Bridge Inverse-Dynamics Inference

Runs `Cosmos3-Nano` on every line of the spec. Inverse dynamics predicts an action rather than a video, so each run writes its result to:

```text
<output>/<name>/sample_outputs.json
```

The predicted action trajectory is stored under `outputs[0].content["action"]`.

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_NANO_TEXT_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_BRIDGE_ID_INPUT" \
  -o "$COSMOS3_BRIDGE_ID_OUTPUT" \
  --checkpoint-path "$COSMOS3_CHECKPOINT_PATH" \
  --seed 0 \
  --benchmark
```

## Visualize the Bridge Iput Video and Predicted Action

The output action of Bridge dataset are normalized, we use `denormalize_action` and `pose_rel_to_abs` to convert back to camera poses (camera-to-world transformation, opencv convention, unit in meter).

We plot the camera poses predicted from each input video, as a 3D camera path (with frustums) and a top-down bird's-eye view.

```python
# `records` and `id_output_dir` come from the prepare cell; read each run's
# predicted action from its sample_outputs.json.
for record in bridge_records:
    name = record["name"]

    # Display video.
    src = Path(record["vision_path"])
    preview = media.read_video(src)
    fps = record["fps"]
    print(f"{name} fps={fps}")
    media.show_video(preview, fps=fps)

    # Visualize Poses.
    outputs = json.loads((bridge_id_output_dir / name / "sample_outputs.json").read_text())
    action = torch.as_tensor(outputs["outputs"][0]["content"]["action"])
    action = denormalize_action(action, method="quantile", stats=norm_stats)
    poses_rel = np.asarray(action[..., :9])  # [T-1, 9] = [translation(3), rot6d(6)]
    poses_abs = pose_rel_to_abs(
        poses_rel,
        rotation_format="rot6d",
        pose_convention="backward_framewise",
    )  # [T, 4, 4] camera-to-world
    visualize_pose(poses_abs, title=f"{name}: predicted end-effector trajectory + frustums ({len(poses_abs)} frames)", show=True)
```

## 8. AgiBotWorld-Beta Inverse Dynamics

In this example, we show how to start from a LeRobot dataset of AgiBotWorld-Beta and generate end-effector poses for robotics manipulation.

### Create the AgiBotWorld-Beta Autoregressive Inverse-Dynamics Input Spec

This cell builds input spec from AgiBotWorld-Beta dataset, writing it under:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/generator/action/inputs/action_inverse_dynamics_agibotworld_custom.jsonl
```

```python
from cosmos_framework.data.generator.action.datasets import AgiBotWorldBetaLeRobotDataset

num_chunks = 5
chunk_length = 16

dataset_root = resolve_input("cookbooks/cosmos3/generator/action/assets/agibotworld_beta_lerobot_example")
agibot_dataset = AgiBotWorldBetaLeRobotDataset(
    root=dataset_root,
    chunk_length=chunk_length,
    fps=10,
    sample_stride=3,  # We downsampled the frames by 3 (from an original 30 FPS).
)
agibot_records = create_record_from_dataset(agibot_dataset, num_chunks, chunk_length)
norm_stats = agibot_dataset.load_action_stats()
domain_name = agibot_dataset.domain_name

agibot_id_input_path = COSMOS3_INPUT_DIR / f"action_inverse_dynamics_{domain_name}_custom.jsonl"
agibot_id_input_path.write_text("".join(json.dumps(r) + "\n" for r in agibot_records))
agibot_id_output_dir = COSMOS3_OUTPUT_ROOT / f"action_inverse_dynamics_{domain_name}_custom"

os.environ["COSMOS3_AGIBOT_ID_INPUT"] = str(agibot_id_input_path)
os.environ["COSMOS3_AGIBOT_ID_OUTPUT"] = str(agibot_id_output_dir)
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_NANO_TEXT_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_AGIBOT_ID_INPUT" \
  -o "$COSMOS3_AGIBOT_ID_OUTPUT" \
  --checkpoint-path "$COSMOS3_CHECKPOINT_PATH" \
  --seed 0 \
  --benchmark
```

```python
actions = []
input_videos = []
for record in agibot_records:
    name = record["name"]

    # Gather input video previews.
    src = record["vision_path"]
    preview = media.read_video(src)
    input_videos.append(preview[1:])  # Discards last frame.

    # Gather Generated Poses.
    outputs = json.loads((agibot_id_output_dir / name / "sample_outputs.json").read_text())
    action = torch.as_tensor(outputs["outputs"][0]["content"]["action"])
    action = denormalize_action(action, method="quantile", stats=norm_stats)
    actions.append(action)

# Visualize Input video.
input_videos = np.concat(input_videos, axis=0)
fps = agibot_records[0]["fps"]
print(f"Input Video, fps={fps}")
media.show_video(input_videos, fps=fps, width=512)

# Visualize Right Wrist Poses.
pose_to_viz = "right_wrist"
pose_name_to_idx = { "head_camera": 0, "right_wrist": 9, "left_wrist":19 }

pose_idx = pose_name_to_idx[pose_to_viz]
actions = torch.concat(actions, axis=0)
poses_rel = np.asarray(actions[..., pose_idx:pose_idx + 9])  # [T-1, 9] = [translation(3), rot6d(6)]
poses_abs = pose_rel_to_abs(
    poses_rel,
    rotation_format="rot6d",
    pose_convention="backward_framewise",
)  # [T, 4, 4] camera-to-world

visualize_pose(poses_abs, title=f"AgiBotWorld-Beta: predicted `{pose_to_viz}` trajectory ({len(poses_abs)} frames)", show=True)
```

## 9. RoboMIND Franka Inverse Dynamics

In this example, we show how to start from a LeRobot dataset of RoboMIND Franka and generate end-effector poses for robotics manipulation.

### Create the RoboMIND Franka Autoregressive Inverse-Dynamics Input Spec

This cell builds input spec from RoboMIND Franka dataset, writing it under:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/generator/action/inputs/action_inverse_dynamics_robomind-franka_custom.jsonl
```

```python
from cosmos_framework.data.generator.action.datasets import RoboMINDFrankaDataset

num_chunks = 4
chunk_length = 16

dataset_root = resolve_input("cookbooks/cosmos3/generator/action/assets/robomind_lerobot_example/franka")
robomind_franka_dataset = RoboMINDFrankaDataset(
    root=dataset_root,
    chunk_length=chunk_length,
    embodiment_type="robomind-franka",
    fps=10,
)

robomind_franka_records = create_record_from_dataset(robomind_franka_dataset, num_chunks, chunk_length)
norm_stats = robomind_franka_dataset.load_action_stats()
domain_name = robomind_franka_dataset.domain_name

robomind_franka_id_input_path = COSMOS3_INPUT_DIR / f"action_inverse_dynamics_{domain_name}_custom.jsonl"
robomind_franka_id_input_path.write_text("".join(json.dumps(r) + "\n" for r in robomind_franka_records))
robomind_franka_id_output_dir = COSMOS3_OUTPUT_ROOT / f"action_inverse_dynamics_{domain_name}_custom"

os.environ["COSMOS3_ROBMIND_FRANKA_ID_INPUT"] = str(robomind_franka_id_input_path)
os.environ["COSMOS3_ROBMIND_FRANKA_ID_OUTPUT"] = str(robomind_franka_id_output_dir)
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_NANO_TEXT_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_ROBMIND_FRANKA_ID_INPUT" \
  -o "$COSMOS3_ROBMIND_FRANKA_ID_OUTPUT" \
  --checkpoint-path "$COSMOS3_CHECKPOINT_PATH" \
  --seed 0 \
  --benchmark
```

```python
actions = []
input_videos = []
for record in robomind_franka_records:
    name = record["name"]

    # Gather input video previews.
    src = record["vision_path"]
    preview = media.read_video(src)
    input_videos.append(preview[1:])  # Discards last frame.

    # Gather Generated Poses.
    outputs = json.loads((robomind_franka_id_output_dir / name / "sample_outputs.json").read_text())
    action = torch.as_tensor(outputs["outputs"][0]["content"]["action"])
    action = denormalize_action(action, method="quantile", stats=norm_stats)
    actions.append(action)

# Visualize Input video.
input_videos = np.concat(input_videos, axis=0)
fps = robomind_franka_records[0]["fps"]
print(f"Input Video, fps={fps}")
media.show_video(input_videos, fps=fps, width=512)

# Visualize Left Wrist Poses.
pose_to_viz = "left"
pose_name_to_idx = { "left": 0, "right": 10}

pose_idx = pose_name_to_idx[pose_to_viz]
actions = torch.concat(actions, axis=0)
poses_rel = np.asarray(actions[..., pose_idx:pose_idx + 9])  # [T-1, 9] = [translation(3), rot6d(6)]
poses_abs = pose_rel_to_abs(
    poses_rel,
    rotation_format="rot6d",
    pose_convention="backward_framewise",
)  # [T, 4, 4] camera-to-world

visualize_pose(poses_abs, title=f"RoboMIND Franka: predicted `{pose_to_viz}` trajectory ({len(poses_abs)} frames)", show=True)
```

## 10. RoboMIND Franka dual-arm Inverse Dynamics

In this example, we show how to start from a LeRobot dataset of RoboMIND Franka dual-arm and generate end-effector poses for robotics manipulation.

### Create the RoboMIND Franka dual-arm Autoregressive Inverse-Dynamics Input Spec

This cell builds input spec from RoboMIND Franka dual-arm dataset, writing it under:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/generator/action/inputs/action_inverse_dynamics_robomind-franka-dual_custom.jsonl
```

```python
from cosmos_framework.data.generator.action.datasets import RoboMINDFrankaDataset

num_chunks = 5
chunk_length = 16

dataset_root = resolve_input("cookbooks/cosmos3/generator/action/assets/robomind_lerobot_example/franka_dual")
robomind_franka_dual_dataset = RoboMINDFrankaDataset(
    root=dataset_root,
    chunk_length=chunk_length,
    embodiment_type="robomind-franka-dual",
    fps=10,
)

robomind_franka_dual_records = create_record_from_dataset(robomind_franka_dual_dataset, num_chunks, chunk_length)
norm_stats = robomind_franka_dual_dataset.load_action_stats()
domain_name = robomind_franka_dual_dataset.domain_name

robomind_franka_dual_id_input_path = COSMOS3_INPUT_DIR / f"action_inverse_dynamics_{domain_name}_custom.jsonl"
robomind_franka_dual_id_input_path.write_text("".join(json.dumps(r) + "\n" for r in robomind_franka_dual_records))
robomind_franka_dual_id_output_dir = COSMOS3_OUTPUT_ROOT / f"action_inverse_dynamics_{domain_name}_custom"

os.environ["COSMOS3_ROBMIND_FRANKA_DUAL_ID_INPUT"] = str(robomind_franka_dual_id_input_path)
os.environ["COSMOS3_ROBMIND_FRANKA_DUAL_ID_OUTPUT"] = str(robomind_franka_dual_id_output_dir)
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_NANO_TEXT_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_ROBMIND_FRANKA_DUAL_ID_INPUT" \
  -o "$COSMOS3_ROBMIND_FRANKA_DUAL_ID_OUTPUT" \
  --checkpoint-path "$COSMOS3_CHECKPOINT_PATH" \
  --seed 0 \
  --benchmark
```

```python
actions = []
input_videos = []
for record in robomind_franka_dual_records:
    name = record["name"]

    # Gather input video previews.
    src = record["vision_path"]
    preview = media.read_video(src)
    input_videos.append(preview[1:])  # Discards last frame.

    # Gather Generated Poses.
    outputs = json.loads((robomind_franka_dual_id_output_dir / name / "sample_outputs.json").read_text())
    action = torch.as_tensor(outputs["outputs"][0]["content"]["action"])
    action = denormalize_action(action, method="quantile", stats=norm_stats)
    actions.append(action)

# Visualize Input video.
input_videos = np.concat(input_videos, axis=0)
fps = robomind_franka_dual_records[0]["fps"]
print(f"Input Video, fps={fps}")
media.show_video(input_videos, fps=fps, width=512)

# Visualize Left Wrist Poses.
pose_to_viz = "right"
pose_name_to_idx = { "left": 0, "right": 10}

pose_idx = pose_name_to_idx[pose_to_viz]
actions = torch.concat(actions, axis=0)
poses_rel = np.asarray(actions[..., pose_idx:pose_idx + 9])  # [T-1, 9] = [translation(3), rot6d(6)]
poses_abs = pose_rel_to_abs(
    poses_rel,
    rotation_format="rot6d",
    pose_convention="backward_framewise",
)  # [T, 4, 4] camera-to-world

visualize_pose(poses_abs, title=f"RoboMIND Franka dual-arm : predicted `{pose_to_viz}` trajectory ({len(poses_abs)} frames)", show=True)
```

## 11. RoboMIND UR Inverse Dynamics

In this example, we show how to start from a LeRobot dataset of RoboMIND UR and generate end-effector poses for robotics manipulation.

### Create the RoboMIND UR Autoregressive Inverse-Dynamics Input Spec

This cell builds input spec from RoboMIND UR dataset, writing it under:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/generator/action/inputs/action_inverse_dynamics_robomind-ur_custom.jsonl
```

```python
from cosmos_framework.data.generator.action.datasets import RoboMINDURDataset

num_chunks = 5
chunk_length = 16

dataset_root = resolve_input("cookbooks/cosmos3/generator/action/assets/robomind_lerobot_example/ur")
robomind_ur_dataset = RoboMINDURDataset(
    root=dataset_root,
    chunk_length=chunk_length,
    fps=10,
)

robomind_ur_records = create_record_from_dataset(robomind_ur_dataset, num_chunks, chunk_length)
norm_stats = robomind_ur_dataset.load_action_stats()
domain_name = robomind_ur_dataset.domain_name

robomind_ur_id_input_path = COSMOS3_INPUT_DIR / f"action_inverse_dynamics_{domain_name}_custom.jsonl"
robomind_ur_id_input_path.write_text("".join(json.dumps(r) + "\n" for r in robomind_ur_records))
robomind_ur_id_output_dir = COSMOS3_OUTPUT_ROOT / f"action_inverse_dynamics_{domain_name}_custom"

os.environ["COSMOS3_ROBMIND_UR_ID_INPUT"] = str(robomind_ur_id_input_path)
os.environ["COSMOS3_ROBMIND_UR_ID_OUTPUT"] = str(robomind_ur_id_output_dir)
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_NANO_TEXT_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_ROBMIND_UR_ID_INPUT" \
  -o "$COSMOS3_ROBMIND_UR_ID_OUTPUT" \
  --checkpoint-path "$COSMOS3_CHECKPOINT_PATH" \
  --seed 0 \
  --benchmark
```

```python
actions = []
input_videos = []
for record in robomind_ur_records:
    name = record["name"]

    # Gather input video previews.
    src = record["vision_path"]
    preview = media.read_video(src)
    input_videos.append(preview[1:])  # Discards last frame.

    # Gather Generated Poses.
    outputs = json.loads((robomind_ur_id_output_dir / name / "sample_outputs.json").read_text())
    action = torch.as_tensor(outputs["outputs"][0]["content"]["action"])
    action = denormalize_action(action, method="quantile", stats=norm_stats)
    actions.append(action)

# Visualize Input video.
input_videos = np.concat(input_videos, axis=0)
fps = robomind_ur_records[0]["fps"]
print(f"Input Video, fps={fps}")
media.show_video(input_videos, fps=fps, width=512)

# Visualize Left Wrist Poses.
pose_to_viz = "left"
pose_name_to_idx = { "left": 0, "right": 9}

pose_idx = pose_name_to_idx[pose_to_viz]
actions = torch.concat(actions, axis=0)
poses_rel = np.asarray(actions[..., pose_idx:pose_idx + 9])  # [T-1, 9] = [translation(3), rot6d(6)]
poses_abs = pose_rel_to_abs(
    poses_rel,
    rotation_format="rot6d",
    pose_convention="backward_framewise",
)  # [T, 4, 4] camera-to-world

visualize_pose(poses_abs, title=f"RoboMIND UR : predicted `{pose_to_viz}` trajectory ({len(poses_abs)} frames)", show=True)
```

## 12. UMI Inverse Dynamics

In this example, we show how to start from a LeRobot dataset of UMI and generate end-effector poses for robotics manipulation.

### Create the UMI Autoregressive Inverse-Dynamics Input Spec

This cell builds input spec from UMI dataset, writing it under:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/generator/action/inputs/action_inverse_dynamics_umi_custom.jsonl
```

```python
from cosmos_framework.data.generator.action.datasets import UMILeRobotDataset

num_chunks = 5
chunk_length = 16

dataset_root = resolve_input("cookbooks/cosmos3/generator/action/assets/umi_lerobot_example")
umi_dataset = UMILeRobotDataset(root=dataset_root,chunk_length=chunk_length)
umi_records = create_record_from_dataset(umi_dataset, num_chunks, chunk_length)

norm_stats = umi_dataset.load_action_stats()
domain_name = umi_dataset.domain_name
umi_id_input_path = COSMOS3_INPUT_DIR / f"action_inverse_dynamics_{domain_name}_custom.jsonl"
umi_id_input_path.write_text("".join(json.dumps(r) + "\n" for r in umi_records))
umi_id_output_dir = COSMOS3_OUTPUT_ROOT / f"action_inverse_dynamics_{domain_name}_custom"

os.environ["COSMOS3_UMI_ID_INPUT"] = str(umi_id_input_path)
os.environ["COSMOS3_UMI_ID_OUTPUT"] = str(umi_id_output_dir)
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_NANO_TEXT_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_UMI_ID_INPUT" \
  -o "$COSMOS3_UMI_ID_OUTPUT" \
  --checkpoint-path "$COSMOS3_CHECKPOINT_PATH" \
  --seed 0 \
  --benchmark
```

```python
actions = []
input_videos = []
for record in umi_records:
    name = record["name"]

    # Gather input video previews.
    src = record["vision_path"]
    preview = media.read_video(src)
    input_videos.append(preview[1:])  # Discards last frame.

    # Gather Generated Poses.
    outputs = json.loads((umi_id_output_dir / name / "sample_outputs.json").read_text())
    action = torch.as_tensor(outputs["outputs"][0]["content"]["action"])
    action = denormalize_action(action, method="quantile", stats=norm_stats)
    actions.append(action)

# Visualize Input video.
input_videos = np.concat(input_videos, axis=0)
fps = umi_records[0]["fps"]
print(f"Input Video, fps={fps}")
media.show_video(input_videos, fps=fps, width=512)

actions = torch.concat(actions, axis=0)
poses_rel = actions[..., :9].numpy()  # [T-1, 9] = [translation(3), rot6d(6)]

poses_abs = pose_rel_to_abs(
    poses_rel,
    rotation_format="rot6d",
    pose_convention="backward_framewise",
)  # [T, 4, 4] camera-to-world

visualize_pose(poses_abs, title=f"UMI : predicted end-effector trajectory ({len(poses_abs)} frames)", show=True)
```

## 13. Fractal Inverse Dynamics

In this example, we show how to start from a LeRobot dataset of Fractal (Google RT-1) and generate end-effector poses for robotics manipulation.

### Create the Fractal Autoregressive Inverse-Dynamics Input Spec

This cell builds input spec from Fractal dataset, writing it under:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/generator/action/inputs/action_inverse_dynamics_fractal_custom.jsonl
```

```python
from cosmos_framework.data.generator.action.datasets import FractalLeRobotDataset

num_chunks = 2
chunk_length = 16

dataset_root = resolve_input("cookbooks/cosmos3/generator/action/assets/fractal_lerobot_example")
fractal_dataset = FractalLeRobotDataset(root=dataset_root,chunk_length=chunk_length)
fractal_records = create_record_from_dataset(fractal_dataset, num_chunks, chunk_length)

norm_stats = fractal_dataset.load_action_stats()
domain_name = fractal_dataset.domain_name
fractal_id_input_path = COSMOS3_INPUT_DIR / f"action_inverse_dynamics_{domain_name}_custom.jsonl"
fractal_id_input_path.write_text("".join(json.dumps(r) + "\n" for r in fractal_records))
fractal_id_output_dir = COSMOS3_OUTPUT_ROOT / f"action_inverse_dynamics_{domain_name}_custom"

os.environ["COSMOS3_FRACTAL_ID_INPUT"] = str(fractal_id_input_path)
os.environ["COSMOS3_FRACTAL_ID_OUTPUT"] = str(fractal_id_output_dir)
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_NANO_TEXT_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_FRACTAL_ID_INPUT" \
  -o "$COSMOS3_FRACTAL_ID_OUTPUT" \
  --checkpoint-path "$COSMOS3_CHECKPOINT_PATH" \
  --seed 0 \
  --benchmark
```

```python
actions = []
input_videos = []
for record in fractal_records:
    name = record["name"]

    # Gather input video previews.
    src = record["vision_path"]
    preview = media.read_video(src)
    input_videos.append(preview[1:])  # Discards last frame.

    # Gather Generated Poses.
    outputs = json.loads((fractal_id_output_dir / name / "sample_outputs.json").read_text())
    action = torch.as_tensor(outputs["outputs"][0]["content"]["action"])
    action = denormalize_action(action, method="quantile", stats=norm_stats)
    actions.append(action)

# Visualize Input video.
input_videos = np.concat(input_videos, axis=0)
fps = fractal_records[0]["fps"]
print(f"Input Video, fps={fps}")
media.show_video(input_videos, fps=fps, width=512)

actions = torch.concat(actions, axis=0)
poses_rel = actions[..., :9].numpy()  # [T-1, 9] = [translation(3), rot6d(6)]

poses_abs = pose_rel_to_abs(
    poses_rel,
    rotation_format="rot6d",
    pose_convention="backward_framewise",
)  # [T, 4, 4] camera-to-world

visualize_pose(poses_abs, title=f"Fractal (Google RT-1) : predicted end-effector trajectory ({len(poses_abs)} frames)", show=True)
```
