<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Action Forward Dynamics with Cosmos Framework

This notebook runs Cosmos3 **action forward-dynamics** inference through the native Cosmos Framework PyTorch entrypoint:

```bash
python -m cosmos_framework.scripts.inference
```

Forward dynamics predicts future visual observations from an initial image and an action trajectory. This notebook is written as a first-run cookbook: clone or locate the framework source, install dependencies from scratch, verify the GPU environment, build AV, robotics, UMI, and human hand-pose input specs, run inference, and visualize generated videos.

Tested path from the audit:

- Framework checkout: `packages/cosmos3`
- Install command: `uv sync --all-extras --group=cu130-train`
- Backend: Cosmos Framework / `cosmos_framework.scripts.inference`
- Models: `Cosmos3-Nano` (default), `Cosmos3-Edge`, and `Cosmos3-Super`

Every inference cell below reads the checkpoint from `COSMOS3_CHECKPOINT_PATH` (set in step 2, default `Cosmos3-Nano`). To run the same forward-dynamics examples on Edge instead, set `COSMOS3_CHECKPOINT_PATH=Cosmos3-Edge` or `Cosmos3-Super` before the configuration cell — nothing else changes.

> **Skill note (not in the source):** this file is the cookbook notebook `run_fd_with_cosmos_framework.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/generator/action/run_fd_with_cosmos_framework.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

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

> Skill note (not in the source): the first line of the block below is a skill addition — the notebook only checks that `uv` exists, but `uv sync` aborts unless `uv --version` is 0.11.29 (the checkout's `required-version`); the guard installs 0.11.29 beside a different host uv (`uv self update` works only for a standalone-installer uv; eval 2026-11-08, 3 of 4 agents). Every other line is the notebook's.

```bash
uv --version 2>/dev/null | grep -q '^uv 0\.11\.29' || { curl -LsSf https://astral.sh/uv/0.11.29/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }  # skill note (not in the source): the checkout pins uv ==0.11.29; `uv self update` works only for a standalone-installer uv
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
from PIL import Image
import imageio_ffmpeg

# The notebook kernel may differ from the framework venv, so put the repo on the
# path before importing `cosmos_framework`.
if str(COSMOS3_REPO) not in sys.path:
    sys.path.insert(0, str(COSMOS3_REPO))

from cosmos_framework.data.generator.action.action_normalization import denormalize_action
from cosmos_framework.data.generator.action.utils.pose_utils import pose_abs_to_rel, pose_rel_to_abs
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
        # LeRobot-backed datasets (e.g. DROID) name the embodiment with a class
        # attribute instead of the `domain_name` / `viewpoint` properties, and carry
        # the viewpoint on each sample.
        domain_name = getattr(dataset, "domain_name", None) or dataset.EMBODIMENT_TYPE
        view_point = getattr(dataset, "viewpoint", None) or data_sample["viewpoint"]
        chunk_name = f"{domain_name}_id_chunk_{chunk_idx:02d}"

        action = data_sample["action"]
        action_path = COSMOS3_INPUT_DIR / f"{chunk_name}.json"
        action_path.write_text(json.dumps(action.cpu().tolist()))

        vision_path = COSMOS3_INPUT_DIR / f"{domain_name}_autoregressive_input_chunk_{chunk_idx:02d}.png"
        if chunk_idx == 0:
            first_frame = data_sample["video"][:, 0].permute(1, 2, 0).cpu().numpy()
            Image.fromarray(first_frame).save(vision_path)

        records.append(
            {
                "action_chunk_size": chunk_length,
                "action_path": str(action_path),
                "domain_name": domain_name,
                "fps": int(data_sample["conditioning_fps"]),
                "image_size": 480,
                "view_point": view_point,
                "model_mode": "forward_dynamics",
                "name": chunk_name,
                "prompt": data_sample["ai_caption"],
                "seed": 0,
                "vision_path": str(vision_path),
            }
        )
    return records
```

## 6. AV Forward Dynamics

In this example, we show how to provide a set of ego poses of an autonomous vehicle and an image to generate driving videos using Cosmos3-Nano.

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
) # [T-1, 9], translation(3), rot6d(6), framewise relative transformation

with open("custom_traj.json", "w") as f:
    json.dump(poses_rel, f)
```

```python
# `resolve_input` and the COSMOS3_* paths come from the configuration cell.

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

Runs `Cosmos3-Nano` on every line of the AV spec. Each run writes its video to:

```text
<output>/action_forward_dynamics_av_custom/<name>/vision.mp4
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
echo "checkpoint path: $COSMOS3_CHECKPOINT_PATH"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_NANO_TEXT_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_AV_FD_INPUT" \
  -o "$COSMOS3_AV_FD_OUTPUT" \
  --checkpoint-path "$COSMOS3_CHECKPOINT_PATH" \
  --video-save-quality 8 \
  --image_size 480 \
  --seed 0 \
  --benchmark
```

### Visualize AV Generated Videos

```python
for record in av_records:
    name = record["name"]
    src = av_fd_output_dir / name / "vision.mp4"
    assert src.exists(), f"missing: {src}"
    preview = media.read_video(src)
    fps = record["fps"]
    print(f"AV: name = {name}, fps = {fps}, num_frames = {preview.shape[0]}")
    media.show_video(preview, width=512, fps=fps)
```

## 7. Camera Forward Dynamics

In this example, we show how to generate videos with Cosmos3-Nano by providing an input image along with a set of ego poses from a moving camera.

### Create the Camera Forward-Dynamics Input Spec

Camera forward-dynamics inference is driven by a JSONL spec, one line per run. Each line shares the same start frame (`vision_path`) but uses a different ego trajectory (`action_path`), so we get one generated video per trajectory.

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
) # [T-1, 9], translation(3), rot6d(6), framewise relative transformation

with open("custom_traj.json", "w") as f:
    json.dump(poses_rel, f)
```

```python
# `resolve_input` and the COSMOS3_* paths come from the configuration cell.

# Local Camera inputs, relative to the cookbook root.
camera_input_images = {
    "lighthouse": "cookbooks/cosmos3/generator/action/assets/images/lighthouse_720.png",
    "solar": "cookbooks/cosmos3/generator/action/assets/images/solar_720.png",
    "mountain": "cookbooks/cosmos3/generator/action/assets/images/mountain_720.png",
}

camera_prompts = {
    "lighthouse": "cookbooks/cosmos3/generator/action/assets/prompts/lighthouse.txt",
    "solar": "cookbooks/cosmos3/generator/action/assets/prompts/solar.txt",
    "mountain": "cookbooks/cosmos3/generator/action/assets/prompts/mountain.txt",
}

camera_input_action = "cookbooks/cosmos3/generator/action/assets/actions/camera_action.json"

camera_input_action_path = resolve_input(camera_input_action)
camera_records = [
    {
        "action_chunk_size": 60,
        "action_path": camera_input_action_path,
        "domain_name": "camera_pose",
        "fps": 30,
        "image_size": 480,
        "view_point": "ego_view",
        "model_mode": "forward_dynamics",
        "name": name,
        "prompt_path": resolve_input(camera_prompts[name]),
        "seed": 0,
        "vision_path": resolve_input(vision_rel),
    }
    for name, vision_rel in camera_input_images.items()
]

COSMOS3_INPUT_DIR.mkdir(parents=True, exist_ok=True)
camera_fd_input_path = COSMOS3_INPUT_DIR / "action_forward_dynamics_camera_custom.jsonl"
camera_fd_input_path.write_text("".join(json.dumps(r) + "\n" for r in camera_records))
camera_fd_output_dir = COSMOS3_OUTPUT_ROOT / "action_forward_dynamics_camera_custom"

os.environ["COSMOS3_CAMERA_FD_INPUT"] = str(camera_fd_input_path)
os.environ["COSMOS3_CAMERA_FD_OUTPUT"] = str(camera_fd_output_dir)

print("Wrote Camera spec:", camera_fd_input_path)
print("Camera runs:", list(camera_input_images))
print(camera_fd_input_path.read_text())
```

### Visualize Camera Input Trajectories

Before generating any video, plot the input ego trajectory as a 3D camera path with frustums and a top-down bird's-eye view.

```python
record = camera_records[0]
with open(record["action_path"]) as f:
    poses_rel = np.array(json.load(f))

# Camera action convention: rot6d rotation, backward_framewise, translation_scale = 1.0.
poses_abs = pose_rel_to_abs(
    poses_rel,
    rotation_format="rot6d",
    pose_convention="backward_framewise",
    translation_scale=1.0,
)
print(name, poses_rel.shape, poses_abs.shape)
visualize_pose(poses_abs, title=f"Camera: camera trajectory + frustums ({len(poses_abs)} frames)", show=True)
```

### Run Camera Forward-Dynamics Inference

Runs `Cosmos3-Nano` on every line of the Camera spec. Each run writes its video to:

```text
<output>/action_forward_dynamics_camera_custom/<name>/vision.mp4
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
echo "checkpoint path: $COSMOS3_CHECKPOINT_PATH"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_NANO_TEXT_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_CAMERA_FD_INPUT" \
  -o "$COSMOS3_CAMERA_FD_OUTPUT" \
  --checkpoint-path "$COSMOS3_CHECKPOINT_PATH" \
  --video-save-quality 8 \
  --image_size 480 \
  --seed 0 \
  --benchmark
```

### Visualize Camera Generated Videos

```python
for record in camera_records:
    name = record["name"]
    src = camera_fd_output_dir / name / "vision.mp4"
    assert src.exists(), f"missing: {src}"
    preview = media.read_video(src)
    fps = record["fps"]
    print(f"Camera: name = {name}, fps = {fps}, num_frames = {preview.shape[0]}")
    media.show_video(preview, width=512, fps=fps)
```

## 8. DROID Forward Dynamics

In this example, we show how to start from a LeRobot dataset of DROID and run **multiview** generation for robotics manipulation **autoregressively**.

### Create the DROID Autoregressive Forward-Dynamics Plan

DROID forward-dynamics runs autoregressively over five contiguous 16-action DROID chunks. This cell writes the GT first conditioning image for chunk 0 and one action JSON per chunk. Later chunks receive their conditioning image from the previous chunk's generated last frame during the inference loop. Raw DROID actions are normalized into the model's `quantile_rot` action space before the JSON files are written.

```python
from cosmos_framework.data.generator.action.datasets import DROIDMergedLeRobotDataset

num_chunks = 5
chunk_length = 16

droid_dataset_root = resolve_input("cookbooks/cosmos3/generator/action/assets/droid_lerobot_example")
# Cosmos3-Nano was trained on quantile-normalized DROID actions.
droid_dataset = DROIDMergedLeRobotDataset(
    root=droid_dataset_root,
    chunk_length=chunk_length,
    action_normalization="quantile_rot",
    apply_forward_clamp=True,
)
droid_records = create_record_from_dataset(droid_dataset, num_chunks, chunk_length)

domain_name = droid_dataset.EMBODIMENT_TYPE
droid_fd_input_path = COSMOS3_INPUT_DIR / f"action_forward_dynamics_{domain_name}_custom.jsonl"
droid_fd_input_path.write_text("".join(json.dumps(r) + "\n" for r in droid_records))
droid_fd_output_dir = COSMOS3_OUTPUT_ROOT / f"action_forward_dynamics_{domain_name}_custom"

print("Loaded DROID samples from:", droid_dataset_root)
print("Wrote droid autoregressive plan:", droid_fd_input_path)
print(droid_fd_input_path.read_text())
```

### Run DROID Autoregressive Forward-Dynamics Inference

Runs `Cosmos3-Nano` once per DROID chunk. Chunk 0 uses the DROID GT first frame. After each chunk finishes, the cell extracts that chunk's last generated frame and uses it as the conditioning image for the next chunk. Guardrails are disabled for this DROID run.

```python
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
droid_actual_records = []
current_vision_path = Path(droid_records[0]["vision_path"])
assert current_vision_path.exists(), f"missing initial conditioning image: {current_vision_path}"

for chunk_idx, base_record in enumerate(droid_records):
    record = dict(base_record)
    record["vision_path"] = str(current_vision_path)
    droid_records[chunk_idx]["vision_path"] = str(current_vision_path)

    chunk_input_path = COSMOS3_INPUT_DIR / f"action_forward_dynamics_droid_chunk_{chunk_idx:02d}.jsonl"
    chunk_input_path.write_text(json.dumps(record) + "\n")
    droid_actual_records.append(record)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env["MASTER_ADDR"] = env.get("COSMOS3_MASTER_ADDR", "127.0.0.1")
    env["MASTER_PORT"] = env.get("COSMOS3_NANO_TEXT_MASTER_PORT", "29500")
    env["RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env["LOCAL_RANK"] = "0"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"running chunk {chunk_idx}: {record['name']}")
    print("conditioning image:", current_vision_path)
    subprocess.run(
        [
            str(COSMOS3_REPO / ".venv" / "bin" / "python"),
            "-m",
            "cosmos_framework.scripts.inference",
            "--parallelism-preset=latency",
            "--no-guardrails",
            "-i",
            str(chunk_input_path),
            "-o",
            str(droid_fd_output_dir),
            "--checkpoint-path",
            os.environ["COSMOS3_CHECKPOINT_PATH"],
            "--video-save-quality",
            "8",
            "--image_size",
            "480",
            "--seed",
            str(record["seed"]),
            "--benchmark",
        ],
        cwd=str(COSMOS3_REPO),
        env=env,
        check=True,
    )

    output_video = droid_fd_output_dir / record["name"] / "vision.mp4"
    assert output_video.exists(), f"missing generated video: {output_video}"

    if chunk_idx + 1 < len(droid_records):
        next_vision_path = COSMOS3_INPUT_DIR / f"droid_droid_autoregressive_input_chunk_{chunk_idx + 1:02d}.png"
        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(output_video),
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

droid_fd_input_path.write_text("".join(json.dumps(r) + "\n" for r in droid_actual_records))
print("Wrote autoregressive run spec:", droid_fd_input_path)
print("Completed chunks:", [record["name"] for record in droid_actual_records])
```

### Stitch and Visualize DROID Generated Video

Each autoregressive video chunk includes its conditioning frame at frame 0. This cell drops that first frame from every chunk and concatenates the remaining 16 generated frames per chunk into one 80-frame video for visualization.

```python
stitched_preview = []
for record in droid_records:
    name = record["name"]
    src = droid_fd_output_dir / name / "vision.mp4"
    assert src.exists(), f"missing: {src}"
    preview = media.read_video(src)
    stitched_preview.append(preview[1:])  # Drops the first drops that first frame.

fps = droid_records[0]["fps"]
stitched_preview = np.concat(stitched_preview, axis=0)
print(f"DROID Generated Video: fps = {fps}, num_frames = {stitched_preview.shape[0]}")
media.show_video(stitched_preview, width=512, fps=fps)
```

## 9. UMI Forward Dynamics

This example runs UMI forward dynamics autoregressively over all 16-action chunks in `assets/actions/umi.json`. The checked-in action file stores the raw UMI 10D action representation, so the setup cell validates the row dimension, writes one action JSON per chunk, and prepares a run plan.

### Create the UMI Autoregressive Forward-Dynamics Plan

The UMI action file is stored as one JSON array with `16 * n` action rows. Cosmos Framework expects each forward-dynamics sample to contain one 16-action chunk whose action dimension matches `domain_name="umi"`, so this cell writes one derived 10D action JSON per chunk and creates a JSONL plan for all chunks.

```python
from cosmos_framework.data.generator.action.datasets import UMILeRobotDataset

num_chunks = 5
chunk_length = 16

umi_dataset_root = resolve_input("cookbooks/cosmos3/generator/action/assets/umi_lerobot_example")
umi_dataset = UMILeRobotDataset(root=umi_dataset_root,chunk_length=chunk_length)
umi_records = create_record_from_dataset(umi_dataset, num_chunks, chunk_length)

domain_name = umi_dataset.domain_name
umi_fd_input_path = COSMOS3_INPUT_DIR / f"action_forward_dynamics_{domain_name}_custom.jsonl"
umi_fd_input_path.write_text("".join(json.dumps(r) + "\n" for r in umi_records))
umi_fd_output_dir = COSMOS3_OUTPUT_ROOT / f"action_forward_dynamics_{domain_name}_custom"

print("Loaded UMI samples from:", umi_dataset_root)
print("Wrote UMI autoregressive plan:", umi_fd_input_path)
```

### Run UMI Autoregressive Forward-Dynamics Inference

Runs `Cosmos3-Nano` once per UMI action chunk. Chunk 0 uses the checked-in UMI conditioning image; after each chunk finishes, the cell extracts that chunk's last generated frame and uses it as the conditioning image for the next chunk.

```python
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
umi_actual_records = []
current_vision_path = Path(umi_records[0]["vision_path"])
assert current_vision_path.exists(), f"missing initial conditioning image: {current_vision_path}"

for chunk_idx, base_record in enumerate(umi_records):
    record = dict(base_record)
    record["vision_path"] = str(current_vision_path)
    umi_records[chunk_idx]["vision_path"] = str(current_vision_path)

    chunk_input_path = COSMOS3_INPUT_DIR / f"action_forward_dynamics_umi_chunk_{chunk_idx:02d}.jsonl"
    chunk_input_path.write_text(json.dumps(record) + "\n")
    umi_actual_records.append(record)

    umi_env = os.environ.copy()
    umi_env["CUDA_VISIBLE_DEVICES"] = umi_env.get("CUDA_VISIBLE_DEVICES", "0")
    umi_env["MASTER_ADDR"] = umi_env.get("COSMOS3_MASTER_ADDR", "127.0.0.1")
    umi_env["MASTER_PORT"] = free_local_port()
    umi_env["RANK"] = "0"
    umi_env["WORLD_SIZE"] = "1"
    umi_env["LOCAL_RANK"] = "0"
    umi_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"running UMI chunk {chunk_idx}: {record['name']}")
    print("conditioning image:", current_vision_path)
    subprocess.run(
        [
            str(COSMOS3_REPO / ".venv" / "bin" / "python"),
            "-m",
            "cosmos_framework.scripts.inference",
            "--parallelism-preset=latency",
            "--no-guardrails",
            "-i",
            str(chunk_input_path),
            "-o",
            str(umi_fd_output_dir),
            "--checkpoint-path",
            umi_env["COSMOS3_CHECKPOINT_PATH"],
            "--video-save-quality",
            "8",
            "--image_size",
            str(record["image_size"]),
            "--seed",
            str(record["seed"]),
            "--benchmark",
        ],
        cwd=str(COSMOS3_REPO),
        env=umi_env,
        check=True,
    )

    output_video = umi_fd_output_dir / record["name"] / "vision.mp4"
    assert output_video.exists(), f"missing generated UMI video: {output_video}"

    if chunk_idx + 1 < len(umi_records):
        next_vision_path = COSMOS3_INPUT_DIR / f"umi_autoregressive_input_chunk_{chunk_idx + 1:02d}.png"
        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(output_video),
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

### Stitch and Visualize UMI Generated Video

Each autoregressive video chunk includes its conditioning frame at frame 0. This cell drops that first frame from every chunk, concatenates the generated frames into one rollout video for visualization.

```python
stitched_preview = []
for record in umi_records:
    name = record["name"]
    src = umi_fd_output_dir / name / "vision.mp4"
    assert src.exists(), f"missing: {src}"
    preview = media.read_video(src)
    stitched_preview.append(preview[1:])  # Drops the first drops that first frame.

fps = umi_records[0]["fps"]
stitched_preview = np.concat(stitched_preview, axis=0)
print(f"UMI Generated Video: fps = {fps}, num_frames = {stitched_preview.shape[0]}")
media.show_video(stitched_preview, width=512, fps=fps)
```

## 9. Human Hand-Pose Forward Dynamics

This example uses one validation episode of a person assembling a wooden object with a screwdriver. The checked-in LeRobot asset contains synchronized ego video, camera pose, and bimanual 21-keypoint hand annotations. `HumanHandPoseLeRobotDataset` converts them into the released 57D Cosmos hand-pose representation.

### Create the Hand-Pose Forward-Dynamics Input Spec

The example samples the 30 FPS source episode at 15 FPS and prepares one 16-action chunk. The action layout is camera pose, right wrist and fingertips, then left wrist and fingertips.

```python
# `resolve_input` and the COSMOS3_* paths come from the configuration cell.
import json
import os
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

COSMOS3_INPUT_DIR.mkdir(parents=True, exist_ok=True)
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

os.environ["COSMOS3_HAND_POSE_FD_INPUT"] = str(hand_pose_fd_input_path)
os.environ["COSMOS3_HAND_POSE_FD_OUTPUT"] = str(hand_pose_fd_output_dir)

print("loaded hand-pose sample from:", hand_pose_dataset_root)
print("caption:", hand_pose_record["prompt"])
print("action shape:", tuple(hand_pose_sample["action"].shape))
print("wrote hand-pose spec:", hand_pose_fd_input_path)
print(hand_pose_fd_input_path.read_text())
```

### Run Hand-Pose Forward-Dynamics Inference

```python
import os
import subprocess

hand_pose_env = os.environ.copy()
hand_pose_env["CUDA_VISIBLE_DEVICES"] = hand_pose_env.get("CUDA_VISIBLE_DEVICES", "0")
hand_pose_env["MASTER_ADDR"] = hand_pose_env.get("COSMOS3_MASTER_ADDR", "127.0.0.1")
hand_pose_env["MASTER_PORT"] = hand_pose_env.get("COSMOS3_NANO_IMAGE_MASTER_PORT", "29501")
hand_pose_env["RANK"] = "0"
hand_pose_env["WORLD_SIZE"] = "1"
hand_pose_env["LOCAL_RANK"] = "0"
hand_pose_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

subprocess.run(
    [
        str(COSMOS3_REPO / ".venv" / "bin" / "python"),
        "-m",
        "cosmos_framework.scripts.inference",
        "--parallelism-preset=latency",
        "--no-guardrails",
        "-i",
        str(hand_pose_fd_input_path),
        "-o",
        str(hand_pose_fd_output_dir),
        "--checkpoint-path",
        os.environ["COSMOS3_CHECKPOINT_PATH"],
        "--video-save-quality",
        "8",
        "--image_size",
        "480",
        "--seed",
        str(hand_pose_record["seed"]),
        "--benchmark",
    ],
    cwd=str(COSMOS3_REPO),
    env=hand_pose_env,
    check=True,
)

hand_pose_generated_video = hand_pose_fd_output_dir / hand_pose_record["name"] / "vision.mp4"
assert hand_pose_generated_video.exists(), f"missing generated video: {hand_pose_generated_video}"
print("generated:", hand_pose_generated_video)
```

### Visualize the Hand-Pose Generated Video

```python
from IPython.display import Video, display

assert hand_pose_generated_video.exists(), f"missing: {hand_pose_generated_video}"
display(Video(str(hand_pose_generated_video), embed=True))
```
