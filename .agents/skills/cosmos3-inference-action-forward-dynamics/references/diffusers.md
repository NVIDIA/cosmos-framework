<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Action-Conditioned Forward Dynamics with Diffusers

Forward dynamics rolls out future video from a start frame plus an action trajectory, running `Cosmos3OmniPipeline` with every action input grouped into a `CosmosActionCondition`.

It runs three AV ego trajectories, three camera-pose trajectories, an autoregressive multiview DROID rollout, an autoregressive UMI rollout, and a bimanual hand-pose chunk. For inverse dynamics, see [`run_id_with_diffusers.ipynb`](../../cosmos3-inference-action-inverse-dynamics/references/diffusers.md).

> **Guardrails:** Disabled by default in this notebook. Set `COSMOS3_DIFFUSERS_GUARDRAILS=true` before running the helper cell to enable them.

FP8 checkpoints: set `COSMOS3_DIFFUSERS_PRECISION=fp8` before loading. Nano and Super use the dedicated Hub repos `nvidia/Cosmos3-Nano-FP8` and `nvidia/Cosmos3-Super-FP8`. Use-case cells stay the same.

> **Skill note (not in the source):** this file is the cookbook notebook `run_fd_with_diffusers.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/generator/action/run_fd_with_diffusers.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

<!-- rumdl-disable MD034 -->
<!-- Skill note (not in the source): the directive above keeps the notebook's bare URL in this section's "uv version" note as written; the rule is re-enabled right after the section. -->

## 1. Prerequisites

Use a Linux machine with NVIDIA GPU access, model access on Hugging Face, and either `uvx hf@latest auth login` or `HF_TOKEN` set.

Trajectories and start frames come from [`assets/`](../../cosmos3-inference-action/assets).

The DROID and hand-pose examples read the checked-in LeRobot episodes with the readers from the Cosmos framework. Point `COSMOS3_REPO` at your framework checkout (it defaults to `packages/cosmos3` beside this repo) before running those sections.

Guardrails are disabled by default in this notebook. If you enable them, request access to the gated [nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail) HF repository and set `COSMOS3_DIFFUSERS_GUARDRAILS=true` before running the helper cell.

> **Headless servers:** if you see an error like `libxcb.so.1: cannot open shared object file` (a missing system graphics library) when importing or running the pipeline, install the required system libraries:
>
> ```bash
> apt-get install -y libxcb1 libgl1 libglib2.0-0
> ```

Skill note (not in the source): run it before the first import — `Cosmos3OmniPipeline` imports `cosmos_guardrail` -> `cv2` even with guardrails disabled — and on a fresh container prefix `apt-get update &&` (empty package lists; eval 2026-11-08): `apt-get update && apt-get install -y libxcb1 libgl1 libglib2.0-0`.

> **uv version:** use `uv 0.11.29`, matching the version pinned by Cosmos Framework. Install it from https://astral.sh/uv/0.11.29/install.sh or update an existing self-managed installation with `uv self update 0.11.29`.

Skill note (not in the source): `uv self update` works only for a standalone-installer uv (a package-managed uv exits 2); the § 3 block below carries a skill-added first line that installs 0.11.29 beside any other uv (eval 2026-11-08).

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
uv --version 2>/dev/null | grep -q '^uv 0\.11\.29' || { curl -LsSf https://astral.sh/uv/0.11.29/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }  # skill note (not in the source): the notebook's uv 0.11.29; inside the checkout any other uv stops at the pyproject pin
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

## 6. Preview Available Inputs

```python
import json
from pathlib import Path
from IPython.display import Image, display

assets_dir = COSMOS3_ACTION_ROOT / "assets"

print("action trajectories:")
for action_path in sorted((assets_dir / "actions").glob("*.json")):
    steps = json.loads(action_path.read_text())
    print(f"  {action_path.name}: {len(steps)} steps x {len(steps[0])}D")
print()

print("prompts:")
for prompt_path in sorted((assets_dir / "prompts").glob("*.txt")):
    print(f"  {prompt_path.name}: {prompt_path.read_text().strip()}")
print()

print("start frames:")
for image_name in ["av_0.jpg", "lighthouse_720.png", "solar_720.png", "mountain_720.png"]:
    image_path = assets_dir / "images" / image_name
    print(f"  {image_name}")
    display(Image(filename=str(image_path), width=420))
```

## 7. Define Action Cases, Runner, and Viewer Helpers

Every action input is grouped into a `CosmosActionCondition`; `height`, `width`, and `num_frames` stay unset because the pipeline derives the frame count from `chunk_size + 1` and the conditioning canvas from `resolution_tier`. Forward dynamics conditions on `action.image` and needs `raw_actions` as a `[T, D]` tensor whose width matches the domain.

Action prompts are plain task descriptions rather than upsampled JSON: the pipeline builds the structured caption the model was trained on from the prompt, `domain_name`, and `view_point`, which is also why `use_system_prompt=False`.

These examples run on Cosmos3-Nano; Cosmos3-Super also ships `action_gen=True`, so passing `model="Cosmos3-Super"` works unchanged.

```python
import base64
import gc
import html
import json
import os
import sys
import time
from pathlib import Path
from IPython.display import HTML, Image, display

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
from diffusers.utils import export_to_video, load_image, load_video


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

    frames = torch.stack([torch.from_numpy(loaded_frames[int(index)]) for index in frame_indexes])
    return frames.permute(0, 3, 1, 2).float() / 255.0


# Patch before the DROID and HumanHandPose reader modules are imported. This lets both readers
# bind the PyAV implementation without importing TorchCodec.
lerobot_video_utils.decode_video_frames = decode_video_frames_av

MODEL_IDS = {
    "Cosmos3-Nano": "nvidia/Cosmos3-Nano",
    "Cosmos3-Super": "nvidia/Cosmos3-Super",
}

# Guardrails are off by default in this notebook. Set COSMOS3_DIFFUSERS_GUARDRAILS=true to enable the safety checker.
GUARDRAILS = os.environ.get("COSMOS3_DIFFUSERS_GUARDRAILS", "false").strip().lower() not in {"0", "false", "no", "off"}

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
    "av_forward": {
        "mode": "forward_dynamics",
        "domain_name": "av",
        "chunk_size": 60,
        "resolution_tier": 480,
        "view_point": "ego_view",
        "fps": 10,
        "prompt": AV_PROMPT,
        "image": "assets/images/av_0.jpg",
        "action": "assets/actions/av_traj_forward.json",
    },
    "av_left": {
        "mode": "forward_dynamics",
        "domain_name": "av",
        "chunk_size": 60,
        "resolution_tier": 480,
        "view_point": "ego_view",
        "fps": 10,
        "prompt": AV_PROMPT,
        "image": "assets/images/av_0.jpg",
        "action": "assets/actions/av_traj_left.json",
    },
    "av_right": {
        "mode": "forward_dynamics",
        "domain_name": "av",
        "chunk_size": 60,
        "resolution_tier": 480,
        "view_point": "ego_view",
        "fps": 10,
        "prompt": AV_PROMPT,
        "image": "assets/images/av_0.jpg",
        "action": "assets/actions/av_traj_right.json",
    },
    "camera_lighthouse": {
        "mode": "forward_dynamics",
        "domain_name": "camera_pose",
        "chunk_size": 60,
        "resolution_tier": 480,
        "view_point": "ego_view",
        "fps": 30,
        "prompt_path": "assets/prompts/lighthouse.txt",
        "image": "assets/images/lighthouse_720.png",
        "action": "assets/actions/camera_action.json",
    },
    "camera_solar": {
        "mode": "forward_dynamics",
        "domain_name": "camera_pose",
        "chunk_size": 60,
        "resolution_tier": 480,
        "view_point": "ego_view",
        "fps": 30,
        "prompt_path": "assets/prompts/solar.txt",
        "image": "assets/images/solar_720.png",
        "action": "assets/actions/camera_action.json",
    },
    "camera_mountain": {
        "mode": "forward_dynamics",
        "domain_name": "camera_pose",
        "chunk_size": 60,
        "resolution_tier": 480,
        "view_point": "ego_view",
        "fps": 30,
        "prompt_path": "assets/prompts/mountain.txt",
        "image": "assets/images/mountain_720.png",
        "action": "assets/actions/camera_action.json",
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


def generate_chunk(
    *,
    prompt: str,
    mode: str,
    domain_name: str,
    chunk_size: int,
    resolution_tier: int,
    view_point: str,
    fps: int,
    raw_actions: "torch.Tensor | None" = None,
    image=None,
    video=None,
    model: str = "Cosmos3-Nano",
    seed: int = 0,
) -> tuple[list, "list | None"]:
    """Run one pipeline call and return its frames and any predicted actions."""
    pipe = get_pipe(model)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=FIXED_SAMPLING["shift"], use_karras_sigmas=False
    )
    generator = torch.Generator(device="cuda").manual_seed(seed)
    t0 = time.time()
    result = pipe(
        prompt=prompt,
        action=CosmosActionCondition(
            mode=mode,
            chunk_size=chunk_size,
            domain_name=domain_name,
            resolution_tier=resolution_tier,
            raw_actions=raw_actions,
            image=image,
            video=video,
            view_point=view_point,
        ),
        fps=fps,
        num_inference_steps=FIXED_SAMPLING["num_steps"],
        guidance_scale=FIXED_SAMPLING["guidance"],
        use_system_prompt=False,
        generator=generator,
    )
    print(f"generated in {time.time() - t0:.1f}s")
    return result.video, result.action


def run_action(case: str, *, model: str = "Cosmos3-Nano") -> Path:
    """Run one action case. Writes `<case>.mp4` and, for inverse dynamics, `<case>_action.json`."""
    spec = ACTION_SETS[case]
    output_dir = case_output_dir(case)
    output_path = output_dir / f"{case}.mp4"

    raw_actions = None
    if "action" in spec:
        raw_actions = torch.as_tensor(json.loads(asset_path(spec["action"]).read_text()), dtype=torch.float32)
    image = load_image(str(asset_path(spec["image"]))) if "image" in spec else None
    video = load_video(str(asset_path(spec["video"]))) if "video" in spec else None

    print(f"case:    {case} ({spec['mode']}, domain {spec['domain_name']}) with {model}")
    print(f"input:   {asset_path(spec.get('image') or spec['video']).relative_to(COSMOS_ROOT)}")
    if raw_actions is not None:
        print(f"actions: {tuple(raw_actions.shape)} from {asset_path(spec['action']).relative_to(COSMOS_ROOT)}")
    print(f"frames:  {spec['chunk_size'] + 1} at {spec['fps']} fps, resolution tier {spec['resolution_tier']}")
    print(f"output:  {output_path}")

    frames, actions = generate_chunk(
        prompt=case_prompt(spec),
        mode=spec["mode"],
        domain_name=spec["domain_name"],
        chunk_size=spec["chunk_size"],
        resolution_tier=spec["resolution_tier"],
        view_point=spec["view_point"],
        fps=spec["fps"],
        raw_actions=raw_actions,
        image=image,
        video=video,
        model=model,
        seed=FIXED_SAMPLING["seed"],
    )

    # macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
    export_to_video(frames, str(output_path), fps=spec["fps"], macro_block_size=1)
    print(f"wrote {output_path}")
    if actions is not None:
        action_path = output_dir / f"{case}_action.json"
        action_path.write_text(json.dumps(actions[0].tolist()) + "\n")
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


def view_action(case: str) -> None:
    """Show the start frame or input clip next to the generated video, plus any predicted actions."""
    spec = ACTION_SETS[case]
    output_dir = case_output_dir(case)
    if "image" in spec:
        print(f"start frame: {asset_path(spec['image']).relative_to(COSMOS_ROOT)}")
        display(Image(filename=str(asset_path(spec["image"])), width=420))
    else:
        print(f"input video: {asset_path(spec['video']).relative_to(COSMOS_ROOT)}")
        display_video(asset_path(spec["video"]), width=420)

    output_path = output_dir / f"{case}.mp4"
    if not output_path.is_file():
        print(f"No generated video at {output_path}; run the case first.")
        return
    print(f"generated: {output_path} ({output_path.stat().st_size // 1024} KB)")
    display_video(output_path)

    action_path = output_dir / f"{case}_action.json"
    if action_path.is_file():
        steps = json.loads(action_path.read_text())
        print(f"predicted actions: {len(steps)} steps x {len(steps[0])}D -> {action_path}")
        print("first step:", [round(value, 4) for value in steps[0]])


def run_rollout(case: str, chunks: list[dict], first_frame, *, fps: int, model: str = "Cosmos3-Nano") -> Path:
    """Chain forward-dynamics chunks, feeding each chunk's last frame into the next one.

    Every chunk regenerates its conditioning frame as frame 0, so only frames 1.. are kept.
    """
    output_path = case_output_dir(case) / f"{case}.mp4"
    frames: list = []
    conditioning = first_frame
    for chunk_index, chunk in enumerate(chunks):
        print(f"chunk {chunk_index + 1}/{len(chunks)}: {tuple(chunk['raw_actions'].shape)} actions")
        chunk_frames, _ = generate_chunk(
            prompt=chunk["prompt"],
            mode="forward_dynamics",
            domain_name=chunk["domain_name"],
            chunk_size=chunk["raw_actions"].shape[0],
            resolution_tier=chunk["resolution_tier"],
            view_point=chunk["view_point"],
            fps=fps,
            raw_actions=chunk["raw_actions"],
            image=conditioning,
            model=model,
            seed=chunk_index,
        )
        frames.extend(chunk_frames[1:])
        conditioning = chunk_frames[-1]
    export_to_video(frames, str(output_path), fps=fps, macro_block_size=1)
    print(f"wrote {output_path} ({len(frames)} frames)")
    return output_path
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

Run each case top-to-bottom: generate, then view the start frame alongside the result.

## Forward Dynamics: AV Driving Forward

Roll out the `av_traj_forward` ego trajectory from the AV start frame.

### Run

```python
av_forward_output = run_action("av_forward")
```

### View Results

```python
view_action("av_forward")
```

## Forward Dynamics: AV Turning Left

The same start frame and settings with the `av_traj_left` trajectory.

### Run

```python
av_left_output = run_action("av_left")
```

### View Results

```python
view_action("av_left")
```

## Forward Dynamics: AV Turning Right

The same start frame and settings with the `av_traj_right` trajectory.

### Run

```python
av_right_output = run_action("av_right")
```

### View Results

```python
view_action("av_right")
```

## Forward Dynamics: Camera Pose, Lighthouse

Camera-pose control moves the virtual camera through a still image, driven by the same pose representation as the AV cases.

### Run

```python
camera_lighthouse_output = run_action("camera_lighthouse")
```

### View Results

```python
view_action("camera_lighthouse")
```

## Forward Dynamics: Camera Pose, Solar Terrace

The same camera trajectory applied to a different scene.

### Run

```python
camera_solar_output = run_action("camera_solar")
```

### View Results

```python
view_action("camera_solar")
```

## Forward Dynamics: Camera Pose, Mountain

The same camera trajectory applied to a third scene.

### Run

```python
camera_mountain_output = run_action("camera_mountain")
```

### View Results

```python
view_action("camera_mountain")
```

## DROID: Autoregressive Multiview Rollout

DROID conditions on a single frame that tiles three camera views, and the reader composes that canvas along with the 10D action chunks. Chaining five chunks rolls the manipulation forward well past a single chunk's horizon.

The reader resolves its feature layout from the name of the directory it is given. By default, the setup cell exposes the bundled sample through the matching `droid_plus_lerobot_640x360_20260412/success` layout. To use another dataset, set `COSMOS3_DROID_ROOT` to its version-named root directory.

```python
import os
import sys
from pathlib import Path

from PIL import Image as PILImage

# The notebook kernel may differ from the framework venv, so put the repo on the
# path before importing `cosmos_framework`.
if str(COSMOS3_REPO) not in sys.path:
    sys.path.insert(0, str(COSMOS3_REPO))
from cosmos_framework.data.generator.action.datasets import DROIDLeRobotDataset

droid_case = "droid_forward"
droid_chunk_length = 16
droid_num_chunks = 5

# `DROIDLeRobotDataset` selects its feature configuration from the root directory
# name. The bundled sample comes from the 640x360 release but is stored under a
# friendly asset name, so expose it through the release layout expected by the loader.
droid_root_override = os.environ.get("COSMOS3_DROID_ROOT")
if droid_root_override:
    droid_root = str(Path(droid_root_override).expanduser().resolve())
else:
    bundled_droid_root = asset_path("assets/droid_lerobot_example")
    versioned_droid_root = (
        Path(os.environ["COSMOS3_ACTION_OUTPUT_ROOT"])
        / "datasets"
        / "droid_plus_lerobot_640x360_20260412"
    )
    success_root = versioned_droid_root / "success"
    versioned_droid_root.mkdir(parents=True, exist_ok=True)
    if success_root.is_symlink():
        if success_root.resolve() != bundled_droid_root:
            raise RuntimeError(f"{success_root} points to {success_root.resolve()}, expected {bundled_droid_root}")
    elif success_root.exists():
        if success_root.resolve() != bundled_droid_root:
            raise RuntimeError(f"{success_root} already exists and is not the bundled DROID sample")
    else:
        success_root.symlink_to(bundled_droid_root, target_is_directory=True)
    droid_root = str(versioned_droid_root)

# Cosmos3-Nano was trained on quantile-normalized DROID actions.
droid_dataset = DROIDLeRobotDataset(
    root=droid_root,
    chunk_length=droid_chunk_length,
    use_success_only=True,
    action_normalization="quantile_rot",
    apply_forward_clamp=True,
)

droid_chunks = []
for chunk_index in range(droid_num_chunks):
    sample = droid_dataset[chunk_index * droid_chunk_length]
    droid_chunks.append(
        {
            "raw_actions": sample["action"].float(),
            "prompt": sample["ai_caption"],
            "domain_name": "droid_lerobot",
            "resolution_tier": 480,
            "view_point": sample["viewpoint"],
        }
    )

droid_fps = int(droid_dataset[0]["conditioning_fps"])
# `video` is [C, T, H, W] uint8; frame 0 is the ground-truth conditioning canvas.
droid_first_frame = PILImage.fromarray(droid_dataset[0]["video"][:, 0].permute(1, 2, 0).cpu().numpy())

print(f"root:     {droid_root}")
print(f"chunks:   {len(droid_chunks)} x {droid_chunk_length} actions of width {droid_chunks[0]['raw_actions'].shape[1]}")
print(f"fps:      {droid_fps}, view point {droid_chunks[0]['view_point']}")
print(f"prompt:   {droid_chunks[0]['prompt']}")
print(f"conditioning frame: {droid_first_frame.size}")
display(droid_first_frame)
```

### Run

```python
droid_output = run_rollout(droid_case, droid_chunks, droid_first_frame, fps=droid_fps)
```

### View Results

```python
print(f"conditioning frame {droid_first_frame.size}")
display(droid_first_frame)
print(f"generated: {droid_output} ({droid_output.stat().st_size // 1024} KB)")
display_video(droid_output)
```

## UMI: Autoregressive Rollout

The UMI actions ship as a plain trajectory of 10D rows, which splits evenly into chunks that chain the same way as DROID.

```python
umi_case = "umi_forward"
umi_chunk_length = 16
umi_raw_action_dim = 10
umi_fps = 20
umi_prompt = "mouse arrangement"

umi_actions = json.loads(asset_path("assets/actions/umi.json").read_text())
assert len(umi_actions) % umi_chunk_length == 0, (
    f"expected the action count to be divisible by {umi_chunk_length}, got {len(umi_actions)}"
)
assert all(len(row) == umi_raw_action_dim for row in umi_actions), "UMI action rows must be 10D"

umi_chunks = [
    {
        "raw_actions": torch.as_tensor(
            umi_actions[start : start + umi_chunk_length], dtype=torch.float32
        ),
        "prompt": umi_prompt,
        "domain_name": "umi",
        "resolution_tier": 256,
        "view_point": "ego_view",
    }
    for start in range(0, len(umi_actions), umi_chunk_length)
]
umi_first_frame = load_image(str(asset_path("assets/images/umi.png")))

print(f"chunks: {len(umi_chunks)} x {umi_chunk_length} actions of width {umi_raw_action_dim}")
print(f"fps:    {umi_fps}, prompt {umi_prompt!r}")
print(f"conditioning frame: {umi_first_frame.size}")
display(umi_first_frame)
```

### Run

```python
umi_output = run_rollout(umi_case, umi_chunks, umi_first_frame, fps=umi_fps)
```

### View Results

```python
print(f"conditioning frame {umi_first_frame.size}")
display(umi_first_frame)
print(f"generated: {umi_output} ({umi_output.stat().st_size // 1024} KB)")
display_video(umi_output)
```

## Human Hand Pose

Egocentric two-hand motion uses the widest action layout in the cookbook: a 9D ego pose plus, for each hand, a 9D wrist pose and five 3D fingertip positions. The reader assembles that from the pose annotations alongside the ego clip.

```python
from cosmos_framework.data.generator.action.datasets import HumanHandPoseLeRobotDataset

hand_pose_case = "hand_pose_forward"
hand_pose_chunk_length = 16
hand_pose_dataset = HumanHandPoseLeRobotDataset(
    root=str(asset_path("assets/human_hand_pose_lerobot_example"))
)
hand_pose_sample = hand_pose_dataset[0]
hand_pose_actions = hand_pose_sample["action"].float()
hand_pose_fps = int(hand_pose_sample["conditioning_fps"])
hand_pose_first_frame = PILImage.fromarray(
    hand_pose_sample["video"][:, 0].permute(1, 2, 0).cpu().numpy()
)

print(f"actions: {tuple(hand_pose_actions.shape)}")
print(f"fps:     {hand_pose_fps}, view point {hand_pose_sample['viewpoint']}")
print(f"prompt:  {hand_pose_sample['ai_caption']}")
print(f"conditioning frame: {hand_pose_first_frame.size}")
display(hand_pose_first_frame)
```

### Run

```python
hand_pose_output = case_output_dir(hand_pose_case) / f"{hand_pose_case}.mp4"
hand_pose_frames, _ = generate_chunk(
    prompt=hand_pose_sample["ai_caption"],
    mode="forward_dynamics",
    domain_name="hand_pose",
    chunk_size=hand_pose_chunk_length,
    resolution_tier=480,
    view_point=hand_pose_sample["viewpoint"],
    fps=hand_pose_fps,
    raw_actions=hand_pose_actions,
    image=hand_pose_first_frame,
)
export_to_video(hand_pose_frames, str(hand_pose_output), fps=hand_pose_fps, macro_block_size=1)
print(f"wrote {hand_pose_output}")
```

### View Results

```python
print(f"conditioning frame {hand_pose_first_frame.size}")
display(hand_pose_first_frame)
print(f"generated: {hand_pose_output} ({hand_pose_output.stat().st_size // 1024} KB)")
display_video(hand_pose_output)
```
