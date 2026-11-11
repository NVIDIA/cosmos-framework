<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Reasoner inference with Cosmos Framework

This notebook runs Cosmos3 Reasoner inference through the Cosmos Framework inference entrypoint:

```bash
python -m cosmos_framework.scripts.inference
```

It is intentionally written as a first-run cookbook: clone or locate the framework source, install dependencies from scratch, create working Reasoner input JSON files, run Nano text/image examples, try several image-based capability prompts, and run the same text/image prompts on the Edge and Super models.

Tested path from the audit:

- Framework checkout: `packages/cosmos3`
- Install command: `uv sync --all-extras --group=cu130-train`
- Backend: Cosmos Framework / `cosmos_framework.scripts.inference`
- Models: `Cosmos3-Nano` (default), `Cosmos3-Edge`, and `Cosmos3-Super`. All run the same Reasoner entrypoint; only `--checkpoint-path` differs. `Cosmos3-Super` needs a large-memory GPU (or multiple GPUs).

> **Skill note (not in the source):** this file is the cookbook notebook `run_with_cosmos_framework.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/reasoner/run_with_cosmos_framework.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

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

4. Use a disk/cache location with enough free space. Nano downloads can use tens of GiB in the Hugging Face cache.

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
export HF_HOME=/path/to/large/huggingface/cache
export CUDA_VISIBLE_DEVICES=0
```

For SSH access, set `COSMOS3_GIT_URL=git@github.com:NVIDIA/cosmos-framework.git`.

> **Skill note (not in the source):** from this skill directory `find_repo_root()` raises `FileNotFoundError` (no `cookbooks/cosmos3` tree here); replace the assignments that call it with `Path.cwd()` and `Path.cwd() / "assets"` (SKILL.md Read this first 3).

```python
from pathlib import Path
import os
import socket

def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "cookbooks" / "cosmos3").is_dir():
            return path
    return start

def free_local_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])

COSMOS_ROOT = find_repo_root(Path.cwd().resolve())
COSMOS_REASONER_ASSETS = COSMOS_ROOT / "cookbooks" / "cosmos3" / "reasoner" / "assets"
COSMOS3_REPO = Path(os.environ.get("COSMOS3_REPO", COSMOS_ROOT / "packages" / "cosmos3")).resolve()
COSMOS3_GIT_URL = os.environ.get(
    "COSMOS3_GIT_URL",
    "https://github.com/NVIDIA/cosmos-framework.git",
)
COSMOS3_UV_GROUP = os.environ.get("COSMOS3_UV_GROUP", "cu130-train")
COSMOS3_OUTPUT_ROOT = Path(
    os.environ.get("COSMOS3_OUTPUT_ROOT", COSMOS3_REPO / "outputs" / "cookbooks" / "cosmos3" / "reasoner" / "nano")
).resolve()
COSMOS3_INPUT_DIR = COSMOS3_OUTPUT_ROOT / "inputs"

# Keep these available to bash cells. Override any of them before running this cell.
os.environ["COSMOS_ROOT"] = str(COSMOS_ROOT)
os.environ["COSMOS_REASONER_ASSETS"] = str(COSMOS_REASONER_ASSETS)
os.environ["COSMOS3_REPO"] = str(COSMOS3_REPO)
os.environ["COSMOS3_GIT_URL"] = COSMOS3_GIT_URL
os.environ["COSMOS3_UV_GROUP"] = COSMOS3_UV_GROUP
os.environ["COSMOS3_OUTPUT_ROOT"] = str(COSMOS3_OUTPUT_ROOT)
os.environ["COSMOS3_INPUT_DIR"] = str(COSMOS3_INPUT_DIR)
os.environ.setdefault("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv"))
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("COSMOS3_MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("COSMOS3_NANO_TEXT_MASTER_PORT", free_local_port())
os.environ.setdefault("COSMOS3_NANO_IMAGE_MASTER_PORT", free_local_port())
os.environ.setdefault("COSMOS3_CAPABILITY_MASTER_PORT", free_local_port())

print("cosmos root:", COSMOS_ROOT)
print("Reasoner assets:", COSMOS_REASONER_ASSETS)
print("Cosmos Framework path:", COSMOS3_REPO)
print("Framework git URL:", COSMOS3_GIT_URL)
print("uv dependency group:", COSMOS3_UV_GROUP)
print("output root:", COSMOS3_OUTPUT_ROOT)
print("UV_CACHE_DIR:", os.environ["UV_CACHE_DIR"])
print("HF_HOME:", os.environ["HF_HOME"])
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
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
- May print a uv cache hardlink warning if your cache and repo are on different filesystems; this is usually harmless.

```bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

cd "$COSMOS3_REPO"
uv sync --all-extras --group="$COSMOS3_UV_GROUP"
```

## 5. Verify GPU and Python Environment

The Cosmos Framework commands below use `CUDA_VISIBLE_DEVICES=0` by default. Adjust this if you want a different GPU.

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" .venv/bin/python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

## 6. Create Reasoner Input Files

The current shipped Reasoner examples fail without `enable_sound=false`. This cell writes patched Nano smoke-test inputs and image-based capability inputs under:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/reasoner/nano/inputs/
packages/cosmos3/outputs/cookbooks/cosmos3/reasoner/nano/inputs/capabilities/
```

These are local cookbook inputs; they do not modify the shipped framework examples. Cosmos Framework Reasoner currently treats `vision_path` as a PIL image input, so video Reasoner examples should be run with [`run_with_vllm.ipynb`](./vllm.md).

```python
import json
from pathlib import Path
import os

input_dir = Path(os.environ["COSMOS3_INPUT_DIR"])
assets_dir = Path(os.environ["COSMOS_REASONER_ASSETS"])
capability_dir = input_dir / "capabilities"
input_dir.mkdir(parents=True, exist_ok=True)
capability_dir.mkdir(parents=True, exist_ok=True)


def write_reasoner_input(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(path)
    print(path.read_text())


write_reasoner_input(
    input_dir / "nano_text.json",
    {
        "model_mode": "reasoner",
        "name": "nano_text",
        "prompt": "Describe a modern robotics research laboratory in one sentence.",
        "enable_sound": False,
    },
)

write_reasoner_input(
    input_dir / "nano_image.json",
    {
        "model_mode": "reasoner",
        "name": "nano_image",
        "prompt": "Describe what is happening in this image in one sentence.",
        "vision_path": str((assets_dir / "robot_153.jpg").resolve()),
        "enable_sound": False,
    },
)

write_reasoner_input(
    capability_dir / "image_caption_detail.json",
    {
        "model_mode": "reasoner",
        "name": "image_caption_detail",
        "prompt": "Caption the image in detail.",
        "vision_path": str((assets_dir / "robot_153.jpg").resolve()),
        "enable_sound": False,
        "max_new_tokens": 4096,
    },
)

write_reasoner_input(
    capability_dir / "robot_planning.json",
    {
        "model_mode": "reasoner",
        "name": "robot_planning",
        "prompt": "The task is to put flower into the red bottle. Generate a plan consisting of subtasks for accomplish the task.",
        "vision_path": str((assets_dir / "robot_planning.png").resolve()),
        "enable_sound": False,
        "max_new_tokens": 4096,
    },
)

write_reasoner_input(
    capability_dir / "ground_load_bbox.json",
    {
        "model_mode": "reasoner",
        "name": "ground_load_bbox",
        "prompt": "Locate the accurate bounding box of the load as a whole. Return a json.",
        "vision_path": str((assets_dir / "grounding_2d.png").resolve()),
        "enable_sound": False,
        "max_new_tokens": 4096,
    },
)

write_reasoner_input(
    capability_dir / "describe_marked_subjects.json",
    {
        "model_mode": "reasoner",
        "name": "describe_marked_subjects",
        "prompt": 'Please caption the notable attributes in the provided image. List and describe all marked subjects in the image with their categories and detailed captions using a json with keyword "subject_id", "category" and "caption".',
        "vision_path": str((assets_dir / "describe_anything.png").resolve()),
        "enable_sound": False,
        "max_new_tokens": 4096,
    },
)

write_reasoner_input(
    capability_dir / "trajectory_bowl.json",
    {
        "model_mode": "reasoner",
        "name": "trajectory_bowl",
        "prompt": """You are given the task "Move the pink bowl to the right". Specify the 2D trajectory your end effector should follow in pixel space. Return the trajectory coordinates in JSON format like this: {"point_2d": [x, y], "label": "gripper trajectory"}.
Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag.
""",
        "vision_path": str((assets_dir / "action_cot_trajectory.png").resolve()),
        "enable_sound": False,
        "max_new_tokens": 4096,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
    },
)

write_reasoner_input(
    capability_dir / "trajectory_flower.json",
    {
        "model_mode": "reasoner",
        "name": "trajectory_flower",
        "prompt": """You are given the task "Put flower into the red bottle". Specify the 2D trajectory your end effector should follow in pixel space. Return the trajectory coordinates in JSON format like this: {"point_2d": [x, y], "label": "gripper trajectory"}.
Answer the question using the following format:

<think> Your reasoning. </think>
Write your final answer immediately after the </think> tag.
""",
        "vision_path": str((assets_dir / "robot_planning.png").resolve()),
        "enable_sound": False,
        "max_new_tokens": 4096,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
    },
)
```

````python
from html import escape
from pathlib import Path
import json
import re

from IPython.display import HTML, display
from PIL import Image as PILImage, ImageDraw

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}


def _media_kind(path_or_url: str) -> str:
    suffix = Path(path_or_url.split("?")[0]).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return "unknown"


def show_reasoner_io(sample: dict, output_text: str | None = None) -> None:
    # Render prompt, media input, and model output together for notebook review.
    prompt = escape(sample.get("prompt", ""))
    name = escape(sample.get("name", "sample"))
    media_path = sample.get("vision_path") or sample.get("video_path")

    if media_path:
        safe_media_path = escape(str(media_path), quote=True)
        media_kind = _media_kind(str(media_path))
        if media_kind == "image":
            media_html = f'<img src="{safe_media_path}" style="max-width:100%; max-height:360px; object-fit:contain; border:1px solid #ddd;">'
        elif media_kind == "video":
            media_html = f'<video src="{safe_media_path}" controls style="max-width:100%; max-height:360px; border:1px solid #ddd;"></video>'
        else:
            media_html = f'<a href="{safe_media_path}" target="_blank">{safe_media_path}</a>'
    else:
        media_html = "<em>No image or video input for this sample.</em>"

    output_block = ""
    if output_text is not None:
        output_block = (
            '<section style="margin-top:12px;">'
            '<div style="font-weight:600; margin-bottom:4px;">Model output</div>'
            f'<div style="white-space:pre-wrap; border:1px solid #ddd; padding:10px; background:#fafafa;">{escape(output_text.strip())}</div>'
            '</section>'
        )

    html = (
        '<div style="display:grid; grid-template-columns:minmax(260px, 1fr) minmax(260px, 1fr); gap:16px; align-items:start;">'
        '<section>'
        f'<div style="font-weight:600; margin-bottom:6px;">Input media: {name}</div>'
        f'{media_html}'
        '</section>'
        '<section>'
        '<div style="font-weight:600; margin-bottom:4px;">Text prompt</div>'
        f'<div style="white-space:pre-wrap; border:1px solid #ddd; padding:10px; background:#fafafa;">{prompt}</div>'
        f'{output_block}'
        '</section>'
        '</div>'
    )
    display(HTML(html))


def load_reasoner_result(input_path: Path, output_root: Path) -> tuple[dict, str | None]:
    sample = json.loads(input_path.read_text())
    text_path = output_root / sample["name"] / "reasoner_text.txt"
    output_text = text_path.read_text() if text_path.exists() else None
    show_reasoner_io(sample, output_text)
    print("output file:", text_path)
    return sample, output_text


def extract_json_payload(text: str):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    match = re.search(r"\[.*\]|\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def draw_boxes_if_present(sample: dict, output_text: str | None) -> None:
    if not output_text:
        return
    data = extract_json_payload(output_text)
    if data is None:
        return
    items = data if isinstance(data, list) else [data]
    boxes = []
    for item in items:
        if not isinstance(item, dict):
            continue
        box = item.get("bbox_2d") or item.get("bbox") or item.get("box")
        if box and len(box) == 4:
            boxes.append((box, item.get("label") or item.get("name") or item.get("category")))
    if not boxes:
        return

    image_path = Path(sample["vision_path"])
    if not image_path.exists():
        return
    img = PILImage.open(image_path).convert("RGB")
    width, height = img.size
    draw = ImageDraw.Draw(img)
    for box, label in boxes:
        x1, y1, x2, y2 = box
        # Cosmos/Qwen grounding prompts commonly return normalized 0-1000 coordinates.
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1000:
            x1, x2 = x1 / 1000 * width, x2 / 1000 * width
            y1, y2 = y1 / 1000 * height, y2 / 1000 * height
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        if label:
            draw.text((x1, max(0, y1 - 14)), str(label), fill="red")
    img.thumbnail((768, 768))
    display(img)


def draw_trajectory_if_present(sample: dict, output_text: str | None) -> None:
    if not output_text:
        return
    data = extract_json_payload(output_text)
    if data is None:
        return
    items = data if isinstance(data, list) else [data]
    points = []
    for item in items:
        if isinstance(item, dict) and "point_2d" in item and len(item["point_2d"]) == 2:
            points.append(tuple(item["point_2d"]))
    if not points:
        return

    image_path = Path(sample["vision_path"])
    if not image_path.exists():
        return
    img = PILImage.open(image_path).convert("RGB")
    width, height = img.size
    scaled = []
    for x, y in points:
        if max(abs(x), abs(y)) <= 1000:
            scaled.append((x / 1000 * width, y / 1000 * height))
        else:
            scaled.append((x, y))
    draw = ImageDraw.Draw(img)
    if len(scaled) > 1:
        draw.line(scaled, fill="lime", width=5)
    for idx, (x, y) in enumerate(scaled):
        radius = 12
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="red", outline="white", width=3)
        draw.text((x + 14, y - 14), str(idx), fill="yellow")
    img.thumbnail((900, 900))
    display(img)
````

## 7. Run Nano Text Inference

This runs `Cosmos3-Nano` on a text-only Reasoner prompt.

Expected output file:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/reasoner/nano/cosmos_framework_nano_text/nano_text/reasoner_text.txt
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
COSMOS_TRAINING=false CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_NANO_TEXT_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_INPUT_DIR/nano_text.json" \
  -o "$COSMOS3_OUTPUT_ROOT/cosmos_framework_nano_text" \
  --checkpoint-path Cosmos3-Nano \
  --seed=0 \
  --benchmark
```

```python
from pathlib import Path
import os, json

output_root = Path(os.environ["COSMOS3_OUTPUT_ROOT"])
text_path = output_root / "cosmos_framework_nano_text" / "nano_text" / "reasoner_text.txt"
benchmark_path = output_root / "cosmos_framework_nano_text" / "benchmark.json"

print(text_path)
print(text_path.read_text())
if benchmark_path.exists():
    print(json.dumps(json.loads(benchmark_path.read_text()).get("average", {}), indent=2))
```

### 7a. Optional: Run Nano FP8 Text Inference

FP8 checkpoints are published in the same Hugging Face repositories as the standard checkpoints, under the `fp8` revision rather than in separate FP8 repositories. This example downloads `nvidia/Cosmos3-Nano` at revision `fp8`. To use another released FP8 checkpoint, change `FP8_MODEL_ID` and `FP8_CHECKPOINT` below; keep `FP8_REVISION=fp8`.

Set `FP8_NUM_GPUS=1` to run without FSDP. To shard the model with FSDP, expose multiple GPUs in `CUDA_VISIBLE_DEVICES` and set `FP8_NUM_GPUS` to their count (for example, `CUDA_VISIBLE_DEVICES=0,1` and `FP8_NUM_GPUS=2`).

Expected output file:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/reasoner/nano/cosmos_framework_nano_fp8_text/nano_text/reasoner_text.txt
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
FP8_MODEL_ID="nvidia/Cosmos3-Nano"
FP8_REVISION="fp8"
FP8_CHECKPOINT="$COSMOS3_REPO/checkpoints/Cosmos3-Nano-FP8"
FP8_NUM_GPUS="${FP8_NUM_GPUS:-1}"
uvx hf@latest download "$FP8_MODEL_ID" \
  --revision "$FP8_REVISION" \
  --local-dir "$FP8_CHECKPOINT"

COSMOS_TRAINING=false CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
.venv/bin/torchrun \
  --nproc-per-node="$FP8_NUM_GPUS" \
  --master-addr="$COSMOS3_MASTER_ADDR" \
  --master-port="$COSMOS3_NANO_TEXT_MASTER_PORT" \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=throughput \
  --dp-shard-size="$FP8_NUM_GPUS" \
  --dp-replicate-size=1 \
  --cp-size=1 \
  --cfgp-size=1 \
  --no-use-torch-compile \
  -i "$COSMOS3_INPUT_DIR/nano_text.json" \
  -o "$COSMOS3_OUTPUT_ROOT/cosmos_framework_nano_fp8_text" \
  --checkpoint-path "$FP8_CHECKPOINT" \
  --seed=0 \
  --benchmark
```

```python
from pathlib import Path
import os, json

output_root = Path(os.environ["COSMOS3_OUTPUT_ROOT"])
text_path = output_root / "cosmos_framework_nano_fp8_text" / "nano_text" / "reasoner_text.txt"
benchmark_path = output_root / "cosmos_framework_nano_fp8_text" / "benchmark.json"

print(text_path)
print(text_path.read_text())
if benchmark_path.exists():
    print(json.dumps(json.loads(benchmark_path.read_text()).get("average", {}), indent=2))
```

## 8. Run Nano Image Inference

This runs `Cosmos3-Nano` on an image-conditioned Reasoner prompt. The result cell below renders the input image, text prompt, and model output side by side. The same display helper supports video URLs if future inputs use video files.

Expected output file:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/reasoner/nano/cosmos_framework_nano_image/nano_image/reasoner_text.txt
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
COSMOS_TRAINING=false CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_NANO_IMAGE_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_INPUT_DIR/nano_image.json" \
  -o "$COSMOS3_OUTPUT_ROOT/cosmos_framework_nano_image" \
  --checkpoint-path Cosmos3-Nano \
  --seed=0 \
  --benchmark
```

```python
from pathlib import Path
import os, json

input_dir = Path(os.environ["COSMOS3_INPUT_DIR"])
output_root = Path(os.environ["COSMOS3_OUTPUT_ROOT"])
sample = json.loads((input_dir / "nano_image.json").read_text())
text_path = output_root / "cosmos_framework_nano_image" / "nano_image" / "reasoner_text.txt"
benchmark_path = output_root / "cosmos_framework_nano_image" / "benchmark.json"
output_text = text_path.read_text()

show_reasoner_io(sample, output_text)

print("output file:", text_path)
if benchmark_path.exists():
    print(json.dumps(json.loads(benchmark_path.read_text()).get("average", {}), indent=2))
```

## 9. Image Caption

> **Note:** The Cosmos Framework Reasoner examples in this notebook are image-only. The current framework entrypoint treats `vision_path` as a PIL image source, so video Reasoner inputs should be run with [`run_with_vllm.ipynb`](./vllm.md).

Detailed image captioning example using the same robot image as the smoke test.

Expected output file:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/reasoner/nano/cosmos_framework_image_caption_detail/image_caption_detail/reasoner_text.txt
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
COSMOS_TRAINING=false CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_CAPABILITY_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_INPUT_DIR/capabilities/image_caption_detail.json" \
  -o "$COSMOS3_OUTPUT_ROOT/cosmos_framework_image_caption_detail" \
  --checkpoint-path Cosmos3-Nano \
  --seed=0 \
  --benchmark
```

```python
from pathlib import Path
import os

sample, output_text = load_reasoner_result(
    Path(os.environ["COSMOS3_INPUT_DIR"]) / "capabilities" / "image_caption_detail.json",
    Path(os.environ["COSMOS3_OUTPUT_ROOT"]) / "cosmos_framework_image_caption_detail",
)
```

## 10. Robot Planning

Embodied planning example for moving the flower into the red bottle.

Expected output file:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/reasoner/nano/cosmos_framework_robot_planning/robot_planning/reasoner_text.txt
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
COSMOS_TRAINING=false CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_CAPABILITY_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_INPUT_DIR/capabilities/robot_planning.json" \
  -o "$COSMOS3_OUTPUT_ROOT/cosmos_framework_robot_planning" \
  --checkpoint-path Cosmos3-Nano \
  --seed=0 \
  --benchmark
```

```python
from pathlib import Path
import os

sample, output_text = load_reasoner_result(
    Path(os.environ["COSMOS3_INPUT_DIR"]) / "capabilities" / "robot_planning.json",
    Path(os.environ["COSMOS3_OUTPUT_ROOT"]) / "cosmos_framework_robot_planning",
)
```

## 11. 2D Grounding

Grounding example that asks the model to locate the load as a bounding box and renders the parsed box when the output is valid JSON.

Expected output file:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/reasoner/nano/cosmos_framework_ground_load_bbox/ground_load_bbox/reasoner_text.txt
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
COSMOS_TRAINING=false CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_CAPABILITY_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_INPUT_DIR/capabilities/ground_load_bbox.json" \
  -o "$COSMOS3_OUTPUT_ROOT/cosmos_framework_ground_load_bbox" \
  --checkpoint-path Cosmos3-Nano \
  --seed=0 \
  --benchmark
```

```python
from pathlib import Path
import os

sample, output_text = load_reasoner_result(
    Path(os.environ["COSMOS3_INPUT_DIR"]) / "capabilities" / "ground_load_bbox.json",
    Path(os.environ["COSMOS3_OUTPUT_ROOT"]) / "cosmos_framework_ground_load_bbox",
)
draw_boxes_if_present(sample, output_text)
```

## 12. Describe Anything

Marked-subject description example that asks for a JSON list of subject IDs, categories, and captions.

Expected output file:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/reasoner/nano/cosmos_framework_describe_marked_subjects/describe_marked_subjects/reasoner_text.txt
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
COSMOS_TRAINING=false CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_CAPABILITY_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_INPUT_DIR/capabilities/describe_marked_subjects.json" \
  -o "$COSMOS3_OUTPUT_ROOT/cosmos_framework_describe_marked_subjects" \
  --checkpoint-path Cosmos3-Nano \
  --seed=0 \
  --benchmark
```

```python
from pathlib import Path
import os

sample, output_text = load_reasoner_result(
    Path(os.environ["COSMOS3_INPUT_DIR"]) / "capabilities" / "describe_marked_subjects.json",
    Path(os.environ["COSMOS3_OUTPUT_ROOT"]) / "cosmos_framework_describe_marked_subjects",
)
```

## 13. Action CoT: Trajectory Coordinates

Action trajectory example that asks for 2D gripper coordinates for moving the pink bowl to the right. The display cell renders parsed points when the output is valid JSON.

Expected output file:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/reasoner/nano/cosmos_framework_trajectory_bowl/trajectory_bowl/reasoner_text.txt
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
COSMOS_TRAINING=false CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_CAPABILITY_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_INPUT_DIR/capabilities/trajectory_bowl.json" \
  -o "$COSMOS3_OUTPUT_ROOT/cosmos_framework_trajectory_bowl" \
  --checkpoint-path Cosmos3-Nano \
  --seed=0 \
  --benchmark
```

```python
from pathlib import Path
import os

sample, output_text = load_reasoner_result(
    Path(os.environ["COSMOS3_INPUT_DIR"]) / "capabilities" / "trajectory_bowl.json",
    Path(os.environ["COSMOS3_OUTPUT_ROOT"]) / "cosmos_framework_trajectory_bowl",
)
draw_trajectory_if_present(sample, output_text)
```

## 14. Action CoT: Robot Plan Trajectory

Second trajectory example using the robot planning image and the flower-to-red-bottle task.

Expected output file:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/reasoner/nano/cosmos_framework_trajectory_flower/trajectory_flower/reasoner_text.txt
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
COSMOS_TRAINING=false CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_CAPABILITY_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_INPUT_DIR/capabilities/trajectory_flower.json" \
  -o "$COSMOS3_OUTPUT_ROOT/cosmos_framework_trajectory_flower" \
  --checkpoint-path Cosmos3-Nano \
  --seed=0 \
  --benchmark
```

```python
from pathlib import Path
import os

sample, output_text = load_reasoner_result(
    Path(os.environ["COSMOS3_INPUT_DIR"]) / "capabilities" / "trajectory_flower.json",
    Path(os.environ["COSMOS3_OUTPUT_ROOT"]) / "cosmos_framework_trajectory_flower",
)
draw_trajectory_if_present(sample, output_text)
```

## 15. Run Edge Text Inference

`Cosmos3-Edge` runs the same Reasoner entrypoint as Nano. This runs it on the same text-only prompt used for Nano in step 7 — only `--checkpoint-path` changes; the input JSON (`nano_text.json`) is identical.

Expected output file:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/reasoner/nano/cosmos_framework_edge_text/nano_text/reasoner_text.txt
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
COSMOS_TRAINING=false CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_NANO_TEXT_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_INPUT_DIR/nano_text.json" \
  -o "$COSMOS3_OUTPUT_ROOT/cosmos_framework_edge_text" \
  --checkpoint-path Cosmos3-Edge \
  --seed=0 \
  --benchmark
```

```python
from pathlib import Path
import os, json

output_root = Path(os.environ["COSMOS3_OUTPUT_ROOT"])
text_path = output_root / "cosmos_framework_edge_text" / "nano_text" / "reasoner_text.txt"
benchmark_path = output_root / "cosmos_framework_edge_text" / "benchmark.json"

print(text_path)
print(text_path.read_text())
if benchmark_path.exists():
    print(json.dumps(json.loads(benchmark_path.read_text()).get("average", {}), indent=2))
```

## 16. Run Edge Image Inference

This runs `Cosmos3-Edge` on the same image-conditioned Reasoner prompt used for Nano in step 8. Only `--checkpoint-path` changes; the input JSON (`nano_image.json`) is identical. The result cell renders the input image, text prompt, and model output side by side.

Expected output file:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/reasoner/nano/cosmos_framework_edge_image/nano_image/reasoner_text.txt
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
COSMOS_TRAINING=false CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_NANO_IMAGE_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_INPUT_DIR/nano_image.json" \
  -o "$COSMOS3_OUTPUT_ROOT/cosmos_framework_edge_image" \
  --checkpoint-path Cosmos3-Edge \
  --seed=0 \
  --benchmark
```

```python
from pathlib import Path
import os, json

input_dir = Path(os.environ["COSMOS3_INPUT_DIR"])
output_root = Path(os.environ["COSMOS3_OUTPUT_ROOT"])
sample = json.loads((input_dir / "nano_image.json").read_text())
text_path = output_root / "cosmos_framework_edge_image" / "nano_image" / "reasoner_text.txt"
benchmark_path = output_root / "cosmos_framework_edge_image" / "benchmark.json"
output_text = text_path.read_text()

show_reasoner_io(sample, output_text)

print("output file:", text_path)
if benchmark_path.exists():
    print(json.dumps(json.loads(benchmark_path.read_text()).get("average", {}), indent=2))
```

## 17. Run Super Text Inference

`Cosmos3-Super` runs the same Reasoner entrypoint as Nano/Edge. This runs it on the same text-only prompt used for Nano in step 7 — only `--checkpoint-path` changes; the input JSON (`nano_text.json`) is identical. Super needs a large-memory GPU (or multiple GPUs).

Expected output file:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/reasoner/nano/cosmos_framework_super_text/nano_text/reasoner_text.txt
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
COSMOS_TRAINING=false CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_NANO_TEXT_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_INPUT_DIR/nano_text.json" \
  -o "$COSMOS3_OUTPUT_ROOT/cosmos_framework_super_text" \
  --checkpoint-path Cosmos3-Super \
  --seed=0 \
  --benchmark
```

```python
from pathlib import Path
import os, json

output_root = Path(os.environ["COSMOS3_OUTPUT_ROOT"])
text_path = output_root / "cosmos_framework_super_text" / "nano_text" / "reasoner_text.txt"
benchmark_path = output_root / "cosmos_framework_super_text" / "benchmark.json"

print(text_path)
print(text_path.read_text())
if benchmark_path.exists():
    print(json.dumps(json.loads(benchmark_path.read_text()).get("average", {}), indent=2))
```

## 18. Run Super Image Inference

This runs `Cosmos3-Super` on the same image-conditioned Reasoner prompt used for Nano in step 8. Only `--checkpoint-path` changes; the input JSON (`nano_image.json`) is identical. The result cell renders the input image, text prompt, and model output side by side.

Expected output file:

```text
packages/cosmos3/outputs/cookbooks/cosmos3/reasoner/nano/cosmos_framework_super_image/nano_image/reasoner_text.txt
```

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
COSMOS_TRAINING=false CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_NANO_IMAGE_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_INPUT_DIR/nano_image.json" \
  -o "$COSMOS3_OUTPUT_ROOT/cosmos_framework_super_image" \
  --checkpoint-path Cosmos3-Super \
  --seed=0 \
  --benchmark
```

```python
from pathlib import Path
import os, json

input_dir = Path(os.environ["COSMOS3_INPUT_DIR"])
output_root = Path(os.environ["COSMOS3_OUTPUT_ROOT"])
sample = json.loads((input_dir / "nano_image.json").read_text())
text_path = output_root / "cosmos_framework_super_image" / "nano_image" / "reasoner_text.txt"
benchmark_path = output_root / "cosmos_framework_super_image" / "benchmark.json"
output_text = text_path.read_text()

show_reasoner_io(sample, output_text)

print("output file:", text_path)
if benchmark_path.exists():
    print(json.dumps(json.loads(benchmark_path.read_text()).get("average", {}), indent=2))
```
