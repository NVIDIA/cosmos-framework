<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Reasoner with Transformers

This notebook runs Cosmos3 Reasoner inference directly with Hugging Face
Transformers — a Python-first path with no server to launch.

1. Sets up an isolated venv with the Cosmos3 Transformers integration.
2. Registers a Jupyter kernel and loads the Reasoner in process.
3. Runs image and video reasoning requests for **Nano**, **Super**, or **Edge**.

**Nano / Super** use `Cosmos3OmniForConditionalGeneration` with the unified
`nvidia/Cosmos3-Nano` or `nvidia/Cosmos3-Super` checkpoint.
**Edge** uses `AutoModelForImageTextToText` (`Cosmos3EdgeForConditionalGeneration`)
with `nvidia/Cosmos3-Edge` — do not load Edge with the Omni class.

The integration loads **only the Reasoner tower** and returns text for text,
image, and video understanding. It does not generate images, video, audio, or
actions — use the Diffusers or vLLM-Omni cookbooks for those.

This notebook installs Transformers from GitHub `main` so Edge works.
Nano/Super also run on that build (PyPI `transformers>=5.11.0` is enough if you
only need Nano/Super).

Note: if you have already completed steps 1-4 and installed the
`Cosmos3 Transformers (Python 3.13)` kernel, switch to that kernel, run the
Restore Environment cell in step 4, then continue from step 5.

> **Skill note (not in the source):** this file is the cookbook notebook `run_with_transformers.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/reasoner/run_with_transformers.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

<!-- rumdl-disable MD034 -->
<!-- Skill note (not in the source): the directive above keeps the notebook's bare URL in this section's "uv version" note as written; the rule is re-enabled right after the section. -->

## 1. Prerequisites

Use a Linux machine with NVIDIA GPU access, model access on Hugging Face, and
either `uvx hf@latest auth login` or `HF_TOKEN` set.

> **Headless servers:** if you see an error like `libxcb.so.1: cannot open shared
> object file` when importing, install the required system libraries:
>
> ```bash
> apt-get install -y libxcb1 libgl1 libglib2.0-0
> ```

> **uv version:** use `uv 0.11.29`, matching the version pinned by Cosmos Framework.
> Install it from https://astral.sh/uv/0.11.29/install.sh or update an existing
> self-managed installation with `uv self update 0.11.29`.

<!-- rumdl-enable MD034 -->

## 2. Configure Paths and Environment

The defaults are relative to this `cosmos` checkout. Override any of these before
running the next cell if needed:

```bash
export COSMOS3_TRANSFORMERS_VENV=/path/to/.venv-cosmos3-transformers
export COSMOS3_TORCH_BACKEND=auto   # or cu130 / cu128 to pin an explicit CUDA wheel
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


def configure_transformers_environment() -> None:
    global COSMOS_ROOT
    global COSMOS_REASONER_ASSETS
    global COSMOS3_TRANSFORMERS_VENV
    global COSMOS3_TORCH_BACKEND

    COSMOS_ROOT = find_repo_root(Path.cwd().resolve())
    COSMOS_REASONER_ASSETS = COSMOS_ROOT / "cookbooks" / "cosmos3" / "reasoner" / "assets"
    COSMOS3_TRANSFORMERS_VENV = Path(
        os.environ.get("COSMOS3_TRANSFORMERS_VENV", COSMOS_ROOT / ".venv-cosmos3-transformers")
    ).resolve()
    COSMOS3_TORCH_BACKEND = os.environ.get("COSMOS3_TORCH_BACKEND", "auto")

    os.environ["COSMOS3_TRANSFORMERS_VENV"] = str(COSMOS3_TRANSFORMERS_VENV)
    os.environ["COSMOS3_TORCH_BACKEND"] = COSMOS3_TORCH_BACKEND
    os.environ.setdefault("UV_LINK_MODE", "copy")

    assert COSMOS_REASONER_ASSETS.exists(), COSMOS_REASONER_ASSETS


def asset_path(name: str) -> Path:
    path = COSMOS_REASONER_ASSETS / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


configure_transformers_environment()
print("cosmos root:", COSMOS_ROOT)
print("Reasoner assets:", COSMOS_REASONER_ASSETS)
print("Transformers venv:", COSMOS3_TRANSFORMERS_VENV)
print("Torch backend:", COSMOS3_TORCH_BACKEND)
```

## 3. Install Transformers Dependencies

This cell creates the venv, installs dependencies (including Transformers from
`main`, required for Cosmos3-Edge), and registers a Jupyter kernel so the model
can run in process.

Nano/Super Reasoner support first appears in Transformers `v5.11.0`. Edge support
is on Transformers `main` until a PyPI release includes it.

`--torch-backend` defaults to `auto`, which lets uv pick a CUDA build of
`torch`/`torchvision` that matches your driver. Set `COSMOS3_TORCH_BACKEND=cu130`
(or `cu128`) above to pin an explicit wheel.

```bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
uv venv "$COSMOS3_TRANSFORMERS_VENV" --python 3.13 --seed --managed-python --allow-existing
source "$COSMOS3_TRANSFORMERS_VENV/bin/activate"

uv pip install --torch-backend="$COSMOS3_TORCH_BACKEND" \
  accelerate \
  av \
  ipykernel \
  pillow \
  "safetensors>=0.8.0" \
  torch \
  "torchvision==0.25.0" \
  "transformers @ git+https://github.com/huggingface/transformers.git"

"$COSMOS3_TRANSFORMERS_VENV/bin/python" -m ipykernel install --user \
  --name cosmos3-transformers \
  --display-name "Cosmos3 Transformers (Python 3.13)"

echo
echo "Installed dependencies into: $COSMOS3_TRANSFORMERS_VENV"
echo "Next: switch this notebook kernel to: Cosmos3 Transformers (Python 3.13)"
```

### 3a. Optional: Install ModelOpt for FP8

Skip this subsection for standard BF16 inference. To load the public Nano/Super `fp8`
revision, run the cell below before switching kernels. `requests` is installed
explicitly because ModelOpt `0.44.0` imports it but does not declare it as a dependency.

```bash
set -euo pipefail

source "$COSMOS3_TRANSFORMERS_VENV/bin/activate"
uv pip install \
  "nvidia-modelopt==0.44.0" \
  requests
```

## 4. Select the Transformers Kernel

The install cell registers the `Cosmos3 Transformers (Python 3.13)` Jupyter kernel.

**Switch this notebook to that kernel before running the remaining Python cells**,
then run the Restore Environment cell immediately below. It can take a moment for
the new kernel to appear in the notebook interface.

```python
# Run this cell immediately after switching to the Cosmos3 Transformers kernel.
# It restores the same paths as the configure cell in step 2.
from pathlib import Path
import os


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "cookbooks" / "cosmos3").is_dir():
            return path
    return start


COSMOS_ROOT = find_repo_root(Path.cwd().resolve())
COSMOS_REASONER_ASSETS = COSMOS_ROOT / "cookbooks" / "cosmos3" / "reasoner" / "assets"
COSMOS3_TRANSFORMERS_VENV = Path(
    os.environ.get("COSMOS3_TRANSFORMERS_VENV", COSMOS_ROOT / ".venv-cosmos3-transformers")
).resolve()
os.environ["COSMOS3_TRANSFORMERS_VENV"] = str(COSMOS3_TRANSFORMERS_VENV)


def asset_path(name: str) -> Path:
    path = COSMOS_REASONER_ASSETS / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


print("cosmos root:", COSMOS_ROOT)
print("Reasoner assets:", COSMOS_REASONER_ASSETS)
```

## 5. Verify GPU and Python Environment

```python
import os
import sys
from pathlib import Path

if "COSMOS3_TRANSFORMERS_VENV" not in os.environ:
    raise RuntimeError("Run the Restore Environment cell after switching to the Transformers kernel.")

expected_python = (Path(os.environ["COSMOS3_TRANSFORMERS_VENV"]) / "bin" / "python").resolve()
current_python = Path(sys.executable).resolve()
print("kernel python:", current_python)
print("expected python:", expected_python)
if current_python != expected_python:
    raise RuntimeError(
        "This notebook is not running inside the Transformers venv. "
        "Switch the kernel to 'Cosmos3 Transformers (Python 3.13)', then run the Restore Environment cell above."
    )

import torch
import transformers

print("transformers:", transformers.__version__)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
```

## 6. Load the Reasoner

Load the processor and model once, then reuse them for every request below.

The default is the public BF16 checkpoint. To use the public Nano/Super FP8 weights,
set `USE_MODELOPT_FP8 = True`. The model ID stays the same and Transformers
downloads the `fp8` revision to the Hugging Face cache and loads it from the
resolved snapshot path. ModelOpt restores that snapshot's calibrated E4M3 weights
and static scales; this is not runtime quantization.

Set `model_id` to `nvidia/Cosmos3-Nano`, `nvidia/Cosmos3-Super`, or
`nvidia/Cosmos3-Edge`. `device_map="auto"` places the model on the available
GPU(s) and can shard Super across multiple GPUs when Accelerate is installed.

> **First run downloads the checkpoint** (tens of GiB for Nano/Super; Edge is
> smaller). Nano/Super load only the Reasoner tower from the unified checkpoint
> into memory. Subsequent runs reuse the Hugging Face cache.

```python
import torch

USE_MODELOPT_FP8 = False
model_id = "nvidia/Cosmos3-Nano"  # or "nvidia/Cosmos3-Super" or "nvidia/Cosmos3-Edge"
FP8_REVISION = "fp8"

if USE_MODELOPT_FP8:
    # Enable ModelOpt restoration before importing the Transformers model class.
    # Importing this backend registers the real per-tensor FP8 GEMM implementation.
    from huggingface_hub import snapshot_download
    from modelopt.torch import opt as modelopt_opt
    from modelopt.torch.quantization.backends import fp8_per_tensor_gemm  # noqa: F401

    modelopt_opt.enable_huggingface_checkpointing()
    checkpoint_path = snapshot_download(model_id, revision=FP8_REVISION)
else:
    checkpoint_path = model_id

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Cosmos3OmniForConditionalGeneration,
)

# Nano / Super: Omni class. Edge: Auto image-text API (Cosmos3EdgeForConditionalGeneration).
processor = AutoProcessor.from_pretrained(checkpoint_path)
if model_id == "nvidia/Cosmos3-Edge":
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
else:
    model = Cosmos3OmniForConditionalGeneration.from_pretrained(
        checkpoint_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )


def run_reasoner(content, fps=None, max_new_tokens=512):
    """Run one Reasoner request. `content` is a chat content list (image/video/text blocks)."""
    messages = [{"role": "user", "content": content}]
    template_kwargs = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if fps is not None:
        template_kwargs["fps"] = fps

    inputs = processor.apply_chat_template(messages, **template_kwargs).to(model.device, torch.bfloat16)
    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


print("Loaded", model_id, "as", type(model).__name__)
```

## 7. Image Reasoning

```python
from IPython.display import Image, display

image_path = asset_path("robot_153.jpg")
display(Image(filename=str(image_path), width=512))

output = run_reasoner(
    [
        {"type": "image", "path": str(image_path.resolve())},
        {"type": "text", "text": "Caption the image in detail."},
    ]
)
print(output)
```

## 8. Video Reasoning

Use a `video` content block and pass a frame sampling rate (`fps`) to the helper.

> Video decoding uses the packages installed above. Transformers prints a
> deprecation warning that it fell back to the `torchvision` decoder — this is
> expected and harmless. To switch to the modern `torchcodec` decoder, install it
> along with system FFmpeg libraries (`libavutil`/`libavcodec`).

```python
from IPython.display import Video, display

video_path = asset_path("video_caption.mp4")
display(Video(str(video_path), embed=True, width=640))

output = run_reasoner(
    [
        {"type": "video", "path": str(video_path.resolve())},
        {"type": "text", "text": "Describe the notable events in this video."},
    ],
    fps=2,
)
print(output)
```

## 9. Next Steps

- Run **Cosmos3-Super**: set `model_id = "nvidia/Cosmos3-Super"` in step 6 and
  re-run from there. `device_map="auto"` shards it across multiple GPUs.
- Run **Cosmos3-Edge**: set `model_id = "nvidia/Cosmos3-Edge"` in step 6 and
  re-run from there (uses `AutoModelForImageTextToText`, not the Omni class).
- Run **ModelOpt FP8 Nano/Super**: set `USE_MODELOPT_FP8 = True` and re-run from step 6.
  The same model ID is loaded from its public `fp8` revision (Edge has none).
- Try other Reasoner tasks (temporal localization, grounding, embodied reasoning)
  by changing the prompt and asset — see the
  [Reasoner Prompt Guide](./prompt-guide.md).
- Need an OpenAI-compatible server instead of in-process Python? See
  [`run_with_vllm.ipynb`](./vllm.md) or the [Cosmos3 NIM](../../cookbooks/cosmos3/nim/reasoning.md).
