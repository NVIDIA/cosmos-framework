<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Quantize Cosmos3-Super Text-to-Image to FP8

Cosmos3-Super Text-to-Image ships in two forms: the full-quality model and a **4-step
distilled** version tuned for fast, few-step generation. This notebook quantizes both to
FP8 — the base model first, then the distilled model.

The two are the *same* network; they are served with a different **sampler** (scheduler,
number of steps, guidance). Calibration replays the sampler each version actually uses, so
we quantize each one with its own sampler — spelled out in the distilled section below.

> **Skill note (not in the source):** this file is the cookbook notebook `quantize_super_text2image.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/quantization/quantize_super_text2image.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

## 1. Prerequisites

Use a Linux machine with an NVIDIA GPU, model access on Hugging Face, and either
`uvx hf@latest auth login` or `HF_TOKEN` set. Quantizing the Super (64B) checkpoints
needs a GPU with enough memory to hold the model in bf16 plus the quantizers; Nano (16B)
also needs room for BF16 weights, activations, and quantizer state.

You also need free disk for the Hugging Face checkpoint cache and for the FP8 output
(each output is roughly the size of the bf16 source). Point `HF_HOME` at a large volume
in the next step.

> **Headless servers:** if you see `libGL.so.1: cannot open shared object file` when the
> VAE loads, install the system graphics libraries:
>
> ```bash
> apt-get install -y libgl1 libglib2.0-0
> ```

## 2. Configure Paths and Environment

The defaults are relative to this `cosmos-framework` checkout and use the CUDA 13 (or 12.8) Torch
backend depending on your system. Override any of these before running the cell:

```bash
export COSMOS3_QUANTIZE_VENV=/path/to/cosmos-framework/.venv
export COSMOS3_TORCH_BACKEND=cu130       # or cu128
export HF_HOME=/path/to/large/huggingface/cache
export OUTPUT_ROOT=/path/to/fp8/outputs
```

```python
from pathlib import Path
import os


def find_framework_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").is_file() and (path / "cosmos_framework").is_dir():
            return path
    raise FileNotFoundError("Launch from the Framework checkout or a notebook directory inside it.")


def configure_quantize_environment() -> None:
    global FRAMEWORK_ROOT, COSMOS_ROOT, QUANTIZE_ROOT, COSMOS3_QUANTIZE_VENV
    global COSMOS3_TORCH_BACKEND, COSMOS3_UV_GROUP, OUTPUT_ROOT

    FRAMEWORK_ROOT = find_framework_root(Path.cwd().resolve())
    COSMOS_ROOT = FRAMEWORK_ROOT / ".agents" / "skills"
    QUANTIZE_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3" / "quantization"
    COSMOS3_QUANTIZE_VENV = Path(
        os.environ.get("COSMOS3_QUANTIZE_VENV", FRAMEWORK_ROOT / ".venv")
    ).expanduser().resolve()
    COSMOS3_TORCH_BACKEND = os.environ.get("COSMOS3_TORCH_BACKEND", "cu130")
    COSMOS3_UV_GROUP = os.environ.get("COSMOS3_UV_GROUP", f"{COSMOS3_TORCH_BACKEND}-train")
    OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", QUANTIZE_ROOT / "outputs")).expanduser().resolve()

    os.environ["FRAMEWORK_ROOT"] = str(FRAMEWORK_ROOT)
    os.environ["COSMOS3_QUANTIZE_VENV"] = str(COSMOS3_QUANTIZE_VENV)
    os.environ["COSMOS3_TORCH_BACKEND"] = COSMOS3_TORCH_BACKEND
    os.environ["COSMOS3_UV_GROUP"] = COSMOS3_UV_GROUP
    os.environ["OUTPUT_ROOT"] = str(OUTPUT_ROOT)
    os.environ.setdefault("UV_LINK_MODE", "copy")
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


configure_quantize_environment()
print("Framework checkout:", FRAMEWORK_ROOT)
print("Cookbook root:", COSMOS_ROOT)
print("Quantization root:", QUANTIZE_ROOT)
print("Output root:", OUTPUT_ROOT)
print("Environment:", COSMOS3_QUANTIZE_VENV, COSMOS3_UV_GROUP)
```

## 3. Install Dependencies

Install the local Framework checkout using its documented `uv sync --all-extras`
environment and matching `cu130-train` / `cu128-train` group. Add ModelOpt,
datasets, and notebook tooling to that environment, then register its kernel.
See [shared setup](../../cookbooks/cosmos3/README.md#cosmos-framework) and
[Framework setup](../../../../docs/setup.md) for system libraries and CUDA.
Run this cell once; re-running `uv sync` later may remove these added packages,
so re-run the `uv pip install` command after syncing.

```bash
set -euo pipefail
: "${FRAMEWORK_ROOT:?run Configure Paths first}"
cd "$FRAMEWORK_ROOT"
export GIT_LFS_SKIP_SMUDGE=1
export UV_PROJECT_ENVIRONMENT="$COSMOS3_QUANTIZE_VENV"
uv sync --all-extras --group="$COSMOS3_UV_GROUP"
uv pip install --python "$COSMOS3_QUANTIZE_VENV/bin/python" \
  "nvidia-modelopt[torch]==0.44.0" datasets ipykernel jupyter-client nbconvert
"$COSMOS3_QUANTIZE_VENV/bin/python" -m ipykernel install --user \
  --name cosmos3-quantize --display-name "Cosmos3 Quantize (Python 3.13)"
```

## 4. Select the Quantization Kernel

The install cell registers the `Cosmos3 Quantize (Python 3.13)` Jupyter kernel.

**Switch this notebook to that kernel**, then run the restore cell below before
continuing. It can take a moment for a new kernel to appear in the notebook interface.

```python
from pathlib import Path
import os


def find_framework_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").is_file() and (path / "cosmos_framework").is_dir():
            return path
    raise FileNotFoundError("Launch from the Framework checkout or a notebook directory inside it.")


def configure_quantize_environment() -> None:
    global FRAMEWORK_ROOT, COSMOS_ROOT, QUANTIZE_ROOT, COSMOS3_QUANTIZE_VENV
    global COSMOS3_TORCH_BACKEND, COSMOS3_UV_GROUP, OUTPUT_ROOT

    FRAMEWORK_ROOT = find_framework_root(Path.cwd().resolve())
    COSMOS_ROOT = FRAMEWORK_ROOT / ".agents" / "skills"
    QUANTIZE_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3" / "quantization"
    COSMOS3_QUANTIZE_VENV = Path(
        os.environ.get("COSMOS3_QUANTIZE_VENV", FRAMEWORK_ROOT / ".venv")
    ).expanduser().resolve()
    COSMOS3_TORCH_BACKEND = os.environ.get("COSMOS3_TORCH_BACKEND", "cu130")
    COSMOS3_UV_GROUP = os.environ.get("COSMOS3_UV_GROUP", f"{COSMOS3_TORCH_BACKEND}-train")
    OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", QUANTIZE_ROOT / "outputs")).expanduser().resolve()

    os.environ["FRAMEWORK_ROOT"] = str(FRAMEWORK_ROOT)
    os.environ["COSMOS3_QUANTIZE_VENV"] = str(COSMOS3_QUANTIZE_VENV)
    os.environ["COSMOS3_TORCH_BACKEND"] = COSMOS3_TORCH_BACKEND
    os.environ["COSMOS3_UV_GROUP"] = COSMOS3_UV_GROUP
    os.environ["OUTPUT_ROOT"] = str(OUTPUT_ROOT)
    os.environ.setdefault("UV_LINK_MODE", "copy")
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


configure_quantize_environment()
print("Framework checkout:", FRAMEWORK_ROOT)
print("Cookbook root:", COSMOS_ROOT)
print("Quantization root:", QUANTIZE_ROOT)
print("Output root:", OUTPUT_ROOT)
print("Environment:", COSMOS3_QUANTIZE_VENV, COSMOS3_UV_GROUP)
```

## 5. Verify GPU and Python Environment

Confirm the kernel sees a GPU and the quantization engine imports cleanly.

```python
import torch, modelopt
print('torch          :', torch.__version__)
print('CUDA available :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU            :', torch.cuda.get_device_name(0))
print('ModelOpt       :', modelopt.__version__)
```

## 6. Load the Quantization Toolkit

The FP8 recipe lives in `cosmos_framework.quantization`.
`quantize_fp8_checkpoint(...)` loads, calibrates and exports the model;
the inspection helper below summarizes the resulting checkpoint.

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

```python
import json

from cosmos_framework.quantization import (
    SAMPLER_DISTILLED,
    SAMPLER_IMAGE_BASE,
    SHAPE_IMAGE,
    SHAPE_IMAGE_DEMO,
    quantize_fp8_checkpoint,
)
from safetensors import safe_open

# DEMO=1 (default): one calibration prompt at a small shape, so a run takes minutes.
# Set DEMO=0 for the production shape and 8 calibration prompts (a larger calibration run).
DEMO = os.environ.get('DEMO', '1') == '1'
NUM_SAMPLES = 1 if DEMO else 8
print(f'DEMO={DEMO}  (NUM_SAMPLES={NUM_SAMPLES})')


def inspect_checkpoint(output_dir):
    """Print a short, human-readable summary of an FP8 checkpoint."""
    output_dir = Path(output_dir)
    tdir = output_dir / 'transformer'
    n_fp8 = n_scale = 0
    example = []
    for shard in sorted(tdir.glob('*.safetensors')):
        with safe_open(str(shard), framework='pt') as h:
            for k in h.keys():
                if k.endswith(('.input_scale', '.weight_scale')):
                    n_scale += 1
                    if k.endswith('.input_scale') and len(example) < 3:
                        example.append((k, float(h.get_tensor(k).reshape(-1)[0])))
                elif k.endswith('.weight') and h.get_slice(k).get_dtype() == 'F8_E4M3':
                    n_fp8 += 1
    qcfg = json.load(open(output_dir / 'hf_quant_config.json'))
    print(f'FP8 checkpoint: {output_dir}')
    print(f"  quant_algo   : {qcfg.get('quant_algo')}  (method={qcfg.get('quant_method')})")
    print(f'  FP8 weights  : {n_fp8}')
    print(f'  scale tensors: {n_scale}')
    for k, v in example:
        print(f'    e.g. {k} = {v:.6g}')
```

## Cosmos3-Super Text-to-Image (Base)

The full-quality text-to-image model. It is served with a 50-step UniPC sampler, so that
is what we calibrate.

### Quantize

```python
model_name_or_path = os.environ.get('C3_T2I_DIR', 'nvidia/Cosmos3-Super-Text2Image')
output_dir = OUTPUT_ROOT / 'super-t2i-fp8'

quantize_fp8_checkpoint(
    model_name_or_path=model_name_or_path, output_dir=output_dir,
    profile='t2i', sampler=SAMPLER_IMAGE_BASE, shape=SHAPE_IMAGE_DEMO if DEMO else SHAPE_IMAGE,
    num_samples=NUM_SAMPLES, calibration_behavior='framework',
)
```

### Inspect the FP8 Checkpoint

A quick look at the drop-in checkpoint just written: the quantization method, how many
weights were converted to FP8, and the per-tensor scales that accompany them.

```python
inspect_checkpoint(OUTPUT_ROOT / 'super-t2i-fp8')
```

## Cosmos3-Super Text-to-Image — 4-Step Distilled

The distilled model is the **same network** as the base, but it is served in just 4 steps.
The only thing that changes for quantization is the **sampler** — so calibration replays
the 4-step schedule instead of the 50-step one:

|           | scheduler                      | steps | guidance           |
| --------- | ------------------------------ | ----- | ------------------ |
| base      | UniPC                          | 50    | 6.0                |
| distilled | FlowMatchEuler (4 fixed steps) | 4     | 1.0 (guidance off) |

The recipe reads the correct scheduler and step schedule from each checkpoint, so from
your side the call below is identical to the base — only `sampler=SAMPLER_DISTILLED`
changes.

### Quantize

```python
# Release the previous model's reference cycles and cached CUDA allocations.
import gc

gc.collect()
torch.cuda.empty_cache()
print(f'CUDA memory allocated after cleanup: {torch.cuda.memory_allocated() / 2**30:.2f} GiB')

model_name_or_path = os.environ.get('C3_T2I_4STEP_DIR', 'nvidia/Cosmos3-Super-Text2Image-4Step')
output_dir = OUTPUT_ROOT / 'super-t2i-distilled-fp8'

quantize_fp8_checkpoint(
    model_name_or_path=model_name_or_path, output_dir=output_dir,
    profile='t2i', sampler=SAMPLER_DISTILLED, shape=SHAPE_IMAGE_DEMO if DEMO else SHAPE_IMAGE,
    num_samples=NUM_SAMPLES, calibration_behavior='framework',
)
```

### Inspect the FP8 Checkpoint

A quick look at the drop-in checkpoint just written: the quantization method, how many
weights were converted to FP8, and the per-tensor scales that accompany them.

```python
inspect_checkpoint(OUTPUT_ROOT / 'super-t2i-distilled-fp8')
```
