<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Video Transfer with Diffusers

This notebook runs Cosmos3 structural-control transfer with `Cosmos3OmniModularPipeline`, the pipeline that exposes control inputs.

It runs edge, blur, depth, segmentation, and world-scenario-map transfer on Cosmos3-Nano, then the same five controls on Cosmos3-Super.

FP8 checkpoints: set `COSMOS3_DIFFUSERS_PRECISION=fp8` before loading. Nano and Super use the dedicated Hub repos `nvidia/Cosmos3-Nano-FP8` and `nvidia/Cosmos3-Super-FP8`. Use-case cells stay the same.

> **Skill note (not in the source):** this file is the cookbook notebook `run_video_transfer_with_diffusers.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/generator/transfer/run_video_transfer_with_diffusers.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

## 1. Prerequisites

Use a Linux machine with NVIDIA GPU access, model access on Hugging Face, and either `uvx hf@latest auth login` or `HF_TOKEN` set.

Each control is a precomputed video under [`assets/`](../assets), paired with an upsampled JSON prompt.

Blur, segmentation, and WSM use the Guardrail and require access to the gated [nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail) HF repository. The edge and depth calls explicitly disable the safety checker.

> **Headless servers:** if you see an error like `libxcb.so.1: cannot open shared object file` (a missing system graphics library) when importing or running the pipeline, install the required system libraries:
>
> ```bash
> apt-get install -y libxcb1 libgl1 libglib2.0-0
> ```

> **uv version:** use `uv 0.11.29`, matching the version pinned by Cosmos Framework. Install it from <https://astral.sh/uv/0.11.29/install.sh> or update an existing self-managed installation with `uv self update 0.11.29`.

## 2. Configure Paths and Environment

The defaults are relative to this `cosmos` checkout and use the CUDA 13 or 12.8 Torch backend depending on the CUDA version installed on your system (`cu130` or `cu128`):

```bash
export COSMOS3_DIFFUSERS_VENV=/path/to/.venv-cosmos3-diffusers
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
    global COSMOS3_TRANSFER_ROOT
    global COSMOS3_DIFFUSERS_VENV
    global COSMOS3_TORCH_BACKEND
    global COSMOS3_TRANSFER_OUTPUT_ROOT

    COSMOS_ROOT = find_repo_root(Path.cwd().resolve())
    COSMOS3_TRANSFER_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3" / "generator" / "transfer"
    COSMOS3_DIFFUSERS_VENV = Path(
        os.environ.get("COSMOS3_DIFFUSERS_VENV", COSMOS_ROOT / ".venv-cosmos3-diffusers")
    ).resolve()
    COSMOS3_TORCH_BACKEND = os.environ.get("COSMOS3_TORCH_BACKEND", "cu130")
    COSMOS3_TRANSFER_OUTPUT_ROOT = Path(
        os.environ.get(
            "COSMOS3_TRANSFER_OUTPUT_ROOT", COSMOS3_TRANSFER_ROOT / "outputs" / "notebooks" / "diffusers"
        )
    ).resolve()

    os.environ["COSMOS3_DIFFUSERS_VENV"] = str(COSMOS3_DIFFUSERS_VENV)
    os.environ["COSMOS3_TORCH_BACKEND"] = COSMOS3_TORCH_BACKEND
    # preview_helpers.py resolves specs and outputs from these two.
    os.environ["COSMOS3_TRANSFER_ROOT"] = str(COSMOS3_TRANSFER_ROOT)
    os.environ["COSMOS3_TRANSFER_OUTPUT_ROOT"] = str(COSMOS3_TRANSFER_OUTPUT_ROOT)
    os.environ.setdefault("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv"))
    os.environ.setdefault("UV_LINK_MODE", "copy")
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("COSMOS3_DIFFUSERS_PRECISION", "bf16")

    print(f"COSMOS_ROOT: {COSMOS_ROOT}")
    for key in [
        "COSMOS3_DIFFUSERS_VENV",
        "COSMOS3_TORCH_BACKEND",
        "COSMOS3_TRANSFER_ROOT",
        "COSMOS3_TRANSFER_OUTPUT_ROOT",
        "UV_CACHE_DIR",
        "UV_LINK_MODE",
        "HF_HOME",
        "HF_HUB_DISABLE_XET",
        "CUDA_VISIBLE_DEVICES",
        "COSMOS3_DIFFUSERS_PRECISION",
    ]:
        print(f"{key}: {os.environ[key]}")
    print("HF_TOKEN:", "<set>" if os.environ.get("HF_TOKEN") else "<unset>")


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
uv venv "$COSMOS3_DIFFUSERS_VENV" --python 3.13 --seed --managed-python --allow-existing
source "$COSMOS3_DIFFUSERS_VENV/bin/activate"
uv pip install --torch-backend="$COSMOS3_TORCH_BACKEND" \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  accelerate \
  av \
  cosmos_guardrail \
  huggingface_hub \
  imageio \
  imageio-ffmpeg \
  ipykernel \
  torch \
  torchvision \
  transformers \
  "nvidia-modelopt==0.44.0"

"$COSMOS3_DIFFUSERS_VENV/bin/python" -m ipykernel install --user \
  --name cosmos3-diffusers \
  --display-name "Cosmos3 Diffusers (Python 3.13)"

echo
echo "Installed dependencies into: $COSMOS3_DIFFUSERS_VENV"
echo "Next: switch this notebook kernel to: Cosmos3 Diffusers (Python 3.13)"
echo "After switching kernels, run the Restore Environment cell below, then continue with Verify."
```

## 4. Select the Diffusers Kernel

The install cell creates and registers the `Cosmos3 Diffusers (Python 3.13)` Jupyter kernel.

**Note**: Switch this notebook to that kernel before running the remaining Python cells, then run the restore cell immediately below. It can take some time for the new Jupyter kernel to show up in the notebook interface.

```python
# Run this cell immediately after switching to the Cosmos3 Diffusers kernel.
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
    global COSMOS3_TRANSFER_ROOT
    global COSMOS3_DIFFUSERS_VENV
    global COSMOS3_TORCH_BACKEND
    global COSMOS3_TRANSFER_OUTPUT_ROOT

    COSMOS_ROOT = find_repo_root(Path.cwd().resolve())
    COSMOS3_TRANSFER_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3" / "generator" / "transfer"
    COSMOS3_DIFFUSERS_VENV = Path(
        os.environ.get("COSMOS3_DIFFUSERS_VENV", COSMOS_ROOT / ".venv-cosmos3-diffusers")
    ).resolve()
    COSMOS3_TORCH_BACKEND = os.environ.get("COSMOS3_TORCH_BACKEND", "cu130")
    COSMOS3_TRANSFER_OUTPUT_ROOT = Path(
        os.environ.get(
            "COSMOS3_TRANSFER_OUTPUT_ROOT", COSMOS3_TRANSFER_ROOT / "outputs" / "notebooks" / "diffusers"
        )
    ).resolve()

    os.environ["COSMOS3_DIFFUSERS_VENV"] = str(COSMOS3_DIFFUSERS_VENV)
    os.environ["COSMOS3_TORCH_BACKEND"] = COSMOS3_TORCH_BACKEND
    # preview_helpers.py resolves specs and outputs from these two.
    os.environ["COSMOS3_TRANSFER_ROOT"] = str(COSMOS3_TRANSFER_ROOT)
    os.environ["COSMOS3_TRANSFER_OUTPUT_ROOT"] = str(COSMOS3_TRANSFER_OUTPUT_ROOT)
    os.environ.setdefault("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv"))
    os.environ.setdefault("UV_LINK_MODE", "copy")
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("COSMOS3_DIFFUSERS_PRECISION", "bf16")

    print(f"COSMOS_ROOT: {COSMOS_ROOT}")
    for key in [
        "COSMOS3_DIFFUSERS_VENV",
        "COSMOS3_TORCH_BACKEND",
        "COSMOS3_TRANSFER_ROOT",
        "COSMOS3_TRANSFER_OUTPUT_ROOT",
        "UV_CACHE_DIR",
        "UV_LINK_MODE",
        "HF_HOME",
        "HF_HUB_DISABLE_XET",
        "CUDA_VISIBLE_DEVICES",
        "COSMOS3_DIFFUSERS_PRECISION",
    ]:
        print(f"{key}: {os.environ[key]}")
    print("HF_TOKEN:", "<set>" if os.environ.get("HF_TOKEN") else "<unset>")


configure_diffusers_environment()
```

## 5. Verify GPU and Python Environment

```python
import os
import sys
from pathlib import Path

if "COSMOS3_DIFFUSERS_VENV" not in os.environ:
    raise RuntimeError("Run the Restore Environment cell after switching to the Diffusers kernel.")

expected_python = (Path(os.environ["COSMOS3_DIFFUSERS_VENV"]) / "bin" / "python").resolve()
current_python = Path(sys.executable).resolve()
print("kernel python:", current_python)
print("expected python:", expected_python)
if current_python != expected_python:
    raise RuntimeError(
        "This notebook is not running inside the Diffusers venv. "
        "Switch the notebook kernel to 'Cosmos3 Diffusers (Python 3.13)', then run the Restore Environment cell above."
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

## 6. Preview Available Controls

The sampling values printed below are read from the specs and are what the run cells use.

```python
import json
import sys
from IPython.display import Video, display

if str(COSMOS3_TRANSFER_ROOT) not in sys.path:
    sys.path.insert(0, str(COSMOS3_TRANSFER_ROOT))
from preview_helpers import TRANSFER_CONTROLS, load_transfer_spec, make_preview, resolve_spec_path

for control in TRANSFER_CONTROLS:
    spec = load_transfer_spec(control)
    control_path = resolve_spec_path(spec[control]["control_path"])
    prompt = json.loads(resolve_spec_path(spec["prompt_path"]).read_text())
    caption = (
        prompt.get("temporal_caption")
        or prompt.get("comprehensive_t2i_caption")
        or prompt.get("extra", {}).get("prompt", "")
    )
    print(f"{control}: {control_path.relative_to(COSMOS_ROOT)}")
    print(f"  resolution={spec['resolution']} aspect_ratio={spec['aspect_ratio']} frames={spec['num_frames']} fps={spec['fps']}")
    print(f"  steps={spec['num_steps']} guidance={spec['guidance']} control_guidance={spec['control_guidance']} shift={spec['shift']} seed={spec['seed']}")
    print(f"  prompt={caption[:180]}{'...' if len(caption) > 180 else ''}")
    display(Video(str(make_preview(control_path)), embed=True))
```

## 7. Define the Transfer Runner

`run_transfer` reads a spec, hands the control video to `Cosmos3OmniModularPipeline` as `control_videos={hint: frames}`, and writes `vision.mp4` under `outputs/notebooks/diffusers/<model>/<spec name>/`, which is where `preview_transfer` from [preview_helpers.py](../scripts/preview_helpers.py) reads it back.

`guidance_scale` is the usual text CFG and `control_guidance` amplifies the control signal on top of it. Both come from the spec, as do the step count, flow shift, seed, frame count, and FPS.

```python
import gc
import json
import os
import sys
import time
from pathlib import Path

if "COSMOS3_DIFFUSERS_VENV" not in os.environ:
    raise RuntimeError("Run the Restore Environment cell after switching to the Diffusers kernel.")
expected_python = (Path(os.environ["COSMOS3_DIFFUSERS_VENV"]) / "bin" / "python").resolve()
if Path(sys.executable).resolve() != expected_python:
    raise RuntimeError("Switch the notebook kernel to 'Cosmos3 Diffusers (Python 3.13)' before running Diffusers cells.")

import torch
from diffusers import Cosmos3OmniModularPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

if str(COSMOS3_TRANSFER_ROOT) not in sys.path:
    sys.path.insert(0, str(COSMOS3_TRANSFER_ROOT))
from preview_helpers import TRANSFER_CONTROLS, load_transfer_spec, preview_transfer, resolve_spec_path

MODEL_IDS = {
    "Cosmos3-Nano": "nvidia/Cosmos3-Nano",
    "Cosmos3-Super": "nvidia/Cosmos3-Super",
}
GUARDRAIL_DISABLED_CONTROLS = frozenset({"edge", "depth"})

_pipe = None
_pipe_model = None


def compact_json_file(path: Path) -> str:
    return json.dumps(json.loads(path.read_text()), ensure_ascii=True, separators=(",", ":"))


def spec_dimensions(spec: dict) -> tuple[int, int]:
    """Height and width from the spec's `resolution` (the height) and `aspect_ratio`."""
    height = int(spec["resolution"])
    width_ratio, height_ratio = (int(part) for part in spec["aspect_ratio"].split(",", 1))
    return height, round(height * width_ratio / height_ratio)


def cuda_allocated_gib() -> float:
    return torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0


def release_pipe() -> None:
    """Free the loaded pipeline's GPU memory. Modular pipelines hold their components as
    attributes, so dropping the pipeline alone leaves them resident."""
    global _pipe, _pipe_model
    if _pipe is None:
        return
    pipe, _pipe, _pipe_model = _pipe, None, None
    for name in list(getattr(pipe, "components", None) or {}):
        module = getattr(pipe, name, None)
        if hasattr(module, "to"):
            try:
                module.to("cpu")
            except Exception:
                pass
        try:
            delattr(pipe, name)
        except Exception:
            pass
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
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



def materialize_checkpoint_dir(pretrained_source, token) -> Path:
    """Return a local checkpoint root (download from Hub when given a repo id)."""
    path = Path(str(pretrained_source))
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(pretrained_source), token=token)).resolve()


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


def get_pipe(model: str) -> Cosmos3OmniModularPipeline:
    global _pipe, _pipe_model
    prec = precision_mode()
    pretrained_source = resolve_pretrained_source(model)
    cache_key = f"{pretrained_source}@{prec}"
    if _pipe is not None and _pipe_model == cache_key:
        return _pipe
    release_pipe()
    t0 = time.time()
    fp8 = is_fp8_precision()
    token = os.environ.get("HF_TOKEN") or None
    if fp8:
        # ModelOpt FP8 graph restore + kernels (required before from_pretrained).
        import modelopt.torch.quantization.backends.fp8_per_tensor_gemm  # noqa: F401
        from modelopt.torch.opt import enable_huggingface_checkpointing

        enable_huggingface_checkpointing()
        print(f"loading FP8 {pretrained_source}...")
    else:
        print(f"loading {pretrained_source}...")
    pipe = Cosmos3OmniModularPipeline.from_pretrained(
        pretrained_source,
        torch_dtype=torch.bfloat16,
        token=token,
    )
    if fp8:
        # Pre-load the transformer from its full path so ModelOpt 0.44 finds
        # transformer/modelopt_state.pth. Modular load_components keeps subfolder
        # separate, which ModelOpt ignores.
        from diffusers import AutoModel

        ckpt_dir = materialize_checkpoint_dir(pretrained_source, token)
        transformer_dir = ckpt_dir / "transformer"
        if not (transformer_dir / "modelopt_state.pth").is_file():
            raise RuntimeError(
                f"FP8 requested but no ModelOpt state at {transformer_dir / 'modelopt_state.pth'}. "
                "Point COSMOS3_FP8_MODEL_PATH at a pre-quantized checkpoint directory."
            )
        transformer = AutoModel.from_pretrained(str(transformer_dir), torch_dtype=torch.bfloat16)
        pipe.update_components(transformer=transformer)
        # names=None loads only components that are still unset, so the preloaded transformer stays.
        # Remaining components keep the sources in modular_model_index.json. The FP8 repo's root
        # modelopt_state.pth belongs to the Reasoner and must not be applied to the VAE or audio tokenizer.
        pipe.load_components(torch_dtype=torch.bfloat16, token=token)
    else:
        pipe.load_components(torch_dtype=torch.bfloat16)
    if fp8:
        fp8_weights, quantizers, meta = validate_fp8_transformer(pipe.transformer)
        print(f"FP8 state restored: weights={fp8_weights}, quantizers={quantizers}, meta={meta}")
    pipe.to("cuda")
    _pipe, _pipe_model = pipe, cache_key
    print(f"loaded pipeline in {time.time() - t0:.1f}s (precision={prec}); cuda allocated {cuda_allocated_gib():.1f} GiB")
    return _pipe


def run_transfer(control: str, *, model: str) -> Path:
    """Run one control modality with the sampling values from `specs/<control>.json`."""
    spec = load_transfer_spec(control)
    height, width = spec_dimensions(spec)
    control_path = resolve_spec_path(spec[control]["control_path"])
    output_dir = Path(os.environ["COSMOS3_TRANSFER_OUTPUT_ROOT"]) / model / spec["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vision.mp4"

    pipe = get_pipe(model)
    guardrails_enabled = control not in GUARDRAIL_DISABLED_CONTROLS
    if guardrails_enabled:
        pipe.enable_safety_checker()
    else:
        pipe.disable_safety_checker()
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=spec["shift"], use_karras_sigmas=False
    )
    generator = torch.Generator(device="cuda").manual_seed(spec["seed"])

    print(f"control: {control} ({control_path.relative_to(COSMOS_ROOT)})")
    print(f"size:    {width}x{height} frames: {spec['num_frames']} fps: {spec['fps']}")
    print(f"guardrails: {guardrails_enabled}")
    print(f"steps:   {spec['num_steps']} guidance: {spec['guidance']} control_guidance: {spec['control_guidance']}")
    print(f"output:  {output_path}")
    t0 = time.time()
    videos = pipe(
        prompt=compact_json_file(resolve_spec_path(spec["prompt_path"])),
        negative_prompt=compact_json_file(resolve_spec_path(spec["negative_prompt_file"])),
        control_videos={control: load_video(str(control_path))},
        control_guidance=spec["control_guidance"],
        num_frames=spec["num_frames"],
        num_video_frames_per_chunk=spec["num_video_frames_per_chunk"],
        num_conditional_frames=spec["num_conditional_frames"],
        height=height,
        width=width,
        fps=float(spec["fps"]),
        num_inference_steps=spec["num_steps"],
        guidance_scale=spec["guidance"],
        generator=generator,
        output="videos",
    )
    print(f"generated in {time.time() - t0:.1f}s")
    export_to_video(videos, str(output_path), fps=spec["fps"], macro_block_size=1)
    print(f"wrote {output_path}")
    return output_path
```

## Quantized Checkpoints (FP8)

Default loads are BF16 (`COSMOS3_DIFFUSERS_PRECISION=bf16`). Set the env var to `fp8` before running the helper cell's `get_pipe` (or re-run that cell after changing it). Use-case request cells stay the same.

Nano and Super load `nvidia/Cosmos3-Nano-FP8` and `nvidia/Cosmos3-Super-FP8`. Do not pass `revision=fp8` on the BF16 cards.

Diffusers follows the checkpoint schedule (W8A16 on the first and last few steps, W8A8 in between). See the [Cosmos 3 Diffusers docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cosmos3).

Optional: point `COSMOS3_FP8_MODEL_PATH` at a local Diffusers checkpoint directory. Transfer preloads `transformer/` so ModelOpt 0.44 restores `modelopt_state.pth`.

```python
# Default is bf16. Uncomment to load FP8 checkpoints in the cells below.
# os.environ["COSMOS3_DIFFUSERS_PRECISION"] = "fp8"
print("COSMOS3_DIFFUSERS_PRECISION:", os.environ.get("COSMOS3_DIFFUSERS_PRECISION", "bf16"))
```

## Use Cases

Run each control top-to-bottom: generate, then view the control video next to the result. All five Cosmos3-Nano controls come first, then the same five on Cosmos3-Super.

## Nano: Edge (Canny) Transfer

Run the `edge` spec on Cosmos3-Nano, then display the input control video and generated output.

```python
edge_nano_output = run_transfer("edge", model="Cosmos3-Nano")
```

```python
preview_transfer("edge", model="Cosmos3-Nano")
```

## Nano: Blur Transfer

Run the `blur` spec on Cosmos3-Nano, then display the input control video and generated output.

```python
blur_nano_output = run_transfer("blur", model="Cosmos3-Nano")
```

```python
preview_transfer("blur", model="Cosmos3-Nano")
```

## Nano: Depth Transfer

Run the `depth` spec on Cosmos3-Nano, then display the input control video and generated output.

```python
depth_nano_output = run_transfer("depth", model="Cosmos3-Nano")
```

```python
preview_transfer("depth", model="Cosmos3-Nano")
```

## Nano: Segmentation Transfer

Run the `seg` spec on Cosmos3-Nano, then display the input control video and generated output.

```python
seg_nano_output = run_transfer("seg", model="Cosmos3-Nano")
```

```python
preview_transfer("seg", model="Cosmos3-Nano")
```

## Nano: World Scenario Map Transfer

Run the `wsm` spec on Cosmos3-Nano, then display the input control video and generated output.

```python
wsm_nano_output = run_transfer("wsm", model="Cosmos3-Nano")
```

```python
preview_transfer("wsm", model="Cosmos3-Nano")
```

```python
release_pipe()
```

## Super: Edge (Canny) Transfer

Run the `edge` spec on Cosmos3-Super, then display the input control video and generated output.

```python
edge_super_output = run_transfer("edge", model="Cosmos3-Super")
```

```python
preview_transfer("edge", model="Cosmos3-Super")
```

## Super: Blur Transfer

Run the `blur` spec on Cosmos3-Super, then display the input control video and generated output.

```python
blur_super_output = run_transfer("blur", model="Cosmos3-Super")
```

```python
preview_transfer("blur", model="Cosmos3-Super")
```

## Super: Depth Transfer

Run the `depth` spec on Cosmos3-Super, then display the input control video and generated output.

```python
depth_super_output = run_transfer("depth", model="Cosmos3-Super")
```

```python
preview_transfer("depth", model="Cosmos3-Super")
```

## Super: Segmentation Transfer

Run the `seg` spec on Cosmos3-Super, then display the input control video and generated output.

```python
seg_super_output = run_transfer("seg", model="Cosmos3-Super")
```

```python
preview_transfer("seg", model="Cosmos3-Super")
```

## Super: World Scenario Map Transfer

Run the `wsm` spec on Cosmos3-Super, then display the input control video and generated output.

```python
wsm_super_output = run_transfer("wsm", model="Cosmos3-Super")
```

```python
preview_transfer("wsm", model="Cosmos3-Super")
```
