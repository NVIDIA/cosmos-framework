# Diffusers — distilled from the cookbook notebook

Scope: the in-process Diffusers backend — the README "Run with Diffusers" quickstart and walkthrough, then the notebook's environment, venv install, verification, pipeline loader (`get_pipe` / `release_pipe`), `run_diffusers_payload`, and every use case it runs with `Cosmos3OmniPipeline` and `Cosmos3DistilledModularPipeline`; everything shared with the other backend notebooks (assets, `ASSET_SETS`, sampling tables, `payload_dimensions`, `create_payload`, model ids, path/output/env-var conventions) is in `payloads.md` and is pointed to, not repeated.

## Quickstart (cookbook README)

README section "Run with Diffusers". Its relative links (`../../README.md#diffusers`, `./run_with_diffusers.ipynb`) are rewritten to the same files in this repository (`../cookbooks/cosmos3/README.md#diffusers`); the notebook link is dropped — this file replaces the notebook (payloads.md path-translation note).

### Quickstart

Set up the environment: [Diffusers setup](../../cookbooks/cosmos3/README.md#diffusers).
Run a text-to-video generation with `Cosmos3OmniPipeline`:

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

prompt = json.load(open("assets/prompts/text2video/robot_kitchen.json"))
negative = json.load(open("assets/negative_prompts/text2video/neg_prompt.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0)

result = pipe(
    prompt=json.dumps(prompt),
    negative_prompt=json.dumps(negative),
    image=None,
    num_frames=189,
    height=720,
    width=1280,
    fps=24,
    num_inference_steps=35,
    guidance_scale=6.0,
    enable_sound=False,
    add_resolution_template=False,
    add_duration_template=False,
    generator=torch.Generator(device="cuda").manual_seed(1234),
)
export_to_video(result.video, "/tmp/cosmos3_t2v_diffusers.mp4", fps=24)
```

To run **Cosmos3-Super** instead, load the larger checkpoint:
`Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Super", ...)`.

To run **Cosmos3-Edge** instead, load `nvidia/Cosmos3-Edge` and use its
single-GPU, no-audio settings: `height=480`, `width=832`, `num_frames=121`,
`num_inference_steps=20`, `guidance_scale=5.0`, and `flow_shift=8.0`.

### Notebook walkthrough

`run_with_diffusers.ipynb` is the full tutorial for
the Diffusers backend: it provisions a dedicated venv, then walks through
text-to-image, text-to-video, image-to-video, and video-to-video generation (with
and without audio) using `Cosmos3OmniPipeline`, including a dedicated Cosmos3-Edge
section with a 480p image-to-video example and preview.

## Cosmos3 Generator Audiovisual with Diffusers (notebook introduction)

SPDX licence header: see payloads.md § 2.1.

This notebook runs Cosmos3 audiovisual generation directly with `Cosmos3OmniPipeline`.

Run the Cosmos3-Nano, Cosmos3-Super, or Cosmos3-Edge examples independently. Cosmos3-Edge has no audio modules and uses its 480p generation settings. The notebook also includes Nano and Super video-to-video examples, while Super T2V/I2V examples run without audio. Each section loads the matching model explicitly.

## 1. Prerequisites

- Machine, Hugging Face model access, `uvx hf@latest auth login` / `HF_TOKEN`: see payloads.md § 2.2 (`run_with_diffusers.ipynb` § 1 entry).
- Guardrail prerequisite and the per-backend disable switches as this notebook words them: see payloads.md § 2.3 (`run_with_diffusers.ipynb` row).
- Headless servers and uv version blockquotes, in this notebook's wording: see payloads.md § 2.4, and the headless OpenCV swap in SKILL.md Variant 1 (skill note after the § 3 install block below).

## 2. Configure Paths and Environment

The defaults are relative to this `cosmos` checkout and use the CUDA 13 or 12.8 Torch backend depending on the CUDA version installed on your system (`cu130` or `cu128`):

```bash
export COSMOS3_DIFFUSERS_VENV=/path/to/.venv-cosmos3-diffusers
export COSMOS3_TORCH_BACKEND=cu130
export HF_HOME=/path/to/large/huggingface/cache
export UV_LINK_MODE=copy
export CUDA_VISIBLE_DEVICES=0
```

Path configuration (`COSMOS_ROOT`, `COSMOS3_AUDIOVISUAL_ROOT`, `OUT` = `COSMOS3_AUDIOVISUAL_OUTPUT_ROOT`): see payloads.md § 4.1, which replaces the notebook's `find_repo_root()` and output-root lines. The remainder of the notebook's configuration cell runs on top of it:

```python
from pathlib import Path
import os


def configure_diffusers_environment() -> None:
    global COSMOS3_DIFFUSERS_VENV
    global COSMOS3_TORCH_BACKEND

    COSMOS3_DIFFUSERS_VENV = Path(
        os.environ.get("COSMOS3_DIFFUSERS_VENV", COSMOS_ROOT / ".venv-cosmos3-diffusers")
    ).resolve()
    COSMOS3_TORCH_BACKEND = os.environ.get("COSMOS3_TORCH_BACKEND", "cu130")

    os.environ["COSMOS3_DIFFUSERS_VENV"] = str(COSMOS3_DIFFUSERS_VENV)
    os.environ["COSMOS3_TORCH_BACKEND"] = COSMOS3_TORCH_BACKEND
    os.environ.setdefault("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv"))
    os.environ.setdefault("UV_LINK_MODE", "copy")
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


configure_diffusers_environment()
```

Backend-only variables the cell sets (the notebook's echo loop printed these together with `COSMOS_ROOT`, `COSMOS3_AUDIOVISUAL_OUTPUT_ROOT`, `UV_CACHE_DIR`, `HF_HOME`, `CUDA_VISIBLE_DEVICES` and `HF_TOKEN: <set>`/`<unset>`; those shared variables are in payloads.md § 4.1 and § 4.4):

| Variable                 | Default in the cell                                                   | Documented example (export block above) |
| ------------------------ | --------------------------------------------------------------------- | --------------------------------------- |
| `COSMOS3_DIFFUSERS_VENV` | `COSMOS_ROOT / ".venv-cosmos3-diffusers"`, resolved and exported back | `/path/to/.venv-cosmos3-diffusers`      |
| `COSMOS3_TORCH_BACKEND`  | `"cu130"`, exported back                                              | `cu130` (`cu128` for CUDA 12.8)         |
| `UV_LINK_MODE`           | `"copy"` (`setdefault`)                                               | `copy`                                  |
| `HF_HUB_DISABLE_XET`     | `"1"` (`setdefault`)                                                  | —                                       |

## 3. Install Diffusers Dependencies

```bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
uv venv "$COSMOS3_DIFFUSERS_VENV" --python 3.13 --seed --managed-python --allow-existing --no-config
source "$COSMOS3_DIFFUSERS_VENV/bin/activate"
uv pip install --no-config --python "$COSMOS3_DIFFUSERS_VENV" --torch-backend="$COSMOS3_TORCH_BACKEND" \
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
  && uv pip uninstall --no-config --python "$COSMOS3_DIFFUSERS_VENV" opencv-python \
  && uv pip install --no-config --python "$COSMOS3_DIFFUSERS_VENV" opencv-python-headless==5.0.0.93

echo
echo "Installed dependencies into: $COSMOS3_DIFFUSERS_VENV"
```

Skill note (not in the notebook): the `--no-config` on the `uv venv` and `uv pip install` lines above, the `--python "$COSMOS3_DIFFUSERS_VENV"` on the three `uv pip` commands (next-but-one note), and the `&& uv pip uninstall … opencv-python && uv pip install … opencv-python-headless==5.0.0.93` continuation that closes the `uv pip install` command (next note), are skill additions (declared deviations from the notebook, like the `-e` lines in sglang.md § 2) — the notebook's lines carry neither, and every other line of the block is the notebook's verbatim. It is needed because this skill sits inside a cosmos-framework checkout whose `pyproject.toml` pins `[tool.uv] required-version = "==0.11.29"`, and uv applies that pin from any cwd under the checkout: without the flag, any other uv (0.13.0 in the eval image) exits 2 on both commands with `error: Required uv version ==0.11.29 does not match the running version …`. `--no-config` (or `UV_NO_CONFIG=1`, or running both commands from a cwd outside the checkout with an absolute `COSMOS3_DIFFUSERS_VENV`) keeps that pin, and the framework's other `[tool.uv]` settings, away from this standalone venv; `UV_*` environment variables such as the exported `UV_LINK_MODE` still apply. The presence check above (`command -v uv`) does not catch the pin, and the notebook's remedy `uv self update 0.11.29` refuses for a pip/apt/vendored uv (`Self-update is only available for uv binaries installed via the standalone installation scripts`). This standalone venv does not need 0.11.29 at all; SKILL.md Variant 1 carries the same two flags.

Skill note (not in the notebook): `Cosmos3OmniPipeline` imports `cosmos_guardrail` → `cv2` at import time (diffusers' `pipeline_cosmos3_omni.py` does so at module scope once the package is installed), also with `enable_safety_checker=False`, and `cosmos_guardrail` pulls the GUI `opencv-python` wheel, which needs libxcb/libGL. The setup guide's swap (TensorRT-LLM section) — `uv pip uninstall --no-config opencv-python && uv pip install --no-config opencv-python-headless==5.0.0.93` — is therefore folded into the block's `uv pip install` command above as its closing `&& …` continuation (a declared deviation, same mechanism as the `--no-config` of the previous note), so it cannot be skipped: with the swap as separate lines after the install, an agent that copies this block instead of SKILL.md's can run the import check first and hit the `libxcb.so.1` ImportError (eval run 2026-10-24). Then check `python -c "from diffusers import Cosmos3OmniPipeline; print('pipeline import OK')"` before any checkpoint download. The apt fallback of § 1 / payloads.md § 2.4 (`apt-get update && apt-get install -y libxcb1 libgl1 libglib2.0-0 ffmpeg`) needs root and, on minimal images, the `apt-get update` first.

Skill note (not in the notebook): the `--python "$COSMOS3_DIFFUSERS_VENV"` on the three `uv pip` commands of the block is a skill addition (declared deviation, same mechanism as the two notes above). The notebook runs the block in one kernel cell, where `source "$COSMOS3_DIFFUSERS_VENV/bin/activate"` makes the venv the target of the following `uv pip` lines. An agent that runs the lines in separate shell calls loses the export and the activation between them: in the eval run of 2026-10-26 the `uv pip install` of a later call went into the evaluator's own venv, and the agent's import check then ran against that interpreter. With the explicit target the command installs into this venv whether or not it is activated, and with `COSMOS3_DIFFUSERS_VENV` unset it stops (`error: No virtual environment found …`) instead of installing elsewhere. Run the block as one shell call; a line re-run on its own needs the `export COSMOS3_DIFFUSERS_VENV=…` line with it, and the import check (SKILL.md Variant 1) and the generation snippets of § 8.2 / § 8.4 below run with `"$COSMOS3_DIFFUSERS_VENV/bin/python"` for the same reason.

## 4. Select the Diffusers Kernel

Jupyter-only step (the install cell registered a `Cosmos3 Diffusers (Python 3.13)` kernel and the notebook switched to it, then re-ran a restore cell identical to the § 2 configuration cell).

## 5. Verify GPU and Python Environment

```python
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

Jupyter preview cell (caption print per prompt file, inline image display per image file). Its caption rule, image-extension set and the captions it prints: see payloads.md § 3.1 and § 3.5.

## 7. Define Asset Sets, Payload Helpers, Runner, and Viewer Helpers

Shared definitions, verbatim in payloads.md:

- `IMAGE_EXTENSIONS`, `asset_path`, `compact_json_file`, `resolve_payload_path`: § 9.1.
- `FIXED_SAMPLING`: § 6.1 (`run_with_diffusers.ipynb` table — differs from the other notebooks' table in `seed`, `1234` instead of `0`).
- `ASSET_SETS` with its two comments: § 8.7 (`run_with_diffusers.ipynb` block); matrix rows in §§ 8.1–8.4.
- `payload_dimensions` (this notebook's five pairs): § 7.
- `create_payload` (`backend="diffusers"`): § 9.2 (`run_with_diffusers.ipynb` block; differences from the other notebooks' versions are tabulated there).
- `MODEL_IDS`, `DISTILLED_MODELS`, `resolve_model_id`, `is_distilled_model`: § 5.

The Diffusers runner (verbatim; the kernel/venv guard that preceded the imports is covered by § 4, and the definitions above are pointed to where they sat in the cell):

```python
import json
import gc
import os
import time
from pathlib import Path

import sys

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers import logging as diffusers_logging
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import encode_video, export_to_video, load_image, load_video

try:
    from diffusers import Cosmos3DistilledModularPipeline
except ImportError:
    Cosmos3DistilledModularPipeline = None

# MODEL_IDS and DISTILLED_MODELS: see payloads.md § 5

_pipe = None
_pipe_model = None


# resolve_model_id and is_distilled_model: see payloads.md § 5


def cuda_allocated_gib() -> float:
    return torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0


def release_pipe() -> None:
    """Drop the cached pipeline and free its CUDA memory.

    Modular pipelines expose their models as plain attributes rather than owning a container, so
    each one is moved off the device and unbound explicitly; dropping the pipeline alone leaves
    the weights resident and the next load then runs out of memory.
    """
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


def get_pipe(model: str):
    global _pipe, _pipe_model
    model_id = resolve_model_id(model)
    if _pipe is not None and _pipe_model == model_id:
        return _pipe
    release_pipe()
    diffusers_logging.set_verbosity_info()
    print(f"loading {model_id}...")
    t0 = time.time()
    if is_distilled_model(model):
        if Cosmos3DistilledModularPipeline is None:
            raise RuntimeError(
                "This diffusers build does not provide Cosmos3DistilledModularPipeline. "
                "Reinstall diffusers from a revision that includes the Cosmos3 distilled pipeline."
            )
        pipe = Cosmos3DistilledModularPipeline.from_pretrained(
            model_id,
            token=os.environ.get("HF_TOKEN") or None,
        )
        pipe.load_components(torch_dtype=torch.bfloat16)
        pipe.enable_safety_checker()
    else:
        pipe = Cosmos3OmniPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            safety_checker=None,
            enable_safety_checker=True,
            token=os.environ.get("HF_TOKEN") or None,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=FIXED_SAMPLING["shift"])
    pipe.to("cuda")
    _pipe = pipe
    _pipe_model = model_id
    print(f"loaded pipeline in {time.time() - t0:.1f}s")
    return _pipe


def run_diffusers_payload(payload_path: Path, output_dir: str | Path, *, model: str) -> Path:
    payload_path = Path(payload_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(payload_path.read_text())
    height, width = payload_dimensions(payload)
    distilled = is_distilled_model(model)
    pipe = get_pipe(model)
    if not distilled:
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=payload["shift"])
    generator = torch.Generator(device="cuda").manual_seed(payload["seed"])
    vision_path = resolve_payload_path(payload_path, payload["vision_path"]) if "vision_path" in payload else None
    image = load_image(str(vision_path)) if payload["model_mode"] == "image2video" else None
    video = load_video(str(vision_path)) if payload["model_mode"] == "video2video" else None
    output_ext = ".png" if payload["model_mode"] == "text2image" else ".mp4"
    output_path = output_dir / f"{payload['name']}{output_ext}"

    print(f"generating {payload['model_mode']} {payload['name']} with {model} -> {output_path}")
    t0 = time.time()
    if distilled:
        # The distilled pipeline rejects num_inference_steps and guidance_scale overrides and
        # ignores negative_prompt, so none of them are forwarded.
        call_kwargs = {
            "prompt": payload["prompt"],
            "num_frames": payload["num_frames"],
            "height": height,
            "width": width,
            "generator": generator,
            "add_resolution_template": False,
            "add_duration_template": False,
            "output": "videos",
        }
        if payload["model_mode"] == "image2video":
            call_kwargs["image"] = image
            call_kwargs["fps"] = payload["fps"]
        videos = pipe(**call_kwargs)
        print(f"generated in {time.time() - t0:.1f}s")
        if payload["model_mode"] == "text2image":
            videos[0].save(output_path)
        else:
            export_to_video(videos, str(output_path), fps=payload["fps"], macro_block_size=1)
        print(f"wrote {output_path}")
        return output_path

    if payload["model_mode"] == "text2image":
        result = pipe(
            prompt=payload["prompt"],
            negative_prompt="",
            num_frames=1,
            height=height,
            width=width,
            num_inference_steps=payload["num_steps"],
            guidance_scale=payload["guidance"],
            add_resolution_template=False,
            add_duration_template=False,
            generator=generator,
        )
        print(f"generated in {time.time() - t0:.1f}s")
        result.video[0].save(output_path)
        print(f"wrote {output_path}")
        return output_path

    result = pipe(
        prompt=payload["prompt"],
        negative_prompt=payload["negative_prompt"],
        image=image,
        video=video,
        condition_frame_indexes_vision=payload.get("condition_frame_indexes_vision", [0, 1]),
        condition_video_keep=payload.get("condition_video_keep", "first"),
        num_frames=payload["num_frames"],
        height=height,
        width=width,
        fps=payload["fps"],
        num_inference_steps=payload["num_steps"],
        guidance_scale=payload["guidance"],
        enable_sound=payload["enable_sound"],
        add_resolution_template=False,
        add_duration_template=False,
        generator=generator,
    )
    print(f"generated in {time.time() - t0:.1f}s")
    if payload["enable_sound"] and result.sound is not None:
        encode_video(
            result.video,
            fps=payload["fps"],
            output_path=str(output_path),
            audio=result.sound,
            audio_sample_rate=pipe.sound_tokenizer.config.sampling_rate,
        )
    else:
        export_to_video(result.video, str(output_path), fps=payload["fps"], macro_block_size=1)
    print(f"wrote {output_path}")
    return output_path
```

Viewer helpers (`display_video`, `view_run`) are Jupyter-only and are not migrated; `view_run`'s media discovery rule is in payloads.md § 4.2 (`view_run` discovery row). How each payload key reaches the pipeline call: payloads.md § 10 (diffusers column).

## Use Cases

Workflow wording and the three-step pattern: see payloads.md § 11 (`run_with_diffusers.ipynb` entry). Every section below the notebook's "Use Cases" heading ran the same cell triplet, "Create Payload" / "Run" / "View Results":

```python
<use_case>_payload, <use_case>_output, <use_case>_model = create_payload("<use_case>", backend="diffusers")
run_diffusers_payload(<use_case>_payload, <use_case>_output, model="<model>")
```

followed by `view_run(<use_case>_output)` (Jupyter viewer, not migrated). `backend="diffusers"` puts the payload at `<OUT>/diffusers/payloads/<use_case>/<use_case>.json`, the media under `<OUT>/diffusers/<use_case>/` (the use-case key is `payload["name"]`; media file rule in payloads.md § 4.2) and exports `COSMOS3_DIFFUSERS_<USE_CASE>_INPUT` / `_OUTPUT` (payloads.md § 4.2–4.3). "Sampling override" is the spec's `sampling` dict, which `create_payload` spreads on top of this notebook's `FIXED_SAMPLING` (payloads.md § 6.1; `create_payload` in § 9.2); "—" means the base table applies unchanged. What `create_payload` adds to or removes from the payload per mode and for distilled specs, and the per-mode negative prompt, are in payloads.md § 9.2 and § 3.3. Rows are in the notebook's section order; `assets/videos/car_driving_plain.mp4` is not vendored in this skill.

| #   | Notebook section — description                                                                                                                                                                                              | Use case            | `model=`                          | Mode          | Prompt asset                                               | Image / video asset                                                | `enable_sound` | Sampling override                                                                                                                        | Other spec fields                                                          |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | --------------------------------- | ------------- | ---------------------------------------------------------- | ------------------------------------------------------------------ | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 1   | Nano: Text to Image — Nano text-to-image generation using a structured JSON prompt.                                                                                                                                         | `t2i`               | `Cosmos3-Nano`                    | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                                  | `False`        | —                                                                                                                                        | —                                                                          |
| 2   | Super: Text to Image — Super text-to-image generation using the same structured JSON prompt.                                                                                                                                | `t2i_super`         | `Cosmos3-Super`                   | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                                  | `False`        | —                                                                                                                                        | —                                                                          |
| 3   | Nano: Text to Video Without Audio — Nano text-to-video generation with audio disabled.                                                                                                                                      | `t2v_nano_noaudio`  | `Cosmos3-Nano`                    | `text2video`  | `assets/prompts/text2video/robot_kitchen.json`             | —                                                                  | `False`        | —                                                                                                                                        | —                                                                          |
| 4   | Nano: Text to Video with Audio — Nano text-to-video generation with generated audio.                                                                                                                                        | `t2vs`              | `Cosmos3-Nano`                    | `text2video`  | `assets/prompts/text2video/robot_pouring_water_audio.json` | —                                                                  | `True`         | —                                                                                                                                        | —                                                                          |
| 5   | Nano: Image to Video Without Audio — Nano image-to-video generation using its paired image asset, with audio disabled.                                                                                                      | `i2v_nano_noaudio`  | `Cosmos3-Nano`                    | `image2video` | `assets/prompts/image2video/car_driving.json`              | `assets/images/image2video/car_driving.jpg`                        | `False`        | —                                                                                                                                        | —                                                                          |
| 6   | Nano: Image to Video with Audio — Nano image-to-video generation using its paired image asset and generated audio.                                                                                                          | `i2vs`              | `Cosmos3-Nano`                    | `image2video` | `assets/prompts/image2video/coastal_road_audio.json`       | `assets/images/image2video/coastal_road_audio.jpg`                 | `True`         | —                                                                                                                                        | —                                                                          |
| 7   | Nano: Video to Video Without Audio — Nano video-to-video generation from a reference clip, with audio disabled. The first two latent frames of the reference clip are kept as clean conditioning and the rest are denoised. | `v2v_nano_noaudio`  | `Cosmos3-Nano`                    | `video2video` | `assets/prompts/image2video/car_driving.json`              | `assets/videos/car_driving_plain.mp4` (not vendored in this skill) | `False`        | —                                                                                                                                        | `"negative_prompt": "assets/negative_prompts/image2video/neg_prompt.json"` |
| 8   | Nano: Video to Video with Audio — Nano video-to-video generation from a reference clip, with generated audio. The first two latent frames of the reference clip are kept as clean conditioning and the rest are denoised.   | `v2vs`              | `Cosmos3-Nano`                    | `video2video` | `assets/prompts/image2video/car_driving.json`              | `assets/videos/car_driving_plain.mp4` (not vendored in this skill) | `True`         | —                                                                                                                                        | `"negative_prompt": "assets/negative_prompts/image2video/neg_prompt.json"` |
| 9   | Super: Text to Video Without Audio — Super text-to-video generation with audio disabled.                                                                                                                                    | `t2v_super_noaudio` | `Cosmos3-Super`                   | `text2video`  | `assets/prompts/text2video/robot_kitchen.json`             | —                                                                  | `False`        | —                                                                                                                                        | —                                                                          |
| 10  | Super: Image to Video Without Audio — Super image-to-video generation using its paired image asset, with audio disabled.                                                                                                    | `i2v_super_noaudio` | `Cosmos3-Super`                   | `image2video` | `assets/prompts/image2video/car_driving.json`              | `assets/images/image2video/car_driving.jpg`                        | `False`        | —                                                                                                                                        | —                                                                          |
| 11  | Super 4-Step: Text to Image — Distilled text-to-image with `Cosmos3-Super-Text2Image-4Step`, served by `Cosmos3DistilledModularPipeline`.                                                                                   | `t2i_distilled`     | `Cosmos3-Super-Text2Image-4Step`  | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                                  | `False`        | `{"fps": 30, "num_frames": 1, "resolution": "768", "aspect_ratio": "1,1", "seed": 1}`                                                    | `"distilled": True`                                                        |
| 12  | Super 4-Step: Image to Video — Distilled image-to-video with `Cosmos3-Super-Image2Video-4Step`, served by `Cosmos3DistilledModularPipeline`.                                                                                | `i2v_distilled`     | `Cosmos3-Super-Image2Video-4Step` | `image2video` | `assets/prompts/image2video/car_driving.json`              | `assets/images/image2video/car_driving.jpg`                        | `False`        | `{"fps": 30, "num_frames": 121, "resolution": "480", "aspect_ratio": "4,3", "seed": 1}`                                                  | `"distilled": True`                                                        |
| 13  | Edge: Image to Video Without Audio (under "Cosmos3-Edge Example", below) — Animate the supplied driving image for 121 frames at 832x480.                                                                                    | `i2v_edge_noaudio`  | `Cosmos3-Edge`                    | `image2video` | `assets/prompts/image2video/car_driving_edge.json`         | `assets/images/image2video/car_driving.jpg`                        | `False`        | `{"num_steps": 20, "guidance": 5.0, "shift": 8.0, "fps": 24, "num_frames": 121, "resolution": "480", "aspect_ratio": "16,9", "seed": 0}` | —                                                                          |
| 14  | Super: Video to Video Without Audio — Super video-to-video generation using the same reference clip and conditioning frames as Nano, with audio disabled.                                                                   | `v2v_super_noaudio` | `Cosmos3-Super`                   | `video2video` | `assets/prompts/image2video/car_driving.json`              | `assets/videos/car_driving_plain.mp4` (not vendored in this skill) | `False`        | —                                                                                                                                        | `"negative_prompt": "assets/negative_prompts/image2video/neg_prompt.json"` |
| 15  | Super: Video to Video with Audio — Super video-to-video generation on the same reference clip and conditioning frames as the Nano case, with generated audio.                                                               | `v2vs_super`        | `Cosmos3-Super`                   | `video2video` | `assets/prompts/image2video/car_driving.json`              | `assets/videos/car_driving_plain.mp4` (not vendored in this skill) | `True`         | —                                                                                                                                        | `"negative_prompt": "assets/negative_prompts/image2video/neg_prompt.json"` |

## Cosmos3-Edge Example

Cosmos3-Edge runs on a single GPU and has no audio tokenizer, so the Edge video example keeps audio disabled. The Edge checkpoint is trained for 480p generation; this example uses 832x480, 121 frames, 20 denoising steps, guidance scale 5.0, and flow shift 8.0.

Its single use case, `i2v_edge_noaudio`, is row 13 of the table above (notebook sub-heading "Edge: Image to Video Without Audio").

## 8. Skill notes (not in the notebook) — headless fallback, snippets and Edge settings (from SKILL.md Variant 1)

Skill-authored text moved here from SKILL.md on 2026-10-27 (layout change only; wording unchanged apart from pointers and lead-ins that had to follow the text). Nothing in this section is notebook text; SKILL.md keeps a one-line pointer at each former location.

### 8.1 Headless servers — the apt fallback (SKILL.md Variant 1)

> **Headless servers** (notebook § 1, verbatim in payloads.md § 2.4; the apt line below is
> reworded from the setup guide's Framework section — `apt-get update`, `ffmpeg` and the root
> caveat are not the notebook's): if you see an error like `libxcb.so.1: cannot open
> shared object file` when importing or running the pipeline, install the required
> system libraries — the fallback for hosts that want GUI OpenCV:
> `apt-get update && apt-get install -y libxcb1 libgl1 libglib2.0-0 ffmpeg` (needs root:
> NGC and training containers run as root, otherwise prefix `sudo`; without root use the
> headless swap above, which needs no system packages. `ffmpeg` is for `ffprobe` — see
> SKILL.md Verify the output; the swap does not provide it, the venv's `av` does). Setup guide
> (TensorRT-LLM section): `pip uninstall -y opencv-python && pip install opencv-python-headless==5.0.0.93`.

### 8.2 Skill launch script (not in the notebook) — Nano text-to-image snippet (SKILL.md Variant 1)

Text-to-image as the notebook runs it (`get_pipe` + the `text2image` branch of
`run_diffusers_payload`; `FIXED_SAMPLING`: 35 steps, guidance 6.0, shift 10.0, 720 × 1280,
seed 1234; `get_pipe` loads with `enable_safety_checker=True` — `False` is the § 1 switch; output path per § 4.2).
Save the snippet as a file and run it with the venv's interpreter, `"$COSMOS3_DIFFUSERS_VENV/bin/python" <file>.py`,
from this skill directory (the one with `assets/`) — `python` alone is the venv's only in a shell that activated it:

```python
import json, os, torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

pipe = Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16,
                                           safety_checker=None, enable_safety_checker=False)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0)
pipe.to("cuda")
prompt = json.dumps(json.load(open("assets/prompts/text2image/robot_draping.json")),   # run from this skill directory (the one with assets/)
                    ensure_ascii=True, separators=(",", ":"))
result = pipe(prompt=prompt, negative_prompt="", num_frames=1, height=720, width=1280,
              num_inference_steps=35, guidance_scale=6.0,
              add_resolution_template=False, add_duration_template=False,
              generator=torch.Generator(device="cuda").manual_seed(1234))
os.makedirs("outputs/diffusers/t2i", exist_ok=True)   # output_dir.mkdir(parents=True, exist_ok=True)
result.video[0].save("outputs/diffusers/t2i/t2i.png")
```

### 8.3 Edge settings and video modes, from the README (SKILL.md Variant 1)

README: "To run **Cosmos3-Edge** instead, load `nvidia/Cosmos3-Edge` and use its
single-GPU, no-audio settings: `height=480`, `width=832`, `num_frames=121`,
`num_inference_steps=20`, `guidance_scale=5.0`, and `flow_shift=8.0`" (`num_frames=1`
for text2image). Video modes pass the text2video prompt asset, `negative_prompt=json.dumps(<neg asset>)`,
`num_frames=189` (Nano/Super), `fps=24`, and `export_to_video(result.video, "<path>.mp4", fps=24)`,
then verify with ffprobe or the venv's `av` (see SKILL.md Verify the output).

### 8.4 Skill launch script (not in the notebook) — Edge text-to-image / text-to-video snippet (SKILL.md Variant 1)

Edge, composed from the README sentence above and the text2image / video branches of
`run_diffusers_payload` (diffusers.md § 7; the diffusers notebook has no `t2i_edge` /
`t2v_edge_noaudio` case, payloads.md § 8.2):

```python
import json, os, torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

compact = lambda p: json.dumps(json.load(open(p)), ensure_ascii=True, separators=(",", ":"))   # run from this skill directory (the one with assets/)
pipe = Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Edge", torch_dtype=torch.bfloat16,
                                           safety_checker=None, enable_safety_checker=False)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)
pipe.to("cuda")
gen = torch.Generator(device="cuda").manual_seed(1234)   # FIXED_SAMPLING seed, as in the Nano snippet
# t2i_edge
result = pipe(prompt=compact("assets/prompts/text2image/robot_draping.json"), negative_prompt="",
              num_frames=1, height=480, width=832, num_inference_steps=20, guidance_scale=5.0,
              add_resolution_template=False, add_duration_template=False, generator=gen)
os.makedirs("outputs/diffusers/t2i_edge", exist_ok=True)
result.video[0].save("outputs/diffusers/t2i_edge/t2i_edge.png")
# t2v_edge_noaudio
gen.manual_seed(1234)   # fresh seed per run — the notebook's run_diffusers_payload seeds a generator per call
result = pipe(prompt=compact("assets/prompts/text2video/robot_kitchen.json"),
              negative_prompt=compact("assets/negative_prompts/text2video/neg_prompt.json"),
              image=None, num_frames=121, height=480, width=832, fps=24,
              num_inference_steps=20, guidance_scale=5.0, enable_sound=False,
              add_resolution_template=False, add_duration_template=False, generator=gen)
os.makedirs("outputs/diffusers/t2v_edge_noaudio", exist_ok=True)
export_to_video(result.video, "outputs/diffusers/t2v_edge_noaudio/t2v_edge_noaudio.mp4", fps=24, macro_block_size=1)
```
