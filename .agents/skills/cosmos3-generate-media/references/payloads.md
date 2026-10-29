# Cosmos3 Generator Audiovisual — shared payload reference, distilled from the cookbook notebooks

Scope: everything the seven backend notebooks and the README have in common — the README introduction, the asset catalogue, guardrail/licensing/access notes, the `ASSET_SETS` use-case matrix, the sampling-default tables, the resolution/aspect-ratio → height × width mapping, `create_payload` and its helpers, model identifiers, and the env-var and output-naming conventions; per-backend server launch, request helpers and run sections live in the sibling backend files, which say "see payloads.md § …" for what is here.

Path translation (said once for the whole skill): the notebooks resolved assets under `<cosmos-framework checkout>/.agents/skills/cookbooks/cosmos3/generator/audiovisual/` and wrote payloads and media under `COSMOS3_AUDIOVISUAL_OUTPUT_ROOT` (default `<that folder>/outputs/notebooks`). This skill vendors the same `assets/` tree at its own root, so `COSMOS3_AUDIOVISUAL_ROOT` is the skill directory and a plain `OUT=outputs` replaces the output-root plumbing (`COSMOS3_AUDIOVISUAL_OUTPUT_ROOT` is set to `OUT` once, in § 4.1, so every helper below runs unchanged). Relative `../../README.md` links in README-derived text are rewritten to the same file at the pinned commit. The only asset the notebooks use that is not vendored is `assets/videos/car_driving_plain.mp4` (not vendored in this skill).

## 1. Introduction (cookbook README)

For Nano/Super static FP8 checkpoints, see the [quantization cookbook](../../cookbooks/cosmos3/quantization/README.md)
and the **Quantized Checkpoints** section in the vLLM-Omni notebooks (here: `vllm-omni.md` § 2).

Generate images and video (with optional audio) from text, image or video prompts with
`Cosmos3-Nano`, `Cosmos3-Super`, `Cosmos3-Edge`, and the published four-step
distilled Cosmos3-Super students across Cosmos Framework, Diffusers, vLLM-Omni, TRT-LLM
and NIM backends. Sample prompts live under [`assets/`](../assets).

> **NIM scope:** The Cosmos3 NIM serves text-to-image,
> text-to-video, image-to-video, and video-to-video, plus Action and Transfer.
> It does not generate audio.

Environment setup for every backend is centralized in the shared
[Cosmos3 cookbooks environment setup](../../cookbooks/cosmos3/README.md) guide; each backend below
links to the section you need. The quickstarts are minimal text-to-video examples
to get one generation running per backend — run them from this folder.

Generator requires the Guardrail. Request access to the gated
[nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail)
HF repository before running these examples. To disable the guardrail, set
`enable_safety_checker=False` (Diffusers), `TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1`
or `use_guardrails: false` through `extra_params` (TensorRT-LLM),
`guardrails: false` (vLLM-Omni `extra_params`/`extra_args`), or
`--no-guardrails` (Cosmos Framework). For the
Cosmos3 NIM, set `NIM_ENABLE_TEXT_GUARDRAILS=0 NIM_ENABLE_VIDEO_GUARDRAILS=0 NIM_ENABLE_SIGLIP_GUARDRAILS=0`.

NIM backends use NGC authentication instead of Hugging Face login; see the
[NIM setup](../../cookbooks/cosmos3/README.md#cosmos3-nim) for details.

## 2. Licensing and access

### 2.1 Licence header

Every notebook carries the same SPDX header:

```text
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1
```

### 2.2 Hugging Face / NGC access

- `run_with_cosmos_framework.ipynb` § 1: Use a Linux machine with NVIDIA GPU access, model access on Hugging Face, and either `uvx hf@latest auth login` or `HF_TOKEN` set. Use cache/output paths with enough disk space.
- `run_with_diffusers.ipynb` § 1: Use a Linux machine with NVIDIA GPU access, model access on Hugging Face, and either `uvx hf@latest auth login` or `HF_TOKEN` set.
- README (§ 1 above): NIM backends use NGC authentication instead of Hugging Face login.

### 2.3 Guardrail (applies to every backend)

The README paragraph in § 1 is the cross-backend statement (one disable switch per backend). The notebooks that carry their own guardrail prerequisite word it as follows:

| Notebook                                                                         | Guardrail prerequisite as written                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_with_diffusers.ipynb`                                                       | Generator requires the Guardrail. Request access to the gated [nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail) HF repository before running these examples. To disable the guardrail, set `enable_safety_checker=False` (Diffusers), `guardrails: false` (vLLM-Omni `extra_params`/`extra_args`), or `--no-guardrails` (Cosmos Framework).                                                                                                                                                                  |
| `run_with_vllm_omni.ipynb`                                                       | Generator requires the Guardrail. Request access to the gated [nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail) HF repository before running these examples. When running with guardrails enabled, also set and pass `HF_TOKEN` to the container; it must be a Hugging Face token authorized for that repository. To disable guardrails server-wide, pass `--no-guardrails` to `vllm serve`; alternatively, set `guardrails: false` in vLLM-Omni `extra_params` or `extra_args` to disable them per request. |
| `run_with_trt_llm.ipynb`                                                         | Generator requires the Guardrail. Request access to the gated [nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail) HF repository before running these examples. TensorRT-LLM loads guardrails by default; to disable them, set `TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1` before starting the server or set `use_guardrails` to `False` in the request `extra_params`.                                                                                                                                               |
| `run_distilled_with_trt_llm.ipynb`                                               | Same text as `run_with_trt_llm.ipynb`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `run_with_cosmos_framework.ipynb`, `run_with_sglang.ipynb`, `run_with_nim.ipynb` | No guardrail paragraph in their Prerequisites; the README paragraph in § 1 applies.                                                                                                                                                                                                                                                                                                                                                                                                                                                          |

Skill note (not in the notebook): for `run_with_sglang.ipynb` the server-side switch, which the README paragraph in § 1 does not name, is `SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` on `docker run` (`-e SGLANG_DISABLE_COSMOS3_GUARDRAILS=1`; eval-verified against `lmsysorg/sglang:dev`, not in the notebook/README) — without it the server loads the gated guardrail at startup and exits with `GatedRepoError 401`, and the per-request `guardrails: false` alone does not prevent that. Pass `-e HF_TOKEN` as well — anonymous Hub calls are rate-limited (HTTP 429; the run's first cold launch exited with an incomplete-checkpoint `ValueError`).

Request-level defaults as the shared helpers send them (details in § 6.3): vLLM-Omni and SGLang `guardrails: True`, TensorRT-LLM `use_guardrails: True`, Diffusers `enable_safety_checker=True` / `pipe.enable_safety_checker()`.

### 2.4 Headless servers and uv version (in-process notebooks)

`run_with_cosmos_framework.ipynb` § 1:

> **Headless servers:** if you see an error like `libxcb.so.1: cannot open shared object file` (a missing system graphics library) when importing or running the model, install the required system libraries:
>
> ```bash
> apt-get install -y libxcb1 libgl1 libglib2.0-0
> ```

Skill note (not in the notebook): on minimal images run `apt-get update` first (the eval image returned `E: Unable to locate package` without it); the root-free alternative is the headless OpenCV swap in SKILL.md Variant 1.

> **uv version:** this notebook uses `uv 0.11.29`, the exact version required by the framework's `pyproject.toml`. Install it from <https://astral.sh/uv/0.11.29/install.sh> or update an existing self-managed installation with `uv self update 0.11.29`.

Skill note (not in the notebook): `uv self update` works for a standalone-installer uv only — it refuses for pip/apt/vendored uv (`Self-update is only available for uv binaries installed via the standalone installation scripts`); then install 0.11.29 side-by-side with `curl -LsSf https://astral.sh/uv/0.11.29/install.sh | env UV_INSTALL_DIR=<dir> INSTALLER_NO_MODIFY_PATH=1 sh` and put `<dir>` first on PATH, or run `uvx --from 'uv==0.11.29' uv …`. SKILL.md Variant 2 has this as a preflight before `uv sync`.

`run_with_diffusers.ipynb` § 1 (differs only in wording):

> **Headless servers:** if you see an error like `libxcb.so.1: cannot open shared object file` (a missing system graphics library) when importing or running the pipeline, install the required system libraries:
>
> ```bash
> apt-get install -y libxcb1 libgl1 libglib2.0-0
> ```

Skill note (not in the notebook): on minimal images run `apt-get update` first (the eval image returned `E: Unable to locate package` without it); the root-free alternative is the headless OpenCV swap in SKILL.md Variant 1.

> **uv version:** use `uv 0.11.29`, matching the version pinned by Cosmos Framework. Install it from <https://astral.sh/uv/0.11.29/install.sh> or update an existing self-managed installation with `uv self update 0.11.29`.

Skill note (not in the notebook): `uv self update` works for a standalone-installer uv only — it refuses for pip/apt/vendored uv (`Self-update is only available for uv binaries installed via the standalone installation scripts`); then install 0.11.29 side-by-side with `curl -LsSf https://astral.sh/uv/0.11.29/install.sh | env UV_INSTALL_DIR=<dir> INSTALLER_NO_MODIFY_PATH=1 sh` and put `<dir>` first on PATH, or run `uvx --from 'uv==0.11.29' uv …`. The pin matters for `uv sync` inside the framework checkout (SKILL.md Variant 2); the standalone diffusers venv of SKILL.md Variant 1 keeps the stock uv and passes `--no-config` instead.

Skill note (not in the notebook): ffmpeg (for ffprobe) is in neither headless list above — add it to the same apt-get (`apt-get update && apt-get install -y libxcb1 libgl1 libglib2.0-0 ffmpeg`), or use the venv's `av` package (SKILL.md, Verify the output).

## 3. Asset catalogue

README: Sample prompts live under `assets/`. `ASSET_SETS` comment (`run_with_cosmos_framework.ipynb`, `run_with_diffusers.ipynb`, `run_with_vllm_omni.ipynb`, `run_with_sglang.ipynb`): "All asset paths are repo-relative under cookbooks/cosmos3/generator/audiovisual." — in this skill the same relative paths resolve against the skill directory (§ 4.1). Prompts are structured JSON in the Cosmos3 prompt schema; `create_payload` serialises each file compactly before it is sent (§ 9.1).

### 3.1 Prompt assets

The "Preview caption" column is what the notebooks' preview cell prints for each file: the first 180 characters of `temporal_caption`, else `comprehensive_t2i_caption`, else `extra.prompt`, followed by `...` when longer.

| Asset (skill-relative)                                     | Caption key                 | Used by                                                                                                                                                                                                                                                                                                                                                                                                                            | Preview caption                                                                                                                                                                         |
| ---------------------------------------------------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `assets/prompts/text2image/robot_draping.json`             | `comprehensive_t2i_caption` | Every text2image case: `t2i`, `t2i_super`, `t2i_edge`, `t2i_distilled`, `t2i_distil` (framework, diffusers, vllm_omni, sglang, distilled trt-llm) and `t2i_nano`, `t2i_super`, `t2i_edge` (trt_llm). Notebook wording: "a structured JSON prompt" / "the same structured JSON prompt".                                                                                                                                             | Photorealistic fashion-studio scene shot at eye level: a sleek white-and-aluminum robotic arm enters from the upper left, its precision two-finger gripper delicately pinching the e... |
| `assets/prompts/text2video/robot_kitchen.json`             | `temporal_caption`          | text2video without audio: `t2v_nano_noaudio`, `t2v_super_noaudio`, `t2v_edge_noaudio` (framework, diffusers, vllm_omni, sglang) and `t2v_nano`, `t2v_super`, `t2v_edge` (trt_llm).                                                                                                                                                                                                                                                 | The video opens with a robotic arm positioned above a dirty white plate on a kitchen countertop. In the first two seconds, the arm lowers its sponge-equipped gripper onto the plate... |
| `assets/prompts/text2video/robot_pouring_water_audio.json` | `temporal_caption`          | text2video with generated/synchronized audio: `t2vs` (framework, diffusers, vllm_omni, sglang) and `t2av_nano`, `t2av_super` (trt_llm).                                                                                                                                                                                                                                                                                            | The video opens with a silver robotic arm holding a clear glass jar of water above a white ceramic cup on a clean white table against a gray background. During the first two second... |
| `assets/prompts/image2video/car_driving.json`              | `temporal_caption`          | image2video without audio, paired with `car_driving.jpg`: `i2v_nano_noaudio`, `i2v_super_noaudio`, `i2v_distilled`, `i2v_distil`, `i2v_distil_noaudio`, and `i2v_edge_noaudio` in `run_with_sglang.ipynb` only; also the prompt of every video2video case (`v2v_nano_noaudio`, `v2vs`, `v2v_super_noaudio`, `v2vs_super`, `v2v_nano`, `v2v_super`). `run_distilled_with_trt_llm.ipynb`: "the paired car-driving image and prompt". | The video begins with a dashcam view of a car driving fast along a scenic coastal mountain road, the asphalt rushing beneath and a sharp left curve sign visible on the left. The ca... |
| `assets/prompts/image2video/car_driving_edge.json`         | `temporal_caption`          | Edge image2video paired with `car_driving.jpg`: `i2v_edge_noaudio` (framework, diffusers, vllm_omni) and `i2v_edge` (trt_llm). `run_with_trt_llm.ipynb`: "the Edge variant of the prompt asset, whose timeline describes the 5-second, 121-frame Edge clip rather than the longer Nano/Super clip."                                                                                                                                | The video begins with a dashcam view of a car driving fast along a scenic coastal mountain road, the asphalt rushing beneath and a sharp left curve sign visible on the left. The ca... |
| `assets/prompts/image2video/coastal_road_audio.json`       | `temporal_caption`          | image2video with audio, paired with `coastal_road_audio.jpg`: `i2vs` (framework, diffusers, vllm_omni, sglang) and `i2av_nano`, `i2av_super` (trt_llm: "paired coastal-road image and synchronized audio prompt").                                                                                                                                                                                                                 | The video begins with a dashcam view of a car driving fast along a scenic coastal mountain road, the asphalt rushing beneath and a sharp left curve sign visible on the left. The ca... |
| `assets/prompts/image2video/humanoid_robot.json`           | `temporal_caption`          | image2video without audio, paired with `humanoid_robot.jpg`: `i2v_nano`, `i2v_super` (trt_llm).                                                                                                                                                                                                                                                                                                                                    | The video opens with a humanoid robot standing motionless in a modern living room. At around one second, the robot begins to shift its weight and bend its knees slightly. By two se... |

### 3.2 Reference images

| Asset (skill-relative)                             | Used by                                                                                                                                                                                                                                                                                                                                           |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `assets/images/image2video/car_driving.jpg`        | `i2v_nano_noaudio`, `i2v_super_noaudio`, `i2v_distilled`, `i2v_distil`, `i2v_distil_noaudio`, `i2v_edge_noaudio` (framework, diffusers, vllm_omni, sglang) and `i2v_edge` (trt_llm, whose section calls it "the coastal-road dashcam frame"). `run_with_cosmos_framework.ipynb` Super 4-Step: "the existing public car-driving image and prompt". |
| `assets/images/image2video/coastal_road_audio.jpg` | `i2vs` (framework, diffusers, vllm_omni, sglang) and `i2av_nano`, `i2av_super` (trt_llm).                                                                                                                                                                                                                                                         |
| `assets/images/image2video/humanoid_robot.jpg`     | `i2v_nano`, `i2v_super` (trt_llm); the retired 1.0.0 NIM notebook's Image2Video request, where it was sent inline as a `data:image/jpeg;base64,...` URI with the prompt "The humanoid robot performs a controlled standing backflip in a modern living room, then lands steadily on both feet."                                                   |

Image2video cases upload the image as-is: `-F input_reference=@<path>` (vllm_omni, sglang), `-F input_reference=@<path>;type=image/jpeg` (distilled trt-llm), `-F image_reference=@<path>;type=image/jpeg` (trt_llm), `load_image(<path>)` (diffusers), `vision_path` in the payload JSON (cosmos_framework).

### 3.3 Negative prompts

| Asset (skill-relative)                                | Used by                                                                                                                                                                                                  |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `assets/negative_prompts/text2video/neg_prompt.json`  | every text2video case, through the per-mode rule below.                                                                                                                                                  |
| `assets/negative_prompts/image2video/neg_prompt.json` | every image2video case through the per-mode rule, and every video2video case, which names this file explicitly in its spec (`"negative_prompt": "assets/negative_prompts/image2video/neg_prompt.json"`). |

Per-mode rule (`create_payload`, § 9.2): for any mode other than `text2image`, load `assets/negative_prompts/{mode}/neg_prompt.json`; `run_with_diffusers.ipynb`, `run_with_sglang.ipynb`, `run_with_trt_llm.ipynb` and `run_distilled_with_trt_llm.ipynb` first honour a spec-level `negative_prompt` path when one is given (`run_with_diffusers.ipynb` comment: "There is no negative_prompts/video2video directory; those cases name the file they reuse."). For `text2image`, `run_with_cosmos_framework.ipynb`, `run_with_diffusers.ipynb`, `run_with_vllm_omni.ipynb` and `run_with_sglang.ipynb` put `"negative_prompt": ""` in the payload, while `run_with_trt_llm.ipynb` and `run_distilled_with_trt_llm.ipynb` leave the key out.

### 3.4 Reference videos

| Asset                                                              | Used by                                                                                                                                                                                                                                                                           |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `assets/videos/car_driving_plain.mp4` (not vendored in this skill) | every video2video case: `v2v_nano_noaudio`, `v2vs`, `v2v_super_noaudio`, `v2vs_super` (diffusers), `v2vs` (sglang), `v2v_nano`, `v2v_super` (trt_llm). Uploaded as `-F video_reference=@<path>;type=video/mp4` (sglang, trt_llm) or loaded with `load_video(<path>)` (diffusers). |

### 3.5 Preview rules carried by the notebooks' preview cell

- Caption key precedence: `temporal_caption`, else `comprehensive_t2i_caption`, else `extra.prompt`; printed as `caption[:180]` plus `...` when longer.
- Image files are listed when their suffix is in `IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}` (defined in every notebook's helper cell).
- `run_with_trt_llm.ipynb` additionally lists `assets/videos` files with suffix `.mp4` or `.avi` and their size in KB.
- `run_distilled_with_trt_llm.ipynb` previews only the inputs it uses: `assets/prompts/text2image/robot_draping.json`, `assets/prompts/image2video/car_driving.json`, `assets/images/image2video/car_driving.jpg`.
- `run_with_nim.ipynb` asserts `assets/images/image2video/humanoid_robot.jpg` exists and prints the first 64 characters of its data URI.

## 4. Conventions: paths, output naming, environment variables, endpoints

### 4.1 Skill-relative path configuration

This block replaces the notebooks' `find_repo_root()` / `COSMOS3_AUDIOVISUAL_ROOT` / `COSMOS3_AUDIOVISUAL_OUTPUT_ROOT` setup cells (see the path-translation note at the top). Everything in § 9 runs unchanged on top of it.

```python
from pathlib import Path
import os

COSMOS3_AUDIOVISUAL_ROOT = Path(".").resolve()          # this skill's directory, which holds assets/
COSMOS_ROOT = COSMOS3_AUDIOVISUAL_ROOT                   # used only by the helpers' relative_to() prints
OUT = Path(os.environ.get("OUT", "outputs")).resolve()   # plain output directory
os.environ["COSMOS3_AUDIOVISUAL_OUTPUT_ROOT"] = str(OUT)
```

What the notebooks did instead (all six non-NIM notebooks): `COSMOS3_AUDIOVISUAL_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3" / "generator" / "audiovisual"` and `COSMOS3_AUDIOVISUAL_OUTPUT_ROOT = Path(os.environ.get("COSMOS3_AUDIOVISUAL_OUTPUT_ROOT", COSMOS3_AUDIOVISUAL_ROOT / "outputs" / "notebooks")).resolve()`, then `os.environ["COSMOS3_AUDIOVISUAL_OUTPUT_ROOT"] = str(COSMOS3_AUDIOVISUAL_OUTPUT_ROOT)`. `run_with_nim.ipynb`: `OUTPUT_DIR = AUDIOVISUAL_ROOT / "outputs" / "cosmos3_generator_nim"`, i.e. `<OUT>/cosmos3_generator_nim` here.

### 4.2 Output naming

| Item                                  | Rule                                                                                                                                                                                                                                                                                                            | Notebooks                          |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| Payload file                          | `<OUT>/<backend>/payloads/<use_case>/<use_case>.json`, written as `json.dumps(payload, indent=2) + "\n"`                                                                                                                                                                                                        | all six `create_payload` notebooks |
| Output directory                      | `<OUT>/<backend>/<use_case>/` (created by `create_payload`, again by `run_*_payload`)                                                                                                                                                                                                                           | all six                            |
| `payload["name"]`                     | the use-case key                                                                                                                                                                                                                                                                                                | all six                            |
| `backend` tag                         | `"pytorch"` and `"pytorch_fp8"` (framework, passed explicitly), `"diffusers"`, `"vllm"`, `"sglang"` (passed explicitly), `"trt_llm"` (trt_llm default argument), `"trt_llm_distilled"` (distilled default argument)                                                                                             | —                                  |
| Media file                            | `<output_dir>/<name>.png` for `text2image`, `<output_dir>/<name>.mp4` otherwise                                                                                                                                                                                                                                 | diffusers, vllm_omni, sglang       |
| Media file (trt_llm)                  | `output_stem = output_dir / name`; `post_image` writes `output_stem.with_suffix(".png")`, `post_video` writes `output_stem.with_suffix(".mp4")`                                                                                                                                                                 | trt_llm                            |
| Media file (distilled)                | `output_stem.with_suffix(ext)`, where `_video_extension_from_response` picks `.avi` if the response headers contain `video/x-msvideo` or `video/avi` or the body starts with `RIFF` and has `AVI` in its first 12 bytes, `.mp4` when bytes 4–8 are `ftyp`, and `.mp4` otherwise                                 | distilled trt-llm                  |
| Media file (framework)                | written by `cosmos_framework.scripts.inference` into `-o <output_dir>`                                                                                                                                                                                                                                          | cosmos_framework                   |
| Side files                            | `<stem>.tmp` while downloading (renamed into place on success); `<stem>.error.txt` on failure; `<stem>.headers.txt` (trt_llm, distilled `-D` header dump); `<stem>.response.json` (trt_llm `post_image`)                                                                                                        | server notebooks                   |
| `vision_path` / `video_path`          | stored relative to the payload file's directory: `os.path.relpath(image_path, payload_path.parent)`; resolved back with `resolve_payload_path` (§ 9.1). `run_with_trt_llm.ipynb` stores a video reference under `video_path`; `run_with_diffusers.ipynb` and `run_with_sglang.ipynb` reuse `vision_path` for it | all six                            |
| `view_run` discovery (output listing) | `*.mp4` under the output dir, skipping names ending in `_preview.mp4` or `_browser.mp4`; plus `*.png` (diffusers, vllm_omni, sglang, trt_llm), `*.png` + `*.jpg` + `*.jpeg` (framework), `.mp4` and `.avi` only (distilled)                                                                                     | all six                            |
| NIM                                   | `<OUT>/cosmos3_generator_nim/cosmos3_generator_nim_t2v.mp4` and `cosmos3_generator_nim_i2v.mp4`, written by base64-decoding the JSON response field `b64_video`                                                                                                                                                 | nim                                |

### 4.3 Per-use-case environment variables set by `create_payload`

```python
os.environ[f"COSMOS3_{backend.upper()}_{use_case.upper()}_INPUT"] = str(payload_path)
os.environ[f"COSMOS3_{backend.upper()}_{use_case.upper()}_OUTPUT"] = str(output_dir)
```

Examples as consumed by the framework notebook's `torchrun` cells: `-i "$COSMOS3_PYTORCH_T2I_INPUT" -o "$COSMOS3_PYTORCH_T2I_OUTPUT"`, `$COSMOS3_PYTORCH_FP8_T2I_INPUT` / `$COSMOS3_PYTORCH_FP8_T2I_OUTPUT`, `$COSMOS3_PYTORCH_T2V_NANO_NOAUDIO_INPUT`, `$COSMOS3_PYTORCH_I2V_DISTILLED_OUTPUT`. With the default backend tags the server notebooks produce names such as `COSMOS3_TRT_LLM_T2I_NANO_INPUT` and `COSMOS3_TRT_LLM_DISTILLED_I2V_DISTILLED_OUTPUT`.

### 4.4 Shared environment variables of the in-process notebooks

| Variable                          | `run_with_cosmos_framework.ipynb`                                                                                                                     | `run_with_diffusers.ipynb`                                                                                                                    |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `HF_HOME`                         | `os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))`; documented example `export HF_HOME=/path/to/large/huggingface/cache` | same default and example                                                                                                                      |
| `UV_CACHE_DIR`                    | `os.environ.setdefault("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv"))`                                                                           | same                                                                                                                                          |
| `CUDA_VISIBLE_DEVICES`            | `setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")`; documented example `export CUDA_VISIBLE_DEVICES=0,1,2,3`                                             | `setdefault("CUDA_VISIBLE_DEVICES", "0")`; documented example `export CUDA_VISIBLE_DEVICES=0`                                                 |
| `HF_TOKEN`                        | prerequisite (or `uvx hf@latest auth login`)                                                                                                          | prerequisite; the setup cell prints `HF_TOKEN: <set>` or `<unset>`; passed as `token=os.environ.get("HF_TOKEN") or None` to `from_pretrained` |
| `COSMOS3_AUDIOVISUAL_OUTPUT_ROOT` | § 4.1                                                                                                                                                 | § 4.1                                                                                                                                         |

Backend-only variables (`COSMOS3_REPO`, `COSMOS3_GIT_URL`, `COSMOS3_UV_GROUP`, `UV_PROJECT_ENVIRONMENT`/`COSMOS3_UV_ENV`, `COSMOS3_NUM_GPUS`, `COSMOS3_MASTER_ADDR`, `COSMOS3_TEXT_MASTER_PORT`, `COSMOS3_IMAGE_MASTER_PORT`; `COSMOS3_DIFFUSERS_VENV`, `COSMOS3_TORCH_BACKEND`, `UV_LINK_MODE`, `HF_HUB_DISABLE_XET`) are documented in the backend files.

### 4.5 Server endpoints

Every server notebook resolves one base URL per model short name from a per-model variable that falls back to a backend-wide variable, and reads an API key variable that is sent as `Authorization: Bearer <key>` only when non-empty.

| Notebook                           | Backend-wide variable (default)                                   | Per-model variable → model key                                                                                                                                                                                     | API key variable (default)                | Routes                                                                                                                                                                                                                                                                                                                                                           |
| ---------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_with_vllm_omni.ipynb`         | `COSMOS3_VLLM_BASE_URL` (`http://localhost:8000`)                 | `COSMOS3_VLLM_NANO_BASE_URL` → `Cosmos3-Nano`; `COSMOS3_VLLM_EDGE_BASE_URL` → `Cosmos3-Edge`; `COSMOS3_VLLM_SUPER_BASE_URL` → `Cosmos3-Super`, `Cosmos3-Super-Text2Image-4Step`, `Cosmos3-Super-Image2Video-4Step` | `COSMOS3_VLLM_API_KEY` (`""`)             | text-to-image `/v1/images/generations`; video modes `/v1/videos/sync`; readiness `GET /v1/models` (setup guide)                                                                                                                                                                                                                                                  |
| `run_with_sglang.ipynb`            | `COSMOS3_SGLANG_BASE_URL` (`http://localhost:30000`)              | `COSMOS3_SGLANG_NANO_BASE_URL` → `Cosmos3-Nano`; `COSMOS3_SGLANG_SUPER_BASE_URL` → `Cosmos3-Super`; `COSMOS3_SGLANG_EDGE_BASE_URL` → `Cosmos3-Edge`; `COSMOS3_SGLANG_DISTIL_BASE_URL` → `Cosmos3-Distil`           | `COSMOS3_SGLANG_API_KEY` (`""`)           | text-to-image `/v1/images/generations`; video modes `/v1/videos` (asynchronous: the POST returns `id`, poll `GET /v1/videos/{id}` every 5 s until `status` is `completed` — `failed`/`cancelled` raise — then download `GET /v1/videos/{id}/content`); readiness `GET <server root>/health` (skill note: 503 while warming up, 200 `{"status":"ok"}` when ready) |
| `run_with_trt_llm.ipynb`           | `COSMOS3_TRTLLM_BASE_URL` (`http://localhost:8000`)               | `COSMOS3_TRTLLM_NANO_BASE_URL` → `Cosmos3-Nano`; `COSMOS3_TRTLLM_SUPER_BASE_URL` → `Cosmos3-Super`; `COSMOS3_TRTLLM_EDGE_BASE_URL` → `Cosmos3-Edge`                                                                | `COSMOS3_TRTLLM_API_KEY` (`tensorrt_llm`) | text-to-image `/v1/images/generations`; video generation `/v1/videos/sync`; readiness `GET <server root>/health`                                                                                                                                                                                                                                                 |
| `run_distilled_with_trt_llm.ipynb` | `COSMOS3_TRTLLM_BASE_URL` (`http://localhost:8000`)               | `COSMOS3_TRTLLM_T2I_4STEP_BASE_URL` → `Cosmos3-Super-Text2Image-4Step`; `COSMOS3_TRTLLM_I2V_4STEP_BASE_URL` → `Cosmos3-Super-Image2Video-4Step`                                                                    | `COSMOS3_TRTLLM_API_KEY` (`tensorrt_llm`) | both modes `/v1/videos/generations` (text-to-image runs as a one-frame video request); readiness `GET <server root>/health`                                                                                                                                                                                                                                      |
| `run_with_nim.ipynb`               | `NIM_PORT` (`8000`) → `BASE_URL = f"http://127.0.0.1:{NIM_PORT}"` | —                                                                                                                                                                                                                  | — (NGC key at container launch)           | `POST /v1/infer`; readiness `GET /v1/health/ready`; `/v1/models`, `/v1/metadata`, `/v1/version`, live OpenAPI schema `/openapi.json`                                                                                                                                                                                                                             |

The Prerequisites cells document the local defaults as exports:

```bash
# run_with_vllm_omni.ipynb
export COSMOS3_VLLM_BASE_URL=http://localhost:8000
export COSMOS3_VLLM_NANO_BASE_URL=http://localhost:8000
export COSMOS3_VLLM_SUPER_BASE_URL=http://localhost:8000
export COSMOS3_VLLM_EDGE_BASE_URL=http://localhost:8000

# run_with_sglang.ipynb
export COSMOS3_SGLANG_BASE_URL=http://localhost:30000
export COSMOS3_SGLANG_NANO_BASE_URL=http://localhost:30000
export COSMOS3_SGLANG_SUPER_BASE_URL=http://localhost:30000

# run_with_trt_llm.ipynb
export COSMOS3_TRTLLM_BASE_URL=http://localhost:8000
export COSMOS3_TRTLLM_NANO_BASE_URL=http://localhost:8000
export COSMOS3_TRTLLM_SUPER_BASE_URL=http://localhost:8000
export COSMOS3_TRTLLM_EDGE_BASE_URL=http://localhost:8000
export COSMOS3_TRTLLM_API_KEY=tensorrt_llm

# run_distilled_with_trt_llm.ipynb
export COSMOS3_TRTLLM_BASE_URL=http://localhost:8000
export COSMOS3_TRTLLM_T2I_4STEP_BASE_URL=http://localhost:8000
export COSMOS3_TRTLLM_I2V_4STEP_BASE_URL=http://localhost:8000
export COSMOS3_TRTLLM_API_KEY=tensorrt_llm
```

Endpoint dictionaries as the setup cells build them:

```python
# run_with_vllm_omni.ipynb
DEFAULT_VLLM_BASE_URL = os.environ.get("COSMOS3_VLLM_BASE_URL", "http://localhost:8000")
VLLM_ENDPOINTS = {
    "Cosmos3-Nano": os.environ.get("COSMOS3_VLLM_NANO_BASE_URL", DEFAULT_VLLM_BASE_URL),
    "Cosmos3-Edge": os.environ.get("COSMOS3_VLLM_EDGE_BASE_URL", DEFAULT_VLLM_BASE_URL),
    "Cosmos3-Super": os.environ.get("COSMOS3_VLLM_SUPER_BASE_URL", DEFAULT_VLLM_BASE_URL),
    "Cosmos3-Super-Text2Image-4Step": os.environ.get("COSMOS3_VLLM_SUPER_BASE_URL", DEFAULT_VLLM_BASE_URL),
    "Cosmos3-Super-Image2Video-4Step": os.environ.get("COSMOS3_VLLM_SUPER_BASE_URL", DEFAULT_VLLM_BASE_URL),
}
os.environ.setdefault("COSMOS3_VLLM_API_KEY", "")

# run_with_sglang.ipynb
DEFAULT_SGLANG_BASE_URL = os.environ.get("COSMOS3_SGLANG_BASE_URL", "http://localhost:30000")
SGLANG_ENDPOINTS = {
    "Cosmos3-Nano": os.environ.get("COSMOS3_SGLANG_NANO_BASE_URL", DEFAULT_SGLANG_BASE_URL),
    "Cosmos3-Super": os.environ.get("COSMOS3_SGLANG_SUPER_BASE_URL", DEFAULT_SGLANG_BASE_URL),
    "Cosmos3-Edge": os.environ.get("COSMOS3_SGLANG_EDGE_BASE_URL", DEFAULT_SGLANG_BASE_URL),
    "Cosmos3-Distil": os.environ.get("COSMOS3_SGLANG_DISTIL_BASE_URL", DEFAULT_SGLANG_BASE_URL),
}
os.environ.setdefault("COSMOS3_SGLANG_API_KEY", "")

# run_with_trt_llm.ipynb
DEFAULT_TRTLLM_BASE_URL = os.environ.get("COSMOS3_TRTLLM_BASE_URL", "http://localhost:8000")
TRTLLM_ENDPOINTS = {
    "Cosmos3-Nano": os.environ.get("COSMOS3_TRTLLM_NANO_BASE_URL", DEFAULT_TRTLLM_BASE_URL),
    "Cosmos3-Super": os.environ.get("COSMOS3_TRTLLM_SUPER_BASE_URL", DEFAULT_TRTLLM_BASE_URL),
    "Cosmos3-Edge": os.environ.get("COSMOS3_TRTLLM_EDGE_BASE_URL", DEFAULT_TRTLLM_BASE_URL),
}
os.environ.setdefault("COSMOS3_TRTLLM_API_KEY", "tensorrt_llm")

# run_distilled_with_trt_llm.ipynb
DEFAULT_TRTLLM_BASE_URL = os.environ.get("COSMOS3_TRTLLM_BASE_URL", "http://localhost:8000")
TRTLLM_ENDPOINTS = {
    "Cosmos3-Super-Text2Image-4Step": os.environ.get(
        "COSMOS3_TRTLLM_T2I_4STEP_BASE_URL", DEFAULT_TRTLLM_BASE_URL
    ),
    "Cosmos3-Super-Image2Video-4Step": os.environ.get(
        "COSMOS3_TRTLLM_I2V_4STEP_BASE_URL", DEFAULT_TRTLLM_BASE_URL
    ),
}
os.environ.setdefault("COSMOS3_TRTLLM_API_KEY", "tensorrt_llm")

# run_with_nim.ipynb
NIM_PORT = int(os.environ.get("NIM_PORT", "8000"))
BASE_URL = f"http://127.0.0.1:{NIM_PORT}"
```

URL helpers shared by the server notebooks (`api_root_url` in vllm_omni, sglang, trt_llm, distilled — the vllm_omni and sglang "Verify Endpoint Configuration" cells inline the same normalisation; `server_root_url` and `health_url` in trt_llm and distilled):

```python
def api_root_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def server_root_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized.rstrip("/")


def health_url(base_url: str) -> str:
    return f"{server_root_url(base_url)}/health"
```

Per-backend route helpers built on `api_root_url` (they differ, so each backend file keeps its own):

| Notebook                           | `video_api_url(base_url)`                        | `image_api_url(base_url)`                        |
| ---------------------------------- | ------------------------------------------------ | ------------------------------------------------ |
| `run_with_vllm_omni.ipynb`         | `f"{api_root_url(base_url)}/videos/sync"`        | `f"{api_root_url(base_url)}/images/generations"` |
| `run_with_sglang.ipynb`            | `f"{api_root_url(base_url)}/videos"`             | `f"{api_root_url(base_url)}/images/generations"` |
| `run_with_trt_llm.ipynb`           | `f"{api_root_url(base_url)}/videos/sync"`        | `f"{api_root_url(base_url)}/images/generations"` |
| `run_distilled_with_trt_llm.ipynb` | `f"{api_root_url(base_url)}/videos/generations"` | — (text-to-image goes through `video_api_url`)   |

## 5. Model identifiers

`run_with_diffusers.ipynb` is the only notebook that spells the short-name → Hugging Face id map out as data; every other notebook passes the ids directly on the command line or uses the short names as `ASSET_SETS["model"]` / endpoint keys.

```python
MODEL_IDS = {
    "Cosmos3-Nano": "nvidia/Cosmos3-Nano",
    "Cosmos3-Super": "nvidia/Cosmos3-Super",
    "Cosmos3-Super-Text2Image-4Step": "nvidia/Cosmos3-Super-Text2Image-4Step",
    "Cosmos3-Super-Image2Video-4Step": "nvidia/Cosmos3-Super-Image2Video-4Step",
    "Cosmos3-Edge": "nvidia/Cosmos3-Edge",
}

# Distilled checkpoints are served only by Cosmos3DistilledModularPipeline; Cosmos3OmniPipeline
# cannot load them.
DISTILLED_MODELS = {
    "Cosmos3-Super-Text2Image-4Step",
    "Cosmos3-Super-Image2Video-4Step",
}


def resolve_model_id(model: str) -> str:
    return MODEL_IDS.get(model, model)


def is_distilled_model(model: str) -> bool:
    return model in DISTILLED_MODELS
```

How each notebook names the same checkpoints:

| Short name (`ASSET_SETS["model"]`, endpoint key)  | cosmos_framework `--checkpoint-path`                                                                                                                                                        | diffusers `MODEL_IDS`                    | vllm_omni `vllm serve`                   | sglang `--model-path`                                                                                   | trt_llm / distilled `trtllm-serve`       | NIM                                |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ---------------------------------- |
| `Cosmos3-Nano`                                    | `"Cosmos3-Nano"`                                                                                                                                                                            | `nvidia/Cosmos3-Nano`                    | `nvidia/Cosmos3-Nano`                    | `nvidia/Cosmos3-Nano`                                                                                   | `nvidia/Cosmos3-Nano`                    | `NIM_MODEL_VARIANT=nano` (default) |
| `Cosmos3-Super`                                   | `"Cosmos3-Super"`                                                                                                                                                                           | `nvidia/Cosmos3-Super`                   | `nvidia/Cosmos3-Super`                   | `nvidia/Cosmos3-Super`                                                                                  | `nvidia/Cosmos3-Super`                   | `NIM_MODEL_VARIANT=super`          |
| `Cosmos3-Edge`                                    | `"Cosmos3-Edge"`                                                                                                                                                                            | `nvidia/Cosmos3-Edge`                    | `nvidia/Cosmos3-Edge`                    | `nvidia/Cosmos3-Edge`                                                                                   | `nvidia/Cosmos3-Edge`                    | —                                  |
| `Cosmos3-Super-Text2Image-4Step`                  | `"Cosmos3-Super-Text2Image-4Step"`                                                                                                                                                          | `nvidia/Cosmos3-Super-Text2Image-4Step`  | `nvidia/Cosmos3-Super-Text2Image-4Step`  | (served under the `Cosmos3-Distil` alias, see next row)                                                 | `nvidia/Cosmos3-Super-Text2Image-4Step`  | —                                  |
| `Cosmos3-Super-Image2Video-4Step`                 | `"Cosmos3-Super-Image2Video-4Step"`                                                                                                                                                         | `nvidia/Cosmos3-Super-Image2Video-4Step` | `nvidia/Cosmos3-Super-Image2Video-4Step` | `Cosmos3-Distil` → `nvidia/Cosmos3-Super-Image2Video-4Step` (the "Docker Image: Cosmos3-Distil" launch) | `nvidia/Cosmos3-Super-Image2Video-4Step` | —                                  |
| Nano FP8 (`run_with_cosmos_framework.ipynb` only) | `FP8_MODEL_ID="nvidia/Cosmos3-Nano"`, `FP8_REVISION="fp8"`, downloaded to `FP8_CHECKPOINT="$COSMOS3_REPO/checkpoints/Cosmos3-Nano-FP8"` and passed as `--checkpoint-path "$FP8_CHECKPOINT"` | —                                        | —                                        | —                                                                                                       | —                                        | —                                  |

Model notes that recur across notebooks:

- `run_with_vllm_omni.ipynb` and `run_with_sglang.ipynb`: "`Cosmos3-Super` is the larger 64B model, so it usually needs more GPU memory than `Cosmos3-Nano`."
- `run_with_cosmos_framework.ipynb` (Nano FP8): "FP8 checkpoints are published in the same Hugging Face repositories as the standard checkpoints, under the `fp8` revision rather than in separate FP8 repositories."
- `run_distilled_with_trt_llm.ipynb` model pages: [`nvidia/Cosmos3-Super-Text2Image-4Step`](https://huggingface.co/nvidia/Cosmos3-Super-Text2Image-4Step) — text-to-image; [`nvidia/Cosmos3-Super-Image2Video-4Step`](https://huggingface.co/nvidia/Cosmos3-Super-Image2Video-4Step) — image-to-video. `run_with_vllm_omni.ipynb`: "`Cosmos3-Super-Text2Image-4Step` and `Cosmos3-Super-Image2Video-4Step` are distilled models which require fewer inference steps for Text2Image and Image2Video tasks respectively." `run_with_diffusers.ipynb` `ASSET_SETS` comment: "Distilled 4-step checkpoints bake classifier-free guidance into the weights and read their sigma schedule from the checkpoint, so they carry no num_steps/guidance/shift."
- Edge, as stated in the shared cells: `run_with_cosmos_framework.ipynb` "Cosmos3-Edge has no audio modules, so its examples run without audio."; `run_with_diffusers.ipynb` "Cosmos3-Edge has no audio modules and uses its 480p generation settings."; `run_with_trt_llm.ipynb` sampling-table comment "Cosmos3-Edge is 480p-native and has no audio tower, so it runs a different shape from Nano/Super." (§ 6.1).
- Guardrail model: `nvidia/Cosmos-1.0-Guardrail` (§ 2.3). NIM container image: `nvcr.io/nim/nvidia/cosmos3:2.1.0` with `NIM_MODEL_TYPE=generator` (the retired Generator NIM image was `nvcr.io/nim/nvidia/cosmos3-generator:1.0.0`).

## 6. Sampling defaults

### 6.1 Default tables, verbatim per notebook

`run_with_cosmos_framework.ipynb`, `run_with_vllm_omni.ipynb`, `run_with_sglang.ipynb`:

```python
FIXED_SAMPLING = {
    "num_steps": 35,
    "guidance": 6.0,
    "shift": 10.0,
    "fps": 24,
    "num_frames": 189,
    "resolution": "720",
    "aspect_ratio": "16,9",
    "seed": 0,
}
```

`run_with_diffusers.ipynb` (differs: `seed` is `1234`):

```python
FIXED_SAMPLING = {
    "num_steps": 35,
    "guidance": 6.0,
    "shift": 10.0,
    "fps": 24,
    "num_frames": 189,
    "resolution": "720",
    "aspect_ratio": "16,9",
    "seed": 1234,
}
```

`run_with_trt_llm.ipynb` (differs: no `shift`; adds `max_sequence_length`; plus the two Edge tables):

```python
FIXED_SAMPLING = {
    "num_steps": 35,
    "guidance": 6.0,
    "fps": 24,
    "num_frames": 189,
    "max_sequence_length": 4096,
    "resolution": "720",
    "aspect_ratio": "16,9",
    "seed": 0,
}

# Cosmos3-Edge is 480p-native and has no audio tower, so it runs a different
# shape from Nano/Super. These are the checkpoint's deployed TensorRT-LLM
# defaults, sent explicitly so each request is self-describing. Flow shift (3.0)
# rides the checkpoint-declared native flow schedule and is not a request field.
EDGE_VIDEO_SAMPLING = {
    "num_steps": 50,
    "guidance": 5.0,
    "fps": 24,
    "num_frames": 121,
    "max_sequence_length": 4096,
    "resolution": "480",
    "aspect_ratio": "16,9",
    "seed": 0,
}

# Edge text-to-image runs at its native 640x640 with full-range CFG, through the
# image generation endpoint in native image mode. That path carries no frame or
# frame-rate fields, so this table has none either.
EDGE_T2I_SAMPLING = {
    "num_steps": 50,
    "guidance": 4.0,
    "max_sequence_length": 4096,
    "resolution": "640",
    "aspect_ratio": "1,1",
    "seed": 0,
}
```

`run_distilled_with_trt_llm.ipynb` (no `num_steps`, `guidance` or `shift` at all):

```python
# Distilled checkpoints own their step count and guidance, so neither appears
# here and neither is sent in a request.
DISTILLED_SAMPLING = {
    "fps": 24,
    "num_frames": 189,
    "max_sequence_length": 4096,
    "resolution": "720",
    "aspect_ratio": "16,9",
    "seed": 0,
}
```

The NIM has no sampling table of its own (nim/generation.md): T2V, I2V and V2V use these defaults when fields are omitted — `resolution` `720` (1280 × 720), `num_frames` `189`, `num_inference_steps` `35`, `guidance_scale` `6.0`, `flow_shift` `10.0`, `fps` `24.0`; text2image — `720_1_1` (960 × 960), `num_inference_steps` `50`, `guidance_scale` `4.0`, `flow_shift` `3.0`, empty `negative_prompt`. The skill's smoke request sends `resolution` `"256"` and `num_frames` 25 (skill choice, SKILL.md Variant 7; the retired 1.0.0 notebook's examples used the same values as `resolution="256"` / `num_output_frames=25`).

### 6.2 Comparison

| Payload key           | `FIXED_SAMPLING` framework / vllm_omni / sglang | `FIXED_SAMPLING` diffusers | `FIXED_SAMPLING` trt_llm | `EDGE_VIDEO_SAMPLING` trt_llm | `EDGE_T2I_SAMPLING` trt_llm | `DISTILLED_SAMPLING` distilled |
| --------------------- | ----------------------------------------------- | -------------------------- | ------------------------ | ----------------------------- | --------------------------- | ------------------------------ |
| `num_steps`           | 35                                              | 35                         | 35                       | 50                            | 50                          | —                              |
| `guidance`            | 6.0                                             | 6.0                        | 6.0                      | 5.0                           | 4.0                         | —                              |
| `shift`               | 10.0                                            | 10.0                       | —                        | —                             | —                           | —                              |
| `fps`                 | 24                                              | 24                         | 24                       | 24                            | —                           | 24                             |
| `num_frames`          | 189                                             | 189                        | 189                      | 121                           | —                           | 189                            |
| `max_sequence_length` | —                                               | —                          | 4096                     | 4096                          | 4096                        | 4096                           |
| `resolution`          | `"720"`                                         | `"720"`                    | `"720"`                  | `"480"`                       | `"640"`                     | `"720"`                        |
| `aspect_ratio`        | `"16,9"`                                        | `"16,9"`                   | `"16,9"`                 | `"16,9"`                      | `"1,1"`                     | `"16,9"`                       |
| `seed`                | 0                                               | 1234                       | 0                        | 0                             | 0                           | 0                              |

How the base table combines with a use case's `sampling` override differs per notebook (code in § 9.2): framework, diffusers and vllm_omni spread `**FIXED_SAMPLING, **spec.get("sampling", {})` (override on top); sglang spreads `**FIXED_SAMPLING` only and has no `sampling` keys; trt_llm spreads `**spec.get("sampling", FIXED_SAMPLING)` (a `sampling` table replaces the base table wholesale); distilled spreads `**DISTILLED_SAMPLING, **spec.get("sampling", {})`. `run_with_sglang.ipynb` hard-codes `max_sequence_length` `"4096"` / `4096` in its request builders rather than in the payload.

### 6.3 `extra_params` / `extra_args` defaults

Module-level tables:

```python
# run_with_trt_llm.ipynb
TRTLLM_EXTRA_PARAMS = {
    "use_resolution_template": False,
    "use_duration_template": False,
    "use_system_prompt": False,
    "use_guardrails": True,
}
```

```python
# run_distilled_with_trt_llm.ipynb
# `use_system_prompt` is deliberately absent: the checkpoint's
# `default_use_system_prompt` applies only while the request leaves it unset.
TRTLLM_EXTRA_PARAMS = {
    "use_resolution_template": False,
    "use_duration_template": False,
    "use_guardrails": True,
    "enable_audio": False,
}
```

What each notebook's request builders send (the builders themselves are in the backend files):

| Notebook                           | Video request                                                                                                                                                                                                                                                | Image request                                                                              |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| `run_with_vllm_omni.ipynb`         | form field `extra_params` = `{"use_resolution_template": False, "use_duration_template": False, "guardrails": True}` (compact JSON)                                                                                                                          | JSON `extra_args` = `{"use_resolution_template": False, "guardrails": True}`               |
| `run_with_sglang.ipynb`            | form field `extra_params` = `{"use_resolution_template": False, "use_duration_template": False, "guardrails": True}` (compact JSON)                                                                                                                          | JSON `extra_params` = `{"use_resolution_template": False, "guardrails": True}`             |
| `run_with_trt_llm.ipynb`           | `payload["extra_params"]` = `TRTLLM_EXTRA_PARAMS` + `enable_audio` (`bool(spec.get("enable_audio", False))`) + the spec's `extra_params` (video2video: `use_system_prompt: True`, `condition_video_latent_indexes: [0, 1]`, `condition_video_keep: "first"`) | same table plus `output_type: "image"` (set for every text2image spec by `create_payload`) |
| `run_distilled_with_trt_llm.ipynb` | `payload["extra_params"]` = `TRTLLM_EXTRA_PARAMS` + the spec's `extra_params` (none defined)                                                                                                                                                                 | same (text-to-image is a one-frame video request)                                          |
| `run_with_diffusers.ipynb`         | pipeline kwargs `add_resolution_template=False`, `add_duration_template=False`; `enable_safety_checker=True` at load (`pipe.enable_safety_checker()` for the distilled pipeline)                                                                             | same                                                                                       |
| `run_with_cosmos_framework.ipynb`  | none (payload JSON + CLI flags; `--no-guardrails` only on the Super 4-Step runs)                                                                                                                                                                             | same                                                                                       |
| `run_with_nim.ipynb`               | none (`Unknown request fields are rejected by the server.`)                                                                                                                                                                                                  | —                                                                                          |

## 7. Resolution / aspect ratio → height × width

`payload_dimensions(payload)` returns `(height, width)`; the server notebooks send `size = f"{width}x{height}"` and Diffusers passes `height=`/`width=`. Any pair outside a notebook's table raises `ValueError(f"Unsupported payload resolution/aspect ratio: {payload.get('resolution')} {payload.get('aspect_ratio')}")`.

| `resolution` | `aspect_ratio` | height × width (`size`)   | framework | diffusers | vllm_omni | sglang | trt_llm | distilled |
| ------------ | -------------- | ------------------------- | --------- | --------- | --------- | ------ | ------- | --------- |
| `"720"`      | `"16,9"`       | 720 × 1280 (`1280x720`)   | yes       | yes       | yes       | yes    | yes     | yes       |
| `"256"`      | `"16,9"`       | 192 × 320 (`320x192`)     | yes       | yes       | yes       | —      | yes     | —         |
| `"480"`      | `"16,9"`       | 480 × 832 (`832x480`)     | —         | yes       | yes       | yes    | yes     | —         |
| `"768"`      | `"1,1"`        | 768 × 768 (`768x768`)     | —         | yes       | —         | —      | —       | —         |
| `"480"`      | `"4,3"`        | 480 × 640 (`640x480`)     | —         | yes       | —         | —      | —       | —         |
| `"640"`      | `"1,1"`        | 640 × 640 (`640x640`)     | —         | —         | —         | yes    | yes     | —         |
| `"256"`      | `"1,1"`        | 256 × 256 (`256x256`)     | —         | —         | —         | yes    | —       | —         |
| `"1024"`     | `"1,1"`        | 1024 × 1024 (`1024x1024`) | —         | —         | —         | —      | —       | yes       |

Verbatim per notebook:

```python
# run_with_cosmos_framework.ipynb
def payload_dimensions(payload: dict) -> tuple[int, int]:
    if payload.get("resolution") == "720" and payload.get("aspect_ratio") == "16,9":
        return 720, 1280
    if payload.get("resolution") == "256" and payload.get("aspect_ratio") == "16,9":
        return 192, 320
    raise ValueError(f"Unsupported payload resolution/aspect ratio: {payload.get('resolution')} {payload.get('aspect_ratio')}")
```

```python
# run_with_diffusers.ipynb
def payload_dimensions(payload: dict) -> tuple[int, int]:
    if payload.get("resolution") == "720" and payload.get("aspect_ratio") == "16,9":
        return 720, 1280
    if payload.get("resolution") == "256" and payload.get("aspect_ratio") == "16,9":
        return 192, 320
    if payload.get("resolution") == "768" and payload.get("aspect_ratio") == "1,1":
        return 768, 768
    if payload.get("resolution") == "480" and payload.get("aspect_ratio") == "4,3":
        return 480, 640
    if payload.get("resolution") == "480" and payload.get("aspect_ratio") == "16,9":
        return 480, 832
    raise ValueError(f"Unsupported payload resolution/aspect ratio: {payload.get('resolution')} {payload.get('aspect_ratio')}")
```

```python
# run_with_vllm_omni.ipynb
def payload_dimensions(payload: dict) -> tuple[int, int]:
    if payload.get("resolution") == "720" and payload.get("aspect_ratio") == "16,9":
        return 720, 1280
    if payload.get("resolution") == "256" and payload.get("aspect_ratio") == "16,9":
        return 192, 320
    if payload.get("resolution") == "480" and payload.get("aspect_ratio") == "16,9":
        return 480, 832
    raise ValueError(f"Unsupported payload resolution/aspect ratio: {payload.get('resolution')} {payload.get('aspect_ratio')}")
```

```python
# run_with_sglang.ipynb
def payload_dimensions(payload: dict) -> tuple[int, int]:
    if payload.get("resolution") == "720" and payload.get("aspect_ratio") == "16,9":
        return 720, 1280
    if payload.get("resolution") == "480" and payload.get("aspect_ratio") == "16,9":
        return 480, 832
    if payload.get("resolution") == "640" and payload.get("aspect_ratio") == "1,1":
        return 640, 640
    if payload.get("resolution") == "256" and payload.get("aspect_ratio") == "1,1":
        return 256, 256
    raise ValueError(f"Unsupported payload resolution/aspect ratio: {payload.get('resolution')} {payload.get('aspect_ratio')}")
```

```python
# run_with_trt_llm.ipynb
def payload_dimensions(payload: dict) -> tuple[int, int]:
    if payload.get("resolution") == "720" and payload.get("aspect_ratio") == "16,9":
        return 720, 1280
    if payload.get("resolution") == "480" and payload.get("aspect_ratio") == "16,9":
        return 480, 832
    if payload.get("resolution") == "640" and payload.get("aspect_ratio") == "1,1":
        return 640, 640
    if payload.get("resolution") == "256" and payload.get("aspect_ratio") == "16,9":
        return 192, 320
    raise ValueError(f"Unsupported payload resolution/aspect ratio: {payload.get('resolution')} {payload.get('aspect_ratio')}")
```

```python
# run_distilled_with_trt_llm.ipynb
def payload_dimensions(payload: dict) -> tuple[int, int]:
    if payload.get("resolution") == "720" and payload.get("aspect_ratio") == "16,9":
        return 720, 1280
    if payload.get("resolution") == "1024" and payload.get("aspect_ratio") == "1,1":
        return 1024, 1024
    raise ValueError(f"Unsupported payload resolution/aspect ratio: {payload.get('resolution')} {payload.get('aspect_ratio')}")
```

## 8. Use-case matrix (`ASSET_SETS`)

Every named use case from every notebook. "Defined in" names the notebook(s) whose `ASSET_SETS` carry the entry, with the `backend` tag `create_payload` is called with; rows are split wherever two notebooks define the same key differently. "Sampling override" is the spec's `sampling` dict (how it combines with the base table is in § 6.2); "—" means the notebook's base table applies unchanged. Prompt, image and video paths are skill-relative `assets/...` paths as in the specs. Output: payload `<OUT>/<backend>/payloads/<use_case>/<use_case>.json`, media `<OUT>/<backend>/<use_case>/<use_case>.png|.mp4` (§ 4.2).

`ASSET_SETS` comments, verbatim: `run_with_cosmos_framework.ipynb`, `run_with_diffusers.ipynb`: "All asset paths are repo-relative under cookbooks/cosmos3/generator/audiovisual. The Nano/Super cases without audio are intentionally separate so each run uses the matching model." `run_with_vllm_omni.ipynb`, `run_with_sglang.ipynb`: "All asset paths are repo-relative under cookbooks/cosmos3/generator/audiovisual. Model and sound choices live in this manifest; folders are organized only by modality." `run_with_trt_llm.ipynb`: "TensorRT-LLM uses the image endpoint for T2I and the synchronous video endpoint for T2V, I2V, and V2V."

### 8.1 Nano / Super base cases (framework `pytorch`, diffusers `diffusers`, vllm_omni `vllm`, sglang `sglang`)

| Use case            | Defined in                              | Model           | Mode          | Prompt asset                                               | Image / video asset                                | `enable_sound` | Sampling override | Other spec fields                                                                  |
| ------------------- | --------------------------------------- | --------------- | ------------- | ---------------------------------------------------------- | -------------------------------------------------- | -------------- | ----------------- | ---------------------------------------------------------------------------------- |
| `t2i`               | framework, diffusers, vllm_omni, sglang | `Cosmos3-Nano`  | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                  | `False`        | —                 | framework also runs this key with `backend="pytorch_fp8"` for its Nano FP8 example |
| `t2i_super`         | framework, diffusers, vllm_omni, sglang | `Cosmos3-Super` | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                  | `False`        | —                 | —                                                                                  |
| `t2v_nano_noaudio`  | framework, diffusers, vllm_omni, sglang | `Cosmos3-Nano`  | `text2video`  | `assets/prompts/text2video/robot_kitchen.json`             | —                                                  | `False`        | —                 | —                                                                                  |
| `t2vs`              | framework, diffusers, vllm_omni, sglang | `Cosmos3-Nano`  | `text2video`  | `assets/prompts/text2video/robot_pouring_water_audio.json` | —                                                  | `True`         | —                 | —                                                                                  |
| `i2v_nano_noaudio`  | framework, diffusers, vllm_omni, sglang | `Cosmos3-Nano`  | `image2video` | `assets/prompts/image2video/car_driving.json`              | `assets/images/image2video/car_driving.jpg`        | `False`        | —                 | —                                                                                  |
| `i2vs`              | framework, diffusers, vllm_omni, sglang | `Cosmos3-Nano`  | `image2video` | `assets/prompts/image2video/coastal_road_audio.json`       | `assets/images/image2video/coastal_road_audio.jpg` | `True`         | —                 | —                                                                                  |
| `t2v_super_noaudio` | framework, diffusers, vllm_omni, sglang | `Cosmos3-Super` | `text2video`  | `assets/prompts/text2video/robot_kitchen.json`             | —                                                  | `False`        | —                 | —                                                                                  |
| `i2v_super_noaudio` | framework, diffusers, vllm_omni, sglang | `Cosmos3-Super` | `image2video` | `assets/prompts/image2video/car_driving.json`              | `assets/images/image2video/car_driving.jpg`        | `False`        | —                 | —                                                                                  |

### 8.2 Edge cases

| Use case           | Defined in                                                     | Model          | Mode          | Prompt asset                                       | Image asset                                 | `enable_sound`                    | Sampling override                                                                                                                        | Other spec fields                                                                                                                                                                                                                                                            |
| ------------------ | -------------------------------------------------------------- | -------------- | ------------- | -------------------------------------------------- | ------------------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `t2i_edge`         | framework (entry only; the notebook has no run section for it) | `Cosmos3-Edge` | `text2image`  | `assets/prompts/text2image/robot_draping.json`     | —                                           | `False`                           | `{"resolution": "480"}`                                                                                                                  | —                                                                                                                                                                                                                                                                            |
| `t2i_edge`         | sglang                                                         | `Cosmos3-Edge` | `text2image`  | `assets/prompts/text2image/robot_draping.json`     | —                                           | `False`                           | — (`create_payload` removes `guidance` and `shift` and sets `resolution` to `"480"` for Edge models)                                     | —                                                                                                                                                                                                                                                                            |
| `t2v_edge_noaudio` | framework (entry only; no run section)                         | `Cosmos3-Edge` | `text2video`  | `assets/prompts/text2video/robot_kitchen.json`     | —                                           | `False`                           | `{"resolution": "480", "num_frames": 121, "fps": 24}`                                                                                    | —                                                                                                                                                                                                                                                                            |
| `t2v_edge_noaudio` | sglang                                                         | `Cosmos3-Edge` | `text2video`  | `assets/prompts/text2video/robot_kitchen.json`     | —                                           | `False`                           | — (Edge rule as above)                                                                                                                   | —                                                                                                                                                                                                                                                                            |
| `i2v_edge_noaudio` | framework                                                      | `Cosmos3-Edge` | `image2video` | `assets/prompts/image2video/car_driving_edge.json` | `assets/images/image2video/car_driving.jpg` | `False`                           | `{"resolution": "480", "num_frames": 121, "fps": 24, "num_steps": 20, "guidance": 5.0, "shift": 8.0}`                                    | —                                                                                                                                                                                                                                                                            |
| `i2v_edge_noaudio` | diffusers, vllm_omni                                           | `Cosmos3-Edge` | `image2video` | `assets/prompts/image2video/car_driving_edge.json` | `assets/images/image2video/car_driving.jpg` | `False`                           | `{"num_steps": 20, "guidance": 5.0, "shift": 8.0, "fps": 24, "num_frames": 121, "resolution": "480", "aspect_ratio": "16,9", "seed": 0}` | —                                                                                                                                                                                                                                                                            |
| `i2v_edge_noaudio` | sglang                                                         | `Cosmos3-Edge` | `image2video` | `assets/prompts/image2video/car_driving.json`      | `assets/images/image2video/car_driving.jpg` | `False`                           | — (Edge rule as above)                                                                                                                   | —                                                                                                                                                                                                                                                                            |
| `t2i_edge`         | trt_llm (`trt_llm`)                                            | `Cosmos3-Edge` | `text2image`  | `assets/prompts/text2image/robot_draping.json`     | —                                           | (no `enable_audio` key → `False`) | `EDGE_T2I_SAMPLING` (§ 6.1)                                                                                                              | `extra_params: {"output_type": "image"}` — spec comment: "Native text-to-image. Without this the request runs the video path, which defaults the negative prompt to Cosmos3's video negative prompt (motion and frame-to-frame artifact terms) instead of leaving it empty." |
| `t2v_edge`         | trt_llm                                                        | `Cosmos3-Edge` | `text2video`  | `assets/prompts/text2video/robot_kitchen.json`     | —                                           | `False`                           | `EDGE_VIDEO_SAMPLING` (§ 6.1)                                                                                                            | —                                                                                                                                                                                                                                                                            |
| `i2v_edge`         | trt_llm                                                        | `Cosmos3-Edge` | `image2video` | `assets/prompts/image2video/car_driving_edge.json` | `assets/images/image2video/car_driving.jpg` | `False`                           | `EDGE_VIDEO_SAMPLING` (§ 6.1)                                                                                                            | —                                                                                                                                                                                                                                                                            |

### 8.3 Distilled four-step students

| Use case             | Defined in                              | Model                             | Mode          | Prompt asset                                   | Image asset                                 | `enable_sound` | Sampling override                                                                                                                        | Other spec fields   |
| -------------------- | --------------------------------------- | --------------------------------- | ------------- | ---------------------------------------------- | ------------------------------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| `t2i_distilled`      | framework                               | `Cosmos3-Super-Text2Image-4Step`  | `text2image`  | `assets/prompts/text2image/robot_draping.json` | —                                           | `False`        | `{"num_steps": 50, "guidance": 1.0, "shift": 3.0, "fps": 30, "num_frames": 1, "resolution": "768", "aspect_ratio": "1,1", "seed": 1}`    | —                   |
| `t2i_distilled`      | diffusers                               | `Cosmos3-Super-Text2Image-4Step`  | `text2image`  | `assets/prompts/text2image/robot_draping.json` | —                                           | `False`        | `{"fps": 30, "num_frames": 1, "resolution": "768", "aspect_ratio": "1,1", "seed": 1}`                                                    | `"distilled": True` |
| `t2i_distilled`      | distilled trt-llm (`trt_llm_distilled`) | `Cosmos3-Super-Text2Image-4Step`  | `text2image`  | `assets/prompts/text2image/robot_draping.json` | —                                           | —              | `{"resolution": "1024", "aspect_ratio": "1,1"}` — spec comment: "The deployed text-to-image shape warmed by cosmos3-t2i-1gpu.yaml."      | —                   |
| `i2v_distilled`      | framework                               | `Cosmos3-Super-Image2Video-4Step` | `image2video` | `assets/prompts/image2video/car_driving.json`  | `assets/images/image2video/car_driving.jpg` | `False`        | `{"num_steps": 35, "guidance": 1.0, "shift": 10.0, "fps": 30, "num_frames": 121, "resolution": "480", "aspect_ratio": "4,3", "seed": 1}` | —                   |
| `i2v_distilled`      | diffusers                               | `Cosmos3-Super-Image2Video-4Step` | `image2video` | `assets/prompts/image2video/car_driving.json`  | `assets/images/image2video/car_driving.jpg` | `False`        | `{"fps": 30, "num_frames": 121, "resolution": "480", "aspect_ratio": "4,3", "seed": 1}`                                                  | `"distilled": True` |
| `i2v_distilled`      | distilled trt-llm                       | `Cosmos3-Super-Image2Video-4Step` | `image2video` | `assets/prompts/image2video/car_driving.json`  | `assets/images/image2video/car_driving.jpg` | —              | —                                                                                                                                        | —                   |
| `t2i_distil`         | vllm_omni                               | `Cosmos3-Super-Text2Image-4Step`  | `text2image`  | `assets/prompts/text2image/robot_draping.json` | —                                           | `False`        | —                                                                                                                                        | —                   |
| `i2v_distil`         | vllm_omni                               | `Cosmos3-Super-Image2Video-4Step` | `image2video` | `assets/prompts/image2video/car_driving.json`  | `assets/images/image2video/car_driving.jpg` | `False`        | —                                                                                                                                        | —                   |
| `t2i_distil`         | sglang                                  | `Cosmos3-Distil`                  | `text2image`  | `assets/prompts/text2image/robot_draping.json` | —                                           | `False`        | — (`create_payload` removes `guidance`, `shift` and `num_steps` for Distil models)                                                       | —                   |
| `i2v_distil_noaudio` | sglang                                  | `Cosmos3-Distil`                  | `image2video` | `assets/prompts/image2video/car_driving.json`  | `assets/images/image2video/car_driving.jpg` | `False`        | — (Distil rule as above)                                                                                                                 | —                   |

### 8.4 Video-to-video cases

| Use case            | Defined in        | Model           | Mode          | Prompt asset                                  | Negative prompt (explicit)                            | Video asset                                                        | `enable_sound`                    | Other payload fields                                                                                                                           |
| ------------------- | ----------------- | --------------- | ------------- | --------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------ | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `v2v_nano_noaudio`  | diffusers         | `Cosmos3-Nano`  | `video2video` | `assets/prompts/image2video/car_driving.json` | `assets/negative_prompts/image2video/neg_prompt.json` | `assets/videos/car_driving_plain.mp4` (not vendored in this skill) | `False`                           | `condition_frame_indexes_vision: [0, 1]`, `condition_video_keep: "first"` (defaults added by `create_payload`)                                 |
| `v2vs`              | diffusers, sglang | `Cosmos3-Nano`  | `video2video` | `assets/prompts/image2video/car_driving.json` | `assets/negative_prompts/image2video/neg_prompt.json` | `assets/videos/car_driving_plain.mp4` (not vendored in this skill) | `True`                            | diffusers adds the two condition fields; sglang adds none                                                                                      |
| `v2v_super_noaudio` | diffusers         | `Cosmos3-Super` | `video2video` | `assets/prompts/image2video/car_driving.json` | `assets/negative_prompts/image2video/neg_prompt.json` | `assets/videos/car_driving_plain.mp4` (not vendored in this skill) | `False`                           | condition fields as above                                                                                                                      |
| `v2vs_super`        | diffusers         | `Cosmos3-Super` | `video2video` | `assets/prompts/image2video/car_driving.json` | `assets/negative_prompts/image2video/neg_prompt.json` | `assets/videos/car_driving_plain.mp4` (not vendored in this skill) | `True`                            | condition fields as above                                                                                                                      |
| `v2v_nano`          | trt_llm           | `Cosmos3-Nano`  | `video2video` | `assets/prompts/image2video/car_driving.json` | `assets/negative_prompts/image2video/neg_prompt.json` | `assets/videos/car_driving_plain.mp4` (not vendored in this skill) | (`enable_audio` absent → `False`) | `extra_params: {"use_system_prompt": True, "condition_video_latent_indexes": [0, 1], "condition_video_keep": "first"}`; stored as `video_path` |
| `v2v_super`         | trt_llm           | `Cosmos3-Super` | `video2video` | `assets/prompts/image2video/car_driving.json` | `assets/negative_prompts/image2video/neg_prompt.json` | `assets/videos/car_driving_plain.mp4` (not vendored in this skill) | `False`                           | same `extra_params` as `v2v_nano`                                                                                                              |

### 8.5 TensorRT-LLM Nano / Super cases (`run_with_trt_llm.ipynb`, backend `trt_llm`; `FIXED_SAMPLING` of § 6.1 for all; no `enable_sound` key — audio is `enable_audio`)

| Use case     | Model           | Mode          | Prompt asset                                               | Image asset                                        | `enable_audio` |
| ------------ | --------------- | ------------- | ---------------------------------------------------------- | -------------------------------------------------- | -------------- |
| `t2i_nano`   | `Cosmos3-Nano`  | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                  | `False`        |
| `t2v_nano`   | `Cosmos3-Nano`  | `text2video`  | `assets/prompts/text2video/robot_kitchen.json`             | —                                                  | `False`        |
| `t2av_nano`  | `Cosmos3-Nano`  | `text2video`  | `assets/prompts/text2video/robot_pouring_water_audio.json` | —                                                  | `True`         |
| `i2v_nano`   | `Cosmos3-Nano`  | `image2video` | `assets/prompts/image2video/humanoid_robot.json`           | `assets/images/image2video/humanoid_robot.jpg`     | `False`        |
| `i2av_nano`  | `Cosmos3-Nano`  | `image2video` | `assets/prompts/image2video/coastal_road_audio.json`       | `assets/images/image2video/coastal_road_audio.jpg` | `True`         |
| `v2v_nano`   | `Cosmos3-Nano`  | `video2video` | see § 8.4                                                  | —                                                  | `False`        |
| `t2i_super`  | `Cosmos3-Super` | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                  | `False`        |
| `t2v_super`  | `Cosmos3-Super` | `text2video`  | `assets/prompts/text2video/robot_kitchen.json`             | —                                                  | `False`        |
| `t2av_super` | `Cosmos3-Super` | `text2video`  | `assets/prompts/text2video/robot_pouring_water_audio.json` | —                                                  | `True`         |
| `i2v_super`  | `Cosmos3-Super` | `image2video` | `assets/prompts/image2video/humanoid_robot.json`           | `assets/images/image2video/humanoid_robot.jpg`     | `False`        |
| `i2av_super` | `Cosmos3-Super` | `image2video` | `assets/prompts/image2video/coastal_road_audio.json`       | `assets/images/image2video/coastal_road_audio.jpg` | `True`         |
| `v2v_super`  | `Cosmos3-Super` | `video2video` | see § 8.4                                                  | —                                                  | `False`        |

For every trt_llm `text2image` spec `create_payload` adds `extra_params["output_type"] = "image"` and removes `num_frames` and `fps` from the payload; every other mode receives the per-mode negative prompt (§ 3.3).

### 8.6 NIM requests (cookbook README `## Run with NIM` and nim/generation.md; inline requests, no `ASSET_SETS`)

| Request                                                                             | Model                                                     | Fields                                                                                                                                                                                                                | Output                                                                        |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| text2video (`"model_mode": "text2video"`, non-empty `prompt`, no `input_reference`) | `NIM_MODEL_VARIANT` `nano` (the README launch) or `super` | `prompt`: "A modern industrial robotic arm cleans a kitchen counter with a green sponge."; `resolution`: "720_16_9"; `num_frames`: 189; `fps`: 24.0; `seed`: 0                                                        | `/tmp/cosmos3_nim_t2v.mp4` (README) / `t2v_robot_kitchen.mp4` (generation.md) |
| image2video (`"model_mode": "image2video"`, image `input_reference`)                | same                                                      | `prompt`: "A car travels along a coastal mountain road with natural motion."; `input_reference`: data URL of `assets/images/image2video/car_driving.jpg`; `seed`: 0 (omitted fields take the video defaults of § 6.1) | `i2v_car_driving.mp4` (generation.md)                                         |

The skill's own I2V request (bundled prompt and negative prompt as compact JSON, `"256"` / 25 frames) is in SKILL.md Variant 7; it writes `<OUT>/cosmos3_generator_nim/cosmos3_generator_nim_i2v.mp4` (§ 4.2).

### 8.7 `ASSET_SETS` source, verbatim per notebook

```python
# run_with_cosmos_framework.ipynb
# All asset paths are repo-relative under cookbooks/cosmos3/generator/audiovisual.
# The Nano/Super cases without audio are intentionally separate so each run uses the matching model.
ASSET_SETS = {
    "t2i": {
        "model": "Cosmos3-Nano",
        "mode": "text2image",
        "prompt": "assets/prompts/text2image/robot_draping.json",
        "enable_sound": False,
    },
    "t2i_super": {
        "model": "Cosmos3-Super",
        "mode": "text2image",
        "prompt": "assets/prompts/text2image/robot_draping.json",
        "enable_sound": False,
    },
    "t2i_distilled": {
        "model": "Cosmos3-Super-Text2Image-4Step",
        "mode": "text2image",
        "prompt": "assets/prompts/text2image/robot_draping.json",
        "enable_sound": False,
        "sampling": {
            "num_steps": 50,
            "guidance": 1.0,
            "shift": 3.0,
            "fps": 30,
            "num_frames": 1,
            "resolution": "768",
            "aspect_ratio": "1,1",
            "seed": 1,
        },
    },
    "t2v_nano_noaudio": {
        "model": "Cosmos3-Nano",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_kitchen.json",
        "enable_sound": False,
    },
    "t2vs": {
        "model": "Cosmos3-Nano",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_pouring_water_audio.json",
        "enable_sound": True,
    },
    "i2v_nano_noaudio": {
        "model": "Cosmos3-Nano",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "image": "assets/images/image2video/car_driving.jpg",
        "enable_sound": False,
    },
    "i2vs": {
        "model": "Cosmos3-Nano",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/coastal_road_audio.json",
        "image": "assets/images/image2video/coastal_road_audio.jpg",
        "enable_sound": True,
    },
    "t2v_super_noaudio": {
        "model": "Cosmos3-Super",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_kitchen.json",
        "enable_sound": False,
    },
    "i2v_super_noaudio": {
        "model": "Cosmos3-Super",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "image": "assets/images/image2video/car_driving.jpg",
        "enable_sound": False,
    },
    "i2v_distilled": {
        "model": "Cosmos3-Super-Image2Video-4Step",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "image": "assets/images/image2video/car_driving.jpg",
        "enable_sound": False,
        "sampling": {
            "num_steps": 35,
            "guidance": 1.0,
            "shift": 10.0,
            "fps": 30,
            "num_frames": 121,
            "resolution": "480",
            "aspect_ratio": "4,3",
            "seed": 1,
        },
    },
    "t2i_edge": {
        "model": "Cosmos3-Edge",
        "mode": "text2image",
        "prompt": "assets/prompts/text2image/robot_draping.json",
        "enable_sound": False,
        "sampling": {
            "resolution": "480",
        },
    },
    "t2v_edge_noaudio": {
        "model": "Cosmos3-Edge",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_kitchen.json",
        "enable_sound": False,
        "sampling": {
            "resolution": "480",
            "num_frames": 121,
            "fps": 24,
        },
    },
    "i2v_edge_noaudio": {
        "model": "Cosmos3-Edge",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/car_driving_edge.json",
        "image": "assets/images/image2video/car_driving.jpg",
        "enable_sound": False,
        "sampling": {
            "resolution": "480",
            "num_frames": 121,
            "fps": 24,
            "num_steps": 20,
            "guidance": 5.0,
            "shift": 8.0,
        },
    },
}
```

```python
# run_with_diffusers.ipynb
# All asset paths are repo-relative under cookbooks/cosmos3/generator/audiovisual.
# The Nano/Super cases without audio are intentionally separate so each run uses the matching model.
ASSET_SETS = {
    "t2i": {
        "model": "Cosmos3-Nano",
        "mode": "text2image",
        "prompt": "assets/prompts/text2image/robot_draping.json",
        "enable_sound": False,
    },
    "t2i_super": {
        "model": "Cosmos3-Super",
        "mode": "text2image",
        "prompt": "assets/prompts/text2image/robot_draping.json",
        "enable_sound": False,
    },
    "t2v_nano_noaudio": {
        "model": "Cosmos3-Nano",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_kitchen.json",
        "enable_sound": False,
    },
    "t2vs": {
        "model": "Cosmos3-Nano",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_pouring_water_audio.json",
        "enable_sound": True,
    },
    "i2v_nano_noaudio": {
        "model": "Cosmos3-Nano",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "image": "assets/images/image2video/car_driving.jpg",
        "enable_sound": False,
    },
    "i2v_edge_noaudio": {
        "model": "Cosmos3-Edge",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/car_driving_edge.json",
        "image": "assets/images/image2video/car_driving.jpg",
        "enable_sound": False,
        "sampling": {
            "num_steps": 20,
            "guidance": 5.0,
            "shift": 8.0,
            "fps": 24,
            "num_frames": 121,
            "resolution": "480",
            "aspect_ratio": "16,9",
            "seed": 0,
        },
    },
    "i2vs": {
        "model": "Cosmos3-Nano",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/coastal_road_audio.json",
        "image": "assets/images/image2video/coastal_road_audio.jpg",
        "enable_sound": True,
    },
    "v2v_nano_noaudio": {
        "model": "Cosmos3-Nano",
        "mode": "video2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "negative_prompt": "assets/negative_prompts/image2video/neg_prompt.json",
        "video": "assets/videos/car_driving_plain.mp4",  # not vendored in this skill
        "enable_sound": False,
    },
    "v2vs": {
        "model": "Cosmos3-Nano",
        "mode": "video2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "negative_prompt": "assets/negative_prompts/image2video/neg_prompt.json",
        "video": "assets/videos/car_driving_plain.mp4",  # not vendored in this skill
        "enable_sound": True,
    },
    "t2v_super_noaudio": {
        "model": "Cosmos3-Super",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_kitchen.json",
        "enable_sound": False,
    },
    "i2v_super_noaudio": {
        "model": "Cosmos3-Super",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "image": "assets/images/image2video/car_driving.jpg",
        "enable_sound": False,
    },
    "v2v_super_noaudio": {
        "model": "Cosmos3-Super",
        "mode": "video2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "negative_prompt": "assets/negative_prompts/image2video/neg_prompt.json",
        "video": "assets/videos/car_driving_plain.mp4",  # not vendored in this skill
        "enable_sound": False,
    },
    "v2vs_super": {
        "model": "Cosmos3-Super",
        "mode": "video2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "negative_prompt": "assets/negative_prompts/image2video/neg_prompt.json",
        "video": "assets/videos/car_driving_plain.mp4",  # not vendored in this skill
        "enable_sound": True,
    },
    # Distilled 4-step checkpoints bake classifier-free guidance into the weights and read their
    # sigma schedule from the checkpoint, so they carry no num_steps/guidance/shift.
    "t2i_distilled": {
        "model": "Cosmos3-Super-Text2Image-4Step",
        "mode": "text2image",
        "prompt": "assets/prompts/text2image/robot_draping.json",
        "enable_sound": False,
        "distilled": True,
        "sampling": {
            "fps": 30,
            "num_frames": 1,
            "resolution": "768",
            "aspect_ratio": "1,1",
            "seed": 1,
        },
    },
    "i2v_distilled": {
        "model": "Cosmos3-Super-Image2Video-4Step",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "image": "assets/images/image2video/car_driving.jpg",
        "enable_sound": False,
        "distilled": True,
        "sampling": {
            "fps": 30,
            "num_frames": 121,
            "resolution": "480",
            "aspect_ratio": "4,3",
            "seed": 1,
        },
    },
}
```

```python
# run_with_vllm_omni.ipynb
# All asset paths are repo-relative under cookbooks/cosmos3/generator/audiovisual.
# Model and sound choices live in this manifest; folders are organized only by modality.
ASSET_SETS = {
    "t2i": {
        "model": "Cosmos3-Nano",
        "mode": "text2image",
        "prompt": "assets/prompts/text2image/robot_draping.json",
        "enable_sound": False,
    },
    "t2i_super": {
        "model": "Cosmos3-Super",
        "mode": "text2image",
        "prompt": "assets/prompts/text2image/robot_draping.json",
        "enable_sound": False,
    },
    "t2v_nano_noaudio": {
        "model": "Cosmos3-Nano",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_kitchen.json",
        "enable_sound": False,
    },
    "t2vs": {
        "model": "Cosmos3-Nano",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_pouring_water_audio.json",
        "enable_sound": True,
    },
    "i2v_nano_noaudio": {
        "model": "Cosmos3-Nano",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "image": "assets/images/image2video/car_driving.jpg",
        "enable_sound": False,
    },
    "i2v_edge_noaudio": {
        "model": "Cosmos3-Edge",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/car_driving_edge.json",
        "image": "assets/images/image2video/car_driving.jpg",
        "enable_sound": False,
        "sampling": {
            "num_steps": 20,
            "guidance": 5.0,
            "shift": 8.0,
            "fps": 24,
            "num_frames": 121,
            "resolution": "480",
            "aspect_ratio": "16,9",
            "seed": 0,
        },
    },
    "i2vs": {
        "model": "Cosmos3-Nano",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/coastal_road_audio.json",
        "image": "assets/images/image2video/coastal_road_audio.jpg",
        "enable_sound": True,
    },
    "t2v_super_noaudio": {
        "model": "Cosmos3-Super",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_kitchen.json",
        "enable_sound": False,
    },
    "i2v_super_noaudio": {
        "model": "Cosmos3-Super",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "image": "assets/images/image2video/car_driving.jpg",
        "enable_sound": False,
    },
    "t2i_distil": {
        "model": "Cosmos3-Super-Text2Image-4Step",
        "mode": "text2image",
        "prompt": "assets/prompts/text2image/robot_draping.json",
        "enable_sound": False,
    },
    "i2v_distil": {
        "model": "Cosmos3-Super-Image2Video-4Step",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "image": "assets/images/image2video/car_driving.jpg",
        "enable_sound": False,
    },
}
```

```python
# run_with_sglang.ipynb
# All asset paths are repo-relative under cookbooks/cosmos3/generator/audiovisual.
# Model and sound choices live in this manifest; folders are organized only by modality.
ASSET_SETS = {
    "t2i": {
        "model": "Cosmos3-Nano",
        "mode": "text2image",
        "prompt": "assets/prompts/text2image/robot_draping.json",
        "enable_sound": False,
    },
    "t2i_super": {
        "model": "Cosmos3-Super",
        "mode": "text2image",
        "prompt": "assets/prompts/text2image/robot_draping.json",
        "enable_sound": False,
    },
    "t2i_edge": {
        "model": "Cosmos3-Edge",
        "mode": "text2image",
        "prompt": "assets/prompts/text2image/robot_draping.json",
        "enable_sound": False,
    },
    "t2i_distil": {
        "model": "Cosmos3-Distil",
        "mode": "text2image",
        "prompt": "assets/prompts/text2image/robot_draping.json",
        "enable_sound": False,
    },
    "t2v_nano_noaudio": {
        "model": "Cosmos3-Nano",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_kitchen.json",
        "enable_sound": False,
    },
    "t2v_edge_noaudio": {
        "model": "Cosmos3-Edge",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_kitchen.json",
        "enable_sound": False,
    },
    "t2vs": {
        "model": "Cosmos3-Nano",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_pouring_water_audio.json",
        "enable_sound": True,
    },
    "i2v_nano_noaudio": {
        "model": "Cosmos3-Nano",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "image": "assets/images/image2video/car_driving.jpg",
        "enable_sound": False,
    },
    "i2v_edge_noaudio": {
        "model": "Cosmos3-Edge",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "image": "assets/images/image2video/car_driving.jpg",
        "enable_sound": False,
    },
    "i2v_distil_noaudio": {
        "model": "Cosmos3-Distil",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "image": "assets/images/image2video/car_driving.jpg",
        "enable_sound": False,
    },
    "i2vs": {
        "model": "Cosmos3-Nano",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/coastal_road_audio.json",
        "image": "assets/images/image2video/coastal_road_audio.jpg",
        "enable_sound": True,
    },
    "v2vs": {
        "model": "Cosmos3-Nano",
        "mode": "video2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "negative_prompt": "assets/negative_prompts/image2video/neg_prompt.json",
        "video": "assets/videos/car_driving_plain.mp4",  # not vendored in this skill
        "enable_sound": True,
    },
    "t2v_super_noaudio": {
        "model": "Cosmos3-Super",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_kitchen.json",
        "enable_sound": False,
    },
    "i2v_super_noaudio": {
        "model": "Cosmos3-Super",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "image": "assets/images/image2video/car_driving.jpg",
        "enable_sound": False,
    },
}
```

```python
# run_with_trt_llm.ipynb
# TensorRT-LLM uses the image endpoint for T2I and the synchronous video
# endpoint for T2V, I2V, and V2V.
ASSET_SETS = {
    "t2i_nano": {
        "model": "Cosmos3-Nano",
        "mode": "text2image",
        "prompt": "assets/prompts/text2image/robot_draping.json",
    },
    "t2v_nano": {
        "model": "Cosmos3-Nano",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_kitchen.json",
    },
    "t2av_nano": {
        "model": "Cosmos3-Nano",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_pouring_water_audio.json",
        "enable_audio": True,
    },
    "i2v_nano": {
        "model": "Cosmos3-Nano",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/humanoid_robot.json",
        "image": "assets/images/image2video/humanoid_robot.jpg",
    },
    "i2av_nano": {
        "model": "Cosmos3-Nano",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/coastal_road_audio.json",
        "image": "assets/images/image2video/coastal_road_audio.jpg",
        "enable_audio": True,
    },
    "v2v_nano": {
        "model": "Cosmos3-Nano",
        "mode": "video2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "negative_prompt": "assets/negative_prompts/image2video/neg_prompt.json",
        "video": "assets/videos/car_driving_plain.mp4",  # not vendored in this skill
        "extra_params": {
            "use_system_prompt": True,
            "condition_video_latent_indexes": [0, 1],
            "condition_video_keep": "first",
        },
    },
    "t2i_super": {
        "model": "Cosmos3-Super",
        "mode": "text2image",
        "prompt": "assets/prompts/text2image/robot_draping.json",
    },
    "t2v_super": {
        "model": "Cosmos3-Super",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_kitchen.json",
    },
    "t2av_super": {
        "model": "Cosmos3-Super",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_pouring_water_audio.json",
        "enable_audio": True,
    },
    "i2v_super": {
        "model": "Cosmos3-Super",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/humanoid_robot.json",
        "image": "assets/images/image2video/humanoid_robot.jpg",
    },
    "i2av_super": {
        "model": "Cosmos3-Super",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/coastal_road_audio.json",
        "image": "assets/images/image2video/coastal_road_audio.jpg",
        "enable_audio": True,
    },
    "v2v_super": {
        "model": "Cosmos3-Super",
        "mode": "video2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "negative_prompt": "assets/negative_prompts/image2video/neg_prompt.json",
        "video": "assets/videos/car_driving_plain.mp4",  # not vendored in this skill
        "extra_params": {
            "use_system_prompt": True,
            "condition_video_latent_indexes": [0, 1],
            "condition_video_keep": "first",
        },
    },
    "t2i_edge": {
        "model": "Cosmos3-Edge",
        "mode": "text2image",
        "prompt": "assets/prompts/text2image/robot_draping.json",
        "sampling": EDGE_T2I_SAMPLING,
        # Native text-to-image. Without this the request runs the video path,
        # which defaults the negative prompt to Cosmos3's video negative prompt
        # (motion and frame-to-frame artifact terms) instead of leaving it empty.
        "extra_params": {"output_type": "image"},
    },
    "t2v_edge": {
        "model": "Cosmos3-Edge",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_kitchen.json",
        "sampling": EDGE_VIDEO_SAMPLING,
    },
    "i2v_edge": {
        "model": "Cosmos3-Edge",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/car_driving_edge.json",
        "image": "assets/images/image2video/car_driving.jpg",
        "sampling": EDGE_VIDEO_SAMPLING,
    },
}
```

```python
# run_distilled_with_trt_llm.ipynb
ASSET_SETS = {
    "t2i_distilled": {
        "model": "Cosmos3-Super-Text2Image-4Step",
        "mode": "text2image",
        "prompt": "assets/prompts/text2image/robot_draping.json",
        # The deployed text-to-image shape warmed by cosmos3-t2i-1gpu.yaml.
        "sampling": {"resolution": "1024", "aspect_ratio": "1,1"},
    },
    "i2v_distilled": {
        "model": "Cosmos3-Super-Image2Video-4Step",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "image": "assets/images/image2video/car_driving.jpg",
    },
}
```

## 9. Payload helpers

De-Jupyterised once for this whole section: the `from IPython.display import ...` imports and the inline `display(Image(...))`, `display(Video(...))` and `display_video(...)` preview calls that followed the `image:`/`video:` prints are removed; every other line is verbatim. The helpers assume the configuration block of § 4.1 and the tables of §§ 6–8.

### 9.1 Helpers identical in all six `create_payload` notebooks

```python
import json
import os
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def asset_path(relative_path: str) -> Path:
    path = COSMOS3_AUDIOVISUAL_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    return path.resolve()


def compact_json_file(path: Path) -> str:
    return json.dumps(json.loads(path.read_text()), ensure_ascii=True, separators=(",", ":"))


def resolve_payload_path(payload_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (payload_path.parent / path).resolve()
```

Prompt serialisation rule: every prompt and negative prompt is sent as the compact JSON string of the asset file — `json.dumps(json.loads(path.read_text()), ensure_ascii=True, separators=(",", ":"))` — never as the raw file. The README quickstarts use the same serialisation (`json.dumps(..., ensure_ascii=True, separators=(",", ":"))` for Cosmos Framework and TensorRT-LLM; `json.dumps(prompt)` for Diffusers and vLLM-Omni).

### 9.2 `create_payload`, per notebook

Common to all six: `payload_dir = <OUT>/<backend>/payloads/<use_case>` and `output_dir = <OUT>/<backend>/<use_case>` are created; the payload is written to `<payload_dir>/<use_case>.json` with `indent=2` and a trailing newline; `COSMOS3_<BACKEND>_<USE_CASE>_INPUT/_OUTPUT` are exported (§ 4.3); the function returns `(payload_path, output_dir, spec["model"])`. Where they differ:

| Notebook          | Base table                             | `sampling` handling                 | text2image                                                                                     | Negative prompt                                                   | Image / video reference                                                                                                                                               | Model-specific rules                                                                                          |
| ----------------- | -------------------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| cosmos_framework  | `FIXED_SAMPLING`                       | `**spec.get("sampling", {})` on top | `num_frames = 1`                                                                               | `""` for text2image, else per-mode file                           | `vision_path` (image2video)                                                                                                                                           | —                                                                                                             |
| diffusers         | `FIXED_SAMPLING`                       | same                                | `num_frames = 1`                                                                               | `""` for text2image, else spec `negative_prompt` or per-mode file | `vision_path` (image2video and video2video); video2video also sets `condition_frame_indexes_vision` (spec or `[0, 1]`) and `condition_video_keep` (spec or `"first"`) | `spec["distilled"]` → `payload["distilled"] = True` and `num_steps`, `guidance`, `shift` removed              |
| vllm_omni         | `FIXED_SAMPLING`                       | same                                | (no `num_frames` change)                                                                       | `""` for text2image, else per-mode file                           | `vision_path` (image2video)                                                                                                                                           | —                                                                                                             |
| sglang            | `FIXED_SAMPLING`                       | none (no `sampling` keys)           | (no `num_frames` change)                                                                       | `""` for text2image, else spec `negative_prompt` or per-mode file | `vision_path` (image2video and video2video)                                                                                                                           | Distil or Edge model → `guidance`, `shift` removed; Distil → `num_steps` removed; Edge → `resolution = "480"` |
| trt_llm           | `spec.get("sampling", FIXED_SAMPLING)` | replaces the base table             | `extra_params["output_type"] = "image"`; `num_frames`, `fps` removed; no `negative_prompt` key | spec `negative_prompt` or per-mode file (non-text2image only)     | `vision_path` (image2video), `video_path` (video2video)                                                                                                               | `extra_params` = `TRTLLM_EXTRA_PARAMS` + `enable_audio` + spec `extra_params`                                 |
| distilled trt-llm | `DISTILLED_SAMPLING`                   | `**spec.get("sampling", {})` on top | `num_frames = 1`, `fps = 8`, `seconds = 1`; no `negative_prompt` key                           | spec `negative_prompt` or per-mode file (non-text2image only)     | `vision_path` (image2video)                                                                                                                                           | `extra_params` = `TRTLLM_EXTRA_PARAMS` + spec `extra_params`                                                  |

```python
# run_with_cosmos_framework.ipynb
def create_payload(use_case: str, *, backend: str) -> tuple[Path, Path, str]:
    spec = ASSET_SETS[use_case]
    payload_dir = Path(os.environ["COSMOS3_AUDIOVISUAL_OUTPUT_ROOT"]) / backend / "payloads" / use_case
    output_dir = Path(os.environ["COSMOS3_AUDIOVISUAL_OUTPUT_ROOT"]) / backend / use_case
    payload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = asset_path(spec["prompt"])
    negative_prompt = ""
    if spec["mode"] != "text2image":
        negative_prompt_path = asset_path(f"assets/negative_prompts/{spec['mode']}/neg_prompt.json")
        negative_prompt = compact_json_file(negative_prompt_path)
    payload_path = payload_dir / f"{use_case}.json"
    payload = {
        "model_mode": spec["mode"],
        "name": use_case,
        "prompt": compact_json_file(prompt_path),
        "negative_prompt": negative_prompt,
        "enable_sound": spec["enable_sound"],
        **FIXED_SAMPLING,
        **spec.get("sampling", {}),
    }
    if spec["mode"] == "text2image":
        payload["num_frames"] = 1
    if spec["mode"] == "image2video":
        image_path = asset_path(spec["image"])
        payload["vision_path"] = os.path.relpath(image_path, payload_path.parent)

    payload_path.write_text(json.dumps(payload, indent=2) + "\n")

    os.environ[f"COSMOS3_{backend.upper()}_{use_case.upper()}_INPUT"] = str(payload_path)
    os.environ[f"COSMOS3_{backend.upper()}_{use_case.upper()}_OUTPUT"] = str(output_dir)

    print(f"model:   {spec['model']}")
    print(f"payload: {payload_path}")
    print(f"output:  {output_dir}")
    print(f"prompt:  {prompt_path.relative_to(COSMOS_ROOT)}")
    if "vision_path" in payload:
        image_display_path = resolve_payload_path(payload_path, payload["vision_path"])
        print(f"image:   {image_display_path.relative_to(COSMOS_ROOT)}")
    print(json.dumps({k: payload[k] for k in ["model_mode", "name", "enable_sound", "num_steps", "guidance", "shift", "fps", "num_frames", "resolution", "aspect_ratio", "seed"]}, indent=2))
    return payload_path, output_dir, spec["model"]
```

```python
# run_with_diffusers.ipynb
def create_payload(use_case: str, *, backend: str) -> tuple[Path, Path, str]:
    spec = ASSET_SETS[use_case]
    payload_dir = Path(os.environ["COSMOS3_AUDIOVISUAL_OUTPUT_ROOT"]) / backend / "payloads" / use_case
    output_dir = Path(os.environ["COSMOS3_AUDIOVISUAL_OUTPUT_ROOT"]) / backend / use_case
    payload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = asset_path(spec["prompt"])
    negative_prompt = ""
    if spec["mode"] != "text2image":
        # There is no negative_prompts/video2video directory; those cases name the file they reuse.
        negative_prompt_path = asset_path(
            spec.get("negative_prompt") or f"assets/negative_prompts/{spec['mode']}/neg_prompt.json"
        )
        negative_prompt = compact_json_file(negative_prompt_path)
    payload_path = payload_dir / f"{use_case}.json"
    payload = {
        "model_mode": spec["mode"],
        "name": use_case,
        "prompt": compact_json_file(prompt_path),
        "negative_prompt": negative_prompt,
        "enable_sound": spec["enable_sound"],
        **FIXED_SAMPLING,
        **spec.get("sampling", {}),
    }
    if spec.get("distilled"):
        payload["distilled"] = True
        # Distilled checkpoints resolve steps, guidance, and sigmas themselves; keeping these
        # in the payload would imply they are applied.
        for key in ("num_steps", "guidance", "shift"):
            payload.pop(key, None)
    if spec["mode"] == "text2image":
        payload["num_frames"] = 1
    if spec["mode"] == "image2video":
        image_path = asset_path(spec["image"])
        payload["vision_path"] = os.path.relpath(image_path, payload_path.parent)
    if spec["mode"] == "video2video":
        video_path = asset_path(spec["video"])
        payload["vision_path"] = os.path.relpath(video_path, payload_path.parent)
        payload["condition_frame_indexes_vision"] = spec.get("condition_frame_indexes_vision", [0, 1])
        payload["condition_video_keep"] = spec.get("condition_video_keep", "first")

    payload_path.write_text(json.dumps(payload, indent=2) + "\n")

    os.environ[f"COSMOS3_{backend.upper()}_{use_case.upper()}_INPUT"] = str(payload_path)
    os.environ[f"COSMOS3_{backend.upper()}_{use_case.upper()}_OUTPUT"] = str(output_dir)

    print(f"model:   {spec['model']}")
    print(f"payload: {payload_path}")
    print(f"output:  {output_dir}")
    print(f"prompt:  {prompt_path.relative_to(COSMOS_ROOT)}")
    if "vision_path" in payload:
        vision_display_path = resolve_payload_path(payload_path, payload["vision_path"])
        if payload["model_mode"] == "video2video":
            print(f"video:   {vision_display_path.relative_to(COSMOS_ROOT)}")
        else:
            print(f"image:   {vision_display_path.relative_to(COSMOS_ROOT)}")
    print(json.dumps({k: payload[k] for k in ["model_mode", "name", "enable_sound", "num_steps", "guidance", "shift", "fps", "num_frames", "resolution", "aspect_ratio", "seed"]}, indent=2))
    summary_keys = ["model_mode", "name", "enable_sound", "num_steps", "guidance", "shift", "fps", "num_frames", "resolution", "aspect_ratio", "seed"]
    print(json.dumps({k: payload[k] for k in summary_keys if k in payload}, indent=2))
    if payload.get("distilled"):
        print("distilled: step count and guidance come from the checkpoint config")
    return payload_path, output_dir, spec["model"]
```

```python
# run_with_vllm_omni.ipynb
def create_payload(use_case: str, *, backend: str) -> tuple[Path, Path, str]:
    spec = ASSET_SETS[use_case]
    payload_dir = Path(os.environ["COSMOS3_AUDIOVISUAL_OUTPUT_ROOT"]) / backend / "payloads" / use_case
    output_dir = Path(os.environ["COSMOS3_AUDIOVISUAL_OUTPUT_ROOT"]) / backend / use_case
    payload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = asset_path(spec["prompt"])
    negative_prompt = ""
    if spec["mode"] != "text2image":
        negative_prompt_path = asset_path(f"assets/negative_prompts/{spec['mode']}/neg_prompt.json")
        negative_prompt = compact_json_file(negative_prompt_path)
    payload_path = payload_dir / f"{use_case}.json"
    payload = {
        "model_mode": spec["mode"],
        "name": use_case,
        "prompt": compact_json_file(prompt_path),
        "negative_prompt": negative_prompt,
        "enable_sound": spec["enable_sound"],
        **FIXED_SAMPLING,
        **spec.get("sampling", {}),
    }
    if spec["mode"] == "image2video":
        image_path = asset_path(spec["image"])
        payload["vision_path"] = os.path.relpath(image_path, payload_path.parent)

    payload_path.write_text(json.dumps(payload, indent=2) + "\n")

    os.environ[f"COSMOS3_{backend.upper()}_{use_case.upper()}_INPUT"] = str(payload_path)
    os.environ[f"COSMOS3_{backend.upper()}_{use_case.upper()}_OUTPUT"] = str(output_dir)

    print(f"model:   {spec['model']}")
    print(f"payload: {payload_path}")
    print(f"output:  {output_dir}")
    print(f"prompt:  {prompt_path.relative_to(COSMOS_ROOT)}")
    if "vision_path" in payload:
        image_display_path = resolve_payload_path(payload_path, payload["vision_path"])
        print(f"image:   {image_display_path.relative_to(COSMOS_ROOT)}")
    print(json.dumps({k: payload[k] for k in ["model_mode", "name", "enable_sound", "num_steps", "guidance", "shift", "fps", "num_frames", "resolution", "aspect_ratio", "seed"]}, indent=2))
    return payload_path, output_dir, spec["model"]
```

```python
# run_with_sglang.ipynb
def _is_distil(model_name: str) -> bool:
    return "Distil" in model_name

def _is_edge(model_name: str) -> bool:
    return "Edge" in model_name


def create_payload(use_case: str, *, backend: str) -> tuple[Path, Path, str]:
    spec = ASSET_SETS[use_case]
    payload_dir = Path(os.environ["COSMOS3_AUDIOVISUAL_OUTPUT_ROOT"]) / backend / "payloads" / use_case
    output_dir = Path(os.environ["COSMOS3_AUDIOVISUAL_OUTPUT_ROOT"]) / backend / use_case
    payload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = asset_path(spec["prompt"])
    negative_prompt = ""
    if spec["mode"] != "text2image":
        negative_prompt_path = asset_path(f"assets/negative_prompts/{spec['mode']}/neg_prompt.json") if not spec.get("negative_prompt") else asset_path(spec["negative_prompt"])
        negative_prompt = compact_json_file(negative_prompt_path)
    payload_path = payload_dir / f"{use_case}.json"
    payload = {
        "model_mode": spec["mode"],
        "name": use_case,
        "prompt": compact_json_file(prompt_path),
        "negative_prompt": negative_prompt,
        "enable_sound": spec["enable_sound"],
        **FIXED_SAMPLING,
    }
    if spec["mode"] in {"image2video"}:
        image_path = asset_path(spec["image"])
        payload["vision_path"] = os.path.relpath(image_path, payload_path.parent)

    if spec["mode"] in {"video2video"}:
        video_path = asset_path(spec["video"])
        payload["vision_path"] = os.path.relpath(video_path, payload_path.parent)

    if _is_distil(spec["model"]) or _is_edge(spec["model"]):
        payload.pop("guidance", None)
        payload.pop("shift", None)

    if _is_distil(spec["model"]):
        payload.pop("num_steps", None)

    if _is_edge(spec["model"]):
        payload["resolution"] = "480"

    payload_path.write_text(json.dumps(payload, indent=2) + "\n")

    os.environ[f"COSMOS3_{backend.upper()}_{use_case.upper()}_INPUT"] = str(payload_path)
    os.environ[f"COSMOS3_{backend.upper()}_{use_case.upper()}_OUTPUT"] = str(output_dir)

    print(f"model:   {spec['model']}")
    print(f"payload: {payload_path}")
    print(f"output:  {output_dir}")
    print(f"prompt:  {prompt_path.relative_to(COSMOS_ROOT)}")
    if "vision_path" in payload:
        if payload["model_mode"] == "image2video":
            image_display_path = resolve_payload_path(payload_path, payload["vision_path"])
            print(f"image:   {image_display_path.relative_to(COSMOS_ROOT)}")
        elif payload["model_mode"] == "video2video":
            video_display_path = resolve_payload_path(payload_path, payload["vision_path"])
            print(f"video:   {video_display_path.relative_to(COSMOS_ROOT)}")
    print(json.dumps({k: payload[k] for k in ["model_mode", "name", "enable_sound", "num_steps", "guidance", "shift", "fps", "num_frames", "resolution", "aspect_ratio", "seed"] if k in payload}, indent=2))
    return payload_path, output_dir, spec["model"]
```

```python
# run_with_trt_llm.ipynb
def create_payload(use_case: str, *, backend: str = "trt_llm") -> tuple[Path, Path, str]:
    spec = ASSET_SETS[use_case]
    payload_dir = Path(os.environ["COSMOS3_AUDIOVISUAL_OUTPUT_ROOT"]) / backend / "payloads" / use_case
    output_dir = Path(os.environ["COSMOS3_AUDIOVISUAL_OUTPUT_ROOT"]) / backend / use_case
    payload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = asset_path(spec["prompt"])
    payload_path = payload_dir / f"{use_case}.json"
    extra_params = dict(TRTLLM_EXTRA_PARAMS)
    extra_params["enable_audio"] = bool(spec.get("enable_audio", False))
    extra_params.update(spec.get("extra_params", {}))
    if spec["mode"] == "text2image":
        extra_params["output_type"] = "image"
    payload = {
        "model_mode": spec["mode"],
        "name": use_case,
        "prompt": compact_json_file(prompt_path),
        "extra_params": extra_params,
        **spec.get("sampling", FIXED_SAMPLING),
    }
    if spec["mode"] == "text2image":
        payload.pop("num_frames", None)
        payload.pop("fps", None)
    else:
        negative_prompt_path = asset_path(
            spec.get("negative_prompt") or f"assets/negative_prompts/{spec['mode']}/neg_prompt.json"
        )
        payload["negative_prompt"] = compact_json_file(negative_prompt_path)
    if spec["mode"] == "image2video":
        image_path = asset_path(spec["image"])
        payload["vision_path"] = os.path.relpath(image_path, payload_path.parent)
    if spec["mode"] == "video2video":
        video_path = asset_path(spec["video"])
        payload["video_path"] = os.path.relpath(video_path, payload_path.parent)

    payload_path.write_text(json.dumps(payload, indent=2) + "\n")

    os.environ[f"COSMOS3_{backend.upper()}_{use_case.upper()}_INPUT"] = str(payload_path)
    os.environ[f"COSMOS3_{backend.upper()}_{use_case.upper()}_OUTPUT"] = str(output_dir)

    print(f"model:   {spec['model']}")
    print(f"payload: {payload_path}")
    print(f"output:  {output_dir}")
    print(f"prompt:  {prompt_path.relative_to(COSMOS_ROOT)}")
    if spec["mode"] == "text2image":
        print("note:   TensorRT-LLM Cosmos3 text-to-image uses /v1/images/generations")
    if "vision_path" in payload:
        image_display_path = resolve_payload_path(payload_path, payload["vision_path"])
        print(f"image:   {image_display_path.relative_to(COSMOS_ROOT)}")
    if "video_path" in payload:
        video_display_path = resolve_payload_path(payload_path, payload["video_path"])
        print(f"video:   {video_display_path.relative_to(COSMOS_ROOT)}")
        print("note:    condition_video_latent_indexes=[0, 1] consumes five input pixel frames")
    if payload["extra_params"]["enable_audio"]:
        print("audio:   enabled; ffmpeg is required on the server to mux it into MP4")
    preview_keys = ["model_mode", "name", "num_steps", "guidance"]
    if spec["mode"] != "text2image":
        preview_keys.extend(["fps", "num_frames"])
    preview_keys.extend(["max_sequence_length", "resolution", "aspect_ratio", "seed", "extra_params"])
    if "seconds" in payload:
        preview_keys.insert(5, "seconds")
    print(json.dumps({k: payload[k] for k in preview_keys}, indent=2))
    return payload_path, output_dir, spec["model"]
```

```python
# run_distilled_with_trt_llm.ipynb
def create_payload(use_case: str, *, backend: str = "trt_llm_distilled") -> tuple[Path, Path, str]:
    spec = ASSET_SETS[use_case]
    payload_dir = Path(os.environ["COSMOS3_AUDIOVISUAL_OUTPUT_ROOT"]) / backend / "payloads" / use_case
    output_dir = Path(os.environ["COSMOS3_AUDIOVISUAL_OUTPUT_ROOT"]) / backend / use_case
    payload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = asset_path(spec["prompt"])
    payload_path = payload_dir / f"{use_case}.json"
    extra_params = dict(TRTLLM_EXTRA_PARAMS)
    extra_params.update(spec.get("extra_params", {}))
    payload = {
        "model_mode": spec["mode"],
        "name": use_case,
        "prompt": compact_json_file(prompt_path),
        "extra_params": extra_params,
        **DISTILLED_SAMPLING,
        **spec.get("sampling", {}),
    }
    if spec["mode"] == "text2image":
        payload["num_frames"] = 1
        payload["fps"] = 8
        payload["seconds"] = 1
    else:
        negative_prompt_path = asset_path(
            spec.get("negative_prompt") or f"assets/negative_prompts/{spec['mode']}/neg_prompt.json"
        )
        payload["negative_prompt"] = compact_json_file(negative_prompt_path)
    if spec["mode"] == "image2video":
        image_path = asset_path(spec["image"])
        payload["vision_path"] = os.path.relpath(image_path, payload_path.parent)

    payload_path.write_text(json.dumps(payload, indent=2) + "\n")

    os.environ[f"COSMOS3_{backend.upper()}_{use_case.upper()}_INPUT"] = str(payload_path)
    os.environ[f"COSMOS3_{backend.upper()}_{use_case.upper()}_OUTPUT"] = str(output_dir)

    print(f"model:   {spec['model']}")
    print(f"payload: {payload_path}")
    print(f"output:  {output_dir}")
    print(f"prompt:  {prompt_path.relative_to(COSMOS_ROOT)}")
    print("steps:   fixed by the checkpoint (4); guidance is baked in — neither is sent")
    if spec["mode"] == "text2image":
        print("note:    TensorRT-LLM Cosmos3 text-to-image is served as a one-frame video response")
    if "vision_path" in payload:
        image_display_path = resolve_payload_path(payload_path, payload["vision_path"])
        print(f"image:   {image_display_path.relative_to(COSMOS_ROOT)}")
    preview_keys = ["model_mode", "name", "fps", "num_frames", "max_sequence_length", "resolution", "aspect_ratio", "seed", "extra_params"]
    if "seconds" in payload:
        preview_keys.insert(4, "seconds")
    print(json.dumps({k: payload[k] for k in preview_keys}, indent=2))
    return payload_path, output_dir, spec["model"]
```

### 9.3 Helpers shared by both TensorRT-LLM notebooks (`run_with_trt_llm.ipynb`, `run_distilled_with_trt_llm.ipynb`), verbatim

```python
import time
import urllib.error
import urllib.request


def check_trtllm_server(model: str, timeout_s: int = 1800, interval_s: int = 10) -> None:
    url = health_url(TRTLLM_ENDPOINTS[model])
    deadline = time.time() + timeout_s
    print(f"waiting for {model} server: {url}")
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    print(f"{model} server is ready")
                    return
                last_error = f"HTTP {response.status}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        print(f"not ready yet: {last_error}")
        time.sleep(interval_s)
    raise TimeoutError(f"Timed out waiting for {model} server at {url}. Last error: {last_error}")


def _auth_headers() -> list[str]:
    api_key = os.environ.get("COSMOS3_TRTLLM_API_KEY", "")
    return ["-H", f"Authorization: Bearer {api_key}"] if api_key else []
```

## 10. Payload field → request field, per backend

How the request helpers of each notebook translate the payload JSON written by `create_payload` (the helpers themselves — `build_*`, `post_*`, `run_*_payload`, `get_pipe`/`run_diffusers_payload`, the `torchrun` cells — are in the backend files).

| Payload key                                    | vllm_omni (`/v1/videos/sync` multipart form; `/v1/images/generations` JSON)    | sglang (`/v1/videos` multipart form, async; `/v1/images/generations` JSON)                    | trt_llm (`/v1/videos/sync`; `/v1/images/generations`)                                                                                                                        | distilled trt-llm (`/v1/videos/generations`)                                                              | diffusers (`Cosmos3OmniPipeline` / `Cosmos3DistilledModularPipeline` kwargs)                                                                                         | cosmos_framework (`-i payload.json` + CLI)                                                                             |
| ---------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `prompt`                                       | `prompt`                                                                       | `prompt`                                                                                      | `prompt`                                                                                                                                                                     | `prompt`                                                                                                  | `prompt=`                                                                                                                                                            | payload field                                                                                                          |
| `negative_prompt`                              | video: `negative_prompt`; image body: not sent                                 | video: `negative_prompt`; image body: `payload.get("negative_prompt", "")`                    | sent when present (`is not None`)                                                                                                                                            | sent when present                                                                                         | video: `negative_prompt=payload["negative_prompt"]`; text2image: `negative_prompt=""`; distilled pipeline: not forwarded                                             | payload field                                                                                                          |
| `resolution` + `aspect_ratio`                  | `size` = `f"{width}x{height}"` from `payload_dimensions`                       | same                                                                                          | same                                                                                                                                                                         | same                                                                                                      | `height=`, `width=` from `payload_dimensions`                                                                                                                        | payload fields; distilled runs also pass `--resolution=768 --aspect-ratio=1,1` / `--resolution=480 --aspect-ratio=4,3` |
| `num_frames`                                   | `num_frames` (string)                                                          | `num_frames` (string)                                                                         | video: `num_frames` and `seconds` = `payload.get("seconds", num_frames / fps)`; image: none                                                                                  | `num_frames` and `seconds` (text2image: `num_frames=1`, `seconds=1`)                                      | `num_frames=` (`1` for text2image)                                                                                                                                   | payload field; distilled runs `--num-frames=1` / `--num-frames=121`                                                    |
| `fps`                                          | `fps` (string)                                                                 | `fps` (string)                                                                                | video: `fps`; image: none                                                                                                                                                    | `fps` (text2image: `8`)                                                                                   | `fps=` (video modes; distilled image2video) and `export_to_video(..., fps=payload["fps"], macro_block_size=1)` / `encode_video(..., fps=payload["fps"], ...)`        | payload field; distilled runs `--fps=30`                                                                               |
| `num_steps`                                    | `num_inference_steps`                                                          | `num_inference_steps` when present                                                            | `num_inference_steps` when `is not None`                                                                                                                                     | never sent (checkpoint-fixed)                                                                             | `num_inference_steps=`; distilled pipeline: not forwarded                                                                                                            | payload field                                                                                                          |
| `guidance`                                     | `guidance_scale`                                                               | `guidance_scale` when present                                                                 | `guidance_scale` when `is not None`                                                                                                                                          | never sent (baked in)                                                                                     | `guidance_scale=`; distilled: not forwarded                                                                                                                          | payload field; distilled runs `--guidance=1.0`                                                                         |
| `shift`                                        | `flow_shift`                                                                   | `flow_shift` when present                                                                     | not a request field                                                                                                                                                          | not a request field                                                                                       | `UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=payload["shift"])` (non-distilled)                                                            | payload field                                                                                                          |
| `seed`                                         | `seed`                                                                         | `seed`                                                                                        | `seed`                                                                                                                                                                       | `seed`                                                                                                    | `torch.Generator(device="cuda").manual_seed(payload["seed"])`                                                                                                        | payload field and `--seed=0` (`--seed=1` on the distilled runs)                                                        |
| `max_sequence_length`                          | —                                                                              | `"4096"` (form) / `4096` (image body), hard-coded                                             | `max_sequence_length` (payload, 4096)                                                                                                                                        | `max_sequence_length` (payload, 4096)                                                                     | —                                                                                                                                                                    | —                                                                                                                      |
| `enable_sound`                                 | `generate_sound="true"` + `sound_duration=f"{num_frames / fps:.3f}"` when true | same                                                                                          | — (audio is `extra_params.enable_audio`)                                                                                                                                     | — (`enable_audio` is `False` in `TRTLLM_EXTRA_PARAMS`)                                                    | `enable_sound=`; with sound, `encode_video(result.video, fps=..., output_path=..., audio=result.sound, audio_sample_rate=pipe.sound_tokenizer.config.sampling_rate)` | payload field                                                                                                          |
| `extra_params` / templates                     | video `extra_params` and image `extra_args` as in § 6.3                        | video/image `extra_params` as in § 6.3                                                        | `extra_params` from the payload (JSON object, or compact-JSON string in multipart)                                                                                           | same                                                                                                      | `add_resolution_template=False`, `add_duration_template=False`; distilled also `output="videos"`                                                                     | —                                                                                                                      |
| image reference (`vision_path`)                | `-F input_reference=@<path>`                                                   | `-F input_reference=@<path>`                                                                  | `--form-string` fields + `-F image_reference=@<path>;type=image/jpeg`                                                                                                        | `--form-string` fields + `-F input_reference=@<path>;type=image/jpeg`                                     | `image=load_image(<path>)`                                                                                                                                           | `vision_path` in the payload                                                                                           |
| video reference (`vision_path` / `video_path`) | —                                                                              | `-F video_reference=@<path>;type=video/mp4`                                                   | `-F video_reference=@<path>;type=video/mp4` with `extra_params.condition_video_latent_indexes` / `condition_video_keep`                                                      | —                                                                                                         | `video=load_video(<path>)`, `condition_frame_indexes_vision=payload.get(..., [0, 1])`, `condition_video_keep=payload.get(..., "first")`                              | —                                                                                                                      |
| response                                       | video: MP4 bytes (`Accept: video/mp4`); image: `data[0].b64_json`              | video: `id` → poll → `/content` MP4; image: `response_format: "b64_json"`, `data[0].b64_json` | video: `format: "mp4"`, `response_format: "file"`, must be `video/mp4` with `ftyp`; image: `format: "png"`, `response_format: "b64_json"`, must start with the PNG signature | video bytes (`Accept: video/mp4, video/x-msvideo, application/octet-stream`), `.mp4`/`.avi` by inspection | `result.video` / `result.sound`; distilled returns `videos` (`videos[0].save` for text2image)                                                                        | files written by the inference script under `-o`                                                                       |

The NIM uses its own schema (`model_mode` — required —, `prompt`, `negative_prompt`, `input_reference`, `seed`, `guidance_scale`, `num_inference_steps`, `flow_shift`, `resolution`, `num_frames`, `fps`, plus the V2V / Action / Transfer fields; one JSON object to `POST /v1/infer`; video returned base64-encoded in `b64_video`, text2image as JPEG in `b64_image`); it is documented in the NIM cookbook (`../../cookbooks/cosmos3/nim/api-reference.md`, Generator Endpoint). The retired 1.0.0 notebook's fields `image`, `steps` and `num_output_frames` are not accepted by the 2.1.0 image.

## 11. Workflow

Every `create_payload` notebook runs each use case as the same three steps — "Create Payload" (`create_payload(<use_case>, backend=...)`), "Run" (`run_<backend>_payload(payload_path, output_dir, model=...)`; the two TensorRT-LLM notebooks call `check_trtllm_server(model)` first), "View Results" (`view_run(output_dir)`, whose discovery rule is in § 4.2; the inline HTML viewer itself is Jupyter-only and is not migrated). The notebooks' own wording:

- `run_with_cosmos_framework.ipynb`: Run each use case top-to-bottom: create the JSON payload, run inference, then view the generated media. The published four-step students use four-GPU sharding for memory and are smoke examples, not production-quality reproduction claims.
- `run_with_diffusers.ipynb`: Run each use case top-to-bottom: create the JSON payload, run inference, then view the generated media. Nano examples come first; Super examples are last.
- `run_with_vllm_omni.ipynb`, `run_with_sglang.ipynb`: Run each use case top-to-bottom: create the JSON payload, run inference, then view the generated media. The examples are grouped by model size below. The Cosmos3-Nano and Cosmos3-Super sections are independent, so you can run just one.
- `run_with_trt_llm.ipynb`: Run each use case top-to-bottom: create the JSON payload, wait for the matching server, run inference, then view the generated media. The Cosmos3-Nano and Cosmos3-Super sections are independent, so you can run just one.
- `run_distilled_with_trt_llm.ipynb`: Run each use case top-to-bottom: create the JSON payload, wait for the matching server, run inference, then view the generated media. The two sections are independent, so you can run just one.
