# TensorRT-LLM — distilled from the cookbook notebook

Scope: the TensorRT-LLM VisualGen server flow of `run_with_trt_llm.ipynb` — Cosmos3-Nano, Cosmos3-Super and Cosmos3-Edge text-to-image, text-to-video and image-to-video with or without synchronized audio, and video-to-video — plus the README's "Run with TensorRT-LLM" Quickstart and Notebook walkthrough; the cells this notebook shares with the other backend notebooks are held by payloads.md and are referenced here by section.

## Cosmos3 Generator Audiovisual with TensorRT-LLM

SPDX licence header: see payloads.md § 2.1.

This notebook calls already-running TensorRT-LLM VisualGen servers with direct `curl` requests from Python.

The examples are split into Cosmos3-Nano, Cosmos3-Super, and Cosmos3-Edge sections. Each section is self-contained, so you can run just one. The notebook covers TensorRT-LLM's stable server flow for text-to-image, text-to-video and image-to-video with or without synchronized audio, and video-to-video generation.

`Cosmos3-Edge` is the compact single-GPU checkpoint. TensorRT-LLM serves it for text-to-image, text-to-video, and image-to-video only: Edge has no audio tower, its action weights are not served by this pipeline, and video-to-video is validated for Nano and Super only.

## 1. Prerequisites

Use a running TensorRT-LLM server with Cosmos3 VisualGen audio and video-to-video support, and set endpoint environment variables before the setup cell if you are not using the local default. Text-to-image uses `/v1/images/generations`; video generation uses `/v1/videos/sync`. Image-to-video uploads multipart `image_reference`, while video-to-video uploads multipart `video_reference`.

Install the `ffmpeg` CLI in the server environment. This notebook explicitly requests browser-compatible MP4, so TensorRT-LLM returns an actionable error if the encoder is unavailable; `ffmpeg` also muxes synchronized audio.

Guardrail prerequisite: see payloads.md § 2.3 (`run_with_trt_llm.ipynb` row).

Endpoint environment variables and their local defaults (`COSMOS3_TRTLLM_BASE_URL`, `COSMOS3_TRTLLM_NANO_BASE_URL`, `COSMOS3_TRTLLM_SUPER_BASE_URL`, `COSMOS3_TRTLLM_EDGE_BASE_URL`, `COSMOS3_TRTLLM_API_KEY`): see payloads.md § 4.5 (`run_with_trt_llm.ipynb` export block).

## 2. Start the Server

Run the TensorRT-LLM VisualGen server before running the request cells. The config YAMLs below come from TensorRT-LLM's Cosmos3 support. To build a checkout from source, follow NVIDIA's [Build from Source](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source.html) guide and then run the commands below from that checkout. Install `ffmpeg` in that environment before serving (`apt-get update && apt-get install -y ffmpeg` on Ubuntu/Debian images).

Skill note (not in the notebook): the stable PyPI `tensorrt-llm` 1.2.x has no VisualGen (`trtllm-serve` rejects `--visual_gen_args`, and Edge fails in the generic LLM loader with KeyError/ValueError `cosmos3_edge`), so `pip install tensorrt-llm` is not a route. The prebuilt-wheel route verified by the skill's eval runs is `tensorrt_llm==1.3.0rc27` — the first release that contains the pinned commit bca6761 — with `torch==2.13.0+cu130` and `torchvision==0.28.0+cu130`, from `https://pypi.nvidia.com` plus the cu130 torch index: `uv pip install --no-config --index-strategy unsafe-best-match --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu130 "tensorrt_llm==1.3.0rc27" "torch==2.13.0+cu130" "torchvision==0.28.0+cu130"` (PyPI hosts only sdists of tensorrt_llm, so both flags are required; the rc27 binaries are built against torch 2.13). Do not use 1.3.0rc26 for the image route — its `/v1/images/generations` returns HTTP 500 (see the skill note after the Edge block). Outside the devel container a pip-installed wheel also needs system OpenMPI (`apt-get update && apt-get install -y libopenmpi-dev openmpi-bin ffmpeg`), a seeded venv (`uv venv … --python 3.12 --seed --managed-python --allow-existing --no-config`, so `pip` exists), its CUDA-13 libraries from the cu130 torch index (never symlink `.so.12` to `.so.13`) and a `CUDA_HOME` — point it at the venv's `nvidia/cu13` when no toolkit is installed: `export CUDA_HOME="$(python -c 'import nvidia,os;print(os.path.join(nvidia.__path__[0],"cu13"))')"`; the upstream TensorRT-LLM [Installing on Linux via pip](https://nvidia.github.io/TensorRT-LLM/installation/linux.html) page documents the `libopenmpi-dev` and cu130 torch steps. The upstream `TRTLLM_USE_PRECOMPILED=1 pip install -e .` shortcut does not work for the pinned commit bca6761 (it resolves the unpublished `1.3.0rc26+bca6761ab8`; a wheel path or URL belongs in `TRTLLM_PRECOMPILED_LOCATION`; and the published rc26 binaries lack `trtllm::silu_and_mul_fp8_quantize_1x128_packed_ue8m0`, which the pinned Python needs) — install a complete wheel ≥ 1.3.0rc27 or do the full devel-container build, and keep the checkout only for `examples/visual_gen/configs`. Full recipe: § 7.2 below (Route A, the wheel; § 7.4 is the source build).

### Cosmos3-Nano

From inside the TensorRT-LLM checkout — the directory that contains `examples/` — with TensorRT-LLM installed in the active environment:

```bash
export TRTLLM_ROOT="${TRTLLM_ROOT:-$PWD}"

trtllm-serve nvidia/Cosmos3-Nano \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-nano-1gpu.yaml" \
  --port 8000
```

Skill note (not in the notebook; measured by the skill's eval runs on one H100 80 GB, 2026-10-21/23 — not a notebook figure): the shipped `cosmos3-nano-1gpu.yaml` fits one 80 GB GPU as is — about 50–52 GB peak (50,623–52,147 MiB in `nvidia-smi`) after the default 720x1280x189 warm-up, `/health` ready in ~110–180 s (the long end includes the Hugging Face weight download), and a 1280x720 35-step text-to-image request takes ~6 s. Memory for this pipeline is controlled by the `--visual_gen_args` YAML, not by `trtllm-serve`'s LLM flags (`--max_batch_size`, KV-cache fraction). For text-to-image-only serving on smaller GPUs, copy the YAML, keep `attention_config` VANILLA and `cfg_size=ulysses_size=1`, and restrict the warm-up with `compilation_config: {resolutions: [[192, 320]], num_frames: [1]}` — one eval run sampled 37,705 MiB peak against 52,147 MiB with the default warm-up for the same 1280x720 request (not a controlled benchmark); a `[[720, 1280]]` x 1 warm-up is the natural variant but was not exercised.

### Cosmos3-Super

`Cosmos3-Super` uses the four-GPU config from TensorRT-LLM. The config sets `cfg_size=2`, `ulysses_size=2`, and `parallel_vae_size=4`, so launch exactly four processes.

```bash
export TRTLLM_ROOT="${TRTLLM_ROOT:-$PWD}"

torchrun --nproc_per_node=4 -m tensorrt_llm.commands.serve \
  nvidia/Cosmos3-Super \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-super-4gpu.yaml" \
  --port 8000
```

### Cosmos3-Edge

`Cosmos3-Edge` is the compact 4B Nemotron-dense checkpoint and fits on a single GPU. Its 480p-native generation defaults — 832x480 with 121 frames, 50 UniPC steps on the checkpoint-declared native flow schedule, guidance 5.0, flow shift 3.0 — are read from the checkpoint, so Edge needs no config override:

```bash
trtllm-serve nvidia/Cosmos3-Edge --port 8000
```

TensorRT-LLM serves Edge for text-to-image, text-to-video, and image-to-video. Text-to-image goes to `/v1/images/generations` with `output_type="image"`; the two video modes go to `/v1/videos/sync`. Edge has no audio tower, so `enable_audio` is unavailable, and the checkpoint's action weights are not served by this pipeline. Requests outside the model card's validated envelope (256p/480p, 50-150 frames, 12-30 FPS) still run and log an advisory line.

TensorRT-LLM exposes `/health` when the server is ready. This notebook sends Cosmos3 model-specific controls through `extra_params`, so use a TensorRT-LLM release that includes the Cosmos3 VisualGen API schema.

> Skill note (not in the notebook): that release is tensorrt_llm 1.3.0rc27 or later (tested: 1.3.0rc27 with torch 2.13.0+cu130; 1.4.0rc1 also served t2i). 1.2.x has no VisualGen module; 1.3.0rc26 — the version string of the pinned checkout — returns HTTP 500 on `/v1/images/generations` (NameError `get_steady_clock_now_in_seconds`; a 200 on `/health` does not exercise the image route); 1.3.0rc29 and the 1.4.0rc* line require torch 2.14, a different pairing from this stack's torch 2.13.0+cu130 — do not float to them (1.4.0rc1 served Nano t2i but is unverified for Edge and the distilled students). Complete a `/v1/images/generations` smoke request (HTTP 200 plus a decoded PNG) before trusting readiness.

## 3. Configure Paths and Endpoints

This setup cell only configures repo/output paths and TensorRT-LLM endpoint settings.

Paths: a plain `OUT=outputs` replaces the notebook's `find_repo_root()` / `COSMOS3_AUDIOVISUAL_ROOT` / `COSMOS3_AUDIOVISUAL_OUTPUT_ROOT` plumbing — see payloads.md § 4.1 (its configuration block sets `COSMOS3_AUDIOVISUAL_OUTPUT_ROOT` to `OUT`, so `create_payload` runs unchanged). Endpoints and API key (`DEFAULT_TRTLLM_BASE_URL`; `TRTLLM_ENDPOINTS` for `Cosmos3-Nano`, `Cosmos3-Super`, `Cosmos3-Edge`; `os.environ.setdefault("COSMOS3_TRTLLM_API_KEY", "tensorrt_llm")`): see payloads.md § 4.5 (`run_with_trt_llm.ipynb` block).

## 4. Verify Endpoint Configuration

`api_root_url`, `server_root_url` and `health_url`: see payloads.md § 4.5. The two route helpers of this notebook:

```python
def video_api_url(base_url: str) -> str:
    return f"{api_root_url(base_url)}/videos/sync"


def image_api_url(base_url: str) -> str:
    return f"{api_root_url(base_url)}/images/generations"
```

## 5. Preview Available Inputs

See payloads.md § 3 (asset catalogue; § 3.5 holds this cell's preview rules, including its `assets/videos` listing).

## 6. Define Asset Sets, Payload Helpers, Request Helpers, and Viewer Helpers

The cell defines, in this order:

1. the imports — below;
2. `api_root_url`, `server_root_url`, `health_url`, `video_api_url`, `image_api_url` — a second, identical copy of § 4 (kept once, above);
3. `IMAGE_EXTENSIONS` — see payloads.md § 9.1;
4. `FIXED_SAMPLING`, `EDGE_VIDEO_SAMPLING`, `EDGE_T2I_SAMPLING`, with their comments — see payloads.md § 6.1 (`run_with_trt_llm.ipynb` block);
5. `TRTLLM_EXTRA_PARAMS` — see payloads.md § 6.3;
6. `ASSET_SETS`, with its comment — see payloads.md § 8.7 (`run_with_trt_llm.ipynb` block) and the use-case table below;
7. `asset_path`, `compact_json_file` — see payloads.md § 9.1;
8. `payload_dimensions` — see payloads.md § 7 (`run_with_trt_llm.ipynb` block);
9. `resolve_payload_path` — see payloads.md § 9.1;
10. `create_payload` — see payloads.md § 9.2 (`run_with_trt_llm.ipynb` block);
11. `check_trtllm_server` — see payloads.md § 9.3;
12. `build_trtllm_video_body`, `build_trtllm_image_body` — below;
13. `_auth_headers` — see payloads.md § 9.3;
14. `_require_mp4_response`, `post_video`, `post_image`, `run_trtllm_payload` — below;
15. `display_video`, `view_run` — Jupyter viewer helpers, not migrated; the output discovery rule of `view_run` is in payloads.md § 4.2.

```python
import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def build_trtllm_video_body(payload: dict) -> dict:
    height, width = payload_dimensions(payload)
    body = {
        "prompt": payload["prompt"],
        "size": f"{width}x{height}",
        "seconds": payload.get("seconds", payload["num_frames"] / payload["fps"]),
        "fps": payload["fps"],
        "num_frames": payload["num_frames"],
        "max_sequence_length": payload["max_sequence_length"],
        "seed": payload["seed"],
        "format": "mp4",
        "response_format": "file",
    }
    if payload.get("num_steps") is not None:
        body["num_inference_steps"] = payload["num_steps"]
    if payload.get("guidance") is not None:
        body["guidance_scale"] = payload["guidance"]
    if payload.get("negative_prompt") is not None:
        body["negative_prompt"] = payload["negative_prompt"]
    body["extra_params"] = payload["extra_params"]
    return body


def build_trtllm_image_body(payload: dict) -> dict:
    height, width = payload_dimensions(payload)
    body = {
        "prompt": payload["prompt"],
        "size": f"{width}x{height}",
        "format": "png",
        "response_format": "b64_json",
        "max_sequence_length": payload["max_sequence_length"],
        "seed": payload["seed"],
        "extra_params": payload["extra_params"],
    }
    if payload.get("num_steps") is not None:
        body["num_inference_steps"] = payload["num_steps"]
    if payload.get("guidance") is not None:
        body["guidance_scale"] = payload["guidance"]
    if payload.get("negative_prompt") is not None:
        body["negative_prompt"] = payload["negative_prompt"]
    return body


# _auth_headers: see payloads.md § 9.3


def _require_mp4_response(header_text: str, content_path: Path) -> None:
    if "video/mp4" not in header_text.lower():
        raise RuntimeError(f"Expected video/mp4 response, got headers:\n{header_text}")
    head = content_path.read_bytes()[:16]
    if len(head) < 12 or head[4:8] != b"ftyp":
        raise RuntimeError("TensorRT-LLM returned a non-MP4 payload")


def post_video(*, payload_path: Path, payload: dict, output_stem: Path, model: str) -> Path:
    url = video_api_url(TRTLLM_ENDPOINTS[model])
    tmp_path = Path(f"{output_stem}.tmp")
    header_path = Path(f"{output_stem}.headers.txt")
    error_path = Path(f"{output_stem}.error.txt")
    for path in [tmp_path, header_path, error_path]:
        if path.exists():
            path.unlink()

    body = build_trtllm_video_body(payload)
    cmd = [
        "curl",
        "-sS",
        "--fail-with-body",
        "-X",
        "POST",
        url,
        "-D",
        str(header_path),
        "-H",
        "Accept: video/mp4",
    ]
    cmd += _auth_headers()

    if payload["model_mode"] in {"image2video", "video2video"}:
        form_body = dict(body)
        if "extra_params" in form_body:
            form_body["extra_params"] = json.dumps(form_body["extra_params"], separators=(",", ":"))
        for key, value in form_body.items():
            cmd += ["--form-string", f"{key}={value}"]
        if payload["model_mode"] == "image2video":
            reference_path = resolve_payload_path(payload_path, payload["vision_path"])
            reference_type = "image/jpeg"
            reference_field = "image_reference"
        else:
            reference_path = resolve_payload_path(payload_path, payload["video_path"])
            reference_type = "video/mp4"
            reference_field = "video_reference"
        cmd += ["-F", f"{reference_field}=@{reference_path};type={reference_type}"]
    else:
        cmd += ["-H", "Content-Type: application/json"]
        cmd += ["-d", json.dumps(body, separators=(",", ":"))]

    cmd += ["-o", str(tmp_path)]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        error_path.write_text((result.stdout or "") + (result.stderr or ""))
        raise RuntimeError(f"TensorRT-LLM request failed with exit code {result.returncode}; see {error_path}")

    header_text = header_path.read_text() if header_path.exists() else ""
    _require_mp4_response(header_text, tmp_path)
    output_path = output_stem.with_suffix(".mp4")
    if output_path.exists():
        output_path.unlink()
    tmp_path.replace(output_path)
    return output_path


def post_image(*, payload: dict, output_stem: Path, model: str) -> Path:
    url = image_api_url(TRTLLM_ENDPOINTS[model])
    response_path = Path(f"{output_stem}.response.json")
    error_path = Path(f"{output_stem}.error.txt")
    for path in [response_path, error_path]:
        if path.exists():
            path.unlink()

    cmd = [
        "curl",
        "-sS",
        "--fail-with-body",
        "-X",
        "POST",
        url,
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(build_trtllm_image_body(payload), separators=(",", ":")),
        "-o",
        str(response_path),
    ]
    cmd += _auth_headers()
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        server_body = response_path.read_text() if response_path.exists() else ""
        error_path.write_text(server_body + (result.stdout or "") + (result.stderr or ""))
        raise RuntimeError(f"TensorRT-LLM request failed with exit code {result.returncode}; see {error_path}")

    response = json.loads(response_path.read_text())
    try:
        image_bytes = base64.b64decode(response["data"][0]["b64_json"], validate=True)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Unexpected TensorRT-LLM image response: {response}") from exc
    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("TensorRT-LLM returned a non-PNG image payload")
    output_path = output_stem.with_suffix(".png")
    output_path.write_bytes(image_bytes)
    return output_path


def run_trtllm_payload(payload_path: Path, output_dir: str | Path, *, model: str) -> Path:
    payload_path = Path(payload_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(payload_path.read_text())
    output_stem = output_dir / payload["name"]
    endpoint = (
        image_api_url(TRTLLM_ENDPOINTS[model])
        if payload["model_mode"] == "text2image"
        else video_api_url(TRTLLM_ENDPOINTS[model])
    )
    print("endpoint:", endpoint)
    print("payload:", payload_path)
    print("output stem:", output_stem)
    if payload["model_mode"] == "image2video":
        print("input image:", resolve_payload_path(payload_path, payload["vision_path"]))
    if payload["model_mode"] == "video2video":
        print("input video:", resolve_payload_path(payload_path, payload["video_path"]))
    t0 = time.time()
    if payload["model_mode"] == "text2image":
        output_path = post_image(payload=payload, output_stem=output_stem, model=model)
    else:
        output_path = post_video(payload_path=payload_path, payload=payload, output_stem=output_stem, model=model)
    print(f"wrote {output_path} in {time.time() - t0:.1f}s")
    return output_path
```

Workflow sentence that closes the cell: see payloads.md § 11 (`run_with_trt_llm.ipynb` bullet).

## Use cases (collapsed Create Payload / Run / View Results cells)

Every use case ran the same three cells with its own key (the notebook names the variables `<use_case>_payload`, `<use_case>_output`, `<use_case>_model`); shown for `t2i_nano`:

```python
# Create Payload
t2i_nano_payload, t2i_nano_output, t2i_nano_model = create_payload("t2i_nano")

# Run
check_trtllm_server(t2i_nano_model)
run_trtllm_payload(t2i_nano_payload, t2i_nano_output, model=t2i_nano_model)

# View Results: view_run(t2i_nano_output) — Jupyter viewer, not migrated; its output discovery rule is in payloads.md § 4.2
```

The table lists every spec exactly as `ASSET_SETS` defines it (verbatim in payloads.md § 8.7). What `create_payload` adds on top (`TRTLLM_EXTRA_PARAMS`, `enable_audio`, `output_type` for `text2image`, the per-mode negative prompt, the `sampling` tables): see payloads.md § 6.1–6.3, § 3.3 and § 9.2 (`run_with_trt_llm.ipynb` block); payload field → request field: see payloads.md § 10; output naming and the exported `COSMOS3_TRT_LLM_*` variables: see payloads.md § 4.2 and § 4.3.

| Use case     | Notebook section                    | `model`         | `mode`        | `prompt`                                                   | `image` / `video`                                                  | `negative_prompt`                                     | `enable_audio` | `sampling` (absent → `FIXED_SAMPLING`) | `extra_params`                                                                                           | Output (`<OUT>/trt_llm/<use_case>/`) |
| ------------ | ----------------------------------- | --------------- | ------------- | ---------------------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------- | -------------- | -------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `t2i_nano`   | Nano: Text to Image                 | `Cosmos3-Nano`  | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                                  | —                                                     | —              | —                                      | —                                                                                                        | `t2i_nano.png`                       |
| `t2v_nano`   | Nano: Text to Video Without Audio   | `Cosmos3-Nano`  | `text2video`  | `assets/prompts/text2video/robot_kitchen.json`             | —                                                                  | —                                                     | —              | —                                      | —                                                                                                        | `t2v_nano.mp4`                       |
| `t2av_nano`  | Nano: Text to Video with Audio      | `Cosmos3-Nano`  | `text2video`  | `assets/prompts/text2video/robot_pouring_water_audio.json` | —                                                                  | —                                                     | `True`         | —                                      | —                                                                                                        | `t2av_nano.mp4`                      |
| `i2v_nano`   | Nano: Image to Video Without Audio  | `Cosmos3-Nano`  | `image2video` | `assets/prompts/image2video/humanoid_robot.json`           | `assets/images/image2video/humanoid_robot.jpg`                     | —                                                     | —              | —                                      | —                                                                                                        | `i2v_nano.mp4`                       |
| `i2av_nano`  | Nano: Image to Video with Audio     | `Cosmos3-Nano`  | `image2video` | `assets/prompts/image2video/coastal_road_audio.json`       | `assets/images/image2video/coastal_road_audio.jpg`                 | —                                                     | `True`         | —                                      | —                                                                                                        | `i2av_nano.mp4`                      |
| `v2v_nano`   | Nano: Video to Video                | `Cosmos3-Nano`  | `video2video` | `assets/prompts/image2video/car_driving.json`              | `assets/videos/car_driving_plain.mp4` (not vendored in this skill) | `assets/negative_prompts/image2video/neg_prompt.json` | —              | —                                      | `{"use_system_prompt": True, "condition_video_latent_indexes": [0, 1], "condition_video_keep": "first"}` | `v2v_nano.mp4`                       |
| `t2i_super`  | Super: Text to Image                | `Cosmos3-Super` | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                                  | —                                                     | —              | —                                      | —                                                                                                        | `t2i_super.png`                      |
| `t2v_super`  | Super: Text to Video Without Audio  | `Cosmos3-Super` | `text2video`  | `assets/prompts/text2video/robot_kitchen.json`             | —                                                                  | —                                                     | —              | —                                      | —                                                                                                        | `t2v_super.mp4`                      |
| `t2av_super` | Super: Text to Video with Audio     | `Cosmos3-Super` | `text2video`  | `assets/prompts/text2video/robot_pouring_water_audio.json` | —                                                                  | —                                                     | `True`         | —                                      | —                                                                                                        | `t2av_super.mp4`                     |
| `i2v_super`  | Super: Image to Video Without Audio | `Cosmos3-Super` | `image2video` | `assets/prompts/image2video/humanoid_robot.json`           | `assets/images/image2video/humanoid_robot.jpg`                     | —                                                     | —              | —                                      | —                                                                                                        | `i2v_super.mp4`                      |
| `i2av_super` | Super: Image to Video with Audio    | `Cosmos3-Super` | `image2video` | `assets/prompts/image2video/coastal_road_audio.json`       | `assets/images/image2video/coastal_road_audio.jpg`                 | —                                                     | `True`         | —                                      | —                                                                                                        | `i2av_super.mp4`                     |
| `v2v_super`  | Super: Video to Video               | `Cosmos3-Super` | `video2video` | `assets/prompts/image2video/car_driving.json`              | `assets/videos/car_driving_plain.mp4` (not vendored in this skill) | `assets/negative_prompts/image2video/neg_prompt.json` | —              | —                                      | `{"use_system_prompt": True, "condition_video_latent_indexes": [0, 1], "condition_video_keep": "first"}` | `v2v_super.mp4`                      |
| `t2i_edge`   | Edge: Text to Image                 | `Cosmos3-Edge`  | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                                  | —                                                     | —              | `EDGE_T2I_SAMPLING`                    | `{"output_type": "image"}`                                                                               | `t2i_edge.png`                       |
| `t2v_edge`   | Edge: Text to Video                 | `Cosmos3-Edge`  | `text2video`  | `assets/prompts/text2video/robot_kitchen.json`             | —                                                                  | —                                                     | —              | `EDGE_VIDEO_SAMPLING`                  | —                                                                                                        | `t2v_edge.mp4`                       |
| `i2v_edge`   | Edge: Image to Video                | `Cosmos3-Edge`  | `image2video` | `assets/prompts/image2video/car_driving_edge.json`         | `assets/images/image2video/car_driving.jpg`                        | —                                                     | —              | `EDGE_VIDEO_SAMPLING`                  | —                                                                                                        | `i2v_edge.mp4`                       |

## Cosmos3-Nano Examples

Use cases for the `Cosmos3-Nano` model. This section is self-contained; you can run it without the Cosmos3-Super section below.

### Nano: Text to Image

Nano text-to-image generation using a structured JSON prompt. TensorRT-LLM returns a base64-encoded PNG from `/v1/images/generations`.

### Nano: Text to Video Without Audio

Nano text-to-video generation using a structured JSON prompt, with audio disabled.

### Nano: Text to Video with Audio

Nano text-to-video generation with synchronized audio. The payload enables `enable_audio` in TensorRT-LLM's Cosmos3 `extra_params`; `ffmpeg` muxes the generated audio into the MP4 response.

### Nano: Image to Video Without Audio

Nano image-to-video generation using its paired image asset, with audio disabled.

### Nano: Image to Video with Audio

Nano image-to-video generation using its paired coastal-road image and synchronized audio prompt. The reference image is uploaded as multipart `image_reference`, and `enable_audio` is set in `extra_params`.

### Nano: Video to Video

Nano video-to-video generation using a checked-in MP4 `video_reference`. TensorRT-LLM forwards the encoded video to each worker for NVDEC decoding. `condition_video_latent_indexes=[0, 1]` pins the first two output latent frames, which consumes the first five pixel frames of the reference.

## Cosmos3-Super Examples

The same use cases for the larger `Cosmos3-Super` model. This section is self-contained; you can run it without the Cosmos3-Nano section above.

### Super: Text to Image

Super text-to-image generation using the same structured JSON prompt. TensorRT-LLM returns a base64-encoded PNG from `/v1/images/generations`.

### Super: Text to Video Without Audio

Super text-to-video generation using the same structured JSON prompt, with audio disabled.

### Super: Text to Video with Audio

Super text-to-video generation with synchronized audio using the same `enable_audio` request contract as Nano.

### Super: Image to Video Without Audio

Super image-to-video generation using its paired image asset, with audio disabled.

### Super: Image to Video with Audio

Super image-to-video generation using the paired coastal-road image and synchronized audio prompt.

### Super: Video to Video

Super video-to-video generation using the same reference-video request shape and conditioning controls as Nano, against the `Cosmos3-Super` endpoint.

## Cosmos3-Edge Examples

Use cases for the compact single-GPU `Cosmos3-Edge` checkpoint. This section is self-contained; you can run it without the Cosmos3-Nano and Cosmos3-Super sections above.

TensorRT-LLM serves Edge for text-to-image, text-to-video, and image-to-video. There is no audio variant (Edge has no audio tower), no action mode (the checkpoint's action weights are not served by this pipeline), and no video-to-video case (validated for Nano and Super only).

Edge runs a different shape from Nano and Super: video requests use 832x480 with 121 frames, 50 UniPC steps and guidance 5.0, and text-to-image uses Edge's native 640x640 with guidance 4.0. These come from `EDGE_VIDEO_SAMPLING` and `EDGE_T2I_SAMPLING` in the helper cell above.

### Edge: Text to Image

Edge text-to-image generation using the same structured JSON prompt as the Nano and Super sections, at Edge's native 640x640.

This case runs **native image mode**: it posts to `/v1/images/generations` with `output_type="image"` in `extra_params` and decodes the base64 PNG from the response. That flag is what selects the image path — without it the server runs video mode, and video mode defaults the negative prompt to Cosmos3's video negative prompt, whose motion and frame-to-frame artifact terms have no meaning for a still. The images API carries no frame or frame-rate fields, so the request sends none.

### Edge: Text to Video

Edge text-to-video generation using the same structured JSON prompt, at 832x480 with 121 frames. Audio is never enabled for Edge.

### Edge: Image to Video

Edge image-to-video generation from the coastal-road dashcam frame. This case uses the Edge variant of the prompt asset, whose timeline describes the 5-second, 121-frame Edge clip rather than the longer Nano/Super clip. The reference image is uploaded as multipart `input_reference`.

## Quickstart (cookbook README)

### Quickstart

Set up the environment and start the server:
[TensorRT-LLM setup](../../cookbooks/cosmos3/README.md#tensorrt-llm-generator). The notebook targets the
OpenAI-compatible VisualGen API served by `trtllm-serve`.

Send a text-to-video request with the synchronous video API:

```python
import json
from pathlib import Path

import requests

prompt = json.load(open("assets/prompts/text2video/robot_kitchen.json"))
negative = json.load(open("assets/negative_prompts/text2video/neg_prompt.json"))

response = requests.post(
    "http://localhost:8000/v1/videos/sync",
    json={
        "prompt": json.dumps(prompt, ensure_ascii=True, separators=(",", ":")),
        "negative_prompt": json.dumps(negative, ensure_ascii=True, separators=(",", ":")),
        "size": "1280x720",
        "seconds": 189 / 24,
        "fps": 24,
        "num_frames": 189,
        "num_inference_steps": 35,
        "guidance_scale": 6.0,
        "max_sequence_length": 4096,
        "seed": 0,
        "format": "mp4",
        "response_format": "file",
        "extra_params": {
            "use_resolution_template": False,
            "use_duration_template": False,
            "use_system_prompt": False,
            "use_guardrails": True,
        },
    },
    headers={"Accept": "video/mp4"},
)
response.raise_for_status()
if (
    "video/mp4" not in response.headers.get("content-type", "")
    or response.content[4:8] != b"ftyp"
):
    raise RuntimeError("TensorRT-LLM did not return browser-compatible MP4")
Path("/tmp/cosmos3_t2v_trtllm.mp4").write_bytes(response.content)
```

For image-to-video, post multipart form data to the same endpoint with the
reference image under `image_reference`. To generate synchronized audio for a
text-to-video or image-to-video request, add `"enable_audio": True` to
`extra_params`. Keep `ffmpeg` installed in the server environment so TensorRT-LLM
can mux the generated audio into MP4. Explicit `format=mp4` makes a missing
encoder fail early instead of returning browser-incompatible AVI.

For video-to-video, upload an MP4 reference under `video_reference`.
TensorRT-LLM forwards the encoded bytes to
the Cosmos3 workers, which decode the conditioning window with NVDEC:

```python
import json
from pathlib import Path

import requests

source_video = Path("assets/videos/car_driving_plain.mp4").resolve()  # not vendored in this skill
v2v_prompt = json.load(open("assets/prompts/image2video/car_driving.json"))
v2v_negative = json.load(open("assets/negative_prompts/image2video/neg_prompt.json"))

with source_video.open("rb") as video_file:
    response = requests.post(
        "http://localhost:8000/v1/videos/sync",
        data={
            "prompt": json.dumps(v2v_prompt, ensure_ascii=True, separators=(",", ":")),
            "negative_prompt": json.dumps(v2v_negative, ensure_ascii=True, separators=(",", ":")),
            "size": "1280x720",
            "num_frames": "189",
            "fps": "24",
            "num_inference_steps": "35",
            "guidance_scale": "6.0",
            "max_sequence_length": "4096",
            "seed": "0",
            "format": "mp4",
            "response_format": "file",
            "extra_params": json.dumps(
                {
                    "use_resolution_template": False,
                    "use_duration_template": False,
                    "use_system_prompt": True,
                    "use_guardrails": True,
                    "condition_video_latent_indexes": [0, 1],
                    "condition_video_keep": "first",
                },
                separators=(",", ":"),
            ),
        },
        files={"video_reference": (source_video.name, video_file, "video/mp4")},
        headers={"Accept": "video/mp4"},
    )
response.raise_for_status()
if (
    "video/mp4" not in response.headers.get("content-type", "")
    or response.content[4:8] != b"ftyp"
):
    raise RuntimeError("TensorRT-LLM did not return browser-compatible MP4")
Path("/tmp/cosmos3_v2v_trtllm.mp4").write_bytes(response.content)
```

`condition_video_latent_indexes` identifies clean latent frames in the output;
with `[0, 1]`, TensorRT-LLM consumes the first five pixel frames from the input.
Set `condition_video_keep` to `"last"` to condition on the corresponding tail
window instead.

For text-to-image, send JSON to `/v1/images/generations` with `format=png`,
`response_format=b64_json`, and `extra_params.output_type="image"`, then
base64-decode `data[0].b64_json`. This produces a PNG image instead of wrapping
one frame in a video container.

To run **Cosmos3-Edge** instead, serve `nvidia/Cosmos3-Edge` on a single GPU with
no config override (`trtllm-serve nvidia/Cosmos3-Edge --port 8000`) and send
Edge's 480p-native shape: `"size": "832x480"`, `"num_frames": 121`,
`"num_inference_steps": 50`, and `"guidance_scale": 5.0`. Text-to-image is a
native image request: post to `/v1/images/generations` with
`"output_type": "image"` in `extra_params`, Edge's native `"size": "640x640"`,
and `"guidance_scale": 4.0`. That flag selects the image path; without it the
server runs video mode and defaults the negative prompt to Cosmos3's video
negative prompt, whose motion and frame-to-frame artifact terms do not apply to
a still. The images API carries no frame or frame-rate fields. Flow shift (3.0)
rides the checkpoint-declared native flow schedule, so requests do not send it.
TensorRT-LLM serves Edge for text-to-image, text-to-video, and image-to-video
only: Edge has no audio tower, so `enable_audio` is unavailable, its action
weights are not served by this pipeline, and video-to-video is validated for Nano
and Super. Edge support landed in TensorRT-LLM PR
[#16773](https://github.com/NVIDIA/TensorRT-LLM/pull/16773).

The TRT-LLM notebook always sends model-specific `extra_params`, so use a
TensorRT-LLM release with the Cosmos3 VisualGen API schema. The notebook sets
request-level `max_sequence_length=4096` for longer structured JSON prompts.

Skill note (not in the README): see the skill note under § 2 Cosmos3-Edge (after the Edge launch block) for the concrete release (≥ 1.3.0rc27).

### Notebook walkthrough

`run_with_trt_llm.ipynb` is the full tutorial for the
TensorRT-LLM backend: it walks through text-to-image, text-to-video and
image-to-video with or without synchronized audio, and video-to-video requests
against an already-running VisualGen server. It also includes a dedicated
Cosmos3-Edge section with 480p text-to-image, text-to-video, and image-to-video
examples. Server launch options (Nano, Super, and Edge, FP8 dynamic
quantization, CFG parallelism, Ulysses, and parallel VAE) live in the
[shared environment setup guide](../../cookbooks/cosmos3/README.md#tensorrt-llm-generator).

## 7. Skill notes (not in the notebook) — install routes, serve commands and image-route facts (from SKILL.md Variant 5)

Skill-authored text moved here from SKILL.md on 2026-10-27 (layout change only; wording unchanged apart from pointers and lead-ins that had to follow the text). Nothing in this section is notebook text; SKILL.md keeps a one-line pointer at each former location.

### 7.1 Hard prerequisite: a VisualGen-capable TensorRT-LLM (SKILL.md Variant 5)

Setup guide: "Use a TensorRT-LLM build that includes the Cosmos3 VisualGen backend
and the modality required by your notebook. Install TensorRT-LLM following its
upstream documentation." Hard prerequisite (eval-verified): Cosmos3 VisualGen is NOT in
stable PyPI tensorrt-llm 1.2.x (`trtllm-serve` has no `--visual_gen_args`; Edge fails in
the generic loader with `KeyError` / `ValueError` `cosmos3_edge`). Use a pre-release wheel
that contains commit bca6761 — first: 1.3.0rc27 — or the devel-container source build. Do
not reuse the Reasoner section's `release:1.3.0rc22`.

### 7.2 Skill launch script (not in the notebook) — Route A: prebuilt wheel on a bare GPU host (SKILL.md Variant 5)

**Route A (default: bare GPU host, no Docker)** — the prebuilt wheel (verified by the eval
runs on 1x H100 80 GB with a CUDA 13 driver and no CUDA toolkit; the upstream TensorRT-LLM
pip-install guide, <https://nvidia.github.io/TensorRT-LLM/installation/linux.html>, lists
the same OpenMPI and cu130 torch-index steps):

```bash
apt-get update && apt-get install -y git libopenmpi-dev openmpi-bin ffmpeg   # mpi4py needs libmpi.so; ffmpeg for MP4
export TRTLLM_VENV="${TRTLLM_VENV:-$PWD/.venv-trtllm}"
# --seed: the venv has pip (a bare host's python may have none); a uv-managed interpreter ships libpython
#   (system interpreter instead: apt-get install -y libpython3.X)
# --no-config: the uv pin — stock uv 0.13+, not Variant 2's 0.11.29 (see SKILL.md, Read this first)
uv venv "$TRTLLM_VENV" --python 3.12 --seed --managed-python --allow-existing --no-config
source "$TRTLLM_VENV/bin/activate"
# PyPI has only sdists of tensorrt_llm; the wheels live on pypi.nvidia.com, so both index flags are
# required (uv's default first-index strategy would stop at the PyPI sdist). rc27 is the first tag
# containing the pinned commit bca6761 and its binaries are built against torch 2.13 (rc27 + torch
# 2.12 -> undefined symbol in libth_common.so). Keep rc27 even though the checkout reports
# __version__ 1.3.0rc26: the rc26 wheel passes /health and generates, then /v1/images/generations
# (and /v1/images/edits) return HTTP 500 — NameError get_steady_clock_now_in_seconds,
# openai_server.py:3164 — fixed in bca6761, first shipped in rc27; video routes (/v1/videos/*) are
# unaffected, so Variant 6 on rc26 is fine (video route; its image-route fallback needs rc27). Do NOT float to newer rcs: rc28 needs torch>=2.13,
# rc29/1.4.0rc* need torch 2.14; 1.4.0rc1 served Nano but is unverified for Edge/distilled.
uv pip install --no-config --index-strategy unsafe-best-match \
  --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu130 \
  "tensorrt_llm==1.3.0rc27" "torch==2.13.0+cu130" "torchvision==0.28.0+cu130"
# torch/nvidia-* from the cu130 index provide the wheel's CUDA-13 libraries (libcublasLt.so.13 ...)
# under site-packages/nvidia/cu13; the system's CUDA 11/12 libraries do not satisfy them — never
# symlink .so.12 to .so.13. CC=gcc may be needed if a dependency has no cp312 wheel; clang and a
# system CUDA toolkit are NOT needed (clang only matters for `pip install -e .`).
export CUDA_HOME="$(python -c 'import nvidia,os;print(os.path.join(nvidia.__path__[0],"cu13"))')"
# DeepGEMM asserts _find_cuda_home() at import ('Decoding operators failed to load' / serve exits
# before /health); no nvcc is used for bf16 serving. If FP8 block-scale kernels are ever JIT-compiled,
# add nvidia-cuda-nvcc-cu13 to the same tree.
# Preflight before any checkpoint download — two checks, so an import failure is not misreported as a
# VisualGen-less build (run the block as a script: `exit 1` would close an interactive shell)
python -c "import tensorrt_llm" || { echo "tensorrt_llm import failed — OpenMPI / CUDA_HOME prerequisites above"; exit 1; }
trtllm-serve serve --help | grep -q visual_gen_args || { echo "this tensorrt_llm has no Cosmos3 VisualGen"; exit 1; }   # the group-level --help does not list serve options
# Pinned checkout, YAML configs only (Edge needs none). Its version.py reports 1.3.0rc26 — that is NOT the
# wheel to install (see the comment above); fetch the SHA directly.
# Optional minimal form: --filter=blob:none on the fetch, then `git sparse-checkout set examples/visual_gen/configs` before the checkout.
git init TensorRT-LLM && cd TensorRT-LLM && git remote add origin https://github.com/NVIDIA/TensorRT-LLM.git
git fetch --depth 1 origin bca6761ab84fbcd58fc7f914eade7de48b32e35e
GIT_LFS_SKIP_SMUDGE=1 git checkout FETCH_HEAD
export TRTLLM_ROOT="$PWD"
```

### 7.3 Pinned checkout: fetch the SHA directly; configs only (SKILL.md Variant 5)

Do not `git clone --depth 1` the repo and then `git checkout <pin>`: a shallow clone of main
does not contain the pinned commit and the checkout fails as if the commit did not exist —
fetch the SHA directly as above (the commit exists: 2026-09-09, 1.3.0rc26). The checkout is
only the source of `examples/visual_gen/configs/*.yaml` for Nano, Super and the 4-Step
students; never put it on `PYTHONPATH` or `pip install -e` it over an installed wheel. Edge
(`trtllm-serve nvidia/Cosmos3-Edge`, no config override) needs no checkout on the wheel path.

### 7.4 Skill launch script (not in the notebook) — Route B: source build in the devel container (SKILL.md Variant 5)

**Route B (source build in the devel container)** — the setup guide's path, "the right path
when you need a checkout that contains a recent Cosmos3 VisualGen change before it is
available in your installed package or release image":

```bash
apt-get update && apt-get -y install git git-lfs && git lfs install
git clone https://github.com/NVIDIA/TensorRT-LLM.git && cd TensorRT-LLM
git checkout bca6761ab84fbcd58fc7f914eade7de48b32e35e && git submodule update --init --recursive && git lfs pull
# Pick a devel tag from the upstream build-from-source guide or NGC, then inside
#   docker run --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all \
#     --volume "$PWD":"$PWD" --workdir "$PWD" nvcr.io/nvidia/tensorrt-llm/devel:<tag>
apt-get update && apt-get -y install ffmpeg
python3 scripts/build_wheel.py --use_ccache --skip_building_wheel --linking_install_binary
pip install -e .
```

### 7.5 Why not the `TRTLLM_USE_PRECOMPILED=1` shortcut (SKILL.md Variant 5)

Do not use the setup guide's Python-only shortcut `TRTLLM_USE_PRECOMPILED=1 pip install -e .`
on this checkout. For an editable install setup.py resolves `tensorrt_llm==1.3.0rc26+bca6761ab8`,
which is not published (pypi.nvidia.com carries only tagged builds — 1.3.0rc26 and 1.3.0rc27
around the pin — never a `+<sha>` local version); the value of
`TRTLLM_USE_PRECOMPILED` is a version string, so a local wheel path or URL has to go in
`TRTLLM_PRECOMPILED_LOCATION`; and the published 1.3.0rc26 binaries predate bca6761 by 126
commits, so the pinned Python fails at import with
`operator trtllm::silu_and_mul_fp8_quantize_1x128_packed_ue8m0 does not exist`. The download
also needs `pip` in the build environment (absent in a uv venv) and the editable install
compiles `llist` (needs a C compiler). Either run the full source build above inside the
devel container, or install one complete published wheel — `tensorrt_llm==1.3.0rc27`, the
first tag that contains bca6761 (Route A) — and keep the checkout only for
`examples/visual_gen/configs`; never mix the checkout's Python or its dependency set with
release binaries.

### 7.6 Only if you keep guardrails on (SKILL.md Variant 5)

**Only if you keep guardrails on** (not the smoke path: the serve blocks (§ 7.7 below)
`export TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1` and the request bodies send `use_guardrails: false`).
Setup guide, TensorRT-LLM Generator: `pip install cosmos_guardrail==0.3.0`. It requires
`transformers>=5`, so it resolves only on a VisualGen-capable TensorRT-LLM (1.3.0rc26+ pins
`transformers==5.5.4`); on the stable 1.2.1 wheel (`transformers==4.57.3`) pip refuses it. On
headless servers follow with the headless OpenCV swap of Variant 1
(`pip uninstall -y opencv-python && pip install opencv-python-headless==5.0.0.93`), then do
the guide's "Server-side NLTK data setup"
(<../cookbooks/cosmos3/README.md#tensorrt-llm-generator>).

### 7.7 Skill launch script (not in the notebook) — serve commands: Edge, Nano, Super (SKILL.md Variant 5)

Run the serve commands with `TRTLLM_ROOT` pointing at the pinned checkout (Nano/Super/4-Step
read their YAML from it; Edge needs none) and TensorRT-LLM, OpenMPI and `ffmpeg` installed
and `CUDA_HOME` set (install block: § 7.2 above). Notebook: "to disable [guardrails], set `TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1`
before starting the server or set `use_guardrails` to `False` in the request `extra_params`":

```bash
export TRTLLM_ROOT="${TRTLLM_ROOT:-$PWD}"
test -d "$TRTLLM_ROOT/examples/visual_gen/configs" \
  || echo "TRTLLM_ROOT=$TRTLLM_ROOT does not look like a TensorRT-LLM checkout"
export TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1
trtllm-serve nvidia/Cosmos3-Edge --port 8000              # Edge: single GPU, no config override
trtllm-serve nvidia/Cosmos3-Nano \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-nano-1gpu.yaml" --port 8000
# Super: torchrun --nproc_per_node=4 -m tensorrt_llm.commands.serve nvidia/Cosmos3-Super \
#   --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-super-4gpu.yaml" --port 8000
```

### 7.8 Explicit serve form, Nano YAML memory, `compilation_config`, headless pointer (SKILL.md Variant 5)

`trtllm-serve serve <model> …` is the explicit equivalent of the implicit form of § 7.7
(the form one eval run used on 1.4.0rc1; the implicit form is verified on 1.3.0rc26 and
1.3.0rc27). The shipped Nano YAML fits one 80 GB GPU
(~50–52 GB peak after the default 720x1280x189 warm-up; ready in ~2–3 min — measured by
the eval runs on one H100 80 GB, not a notebook figure). Memory for this pipeline is
controlled by the `--visual_gen_args` YAML, not by trtllm-serve's LLM flags
(`--max_batch_size`, KV-cache fraction). For t2i-only serving on smaller GPUs copy
`cosmos3-nano-1gpu.yaml`, keep `attention_config` VANILLA and `cfg_size=ulysses_size=1`,
and restrict the warm-up with `compilation_config: {resolutions: [[192, 320]], num_frames: [1]}`
(one eval run: 37,705 MiB sampled peak vs 52,147 MiB, same 1280x720 request; not a
controlled benchmark); a `[[720, 1280]]` x 1 warm-up is the natural variant but was not
exercised. Headless launch / teardown: see the pattern in skill-notes.md § 4 (the notebook's `check_trtllm_server`, payloads.md § 9.3, is the poll).

### 7.9 Image-route facts: rc26 diagnosis, `output_type`, sampling per model (SKILL.md Variant 5)

A 200 on `/health` does not exercise the image route: if the first
`/v1/images/generations` call answers HTTP 500 with "name 'get_steady_clock_now_in_seconds'
is not defined", the server is tensorrt_llm 1.3.0rc26 — reinstall 1.3.0rc27 with torch
2.13.0+cu130; do not debug the payload.

Text-to-image is a native image request (README, "Edge: Text to Image"): it posts to
`/v1/images/generations` with `output_type="image"` in `extra_params` — "that flag is
what selects the image path; without it the server runs video mode, and video mode
defaults the negative prompt to Cosmos3's video negative prompt" — with `format: "png"`,
`response_format: "b64_json"` and no frame or frame-rate fields; decode `data[0].b64_json`.
Edge t2i runs at its native 640x640, guidance 4.0, 50 steps (`EDGE_T2I_SAMPLING`);
Nano/Super use `FIXED_SAMPLING` (1280x720, 35 steps, guidance 6.0); `max_sequence_length`
4096.
