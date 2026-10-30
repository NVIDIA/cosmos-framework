# Cosmos3 Generator Audiovisual with SGLang — distilled from the cookbook notebook

Scope: everything SGLang-specific in the notebook — prerequisites, server launch (Nano, Super, CFG Parallel, Edge, Distil), endpoint verification, the SGLang request helpers (`video_api_url`, `image_api_url`, `build_sglang_video_form`, `build_sglang_image_body`, `post_video`, `post_image`, `run_sglang_payload`) and all 14 use cases in one table; the material the notebook shares with the other backend notebooks (licence header, asset catalogue, path configuration, `FIXED_SAMPLING`, `ASSET_SETS`, `payload_dimensions`, `create_payload`, endpoints, output naming) lives in payloads.md and is referenced here, not repeated.

Licence header: see payloads.md § 2.1.

This notebook calls already-running SGLang Cosmos3 servers with direct `curl` requests from Python.

The examples are split into Cosmos3-Nano and Cosmos3-Super sections. Each section is self-contained, so you can run just one. Each section targets the matching model endpoint.

## 1. Prerequisites

Use a running SGLang server and set endpoint environment variables before the setup cell if you are not using the local default. Text-to-image uses `/v1/images/generations`; video modes use `/v1/videos`.

The notebook's export block (`COSMOS3_SGLANG_BASE_URL`, `COSMOS3_SGLANG_NANO_BASE_URL`, `COSMOS3_SGLANG_SUPER_BASE_URL`, each `http://localhost:30000`): see payloads.md § 4.5, "The Prerequisites cells document the local defaults as exports", `run_with_sglang.ipynb` block.

## 2. Start the Server

Run the SGLang server before running the request cells. Use the Docker image for every modality on this page. Mount any directory that contains local media or action files you want the server to read.

Skill note (not in the notebook): readiness — poll `GET http://localhost:30000/health` until it returns HTTP 200 `{"status":"ok"}`. It returns 503 after `Application startup complete.` while the warm-up runs and turns 200 after `The server is fired up and ready to roll!` (observed 60–171 s for Edge on 1x H100 across four eval runs, 2026-10-21/23; the long end includes the in-container weight download). Headless automation: `docker run -d --name cosmos3-sglang …` with the rest of the block unchanged, poll `/health`, then `docker stop cosmos3-sglang && docker rm cosmos3-sglang` (pattern in skill-notes.md § 4).

Skill note (not in the notebook): the `-e SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` and `-e HF_TOKEN` lines in the four `docker run` blocks below, and the `export SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` before the bare `sglang serve` CFG-Parallel command, are skill additions verified against `lmsysorg/sglang:dev` by the eval runs — the notebook passes no `-e` at all; every other line is the notebook's command verbatim. The variable is read by the image's Cosmos3 pipeline at startup (`cosmos3_pipeline.py`); the HF-cache note below says why both lines are needed.

### Set up HF Cache

SGLang container is run under root by default. User can create a writable cache directory on host, or start the container with host user id.

```bash
export SGLANG_HF_CACHE="${SGLANG_HF_CACHE:-$HOME/.cache/sglang-huggingface}"
mkdir -p "$SGLANG_HF_CACHE"
chmod 777 "$SGLANG_HF_CACHE"
```

Skill note (not in the notebook): pass `-e HF_TOKEN` to every `docker run` below — the first launch downloads `nvidia/Cosmos3-<model>` into `SGLANG_HF_CACHE`, and anonymous Hub calls are rate-limited (HTTP 429 → the server exits with an incomplete-checkpoint ValueError). With a token authorised for `nvidia/Cosmos-1.0-Guardrail` the server can keep guardrails enabled; otherwise keep the `-e SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` line the blocks below already carry (drop it only with an authorised token) — the gated guardrail is loaded at startup, and without the switch the server exits with GatedRepoError 401.

### Docker Image: Cosmos3-Nano

```bash
docker run --runtime nvidia --gpus all \
  -e SGLANG_DISABLE_COSMOS3_GUARDRAILS=1 \
  -e HF_TOKEN \
  -v $SGLANG_HF_CACHE:/root/.cache/huggingface \
  -v "$(pwd):/workspace" \
  -p 30000:30000 \
  --ipc=host \
  lmsysorg/sglang:dev \
  sglang serve \
  --model-path nvidia/Cosmos3-Nano \
  --host 0.0.0.0
```

### Docker Image: Cosmos3-Super

`Cosmos3-Super` is the larger 64B model, so it usually needs more GPU memory than `Cosmos3-Nano`. `--tp-size` splits model weights across multiple GPUs and reduces per-GPU memory use. Use `--performance-mode memory` preset to preserve memory.

For example, on four GPUs:

```bash
docker run --runtime nvidia --gpus all \
  -e SGLANG_DISABLE_COSMOS3_GUARDRAILS=1 \
  -e HF_TOKEN \
  -v $SGLANG_HF_CACHE:/root/.cache/huggingface \
  -v "$(pwd):/workspace" \
  -p 30000:30000 \
  --ipc=host \
  lmsysorg/sglang:dev \
  sglang serve \
  --model-path nvidia/Cosmos3-Super \
  --host 0.0.0.0 \
  --num-gpus 4 \
  --performance-mode memory
```

### CFG Parallel

Use `--cfg-parallel-size 2` to run the positive and negative CFG branches in parallel on two GPUs:

```bash
export SGLANG_DISABLE_COSMOS3_GUARDRAILS=1
sglang serve \
  --model-path nvidia/Cosmos3-Nano \
  --host 0.0.0.0 \
  --num-gpus 2 \
  --enable-cfg-parallel \
  --cfg-parallel-size 2
```

For Cosmos3, set CFG strength with the request-level `guidance_scale` field. Do not use `true_cfg_scale` for CFG Parallel with these Cosmos3 examples.

### Docker Image: Cosmos3-Edge

```bash
docker run --runtime nvidia --gpus all \
  -e SGLANG_DISABLE_COSMOS3_GUARDRAILS=1 \
  -e HF_TOKEN \
  -v $SGLANG_HF_CACHE:/root/.cache/huggingface \
  -v "$(pwd):/workspace" \
  -p 30000:30000 \
  --ipc=host \
  lmsysorg/sglang:dev \
  sglang serve \
  --model-path nvidia/Cosmos3-Edge \
  --host 0.0.0.0
```

### Docker Image: Cosmos3-Distil

```bash
docker run --runtime nvidia --gpus all \
  -e SGLANG_DISABLE_COSMOS3_GUARDRAILS=1 \
  -e HF_TOKEN \
  -v $SGLANG_HF_CACHE:/root/.cache/huggingface \
  -v "$(pwd):/workspace" \
  -p 30000:30000 \
  --ipc=host \
  lmsysorg/sglang:dev \
  sglang serve \
  --model-path nvidia/Cosmos3-Super-Image2Video-4Step \
  --host 0.0.0.0
```

## 3. Configure Paths and Endpoints

This setup cell only configures repo/output paths and SGLang endpoint settings.

Paths (said once for this file): run the skill-relative configuration block of payloads.md § 4.1 in place of the notebook's `find_repo_root()` / `COSMOS_ROOT` / `COSMOS3_AUDIOVISUAL_ROOT` / `COSMOS3_AUDIOVISUAL_OUTPUT_ROOT` setup. The skill directory holds `assets/`, and a plain output directory `OUT=outputs` replaces the notebook's output-root plumbing, so every payload lands in `<OUT>/sglang/payloads/<use_case>/<use_case>.json` and every media file in `<OUT>/sglang/<use_case>/` (payloads.md § 4.2).

Endpoints: `DEFAULT_SGLANG_BASE_URL` (`COSMOS3_SGLANG_BASE_URL`, default `http://localhost:30000`), the four-entry `SGLANG_ENDPOINTS` dictionary (`Cosmos3-Nano`, `Cosmos3-Super`, `Cosmos3-Edge`, `Cosmos3-Distil`, each read from its `COSMOS3_SGLANG_<MODEL>_BASE_URL` variable with `DEFAULT_SGLANG_BASE_URL` as fallback) and `os.environ.setdefault("COSMOS3_SGLANG_API_KEY", "")`: see payloads.md § 4.5, `run_with_sglang.ipynb` block (identical to the notebook's cell).

## 4. Verify Endpoint Configuration

```python
from urllib.parse import urlparse

for model, base_url in SGLANG_ENDPOINTS.items():
    api_root = base_url.rstrip("/")
    if not api_root.endswith("/v1"):
        api_root = f"{api_root}/v1"
    parsed = urlparse(api_root)
    print(model)
    print("  api root:", api_root)
    print("  images generations:", f"{api_root}/images/generations")
    print("  videos async:", f"{api_root}/videos")
    print("  scheme:", parsed.scheme)
    print("  host:", parsed.netloc)
```

## 5. Preview Available Inputs

The preview cell (prompt captions and inline reference images) is Jupyter display plumbing and is not migrated; its caption rule and `IMAGE_EXTENSIONS` are in payloads.md § 3.5, and the assets it lists are catalogued in payloads.md §§ 3.1–3.3.

## 6. Define Asset Sets, Payload Helpers, Request Helpers, and Viewer Helpers

### 6.1 Shared definitions (in payloads.md)

| Definition in the notebook's cell                                                                                                                                                                                       | See                                                                                            |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `import json`, `import os`, `from pathlib import Path`; `IMAGE_EXTENSIONS`                                                                                                                                              | payloads.md § 9.1                                                                              |
| `FIXED_SAMPLING` (`num_steps` 35, `guidance` 6.0, `shift` 10.0, `fps` 24, `num_frames` 189, `resolution` `"720"`, `aspect_ratio` `"16,9"`, `seed` 0)                                                                    | payloads.md § 6.1 (framework / vllm_omni / sglang table; identical)                            |
| `ASSET_SETS` — 14 use cases, with the comment "All asset paths are repo-relative under cookbooks/cosmos3/generator/audiovisual. Model and sound choices live in this manifest; folders are organized only by modality." | payloads.md § 8.7 (`run_with_sglang.ipynb` block, verbatim) and §§ 8.1–8.4 (matrix)            |
| `_is_distil`, `_is_edge`, `create_payload(use_case, *, backend)`                                                                                                                                                        | payloads.md § 9.2 (`run_with_sglang.ipynb` version; de-Jupyterised there, otherwise identical) |
| `asset_path`, `compact_json_file`, `resolve_payload_path`                                                                                                                                                               | payloads.md § 9.1                                                                              |
| `payload_dimensions` (`"720"`/`"16,9"` → 720 × 1280; `"480"`/`"16,9"` → 480 × 832; `"640"`/`"1,1"` → 640 × 640; `"256"`/`"1,1"` → 256 × 256)                                                                            | payloads.md § 7 (`run_with_sglang.ipynb` version)                                              |
| `api_root_url`                                                                                                                                                                                                          | payloads.md § 4.5                                                                              |

### 6.2 SGLang request helpers

`api_root_url`: see payloads.md § 4.5.

```python
import base64
import html
import json
import os
import subprocess
import time
from pathlib import Path


def video_api_url(base_url: str) -> str:
    return f"{api_root_url(base_url)}/videos"


def image_api_url(base_url: str) -> str:
    return f"{api_root_url(base_url)}/images/generations"

def build_sglang_video_form(payload: dict) -> dict[str, str]:
    height, width = payload_dimensions(payload)
    extra_params = {
        "use_resolution_template": False,
        "use_duration_template": False,
        "guardrails": True,
    }
    form = {
        "prompt": payload["prompt"],
        "negative_prompt": payload["negative_prompt"],
        "size": f"{width}x{height}",
        "num_frames": str(payload["num_frames"]),
        "fps": str(payload["fps"]),
        "num_inference_steps": str(payload["num_steps"]) if "num_steps" in payload else None,
        "guidance_scale": str(payload["guidance"]) if "guidance" in payload else None,
        "flow_shift": str(payload["shift"]) if "shift" in payload else None,
        "seed": str(payload["seed"]),
        "max_sequence_length": "4096",
        "extra_params": json.dumps(extra_params, separators=(",", ":")),
    }
    if payload["enable_sound"]:
        form["generate_sound"] = "true"
        form["sound_duration"] = f"{payload['num_frames'] / payload['fps']:.3f}"
    return form


def build_sglang_image_body(payload: dict) -> dict:
    height, width = payload_dimensions(payload)
    return {
        "prompt": payload["prompt"],
        "negative_prompt": payload.get("negative_prompt", ""),
        "size": f"{width}x{height}",
        "n": 1,
        "num_inference_steps": payload["num_steps"] if "num_steps" in payload else None,
        "guidance_scale": payload["guidance"] if "guidance" in payload else None,
        "flow_shift": payload["shift"] if "shift" in payload else None,
        "seed": payload["seed"],
        "max_sequence_length": 4096,
        "response_format": "b64_json",
        "extra_params": {
            "use_resolution_template": False,
            "guardrails": True,
        },
    }


def post_video(*, payload_path: Path, payload: dict, output_path: Path, model: str) -> None:
    url = video_api_url(SGLANG_ENDPOINTS[model])
    api_key = os.environ.get("COSMOS3_SGLANG_API_KEY") or None
    tmp_path = Path(f"{output_path}.tmp")
    error_path = Path(f"{output_path}.error.txt")
    if tmp_path.exists():
        tmp_path.unlink()
    if error_path.exists():
        error_path.unlink()

    cmd = [
        "curl",
        "-sS",
        "--fail-with-body",
        "-X",
        "POST",
        url,
        "-H",
        "Accept: video/mp4",
    ]
    if api_key is not None:
        cmd += ["-H", f"Authorization: Bearer {api_key}"]

    for key, value in build_sglang_video_form(payload).items():
        if value is None:
            continue
        cmd += ["--form-string", f"{key}={value}"]

    if payload["model_mode"] == "image2video":
        image_path = resolve_payload_path(payload_path, payload["vision_path"])
        cmd += ["-F", f"input_reference=@{image_path}"]

    if payload["model_mode"] == "video2video":
        video_path = resolve_payload_path(payload_path, payload["vision_path"])
        cmd += ["-F", f"video_reference=@{video_path};type=video/mp4"]

    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        error_path.write_text((result.stdout or "") + (result.stderr or ""))
        raise RuntimeError(f"SGLang request failed with exit code {result.returncode}; see {error_path}")

    # poll output video
    initial = json.loads(result.stdout)
    video_id = initial["id"]

    while True:
        poll = subprocess.run(
            ["curl", "-sS", "--fail-with-body", f"{url}/{video_id}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if poll.returncode != 0:
            error_path.write_text((poll.stdout or "") + (poll.stderr or ""))
            raise RuntimeError(f"SGLang video poll failed; see {error_path}")

        final = json.loads(poll.stdout)
        if final.get("status") == "completed":
            break
        if final.get("status") in {"failed", "cancelled"}:
            error_path.write_text(json.dumps(final, indent=2))
            raise RuntimeError(f"SGLang video failed; see {error_path}")
        time.sleep(5)

    download = subprocess.run(
        ["curl", "-sS", "--fail-with-body", f"{url}/{video_id}/content", "-o", str(tmp_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if download.returncode != 0:
        error_path.write_text(json.dumps(final, indent=2) + "\n" + (download.stderr or ""))
        raise RuntimeError(f"SGLang video download failed; see {error_path}")

    tmp_path.replace(output_path)


def post_image(*, payload: dict, output_path: Path, model: str) -> None:
    url = image_api_url(SGLANG_ENDPOINTS[model])
    api_key = os.environ.get("COSMOS3_SGLANG_API_KEY") or None
    tmp_path = Path(f"{output_path}.tmp")
    error_path = Path(f"{output_path}.error.txt")
    if tmp_path.exists():
        tmp_path.unlink()
    if error_path.exists():
        error_path.unlink()

    cmd = [
        "curl",
        "-sS",
        "--fail-with-body",
        "-X",
        "POST",
        url,
        "-H",
        "Content-Type: application/json",
    ]
    if api_key is not None:
        cmd += ["-H", f"Authorization: Bearer {api_key}"]
    cmd += ["-d", json.dumps(build_sglang_image_body(payload), separators=(",", ":"))]

    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        error_path.write_text((result.stdout or "") + (result.stderr or ""))
        raise RuntimeError(f"SGLang image request failed with exit code {result.returncode}; see {error_path}")
    try:
        response = json.loads(result.stdout)
        b64_json = response["data"][0]["b64_json"]
        tmp_path.write_bytes(base64.b64decode(b64_json))
    except Exception as exc:
        error_path.write_text((result.stdout or "") + (result.stderr or ""))
        raise RuntimeError(f"Could not decode SGLang image response; see {error_path}") from exc
    tmp_path.replace(output_path)


def run_sglang_payload(payload_path: Path, output_dir: str | Path, *, model: str) -> Path:
    payload_path = Path(payload_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(payload_path.read_text())
    output_ext = ".png" if payload["model_mode"] == "text2image" else ".mp4"
    output_path = output_dir / f"{payload['name']}{output_ext}"
    endpoint = image_api_url(SGLANG_ENDPOINTS[model]) if payload["model_mode"] == "text2image" else video_api_url(SGLANG_ENDPOINTS[model])
    print("endpoint:", endpoint)
    print("payload:", payload_path)
    print("output:", output_path)
    if payload["model_mode"] == "image2video":
        print("input image:", resolve_payload_path(payload_path, payload["vision_path"]))
    t0 = time.time()
    if payload["model_mode"] == "text2image":
        post_image(payload=payload, output_path=output_path, model=model)
    else:
        post_video(payload_path=payload_path, payload=payload, output_path=output_path, model=model)
    print(f"wrote {output_path} in {time.time() - t0:.1f}s")
    return output_path
```

Skill note (not in the notebook): `build_sglang_video_form` and `build_sglang_image_body` above send `"guardrails": True` in `extra_params` — the notebook's value, kept verbatim. When the server runs with `SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` (§ 2) send `False` instead; the request-level flag alone does not prevent the startup exit described in § 2, and SKILL.md Variant 4 shows the smoke-run body.

### 6.3 Viewer helpers (not migrated)

`display_video` and `view_run` are Jupyter display plumbing and are not migrated; the `view_run` discovery rule is in payloads.md § 4.2 ("`view_run` discovery").

### 6.4 Workflow

Running the examples: see payloads.md § 11 (`run_with_vllm_omni.ipynb`, `run_with_sglang.ipynb` entry).

## Cosmos3-Nano Examples

Use cases for the `Cosmos3-Nano` model. This section is self-contained; you can run it without the Cosmos3-Super section below.

## Cosmos3-Super Examples

The same use cases for the larger `Cosmos3-Super` model. This section is self-contained; you can run it without the Cosmos3-Nano section above.

## Cosmos3-Edge Examples

Use cases for the smaller `Cosmos3-Edge` model. This section is self-contained; you can run it without the other sections above.

## Cosmos3-Distil Examples

Use cases for the smaller `Cosmos3-Distil` model. This section is self-contained; you can run it without the other sections above.

## Use cases (the Create Payload / Run / View Results triplets of the four sections above, in notebook order)

| Notebook heading — description                                                                                             | Use case (`create_payload` key) | `model=` (`run_sglang_payload`) | Mode          | Prompt asset                                               | Image / video asset                                                | Negative prompt                                                                      | `enable_sound` | Payload sampling `num_steps` / `guidance` / `shift` / `fps` / `num_frames` / `resolution` / `aspect_ratio` / `seed` (`—` = absent) → `size` | Request                                                                                                                                 | Output (under `<OUT>/sglang/`)                                                                     |
| -------------------------------------------------------------------------------------------------------------------------- | ------------------------------- | ------------------------------- | ------------- | ---------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------ | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Nano: Text to Image — Nano text-to-image generation using a structured JSON prompt.                                        | `t2i`                           | `Cosmos3-Nano`                  | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                                  | `""`                                                                                 | `False`        | 35 / 6.0 / 10.0 / 24 / 189 / `"720"` / `"16,9"` / 0 → `1280x720`                                                                            | `post_image`                                                                                                                            | `payloads/t2i/t2i.json`, `t2i/t2i.png`                                                             |
| Nano: Text to Video Without Audio — Nano text-to-video generation with audio disabled.                                     | `t2v_nano_noaudio`              | `Cosmos3-Nano`                  | `text2video`  | `assets/prompts/text2video/robot_kitchen.json`             | —                                                                  | `assets/negative_prompts/text2video/neg_prompt.json`                                 | `False`        | 35 / 6.0 / 10.0 / 24 / 189 / `"720"` / `"16,9"` / 0 → `1280x720`                                                                            | `post_video`                                                                                                                            | `payloads/t2v_nano_noaudio/t2v_nano_noaudio.json`, `t2v_nano_noaudio/t2v_nano_noaudio.mp4`         |
| Nano: Text to Video with Audio — Nano text-to-video generation with generated audio.                                       | `t2vs`                          | `Cosmos3-Nano`                  | `text2video`  | `assets/prompts/text2video/robot_pouring_water_audio.json` | —                                                                  | `assets/negative_prompts/text2video/neg_prompt.json`                                 | `True`         | 35 / 6.0 / 10.0 / 24 / 189 / `"720"` / `"16,9"` / 0 → `1280x720`                                                                            | `post_video` + `generate_sound=true`, `sound_duration=7.875` (`189 / 24`, `.3f`)                                                        | `payloads/t2vs/t2vs.json`, `t2vs/t2vs.mp4`                                                         |
| Nano: Image to Video Without Audio — Nano image-to-video generation using its paired image asset, with audio disabled.     | `i2v_nano_noaudio`              | `Cosmos3-Nano`                  | `image2video` | `assets/prompts/image2video/car_driving.json`              | `assets/images/image2video/car_driving.jpg`                        | `assets/negative_prompts/image2video/neg_prompt.json`                                | `False`        | 35 / 6.0 / 10.0 / 24 / 189 / `"720"` / `"16,9"` / 0 → `1280x720`                                                                            | `post_video` + `-F input_reference=@assets/images/image2video/car_driving.jpg`                                                          | `payloads/i2v_nano_noaudio/i2v_nano_noaudio.json`, `i2v_nano_noaudio/i2v_nano_noaudio.mp4`         |
| Nano: Image to Video with Audio — Nano image-to-video generation using its paired image asset and generated audio.         | `i2vs`                          | `Cosmos3-Nano`                  | `image2video` | `assets/prompts/image2video/coastal_road_audio.json`       | `assets/images/image2video/coastal_road_audio.jpg`                 | `assets/negative_prompts/image2video/neg_prompt.json`                                | `True`         | 35 / 6.0 / 10.0 / 24 / 189 / `"720"` / `"16,9"` / 0 → `1280x720`                                                                            | `post_video` + `-F input_reference=@assets/images/image2video/coastal_road_audio.jpg` + `generate_sound=true`, `sound_duration=7.875`   | `payloads/i2vs/i2vs.json`, `i2vs/i2vs.mp4`                                                         |
| Nano: Video to Video with Audio — Nano video-to-video generation with sound.                                               | `v2vs`                          | `Cosmos3-Nano`                  | `video2video` | `assets/prompts/image2video/car_driving.json`              | `assets/videos/car_driving_plain.mp4` (not vendored in this skill) | `assets/negative_prompts/image2video/neg_prompt.json` (named explicitly in the spec) | `True`         | 35 / 6.0 / 10.0 / 24 / 189 / `"720"` / `"16,9"` / 0 → `1280x720`                                                                            | `post_video` + `-F video_reference=@assets/videos/car_driving_plain.mp4;type=video/mp4` + `generate_sound=true`, `sound_duration=7.875` | `payloads/v2vs/v2vs.json`, `v2vs/v2vs.mp4`                                                         |
| Super: Text to Image — Super text-to-image generation using the same structured JSON prompt.                               | `t2i_super`                     | `Cosmos3-Super`                 | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                                  | `""`                                                                                 | `False`        | 35 / 6.0 / 10.0 / 24 / 189 / `"720"` / `"16,9"` / 0 → `1280x720`                                                                            | `post_image`                                                                                                                            | `payloads/t2i_super/t2i_super.json`, `t2i_super/t2i_super.png`                                     |
| Super: Text to Video Without Audio — Super text-to-video generation with audio disabled.                                   | `t2v_super_noaudio`             | `Cosmos3-Super`                 | `text2video`  | `assets/prompts/text2video/robot_kitchen.json`             | —                                                                  | `assets/negative_prompts/text2video/neg_prompt.json`                                 | `False`        | 35 / 6.0 / 10.0 / 24 / 189 / `"720"` / `"16,9"` / 0 → `1280x720`                                                                            | `post_video`                                                                                                                            | `payloads/t2v_super_noaudio/t2v_super_noaudio.json`, `t2v_super_noaudio/t2v_super_noaudio.mp4`     |
| Super: Image to Video Without Audio — Super image-to-video generation using its paired image asset, with audio disabled.   | `i2v_super_noaudio`             | `Cosmos3-Super`                 | `image2video` | `assets/prompts/image2video/car_driving.json`              | `assets/images/image2video/car_driving.jpg`                        | `assets/negative_prompts/image2video/neg_prompt.json`                                | `False`        | 35 / 6.0 / 10.0 / 24 / 189 / `"720"` / `"16,9"` / 0 → `1280x720`                                                                            | `post_video` + `-F input_reference=@assets/images/image2video/car_driving.jpg`                                                          | `payloads/i2v_super_noaudio/i2v_super_noaudio.json`, `i2v_super_noaudio/i2v_super_noaudio.mp4`     |
| Edge: Text to Image — Edge text-to-image generation using the same structured JSON prompt.                                 | `t2i_edge`                      | `Cosmos3-Edge`                  | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                                  | `""`                                                                                 | `False`        | 35 / — / — / 24 / 189 / `"480"` / `"16,9"` / 0 → `832x480`                                                                                  | `post_image`                                                                                                                            | `payloads/t2i_edge/t2i_edge.json`, `t2i_edge/t2i_edge.png`                                         |
| Edge: Text to Video Without Audio — Edge text-to-video generation with audio disabled.                                     | `t2v_edge_noaudio`              | `Cosmos3-Edge`                  | `text2video`  | `assets/prompts/text2video/robot_kitchen.json`             | —                                                                  | `assets/negative_prompts/text2video/neg_prompt.json`                                 | `False`        | 35 / — / — / 24 / 189 / `"480"` / `"16,9"` / 0 → `832x480`                                                                                  | `post_video`                                                                                                                            | `payloads/t2v_edge_noaudio/t2v_edge_noaudio.json`, `t2v_edge_noaudio/t2v_edge_noaudio.mp4`         |
| Edge: Image to Video Without Audio — Edge image-to-video generation using its paired image asset, with audio disabled.     | `i2v_edge_noaudio`              | `Cosmos3-Edge`                  | `image2video` | `assets/prompts/image2video/car_driving.json`              | `assets/images/image2video/car_driving.jpg`                        | `assets/negative_prompts/image2video/neg_prompt.json`                                | `False`        | 35 / — / — / 24 / 189 / `"480"` / `"16,9"` / 0 → `832x480`                                                                                  | `post_video` + `-F input_reference=@assets/images/image2video/car_driving.jpg`                                                          | `payloads/i2v_edge_noaudio/i2v_edge_noaudio.json`, `i2v_edge_noaudio/i2v_edge_noaudio.mp4`         |
| Distil: Text to Image — Distil text-to-image generation using the same structured JSON prompt.                             | `t2i_distil`                    | `Cosmos3-Distil`                | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                                  | `""`                                                                                 | `False`        | — / — / — / 24 / 189 / `"720"` / `"16,9"` / 0 → `1280x720`                                                                                  | `post_image`                                                                                                                            | `payloads/t2i_distil/t2i_distil.json`, `t2i_distil/t2i_distil.png`                                 |
| Distil: Image to Video Without Audio — Distil image-to-video generation using its paired image asset, with audio disabled. | `i2v_distil_noaudio`            | `Cosmos3-Distil`                | `image2video` | `assets/prompts/image2video/car_driving.json`              | `assets/images/image2video/car_driving.jpg`                        | `assets/negative_prompts/image2video/neg_prompt.json`                                | `False`        | — / — / — / 24 / 189 / `"720"` / `"16,9"` / 0 → `1280x720`                                                                                  | `post_video` + `-F input_reference=@assets/images/image2video/car_driving.jpg`                                                          | `payloads/i2v_distil_noaudio/i2v_distil_noaudio.json`, `i2v_distil_noaudio/i2v_distil_noaudio.mp4` |

## 7. Skill notes (not in the notebook) — port and readiness figures (from SKILL.md Variant 4)

Skill-authored text moved here from SKILL.md on 2026-10-27 (layout change only; wording unchanged apart from pointers and lead-ins that had to follow the text). Nothing in this section is notebook text; SKILL.md keeps a one-line pointer at each former location.

### 7.1 Port and readiness figures (SKILL.md Variant 4)

Keep `--host 0.0.0.0` and `-p 30000:30000` exactly as in the notebook's Docker serve
blocks (`COSMOS3_SGLANG_BASE_URL` defaults to `http://localhost:30000`).

Readiness poll timing: (observed 60–171 s for Edge across four runs;
the long end includes the in-container weight download).
