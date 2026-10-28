# Distilled 4-step students on TensorRT-LLM — distilled from the cookbook notebook

Source: `run_distilled_with_trt_llm.ipynb` and the cookbook README at <https://github.com/NVIDIA/cosmos/blob/6e2c69afafd6444c1ca14a021aba93fcc6b7996a/cookbooks/cosmos3/generator/audiovisual/run_distilled_with_trt_llm.ipynb> (pinned commit; this file supersedes it and is complete on its own)

Scope: the whole distilled notebook — introduction, prerequisites, server launch, endpoint configuration, the backend-only request helpers and the two use cases (`t2i_distilled`, `i2v_distilled`) — plus the README's "Distilled 4-step notebook" section; cells shared with the other backend notebooks live in payloads.md and are referenced by section.

## Quickstart (cookbook README)

### Distilled 4-step notebook

[`run_distilled_with_trt_llm.ipynb`](https://github.com/NVIDIA/cosmos/blob/6e2c69afafd6444c1ca14a021aba93fcc6b7996a/cookbooks/cosmos3/generator/audiovisual/run_distilled_with_trt_llm.ipynb) is the
tutorial for the two published DMD2-distilled Cosmos3-Super students,
`nvidia/Cosmos3-Super-Text2Image-4Step` and
`nvidia/Cosmos3-Super-Image2Video-4Step`. Each serves on a single GPU and runs a
fixed four-step stochastic schedule with classifier-free guidance baked into the
weights, so requests omit `num_inference_steps` and `guidance_scale` entirely
and let the server read both from the checkpoint; sending a conflicting value is
rejected rather than clamped. Requests also leave `use_system_prompt` unset so
the image-to-video student's `default_use_system_prompt: true` applies. These
students cover text-to-image and image-to-video only. Server launch commands
live in the
[shared environment setup guide](https://github.com/NVIDIA/cosmos/blob/6e2c69afafd6444c1ca14a021aba93fcc6b7996a/cookbooks/cosmos3/README.md#tensorrt-llm-generator).

## Cosmos3 Generator Distilled 4-Step with TensorRT-LLM

Licence header: see payloads.md § 2.1.

This notebook calls already-running TensorRT-LLM VisualGen servers with direct `curl` requests from Python, using the two published DMD2-distilled Cosmos3-Super students:

- [`nvidia/Cosmos3-Super-Text2Image-4Step`](https://huggingface.co/nvidia/Cosmos3-Super-Text2Image-4Step) — text-to-image
- [`nvidia/Cosmos3-Super-Image2Video-4Step`](https://huggingface.co/nvidia/Cosmos3-Super-Image2Video-4Step) — image-to-video

Each section is self-contained, so you can run just one. Both checkpoints serve on a single GPU.

A distilled checkpoint is not a base checkpoint with fewer steps. It ships a `FlowMatchEulerDiscreteScheduler` whose `fixed_step_sampler_config.t_list` locks the schedule to four stochastic (SDE) steps, and classifier-free guidance is baked into the weights, so the only valid `guidance_scale` is `1.0`. TensorRT-LLM reads both facts from the checkpoint:

- **Leave `num_inference_steps` and `guidance_scale` out of the request.** The server fills in the checkpoint's four steps and guidance `1.0`. Sending any other value is rejected with a `400`, not silently clamped.
- **Leave `use_system_prompt` unset.** `Cosmos3-Super-Image2Video-4Step` declares `default_use_system_prompt: true` in its `model_index.json`; sending `false` overrides that default and changes the tokenized prompt.
- `flow_shift` does not apply. It is a UniPC knob, and distilled checkpoints do not use UniPC.

These students cover text-to-image and image-to-video only. For text-to-video, video-to-video, or synchronized audio, use the base Cosmos3-Nano / Cosmos3-Super checkpoints in [`run_with_trt_llm.ipynb`](https://github.com/NVIDIA/cosmos/blob/6e2c69afafd6444c1ca14a021aba93fcc6b7996a/cookbooks/cosmos3/generator/audiovisual/run_with_trt_llm.ipynb).

## 1. Prerequisites

Use a running TensorRT-LLM server with Cosmos3 VisualGen distilled-checkpoint support, and set endpoint environment variables before the setup cell if you are not using the local default. Generation uses `/v1/videos/generations`; Cosmos3 text-to-image runs as a one-frame VisualGen video request with `num_frames=1`, `seconds=1`, and `fps=8`. Image-to-video sends its reference image as multipart `input_reference`.

Install the `ffmpeg` CLI in the server environment. TensorRT-LLM uses it to encode MP4; without it, the fallback AVI encoder is used instead.

Guardrail prerequisite: see payloads.md § 2.3 (this notebook carries the `run_with_trt_llm.ipynb` wording of that table).

The two checkpoints are served separately, so run one server at a time on a single GPU and point the matching environment variable at it. Both variables default to `COSMOS3_TRTLLM_BASE_URL`: for the four `export` lines (`COSMOS3_TRTLLM_BASE_URL`, `COSMOS3_TRTLLM_T2I_4STEP_BASE_URL`, `COSMOS3_TRTLLM_I2V_4STEP_BASE_URL`, `COSMOS3_TRTLLM_API_KEY`) see payloads.md § 4.5, `run_distilled_with_trt_llm.ipynb` block.

## 2. Start the Server

Run the TensorRT-LLM VisualGen server before running the request cells. To build a checkout from source, follow NVIDIA's [Build from Source](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source.html) guide. Install `ffmpeg` in that environment before serving (`apt-get update && apt-get install -y ffmpeg` on Ubuntu/Debian images).

Run the commands below **from inside the TensorRT-LLM checkout** — the directory that contains `examples/` — or set `TRTLLM_ROOT` to that checkout's path yourself. `trtllm-serve` does not validate `--visual_gen_args` until it has already started, so check the path first rather than reading it out of a stack trace:

```bash
export TRTLLM_ROOT="${TRTLLM_ROOT:-$PWD}"

test -f "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-t2i-1gpu.yaml" \
  || echo "TRTLLM_ROOT=$TRTLLM_ROOT does not look like a TensorRT-LLM checkout"
```

### Cosmos3-Super-Text2Image-4Step

Serve the distilled text-to-image student with the shared text-to-image config, which warms the deployed 1024x1024 single-frame shape instead of the omni video shape:

```bash
trtllm-serve nvidia/Cosmos3-Super-Text2Image-4Step \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-t2i-1gpu.yaml" \
  --port 8000
```

### Cosmos3-Super-Image2Video-4Step

The distilled image-to-video student's deployed shape is the default omni video shape (720p x 189 frames), so it needs no config file and therefore no `TRTLLM_ROOT`:

```bash
trtllm-serve nvidia/Cosmos3-Super-Image2Video-4Step \
  --port 8000
```

TensorRT-LLM exposes `/health` when the server is ready. On startup it logs the schedule it read from the checkpoint:

```text
Distilled Cosmos3 checkpoint: fixed 4-step schedule [1.0, 0.9375, 0.8333333333333334, 0.625], classifier-free guidance baked in.
```

This notebook sends Cosmos3 model-specific controls through `extra_params`, so use a TensorRT-LLM release that includes the Cosmos3 VisualGen API schema and the distilled sampling policy.

## 3. Configure Paths and Endpoints

This setup cell only configures repo/output paths and TensorRT-LLM endpoint settings.

In this skill the repo/output part of the cell (`find_repo_root`, `COSMOS_ROOT`, `COSMOS3_AUDIOVISUAL_ROOT`, `COSMOS3_AUDIOVISUAL_OUTPUT_ROOT`) is replaced by the skill-relative configuration block of payloads.md § 4.1: assets resolve against the skill directory and a plain output directory `OUT=outputs` stands in for the notebook's output-root plumbing (said once here; every helper below runs unchanged on top of that block). The endpoint part of the cell — `DEFAULT_TRTLLM_BASE_URL`, the `TRTLLM_ENDPOINTS` dictionary and `os.environ.setdefault("COSMOS3_TRTLLM_API_KEY", "tensorrt_llm")` — is in payloads.md § 4.5, `run_distilled_with_trt_llm.ipynb` block, verbatim.

## 4. Verify Endpoint Configuration

`api_root_url`, `server_root_url` and `health_url` are shared with `run_with_trt_llm.ipynb`: see payloads.md § 4.5. `video_api_url` is this notebook's own:

```python
def video_api_url(base_url: str) -> str:
    return f"{api_root_url(base_url)}/videos/generations"
```

## 5. Preview Available Inputs

The three inputs this notebook previews (`assets/prompts/text2image/robot_draping.json`, `assets/prompts/image2video/car_driving.json`, `assets/images/image2video/car_driving.jpg`) and the caption rule it prints them with: see payloads.md § 3.5 and § 3.1.

## 6. Define Asset Sets, Payload Helpers, Request Helpers, and Viewer Helpers

Shared with the other notebooks and held verbatim in payloads.md:

- `IMAGE_EXTENSIONS`, `asset_path`, `compact_json_file`, `resolve_payload_path` — see payloads.md § 9.1
- `DISTILLED_SAMPLING` (with its comment) — see payloads.md § 6.1
- `TRTLLM_EXTRA_PARAMS` (with its comment) — see payloads.md § 6.3
- `ASSET_SETS` — see payloads.md § 8.7 (source) and § 8.3 (table); collapsed together with the per-use-case cells into the use-case table at the end of this file
- `payload_dimensions` — see payloads.md § 7 (`run_distilled_with_trt_llm.ipynb` block)
- `create_payload` (default `backend="trt_llm_distilled"`) — see payloads.md § 9.2 (`run_distilled_with_trt_llm.ipynb` block)
- `check_trtllm_server`, `_auth_headers` — see payloads.md § 9.3

Backend-only request helpers of this notebook (the imports they need beyond those of payloads.md §§ 9.1 and 9.3):

```python
import json
import subprocess
import time
from pathlib import Path


def build_trtllm_video_body(payload: dict) -> dict:
    """Request body for a distilled checkpoint.

    `num_inference_steps` and `guidance_scale` are intentionally absent: the
    server resolves both from the checkpoint and rejects conflicting values.
    """
    height, width = payload_dimensions(payload)
    body = {
        "prompt": payload["prompt"],
        "size": f"{width}x{height}",
        "seconds": payload.get("seconds", payload["num_frames"] / payload["fps"]),
        "fps": payload["fps"],
        "num_frames": payload["num_frames"],
        "max_sequence_length": payload["max_sequence_length"],
        "seed": payload["seed"],
    }
    if payload.get("negative_prompt") is not None:
        body["negative_prompt"] = payload["negative_prompt"]
    body["extra_params"] = payload["extra_params"]
    return body


def _video_extension_from_response(header_text: str, content_path: Path) -> str:
    lowered = header_text.lower()
    if "video/x-msvideo" in lowered or "video/avi" in lowered:
        return ".avi"
    head = content_path.read_bytes()[:16]
    if head.startswith(b"RIFF") and b"AVI" in head[:12]:
        return ".avi"
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return ".mp4"
    return ".mp4"


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
        "Accept: video/mp4, video/x-msvideo, application/octet-stream",
    ]
    cmd += _auth_headers()

    if payload["model_mode"] == "image2video":
        form_body = dict(body)
        if "extra_params" in form_body:
            form_body["extra_params"] = json.dumps(form_body["extra_params"], separators=(",", ":"))
        for key, value in form_body.items():
            cmd += ["--form-string", f"{key}={value}"]
        reference_path = resolve_payload_path(payload_path, payload["vision_path"])
        cmd += ["-F", f"input_reference=@{reference_path};type=image/jpeg"]
    else:
        cmd += ["-H", "Content-Type: application/json"]
        cmd += ["-d", json.dumps(body, separators=(",", ":"))]

    cmd += ["-o", str(tmp_path)]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        error_path.write_text((result.stdout or "") + (result.stderr or ""))
        raise RuntimeError(f"TensorRT-LLM request failed with exit code {result.returncode}; see {error_path}")

    header_text = header_path.read_text() if header_path.exists() else ""
    ext = _video_extension_from_response(header_text, tmp_path)
    output_path = output_stem.with_suffix(ext)
    if output_path.exists():
        output_path.unlink()
    tmp_path.replace(output_path)
    return output_path


def run_trtllm_payload(payload_path: Path, output_dir: str | Path, *, model: str) -> Path:
    payload_path = Path(payload_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(payload_path.read_text())
    output_stem = output_dir / payload["name"]
    endpoint = video_api_url(TRTLLM_ENDPOINTS[model])
    print("endpoint:", endpoint)
    print("payload:", payload_path)
    print("output stem:", output_stem)
    if payload["model_mode"] == "image2video":
        print("input image:", resolve_payload_path(payload_path, payload["vision_path"]))
    t0 = time.time()
    output_path = post_video(payload_path=payload_path, payload=payload, output_stem=output_stem, model=model)
    print(f"wrote {output_path} in {time.time() - t0:.1f}s")
    return output_path
```

`view_run` output discovery (`.mp4` and `.avi` only, skipping `_preview.mp4` / `_browser.mp4`): see payloads.md § 4.2. The notebook's closing workflow paragraph ("Run each use case top-to-bottom ... The two sections are independent, so you can run just one."): see payloads.md § 11.

## Cosmos3-Super-Text2Image-4Step Examples

Use cases for the distilled text-to-image student. This section is self-contained; you can run it without the image-to-video section below.

### Distilled: Text to Image

Distilled text-to-image generation using a structured JSON prompt at the deployed 1024x1024 shape. TensorRT-LLM Cosmos3 returns this as a one-frame video from the video generation endpoint; the request uses `num_frames=1`, `seconds=1`, and `fps=8`.

Create Payload / Run / View Results: row `t2i_distilled` of the use-case table below.

## Cosmos3-Super-Image2Video-4Step Examples

Use cases for the distilled image-to-video student. This section is self-contained; you can run it without the text-to-image section above. Point `COSMOS3_TRTLLM_I2V_4STEP_BASE_URL` at a server started with this checkpoint before running it.

### Distilled: Image to Video

Distilled image-to-video generation from the paired car-driving image and prompt at the deployed 720p x 189-frame shape. The request leaves `use_system_prompt` unset so the checkpoint's `default_use_system_prompt: true` applies.

Create Payload / Run / View Results: row `i2v_distilled` of the use-case table below.

## Use cases (collapsed Create Payload / Run / View Results cells)

Every use case of `ASSET_SETS` (payloads.md § 8.7) with the parameters `create_payload` (payloads.md § 9.2) and `build_trtllm_video_body` / `post_video` (§ 6 above) derive from it. `<OUT>` is the output directory of payloads.md § 4.1; `<api root>` is `api_root_url(TRTLLM_ENDPOINTS[model])`.

| Use case        | Notebook section                                                     | Model (`TRTLLM_ENDPOINTS` key → endpoint variable)                                                          | Mode (`model_mode`) | Prompt asset                                   | Image asset                                                                                         | Negative prompt                                                                                          | Sampling override → payload sampling fields                                                                                                                                                                                                                                                                                                                                                                                                                                             | Request (`POST <api root>/videos/generations`)                                                                                                                                                                                                                                                                                                                                | Output                                                                                                                                                                                                                                                                          | Cells (Create Payload → Run → View Results)                                                                                                                                                                                                               |
| --------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `t2i_distilled` | Cosmos3-Super-Text2Image-4Step Examples → Distilled: Text to Image   | `Cosmos3-Super-Text2Image-4Step` → `COSMOS3_TRTLLM_T2I_4STEP_BASE_URL` (default `COSMOS3_TRTLLM_BASE_URL`)  | `text2image`        | `assets/prompts/text2image/robot_draping.json` | —                                                                                                   | none (`create_payload` leaves the key out)                                                               | spec `sampling` `{"resolution": "1024", "aspect_ratio": "1,1"}` (spec comment: "The deployed text-to-image shape warmed by cosmos3-t2i-1gpu.yaml.") on top of `DISTILLED_SAMPLING`; `create_payload` then sets `num_frames` 1, `fps` 8, `seconds` 1 → payload `fps` 8, `num_frames` 1, `seconds` 1, `max_sequence_length` 4096, `resolution` `"1024"`, `aspect_ratio` `"1,1"`, `seed` 0; `extra_params` = `TRTLLM_EXTRA_PARAMS` (spec adds none); no `num_steps` / `guidance` / `shift` | JSON body (`-H "Content-Type: application/json" -d`): `prompt`, `size` (`payload_dimensions`, payloads.md § 7), `seconds` 1, `fps` 8, `num_frames` 1, `max_sequence_length` 4096, `seed` 0, `extra_params`; `Accept: video/mp4, video/x-msvideo, application/octet-stream`; `Authorization: Bearer $COSMOS3_TRTLLM_API_KEY` when the key is non-empty                         | `name` `t2i_distilled`; payload file, output directory, media suffix (`_video_extension_from_response`) and side files (`post_video`): payloads.md § 4.2 with `<backend>` `trt_llm_distilled` (the `create_payload` default); `_INPUT` / `_OUTPUT` variables: payloads.md § 4.3 | `t2i_payload, t2i_output, t2i_model = create_payload("t2i_distilled")` → `check_trtllm_server(t2i_model)` then `run_trtllm_payload(t2i_payload, t2i_output, model=t2i_model)` → `view_run(t2i_output)` (Jupyter viewer; discovery rule payloads.md § 4.2) |
| `i2v_distilled` | Cosmos3-Super-Image2Video-4Step Examples → Distilled: Image to Video | `Cosmos3-Super-Image2Video-4Step` → `COSMOS3_TRTLLM_I2V_4STEP_BASE_URL` (default `COSMOS3_TRTLLM_BASE_URL`) | `image2video`       | `assets/prompts/image2video/car_driving.json`  | `assets/images/image2video/car_driving.jpg` (stored as `vision_path`, relative to the payload file) | `assets/negative_prompts/image2video/neg_prompt.json` (compact JSON, per-mode rule of payloads.md § 3.3) | none — `DISTILLED_SAMPLING` unchanged → payload `fps` 24, `num_frames` 189, `max_sequence_length` 4096, `resolution` `"720"`, `aspect_ratio` `"16,9"`, `seed` 0; `extra_params` = `TRTLLM_EXTRA_PARAMS` (spec adds none); no `num_steps` / `guidance` / `shift`                                                                                                                                                                                                                         | multipart: every body field as `--form-string key=value` (`prompt`, `size` (`payload_dimensions`, payloads.md § 7), `seconds` `189 / 24`, `fps` 24, `num_frames` 189, `max_sequence_length` 4096, `seed` 0, `negative_prompt`, `extra_params` as compact JSON string) plus `-F input_reference=@<car_driving.jpg>;type=image/jpeg`; same `Accept` and `Authorization` headers | `name` `i2v_distilled`; payload file, output directory, media suffix (`_video_extension_from_response`) and side files (`post_video`): payloads.md § 4.2 with `<backend>` `trt_llm_distilled`; `_INPUT` / `_OUTPUT` variables: payloads.md § 4.3                                | `i2v_payload, i2v_output, i2v_model = create_payload("i2v_distilled")` → `check_trtllm_server(i2v_model)` then `run_trtllm_payload(i2v_payload, i2v_output, model=i2v_model)` → `view_run(i2v_output)` (Jupyter viewer; discovery rule payloads.md § 4.2) |
