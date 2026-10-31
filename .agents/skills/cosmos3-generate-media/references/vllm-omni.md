# vLLM-Omni — distilled from the cookbook notebook

Source: `run_with_vllm_omni.ipynb` and the cookbook README at <https://github.com/NVIDIA/cosmos/blob/6e2c69afafd6444c1ca14a021aba93fcc6b7996a/cookbooks/cosmos3/generator/audiovisual/run_with_vllm_omni.ipynb> (pinned commit; this file supersedes it and is complete on its own) § 2 (the server launches and the Quantized Checkpoints section) and the § 3 note follow the notebook's last version in this
repository, main `7738e9e` (NVIDIA/cosmos-framework#329: launch mounts, `HF_TOKEN`, FP8 checkpoints) — the version this skill retires.

Scope: the vLLM-Omni backend — the notebook's prerequisites, Docker server launches (Nano, Edge, Super with tensor parallelism and layerwise offload, the two Super 4-Step students, CFG parallel), endpoint verification, the `curl`-based request helpers (`video_api_url`, `image_api_url`, `build_vllm_form`, `build_vllm_image_body`, `post_video`, `post_image`, `run_vllm_payload`), its twelve run sections collapsed into one use-case table, and the README's vLLM-Omni Quickstart and Notebook walkthrough; content shared with the other backends (`ASSET_SETS`, `FIXED_SAMPLING`, `payload_dimensions`, `create_payload`, endpoint variables, output naming) is in payloads.md and is referenced by section.

## Cosmos3 Generator Audiovisual with vLLM Omni

This notebook calls already-running vLLM Omni Cosmos3 servers with direct `curl` requests from Python.

The examples are split into Cosmos3-Nano, Cosmos3-Super, Super-4Step, and Cosmos3-Edge sections. Each section is self-contained, so you can run just one. Each section targets the matching model endpoint.

Licence header: see payloads.md § 2.1.

## 1. Prerequisites

Use a running vLLM Omni server and set endpoint environment variables before the setup cell if you are not using the local default. Text-to-image uses `/v1/images/generations`; video modes use `/v1/videos/sync`.

Guardrail prerequisite: see payloads.md § 2.3 — its `run_with_vllm_omni.ipynb` row is this notebook's paragraph verbatim (gated `nvidia/Cosmos-1.0-Guardrail` access; `HF_TOKEN` set and passed to the container; `--no-guardrails` on `vllm serve` server-wide, or `guardrails: false` in `extra_params` / `extra_args` per request).

Local-default endpoint exports (`COSMOS3_VLLM_BASE_URL`, `COSMOS3_VLLM_NANO_BASE_URL`, `COSMOS3_VLLM_SUPER_BASE_URL`, `COSMOS3_VLLM_EDGE_BASE_URL`, each `http://localhost:8000`): see payloads.md § 4.5 (`# run_with_vllm_omni.ipynb` export block).

## 2. Start the Server

Run the vLLM Omni server before running the request cells. Use the Docker image for every modality on this page. Run the launch commands from `cosmos-framework/.agents/skills`. Mount this cookbook root at `/workspace` and retain the host absolute path so client media paths resolve in Docker.

### Docker Image: Cosmos3-Nano

```bash
docker run --runtime nvidia --gpus all \
  -v "$PWD:/workspace" \
  -v "$PWD:$PWD" \
  -w /workspace \
  -e HF_TOKEN \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano \
  --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --allowed-local-media-path / \
  --port 8000 \
  --init-timeout 1800
```

### Docker Image: Cosmos3-Edge

```bash
docker run --runtime nvidia --gpus all \
  -v "$PWD:/workspace" \
  -v "$PWD:$PWD" \
  -w /workspace \
  -e HF_TOKEN \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Edge \
  --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --allowed-local-media-path / \
  --port 8000 \
  --init-timeout 1800
```

### Docker Image: Cosmos3-Super

`Cosmos3-Super` is the larger 64B model, so it usually needs more GPU memory than `Cosmos3-Nano`. `--tensor-parallel-size` splits model weights across multiple GPUs and reduces per-GPU memory use. `--enable-layerwise-offload` reduces peak GPU memory further by offloading transformer blocks between CPU and GPU, with a latency tradeoff and additional CPU RAM use. Set `--tensor-parallel-size` to the number of GPUs you want to use.

For example, on four GPUs:

```bash
docker run --runtime nvidia --gpus all \
  -v "$PWD:/workspace" \
  -v "$PWD:$PWD" \
  -w /workspace \
  -e HF_TOKEN \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Super \
  --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --allowed-local-media-path / \
  --tensor-parallel-size 4 \
  --enable-layerwise-offload \
  --port 8000 \
  --init-timeout 1800
```

### Docker Image: Cosmos3-Super-Text2Image-4Step, Cosmos3-Super-Image2Video-4Step

`Cosmos3-Super-Text2Image-4Step` and `Cosmos3-Super-Image2Video-4Step` are distilled models which require fewer inference steps for Text2Image and Image2Video tasks respectively.

Start one model at a time on port 8000 and run its matching example below. Both models use `COSMOS3_VLLM_SUPER_BASE_URL` (or `COSMOS3_VLLM_BASE_URL` when unset).

#### Cosmos3-Super-Text2Image-4Step

```bash
docker run --runtime nvidia --gpus all \
  -v "$PWD:/workspace" \
  -v "$PWD:$PWD" \
  -w /workspace \
  -e HF_TOKEN \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Super-Text2Image-4Step \
  --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --allowed-local-media-path / \
  --tensor-parallel-size 4 \
  --enable-layerwise-offload \
  --port 8000 \
  --init-timeout 1800
```

#### Cosmos3-Super-Image2Video-4Step

```bash
docker run --runtime nvidia --gpus all \
  -v "$PWD:/workspace" \
  -v "$PWD:$PWD" \
  -w /workspace \
  -e HF_TOKEN \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Super-Image2Video-4Step \
  --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --allowed-local-media-path / \
  --tensor-parallel-size 4 \
  --enable-layerwise-offload \
  --port 8000 \
  --init-timeout 1800
```

### CFG Parallel

Use `--cfg-parallel-size 2` to run the positive and negative CFG branches in parallel on two GPUs:

```bash
vllm serve nvidia/Cosmos3-Nano \
  --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --allowed-local-media-path / \
  --cfg-parallel-size 2 \
  --port 8000 \
  --init-timeout 1800
```

For Cosmos3, set CFG strength with the request-level `guidance_scale` field. Do not use `true_cfg_scale` for CFG Parallel with these Cosmos3 examples.

### Quantized Checkpoints

Published [Edge FP8](https://huggingface.co/nvidia/Cosmos3-Edge-FP8),
[Nano FP8](https://huggingface.co/nvidia/Cosmos3-Nano-FP8) and
[Super FP8](https://huggingface.co/nvidia/Cosmos3-Super-FP8) checkpoints can be
downloaded to a local directory. Or use an output from the
[quantization walkthrough](../../cookbooks/cosmos3/quantization/README.md).
Serve that directory with the original `--served-model-name` so the request
cells keep their existing model IDs. Do not select FP8 with `--revision` alone:
the Cosmos3 vLLM-Omni pipeline must receive a materialized local checkpoint.

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

Run these commands from the **Framework checkout root**. Choose one model at a
time. For a generated output set `FP8_MODEL_DIR` to its absolute directory and
`FP8_SOURCE_DIR` to the absolute source checkpoint directory, or a common parent
containing its snapshot and backing blobs. The source mount preserves exported
symlinks. If another cache holds symlink targets, mount that cache at its original
absolute path too. Keep `HF_HOME` consistent with the quantization run.

| Local output under `OUTPUT_ROOT` | `SERVED_MODEL`                           | Matching example                   |
| -------------------------------- | ---------------------------------------- | ---------------------------------- |
| `edge-fp8`                       | `nvidia/Cosmos3-Edge`                    | Edge image/video (no audio)        |
| `nano-fp8`                       | `nvidia/Cosmos3-Nano`                    | Nano image/video                   |
| `super-fp8`                      | `nvidia/Cosmos3-Super`                   | Super video                        |
| `super-t2i-fp8`                  | `nvidia/Cosmos3-Super-Text2Image`        | Super text-to-image, base sampler  |
| `super-t2i-distilled-fp8`        | `nvidia/Cosmos3-Super-Text2Image-4Step`  | Super-4Step text-to-image          |
| `super-i2v-fp8`                  | `nvidia/Cosmos3-Super-Image2Video`       | Super image-to-video, base sampler |
| `super-i2v-distilled-fp8`        | `nvidia/Cosmos3-Super-Image2Video-4Step` | Super-4Step image-to-video         |

For specialized base models, run the matching Super T2I or I2V example against
the server at `COSMOS3_VLLM_SUPER_BASE_URL`. These request bodies omit a model ID;
the single model loaded at that endpoint handles the request. Verify its served
ID with `/v1/models`. The preserved generic request payloads send 35 steps and
guidance 6.0. For a distilled checkpoint, the Cosmos3 runtime overrides these:
it reads the fixed four-step schedule from the checkpoint and forces guidance
to 1.0 (CFG off). Verify this behavior in the selected image; do not infer the
student's effective sampler from those generic request fields.
No published `fp8` revision is assumed for the four specialized Super models.

Choose one download block, then use the shared launch command below. For a
locally generated checkpoint, skip both downloads and set `FP8_MODEL_DIR` and
`SERVED_MODEL` using the table above.

#### Cosmos3-Edge FP8

```bash
export FP8_MODEL_DIR="$PWD/models/cosmos3-edge-fp8"
export SERVED_MODEL="nvidia/Cosmos3-Edge"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("nvidia/Cosmos3-Edge-FP8", local_dir=os.environ["FP8_MODEL_DIR"])
PY
```

Use the [Edge image-to-video example](#cosmos3-edge-example) with audio disabled
against `COSMOS3_VLLM_EDGE_BASE_URL` (default `http://localhost:8000`).

#### Cosmos3-Nano FP8

```bash
# Download published Nano FP8; use Cosmos3-Super-FP8 and its served ID for Super.
export FP8_MODEL_DIR="${FP8_MODEL_DIR:-$PWD/models/cosmos3-nano-fp8}"
export SERVED_MODEL="${SERVED_MODEL:-nvidia/Cosmos3-Nano}"
# Skip this download when FP8_MODEL_DIR is a locally generated checkpoint.
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("nvidia/Cosmos3-Nano-FP8", local_dir=os.environ["FP8_MODEL_DIR"])
PY
```

#### Serve the selected checkpoint

```bash
export HF_HOME="$(realpath -m "${HF_HOME:-$HOME/.cache/huggingface}")"
FP8_MODEL_DIR="$(realpath "$FP8_MODEL_DIR")"
test -f "$FP8_MODEL_DIR/hf_quant_config.json"
# Mount each destination once, including on hosts whose HOME is /root.
MOUNTS=()
declare -A MOUNT_DESTINATIONS=()
add_mount() {
  local source_path="$1" destination="$2" mode="${3:-rw}"
  if [[ -z "${MOUNT_DESTINATIONS[$destination]+present}" ]]; then
    MOUNTS+=(-v "$source_path:$destination:$mode")
    MOUNT_DESTINATIONS[$destination]=1
  fi
}
add_mount "$HF_HOME" /root/.cache/huggingface
add_mount "$HF_HOME" "$HF_HOME" ro
add_mount "$PWD/.agents/skills" /workspace
add_mount "$PWD" "$PWD"
add_mount "$FP8_MODEL_DIR" "$FP8_MODEL_DIR" ro
if [ -n "${FP8_SOURCE_DIR:-}" ]; then
  FP8_SOURCE_DIR="$(realpath "$FP8_SOURCE_DIR")"
  add_mount "$FP8_SOURCE_DIR" "$FP8_SOURCE_DIR" ro
fi
docker run --runtime nvidia --gpus all \
  -e HF_TOKEN \
  "${MOUNTS[@]}" \
  -w /workspace \
  -p 8000:8000 --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve "$FP8_MODEL_DIR" \
    --served-model-name "$SERVED_MODEL" \
    --omni --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / --port 8000 --init-timeout 1800
```

For Super, add the upstream `--tensor-parallel-size 4` and
`--enable-layerwise-offload` options if needed. Poll
`curl http://localhost:8000/v1/models` until ready and verify the ID equals
`SERVED_MODEL`, then run the matching request and visualization cells below.
The image and activation policy require runtime validation on your GPU setup.

## 3. Configure Paths and Endpoints

This setup cell only configures repo/output paths and vLLM endpoint settings. (The notebook's final version resolves the
Framework checkout root — the directory with `pyproject.toml` and `cosmos_framework/` — and anchors `COSMOS_ROOT` at its
`.agents/skills`; the skill's path convention below stands in for that cell.)

Paths: a plain output directory `OUT=outputs` replaces the notebook's output-root plumbing (payloads.md § 4.1). Endpoints (`DEFAULT_VLLM_BASE_URL`, `VLLM_ENDPOINTS`, `COSMOS3_VLLM_API_KEY`): see payloads.md § 4.5.

## 4. Verify Endpoint Configuration

```python
from urllib.parse import urlparse

for model, base_url in VLLM_ENDPOINTS.items():
    api_root = base_url.rstrip("/")
    if not api_root.endswith("/v1"):
        api_root = f"{api_root}/v1"
    parsed = urlparse(api_root)
    print(model)
    print("  api root:", api_root)
    print("  images generations:", f"{api_root}/images/generations")
    print("  videos sync:", f"{api_root}/videos/sync")
    print("  scheme:", parsed.scheme)
    print("  host:", parsed.netloc)
```

## 5. Preview Available Inputs

See payloads.md § 3.1 (prompt captions) and § 3.5 (listing rules).

## 6. Define Asset Sets, Payload Helpers, Request Helpers, and Viewer Helpers

- `IMAGE_EXTENSIONS`, `asset_path`, `compact_json_file`, `resolve_payload_path`: see payloads.md § 9.1.
- `FIXED_SAMPLING`: see payloads.md § 6.1.
- `ASSET_SETS`: see payloads.md § 8.7 (`# run_with_vllm_omni.ipynb` block).
- `payload_dimensions`: see payloads.md § 7 (`# run_with_vllm_omni.ipynb` block).
- `create_payload`: see payloads.md § 9.2 (`# run_with_vllm_omni.ipynb` block).
- `api_root_url`: see payloads.md § 4.5.
- `view_run` output discovery: see payloads.md § 4.2.

Request helpers:

```python
import base64
import json
import os
import subprocess
import time
from pathlib import Path


def video_api_url(base_url: str) -> str:
    return f"{api_root_url(base_url)}/videos/sync"


def image_api_url(base_url: str) -> str:
    return f"{api_root_url(base_url)}/images/generations"


def build_vllm_form(payload: dict) -> dict[str, str]:
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
        "num_inference_steps": str(payload["num_steps"]),
        "guidance_scale": str(payload["guidance"]),
        "flow_shift": str(payload["shift"]),
        "seed": str(payload["seed"]),
        "extra_params": json.dumps(extra_params, separators=(",", ":")),
    }
    if payload["enable_sound"]:
        form["generate_sound"] = "true"
        form["sound_duration"] = f"{payload['num_frames'] / payload['fps']:.3f}"
    return form


def build_vllm_image_body(payload: dict) -> dict:
    height, width = payload_dimensions(payload)
    return {
        "prompt": payload["prompt"],
        "size": f"{width}x{height}",
        "n": 1,
        "num_inference_steps": payload["num_steps"],
        "guidance_scale": payload["guidance"],
        "flow_shift": payload["shift"],
        "seed": payload["seed"],
        "extra_args": {
            "use_resolution_template": False,
            "guardrails": True,
        },
    }


def post_video(*, payload_path: Path, payload: dict, output_path: Path, model: str) -> None:
    url = video_api_url(VLLM_ENDPOINTS[model])
    api_key = os.environ.get("COSMOS3_VLLM_API_KEY") or None
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

    for key, value in build_vllm_form(payload).items():
        cmd += ["--form-string", f"{key}={value}"]

    if payload["model_mode"] == "image2video":
        image_path = resolve_payload_path(payload_path, payload["vision_path"])
        cmd += ["-F", f"input_reference=@{image_path}"]

    cmd += ["-o", str(tmp_path)]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        error_path.write_text((result.stdout or "") + (result.stderr or ""))
        raise RuntimeError(f"vLLM request failed with exit code {result.returncode}; see {error_path}")
    tmp_path.replace(output_path)


def post_image(*, payload: dict, output_path: Path, model: str) -> None:
    url = image_api_url(VLLM_ENDPOINTS[model])
    api_key = os.environ.get("COSMOS3_VLLM_API_KEY") or None
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
    cmd += ["-d", json.dumps(build_vllm_image_body(payload), separators=(",", ":"))]

    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        error_path.write_text((result.stdout or "") + (result.stderr or ""))
        raise RuntimeError(f"vLLM image request failed with exit code {result.returncode}; see {error_path}")
    try:
        response = json.loads(result.stdout)
        b64_json = response["data"][0]["b64_json"]
        tmp_path.write_bytes(base64.b64decode(b64_json))
    except Exception as exc:
        error_path.write_text((result.stdout or "") + (result.stderr or ""))
        raise RuntimeError(f"Could not decode vLLM image response; see {error_path}") from exc
    tmp_path.replace(output_path)


def run_vllm_payload(payload_path: Path, output_dir: str | Path, *, model: str) -> Path:
    payload_path = Path(payload_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(payload_path.read_text())
    output_ext = ".png" if payload["model_mode"] == "text2image" else ".mp4"
    output_path = output_dir / f"{payload['name']}{output_ext}"
    endpoint = image_api_url(VLLM_ENDPOINTS[model]) if payload["model_mode"] == "text2image" else video_api_url(VLLM_ENDPOINTS[model])
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

Run order: see payloads.md § 11 for this notebook's run-order paragraph (create the JSON payload, run inference, then view the generated media; examples grouped by model size; the Cosmos3-Nano and Cosmos3-Super sections are independent).

## Cosmos3-Nano Examples

Use cases for the `Cosmos3-Nano` model. This section is self-contained; you can run it without the Cosmos3-Super section below.

## Cosmos3-Super Examples

The same use cases for the larger `Cosmos3-Super` model. This section is self-contained; you can run it without the Cosmos3-Nano section above.

## Super-4Step Examples

The same audiovisual use cases for the Cosmos distilled model. This section is self-contained; you can run it without any of the sections above.

## Cosmos3-Edge Example

Image-to-video use case for the `Cosmos3-Edge` model. The video example has audio disabled. This section is self-contained; you can run it without the model-specific sections above.

## Use-case table (all run sections)

Each run section is the same "Create Payload" and "Run" cells with its row's use-case key and model:

```python
# Create Payload
<use_case>_payload, <use_case>_output, <use_case>_model = create_payload("<use_case>", backend="vllm")
# Run
run_vllm_payload(<use_case>_payload, <use_case>_output, model="<Model>")
```

| Notebook section — description                                                                                                                                        | Use case            | Model (`ASSET_SETS["model"]`, also the `model=` argument of `run_vllm_payload`) | Mode          | Prompt asset                                               | Image asset                                        | `enable_sound` | `sampling` override (on top of `FIXED_SAMPLING`, payloads.md § 6.1; — = none)                                                            | Payload → output (under `<OUT>/vllm/`)                                                          |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------------------------------- | ------------- | ---------------------------------------------------------- | -------------------------------------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Nano: Text to Image** — Nano text-to-image generation using a structured JSON prompt.                                                                               | `t2i`               | `Cosmos3-Nano`                                                                  | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                  | `False`        | —                                                                                                                                        | `payloads/t2i/t2i.json` → `t2i/t2i.png`                                                         |
| **Nano: Text to Video Without Audio** — Nano text-to-video generation with audio disabled.                                                                            | `t2v_nano_noaudio`  | `Cosmos3-Nano`                                                                  | `text2video`  | `assets/prompts/text2video/robot_kitchen.json`             | —                                                  | `False`        | —                                                                                                                                        | `payloads/t2v_nano_noaudio/t2v_nano_noaudio.json` → `t2v_nano_noaudio/t2v_nano_noaudio.mp4`     |
| **Nano: Text to Video with Audio** — Nano text-to-video generation with generated audio.                                                                              | `t2vs`              | `Cosmos3-Nano`                                                                  | `text2video`  | `assets/prompts/text2video/robot_pouring_water_audio.json` | —                                                  | `True`         | —                                                                                                                                        | `payloads/t2vs/t2vs.json` → `t2vs/t2vs.mp4`                                                     |
| **Nano: Image to Video Without Audio** — Nano image-to-video generation using its paired image asset, with audio disabled.                                            | `i2v_nano_noaudio`  | `Cosmos3-Nano`                                                                  | `image2video` | `assets/prompts/image2video/car_driving.json`              | `assets/images/image2video/car_driving.jpg`        | `False`        | —                                                                                                                                        | `payloads/i2v_nano_noaudio/i2v_nano_noaudio.json` → `i2v_nano_noaudio/i2v_nano_noaudio.mp4`     |
| **Nano: Image to Video with Audio** — Nano image-to-video generation using its paired image asset and generated audio.                                                | `i2vs`              | `Cosmos3-Nano`                                                                  | `image2video` | `assets/prompts/image2video/coastal_road_audio.json`       | `assets/images/image2video/coastal_road_audio.jpg` | `True`         | —                                                                                                                                        | `payloads/i2vs/i2vs.json` → `i2vs/i2vs.mp4`                                                     |
| **Super: Text to Image** — Super text-to-image generation using the same structured JSON prompt.                                                                      | `t2i_super`         | `Cosmos3-Super`                                                                 | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                  | `False`        | —                                                                                                                                        | `payloads/t2i_super/t2i_super.json` → `t2i_super/t2i_super.png`                                 |
| **Super: Text to Video Without Audio** — Super text-to-video generation with audio disabled.                                                                          | `t2v_super_noaudio` | `Cosmos3-Super`                                                                 | `text2video`  | `assets/prompts/text2video/robot_kitchen.json`             | —                                                  | `False`        | —                                                                                                                                        | `payloads/t2v_super_noaudio/t2v_super_noaudio.json` → `t2v_super_noaudio/t2v_super_noaudio.mp4` |
| **Super: Image to Video Without Audio** — Super image-to-video generation using its paired image asset, with audio disabled.                                          | `i2v_super_noaudio` | `Cosmos3-Super`                                                                 | `image2video` | `assets/prompts/image2video/car_driving.json`              | `assets/images/image2video/car_driving.jpg`        | `False`        | —                                                                                                                                        | `payloads/i2v_super_noaudio/i2v_super_noaudio.json` → `i2v_super_noaudio/i2v_super_noaudio.mp4` |
| **Super: Video to Video Without Audio** — Super video-to-video generation using the same reference-video request shape as Nano, against the `Cosmos3-Super` endpoint. | `v2v_super_noaudio` | `Cosmos3-Super`                                                                 | —             | —                                                          | —                                                  | —              | —                                                                                                                                        | —                                                                                               |
| **Super-4Step: Text to Image** — Cosmos3-Super-Text2Image-4Step generation is using the same structured JSON prompt.                                                  | `t2i_distil`        | `Cosmos3-Super-Text2Image-4Step`                                                | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                  | `False`        | —                                                                                                                                        | `payloads/t2i_distil/t2i_distil.json` → `t2i_distil/t2i_distil.png`                             |
| **Super-4Step: Image to Video Without Audio** — Cosmos3-Super-Image2Video-4Step generation is using its paired image asset, with audio disabled.                      | `i2v_distil`        | `Cosmos3-Super-Image2Video-4Step`                                               | `image2video` | `assets/prompts/image2video/car_driving.json`              | `assets/images/image2video/car_driving.jpg`        | `False`        | —                                                                                                                                        | `payloads/i2v_distil/i2v_distil.json` → `i2v_distil/i2v_distil.mp4`                             |
| **Edge: Image to Video Without Audio** — Edge image-to-video generation using its paired image asset, with audio disabled.                                            | `i2v_edge_noaudio`  | `Cosmos3-Edge`                                                                  | `image2video` | `assets/prompts/image2video/car_driving_edge.json`         | `assets/images/image2video/car_driving.jpg`        | `False`        | `{"num_steps": 20, "guidance": 5.0, "shift": 8.0, "fps": 24, "num_frames": 121, "resolution": "480", "aspect_ratio": "16,9", "seed": 0}` | `payloads/i2v_edge_noaudio/i2v_edge_noaudio.json` → `i2v_edge_noaudio/i2v_edge_noaudio.mp4`     |

## Run with vLLM-Omni

### Quickstart

Set up the environment and start the server:
[vLLM-Omni setup](../../cookbooks/cosmos3/README.md#vllm-omni) (Docker recommended). Run the
Docker command from this folder (`COSMOS3_WORKDIR` defaults to the current
directory) with **`COSMOS3_HOST_PORT=8000`** unless you already have another
server on that port.

Send a text-to-video request with the OpenAI-compatible video API:

```python
import json
from pathlib import Path

import requests

prompt = json.load(open("assets/prompts/text2video/robot_kitchen.json"))
negative = json.load(open("assets/negative_prompts/text2video/neg_prompt.json"))

response = requests.post(
    "http://localhost:8000/v1/videos/sync",
    data={
        "prompt": json.dumps(prompt),
        "negative_prompt": json.dumps(negative),
        "size": "1280x720",
        "num_frames": "189",
        "fps": "24",
        "num_inference_steps": "35",
        "guidance_scale": "6.0",
        "flow_shift": "10.0",
        "seed": "0",
        "extra_params": json.dumps(
            {
                "use_resolution_template": False,
                "use_duration_template": False,
                "guardrails": True,
            }
        ),
    },
    headers={"Accept": "video/mp4"},
)
response.raise_for_status()
Path("/tmp/cosmos3_t2v.mp4").write_bytes(response.content)
```

For image-to-video, post to the same endpoint with an image under
`files={"input_reference": ...}`. For audio, add `"generate_sound": "true"`.
For video-to-video, upload a source video under `input_reference` and choose the
clean conditioning frames through `extra_params`:

```python
from pathlib import Path

source_video = Path("../action/assets/videos/av_0.mp4").resolve()  # not vendored in this skill
with source_video.open("rb") as video_file:
    response = requests.post(
        "http://localhost:8000/v1/videos/sync",
        data={
            "prompt": "Continue the same driving scene with smooth natural motion.",
            "negative_prompt": "blurry, distorted, low quality, jittery, deformed",
            "size": "832x480",
            "num_frames": "61",
            "fps": "10",
            "num_inference_steps": "35",
            "guidance_scale": "6.0",
            "flow_shift": "10.0",
            "seed": "2222",
            "extra_params": json.dumps(
                {
                    "use_resolution_template": False,
                    "use_duration_template": False,
                    "guardrails": True,
                    "condition_frame_indexes_vision": [0, 1],
                    "condition_video_keep": "first",
                }
            ),
        },
        files={"input_reference": (source_video.name, video_file, "video/mp4")},
        headers={"Accept": "video/mp4"},
    )
response.raise_for_status()
Path("/tmp/cosmos3_v2v.mp4").write_bytes(response.content)
```

### Notebook walkthrough

`run_with_vllm_omni.ipynb` is the full tutorial for
the vLLM-Omni backend: it walks through text-to-image, text-to-video, and
image-to-video requests with audio on or off plus standard video-to-video
requests. Server launch options (Nano and
Super, tensor parallelism, layerwise offload, and CFG-parallel variants) live in
the [shared environment setup guide](../../cookbooks/cosmos3/README.md#vllm-omni).

## 7. Skill notes (not in the notebook) — `HF_TOKEN` (from SKILL.md Variant 3)

Skill-authored text moved here from SKILL.md on 2026-10-27 (layout change only; wording unchanged). Nothing in this section is notebook text; SKILL.md keeps a one-line pointer at each former location.

### 7.1 Why the `-e HF_TOKEN` line (SKILL.md Variant 3)

The notebook's final version (#329) passes `HF_TOKEN` into the container in every launch above; this skill had added the
line before that (eval run 2026-10-26, cold cache on a fresh node): the first launch downloads `nvidia/Cosmos3-<model>`
into `HF_HOME`, and anonymous Hub calls are rate-limited — the container crashed with HTTP 429 and its log asked
for `HF_TOKEN`; with guardrails enabled the token must also be authorised for the gated guardrail repo ("also set
and pass `HF_TOKEN` to the container").
