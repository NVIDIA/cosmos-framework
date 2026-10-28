# NIM — distilled from the cookbook notebook

Source: `run_with_nim.ipynb` and the cookbook README at <https://github.com/NVIDIA/cosmos/blob/6e2c69afafd6444c1ca14a021aba93fcc6b7996a/cookbooks/cosmos3/generator/audiovisual/run_with_nim.ipynb> (pinned commit; this file supersedes it and is complete on its own)

Scope: the `Cosmos3-Generator` NIM backend as the notebook runs it — container launch, readiness wait, service inspection, the `POST /v1/infer` request parameters, the Text2Video and Image2Video requests, the playback note and cleanup — followed by the README's `## Run with NIM` quickstart, notebook walkthrough and limitations; what the notebook shares with the other backends (README introduction with the NIM scope, guardrail and NGC notes, licence header, asset catalogue, path/output conventions, endpoints, model identifiers, sampling summary, request table) lives in payloads.md §§ 1, 2.1, 2.3, 3.2, 3.5, 4.1–4.2, 4.5, 5, 6.1, 6.3, 8.6 and 10 and is referenced below rather than repeated.

## Cosmos3 Generator inference with NVIDIA NIM

Licence header: see payloads.md § 2.1. Generator NIM guardrail switch and NGC-authentication note (README): see payloads.md § 1 and § 2.3.

This notebook runs the prebuilt `Cosmos3-Generator` NIM container and sends video-generation requests to its HTTP API. It mirrors the Reasoner NIM cookbook structure, but uses the Generator NIM API contract:

1. Launch `nvcr.io/nim/nvidia/cosmos3-generator:1.0.0`.
2. Wait for `GET /v1/health/ready`.
3. Inspect `/v1/models`, `/v1/metadata`, and `/v1/version`.
4. Send Text2Video and Image2Video requests to `POST /v1/infer`.
5. Decode the JSON response field `b64_video` into `.mp4` files and preview them.

The NIM container downloads model artifacts from NGC into a local cache on first launch. You need an `NGC_API_KEY` and Docker login to `nvcr.io` before starting the container.

Useful links:

- Container [page](https://catalog.ngc.nvidia.com/orgs/nim/nvidia/containers/cosmos3-generator/-) on NGC
- [Documentation](https://docs.nvidia.com/nim/cosmos/latest/introduction.html) for the Cosmos NIMs, each page have a table specific to Cosmos3-Generator NIM

## Capability scope

`Cosmos3-Generator` NIM currently exposes only the two video-generation modes below. It infers the mode automatically from request fields; there is no explicit `mode` field.

| Workflow               | Request shape                  | Output                | Supported here? |
| ---------------------- | ------------------------------ | --------------------- | --------------- |
| Text2Video             | non-empty `prompt`, no `image` | JSON with `b64_video` | Yes             |
| Image2Video            | `image` plus optional `prompt` | JSON with `b64_video` | Yes             |
| Text-to-image          | —                              | image                 | No              |
| Video-to-video         | —                              | video                 | No              |
| Sound/audio generation | —                              | video + audio         | No              |
| Action modes           | —                              | video/action          | No              |
| Transfer controls      | —                              | controlled video      | No              |

The request schema used in this notebook is Cosmos3-specific: `prompt`, `negative_prompt`, `image`, `seed`, `guidance_scale`, `steps`, `resolution`, `num_output_frames`, and `fps`. Unknown request fields are rejected by the server.

## 1. Launch the NIM container

Set `NGC_API_KEY` before launching Jupyter, or set it in the notebook environment. Authenticate Docker once with:

```bash
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

The default launch below uses Nano (`NIM_MODEL_SIZE=nano`), FP8 (`NIM_PRECISION=fp8`), and the latency profile (`NIM_PERF_PROFILE=latency`). Set `NIM_MODEL_SIZE=super` before running the launch cell to serve the larger model.

Paths, said once for this file: `ASSETS_ROOT` is this skill's vendored `assets/` tree and a plain `OUT=outputs` replaces the notebook's repo-root/output-root plumbing, so `OUTPUT_DIR` is `<OUT>/cosmos3_generator_nim` (payloads.md §§ 4.1–4.2).

```python
from pathlib import Path
import base64
import json
import mimetypes
import os
import subprocess
import time

import requests


ASSETS_ROOT = Path(".").resolve() / "assets"               # this skill's directory holds assets/
OUT = Path(os.environ.get("OUT", "outputs")).resolve()     # plain output directory
OUTPUT_DIR = OUT / "cosmos3_generator_nim"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

I2V_IMAGE = ASSETS_ROOT / "images" / "image2video" / "humanoid_robot.jpg"
assert I2V_IMAGE.exists(), I2V_IMAGE

NIM_PORT = int(os.environ.get("NIM_PORT", "8000"))
BASE_URL = f"http://127.0.0.1:{NIM_PORT}"


def file_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def write_b64_video(response_json: dict, path: Path) -> Path:
    data = response_json["b64_video"]
    path.write_bytes(base64.b64decode(data))
    return path


def post_infer(payload: dict, output_name: str) -> Path:
    response = requests.post(
        f"{BASE_URL}/v1/infer",
        json=payload,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        timeout=60 * 60,
    )
    response.raise_for_status()
    output_path = OUTPUT_DIR / output_name
    return write_b64_video(response.json(), output_path)
```

```bash
set -euo pipefail

: "${NGC_API_KEY:?Set NGC_API_KEY before launching the Cosmos3-Generator NIM}"

export CONTAINER_NAME="${CONTAINER_NAME:-nvidia-cosmos3-generator}"
export IMG_NAME="${IMG_NAME:-nvcr.io/nim/nvidia/cosmos3-generator:1.0.0}"
export NIM_MODEL_SIZE="${NIM_MODEL_SIZE:-nano}"
export NIM_PRECISION="${NIM_PRECISION:-fp8}"
export NIM_PERF_PROFILE="${NIM_PERF_PROFILE:-latency}"
export LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim}"
export NIM_PORT="${NIM_PORT:-8000}"
mkdir -p "$LOCAL_NIM_CACHE"
chmod -R 777 "$LOCAL_NIM_CACHE" 2>/dev/null || true

# The container name must be free. If a previous run is still up, stop it first:
#   docker stop nvidia-cosmos3-generator
docker run -d --rm --name="$CONTAINER_NAME" \
  --runtime=nvidia \
  --gpus all \
  --shm-size=32GB \
  --ulimit nofile=65536:65536 \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -e NIM_MODEL_SIZE="$NIM_MODEL_SIZE" \
  -e NIM_PRECISION="$NIM_PRECISION" \
  -e NIM_PERF_PROFILE="$NIM_PERF_PROFILE" \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -p "${NIM_PORT}:8000" \
  "$IMG_NAME"

echo "NIM container '$CONTAINER_NAME' launching on port $NIM_PORT"
echo "model_size=$NIM_MODEL_SIZE precision=$NIM_PRECISION profile=$NIM_PERF_PROFILE"
```

## 2. Wait for readiness

The first run downloads model artifacts and may spend additional time compiling/warming the engine. The server is ready for inference when `GET /v1/health/ready` returns HTTP 200.

```bash
set -euo pipefail

PORT="${NIM_PORT:-8000}"
CONTAINER_NAME="${CONTAINER_NAME:-nvidia-cosmos3-generator}"

echo "Waiting for Cosmos3-Generator NIM on port ${PORT}..."

for i in $(seq 1 3600); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/health/ready" >/dev/null 2>&1; then
    echo "NIM server is ready."
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '${CONTAINER_NAME}' is no longer running. Recent logs:"
    docker logs --tail 160 "$CONTAINER_NAME" 2>&1 || true
    exit 1
  fi
  if (( i % 30 == 0 )); then
    echo "Still waiting... recent logs:"
    docker logs --tail 20 "$CONTAINER_NAME" 2>&1 || true
  fi
  sleep 2
done

echo "Timed out waiting for NIM server. Recent logs:"
docker logs --tail 160 "$CONTAINER_NAME" 2>&1 || true
exit 1
```

## 3. Inspect the running service

`Cosmos3-Generator` exposes standard NIM health/metadata endpoints plus `/v1/version` and a live OpenAPI schema at `/openapi.json`.

```python
for endpoint in ["/v1/models", "/v1/metadata", "/v1/version"]:
    response = requests.get(f"{BASE_URL}{endpoint}", timeout=30)
    response.raise_for_status()
    print(f"\n### {endpoint}")
    print(json.dumps(response.json(), indent=2)[:4000])
```

## 4. Request parameters

Useful constraints for `POST /v1/infer`:

| Field               | Constraint / default                                                                                          |
| ------------------- | ------------------------------------------------------------------------------------------------------------- |
| `guidance_scale`    | float in `[1.0, 7.0]`, default `6.0`                                                                          |
| `steps`             | int in `[1, 100]`, default `35`                                                                               |
| `resolution`        | `256`, `480`, `720`, optionally suffixed with `_16_9`, `_1_1`, `_9_16`, `_4_3`, `_3_4`; bare tiers mean 16:9  |
| `num_output_frames` | must follow the `4k+1` cadence: `25, 29, 33, ...`; per-tier caps are `256 <= 397`, `480 <= 297`, `720 <= 197` |
| `fps`               | float in `[1.0, 60.0]`, recommended `10` to `30`, default `24.0`                                              |

The examples use `resolution="256"` and `num_output_frames=25` so a smoke test completes faster. Increase these values for quality once the workflow is verified.

## 5. Text2Video

For Text2Video, provide a non-empty `prompt` and omit `image`. The response is JSON; decode `response["b64_video"]` to get the MP4 bytes.

The two requests of §§ 5–6 are tabulated (request shape, model, fields, output name) in payloads.md § 8.6, and their output files `<OUT>/cosmos3_generator_nim/cosmos3_generator_nim_t2v.mp4` / `cosmos3_generator_nim_i2v.mp4` in payloads.md § 4.2; the cells below are the notebook's own.

```python
t2v_payload = {
    "prompt": "A humanoid robot walks through a futuristic warehouse, inspecting shelves of mechanical components. Photorealistic, cinematic lighting.",
    "seed": 42,
    "guidance_scale": 6.0,
    "steps": 35,
    "resolution": "256",
    "num_output_frames": 25,
    "fps": 24.0,
}

t2v_path = post_infer(t2v_payload, "cosmos3_generator_nim_t2v.mp4")
print("wrote", t2v_path, t2v_path.stat().st_size, "bytes")
```

## 6. Image2Video

For Image2Video, provide an `image` field. The field accepts raw base64, a `data:image/...;base64,...` URI, or a public URL when URL inputs are enabled. This example uses a checked-in local image encoded as a data URI, so the NIM container does not need to see the host filesystem.

```python
i2v_payload = {
    "prompt": "The humanoid robot performs a controlled standing backflip in a modern living room, then lands steadily on both feet.",
    "image": file_data_uri(I2V_IMAGE),
    "seed": 123,
    "guidance_scale": 6.0,
    "steps": 35,
    "resolution": "256",
    "num_output_frames": 25,
    "fps": 24.0,
}

i2v_path = post_infer(i2v_payload, "cosmos3_generator_nim_i2v.mp4")
print("wrote", i2v_path, i2v_path.stat().st_size, "bytes")
```

## Playback note

The generated `b64_video` payload is an MP4 container. Some environments may not preview VP9-in-MP4 inline. If the notebook preview does not play, verify the written file with a player such as `ffplay`, `mpv`, or IINA, or re-encode externally to H.264.

## 7. Optional cleanup

Stop the detached container when you are done. The cached artifacts stay in `LOCAL_NIM_CACHE` for future launches.

```bash
set -euo pipefail
CONTAINER_NAME="${CONTAINER_NAME:-nvidia-cosmos3-generator}"
docker stop "$CONTAINER_NAME"
```

## Run with NIM

README introduction: see payloads.md § 1.

### Quickstart

Set up the environment: [Generator NIM setup](https://github.com/NVIDIA/cosmos/blob/6e2c69afafd6444c1ca14a021aba93fcc6b7996a/cookbooks/cosmos3/README.md#generator-nim).
`Cosmos3-Generator` NIM is a prebuilt NGC container that serves Text2Video and
Image2Video through `POST /v1/infer`. It returns JSON with a base64-encoded MP4
in `b64_video`; unlike vLLM-Omni, it does not use `/v1/videos/sync` and does not
return MP4 bytes directly.

Authenticate Docker to NGC once:

```bash
export NGC_API_KEY=<your_key>
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

Launch **Cosmos3-Nano** (default model size, FP8, latency profile):

```bash
export LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim}"
mkdir -p "$LOCAL_NIM_CACHE"
chmod -R 777 "$LOCAL_NIM_CACHE" 2>/dev/null || true

docker run -it --rm --name cosmos3-generator \
  --runtime=nvidia \
  --gpus all \
  --shm-size=32GB \
  --ulimit nofile=65536:65536 \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/cosmos3-generator:1.0.0
```

Launch **Cosmos3-Super** by adding `NIM_MODEL_SIZE=super`:

```bash
docker run -it --rm --name cosmos3-generator \
  --runtime=nvidia \
  --gpus all \
  --shm-size=32GB \
  --ulimit nofile=65536:65536 \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -e NIM_MODEL_SIZE=super \
  -e NIM_PRECISION=fp8 \
  -e NIM_PERF_PROFILE=latency \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/cosmos3-generator:1.0.0
```

Wait until the readiness endpoint returns success:

```bash
curl -fsS http://127.0.0.1:8000/v1/health/ready
```

Send a Text2Video request. The NIM infers T2V from a non-empty `prompt` with no
`image` field:

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/infer \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A humanoid robot walks through a futuristic warehouse, inspecting shelves of mechanical components. Photorealistic, cinematic lighting.",
    "seed": 42,
    "guidance_scale": 6.0,
    "steps": 35,
    "resolution": "256",
    "num_output_frames": 25,
    "fps": 24.0
  }' | jq -r '.b64_video' | base64 -d > /tmp/cosmos3_nim_t2v.mp4
```

Send an Image2Video request with a local image encoded as a data URI:

```bash
IMG_B64=$(base64 -w 0 assets/images/image2video/humanoid_robot.jpg)
cat > /tmp/cosmos3_nim_i2v.json <<EOF
{
  "prompt": "The humanoid robot performs a controlled standing backflip in a modern living room, then lands steadily on both feet.",
  "image": "data:image/jpeg;base64,${IMG_B64}",
  "seed": 123,
  "guidance_scale": 6.0,
  "steps": 35,
  "resolution": "256",
  "num_output_frames": 25,
  "fps": 24.0
}
EOF

curl -sS -X POST http://127.0.0.1:8000/v1/infer \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @/tmp/cosmos3_nim_i2v.json | jq -r '.b64_video' | base64 -d > /tmp/cosmos3_nim_i2v.mp4
```

Key request fields and constraints:

| Field               | Constraint / default                                                                                       |
| ------------------- | ---------------------------------------------------------------------------------------------------------- |
| `prompt`            | Required for T2V; optional when `image` is provided; max 20000 chars                                       |
| `negative_prompt`   | Optional; omitted means the server uses the Cosmos3 default                                                |
| `image`             | I2V conditioning image; raw base64, `data:image/...;base64,...`, or public URL when URL inputs are enabled |
| `guidance_scale`    | `1.0` to `7.0`, default `6.0`                                                                              |
| `steps`             | `1` to `100`, default `35`                                                                                 |
| `resolution`        | `256`, `480`, `720`, optionally with `_16_9`, `_1_1`, `_9_16`, `_4_3`, or `_3_4`                           |
| `num_output_frames` | Must follow the 4k+1 cadence (`25, 29, 33, ...`); caps: `256 <= 397`, `480 <= 297`, `720 <= 197`           |
| `fps`               | `1.0` to `60.0`, recommended `10` to `30`, default `24.0`                                                  |

### Notebook walkthrough

[`run_with_nim.ipynb`](https://github.com/NVIDIA/cosmos/blob/6e2c69afafd6444c1ca14a021aba93fcc6b7996a/cookbooks/cosmos3/generator/audiovisual/run_with_nim.ipynb) launches the NIM container,
waits for readiness, inspects the service endpoints, sends T2V and I2V requests,
decodes `b64_video`, and previews the generated MP4 files inline.

### Limitations

`Cosmos3-Generator` NIM currently exposes **Text2Video** and **Image2Video** only.
It does **not** expose text-to-image, video-to-video, sound/audio generation,
action modes, or transfer controls. For those broader Generator API workflows,
use vLLM-Omni or Cosmos Framework as appropriate.
