<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Reasoner inference with vLLM

This notebook serves the Cosmos3 Reasoner with an OpenAI-compatible **vLLM** server,
using the prebuilt `vllm/vllm-openai:cosmos3` Docker image. It:

1. Pulls the vLLM Docker image and installs the client dependencies.
2. Launches an OpenAI-compatible server for the model tier you choose.
3. Sends image and video reasoning requests with the `openai` client.

The notebook can serve `nvidia/Cosmos3-Edge` on one GPU or `nvidia/Cosmos3-Super` across
4 GPUs with tensor parallelism. The query examples in the final section resolve the served
model dynamically.

> **Skill note (not in the source):** this file is the cookbook notebook `run_with_vllm.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/reasoner/run_with_vllm.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

## 1. Setup

Serving runs entirely in Docker, so no local Python environment is required for the model.
You need:

- Docker with the NVIDIA Container Toolkit (`--runtime nvidia` / `--gpus`).
- The `vllm/vllm-openai:cosmos3` image, which ships native Cosmos3 Reasoner support
  (vLLM ≥ 0.23.0). A cell below pulls it (~19 GB on first run).

The Cosmos3-Super Reasoner checkpoint is gated on Hugging Face. If you choose Super,
authenticate with a token that has access after running the setup cells below. Both launch
cells mount `~/.cache/huggingface` into the container so downloaded weights are reused:

```bash
hf auth login
```

> **Skill note (not in the source):** from this skill directory the `COSMOS_REASONER_ASSETS` assertion below fails (no `cookbooks/cosmos3` tree here); set `COSMOS_ROOT = Path.cwd()` and `COSMOS_REASONER_ASSETS = COSMOS_ROOT / "assets"` instead (SKILL.md Read this first 3).

```python
from pathlib import Path
import os
import subprocess


def find_framework_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").is_file() and (path / "cosmos_framework").is_dir():
            return path
    raise FileNotFoundError("Launch from the Framework checkout or a notebook directory inside it.")


FRAMEWORK_ROOT = find_framework_root(Path.cwd().resolve())
COSMOS_ROOT = FRAMEWORK_ROOT / ".agents" / "skills"
COSMOS_REASONER_ASSETS = COSMOS_ROOT / "cookbooks" / "cosmos3" / "reasoner" / "assets"
COSMOS3_MEDIA_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3"

assert COSMOS_REASONER_ASSETS.exists(), COSMOS_REASONER_ASSETS

os.environ["COSMOS_ROOT"] = str(COSMOS_ROOT)
os.environ["COSMOS_REASONER_ASSETS"] = str(COSMOS_REASONER_ASSETS)
os.environ["COSMOS3_MEDIA_ROOT"] = str(COSMOS3_MEDIA_ROOT)


def asset_path(name: str) -> Path:
    path = COSMOS_REASONER_ASSETS / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def asset_url(name: str) -> str:
    return asset_path(name).resolve().as_uri()


print("cosmos root:", COSMOS_ROOT)
print("Reasoner assets:", COSMOS_REASONER_ASSETS)
print("Allowed media root:", COSMOS3_MEDIA_ROOT)
```

```python
# Client-side dependencies for this notebook (not the model): the OpenAI client to
# send requests, and the Hugging Face CLI used by the `hf auth login` step above.
%pip install -q "openai>=1.0" pillow "huggingface_hub[cli]"
```

```bash
set -euo pipefail
# Pull the prebuilt vLLM image with native Cosmos3 Reasoner support (~19 GB; first run only).
docker pull vllm/vllm-openai:cosmos3
```

## 2. Launch a vLLM server

Choose either the Edge or Super tier below. The commands are mutually exclusive: both use
the container name `cosmos3-reasoner-vllm` and publish the container's port 8000 to host
port **8001** on the loopback interface (`127.0.0.1`) only, which matches the endpoint the
query cells use. The server has no API authentication configured; the loopback binding
keeps it unreachable from other machines. To serve remote clients, additional measures
are required, for example `--api-key` and network-level access controls.

The first start downloads the weights and compiles CUDA graphs and can take several
minutes; the readiness cell below polls `/health` and waits automatically.

### Cosmos3-Edge inference with vLLM

`Cosmos3-Edge` is the smaller tier and runs on a single GPU. Run this cell to launch the server.

```bash
set -euo pipefail
: "${COSMOS_ROOT:?run the setup cell first}"
: "${COSMOS3_MEDIA_ROOT:?run the setup cell first}"
CONTAINER=cosmos3-reasoner-vllm

# Remove any existing server first.
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" \
  --runtime nvidia --gpus '"device=0"' \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$COSMOS_ROOT:$COSMOS_ROOT" \
  -p 127.0.0.1:8001:8000 --ipc=host \
  vllm/vllm-openai:cosmos3 \
  nvidia/Cosmos3-Edge \
    --async-scheduling \
    --allowed-local-media-path "$COSMOS3_MEDIA_ROOT" \
    --port 8000
echo "Cosmos3-Edge starting in container '$CONTAINER' — run the readiness cell next."
```

### Cosmos3-Super inference with vLLM

`Cosmos3-Super` is the larger tier, served across **4 GPUs** with `--tensor-parallel-size 4`.
Run this cell to launch the server.

```bash
set -euo pipefail
: "${COSMOS_ROOT:?run the setup cell first}"
: "${COSMOS3_MEDIA_ROOT:?run the setup cell first}"
CONTAINER=cosmos3-reasoner-vllm

# Remove any existing server first.
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" \
  --runtime nvidia --gpus all \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$COSMOS_ROOT:$COSMOS_ROOT" \
  -p 127.0.0.1:8001:8000 --ipc=host \
  vllm/vllm-openai:cosmos3 \
  nvidia/Cosmos3-Super \
    --tensor-parallel-size 4 \
    --mm-encoder-tp-mode data \
    --async-scheduling \
    --allowed-local-media-path "$COSMOS3_MEDIA_ROOT" \
    --port 8000
echo "Cosmos3-Super starting in container '$CONTAINER' — run the readiness cell next."
```

### Quantized Checkpoints

Standalone vLLM uses ModelOpt FP8 for reasoner inference; the generator's
diffusion-step activation policy does not apply. Download a published Edge or Super
FP8 checkpoint to a local directory, or set `FP8_MODEL_DIR` to the local
`edge-fp8` or `super-fp8` output from the [quantization cookbook](../../cosmos3-quantization/SKILL.md)
and skip the download. Choose one tier below. The launches preserve the original
served model ID. Edge FP8 uses `vllm/vllm-openai:nightly`; Super FP8 uses
`vllm/vllm-openai:v0.31.0`. Validate the image on your runtime before relying on it
in production.

The launch adds `--trust-request-mm-kwargs`, which lets each request override media
settings such as `num_frames` and `fps`; the video examples below need it. Enable it only
for trusted clients, because per-request overrides can increase CPU, GPU, and memory use.
Keep the loopback-only port binding, and add access controls (for example `--api-key`)
before exposing the server to remote clients.

For locally generated outputs, set `FP8_SOURCE_DIR` to the source checkpoint
directory (or a common parent containing snapshot and backing blobs) so exported
symlinks remain readable. Mount any additional symlink target caches at their
original absolute paths. Keep `HF_HOME` consistent with quantization.

#### Cosmos3-Edge FP8

The server binds to localhost and trusts the notebook client's per-request video
loading settings (`media_io_kwargs`).

```python
from huggingface_hub import snapshot_download

FP8_MODEL_DIR = Path(os.environ.get("FP8_MODEL_DIR", FRAMEWORK_ROOT / "models" / "cosmos3-edge-fp8")).expanduser().resolve()
os.environ["FP8_MODEL_DIR"] = str(FP8_MODEL_DIR)
# Skip the download for a locally generated output; export FP8_MODEL_DIR before setup.
if not (FP8_MODEL_DIR / "hf_quant_config.json").is_file():
    snapshot_download("nvidia/Cosmos3-Edge-FP8", local_dir=str(FP8_MODEL_DIR))
```

then serve that directory:

```bash
set -euo pipefail
: "${COSMOS_ROOT:?run the setup cell first}"
: "${COSMOS3_MEDIA_ROOT:?run the setup cell first}"

HF_HOME="$(realpath -m "${HF_HOME:-$HOME/.cache/huggingface}")"
: "${FP8_MODEL_DIR:?run the FP8 configuration/download cell first}"
FP8_MODEL_DIR="$(realpath "$FP8_MODEL_DIR")"
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
add_mount "$COSMOS_ROOT" "$COSMOS_ROOT"
add_mount "$FP8_MODEL_DIR" "$FP8_MODEL_DIR" ro
if [ -n "${FP8_SOURCE_DIR:-}" ]; then
  FP8_SOURCE_DIR="$(realpath "$FP8_SOURCE_DIR")"
  add_mount "$FP8_SOURCE_DIR" "$FP8_SOURCE_DIR" ro
fi
CONTAINER=cosmos3-reasoner-vllm
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --rm --name "$CONTAINER" \
  --runtime nvidia --gpus all \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN \
  "${MOUNTS[@]}" \
  -w "$COSMOS_ROOT" \
  -p 127.0.0.1:8001:8000 --ipc=host \
  vllm/vllm-openai:nightly \
  "$FP8_MODEL_DIR" \
    --served-model-name nvidia/Cosmos3-Edge \
    --async-scheduling \
    --trust-request-mm-kwargs \
    --allowed-local-media-path "$COSMOS3_MEDIA_ROOT" \
    --port 8000
```

#### Cosmos3-Super FP8

```python
from huggingface_hub import snapshot_download

FP8_MODEL_DIR = Path(os.environ.get("FP8_MODEL_DIR", FRAMEWORK_ROOT / "models" / "cosmos3-super-fp8")).expanduser().resolve()
os.environ["FP8_MODEL_DIR"] = str(FP8_MODEL_DIR)
# Skip the download for a locally generated output; export FP8_MODEL_DIR before setup.
if not (FP8_MODEL_DIR / "hf_quant_config.json").is_file():
    snapshot_download("nvidia/Cosmos3-Super-FP8", local_dir=str(FP8_MODEL_DIR))
```

then serve that directory:

```bash
set -euo pipefail
: "${COSMOS_ROOT:?run the setup cell first}"
: "${COSMOS3_MEDIA_ROOT:?run the setup cell first}"

HF_HOME="$(realpath -m "${HF_HOME:-$HOME/.cache/huggingface}")"
: "${FP8_MODEL_DIR:?run the FP8 configuration/download cell first}"
FP8_MODEL_DIR="$(realpath "$FP8_MODEL_DIR")"
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
add_mount "$COSMOS_ROOT" "$COSMOS_ROOT"
add_mount "$FP8_MODEL_DIR" "$FP8_MODEL_DIR" ro
if [ -n "${FP8_SOURCE_DIR:-}" ]; then
  FP8_SOURCE_DIR="$(realpath "$FP8_SOURCE_DIR")"
  add_mount "$FP8_SOURCE_DIR" "$FP8_SOURCE_DIR" ro
fi
CONTAINER=cosmos3-reasoner-vllm
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" \
  --runtime nvidia --gpus all \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN \
  "${MOUNTS[@]}" \
  -w "$COSMOS_ROOT" \
  -p 127.0.0.1:8001:8000 --ipc=host \
  vllm/vllm-openai:v0.31.0 \
  "$FP8_MODEL_DIR" \
    --served-model-name nvidia/Cosmos3-Super \
    --tensor-parallel-size 4 \
    --mm-encoder-tp-mode data \
    --async-scheduling \
    --trust-request-mm-kwargs \
    --allowed-local-media-path "$COSMOS3_MEDIA_ROOT" \
    --port 8000
```

### Wait for the server to be ready

Run this after launching either tier. It streams the container log and polls `/health`
until the server is ready, allowing up to 30 minutes on the first run while the weights
download and CUDA graphs compile.

```bash
set -uo pipefail
PORT="${VLLM_PORT:-8001}"
CONTAINER=cosmos3-reasoner-vllm

echo "Streaming server logs; waiting for http://127.0.0.1:${PORT}/health ..."
docker logs -f "$CONTAINER" 2>&1 &
LOGPID=$!

for i in $(seq 1 1800); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo; echo "vLLM server is ready."
    kill "$LOGPID" 2>/dev/null || true
    exit 0
  fi
  if ! docker ps -q -f "name=^${CONTAINER}$" | grep -q .; then
    echo; echo "Container exited early (see logs above)."
    kill "$LOGPID" 2>/dev/null || true
    exit 1
  fi
  sleep 2
done

kill "$LOGPID" 2>/dev/null || true
echo "Timed out waiting for vLLM server."
exit 1
```

## 3. Query the model

### Image Caption

```python
import openai
from IPython.display import Image, display

client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")
MODEL = client.models.list().data[0].id

image_path = asset_path("robot_153.jpg")
image_url = image_path.resolve().as_uri()

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Caption the image in detail."},
            ],
        }
    ],
    max_tokens=4096,
    seed=0,
)
display(Image(filename=str(image_path), width=512))
print(response.choices[0].message.content)
```

### Video Caption

```python
import openai
from pathlib import Path
from IPython.display import Video, display

prompt = "Describe the video in detail."

# Plain filesystem path (used for display)
video_path = str(asset_path("video_caption.mp4"))
# file:// URL (used for the model request)
video_url = Path(video_path).resolve().as_uri()

client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": prompt},
            ],
        },
    ],
    max_tokens=4096,
    extra_body={"media_io_kwargs": {"video": {"num_frames": -1, "fps": 4}}},
)

# Display the input video (plain path, NOT the file:// URL)
display(Video(video_path, embed=True, width=640))

print(response.choices[0].message.content)
```

### Temporal Localization

````python
from pathlib import Path
from IPython.display import Video, display

# Plain filesystem path (used for display)
video_path = str(asset_path("temporal_localization_1.mp4"))

display(Video(video_path, embed=True, width=640))

import openai

prompt = (
    """List all action segments in the video. For each detected event, you must determine:

Provide the result in json format with 'seconds' for time depiction for each event. Use keywords 'start', 'end' and 'caption' in the json output. Please list multiple events if applicable.

```json
[
{
  "start": t_start,
  "end": t_end,
  "caption": EVENT1
},
{
  "start": t_start,
  "end": t_end,
  "caption": EVENT2
},
...
]
``` """
)
video_url = asset_url("temporal_localization_1.mp4")

client = openai.OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8001/v1",
)

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": prompt},
            ],
        },
    ],
    max_tokens=4096,
    extra_body={"media_io_kwargs": {"video": {"num_frames": -1, "fps": 4}}},
)

print(response.choices[0].message.content)
````

```python
from pathlib import Path
from IPython.display import Video, display

# Plain filesystem path (used for display)
video_path = str(asset_path("temporal_localization_2.mp4"))

display(Video(video_path, embed=True, width=640))
```

#### Event Timeline

```python
import openai

prompt = (
    "Describe the notable events in the provided video. Provide the result in json format with 'mm:ss.ff' format for time depiction for each event."
    "Use keywords 'start', 'end' and 'caption' in the json output."
)
video_url = asset_url("temporal_localization_2.mp4")

client = openai.OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8001/v1",
)

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": prompt},
            ],
        },
    ],
    max_tokens=4096,
    extra_body={"media_io_kwargs": {"video": {"num_frames": -1, "fps": 4}}},
)

print(response.choices[0].message.content)
```

#### Timestamp Query

```python
import openai

prompt = """When is "A man in a white sweater walks out of a room carrying a box, closes the door behind him, walks on the floor, and turns left at the end near the wall." depicted in the video? Please provide the result in json format with 'mm:ss.ff' format for time depiction for the event. Use keywords 'start', 'end' in the json output."""

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": prompt},
            ],
        },
    ],
    max_tokens=4096,
    extra_body={"media_io_kwargs": {"video": {"num_frames": -1, "fps": 4}}},
)

print(response.choices[0].message.content)
```

#### Interval Question

```python
import openai

prompt = "What happened between 00:05.64 and 00:17.49?"
response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": prompt},
            ],
        },
    ],
    max_tokens=4096,
    extra_body={"media_io_kwargs": {"video": {"num_frames": -1, "fps": 4}}},
)

print(response.choices[0].message.content)
```

### Embodied Reasoning

#### Robotics Next Action

```python
import openai
from pathlib import Path
from IPython.display import Video, display

prompt = "What can be the next immediate action? Answer the question using the following format: <think> Your reasoning. </think> Write your final answer immediately after the </think> tag."

# Plain filesystem path (used for display)
video_path = str(asset_path("robotics_next_action.mp4"))
# file:// URL (used for the model request)
video_url = Path(video_path).resolve().as_uri()

client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": prompt},
            ],
        },
    ],
    max_tokens=4096,
    extra_body={"media_io_kwargs": {"video": {"num_frames": -1, "fps": 4}}},
)

# Display the input video (plain path, NOT the file:// URL)
display(Video(video_path, embed=True, width=640))

print(response.choices[0].message.content)
```

#### Drive Scene Next Action

```python
import openai
from pathlib import Path
from IPython.display import Video, display

prompt = "You are an autonomous vehicle planning system. The video depicts the observation from the vehicle's camera. You need to observe the critical objects in the environment and reason your next action and the driving trajectory ahead."

# Plain filesystem path (used for display)
video_path = str(asset_path("drive_scene_next_action.mp4"))
# file:// URL (used for the model request)
video_url = Path(video_path).resolve().as_uri()

client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": prompt},
            ],
        },
    ],
    max_tokens=4096,
    extra_body={"media_io_kwargs": {"video": {"num_frames": -1, "fps": 4}}},
)

# Display the input video (plain path, NOT the file:// URL)
display(Video(video_path, embed=True, width=640))

print(response.choices[0].message.content)
```

#### Robot Planning

```python
import json
import re
import openai
from pathlib import Path
from PIL import Image as PILImage, ImageDraw
from IPython.display import display

client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")
MODEL = client.models.list().data[0].id

image_path = str(asset_path("robot_planning.png"))
image_url = Path(image_path).resolve().as_uri()   # file:// URL for the model

# Display the input image (scaled down to fit the cell)
preview = PILImage.open(image_path).convert("RGB")
preview.thumbnail((768, 768))
display(preview)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": 'The task is to put flower into the red bottle. Generate a plan consisting of subtasks for accomplish the task.'},
            ],
        }
    ],
    max_tokens=4096,
    seed=0,
)
print(response.choices[0].message.content)
```

#### Assisted Task Next Action

```python
import openai
from pathlib import Path
from IPython.display import Video, display

prompt = """This is the overall task that the agent is trying to complete: "The student exchanges the black ink cartridge of the printer."
            In the video, the agent is trying to follow the instruction (a single step out of many to complete the overall task): "place old ink_cartridge."
            What should be the next action of the agent?
            Answer the question using the following format:
            <think>
            Your reasoning.
            </think>
            Write your final answer immediately after the </think> tag."""

# Plain filesystem path (used for display)
video_path = str(asset_path("assisted_task_next_action.mp4"))
# file:// URL (used for the model request)
video_url = Path(video_path).resolve().as_uri()

client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": prompt},
            ],
        },
    ],
    max_tokens=4096,
    extra_body={"media_io_kwargs": {"video": {"num_frames": -1, "fps": 4}}},
)

# Display the input video (plain path, NOT the file:// URL)
display(Video(video_path, embed=True, width=640))

print(response.choices[0].message.content)
```

### Common Sense Reasoning

```python
import openai
from pathlib import Path
from IPython.display import Video, display

prompt = """Can the countertop support the weight of the juicers?
            Answer the question using the following format:

            <think>
            Your reasoning.
            </think>

            Write your final answer immediately after the </think> tag."""

# Plain filesystem path (used for display)
video_path = str(asset_path("common_sense_reasoning.mp4"))
# file:// URL (used for the model request)
video_url = Path(video_path).resolve().as_uri()

client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": prompt},
            ],
        },
    ],
    max_tokens=4096,
    extra_body={"media_io_kwargs": {"video": {"num_frames": -1, "fps": 4}}},
)

# Display the input video (plain path, NOT the file:// URL)
display(Video(video_path, embed=True, width=640))

print(response.choices[0].message.content)
```

### 2D Grounding

````python
import json
import re
import openai
from pathlib import Path
from PIL import Image as PILImage, ImageDraw
from IPython.display import display

client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")
MODEL = client.models.list().data[0].id

image_path = str(asset_path("grounding_2d.png"))
image_url = Path(image_path).resolve().as_uri()   # file:// URL for the model

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Locate the accurate bounding box of the load as a whole. Return a json."},
            ],
        }
    ],
    max_tokens=4096,
    seed=0,
)
out = response.choices[0].message.content
print(out)


def parse_boxes(text):
    """Extract JSON box objects from the model text.

    The Reasoner replies in free-form text, so this tolerates a ``<think>`` block,
    ``` fences, and any prose after the JSON (which varies between runs).
    """
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    starts = [i for i in (text.find("["), text.find("{")) if i != -1]
    if not starts:
        return []
    data, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    return data if isinstance(data, list) else [data]


def box_coords(obj):
    """Return ``[x1, y1, x2, y2]`` from the varied shapes the model emits."""
    vals = obj if isinstance(obj, (list, tuple)) else (
        obj.get("bbox_2d") or obj.get("bbox") or obj.get("box") or obj.get("bounding_box")
    )
    if vals is None and isinstance(obj, dict) and all(k in obj for k in ("x1", "y1", "x2")):
        vals = [obj["x1"], obj["y1"], obj["x2"], obj.get("y2", obj.get("y3"))]
    if not vals or len(vals) < 4:
        return None
    try:
        return [float(v) for v in vals[:4]]
    except (TypeError, ValueError):
        return None


# Draw boxes; coords are normalized to 0-1000
img = PILImage.open(image_path).convert("RGB")
W, H = img.size
draw = ImageDraw.Draw(img)

for obj in parse_boxes(out):
    box = box_coords(obj)
    if not box:
        continue
    x1, y1, x2, y2 = box
    x1, x2 = x1 / 1000 * W, x2 / 1000 * W
    y1, y2 = y1 / 1000 * H, y2 / 1000 * H
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    label = (obj.get("label") or obj.get("name")) if isinstance(obj, dict) else None
    if label:
        draw.text((x1, max(0, y1 - 12)), str(label), fill="red")

# Display scaled down so a large image fits the cell
preview = img.copy()
preview.thumbnail((768, 768))
display(preview)
````

### Describe Anything

```python
import json
import re
import openai
from pathlib import Path
from PIL import Image as PILImage, ImageDraw
from IPython.display import display

client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")
MODEL = client.models.list().data[0].id

image_path = str(asset_path("describe_anything.png"))
image_url = Path(image_path).resolve().as_uri()   # file:// URL for the model

# Display the input image (scaled down to fit the cell)
preview = PILImage.open(image_path).convert("RGB")
preview.thumbnail((768, 768))
display(preview)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": 'Please caption the notable attributes in the provided image. List and describe all marked subjects in the image with their categories and detailed captions using a json with keyword "subject_id", "category" and "caption".'},
            ],
        }
    ],
    max_tokens=4096,
    seed=0,
)
print(response.choices[0].message.content)
```

### Action CoT

#### Trajectory Coordinates

````python
import json
import re
import openai
from pathlib import Path
from PIL import Image as PILImage, ImageDraw
from IPython.display import display

client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")
MODEL = client.models.list().data[0].id

image_path = str(asset_path("action_cot_trajectory.png"))
image_url = Path(image_path).resolve().as_uri()

prompt = """You are given the task "Move the pink bowl to the right". Specify the 2D trajectory your end effector should follow in pixel space. Return the trajectory coordinates in JSON format like this: {"point_2d": [x, y], "label": "gripper trajectory"}.
Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag.
"""

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ],
    max_tokens=4096,
    temperature=0.6,
    top_p=0.95,
    presence_penalty=0.0,
    extra_body={"top_k": 20, "repetition_penalty": 1.0},
)
out = response.choices[0].message.content
print(out)


def parse_points(text):
    """Grab the JSON list of {point_2d, label} after the </think> tag."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    data = json.loads(m.group(0) if m else text)
    return data if isinstance(data, list) else [data]


# Visualize the trajectory (points are in pixel space)
img = PILImage.open(image_path).convert("RGB")
draw = ImageDraw.Draw(img)
W, H = img.size

# coords are normalized to 0-1000 (per-axis) -> scale to pixels
pts = [(o["point_2d"][0] / 1000 * W, o["point_2d"][1] / 1000 * H)
       for o in parse_points(out) if isinstance(o, dict) and "point_2d" in o]
if len(pts) > 1:
    draw.line(pts, fill="lime", width=5)
for i, (x, y) in enumerate(pts):
    r = 12
    draw.ellipse([x - r, y - r, x + r, y + r], fill="red", outline="white", width=3)
    draw.text((x + 14, y - 14), str(i), fill="yellow")
preview = img.copy()
preview.thumbnail((900, 900))
display(preview)
````

````python
import json
import re
import openai
from pathlib import Path
from PIL import Image as PILImage, ImageDraw
from IPython.display import display

client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")
MODEL = client.models.list().data[0].id

image_path = str(asset_path("robot_planning.png"))
image_url = Path(image_path).resolve().as_uri()

prompt = """You are given the task "Put flower into the red bottle". Specify the 2D trajectory your end effector should follow in pixel space. Return the trajectory coordinates in JSON format like this: {"point_2d": [x, y], "label": "gripper trajectory"}.
Answer the question using the following format:

<think> Your reasoning. </think>
Write your final answer immediately after the </think> tag.
"""

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ],
    max_tokens=4096,
    temperature=0.6,
    top_p=0.95,
    presence_penalty=0.0,
    extra_body={"top_k": 20, "repetition_penalty": 1.0},
)
out = response.choices[0].message.content
print(out)


def parse_points(text):
    """Grab the JSON list of {point_2d, label} after the </think> tag."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    data = json.loads(m.group(0) if m else text)
    return data if isinstance(data, list) else [data]


# Visualize the trajectory (points are in pixel space)
img = PILImage.open(image_path).convert("RGB")
draw = ImageDraw.Draw(img)
W, H = img.size

# coords are normalized to 0-1000 (per-axis) -> scale to pixels
pts = [(o["point_2d"][0] / 1000 * W, o["point_2d"][1] / 1000 * H)
       for o in parse_points(out) if isinstance(o, dict) and "point_2d" in o]
if len(pts) > 1:
    draw.line(pts, fill="lime", width=5)
for i, (x, y) in enumerate(pts):
    r = 12
    draw.ellipse([x - r, y - r, x + r, y + r], fill="red", outline="white", width=3)
    draw.text((x + 14, y - 14), str(i), fill="yellow")
preview = img.copy()
preview.thumbnail((900, 900))
display(preview)
````

#### Driving Scene

```python
import openai
from pathlib import Path
from IPython.display import Video, display
client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")
MODEL = client.models.list().data[0].id
video_path = str(asset_path("action_cot_driving_scene.mp4"))
video_url = Path(video_path).resolve().as_uri()
prompt = """The video depicts the observation from the vehicle's camera. You need to think step by step and identify the objects in the scene that are critical for safe navigation.
Answer the question using the following format:
<think>
Your reasoning.
</think>
Write your final answer immediately after the </think> tag."""
# Show the input video
display(Video(video_path, embed=True, width=640))
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ],
    max_tokens=4096,
    temperature=0.6,
    top_p=0.95,
    presence_penalty=0.0,
    extra_body={"media_io_kwargs": {"video": {"num_frames": -1, "fps": 4}}, "top_k": 20, "repetition_penalty": 1.0},
)
print(response.choices[0].message.content)
```

### Physical Plausibility Analysis

```python
import openai
from pathlib import Path
from IPython.display import Video, display
client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")
MODEL = client.models.list().data[0].id
video_path = str(asset_path("physical_plausibility.mp4"))
video_url = Path(video_path).resolve().as_uri()
prompt = """Is this video physically plausible/possible according to your understanding of e.g. object permanence, shape constancy (objects maintain shape over time), continuous trajectories of objects? Assume it is the normal laws of physics.
Your answer should be based on the events in the video and ignore the quality of the simulation engine. The rising wall is part of the experiment setup and should not be judged for plausibility.
(A) Possible
(B) Impossible"""
# Show the input video
display(Video(video_path, embed=True, width=640))
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ],
    max_tokens=4096,
    extra_body={"media_io_kwargs": {"video": {"num_frames": -1, "fps": 4}}},
)
print(response.choices[0].message.content)
```

### Situation Understanding

```python
import openai
from pathlib import Path
from IPython.display import Video, display
client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")
MODEL = client.models.list().data[0].id
video_path = str(asset_path("situation_understanding.mp4"))
video_url = Path(video_path).resolve().as_uri()
prompt = "What is the person doing with the skillet? What will the person likely do next in this situation?"
# Show the input video
display(Video(video_path, embed=True, width=640))
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ],
    max_tokens=4096,
    extra_body={"media_io_kwargs": {"video": {"num_frames": -1, "fps": 4}}},
)
print(response.choices[0].message.content)
```

## 4. Shut down the server

When you are finished, stop and remove the container to free the GPUs and host port.

```python
!docker rm -f cosmos3-reasoner-vllm
```
