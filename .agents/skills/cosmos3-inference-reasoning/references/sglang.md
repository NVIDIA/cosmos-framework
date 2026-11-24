<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Reasoner inference with SGLang

This notebook serves the Cosmos3 Reasoner with an OpenAI-compatible **SGLang** server,
using the prebuilt `lmsysorg/sglang:dev` Docker image. It:

1. Pulls the SGLang Docker image and installs the client dependencies.
2. Launches an OpenAI-compatible server for the model tier you choose.
3. Sends image and video reasoning requests with the `openai` client.

The notebook can serve `nvidia/Cosmos3-Edge` on one GPU or `nvidia/Cosmos3-Super` across
4 GPUs with tensor parallelism. The query examples in the final section resolve the served
model dynamically.

> **Skill note (not in the source):** this file is the cookbook notebook `run_with_sglang.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/reasoner/run_with_sglang.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

## 1. Setup

Serving runs entirely in Docker, so no local Python environment is required for the model.
You need:

- Docker with the NVIDIA Container Toolkit (`--runtime nvidia` / `--gpus`).

The Cosmos3-Super Reasoner checkpoint is gated on Hugging Face. If you choose Super,
authenticate with a token that has access after running the setup cells below. Both launch
cells mount `~/.cache/huggingface` into the container so downloaded weights are reused:

```bash
hf auth login
```

> **Skill note (not in the source):** from this skill directory `find_repo_root()` raises `FileNotFoundError` (no `cookbooks/cosmos3` tree here); replace the assignments that call it with `Path.cwd()` and `Path.cwd() / "assets"` (SKILL.md Read this first 3).

```python
from pathlib import Path
import os
import subprocess


def find_repo_root() -> Path:
    start = Path.cwd().resolve()
    for path in [start, *start.parents]:
        if (path / "cookbooks" / "cosmos3").is_dir():
            return path
    return start


COSMOS_ROOT = find_repo_root()
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
# Pull the prebuilt SGLang image with native Cosmos3 Reasoner support (~14 GB; first run only).
docker pull lmsysorg/sglang:dev
```

## 2. Launch a SGLang server

Choose either the Edge or Super tier below. The commands are mutually exclusive: both use
the container name `cosmos3-reasoner-sglang` and publish the container's port 8000 to host
port **8001**, which matches the endpoint the query cells use.

The first start downloads the weights and compiles CUDA graphs and can take several
minutes; the readiness cell below polls `/health` and waits automatically.

### Cosmos3-Edge inference with SGLang

`Cosmos3-Edge` is the smaller tier and runs on a single GPU. Run this cell to launch the server.

```bash
set -euo pipefail
: "${COSMOS_ROOT:?run the setup cell first}"
: "${COSMOS3_MEDIA_ROOT:?run the setup cell first}"
CONTAINER=cosmos3-reasoner-sglang

# setup hf cache dir
export SGLANG_HF_CACHE="${SGLANG_HF_CACHE:-$HOME/.cache/sglang-huggingface}"
mkdir -p "$SGLANG_HF_CACHE"
chmod 777 "$SGLANG_HF_CACHE"

# Remove any existing server first.
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" \
  --runtime nvidia --gpus '"device=0"' \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$SGLANG_HF_CACHE:/root/.cache/huggingface" \
  -v "$COSMOS_ROOT:$COSMOS_ROOT" \
  -p 8001:8000 --ipc=host \
  lmsysorg/sglang:dev \
  sglang serve \
    --model-type llm \
    --model-path nvidia/Cosmos3-Edge \
    --tp-size 1 \
    --enable-multimodal \
    --host 0.0.0.0 \
    --port 8000
echo "Cosmos3-Edge starting in container '$CONTAINER' — run the readiness cell next."
```

### Cosmos3-Super inference with SGLang

`Cosmos3-Super` is the larger tier, served across **4 GPUs** with `--tensor-parallel-size 4`.
Run this cell to launch the server.

```bash
set -euo pipefail
: "${COSMOS_ROOT:?run the setup cell first}"
: "${COSMOS3_MEDIA_ROOT:?run the setup cell first}"
CONTAINER=cosmos3-reasoner-sglang

# setup hf cache dir
export SGLANG_HF_CACHE="${SGLANG_HF_CACHE:-$HOME/.cache/sglang-huggingface}"
mkdir -p "$SGLANG_HF_CACHE"
chmod 777 "$SGLANG_HF_CACHE"

# Remove any existing server first.
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" \
  --runtime nvidia --gpus all \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$SGLANG_HF_CACHE:/root/.cache/huggingface" \
  -v "$COSMOS_ROOT:$COSMOS_ROOT" \
  -p 8001:8000 --ipc=host \
  lmsysorg/sglang:dev \
  sglang serve \
    --model-type llm \
    --model-path nvidia/Cosmos3-Super \
    --tp-size 4 \
    --enable-multimodal \
    --host 0.0.0.0 \
    --port 8000
echo "Cosmos3-Super starting in container '$CONTAINER' — run the readiness cell next."
```

### Wait for the server to be ready

Run this after launching either tier. It streams the container log and polls `/health`
until the server is ready, allowing up to 30 minutes on the first run while the weights
download and CUDA graphs compile.

```bash
set -uo pipefail
PORT="${SGLANG_PORT:-8001}"
CONTAINER=cosmos3-reasoner-sglang

echo "Streaming server logs; waiting for http://127.0.0.1:${PORT}/health ..."
docker logs -f "$CONTAINER" 2>&1 &
LOGPID=$!

for i in $(seq 1 1800); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo; echo "SGLang server is ready."
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
echo "Timed out waiting for SGLang server."
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
!docker rm -f cosmos3-reasoner-sglang
```
