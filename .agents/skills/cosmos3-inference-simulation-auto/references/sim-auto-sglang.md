<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3-Nano-Sim-Auto with SGLang

Cosmos3-Nano-Sim-Auto generates **synchronized multi-camera driving video** from one World Scenario Model (WSM) control video and one RGB first-frame condition per camera. The WSM controls render the scene's lane lines, road surface, and vehicle boxes from each camera's viewpoint; the RGB frame sets each camera's starting appearance; a caption per camera describes the scene. It is the causal, distilled student of Cosmos3-Nano-Transfer-Auto: it generates every camera in 0.4-second chunks with a rolling key/value cache and a fixed **four-step** sampler at guidance 1.

This notebook runs the RGB example of the shared **daytime sample**, one scene recorded by seven cameras, through one SGLang server at **297 frames per camera, 30 FPS, 832 × 480, and seed 0** on one GPU. SGLang reads the inputs from paths the server can open and runs the rollout as one asynchronous Videos API job.

| Component                      | Version                                                                                                                                                                                                                                                                         |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SGLang                         | The published [`lmsysorg/sglang:dev`](https://hub.docker.com/r/lmsysorg/sglang/tags) image, which carries the Cosmos3 Sim-Auto pipeline from [sgl-project/sglang#39475](https://github.com/sgl-project/sglang/pull/39475); the tag moves, so pull it before starting the server |
| `nvidia/Cosmos3-Nano-Sim-Auto` | [`main`](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto/tree/main)                                                                                                                                                                                                         |
| Daytime sample                 | [`main`](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto/tree/main/assets/sample_data) of `nvidia/Cosmos3-Nano-Transfer-Auto`, under `assets/sample_data`                                                                                                              |

Steps 1 and 2 run on the server host. Steps 3 to 6 run in this notebook.

> **Skill note (not in the source):** this file is the cookbook notebook `run_sim-auto_with_sglang.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/simulation/auto/run_sim-auto_with_sglang.ipynb`, cosmos-framework `main` 45fecab) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced as `python`, outputs omitted (the committed notebook has none). Edits are limited to three kinds: link targets that pointed at files this migration moved were re-pointed; path anchors in code that the move invalidated were replaced, each marked in place with a `Skill note (not in the source)` comment; and location text that named the retired cookbook folder (default output folders, where to run from) now names this skill's folder, each marked by a skill note. The changes are recorded in `docs/cookbook_to_skill_migration.md`.

## 1. Prerequisites

**Server host:** Linux with Docker, the NVIDIA Container Toolkit, and one NVIDIA GPU with enough memory for the model. The SGLang rollout runs on FlexAttention's Triton kernels, so it does not need the FlashAttention-4 kernels that the [Framework notebook](sim-auto-cosmos-framework.md) requires. The host driver must support the CUDA runtime of the `lmsysorg/sglang:dev` image. `HF_TOKEN` needs read access to `nvidia/Cosmos3-Nano-Sim-Auto`. The server opens the inputs by path, so the cookbook checkout, including the output folder this notebook writes, must be on the server host or on a filesystem it shares with this notebook, at one absolute path.

**Notebook client:** Python 3.10 or newer, and a Hugging Face token with read access to `nvidia/Cosmos3-Nano-Transfer-Auto`, which publishes the sample; run `hf auth login` or set `HF_TOKEN`. In the notebook kernel environment, install:

```bash
pip install httpx pillow numpy matplotlib ipython huggingface_hub
```

Install FFmpeg on the client host; the package also includes FFprobe. On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

On macOS with Homebrew, use `brew install ffmpeg`. Verify that both `ffmpeg -version` and `ffprobe -version` work.

> **Skill note (not in the source):** the helpers call FFmpeg with `-fps_mode`, which needs FFmpeg 5.1 or newer. Ubuntu 22.04 installs 4.4.2, and the first helper call then stops with `Unrecognized option 'fps_mode'`. Run the block below right after the install above: it looks for the option and, when the installed FFmpeg lacks it, puts the FFmpeg 7.0.2 of `imageio-ffmpeg==0.6.0` first on `PATH`, the version the repository's cookbook checks run these helpers with (`ffprobe` from the distribution package is enough). In a Python environment without `pip`, such as one made by `uv venv`, the block installs the package with `uv pip`, run from `/` because uv stops inside the Framework checkout unless it is the pinned 0.11.29. The `PATH` line holds for the current shell only. Where `/usr/local/bin` is writable and holds no `ffmpeg` (root in a container), the block also links that FFmpeg there, ahead of `/usr/bin` on the default `PATH`, so shells and Python processes started later find it; elsewhere, run the block again in the shell that starts the client:

```bash
# Skill note (not in the source): adds FFmpeg 7.0.2 from imageio-ffmpeg when the installed FFmpeg has no -fps_mode.
if [ -z "$(ffmpeg -hide_banner -h full 2>/dev/null | grep -- '-fps_mode')" ]; then
  if python -m pip --version >/dev/null 2>&1; then python -m pip install imageio-ffmpeg==0.6.0; else (cd / && uv pip install --python "$(command -v python)" imageio-ffmpeg==0.6.0); fi
  mkdir -p "$PWD/ffmpeg7" && ln -sf "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$PWD/ffmpeg7/ffmpeg"
  export PATH="$PWD/ffmpeg7:$PATH"
  if [ -w /usr/local/bin ] && [ ! -e /usr/local/bin/ffmpeg ] && [ -x "$PWD/ffmpeg7/ffmpeg" ]; then ln -sf "$(readlink -f "$PWD/ffmpeg7/ffmpeg")" /usr/local/bin/ffmpeg; fi
fi
```

**Inputs:** The example uses the shared daytime sample published with Transfer-Auto under `assets/sample_data`: one scene with seven synchronized cameras, each with an RGB recording, a WSM control, and a caption. Step 3 downloads the files this example needs, about 100 MiB, into the Hugging Face cache and checks them against the sample's `SHA256SUMS`. To use a copy you already have, set `AUTO_SAMPLE_ROOT` to its `sample_data` folder.

Nothing is uploaded. Step 4 writes the trimmed WSM controls and a copy of each RGB condition under the output folder, inside the mounted cookbook root, and the request names those files by absolute path.

## 2. Start the server

The server runs in the published SGLang image, `lmsysorg/sglang:dev`, which includes the Cosmos3 Sim-Auto pipeline. On the server host, set `COSMOS_ROOT` to the absolute cookbook root (`.agents/skills`), which step 3 prints as `Cookbook root`. `SGLANG_HF_CACHE` is a host directory for the Hugging Face cache. The container runs as root by default, so the setup lines create that directory world-writable; alternatively, start the container with your host user ID. Pull the image, then run the server in a terminal and leave it running while using the notebook:

```bash
docker pull lmsysorg/sglang:dev
export COSMOS_ROOT=/absolute/path/to/cosmos-framework/.agents/skills
export SGLANG_HF_CACHE="${SGLANG_HF_CACHE:-$HOME/.cache/sglang-huggingface}"
mkdir -p "$SGLANG_HF_CACHE"
chmod 777 "$SGLANG_HF_CACHE"
docker run --init --runtime nvidia --gpus all \
  -v $SGLANG_HF_CACHE:/root/.cache/huggingface \
  -v "${COSMOS_ROOT}:${COSMOS_ROOT}" \
  -e HF_TOKEN=$HF_TOKEN \
  -p 30000:30000 \
  --ipc=host \
  lmsysorg/sglang:dev \
  sglang serve \
  --model-path nvidia/Cosmos3-Nano-Sim-Auto \
  --revision main \
  --host 0.0.0.0 --port 30000
```

| Setting                                        | Effect                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-v "${COSMOS_ROOT}:${COSMOS_ROOT}"`           | Mounts the cookbook root at its host path. The request names every WSM control and RGB condition by absolute path, and the trimmed controls and staged RGB conditions that step 4 writes under the output folder must be visible at the same path.                                                                                                                                                                                                                                                                            |
| `-v $SGLANG_HF_CACHE:/root/.cache/huggingface` | Keeps the downloaded checkpoint across container restarts. The setup lines default the directory to `~/.cache/sglang-huggingface`, create it, and make it world-writable.                                                                                                                                                                                                                                                                                                                                                     |
| `-e HF_TOKEN=$HF_TOKEN`                        | Read access to the gated checkpoint.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `--revision`                                   | Loads the checkpoint from `main`. The first start downloads it into the mounted cache.                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `docker pull`                                  | The `dev` tag moves with SGLang's main branch; pulling before each start picks up the current image. This notebook was validated against the image built from SGLang commit [`ab580b2e`](https://github.com/sgl-project/sglang/commit/ab580b2e060059c0f8387b598a9f648dcec13f93), published as the fixed tag `lmsysorg/sglang:nightly-dev-20261020-ab580b2e`; use that tag in place of `dev` to reproduce the validated run.                                                                                                   |
| `--gpus all`                                   | Makes every GPU visible; select one with `--gpus '"device=0"'`. With N GPUs, add `--num-gpus N --ulysses-degree N --sp-degree N` to `sglang serve`: Ulysses sequence parallelism splits every chunk across the GPUs and speeds up generation. Sim-Auto supports no other sequence parallelism, so pass both degrees explicitly, and N must divide the checkpoint's 8 key/value heads (2, 4, or 8). `--tp-size` can be combined when the model does not fit on one GPU, as long as `tp_size × ulysses_degree` still divides 8. |
| `--host 0.0.0.0`                               | Accepts connections from outside the container. For a remote notebook, set `SGLANG_BASE_URL`, or forward the port with `ssh -L 30000:127.0.0.1:30000 <server-host>`.                                                                                                                                                                                                                                                                                                                                                          |

The HTTP port opens only after the weights are loaded and the warmup request finishes, so wait for the `Uvicorn running` line. The start-up log also prints `Cosmos3-Nano-Sim-Auto pipeline stages created`, followed by the checkpoint's chunk, ring, and sampler settings. This pipeline runs no text or video guardrails. Generation runs as an asynchronous job, so no server-side request timeout needs raising.

## 3. Set up the client

Run the notebook from anywhere inside `.agents/skills`, or set `COSMOS_COOKBOOK_ROOT` to it. These optional environment variables override the defaults:

| Variable             | Default                                                                    | Purpose                                                                                                                           |
| -------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `SGLANG_BASE_URL`    | `http://127.0.0.1:30000`                                                   | Server address                                                                                                                    |
| `SGLANG_API_KEY`     | unset                                                                      | Bearer token, if the server was started with `--api-key`                                                                          |
| `COSMOS3_OUTPUT_DIR` | `.agents/skills/cosmos3-inference-simulation-auto/outputs/sglang/sim_auto` | Output folder. Keep it inside the mounted cookbook root: the server reads the trimmed controls and staged RGB conditions from it. |
| `AUTO_SAMPLE_ROOT`   | unset                                                                      | A local `sample_data` folder to use instead of downloading the sample                                                             |

> **Skill note (not in the source):** the `COSMOS3_OUTPUT_DIR` default in the table above was `.agents/skills/cookbooks/cosmos3/simulation/outputs/sglang/sim_auto` in the source; the default output folder moved into this skill with the cookbook.

The next cell sets the configuration and imports the Sim-Auto input helpers from [`helpers/sim_auto_sglang_helpers.py`](../scripts/helpers/sim_auto_sglang_helpers.py) and the sample helpers from [`helpers/daytime_sample.py`](../scripts/helpers/daytime_sample.py). The cell after it defines the client helpers for server readiness, Videos API jobs, and output validation; neither needs GPU libraries. Then fetch the sample.

```python
from pathlib import Path
import copy
import json
import os
import sys
import time

import httpx
import matplotlib.pyplot as plt
from IPython.display import HTML, display


def find_auto_root() -> Path:
    configured = os.environ.get("COSMOS_COOKBOOK_ROOT")
    start = Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()
    for root in [start] if configured else [start, *start.parents]:
        # Skill note (not in the source): the Auto cookbook folder is now this skill's folder; the candidates were root / "auto", root / "cookbooks" / "cosmos3" / "simulation" / "auto", root / ".agents" / "skills" / "cookbooks" / "cosmos3" / "simulation" / "auto".
        for candidate in (root, root / "cosmos3-inference-simulation-auto", root / ".agents" / "skills" / "cosmos3-inference-simulation-auto"):
            # Skill note (not in the source): the helpers are now in this skill's scripts/helpers folder; the marker was candidate / "helpers" / "sim_auto_sglang_helpers.py".
            if (candidate / "scripts" / "helpers" / "sim_auto_sglang_helpers.py").is_file():
                return candidate
    raise FileNotFoundError("Run inside .agents/skills or set COSMOS_COOKBOOK_ROOT to it.")


AUTO_ROOT = find_auto_root()
# Skill note (not in the source): the Simulation cookbook folder is retired and this skill's folder takes its place, so the default outputs below land in AUTO_ROOT / "outputs"; this was SIMULATION_ROOT = AUTO_ROOT.parent.
SIMULATION_ROOT = AUTO_ROOT
# Skill note (not in the source): .agents/skills is now the parent of the skill folder; this was SIMULATION_ROOT.parents[2].
COOKBOOK_ROOT = SIMULATION_ROOT.parent  # The .agents/skills folder the server mounts.
# Skill note (not in the source): the helpers package is now under this skill's scripts folder; this line and the next used str(AUTO_ROOT).
if str(AUTO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(AUTO_ROOT / "scripts"))

from helpers.auto_helpers import prepare_controls, split_cameras, stage_conditions, validate_rgb_conditions
from helpers.daytime_sample import SAMPLE_REPO, SAMPLE_REVISION, fetch_sample
from helpers.sim_auto_sglang_helpers import CAMERA_ORDER, MODEL_ID, load_request, prepare_request, require_inputs

MODEL_REVISION = "main"
SGLANG_BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000").rstrip("/")
SGLANG_API_KEY = os.environ.get("SGLANG_API_KEY")
HEADERS = {"Authorization": f"Bearer {SGLANG_API_KEY}"} if SGLANG_API_KEY else {}
HTTP_TIMEOUT = 600  # Per HTTP request: submission, each poll, and the MP4 download.
GENERATION_TIMEOUT = 3600  # Job wait, from submission to completion.
OUTPUT_DIR = Path(
    os.environ.get("COSMOS3_OUTPUT_DIR", SIMULATION_ROOT / "outputs" / "sglang" / "sim_auto")
).expanduser().resolve()
AUTO_SAMPLE_ROOT = os.environ.get("AUTO_SAMPLE_ROOT")  # Unset: download the sample from Hugging Face
CAPTION_FOLDER = "captions_with_timestamp"  # Alternative: "captions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Cookbook root:", COOKBOOK_ROOT)
print("Server:", SGLANG_BASE_URL, "(with API key)" if SGLANG_API_KEY else "")
print("Model:", MODEL_ID, "@", MODEL_REVISION)
print("Sample:", AUTO_SAMPLE_ROOT or f"{SAMPLE_REPO} @ {SAMPLE_REVISION}")
print("Outputs:", OUTPUT_DIR)
if not OUTPUT_DIR.is_relative_to(COOKBOOK_ROOT):
    print(f"Note: {OUTPUT_DIR} is outside the cookbook root; the server must see it at the same path.")
```

```python
# Client helpers. The same cell appears in each Simulation serving notebook; keep the copies in sync.
import base64
import html
import json
import shutil
import subprocess
import tempfile
import time
from fractions import Fraction
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

_LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


def require_media_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
    if missing:
        raise RuntimeError(f"Install FFmpeg and add {' and '.join(missing)} to PATH.")


def require_file(path: Path) -> Path:
    """Resolve an input file and reject missing, empty, or unhydrated Git LFS files."""
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty input: {path}")
    with path.open("rb") as stream:
        if stream.read(len(_LFS_POINTER)) == _LFS_POINTER:
            raise ValueError(f"Unhydrated Git LFS pointer: {path}. Run git lfs pull in the cookbook checkout.")
    return path


def wait_for_model(client: httpx.Client, model: str, *, timeout: float = 1800, interval: float = 5) -> list[str]:
    """Poll ``GET /v1/models`` until the server answers, then require ``model``.

    Connection errors and 5xx responses mean the server is still starting and are
    retried until ``timeout``. Other HTTP errors, such as a missing API key, and a
    server that answers without ``model`` fail immediately.
    """
    deadline = time.monotonic() + timeout
    waiting = False
    while True:
        try:
            response = client.get("/v1/models", timeout=min(30, timeout))
            if response.status_code < 500:
                break
            reason = f"HTTP {response.status_code}"
        except httpx.TransportError as error:
            reason = type(error).__name__
        if time.monotonic() + interval > deadline:
            raise TimeoutError(f"No ready server at {client.base_url} after {timeout:.0f}s ({reason}).")
        if not waiting:
            print(f"Waiting for the server at {client.base_url} ({reason}) ...")
            waiting = True
        time.sleep(interval)
    response.raise_for_status()
    served = [item["id"] for item in response.json().get("data", []) if item.get("id")]
    if model not in served:
        raise ValueError(f"Expected served model {model!r}; the server at {client.base_url} serves {served}.")
    return served


def submit_video_job(client: httpx.Client, form: dict, output_dir: Path, *, timeout: float, interval: float = 10) -> dict:
    """Send ``POST /v1/videos`` as multipart form fields and poll the job until it completes.

    The form is saved as ``request.json`` beside the outputs so the request can be
    inspected or replayed, and every job response as ``job.json``. A rejected
    submission or a failed job saves the server's response as ``response_error.txt``
    and raises. A polling timeout raises too, but leaves the job running on the
    server at the printed URL.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "request.json").write_text(json.dumps(form, indent=2, ensure_ascii=False) + "\n")
    error_path = output_dir / "response_error.txt"
    error_path.unlink(missing_ok=True)
    job_path = output_dir / "job.json"
    job_path.unlink(missing_ok=True)
    # A None filename makes each part a regular field; the Videos API parses multipart, not URL-encoded, forms.
    response = client.post("/v1/videos", files=[(key, (None, value)) for key, value in form.items()])
    if response.is_error:
        error_path.write_text(response.text)
        raise RuntimeError(f"HTTP {response.status_code} from /v1/videos; see {error_path}\n{response.text[:2000]}")
    job = response.json()
    job_url = f"/v1/videos/{job['id']}"
    deadline = time.monotonic() + timeout
    print(f"Submitted job {job['id']} at {client.base_url.join(job_url)}")
    while True:
        job_path.write_text(json.dumps(job, indent=2) + "\n")
        status = job.get("status")
        if status == "completed":
            return job
        if status in {"failed", "cancelled"}:
            error_path.write_text(json.dumps(job, indent=2) + "\n")
            raise RuntimeError(f"Video generation {status}: {job.get('error')}; see {error_path}")
        if status not in {"queued", "in_progress"}:
            raise RuntimeError(f"Unexpected job status {status!r}; see {job_path}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"No result after {timeout:.0f}s; the job remains at {client.base_url.join(job_url)}. See {job_path}."
            )
        time.sleep(min(interval, remaining))
        response = client.get(job_url)
        if response.is_error:
            error_path.write_text(response.text)
            raise RuntimeError(f"HTTP {response.status_code} from {job_url}; see {error_path}\n{response.text[:2000]}")
        job = response.json()


def download_video(client: httpx.Client, job: dict, output_path: Path) -> Path:
    """Save a completed job's ``GET /v1/videos/{id}/content`` MP4 atomically."""
    output_path = Path(output_path)
    temporary = output_path.with_name(output_path.name + ".part")
    try:
        with client.stream("GET", f"/v1/videos/{job['id']}/content", headers={"Accept": "video/mp4"}) as response:
            if response.is_error:
                response.read()
                raise RuntimeError(f"HTTP {response.status_code} from the content download\n{response.text[:2000]}")
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("video/mp4"):
                raise ValueError(f"Expected a video/mp4 response; got {content_type!r}.")
            with temporary.open("wb") as stream:
                for chunk in response.iter_bytes():
                    stream.write(chunk)
        if temporary.stat().st_size == 0:
            raise ValueError("The content download was empty.")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def probe_video(path: Path) -> dict:
    """Return the first video stream's size, average FPS, and decoded frame count."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No video stream in {path}")
    stream = streams[0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": Fraction(stream["avg_frame_rate"]),
        "frames": int(stream["nb_read_frames"]),
    }


def validate_video(path: Path, *, num_frames: int, fps: float, width: int, height: int) -> dict:
    """Require the requested frame count, FPS, and size, and an error-free full decode."""
    info = probe_video(path)
    problems = []
    if info["frames"] != num_frames:
        problems.append(f"{info['frames']} frames, expected {num_frames}")
    if info["fps"] != Fraction(fps):
        problems.append(f"{float(info['fps']):g} FPS, expected {fps:g}")
    if (info["width"], info["height"]) != (width, height):
        problems.append(f"{info['width']}x{info['height']}, expected {width}x{height}")
    decode = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
        capture_output=True, text=True,
    )
    if decode.returncode or decode.stderr.strip():
        problems.append(f"decode failed: {decode.stderr.strip()[:500]}")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return {**info, "fps": float(info["fps"])}


def read_frame(path: Path, index: int) -> Image.Image:
    """Decode one zero-based frame as RGB."""
    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
            "-vf", rf"select=eq(n\,{index})", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
        ],
        check=True, capture_output=True,
    )
    if not result.stdout:
        raise ValueError(f"{path} has no frame {index}")
    return Image.open(BytesIO(result.stdout)).convert("RGB")


def preview_video(path: Path, *, width: int = 416, frames: int | None = None) -> bytes:
    """Re-encode a small H.264 preview, optionally trimmed to ``frames``, for embedding in a notebook."""
    with tempfile.TemporaryDirectory() as directory:
        preview = Path(directory) / "preview.mp4"
        limit = ["-frames:v", str(frames)] if frames else []
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-an", *limit,
                "-vf", f"scale={width}:-2", "-c:v", "libx264", "-crf", "28", "-preset", "fast",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(preview),
            ],
            check=True, capture_output=True,
        )
        return preview.read_bytes()


def video_row(videos: dict[str, Path], *, width: int = 416, frames: int | None = None) -> str:
    """HTML that embeds labeled previews side by side; they autoplay muted and loop together.

    Each video is re-encoded at ``width`` and trimmed to ``frames``, so a long or
    high-resolution input stays small and loops in step with the others.
    """
    cells = []
    for label, path in videos.items():
        data = base64.b64encode(preview_video(path, width=width, frames=frames)).decode("ascii")
        cells.append(
            f'<figure style="margin:0"><figcaption style="font-family:sans-serif">{html.escape(label)}</figcaption>'
            f'<video width="{width}" autoplay muted loop controls playsinline '
            f'src="data:video/mp4;base64,{data}"></video></figure>'
        )
    return '<div style="display:flex;flex-wrap:wrap;gap:8px">' + "".join(cells) + "</div>"


def write_sample_outputs(output_dir: Path, samples: list[dict]) -> Path:
    """Write ``sample_outputs.json``: one record per generated sample, each with a ``status``."""
    for sample in samples:
        if "status" not in sample:
            raise ValueError("Every sample record needs a status.")
    path = Path(output_dir) / "sample_outputs.json"
    path.write_text(json.dumps(samples, indent=2, default=str) + "\n")
    return path


require_media_tools()
```

### Download and verify the sample

Download the seven cameras' RGB recordings, WSM controls, and both caption styles from `assets/sample_data` on Hugging Face, and verify them against `SHA256SUMS`; later runs reuse the Hugging Face cache. No model weights are downloaded here.

`CAPTION_FOLDER="captions_with_timestamp"` selects timestamped captions, which better match training and empirically improve inference. Set it to `"captions"` for the original dense paragraphs.

```python
SAMPLE_ROOT = fetch_sample(AUTO_SAMPLE_ROOT)
print("Verified sample:", SAMPLE_ROOT)
print("Cameras:", ", ".join(CAMERA_ORDER))
```

### Verify the server

Wait for the server to finish starting, then confirm that it serves the Sim-Auto model. The cell polls for up to 30 minutes, so you can run it while the server is still loading. `/v1/models` reports the model name, not its revision; the `--revision` in the server command pins that.

```python
with httpx.Client(base_url=SGLANG_BASE_URL, headers=HEADERS) as client:
    served = wait_for_model(client, MODEL_ID, timeout=1800)
print("Server ready; serving", served)
```

## 4. Prepare the inputs

The [RGB request](../assets/daytime_scene/sim_auto.json) asks for **297 frames per camera at 30 FPS** with `resolution: "480"`, seed **0**, and `condition_video_as_image: true`. It lists the sample's seven cameras in the checkpoint's camera order, each with its WSM control, RGB recording, and caption file; every file is named after its camera, such as `daytime_scene/wsm/camera_front_wide_120fov.mp4`. The server identifies each camera by its `camera_key`, so the seven keep their trained camera IDs 0 to 6; the generated video follows the request order. The checkpoint fixes guidance 1 and four steps.

`load_request` resolves the paths under the sample and replaces each caption file with its text, which covers all 300 source frames. The Videos API also takes a top-level caption; the request copies the front wide camera's.

Every source video has 300 frames at 30 FPS and 1280 × 720. The server resizes and center-crops inputs to its 832 × 480 output.

The next cell checks the request and reports every missing file at once. It decodes each RGB condition's first frame, which is all that generation uses. It fully decodes the WSM controls, checks that their timelines are synchronized, and trims their first 297 frames without resizing into the output folder; controls that already have 297 frames are copied there unchanged. Because SGLang opens inputs by path and the sample lives in this notebook's Hugging Face cache, the cell also copies each RGB condition into the output folder, so every path in the request lies inside the mounted cookbook root. To use WSM controls from another directory, such as a different render of this scene, set `CONTROL_DIR` to a folder with one `<camera_key>.mp4` per camera.

```python
REQUEST_PATH = AUTO_ROOT / "assets" / "daytime_scene" / "sim_auto.json"
CONTROL_DIR = None  # Or a directory with one <camera_key>.mp4 WSM control per camera

manifest = load_request(REQUEST_PATH, SAMPLE_ROOT, caption_folder=CAPTION_FOLDER, control_dir=Path(CONTROL_DIR) if CONTROL_DIR else None)
require_inputs(manifest)
views = manifest["multiview"]["views"]
validate_rgb_conditions(views, SAMPLE_ROOT)

sample_dir = OUTPUT_DIR / "rgb"
controls = prepare_controls(views, SAMPLE_ROOT, sample_dir / "prepared_controls", manifest["num_frames"], manifest["fps"], materialize=True)
conditions = stage_conditions(views, sample_dir / "staged_conditions")
prepared = copy.deepcopy(manifest)
for view, control, condition in zip(prepared["multiview"]["views"], controls, conditions, strict=True):
    view["control_path"] = str(control)
    view["vision_path"] = str(condition)
rgb_conditions = dict(zip(CAMERA_ORDER, conditions))
wsm_controls = dict(zip(CAMERA_ORDER, controls))
print(f"{manifest['name']}: {manifest['num_frames']} frames per camera, {manifest['fps']:g} FPS, seed {manifest['seed']}")
for camera_key, control, condition in zip(CAMERA_ORDER, controls, conditions):
    print(f"  {camera_key:26}  WSM {control.name}, RGB {condition.name}")
```

### Preview the inputs

The first frame of each camera's WSM control and of its RGB condition.

```python
def camera_frames(paths: dict[str, Path], index: int, title: str) -> None:
    """Show one frame of each camera in a 2 × 4 grid."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 5), layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, (camera_key, path) in zip(axes.flat, paths.items()):
        ax.imshow(read_frame(path, index))
        ax.set_title(camera_key, fontsize=9)
    fig.suptitle(title)
    display(fig)
    plt.close(fig)


camera_frames(wsm_controls, 0, "WSM controls — frame 0")
camera_frames(rgb_conditions, 0, "RGB conditions — frame 0")
```

## 5. Generate

The example submits one `POST /v1/videos` job for the complete rollout, polls `GET /v1/videos/{id}` until it completes, and downloads the finished MP4 from `GET /v1/videos/{id}/content`. The request is multipart form fields with no file uploads:

- **Form fields:** the model, top-level caption, canvas `size`, `num_frames`, `fps`, and `seed`, taken from the request file, plus the checkpoint's fixed `guidance_scale` of 1 and four `num_inference_steps`. The server rejects other values for the last two instead of ignoring them.
- **`extra_params` (JSON):** `wsm: {}`; `multiview.views` with each camera's key, caption, and the absolute `control_path` and `vision_path` the server opens; `multiview.condition_video_as_image: true`; `resolution` and `aspect_ratio`. SGLang takes no `*_reference_index` fields and no session fields: each job is one complete rollout, and the model stays loaded between jobs.

The next cell builds the request and prints it, with the captions shortened and the paths relative to the cookbook root.

```python
form, uploads = prepare_request(prepared, MODEL_ID)
extra_params = json.loads(form["extra_params"])
assert not uploads  # SGLang opens control_path and vision_path itself; nothing is uploaded.
assert [view["camera_key"] for view in extra_params["multiview"]["views"]] == list(CAMERA_ORDER)
assert all(
    Path(view[key]).is_absolute() and not any(field.endswith("_reference_index") for field in view)
    for view in extra_params["multiview"]["views"]
    for key in ("control_path", "vision_path")
)


def short(text: str) -> str:
    return text if len(text) <= 80 else text[:80] + " ..."


def portable(path: str) -> str:
    return str(Path(path).relative_to(COOKBOOK_ROOT)) if Path(path).is_relative_to(COOKBOOK_ROOT) else path


for view in extra_params["multiview"]["views"]:
    view["prompt"] = short(view["prompt"])
    for key in ("control_path", "vision_path"):
        view[key] = portable(view[key])
printable = {**form, "prompt": short(form["prompt"]), "extra_params": extra_params}
print(json.dumps(printable, indent=2, ensure_ascii=False))
```

Generate the rollout. The first job takes longer than later ones, and the cell prints the job URL as soon as the server accepts the request. Each run of this cell submits a new job; a failed run leaves no video for the next step to validate.

```python
video = None
form, _ = prepare_request(prepared, MODEL_ID)
with httpx.Client(base_url=SGLANG_BASE_URL, headers=HEADERS, timeout=HTTP_TIMEOUT) as client:
    started = time.monotonic()
    job = submit_video_job(client, form, sample_dir, timeout=GENERATION_TIMEOUT)
    video = download_video(client, job, sample_dir / "multiview_camera_major.mp4")
    elapsed_seconds = round(time.monotonic() - started, 1)
print(f"{elapsed_seconds:.1f}s, including polling and download  {video}")
```

## 6. Validate and view

The result is one camera-major MP4: 297 frames of the first camera, then 297 of the second, through the seventh. Decode it completely, check its `7 × 297 = 2079` frames, FPS, and canvas, then split it into one labeled video per camera. The folder then holds:

```text
outputs/sglang/sim_auto/rgb/
  request.json                    # The exact form fields that were sent
  job.json                        # The last job response
  prepared_controls/              # The seven trimmed WSM controls the server read
  staged_conditions/              # The seven RGB conditions, copied from the sample for the server
  multiview_camera_major.mp4      # The downloaded result
  vision_viewNN_<camera>.mp4      # One video per camera
  sample_outputs.json             # Status, timing, job, inputs, cameras, and the decoded video properties
```

A rejected submission or a failed job leaves `response_error.txt` with the server's response instead of the MP4. A video that decodes at the right size does not establish control adherence, cross-camera consistency, or visual quality; inspect the frames and playback below, and see the [shared Simulation checklist](../../cosmos3-inference-simulation/references/simulation-readme.md#validate-a-rollout).

```python
if video is None:
    raise RuntimeError("Complete generation before validating outputs.")
num_frames, fps = manifest["num_frames"], manifest["fps"]
info = validate_video(video, num_frames=len(CAMERA_ORDER) * num_frames, fps=fps, width=832, height=480)
cameras = split_cameras(video, sample_dir, CAMERA_ORDER, num_frames, fps)
write_sample_outputs(sample_dir, [{
    "name": manifest["name"],
    "status": "success",
    "job_id": job["id"],
    "model_revision": MODEL_REVISION,
    "sample_revision": SAMPLE_REVISION,
    "video": video,
    "request": sample_dir / "request.json",
    "job": sample_dir / "job.json",
    "inputs": [
        view[key] for view in json.loads(form["extra_params"])["multiview"]["views"] for key in ("control_path", "vision_path")
    ],
    "elapsed_seconds": elapsed_seconds,
    "frames_per_camera": num_frames,
    "cameras": cameras,
    **info,
}])
print(f"{info['frames']} camera-major frames, {info['width']}×{info['height']}, {info['fps']:g} FPS; "
      f"full decode passed; split into {len(cameras)} cameras of {num_frames} frames")
```

### Compare with the conditions

The first generated frame of each camera, to compare with its RGB condition above, and the final frame. The first frame should closely follow the condition; later frames should follow the WSM controls' motion and layout and keep objects consistent where neighboring cameras overlap.

```python
camera_frames(cameras, 0, "SGLang output — frame 0")
camera_frames(cameras, num_frames - 1, f"SGLang output — frame {num_frames - 1}")
```

### Play the videos

Each camera plays muted and looped. The players embed small re-encoded previews, so saved notebook outputs stay compact; the full-resolution MP4s remain in the output folder.

```python
display(HTML(video_row(cameras, width=320)))
```

## 7. Next steps

- **Use your own scene:** copy [`sim_auto.json`](../assets/daytime_scene/sim_auto.json), point its paths at your data, then set `REQUEST_PATH` to the copy and `SAMPLE_ROOT` to the folder its paths are relative to. Keep the seven cameras in the order above, each with a `control_path`, a `vision_path`, and a caption file or inline `prompt`. Keep `num_frames` at 4k + 1, guidance 1, and four steps. The prepared controls and staged conditions always land under the output folder, so keep it inside the mounted cookbook root. For a different WSM render of this scene, set `CONTROL_DIR` instead.
- **Validate further:** apply the [shared Simulation checklist](../../cosmos3-inference-simulation/references/simulation-readme.md#validate-a-rollout). Check camera alignment and temporal consistency before using results downstream, and keep run-specific notes, including the server's GPU and the checkpoint revision it resolved from `main`, alongside your generated outputs.
- **Troubleshooting:**
  - *Missing inputs:* rerun the sample download in step 3 and then **Prepare the inputs**; `AUTO_SAMPLE_ROOT` must point at a complete `sample_data` folder.
  - *HTTP error on submission:* inspect `response_error.txt` and the server log. The pipeline needs a WSM control and an RGB condition for every camera, cameras from the checkpoint's eleven, guidance 1, four steps, and `condition_video_as_image: true`. A path the server cannot open means the checkout or the output folder is not mounted at the same absolute path inside the container.
  - *Job `failed`:* `job.json` and `response_error.txt` carry the server's error message; the server log has the traceback.
  - *`Cosmos3-Nano-Sim-Auto requires transformer/config.json['cosmos3_nano_sim_auto']` at start-up:* the checkpoint is not a Sim-Auto export with the SGLang artifact. Use the published checkpoint on `main`, or an export produced with `--cosmos3-nano-sim-auto`.
  - *Timeout:* the job keeps running on the server at the printed URL; `GET /v1/videos/{id}` reports its status. Raise `GENERATION_TIMEOUT` before resubmitting.
- **Other models and runtimes:** [Cosmos3-Nano-Transfer-Auto with SGLang](transfer-auto-sglang.md) generates all eleven cameras with full diffusion sampling, with an optional joint LiDAR example. The [vLLM-Omni notebook](sim-auto-vllm-omni.md) serves this sample through vLLM-Omni's synchronous endpoint, and the [Framework notebook](sim-auto-cosmos-framework.md) runs Sim-Auto natively, including joint RGB and LiDAR generation.
