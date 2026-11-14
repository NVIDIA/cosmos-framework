<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3-Nano-Transfer-Auto with vLLM-Omni

Cosmos3-Nano-Transfer-Auto generates a **synchronized multi-camera driving clip** from one World Scenario Model (WSM) control video per camera. The WSM controls render the scene's lane lines, road surface, and vehicle boxes from each camera's viewpoint, and a caption per camera describes its appearance. Joint requests also condition every camera on a first RGB frame and on a numeric LiDAR range map, and **generate numeric LiDAR** alongside the cameras.

This notebook runs two examples from the [Transfer-Auto guide](transfer-auto-guide.md) on the shared **daytime sample**, one scene recorded by seven cameras, through one vLLM-Omni server. Each runs at **201 frames per camera, 30 FPS, and 832 × 480**:

1. **WSM-only:** seven WSM controls produce seven camera videos through the synchronous Videos API.
2. **Joint RGB and LiDAR:** seven WSM controls, seven first-frame RGB conditions, and a numeric HD-map LiDAR control produce seven camera videos and 67 numeric LiDAR sweeps through the asynchronous Videos API.

The examples are independent. To run only the LiDAR example, complete steps 1 to 3, then go to step 5.

| Component                           | Version                                                                      |
| ----------------------------------- | ---------------------------------------------------------------------------- |
| Server image                        | `vllm/vllm-omni:cosmos3`                                                     |
| `nvidia/Cosmos3-Nano-Transfer-Auto` | [`main`](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto/tree/main) |

The sample comes from the same repository and includes both caption styles. Steps 1 and 2 run on the server host. Steps 3 to 5 run in this notebook.

> **Skill note (not in the source):** this file is the cookbook notebook `run_auto_with_vllm_omni.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/simulation/auto/run_auto_with_vllm_omni.ipynb`, cosmos-framework `main` 45fecab) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced as `python`, outputs omitted (the committed notebook has none). Edits are limited to three kinds: link targets that pointed at files this migration moved were re-pointed; path anchors in code that the move invalidated were replaced, each marked in place with a `Skill note (not in the source)` comment; and location text that named the retired cookbook folder (default output folders, where to run from) now names this skill's folder, each marked by a skill note. The changes are recorded in `docs/cookbook_to_skill_migration.md`.

## 1. Prerequisites

**Server host:** Linux with at least one NVIDIA GPU, Docker with the NVIDIA Container Toolkit, and a Hugging Face token with read access to `nvidia/Cosmos3-Nano-Transfer-Auto`.

**Notebook client:** Python 3.10 or newer, and a Hugging Face token with read access to `nvidia/Cosmos3-Nano-Transfer-Auto`; run `hf auth login` or set `HF_TOKEN`. In the notebook kernel environment, install:

```bash
pip install httpx pillow numpy attrs safetensors matplotlib ipython huggingface_hub
```

Install FFmpeg with the `libx264` encoder on the client host; the package also includes FFprobe. On Ubuntu/Debian:

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

**Inputs:** The examples use the shared daytime sample published with the model under `assets/sample_data`: one scene with seven synchronized cameras, each with an RGB recording, a WSM control, and a caption. Step 3 downloads the files these examples need, about 100 MiB, into the Hugging Face cache and checks them against the sample's `SHA256SUMS`. To use a copy you already have, set `AUTO_SAMPLE_ROOT` to its `sample_data` folder.

The joint preparation cell converts the published `daytime_scene/lidar/control.tar` to a local `prepared/lidar_control.safetensors` file for vLLM-Omni. Native Framework reads the archive directly; no converted tensor or duplicate media is bundled. See the [format notes](../assets/daytime_scene/README.md).

Both requests upload their files, so the notebook can run on any host that reaches the server. The joint example uploads about 330 MiB in 15 files; raise any proxy request-size limit and give the server enough temporary disk space.

## 2. Start the server

All Simulation vLLM-Omni notebooks use the `vllm/vllm-omni:cosmos3` Docker image. Use a Linux GPU server with Docker and the NVIDIA Container Toolkit. Export `HF_TOKEN` with read access to the model, then pull the image:

```bash
docker pull vllm/vllm-omni:cosmos3
```

Start the container and leave it running while using the notebook:

```bash
docker run --rm --gpus '"device=0"' --ipc=host \
  -e HF_TOKEN \
  -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
  -e VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=1800 \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -p 127.0.0.1:8000:8000 \
  -w /app/vllm-omni \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano-Transfer-Auto \
    --revision main \
    --omni --model-class-name Cosmos3MultiviewPipeline \
    --diffusion-compile-granularity regional --no-diffusion-compile-dynamic \
    --no-guardrails \
    --host 0.0.0.0 --port 8000 --init-timeout 1800
```

The container reuses the host Hugging Face cache and listens on all container interfaces. Docker publishes port 8000 on the host loopback interface, matching `http://127.0.0.1:8000`. For a remote notebook, forward the port with `ssh -L 8000:127.0.0.1:8000 <server-host>`, or publish it on a reachable interface with `-p 8000:8000` and set `VLLM_BASE_URL`. The working directory selects the deployment YAML files bundled in the image.

| Setting                                                                      | Effect                                                                                                                                                                                                       |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800`                                          | Lets one synchronous request run for up to 30 minutes, instead of the 600-second default.                                                                                                                    |
| `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=1800`                                         | Lets the GPU workers spend up to 30 minutes on one request, instead of the 600-second default. A request that exceeds it shuts down the workers.                                                             |
| `--revision`                                                                 | Loads the checkpoint from `main`. The first start downloads it into the Hugging Face cache.                                                                                                                  |
| `--diffusion-compile-granularity regional`, `--no-diffusion-compile-dynamic` | **Required.** Compile the repeated transformer blocks for the fixed Multiview shapes. Without them, prompt-dependent graphs recompile or fall back to an attention score matrix that does not fit in memory. |
| `--no-guardrails`                                                            | Disables the Cosmos3 text and video safety guardrails for this example server.                                                                                                                               |
| `-p 127.0.0.1:8000:8000`, `--host 0.0.0.0`                                   | Exposes the container server on host loopback port 8000. Use the port-forwarding instructions above for a remote notebook.                                                                                   |

The command uses one GPU. For two GPUs, change Docker's GPU selection to `--gpus all` and add `--cfg-parallel-size 2`. For more, also set `--ulysses-degree`; for example, `--cfg-parallel-size 2 --ulysses-degree 2` uses four GPUs. Raise both timeouts above the measured request time on your hardware.

> **Skill note (not in the source):** measured on H100 NVL 95 GB: with the one-GPU launch the published WSM request (201 frames, 35 steps) did not finish within the two 1800-second timeouts (HTTP 504 at step 31 of 35); in the two-GPU form it took 1127 s. On GPUs of that class use the two-GPU form from the start: `--gpus all` in place of `--gpus '"device=0"'`, and `--cfg-parallel-size 2` added to the `vllm serve` options.

## 3. Set up the client

Run the notebook from anywhere inside `.agents/skills`, or set `COSMOS_COOKBOOK_ROOT` to it. These optional environment variables override the defaults:

| Variable             | Default                                                                   | Purpose                                                               |
| -------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `VLLM_BASE_URL`      | `http://127.0.0.1:8000`                                                   | Server address                                                        |
| `VLLM_API_KEY`       | unset                                                                     | Bearer token, if the server was started with `--api-key`              |
| `COSMOS3_OUTPUT_DIR` | `.agents/skills/cosmos3-inference-simulation-auto/outputs/vllm_omni/auto` | Output folder; each example writes to its own subfolder               |
| `AUTO_SAMPLE_ROOT`   | unset                                                                     | A local `sample_data` folder to use instead of downloading the sample |

> **Skill note (not in the source):** the `COSMOS3_OUTPUT_DIR` default in the table above was `.agents/skills/cookbooks/cosmos3/simulation/outputs/vllm_omni/auto` in the source; the default output folder moved into this skill with the cookbook.

Run this notebook from a complete Cosmos Framework checkout. The next cell sets the configuration and imports the Auto input helpers from [`helpers/auto_helpers.py`](../scripts/helpers/auto_helpers.py) and the sample helpers from [`helpers/daytime_sample.py`](../scripts/helpers/daytime_sample.py), along with Framework's shared calibration for CPU LiDAR visualization. The cell after it defines the client helpers for server readiness, requests, and output validation, and the next one the frame-grid and request-printing helpers that both examples use; none needs GPU libraries. Then fetch the sample.

```python
from pathlib import Path
from contextlib import ExitStack
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
            # Skill note (not in the source): the helpers are now in this skill's scripts/helpers folder; the marker was candidate / "helpers" / "auto_helpers.py".
            if (candidate / "scripts" / "helpers" / "auto_helpers.py").is_file():
                return candidate
    raise FileNotFoundError("Run inside .agents/skills or set COSMOS_COOKBOOK_ROOT to it.")


AUTO_ROOT = find_auto_root()
# Skill note (not in the source): the Simulation cookbook folder is retired and this skill's folder takes its place, so the default outputs below land in AUTO_ROOT / "outputs"; this was SIMULATION_ROOT = AUTO_ROOT.parent.
SIMULATION_ROOT = AUTO_ROOT
for root in AUTO_ROOT.parents:
    if (root / "cosmos_framework").is_dir():
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        break
else:
    raise FileNotFoundError("Use the complete Cosmos Framework checkout for shared LiDAR calibration.")
# Skill note (not in the source): the helpers package is now under this skill's scripts folder; this line and the next used str(AUTO_ROOT).
if str(AUTO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(AUTO_ROOT / "scripts"))

from helpers.auto_helpers import (
    download_lidar,
    prepare_controls,
    prepare_request,
    split_cameras,
    submit_joint,
    validate_lidar,
    validate_lidar_output,
    validate_rgb_conditions,
)
from helpers.daytime_sample import SAMPLE_CAMERAS, SAMPLE_REPO, SAMPLE_REVISION, fetch_sample, resolve_request

MODEL_ID = "nvidia/Cosmos3-Nano-Transfer-Auto"
MODEL_REVISION = "main"
VLLM_OMNI_IMAGE = "vllm/vllm-omni:cosmos3"
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY")
HEADERS = {"Authorization": f"Bearer {VLLM_API_KEY}"} if VLLM_API_KEY else {}
HTTP_TIMEOUT = 1800 + 60  # The server's synchronous timeout, plus a margin for the response.
OUTPUT_DIR = Path(
    os.environ.get("COSMOS3_OUTPUT_DIR", SIMULATION_ROOT / "outputs" / "vllm_omni" / "auto")
).expanduser().resolve()
AUTO_SAMPLE_ROOT = os.environ.get("AUTO_SAMPLE_ROOT")  # Unset: download the sample from Hugging Face
CAPTION_FOLDER = "captions_with_timestamp"  # Alternative: "captions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Cookbook:", AUTO_ROOT)
print("Server:", VLLM_BASE_URL, "(with API key)" if VLLM_API_KEY else "")
print("Model:", MODEL_ID, "@", MODEL_REVISION)
print("Server image:", VLLM_OMNI_IMAGE)
print("Sample:", AUTO_SAMPLE_ROOT or f"{SAMPLE_REPO} @ {SAMPLE_REVISION}")
print("Outputs:", OUTPUT_DIR)
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
            raise ValueError(f"Unhydrated Git LFS pointer: {path}. Download the actual file contents instead of a Git LFS pointer.")
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


def post_sync_video(client: httpx.Client, form: dict, output_path: Path, *, files=None) -> Path:
    """Send ``POST /v1/videos/sync`` and save the returned MP4 atomically.

    The form is saved as ``request.json`` beside the video so the request can be
    inspected or replayed. A rejected request saves the server's response body as
    ``response_error.txt`` and raises.
    """
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "request.json").write_text(json.dumps(form, indent=2, ensure_ascii=False) + "\n")
    error_path = output_dir / "response_error.txt"
    error_path.unlink(missing_ok=True)
    response = client.post("/v1/videos/sync", data=form, files=files, headers={"Accept": "video/mp4"})
    if response.is_error:
        error_path.write_text(response.text)
        raise RuntimeError(
            f"HTTP {response.status_code} from /v1/videos/sync; see {error_path}\n{response.text[:2000]}"
        )
    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("video/mp4") or not response.content:
        raise ValueError(
            f"Expected a nonempty video/mp4 response; got {content_type!r}, {len(response.content)} bytes."
        )
    temporary = output_path.with_name(output_path.name + ".part")
    temporary.write_bytes(response.content)
    temporary.replace(output_path)
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

```python
# Display helpers shared by both examples.
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


def show_request(form: dict, uploads: list[Path]) -> None:
    """Print a multipart request with long captions shortened."""
    def short(text: str) -> str:
        return text if len(text) <= 80 else text[:80] + " ..."

    extra_params = json.loads(form["extra_params"])
    for view in extra_params["multiview"]["views"]:
        view["prompt"] = short(view["prompt"])
    if "per_view_negative_prompt" in extra_params:
        extra_params["per_view_negative_prompt"] = short(extra_params["per_view_negative_prompt"])
    printable = {**form, "prompt": short(form["prompt"]), "extra_params": extra_params}
    printable["input_references"] = [str(path.relative_to(SIMULATION_ROOT)) if path.is_relative_to(SIMULATION_ROOT)
                                     else str(path) for path in uploads]
    print(json.dumps(printable, indent=2, ensure_ascii=False))
```

## Download and verify the sample

Download RGB, WSM, both caption styles, native requests and the numeric LiDAR control from `assets/sample_data` on Hugging Face. Verify them against `SHA256SUMS`; later runs reuse the Hugging Face cache. No model weights or recorded LiDAR targets are downloaded here.

`CAPTION_FOLDER="captions_with_timestamp"` selects timestamped captions, which better match training and empirically improve inference. Set it to `"captions"` for the original dense paragraphs. Keep object appearance and motion consistent across views; describe each camera’s visible scene.

```python
SAMPLE_ROOT = fetch_sample(AUTO_SAMPLE_ROOT)
print("Verified sample:", SAMPLE_ROOT)
print("Cameras:", ", ".join(SAMPLE_CAMERAS))
```

### Verify the server

Wait for the server to finish starting, then confirm that it serves the Transfer-Auto model. The cell polls for up to 30 minutes, so you can run it while the server is still loading.

```python
with httpx.Client(base_url=VLLM_BASE_URL, headers=HEADERS) as client:
    served = wait_for_model(client, MODEL_ID, timeout=1800)
print("Server ready; serving", served)
```

## 4. WSM-only example

### Prepare the inputs

The [WSM request](../assets/daytime_scene/transfer_auto_wsm.json) asks for **201 frames per camera at 30 FPS** with `resolution: "480"` and `aspect_ratio: "16,9"`, seed 0, 35 steps, guidance 3, and flow shift 10, matching the recommended Transfer-Auto baseline. It lists the sample's seven cameras in the checkpoint's camera order, each with its WSM control and caption file. The server identifies each camera by its `camera_key`, so the seven keep their trained camera IDs 0 to 6; the generated video follows the request order.

`resolve_request` resolves the paths under the sample and replaces each caption file with its text, which covers all 300 source frames. The Videos API also requires a top-level caption; the request copies the front wide camera's, which Transfer-Auto uses only for the text guardrail. The preparation cell supplies the inference skill's recommended `per_view_negative_prompt`. The runtime adds camera labels and control emphasis.

Every control is 1280 × 720 with 300 frames at 30 FPS. The next cell decodes every control, checks that their timelines are synchronized, and trims the first 201 frames without resizing. It then builds the multipart request: the seven controls become uploads, and each camera's `control_path` becomes a `control_reference_index`.

The preparation cell also loads the inference skill’s recommended per-view negative prompt. Guidance 3 controls caption strength; shift 10 sets the denoising timestep distribution. Use these settings as the starting recipe for your inputs.

```python
WSM_REQUEST = AUTO_ROOT / "assets" / "daytime_scene" / "transfer_auto_wsm.json"
WSM_OUTPUT_DIR = OUTPUT_DIR / "wsm"

wsm_manifest = resolve_request(WSM_REQUEST, SAMPLE_ROOT, caption_folder=CAPTION_FOLDER)
# Skill note (not in the source): the Transfer-Auto negative prompt is now this skill's assets/negative-prompt.txt; this was AUTO_ROOT.parents[3] / "cosmos3-nano-transfer-auto/inference/assets/negative-prompt.txt".
negative_path = AUTO_ROOT / "assets/negative-prompt.txt"
wsm_manifest["per_view_negative_prompt"] = negative_path.read_text().strip()
wsm_views = wsm_manifest["multiview"]["views"]
assert tuple(view["camera_key"] for view in wsm_views) == SAMPLE_CAMERAS
assert wsm_manifest["model_mode"] == "text2video" and not any("vision_path" in view for view in wsm_views)
wsm_controls = prepare_controls(
    wsm_views, SAMPLE_ROOT, WSM_OUTPUT_DIR / "prepared_controls",
    wsm_manifest["num_frames"], wsm_manifest["fps"],
)
wsm_request = copy.deepcopy(wsm_manifest)
for view, control in zip(wsm_request["multiview"]["views"], wsm_controls, strict=True):
    view["control_path"] = str(control)
wsm_form, wsm_uploads = prepare_request(wsm_request, SAMPLE_ROOT, MODEL_ID)
assert len(wsm_uploads) == len(SAMPLE_CAMERAS)
for camera_key, control in zip(SAMPLE_CAMERAS, wsm_controls):
    info = probe_video(control)
    print(f"{camera_key:28}  {info['frames']} frames, {info['width']}×{info['height']}, {float(info['fps']):g} FPS")
```

#### Preview the controls

The middle requested frame of each camera's WSM control.

```python
camera_frames(dict(zip(SAMPLE_CAMERAS, wsm_controls)), wsm_manifest["num_frames"] // 2,
              f"WSM controls — frame {wsm_manifest['num_frames'] // 2}")
```

### Generate

The example sends one `POST /v1/videos/sync` request and receives the finished MP4. The request carries:

- **Form fields:** the model, top-level caption, canvas `size`, `num_frames`, `fps`, `seed`, `num_inference_steps`, `guidance_scale`, and `flow_shift`, taken from the request file.
- **`input_references` (files):** the seven prepared controls, in camera order.
- **`extra_params` (JSON):** `wsm: {}`, the `multiview.views` list with each camera's key, caption, and `control_reference_index`, plus `resolution` and `aspect_ratio`.

The next cell prints the request, with the captions shortened.

```python
show_request(wsm_form, wsm_uploads)
```

Generate the camera videos. The first request also compiles the transformer, so it takes longer than later ones.

```python
with ExitStack() as stack, httpx.Client(base_url=VLLM_BASE_URL, headers=HEADERS, timeout=HTTP_TIMEOUT) as client:
    files = [("input_references", (path.name, stack.enter_context(path.open("rb")), "video/mp4")) for path in wsm_uploads]
    started = time.monotonic()
    wsm_video = post_sync_video(client, wsm_form, WSM_OUTPUT_DIR / "multiview_camera_major.mp4", files=files)
    wsm_elapsed = round(time.monotonic() - started, 1)
print(f"{wsm_elapsed:.1f}s  {wsm_video}")
```

### Validate and view

The response is one camera-major MP4: 201 frames of the first camera, then 201 of the second, through the seventh. Decode it completely, check its `7 × 201 = 1407` frames, FPS, and canvas, then split it into one labeled video per camera. The folder then holds:

```text
outputs/vllm_omni/auto/wsm/
  request.json                    # The exact form that was sent
  prepared_controls/              # The seven trimmed controls that were uploaded
  multiview_camera_major.mp4      # The server's response
  vision_viewNN_<camera>.mp4      # One video per camera
  sample_outputs.json             # Status, timing, uploads, cameras, and the decoded video properties
```

A rejected request leaves `response_error.txt` with the server's response instead of the MP4. Inspect the frames and playback below for control adherence, cross-camera consistency and visual quality, and see [Validate the rollout](transfer-auto-guide.md#validate-the-rollout).

```python
num_frames, fps = wsm_manifest["num_frames"], wsm_manifest["fps"]
wsm_info = validate_video(wsm_video, num_frames=len(SAMPLE_CAMERAS) * num_frames, fps=fps, width=832, height=480)
wsm_cameras = split_cameras(wsm_video, WSM_OUTPUT_DIR, SAMPLE_CAMERAS, num_frames, fps)
write_sample_outputs(WSM_OUTPUT_DIR, [{
    "name": wsm_manifest["name"],
    "status": "success",
    "video": wsm_video,
    "request": WSM_OUTPUT_DIR / "request.json",
    "uploads": wsm_uploads,
    "elapsed_seconds": wsm_elapsed,
    "frames_per_camera": num_frames,
    "cameras": wsm_cameras,
    **wsm_info,
}])
print(f"{wsm_info['frames']} camera-major frames, {wsm_info['width']}×{wsm_info['height']}, {wsm_info['fps']:g} FPS; "
      f"full decode passed; split into {len(wsm_cameras)} cameras of {num_frames} frames")
```

#### Compare with the controls

The middle frame of each generated camera, to compare with the control preview above. Check that lanes, road edges, and vehicles follow the controls, and that cameras agree where their views overlap.

```python
camera_frames(wsm_cameras, num_frames // 2, f"WSM-only vLLM-Omni output — frame {num_frames // 2}")
```

#### Play the videos

Each camera plays muted and looped. The players embed small re-encoded previews, so saved notebook outputs stay compact; the full-resolution MP4s remain in the output folder.

```python
display(HTML(video_row(wsm_cameras, width=320)))
```

## 5. Joint RGB and LiDAR example

### Prepare the inputs

The [joint request](../assets/daytime_scene/transfer_auto_joint.json) uses the same server, cameras, and captions and asks for the same **201 frames per camera at 30 FPS**, 832 × 480, seed 0, 35 steps, guidance 3, flow shift 10, and control guidance 1. Each camera adds the sample's RGB recording under `vision_path`; with `condition_video_as_image: true`, only its first decoded frame conditions generation. The request also adds a numeric LiDAR control with `return_output: true`.

The joint preparation cell converts the published `daytime_scene/lidar/control.tar` to a local `prepared/lidar_control.safetensors` file for vLLM-Omni. Native Framework reads the archive directly; no converted tensor or duplicate media is bundled. See the [format notes](../assets/daytime_scene/README.md).

The next cell checks the first 67 sweeps one at a time, decodes each RGB condition's first frame, and trims the WSM controls to 201 frames. The RGB videos are uploaded unchanged. It then builds the multipart request: camera *i* gets `control_reference_index` 2*i* and `vision_reference_index` 2*i* + 1, and the LiDAR tensor is upload 14.

The preparation cell also loads the inference skill’s recommended per-view negative prompt. Guidance 3 controls caption strength; shift 10 sets the denoising timestep distribution. Use these settings as the starting recipe for your inputs.

```python
LIDAR_REQUEST = AUTO_ROOT / "assets" / "daytime_scene" / "transfer_auto_joint.json"
LIDAR_OUTPUT_DIR = OUTPUT_DIR / "lidar"
LIDAR_FPS = 10.0
LIDAR_POLL_TIMEOUT = 3600  # Seconds to wait for the asynchronous job

lidar_manifest = resolve_request(
    LIDAR_REQUEST, SAMPLE_ROOT, caption_folder=CAPTION_FOLDER,
    lidar_output=LIDAR_OUTPUT_DIR / "prepared" / "lidar_control.safetensors",
)
# Skill note (not in the source): the Transfer-Auto negative prompt is now this skill's assets/negative-prompt.txt; this was AUTO_ROOT.parents[3] / "cosmos3-nano-transfer-auto/inference/assets/negative-prompt.txt".
negative_path = AUTO_ROOT / "assets/negative-prompt.txt"
lidar_manifest["per_view_negative_prompt"] = negative_path.read_text().strip()
lidar_views = lidar_manifest["multiview"]["views"]
assert tuple(view["camera_key"] for view in lidar_views) == SAMPLE_CAMERAS
assert lidar_manifest["multiview"]["condition_video_as_image"] is True
assert lidar_manifest["lidar"]["return_output"] is True
assert lidar_manifest["resolution"] == "480" and lidar_manifest["aspect_ratio"] == "16,9"
lidar_sweeps = round(lidar_manifest["num_frames"] * LIDAR_FPS / lidar_manifest["fps"])

lidar_input = require_file(lidar_manifest["lidar"]["control_path"])
lidar_input_info = validate_lidar(lidar_input, lidar_sweeps)
validate_rgb_conditions(lidar_views, SAMPLE_ROOT)
lidar_controls = prepare_controls(
    lidar_views, SAMPLE_ROOT, LIDAR_OUTPUT_DIR / "prepared_controls",
    lidar_manifest["num_frames"], lidar_manifest["fps"],
)
lidar_request = copy.deepcopy(lidar_manifest)
for view, control in zip(lidar_request["multiview"]["views"], lidar_controls, strict=True):
    view["control_path"] = str(control)
lidar_form, lidar_uploads = prepare_request(lidar_request, SAMPLE_ROOT, MODEL_ID)
lidar_extra = json.loads(lidar_form["extra_params"])
assert len(lidar_uploads) == 2 * len(SAMPLE_CAMERAS) + 1
assert lidar_extra["lidar"]["control_reference_index"] == 2 * len(SAMPLE_CAMERAS)
assert [
    index for view in lidar_extra["multiview"]["views"]
    for index in (view["control_reference_index"], view["vision_reference_index"])
] == list(range(2 * len(SAMPLE_CAMERAS)))
rgb_conditions = {view["camera_key"]: Path(view["vision_path"]) for view in lidar_views}
print(f"LiDAR input: {lidar_input_info['shape']} {lidar_input_info['dtype']}; using {lidar_sweeps} sweeps at {LIDAR_FPS:g} Hz")
print(f"Uploads: {len(SAMPLE_CAMERAS)} WSM controls + {len(SAMPLE_CAMERAS)} RGB conditions + 1 LiDAR tensor, "
      f"{sum(path.stat().st_size for path in lidar_uploads) / 2**20:,.0f} MiB")
```

#### Preview the inputs

The first frame of each RGB condition, which is all that generation uses, followed by the range and validity of the LiDAR control's first sweep. The control is rendered from the scene's HD map rather than measured, so it is sparser than a recorded sweep and its intensity channel equals its validity. The range image spans 1800 azimuth columns and 128 beams.

```python
from safetensors import safe_open

camera_frames(rgb_conditions, 0, "RGB conditions — frame 0")
with safe_open(lidar_input, framework="numpy") as handle:
    first_sweep = handle.get_slice("frames")[:, 0]
fig, axes = plt.subplots(2, 1, figsize=(16, 4.5), layout="constrained")
for ax, channel, label, cmap in zip(axes, (0, 2), ("Range (m)", "Validity"), ("turbo", "gray")):
    image = ax.imshow(first_sweep[channel], aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_title(f"LiDAR control, sweep 0 — {label.lower()}", fontsize=9)
    ax.set_ylabel("Beam")
    fig.colorbar(image, ax=ax, label=label, pad=0.01)
axes[-1].set_xlabel("Azimuth column")
display(fig)
plt.close(fig)
```

### Generate

Numeric LiDAR output needs the asynchronous Videos API; the synchronous endpoint rejects `return_output: true`. The example submits one `POST /v1/videos` job and polls `GET /v1/videos/{id}` until it completes. The request carries:

- **Form fields:** as in the WSM-only example.
- **`input_references` (files):** 15 uploads: the WSM control and RGB condition of each camera, in camera order, then the LiDAR tensor.
- **`extra_params` (JSON):** `wsm: {}`; `multiview.views` with each camera's key, caption, and reference indexes; `multiview.condition_video_as_image: true`; `lidar` with its reference index and `return_output: true`; plus `resolution`, `aspect_ratio`, and `control_guidance`.

The next cell prints the request, with the captions shortened.

```python
show_request(lidar_form, lidar_uploads)
```

Submit the job and download the camera video. The cell saves the latest job status as `job.json` and the camera-major MP4 from `/content`. A failed job or HTTP error leaves `response_error.txt`. If polling times out, the job stays on the server at the printed URL; rerunning the cell submits a new job.

```python
LIDAR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(LIDAR_OUTPUT_DIR / "request.json").write_text(json.dumps(lidar_form, indent=2, ensure_ascii=False) + "\n")
(LIDAR_OUTPUT_DIR / "response_error.txt").unlink(missing_ok=True)
with httpx.Client(base_url=VLLM_BASE_URL, headers=HEADERS, timeout=HTTP_TIMEOUT) as client:
    started = time.monotonic()
    lidar_job = submit_joint(client, lidar_form, lidar_uploads, LIDAR_OUTPUT_DIR, timeout=LIDAR_POLL_TIMEOUT)
    lidar_elapsed = round(time.monotonic() - started, 1)
lidar_video = lidar_job["video"]
print(f"Job {lidar_job['job_id']} completed in {lidar_elapsed:.1f}s; camera video: {lidar_video}")
```

Download the generated LiDAR. `GET /v1/videos/{id}/lidar` returns a numeric `.safetensors` file, separate from the camera MP4. The cell streams it to disk and saves the job's LiDAR descriptor beside it as `.lidar.json`. It reads the job from `job.json`, so you can rerun it to recover the file without generating again.

```python
job = json.loads((LIDAR_OUTPUT_DIR / "job.json").read_text())
with httpx.Client(base_url=VLLM_BASE_URL, headers=HEADERS, timeout=HTTP_TIMEOUT) as client:
    lidar_output, lidar_descriptor = download_lidar(
        client, job, LIDAR_OUTPUT_DIR / "multiview_camera_major.lidar.safetensors"
    )
print(f"{lidar_output.stat().st_size / 2**20:,.0f} MiB  {lidar_output}")
```

### Validate and view

Decode the camera-major MP4 and check its `7 × 201 = 1407` frames, FPS, and canvas, then split it into one video per camera. Check the generated LiDAR on its own 10 Hz timeline: exactly 67 FP32 sweeps of shape `[3,67,128,1800]` that start at time zero, with valid physical values and embedded metadata that matches the job descriptor. The folder then holds:

```text
outputs/vllm_omni/auto/lidar/
  request.json                                # The exact form that was sent
  prepared_controls/                          # The seven trimmed WSM controls that were uploaded
  job.json                                    # The final job status, including the LiDAR descriptor
  multiview_camera_major.mp4                  # The camera-major video from /content
  vision_viewNN_<camera>.mp4                  # One video per camera
  multiview_camera_major.lidar.safetensors    # The numeric LiDAR from /lidar
  multiview_camera_major.lidar.json           # Its descriptor
  sample_outputs.json                         # Status, timing, uploads, cameras, LiDAR, and the decoded video properties
```

```python
num_frames, fps = lidar_manifest["num_frames"], lidar_manifest["fps"]
lidar_info = validate_video(lidar_video, num_frames=len(SAMPLE_CAMERAS) * num_frames, fps=fps, width=832, height=480)
lidar_output_info = validate_lidar_output(lidar_output, lidar_descriptor, lidar_sweeps, lidar_fps=LIDAR_FPS)

lidar_cameras = split_cameras(lidar_video, LIDAR_OUTPUT_DIR, SAMPLE_CAMERAS, num_frames, fps)
write_sample_outputs(LIDAR_OUTPUT_DIR, [{
    "name": lidar_manifest["name"],
    "status": "success",
    "job_id": job["id"],
    "video": lidar_video,
    "request": LIDAR_OUTPUT_DIR / "request.json",
    "uploads": lidar_uploads,
    "elapsed_seconds": lidar_elapsed,
    "frames_per_camera": num_frames,
    "cameras": lidar_cameras,
    "lidar": lidar_output,
    "lidar_descriptor": lidar_descriptor,
    **lidar_info,
}])
print(f"{lidar_info['frames']} camera-major frames, {lidar_info['width']}×{lidar_info['height']}, "
      f"{lidar_info['fps']:g} FPS; full decode passed; split into {len(lidar_cameras)} cameras")
print(f"LiDAR: {lidar_output_info['shape']} {lidar_output_info['dtype']} at {lidar_descriptor['fps']:g} Hz; "
      "values and metadata passed")
```

#### Compare with the RGB conditions

The first generated frame of each camera, to compare with its RGB condition above, and the final frame. The first frame should closely follow the condition; later frames should keep the scene consistent with it while following the controls.

```python
camera_frames(lidar_cameras, 0, "Joint vLLM-Omni output — frame 0")
camera_frames(lidar_cameras, num_frames - 1, f"Joint vLLM-Omni output — frame {num_frames - 1}")
```

#### Play the videos

Each camera plays muted and looped, as compact previews.

```python
display(HTML(video_row(lidar_cameras, width=320)))
```

#### Visualize a LiDAR sweep

Choose `sweep_index` and rerun this cell to view a different sweep. It reads only that sweep from the generated `frames[3,T,128,1800]` tensor and saves a PNG preview. The range and intensity panels show the original data; the top-down and 3D panels use metric XYZ, with X forward, Y left, and Z up. Each sweep stays in its own sensor frame; scans are not accumulated or corrected for vehicle motion.

Framework's shared [Pandar128 geometry](../../../../cosmos_framework/model/generator/tokenizers/lidar/geometry.py) supplies the 128 beam elevations, matching the model's geometry helpers. Azimuth follows the artifact's projection metadata, defaulting to +180° toward −180° with no duplicate endpoint.

Set `smooth_geometry=False` to view unfiltered points. With smoothing enabled, the geometry panels use the examples' validity-aware bilateral filter over five azimuth columns within each beam row, including across the azimuth seam. It can move ranges and drop unsupported returns; it affects only the preview. Height colors share the displayed sweep's 2nd/98th percentile bounds. Both geometry views cover `bev_radius_m` in X and Y; the point limit applies only to the 3D plot. All-invalid sweeps display empty geometry panels.

```python
from safetensors import safe_open
from helpers.lidar_visualization import draw_lidar_sweep, prepare_lidar_sweep

sweep_index = 0  # 0..66 for the default 67-sweep output
smooth_geometry = True
max_plot_points = 20_000
bev_radius_m = 80.0
bev_size_px = 512

with safe_open(lidar_output, framework="numpy") as handle:
    frames = handle.get_slice("frames")
    if not 0 <= sweep_index < frames.get_shape()[1]:
        raise ValueError("sweep_index is outside the generated sequence.")
    sweep = frames[:, sweep_index]
preview = prepare_lidar_sweep(sweep, lidar_descriptor, smooth_geometry=smooth_geometry)
figure = draw_lidar_sweep(
    preview, lidar_descriptor, sweep_index, smooth_geometry=smooth_geometry,
    max_plot_points=max_plot_points, bev_radius_m=bev_radius_m, bev_size_px=bev_size_px,
)
preview_path = lidar_output.with_name(f"lidar_sweep_{sweep_index:03d}.png")
figure.savefig(preview_path, dpi=140)
display(figure)
plt.close(figure)
print(f"Saved {preview_path}; {preview.valid.sum():,} original returns, {len(preview.xyz):,} preview points")
```

#### Visualize LiDAR as video

Render every generated sweep into a four-panel video: original range, original intensity, calibrated top-down view, and a height-colored 3D point cloud. The MP4 uses the descriptor's LiDAR FPS, so the 67 sweeps at 10 Hz play for 6.7 seconds. Each sweep stays in its sensor frame, without accumulation or vehicle-motion correction.

The local [renderer](../scripts/helpers/lidar_visualization.py) keeps height colors and camera bounds fixed across the clip. It processes one sweep at a time on the CPU and streams frames to FFmpeg, saving `lidar_preview.mp4` beside the numeric artifact. Rendering may take a few minutes; progress is printed. No Open3D installation is needed.

Set `video_smooth_geometry=False` for unfiltered geometry. `video_max_plot_points` limits only the 3D panel. Lower `video_dpi` to reduce rendering cost and file size; 80 gives 1280 × 800. Smoothing affects only the preview. The player embeds a compact re-encoded preview; the full-resolution MP4 remains in the output folder.

```python
from helpers.lidar_visualization import render_lidar_video

video_smooth_geometry = True
video_max_plot_points = 20_000
video_bev_radius_m = 80.0
video_bev_size_px = 512
video_dpi = 80

lidar_preview_video = render_lidar_video(
    lidar_output, lidar_descriptor, lidar_output.with_name("lidar_preview.mp4"),
    smooth_geometry=video_smooth_geometry,
    max_plot_points=video_max_plot_points,
    bev_radius_m=video_bev_radius_m,
    bev_size_px=video_bev_size_px,
    dpi=video_dpi,
)
display(HTML(video_row({"Generated LiDAR": lidar_preview_video}, width=832)))
```

## 6. Next steps

- **Use your own scene:** copy a request file from [`assets/daytime_scene/`](../assets/daytime_scene), point its paths at your data, then set `WSM_REQUEST` or `LIDAR_REQUEST` to the copy and `SAMPLE_ROOT` to the folder its paths are relative to. List any of the checkpoint's eleven cameras in its camera order, `CAMERA_ORDER` in [`helpers/auto_helpers.py`](../scripts/helpers/auto_helpers.py), each with a WSM control and a caption file or inline `prompt`. The [WSM generation notebook](generate-wsm.md) writes a native request for the Framework notebooks; serving uses the templates above. For the joint example, add a `vision_path` per camera and a numeric LiDAR control; [`helpers/lidar_rangemap.py`](../scripts/helpers/lidar_rangemap.py) converts a sparse HD-map range-map archive.
- **Validate further:** apply the [shared Simulation checklist](../../cosmos3-inference-simulation/references/simulation-readme.md#validate-a-rollout) and the [Transfer-Auto checks](transfer-auto-guide.md#validate-the-rollout). If a request fails, see [vLLM-Omni troubleshooting](transfer-auto-guide.md#vllm-omni-troubleshooting).
- **Other runtimes:** run the same media with the published native JSONL in the [Cosmos Framework notebook](transfer-auto-cosmos-framework.md), or serve them through SGLang with the [SGLang notebook](transfer-auto-sglang.md). For the autoregressive student of this model, see the [Sim-Auto notebook](sim-auto-cosmos-framework.md).

For custom caption filenames, pass `caption_folder=None` to the request helper to preserve your paths.
