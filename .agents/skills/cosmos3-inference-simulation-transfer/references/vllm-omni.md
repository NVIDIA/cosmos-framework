<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3-Nano-Sim-Transfer with vLLM-Omni

Cosmos3-Nano-Sim-Transfer generates a **480p video from a caption and a control video**. Each sample uses one of four control modalities:

- **Depth** guides scene geometry and depth structure.
- **Edge** guides object boundaries and outlines.
- **Blur** guides coarse appearance, layout, and motion.
- **Segmentation** (`seg`) guides the layout of regions in the scene.

The caption describes the desired scene and appearance, while the control video guides its spatial structure and motion. The distilled causal model generates latents autoregressively with **four denoising steps**, retaining control and generated-latent K/V history across the rollout.

This notebook runs the four published HF `example2` controls through one vLLM-Omni server at **121 frames, 832 × 480, and 30 FPS**. It checks the output frame count, dimensions and FPS, then displays each generated video beside its control.

> [!CAUTION]
> Accept the model license and authenticate with Hugging Face before downloading the checkpoint and sample data.

| Component                          | Version                                                                                                 |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Server image                       | `vllm/vllm-omni:cosmos3`                                                                                |
| `nvidia/Cosmos3-Nano-Sim-Transfer` | [`main`](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer/tree/main)                             |
| HF `example2` assets               | [`main`](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer/tree/main/assets/sample_data/example2) |

Steps 1 and 2 run on the server host. Steps 3 to 6 run in this notebook.

> **Skill note (not in the source):** this file is the cookbook notebook `run_transfer_with_vllm_omni.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/simulation/transfer/run_transfer_with_vllm_omni.ipynb`, cosmos-framework `main` 45fecab) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced as `python`, outputs omitted (the committed notebook has none). Edits are limited to three kinds: link targets that pointed at files this migration moved were re-pointed; path anchors in code that the move invalidated were replaced, each marked in place with a `Skill note (not in the source)` comment; and location text that named the retired cookbook folder (default output folders, where to run from) now names this skill's folder, each marked by a skill note. The changes are recorded in `docs/cookbook_to_skill_migration.md`.

## 1. Prerequisites

**Server host:** Linux with one NVIDIA GPU, Docker with the NVIDIA Container Toolkit, and a Hugging Face token with read access to `nvidia/Cosmos3-Nano-Sim-Transfer`.

**Notebook client:** Python 3.10 or newer. In the notebook kernel environment, install:

```bash
pip install httpx huggingface_hub pillow matplotlib ipython
```

Install FFmpeg on the client host; the package also includes FFprobe. On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

On macOS with Homebrew, use `brew install ffmpeg`. Verify that both `ffmpeg -version` and `ffprobe -version` work.

**Inputs:** The notebook downloads the saved `example2` prompt and four precomputed controls from the same Hugging Face repository as the model. Later runs reuse the Hugging Face cache.

> [!IMPORTANT]
> **The server reads each control video from its own filesystem.** A Sim-Transfer request names the control by absolute `control_path`; it does not upload the file. The Docker command below mounts the Cosmos checkout at its original absolute path. Run the notebook from that checkout on the server host, or use the same absolute checkout path on both hosts.

## 2. Start the server

All Simulation vLLM-Omni notebooks use the `vllm/vllm-omni:cosmos3` Docker image. Use a Linux GPU server with Docker and the NVIDIA Container Toolkit. Export `HF_TOKEN` with read access to the model, then pull the image:

```bash
docker pull vllm/vllm-omni:cosmos3
```

Run this shell command from the Cosmos Framework checkout root to mount the control videos at the same absolute paths the notebook sends:

```bash
export COSMOS3_REPO="$(pwd -P)"
```

Start the container and leave it running while using the notebook:

```bash
docker run --rm --gpus '"device=0"' --ipc=host \
  -e HF_TOKEN \
  -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -v "${COSMOS3_REPO}:${COSMOS3_REPO}:ro" \
  -p 127.0.0.1:8000:8000 \
  -w /app/vllm-omni \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano-Sim-Transfer \
    --revision main \
    --omni --model-class-name Cosmos3NanoSimTransferPipeline \
    --deploy-config vllm_omni/deploy/cosmos3_nano_sim_transfer.yaml \
    --stage-overrides '{"0":{"enforce_eager":false,"diffusion_compile_granularity":"regional","diffusion_compile_dynamic":true,"vae_fast_path":"channels_last","vae_encode_fast_path":"channels_last","model_config":{"guardrails":false,"use_cuda_graphs":true,"overlap_vae_decode":true,"clean_commit_mode":"framewise"}}}' \
    --host 0.0.0.0 --port 8000 --init-timeout 1800
```

The container reuses the host Hugging Face cache and listens on all container interfaces. Docker publishes port 8000 on the host loopback interface, matching `http://127.0.0.1:8000`. For a remote notebook, forward the port with `ssh -L 8000:127.0.0.1:8000 <server-host>`, or publish it on a reachable interface with `-p 8000:8000` and set `VLLM_BASE_URL`. The working directory selects the deployment YAML files bundled in the image.

| Setting                                                                       | Effect                                                                                                                              |
| ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800`                                           | Lets one synchronous request run for up to 30 minutes, instead of the 600-second default.                                           |
| `--revision`                                                                  | Loads the checkpoint from `main`. The first start downloads it into the Hugging Face cache.                                         |
| `--deploy-config`                                                             | Selects the Sim-Transfer pipeline.                                                                                                  |
| `enforce_eager`, `diffusion_compile_granularity`, `diffusion_compile_dynamic` | Compile the repeated transformer blocks with dynamic shapes instead of running eagerly.                                             |
| `guardrails`                                                                  | Disables the Cosmos3 text and video safety guardrails for this example server.                                                      |
| `use_cuda_graphs`                                                             | Replays captured CUDA graphs to reduce kernel-launch overhead.                                                                      |
| `vae_fast_path`, `vae_encode_fast_path`                                       | Use fused channels-last VAE decoder and encoder kernels. These kernels change floating-point rounding relative to the standard VAE. |
| `overlap_vae_decode`                                                          | Decodes finished latent frames while generation continues.                                                                          |
| `clean_commit_mode`                                                           | Commits generated history to the K/V cache frame by frame instead of once per chunk.                                                |
| `-p 127.0.0.1:8000:8000`, `--host 0.0.0.0`                                    | Exposes the container server on host loopback port 8000. Use the port-forwarding instructions above for a remote notebook.          |

## 3. Set up the client

Run the notebook from anywhere inside `.agents/skills`, or set `COSMOS_COOKBOOK_ROOT` to it. These optional environment variables override the defaults:

| Variable             | Default                                                                           | Purpose                                                  |
| -------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `VLLM_BASE_URL`      | `http://127.0.0.1:8000`                                                           | Server address                                           |
| `VLLM_API_KEY`       | unset                                                                             | Bearer token, if the server was started with `--api-key` |
| `COSMOS3_OUTPUT_DIR` | `.agents/skills/cosmos3-inference-simulation-transfer/outputs/vllm_omni/transfer` | Output folder                                            |

> **Skill note (not in the source):** the `COSMOS3_OUTPUT_DIR` default in the table above was `.agents/skills/cookbooks/cosmos3/simulation/outputs/vllm_omni/transfer` in the source; the default output folder moved into this skill with the cookbook.

The next cell sets the configuration. The cell after it defines the client helpers for server readiness, requests, and output validation; they need no GPU libraries.

```python
from pathlib import Path
import json
import os
import time
from uuid import uuid4

import httpx
import matplotlib.pyplot as plt
from IPython.display import HTML, display


def find_transfer_root() -> Path:
    configured = os.environ.get("COSMOS_COOKBOOK_ROOT")
    start = Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()
    for root in [start] if configured else [start, *start.parents]:
        # Skill note (not in the source): the Transfer cookbook folder is now this skill's folder; the candidates were root, root / "transfer", root / "cookbooks" / "cosmos3" / "simulation" / "transfer".
        for candidate in (root, root / "cosmos3-inference-simulation-transfer"):
            # Skill note (not in the source): the guide transfer.md is now references/transfer-guide.md in this skill; the marker was candidate / "transfer.md".
            if (candidate / "references" / "transfer-guide.md").is_file():
                return candidate
    raise FileNotFoundError("Run inside .agents/skills or set COSMOS_COOKBOOK_ROOT to it.")


TRANSFER_ROOT = find_transfer_root()
# Skill note (not in the source): the outputs folder is now inside this skill's folder; the source line was SIMULATION_ROOT = TRANSFER_ROOT.parent.
SIMULATION_ROOT = TRANSFER_ROOT
MODEL_ID = "nvidia/Cosmos3-Nano-Sim-Transfer"
MODEL_REVISION = "main"
ASSET_REVISION = "main"
VLLM_OMNI_IMAGE = "vllm/vllm-omni:cosmos3"
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY")
HEADERS = {"Authorization": f"Bearer {VLLM_API_KEY}"} if VLLM_API_KEY else {}
HTTP_TIMEOUT = 1800 + 60  # The server's synchronous timeout, plus a margin for the response.
OUTPUT_DIR = Path(
    os.environ.get("COSMOS3_OUTPUT_DIR", SIMULATION_ROOT / "outputs" / "vllm_omni" / "transfer")
).expanduser().resolve()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Cookbook:", TRANSFER_ROOT)
print("Server:", VLLM_BASE_URL, "(with API key)" if VLLM_API_KEY else "")
print("Model:", MODEL_ID, "@", MODEL_REVISION)
print("Inputs:", MODEL_ID, "@", ASSET_REVISION)
print("Server image:", VLLM_OMNI_IMAGE)
print("Outputs:", OUTPUT_DIR)
```

```python
# Client helpers for requests, media inspection, and playback.
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

def require_media_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
    if missing:
        raise RuntimeError(f"Install FFmpeg and add {' and '.join(missing)} to PATH.")


def require_file(path: Path) -> Path:
    """Resolve a nonempty input file."""
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty input: {path}")
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

### Verify the server

Wait for the server to finish starting, then confirm that it serves the Sim-Transfer model. The cell polls for up to 30 minutes, so you can run it while the server is still loading.

```python
with httpx.Client(base_url=VLLM_BASE_URL, headers=HEADERS) as client:
    served = wait_for_model(client, MODEL_ID, timeout=1800)
print("Server ready; serving", served)
```

## 4. Prepare the inputs

The notebook uses the four published, precomputed HF `example2` controls directly. Each request uses **121 frames, 832 × 480, 30 FPS, and seed 2025**.

| Control      | Precomputed HF asset        | What it preserves                     |
| ------------ | --------------------------- | ------------------------------------- |
| Depth        | `example2_depth.mp4`        | Scene geometry and depth structure    |
| Edge         | `example2_edge.mp4`         | Object boundaries and outlines        |
| Blur         | `example2_blur.mp4`         | Coarse appearance, layout, and motion |
| Segmentation | `example2_segmentation.mp4` | Semantic region layout                |

The next cell downloads the files under `cosmos3-inference-simulation-transfer/outputs/`, which the Docker command exposes at the same absolute path, compacts the saved appearance prompt, and resolves the control paths. This recipe uses the published precomputed hints. To derive edge or blur from RGB instead, follow the [Cosmos Framework workflow](transfer-guide.md#prepare-the-input).

> **Skill note (not in the source):** the folder named above was `simulation/outputs/` in the source; the default output folder moved into this skill with the cookbook.

```python
from huggingface_hub import snapshot_download

CONTROLS = ("depth", "edge", "blur", "seg")
NUM_FRAMES, FPS, WIDTH, HEIGHT, SEED = 121, 30, 832, 480, 2025
sample_root = Path(snapshot_download(
    MODEL_ID,
    revision=ASSET_REVISION,
    allow_patterns=["assets/sample_data/example2/*"],
    local_dir=SIMULATION_ROOT / "outputs/vllm_omni/transfer/inputs",
)) / "assets/sample_data/example2"
prompt = json.dumps(json.loads((sample_root / "example2_upsampled_prompt.json").read_text()))
files = {
    "depth": "example2_depth.mp4",
    "edge": "example2_edge.mp4",
    "blur": "example2_blur.mp4",
    "seg": "example2_segmentation.mp4",
}
cases = {
    hint: {
        "name": f"example2_{hint}", "prompt": prompt,
        "control": require_file(sample_root / filename),
        "num_frames": NUM_FRAMES, "fps": FPS, "width": WIDTH, "height": HEIGHT,
        "seed": SEED, "output_dir": OUTPUT_DIR / hint,
    }
    for hint, filename in files.items()
}
for hint, case in cases.items():
    print(f"{hint:5}  {case['num_frames']} frames, {case['width']}×{case['height']}, "
          f"{case['fps']} FPS, seed {case['seed']}; caption {len(prompt):,} characters")
```

### Preview the controls

The first, middle, and final requested frames of each precomputed control video.

```python
fig, axes = plt.subplots(len(cases), 3, figsize=(12, 2.6 * len(cases)), layout="constrained", squeeze=False)
for row, (hint, case) in zip(axes, cases.items()):
    last = case["num_frames"] - 1
    for ax, index in zip(row, (0, last // 2, last)):
        ax.imshow(read_frame(case["control"], index))
        ax.set_title(f"{hint} control — frame {index}", fontsize=9)
        ax.axis("off")
display(fig)
plt.close(fig)
```

## 5. Generate

Each example sends one `POST /v1/videos/sync` request and receives the finished MP4. The request carries:

- **Form fields:** the model, saved appearance prompt, canvas `size`, `num_frames`, `fps`, and `seed`.
- **`extra_params` (JSON):** one named hint with its absolute `control_path`, the 480p canvas, `max_prompt_tokens=4096`, and a 30-latent-frame history window with three attention sinks.
- **Session fields:** `session_id`, `reset`, and `close_session` give each example a fresh generation session and release it afterward. The model stays loaded.

The next cell builds the requests and prints the depth request, with its caption shortened.

```python
STREAMING = {"kv_cache_inference_size": 30, "attention_sink_size": 3}  # vLLM-Omni history window and attention sinks
MAX_PROMPT_TOKENS = 4096  # Room for the full upsampled captions.


def build_request(hint: str, case: dict) -> dict:
    extra_params = {
        "session_id": uuid4().hex,
        "reset": True,
        "close_session": True,
        "resolution": "480",
        "aspect_ratio": "16,9",
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        **STREAMING,
        hint: {"control_path": str(case["control"])},
    }
    return {
        "model": MODEL_ID,
        "prompt": case["prompt"],
        "size": f"{case['width']}x{case['height']}",
        "num_frames": str(case["num_frames"]),
        "fps": str(case["fps"]),
        "seed": str(case["seed"]),
        "extra_params": json.dumps(extra_params),
    }


example = build_request("depth", cases["depth"])
example.update(prompt=example["prompt"][:80] + " ...", extra_params=json.loads(example["extra_params"]))
print(json.dumps(example, indent=2))
```

Generate the examples listed in `RUN`. Set it to a subset, such as `("depth",)`, to run fewer. The first request also compiles the transformer and captures CUDA graphs, so it takes longer than the others.

```python
RUN = CONTROLS

results = {}
with httpx.Client(base_url=VLLM_BASE_URL, headers=HEADERS, timeout=HTTP_TIMEOUT) as client:
    for hint in RUN:
        case = cases[hint]
        started = time.monotonic()
        video = post_sync_video(client, build_request(hint, case), case["output_dir"] / f"{hint}.mp4")
        results[hint] = {"video": video, "elapsed_seconds": round(time.monotonic() - started, 1)}
        print(f"{hint:5}  {results[hint]['elapsed_seconds']:7.1f}s  {video}")
```

## 6. Validate and view

Decode each generated video completely and check its frame count, FPS, and canvas. Each example's folder then holds:

```text
outputs/vllm_omni/transfer/<control>/
  request.json          # The exact form that was sent
  <control>.mp4         # The generated video
  sample_outputs.json   # Status, timing, and the decoded video properties
```

A rejected request leaves `response_error.txt` with the server's response instead of the MP4. Inspect the frames and playback below for control adherence and temporal consistency; see [Validate the rollout](transfer-guide.md#validate-the-rollout).

```python
for hint, result in results.items():
    case = cases[hint]
    info = validate_video(
        result["video"], num_frames=case["num_frames"], fps=case["fps"],
        width=case["width"], height=case["height"],
    )
    write_sample_outputs(case["output_dir"], [{
        "name": case["name"],
        "status": "success",
        "control": hint,
        "video": result["video"],
        "request": case["output_dir"] / "request.json",
        "elapsed_seconds": result["elapsed_seconds"],
        **info,
    }])
    print(f"{hint:5}  {info['frames']} frames, {info['width']}×{info['height']}, {info['fps']:g} FPS; full decode passed")
```

### Inspect control adherence

Each figure shows the control and generated output at the first, middle, and final frames. Compare geometry, object boundaries, and motion across the rollout.

```python
for hint, result in results.items():
    case = cases[hint]
    last = case["num_frames"] - 1
    rows = {"control": case["control"], "vLLM-Omni": result["video"]}
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 2.6 * len(rows)), layout="constrained", squeeze=False)
    for row, (label, path) in zip(axes, rows.items()):
        for ax, index in zip(row, (0, last // 2, last)):
            ax.imshow(read_frame(path, index))
            ax.set_title(f"{hint} {label} — frame {index}", fontsize=9)
            ax.axis("off")
    display(fig)
    plt.close(fig)
```

### Play the videos

Each control plays beside its generated video, muted and looped over the requested frames. The players embed small re-encoded previews, so saved notebook outputs stay compact; the full-resolution MP4s remain in each example's folder.

```python
for hint, result in results.items():
    case = cases[hint]
    row = video_row({"Control": case["control"], "vLLM-Omni": result["video"]}, frames=case["num_frames"])
    display(HTML(f"<h4>{hint}</h4>" + row))
```

## 7. Next steps

- **Use your own control:** replace one path in `files` or build a case with your caption and absolute control path. To derive edge or blur from RGB, follow the [Cosmos Framework preprocessing](transfer-guide.md#prepare-the-input).
- **Validate further:** apply the [shared Simulation checklist](../../cosmos3-inference-simulation/references/simulation-readme.md#validate-a-rollout) and the [Sim-Transfer checks](transfer-guide.md#validate-the-rollout) for control adherence and temporal consistency.
- **Cosmos Framework:** use the native [Hugging Face example2 recipe](transfer-guide.md#cosmos-framework) for direct-resize inference and locally computed edge/blur controls.
- **SGLang:** the [SGLang notebook](sglang.md) runs the same four `example2` controls through SGLang's asynchronous Videos API.
