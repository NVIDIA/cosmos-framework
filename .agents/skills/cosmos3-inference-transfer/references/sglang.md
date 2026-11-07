<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Nano Transfer with SGLang

This notebook calls an already-running SGLang Cosmos3 server through the OpenAI-compatible video API. It reuses the checked-in transfer control assets and specs from this cookbook, then sends edge, blur, depth, segmentation, and world-scenario-map transfer requests to `POST /v1/videos`.

The runnable examples use one control at a time. SGLang also accepts unweighted multi-control requests by passing a list of control videos in the API request. Per-hint `weight` is supported only by Cosmos Framework.

The notebook does not modify the SGLang source tree.

> **Skill note (not in the source):** this file is the cookbook notebook `run_video_transfer_with_sglang.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/generator/transfer/run_video_transfer_with_sglang.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

## 1. Prerequisites

Start a SGLang Cosmos3 server before running the request cells. The examples assume `${COSMOS3_WORKDIR}` points to cosmos repo.

Transfer controls are available from SGLang `main` and the released `lmsysorg/sglang:dev` container.

```bash
export SGLANG_HF_CACHE="${SGLANG_HF_CACHE:-$HOME/.cache/sglang-huggingface}"
mkdir -p "$SGLANG_HF_CACHE"
chmod 777 "$SGLANG_HF_CACHE"

export COSMOS3_WORKDIR="$PWD"

docker run --runtime nvidia --gpus all \
  -v $SGLANG_HF_CACHE:/root/.cache/huggingface \
  -v "${COSMOS3_WORKDIR}:${COSMOS3_WORKDIR}" \
  -p 30000:30000 \
  --ipc=host \
  lmsysorg/sglang:dev \
  sglang serve \
  --model-path nvidia/Cosmos3-Nano \
  --host 0.0.0.0
```

Generator guardrails are on by default and require access to the gated `nvidia/Cosmos-1.0-Guardrail` repository. When guardrails are enabled, also set and pass `HF_TOKEN` to the container; it must be a Hugging Face token authorized for that repository. To disable guardrails server-wide, set `SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` when launching server; alternatively, set `"use_guardrails": false` in these sample requests.

### FP8

To serve Cosmos3-Nano in FP8, you can either:

- Download the prebuilt FP8 checkpoint from Hugging Face: `nvidia/Cosmos3-Nano-FP8`.
- Quantize a checkpoint yourself by following the [quantization guide](../../cookbooks/cosmos3/quantization/README.md).

To serve the prebuilt checkpoint, pass its Hugging Face ID to `--model-path`:

```bash
sglang serve \
  --model-path nvidia/Cosmos3-Nano-FP8 \
  --host 0.0.0.0
```

To serve a checkpoint you quantized yourself, pass its local directory to `--model-path`. If the server runs in Docker, mount that directory and use its path inside the container.

```bash
sglang serve \
  --model-path /path/to/fp8/checkpoint \
  --host 0.0.0.0
```

## 2. Configure Paths and Endpoints

Run this cell from anywhere inside `.agents/skills`. It resolves local assets, output paths, the SGLang endpoint, and repo-local `control_path` values.

```python
from pathlib import Path
import base64
import html
import json
import os
import shutil
import subprocess
import time
from IPython.display import HTML, display

def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "cookbooks" / "cosmos3").is_dir():
            return path
    return start


COSMOS_ROOT = find_repo_root(Path.cwd().resolve())
TRANSFER_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3" / "generator" / "transfer"
SPECS_DIR = TRANSFER_ROOT / "specs"
ASSETS_DIR = TRANSFER_ROOT / "assets"
OUTPUT_ROOT = Path(
    os.environ.get("COSMOS3_TRANSFER_SGLANG_OUTPUT_ROOT", TRANSFER_ROOT / "outputs" / "sglang")
).resolve()
BASE_URL = os.environ.get("COSMOS3_SGLANG_BASE_URL", "http://localhost:30000").rstrip("/")
GUARDRAILS = os.environ.get("COSMOS3_SGLANG_GUARDRAILS", "false").strip().lower() not in {"0", "false", "no", "off"}

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("COSMOS_ROOT:", COSMOS_ROOT)
print("TRANSFER_ROOT:", TRANSFER_ROOT)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("BASE_URL:", BASE_URL)
print("GUARDRAILS:", GUARDRAILS)
```

## 3. Verify Endpoint Configuration

```python
from urllib.parse import urlparse

try:
    import requests
except ImportError as exc:
    raise RuntimeError("Install requests in this notebook kernel: pip install requests") from exc


def api_root_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


API_ROOT = api_root_url(BASE_URL)
VIDEOS_GENERATION_URL = f"{API_ROOT}/videos"
MODELS_URL = f"{API_ROOT}/models"
parsed = urlparse(API_ROOT)
print("api root:", API_ROOT)
print("videos generation:", VIDEOS_GENERATION_URL)
print("models:", MODELS_URL)
print("scheme:", parsed.scheme)
print("host:", parsed.netloc)

response = requests.get(MODELS_URL, timeout=30)
response.raise_for_status()
print(response.json())
```

## 4. Define Transfer Request Helpers

```python
TRANSFER_CONTROLS = ("edge", "blur", "depth", "seg", "wsm")


def compact_json_file(path: Path) -> str:
    return json.dumps(json.loads(path.read_text()), ensure_ascii=True, separators=(",", ":"))


def resolve_spec_path(spec_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (spec_path.parent / path).resolve()


def repo_relative_path(local_path: Path) -> str:
    return local_path.resolve().relative_to(COSMOS_ROOT).as_posix()


def spec_size(spec: dict) -> str:
    height = int(spec["resolution"])
    width_ratio, height_ratio = (int(part) for part in spec["aspect_ratio"].split(",", 1))
    width = round(height * width_ratio / height_ratio)
    return f"{width}x{height}"


def load_transfer_spec(control: str) -> tuple[Path, dict]:
    if control not in TRANSFER_CONTROLS:
        raise ValueError(f"control must be one of {TRANSFER_CONTROLS}, got {control!r}")
    spec_path = SPECS_DIR / f"{control}.json"
    if not spec_path.is_file():
        raise FileNotFoundError(spec_path)
    return spec_path, json.loads(spec_path.read_text())


def build_transfer_request(control: str) -> tuple[dict[str, str], Path, Path]:
    spec_path, spec = load_transfer_spec(control)
    hint = dict(spec[control])
    local_control_path = resolve_spec_path(spec_path, hint["control_path"])
    hint["control_path"] = local_control_path.resolve().as_posix()

    extra_params = {
        "use_resolution_template": False,
        "use_duration_template": False,
        "guardrails": GUARDRAILS,
        **hint,
        "control_hint": control,
        "resolution": spec["resolution"],
        "control_guidance": spec["control_guidance"],
        "num_video_frames_per_chunk": spec["num_video_frames_per_chunk"],
        "num_conditional_frames": spec.get("num_conditional_frames", 1),
        "num_first_chunk_conditional_frames": spec.get("num_first_chunk_conditional_frames", 0),
        "share_vision_temporal_positions": spec.get("share_vision_temporal_positions", True),
        "max_frames": spec["num_frames"],
    }
    form = {
        "prompt": compact_json_file(resolve_spec_path(spec_path, spec["prompt_path"])),
        "negative_prompt": compact_json_file(resolve_spec_path(spec_path, spec["negative_prompt_file"])),
        "size": spec_size(spec),
        "num_frames": str(spec["num_frames"]),
        "fps": str(spec["fps"]),
        "num_inference_steps": "50",
        "guidance_scale": str(spec["guidance"]),
        "flow_shift": "10.0",
        "seed": "2026",
        "extra_params": json.dumps(extra_params, separators=(",", ":")),
    }
    output_path = OUTPUT_ROOT / control / f"{spec['name']}.mp4"
    return form, local_control_path, output_path


def run_transfer(control: str) -> Path:
    form, local_control_path, output_path = build_transfer_request(control)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = output_path.with_suffix(".error.txt")
    tmp_path = output_path.with_suffix(".tmp")
    print("control:", control)
    print("local control:", local_control_path)
    print("size:", form["size"], "frames:", form["num_frames"], "fps:", form["fps"])
    print("guardrails:", json.loads(form["extra_params"])["guardrails"])
    print("output:", output_path)

    headers = {"Accept": "video/mp4"}
    api_key = os.environ.get("COSMOS3_SGLANG_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t0 = time.time()
    response = requests.post(VIDEOS_GENERATION_URL, json=form, headers=headers, timeout=3600)
    if not response.ok:
        error_path.write_text(response.text)
        print("request failed:", response.status_code)
        print(response.text[:2000])
        response.raise_for_status()

    initial = response.json()
    (output_path.parent / "response.json").write_text(json.dumps(initial, indent=2))

    while True:
        response = requests.get(f"{API_ROOT}/videos/{initial['id']}", timeout=30)
        response.raise_for_status()
        final = response.json()
        (output_path.parent / "final.json").write_text(json.dumps(final, indent=2))
        print(initial["id"], final.get("status"), f"{final.get('progress', 0)}%")
        if final.get("status") == "completed":
            break
        if final.get("status") in {"failed", "cancelled"}:
            raise RuntimeError(json.dumps(final, indent=2))
        time.sleep(5)

    content_response = requests.get(f"{API_ROOT}/videos/{initial['id']}/content", timeout=300)
    content_response.raise_for_status()
    if content_response.content:
        tmp_path.write_bytes(content_response.content)
        tmp_path.rename(output_path)
    else:
        raise RuntimeError("video content endpoint returned an empty body")

    print(f"wrote {output_path} in {time.time() - t0:.1f}s")
    return output_path


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise RuntimeError("Install imageio-ffmpeg or put ffmpeg on PATH to create compact previews.")


def make_preview(src: Path, *, crf: int = 28) -> Path:
    preview = src.with_name(f"{src.stem}_preview.mp4")
    if not preview.exists() or preview.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(
            [
                _ffmpeg_exe(),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                "veryfast",
                "-an",
                "-pix_fmt",
                "yuv420p",
                str(preview),
            ],
            check=True,
        )
    return preview


def display_video(path: Path, *, width: int = 720) -> None:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    label = html.escape(str(path))
    markup = f'''
<video controls playsinline preload="metadata" width="{width}" style="max-width: 100%; background: #000;">
  <source src="data:video/mp4;base64,{data}" type="video/mp4">
</video>
<div style="font-family: monospace; font-size: 12px; margin-top: 4px;">{label}</div>
'''
    display(HTML(markup))


def view_transfer(control: str, output_path: Path | None = None) -> None:
    form, local_control_path, expected_output = build_transfer_request(control)
    output_path = Path(output_path or expected_output)
    if not output_path.is_file():
        raise FileNotFoundError(f"missing output: {output_path} (run {control} transfer first)")
    for label, src in [("control", local_control_path), ("generated", output_path)]:
        preview = make_preview(src)
        print(f"{control} {label}: {src.name} ({src.stat().st_size // 1024} KB -> {preview.stat().st_size // 1024} KB preview)")
        display_video(preview)
```

## 5. Preview Available Inputs

```python
for control in TRANSFER_CONTROLS:
    form, local_control_path, _ = build_transfer_request(control)
    prompt = json.loads(form["prompt"])
    caption = prompt.get("temporal_caption") or prompt.get("comprehensive_t2i_caption") or prompt.get("extra", {}).get("prompt", "")
    print(f"{control}: {local_control_path.relative_to(COSMOS_ROOT)}")
    print(f"  size={form['size']} frames={form['num_frames']} fps={form['fps']}")
    print(f"  prompt={caption[:180]}{'...' if len(caption) > 180 else ''}")
```

## 6. Edge (Canny) Transfer

Run the `edge` transfer request through SGLang, then display the input control video and generated output.

```python
edge_output = run_transfer("edge")
```

```python
view_transfer("edge", edge_output)
```

## 7. Blur Transfer

Run the `blur` transfer request through SGLang, then display the input control video and generated output.

```python
blur_output = run_transfer("blur")
```

```python
view_transfer("blur", blur_output)
```

## 8. Depth Transfer

Run the `depth` transfer request through SGLang, then display the input control video and generated output.

```python
depth_output = run_transfer("depth")
```

```python
view_transfer("depth", depth_output)
```

## 9. Segmentation Transfer

Run the `seg` transfer request through SGLang, then display the input control video and generated output.

```python
seg_output = run_transfer("seg")
```

```python
view_transfer("seg", seg_output)
```

## 10. World Scenario Map Transfer

Run the `wsm` transfer request through SGLang, then display the input control video and generated output.

```python
wsm_output = run_transfer("wsm")
```

```python
view_transfer("wsm", wsm_output)
```
