<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Nano Transfer with vLLM-Omni

This notebook calls an already-running vLLM-Omni Cosmos3 server through the OpenAI-compatible video API. It reuses the checked-in transfer control assets and specs from this cookbook, then sends edge, blur, depth, segmentation, and world-scenario-map transfer requests to `POST /v1/videos/sync`.

The runnable examples use one control at a time. vLLM-Omni also accepts unweighted multi-control requests: include multiple hint blocks in `extra_params`. Per-hint `weight` is supported only by Cosmos Framework.

The notebook does not modify the vLLM-Omni source tree.

> **Skill note (not in the source):** this file is the cookbook notebook `run_video_transfer_with_vllm_omni.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/generator/transfer/run_video_transfer_with_vllm_omni.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

## 1. Prerequisites

Start a vLLM-Omni Cosmos3 server before running the request cells. The examples assume `.agents/skills` is mounted at `/workspace` and the server runs from that directory inside the container, so repo-local `cookbooks/...` paths resolve correctly.

Transfer controls are available from vLLM-Omni `main` and the released `vllm/vllm-omni:cosmos3` container. For current compatibility details, check the [Cosmos3-Nano recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-Nano.md).

```bash
cd /path/to/cosmos-framework/.agents/skills
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export COSMOS3_WORKDIR="$PWD"
export COSMOS3_HOST_PORT=8000

docker run --runtime nvidia --gpus all \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${COSMOS3_WORKDIR}:/workspace" \
  -v "${COSMOS3_WORKDIR}:${COSMOS3_WORKDIR}" \
  -e HF_TOKEN \
  -p "${COSMOS3_HOST_PORT}:8000" --ipc=host \
  -w /workspace \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano \
    --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / \
    --port 8000 \
    --init-timeout 1800
```

> **Skill note (not in the source):** add `-e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600` to the `docker run` launches in this file: the server caps `/v1/videos/sync` at 600 s by default, and Nano edge transfer took 700–810 s on one H100 (HTTP 504 "Video generation timed out after 600.0s"; the client-side `timeout=3600` does not cover it).

Generator guardrails are on by default and require access to the gated `nvidia/Cosmos-1.0-Guardrail` repository. When guardrails are enabled, also set and pass `HF_TOKEN` to the container; it must be a Hugging Face token authorized for that repository. The edge and depth requests explicitly disable guardrails; blur, segmentation, and WSM use `COSMOS3_VLLM_GUARDRAILS` (default `true`). To disable guardrails server-wide, pass `--no-guardrails` to `vllm serve`; alternatively, set `COSMOS3_VLLM_GUARDRAILS=false` for these sample requests.

### Quantized Checkpoints

FP8 checkpoints are published for Edge, Nano, and Super. This video-to-video
transfer example requires `Cosmos3-Nano`; base `Cosmos3-Edge-FP8` does not
support video-to-video transfer. Use `nvidia/Cosmos3-Nano-FP8` below.

Published [Nano FP8](https://huggingface.co/nvidia/Cosmos3-Nano-FP8) and
[Super FP8](https://huggingface.co/nvidia/Cosmos3-Super-FP8) checkpoints can be
downloaded to a local directory. Or use an output from the
[quantization walkthrough](../../cosmos3-quantization/SKILL.md).
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

```bash
# Download published Nano FP8 for this video-to-video transfer example.
export FP8_MODEL_DIR="${FP8_MODEL_DIR:-$PWD/models/cosmos3-nano-fp8}"
export SERVED_MODEL="${SERVED_MODEL:-nvidia/Cosmos3-Nano}"
# Skip this download when FP8_MODEL_DIR is a locally generated checkpoint.
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("nvidia/Cosmos3-Nano-FP8", local_dir=os.environ["FP8_MODEL_DIR"])
PY
```

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

## 2. Configure Paths and Endpoints

Run this cell from anywhere inside `.agents/skills`. It resolves local assets, output paths, the vLLM-Omni endpoint, and repo-local `control_path` values.

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


def find_framework_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").is_file() and (path / "cosmos_framework").is_dir():
            return path
    raise FileNotFoundError("Launch from the Framework checkout or a notebook directory inside it.")


FRAMEWORK_ROOT = find_framework_root(Path.cwd().resolve())
COSMOS_ROOT = FRAMEWORK_ROOT / ".agents" / "skills"
TRANSFER_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3" / "generator" / "transfer"
SPECS_DIR = TRANSFER_ROOT / "specs"
ASSETS_DIR = TRANSFER_ROOT / "assets"
OUTPUT_ROOT = Path(
    os.environ.get("COSMOS3_TRANSFER_VLLM_OUTPUT_ROOT", TRANSFER_ROOT / "outputs" / "vllm_omni")
).resolve()
VLLM_BASE_URL = os.environ.get("COSMOS3_VLLM_BASE_URL", "http://localhost:8000").rstrip("/")
GUARDRAILS = os.environ.get("COSMOS3_VLLM_GUARDRAILS", "true").strip().lower() not in {"0", "false", "no", "off"}

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("COSMOS_ROOT:", COSMOS_ROOT)
print("TRANSFER_ROOT:", TRANSFER_ROOT)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("VLLM_BASE_URL:", VLLM_BASE_URL)
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


API_ROOT = api_root_url(VLLM_BASE_URL)
VIDEOS_SYNC_URL = f"{API_ROOT}/videos/sync"
MODELS_URL = f"{API_ROOT}/models"
parsed = urlparse(API_ROOT)
print("api root:", API_ROOT)
print("videos sync:", VIDEOS_SYNC_URL)
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
GUARDRAIL_DISABLED_CONTROLS = frozenset({"edge", "depth"})


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
    hint["control_path"] = repo_relative_path(local_control_path)

    extra_params = {
        "use_resolution_template": False,
        "use_duration_template": False,
        "guardrails": GUARDRAILS and control not in GUARDRAIL_DISABLED_CONTROLS,
        control: hint,
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
    api_key = os.environ.get("COSMOS3_VLLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t0 = time.time()
    response = requests.post(VIDEOS_SYNC_URL, data=form, headers=headers, timeout=3600)
    if not response.ok:
        error_path.write_text(response.text)
        print("request failed:", response.status_code)
        print(response.text[:2000])
        response.raise_for_status()
    tmp_path.write_bytes(response.content)
    tmp_path.replace(output_path)
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

Run the `edge` transfer request through vLLM-Omni, then display the input control video and generated output.

```python
edge_output = run_transfer("edge")
```

```python
view_transfer("edge", edge_output)
```

## 7. Blur Transfer

Run the `blur` transfer request through vLLM-Omni, then display the input control video and generated output.

```python
blur_output = run_transfer("blur")
```

```python
view_transfer("blur", blur_output)
```

## 8. Depth Transfer

Run the `depth` transfer request through vLLM-Omni, then display the input control video and generated output.

```python
depth_output = run_transfer("depth")
```

```python
view_transfer("depth", depth_output)
```

## 9. Segmentation Transfer

Run the `seg` transfer request through vLLM-Omni, then display the input control video and generated output.

```python
seg_output = run_transfer("seg")
```

```python
view_transfer("seg", seg_output)
```

## 10. World Scenario Map Transfer

Run the `wsm` transfer request through vLLM-Omni, then display the input control video and generated output.

```python
wsm_output = run_transfer("wsm")
```

```python
view_transfer("wsm", wsm_output)
```
