# Distilled 4-step students on TensorRT-LLM — distilled from the cookbook notebook

Scope: the whole distilled notebook — introduction, prerequisites, server launch, endpoint configuration, the backend-only request helpers and the two use cases (`t2i_distilled`, `i2v_distilled`) — plus the README's "Distilled 4-step notebook" section; cells shared with the other backend notebooks live in payloads.md and are referenced by section.

## Quickstart (cookbook README)

### Distilled 4-step notebook

`run_distilled_with_trt_llm.ipynb` is the
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
[shared environment setup guide](../../cookbooks/cosmos3/README.md#tensorrt-llm-generator).

Skill note (not in the README): "a single GPU" means a single GPU that holds ~125 GiB; the bf16 weights alone are 122.4 GiB per student (measured 131.4 GB on disk) — on GPUs under ~125 GiB use the offload config (≥ ~128 GiB host RAM) or the patched `fp8` branch under § 2 (two gates, § 2 skill note).

## Cosmos3 Generator Distilled 4-Step with TensorRT-LLM

Licence header: see payloads.md § 2.1.

This notebook calls already-running TensorRT-LLM VisualGen servers with direct `curl` requests from Python, using the two published DMD2-distilled Cosmos3-Super students:

- [`nvidia/Cosmos3-Super-Text2Image-4Step`](https://huggingface.co/nvidia/Cosmos3-Super-Text2Image-4Step) — text-to-image
- [`nvidia/Cosmos3-Super-Image2Video-4Step`](https://huggingface.co/nvidia/Cosmos3-Super-Image2Video-4Step) — image-to-video

Each section is self-contained, so you can run just one. Both checkpoints serve on a single GPU.

Skill note (not in the notebook): on what "a single GPU" has to hold (the bf16 weights alone are 122.4 GiB per student), see the skill note after the Cosmos3-Super-Text2Image-4Step launch in § 2.

A distilled checkpoint is not a base checkpoint with fewer steps. It ships a `FlowMatchEulerDiscreteScheduler` whose `fixed_step_sampler_config.t_list` locks the schedule to four stochastic (SDE) steps, and classifier-free guidance is baked into the weights, so the only valid `guidance_scale` is `1.0`. TensorRT-LLM reads both facts from the checkpoint:

- **Leave `num_inference_steps` and `guidance_scale` out of the request.** The server fills in the checkpoint's four steps and guidance `1.0`. Sending any other value is rejected with a `400`, not silently clamped.
- **Leave `use_system_prompt` unset.** `Cosmos3-Super-Image2Video-4Step` declares `default_use_system_prompt: true` in its `model_index.json`; sending `false` overrides that default and changes the tokenized prompt.
- `flow_shift` does not apply. It is a UniPC knob, and distilled checkpoints do not use UniPC.

These students cover text-to-image and image-to-video only. For text-to-video, video-to-video, or synchronized audio, use the base Cosmos3-Nano / Cosmos3-Super checkpoints in `run_with_trt_llm.ipynb`.

## 1. Prerequisites

Use a running TensorRT-LLM server with Cosmos3 VisualGen distilled-checkpoint support, and set endpoint environment variables before the setup cell if you are not using the local default. Generation uses `/v1/videos/generations`; Cosmos3 text-to-image runs as a one-frame VisualGen video request with `num_frames=1`, `seconds=1`, and `fps=8`. Image-to-video sends its reference image as multipart `input_reference`.

Install the `ffmpeg` CLI in the server environment. TensorRT-LLM uses it to encode MP4; without it, the fallback AVI encoder is used instead.

Guardrail prerequisite: see payloads.md § 2.3 (this notebook carries the `run_with_trt_llm.ipynb` wording of that table).

The two checkpoints are served separately, so run one server at a time on a single GPU and point the matching environment variable at it. Both variables default to `COSMOS3_TRTLLM_BASE_URL`: for the four `export` lines (`COSMOS3_TRTLLM_BASE_URL`, `COSMOS3_TRTLLM_T2I_4STEP_BASE_URL`, `COSMOS3_TRTLLM_I2V_4STEP_BASE_URL`, `COSMOS3_TRTLLM_API_KEY`) see payloads.md § 4.5, `run_distilled_with_trt_llm.ipynb` block.

Skill note (not in the notebook): on what "a single GPU" has to hold (the bf16 weights alone are 122.4 GiB per student), see the skill note after the Cosmos3-Super-Text2Image-4Step launch in § 2.

## 2. Start the Server

Run the TensorRT-LLM VisualGen server before running the request cells. To build a checkout from source, follow NVIDIA's [Build from Source](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source.html) guide. Install `ffmpeg` in that environment before serving (`apt-get update && apt-get install -y ffmpeg` on Ubuntu/Debian images).

Skill note (not in the notebook): the stable PyPI `tensorrt-llm` 1.2.x has no VisualGen (`trtllm-serve` rejects `--visual_gen_args`), so `pip install tensorrt-llm` is not a route. The prebuilt-wheel route verified by the skill's eval runs is `tensorrt_llm==1.3.0rc27` — the first release that contains the pinned commit bca6761 — with `torch==2.13.0+cu130` and `torchvision==0.28.0+cu130`, from `https://pypi.nvidia.com` plus the cu130 torch index: `uv pip install --no-config --index-strategy unsafe-best-match --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu130 "tensorrt_llm==1.3.0rc27" "torch==2.13.0+cu130" "torchvision==0.28.0+cu130"` (PyPI hosts only sdists of tensorrt_llm, so both flags are required; the rc27 binaries are built against torch 2.13). This notebook uses only the video route (`/v1/videos/generations`), which 1.3.0rc26 also serves correctly (two eval PASS cells), so 1.3.0rc26 (what both eval PASS cells ran) works here, and 1.3.0rc27 — the SKILL.md Variant 5 Route A default — was exercised on this notebook on 2026-10-24 (the offload launch on two H100 NVL cells and the patched `fp8` launch, video route; eval run 2026-10-24); rc26 is unusable only for `/v1/images/generations` (HTTP 500, see trt-llm.md § 2). Outside the devel container a pip-installed wheel also needs system OpenMPI (`apt-get update && apt-get install -y libopenmpi-dev openmpi-bin ffmpeg`), a seeded venv (`uv venv … --python 3.12 --seed --managed-python --allow-existing --no-config`, so `pip` exists), its CUDA-13 libraries from the cu130 torch index (never symlink `.so.12` to `.so.13`) and a `CUDA_HOME` — point it at the venv's `nvidia/cu13` when no toolkit is installed: `export CUDA_HOME="$(python -c 'import nvidia,os;print(os.path.join(nvidia.__path__[0],"cu13"))')"`; the upstream TensorRT-LLM [Installing on Linux via pip](https://nvidia.github.io/TensorRT-LLM/installation/linux.html) page documents the `libopenmpi-dev` and cu130 torch steps. The upstream `TRTLLM_USE_PRECOMPILED=1 pip install -e .` shortcut does not work for the pinned commit bca6761 (it resolves the unpublished `1.3.0rc26+bca6761ab8`; a wheel path or URL belongs in `TRTLLM_PRECOMPILED_LOCATION`; and the published rc26 binaries lack `trtllm::silu_and_mul_fp8_quantize_1x128_packed_ue8m0`, which the pinned Python needs) — install a complete published wheel (1.3.0rc27, or 1.3.0rc26 for this video-only notebook) or do the full devel-container build, and keep the checkout only for `examples/visual_gen/configs`. Full recipe: trt-llm.md § 7.2 (Route A, the wheel; § 7.4 is the source build).

Skill note (not in the notebook; eval runs 2026-10-21/24): choose the launch by two gates before the first command, because the bf16 weights are 122.4 GiB per student. Gate 1, GPU memory (`nvidia-smi --query-gpu=memory.total --format=csv,noheader`, which prints MiB: the ~125 GiB gate is 128,000 MiB, an H100 80 GB reports 81559): on any GPU under ~125 GiB — every H100 80 GB and H100 NVL 94 GB — the plain `trtllm-serve` launch below OOMs at load (measured 79.11/79.22 GiB in use on the H100 PCIe node after 30 s, 79.18/79.22 on the H100 SXM 80 GB node (251 GiB host RAM) after 41 s, 91.65/93.13 GiB on H100 NVL after ~40 s); never run it there — it is the launch for ≥ ~125 GiB GPUs (H200/B200-class), the notebook's claim, not exercised by the eval runs (none ran on a ≥ ~125 GiB GPU; if it OOMs there, use gate 2). Gate 2, host RAM available (`free -g`, its `available` column = MemAvailable; inside a container also read the cgroup ceiling, `cat /sys/fs/cgroup/memory.max` — cgroup v2, bytes, `max` = unlimited — because `free` reports the host's /proc/meminfo, not the container's limit, and the lower of the two figures is what the offload loader gets — the offload block's preflight takes the lower one): ≥ ~128 GiB → the bf16 + `cpu_offload_config` launch (skill note after the Cosmos3-Super-Text2Image-4Step block); less → the Hugging Face `fp8` branch with the `embed_tokens` config patch (last skill note of that section; commands: the `fp8` route of the § 7.2 script below; FP8 needs an SM 89+ GPU — Ada/Hopper/Blackwell — so on an A100 only the offload launch can apply, and no eval run touched an A100) — Text2Image-4Step only: `nvidia/Cosmos3-Super-Image2Video-4Step` also has an `fp8` branch (220a1b2d today) that no eval run served, and the offload launch was exercised only with the t2i config, so no sub-~125 GiB route is verified for the Image2Video student. Smoke runs put `export TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1` before whichever launch they pick (guardrail prerequisite, § 1 / payloads.md § 2.3; the § 7.2 script exports it before every route). Loop 3 (2026-10-24): § 7.2 below (the launch SKILL.md Variant 6 points at) is ONE gated launch script — it measures the GPU (plain route only at ≥ 128,000 MiB), the host RAM available (`free -g`, capped by the cgroup ceiling) and refuses to start while another server holds the GPU, then starts exactly one `trtllm-serve` (plain / offload / patched fp8). Two eval runs (cursor on H100 NVL, opencode on H100 SXM) had still run the plain launch first and the second launch then OOMed against the loading server (`33.19 MiB free`), so the notebook block is named in § 7.3 only as the command the script's `plain` route reproduces.

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

Skill note (not in the notebook; measured by the skill's eval runs on H100 80 GB (PCIe and SXM) and H100 NVL 94 GB nodes, 2026-10-21/24): on an 80 GB H100 the command above OOMs at load — the student's bf16 safetensors are 122.4 GiB (131.4 GB on disk), and the server died with 79.18 of 79.22 GiB in use after 41 s on the H100 SXM 80 GB node, 251 GiB host RAM (2026-10-21; 79.11/79.22 after 30 s on the H100 PCIe 80 GB node, 125 GiB host RAM, in the 2026-10-24 run; 91.65/93.13 GiB on an H100 NVL 94 GB) — gate 1 of the § 2 note. With ≥ ~128 GiB host RAM available (gate 2; eval-verified on the H100 SXM 80 GB node / 251 GiB host RAM and the H100 NVL 94 GB node / 1 TB host RAM), copy the config and enable CPU offload — a prototype-status TensorRT-LLM option (`visual_gen/args.py` `CpuOffloadConfig`) that stages reasoner → generator through a GPU arena and keeps the rest in pinned host RAM. Less host RAM is fatal, not slow: the loader materialises the offloaded towers on CPU and packs each stage into a new pinned buffer before freeing the original, and on a 125.6 GiB host (the H100 PCIe 80 GB node: 128,664 MiB, no swap; eval run 2026-10-24) the host peak reached 126,136 of 128,664 MiB (sampled at 158 s) and the OOM killer SIGKILLed the server at 170 s (cgroup `memory.events` `oom_kill 1`) — hence the fail-stop preflight at the top of the block (less host RAM → the `fp8` branch, last note of this section):

```bash
# fail-stop preflight — run the block as a script (`exit 1` would close an interactive shell; `|| true`: a missing `free` must not kill a `set -o pipefail` script before the message); `free -g` truncates to whole GiB, so the H100 PCIe 80 GB node's 128,664 MiB total reads as 125 and hosts of that class fail it on purpose; the cgroup v2 ceiling (memory.max, bytes) is taken when lower
avail=$(free -g 2>/dev/null | awk '/Mem:/{print $7}' || true)
lim=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || true); if [ -n "$lim" ] && [ "$lim" != max ] && [ $((lim / 1073741824)) -lt "${avail:-0}" ]; then avail=$((lim / 1073741824)); fi
[ "${avail:-0}" -ge 128 ] || { echo "only ${avail:-?} GiB host RAM available (lower of free -g and cgroup memory.max) — use the fp8 launch below"; exit 1; }
cp "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-t2i-1gpu.yaml" cosmos3-t2i-1gpu-offload.yaml
cat >> cosmos3-t2i-1gpu-offload.yaml <<"YAML"
cpu_offload_config:
  enable: true
  pin_memory: true
YAML
trtllm-serve nvidia/Cosmos3-Super-Text2Image-4Step \
  --visual_gen_args "$PWD/cosmos3-t2i-1gpu-offload.yaml" --port 8000
```

Skill note (not in the notebook; eval runs 2026-10-21/24): measured on TensorRT-LLM 1.3.0rc26, H100 80 GB with ≥ 128 GiB host RAM — `/health` after ~250 s, 70,271 MiB GPU peak, ~116 GiB pinned host RAM (plan for ≥ ~128 GiB free host RAM; eval-verified on the 251 GiB and 1 TB hosts named above, OOM-killed on 125.6 GiB), 1024x1024 text-to-image in ~7 s; the startup log still prints `Distilled Cosmos3 checkpoint: fixed 4-step schedule [...]`. Leave CUDA graphs off (the default; offload and CUDA graphs are mutually exclusive).

> Skill note (not in the notebook; eval runs 2026-10-21/24): `quant_config: {quant_algo: FP8_BLOCK_SCALES, dynamic: true}` on the bf16 weights serves at ~70 GB but returned pure noise for this checkpoint — do not use it; an FP8 + `cpu_offload_config` + `skip_warmup` profile (~41 GB GPU) passed only a pixel-provenance check, not a visual one, and stays unverified; `cosmos3-fp8-1gpu.yaml` does not exist in the pinned `examples/visual_gen/configs`. The Hugging Face `fp8` branch of `nvidia/Cosmos3-Super-Text2Image-4Step` (revision e6e06ab9, 69.0 GB / 64.3 GiB; ModelOpt static per-tensor FP8, `hf_quant_config.json` `quant_algo` FP8) IS verified — two runs (125.6 GiB PCIe host and 251 GiB SXM host), 2026-10-24, TensorRT-LLM 1.3.0rc27, 1× H100 80 GB + 125.6 GiB host — but only with a config patch: unmodified it fails at load with `Static quantization (FP8) is configured for module 'language_model.embed_tokens', but the checkpoint provides a torch.bfloat16 weight without the expected scale tensor(s) ['weight_scale']`, because the branch's two `ignore` lists (top-level `ignore` in `hf_quant_config.json`, `quantization_config.ignore` in `transformer/config.json`) lack `embed_tokens`. Serve an overlay directory — a symlink to every checkpoint file plus patched copies of those two JSON files (`ignore` += `"*embed_tokens*"`, weights untouched) — with the shipped `cosmos3-t2i-1gpu.yaml` and no offload: `/health` 200 after 60 s (incl. the 7.2 s 1024x1024x1 warm-up), 70,983 MiB GPU peak of 81,559, host RAM ~6.7 GB, 1024x1024 text-to-image in 2.5 s (server 2.38 s), image visually correct. It is a workaround for the branch's config until the branch lists `embed_tokens` itself, and the image then comes from the FP8 (ModelOpt) variant, not the bf16 main revision. Verified for the Text2Image-4Step student only: the Image2Video-4Step repo's own `fp8` branch (220a1b2d today) was served by no eval run. Commands (download, `cp -rs` overlay, patch snippet, launch): the `fp8` route of the § 7.2 script below.

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

Skill note (not in the notebook): the notebook viewed the one-frame video inline, so it never wrote a still; see the extraction step under Distilled: Text to Image.

## Cosmos3-Super-Text2Image-4Step Examples

Use cases for the distilled text-to-image student. This section is self-contained; you can run it without the image-to-video section below.

### Distilled: Text to Image

Distilled text-to-image generation using a structured JSON prompt at the deployed 1024x1024 shape. TensorRT-LLM Cosmos3 returns this as a one-frame video from the video generation endpoint; the request uses `num_frames=1`, `seconds=1`, and `fps=8`.

Skill note (not in the notebook): the response is a one-frame video — MP4, or AVI when the server has no `ffmpeg` — written to the path `run_trtllm_payload` returns (`<OUT>/trt_llm_distilled/t2i_distilled/t2i_distilled.mp4` or `.avi`, payloads.md § 4.2), not a PNG. For a still image extract that frame with `ffmpeg` (already the § 1 server prerequisite; `subprocess` is imported in § 6) — tested in the eval runs, 1024x1024 PNG:

Skill note (not in the notebook; eval run 2026-10-24): if `POST /v1/videos/generations` answers HTTP 500 `Video generation failed` although the server log shows the 4 denoising steps and the decode completed (seen once on the patched `fp8` branch; the one-frame video packaging failed server-side), post the same prompt to the native image route — `POST /v1/images/generations` with `size` `1024x1024`, `format` `png`, `response_format` `b64_json`, `max_sequence_length` 4096, `seed` 0, no `num_inference_steps` / `guidance_scale`, and `t2i_distilled`'s `extra_params` (`use_resolution_template` false, `use_duration_template` false, `use_guardrails` false, `enable_audio` false) plus `output_type: "image"` — no `use_system_prompt`, no `seconds` / `fps` / `num_frames`; needs 1.3.0rc27 (on rc26 the image route answers the NameError 500); record in `result_summary.md` that the video route answered 500 and this documented fallback was used — HTTP 200 with a 1024x1024 PNG on the 4-step `fp8` server in that run; decode `data[0].b64_json` and skip the ffmpeg frame extraction.

```python
video_path = run_trtllm_payload(t2i_payload, t2i_output, model=t2i_model)
subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", str(video_path), "-frames:v", "1", str(OUT / "t2i_4step.png")], check=True)
```

Skill note (not in the notebook): verify the PNG is 1024x1024 and keep `<stem>.headers.txt` next to the video for provenance; the still is decoded from the server's compressed one-frame video (inherent to the notebook's video-route design).

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

## 7. Skill notes (not in the notebook) — gated launch script, measurements and fallback (from SKILL.md Variant 6)

Skill-authored text moved here from SKILL.md on 2026-10-27 (layout change only; wording unchanged). Nothing in this section is notebook text; SKILL.md keeps a one-line pointer at each former location.

### 7.1 Why one launch script, gated by GPU and host memory (SKILL.md Variant 6)

**One launch script, gated by GPU and host memory** (not a notebook rule; eval runs 2026-10-21/24 on
H100 PCIe 80 GB, H100 SXM 80 GB and H100 NVL 94 GB nodes). The bf16 weights are 122.4 GiB, so on any
GPU under ~125 GiB — every H100 80 GB and H100 NVL 94 GB — the notebook's plain launch OOMs at load
(measured 79.11/79.22 GiB in use on H100 PCIe after 30 s, 79.18/79.22 on H100 SXM after 41 s,
91.65/93.13 GiB on H100 NVL after ~40 s). Do not "try it first": a server that is still loading holds
the GPU, and the next launch OOMs against it (eval run 2026-10-24: the offload launch died with
`33.19 MiB free` — the abandoned plain launch was still on the GPU; the same route passed on an empty
80 GB SXM in loop 1). The script below is the
documented first attempt on every host: it measures the GPU and the host, picks ONE route and starts
exactly one server —

- GPU ≥ ~125 GiB (128,000 MiB; H200/B200-class): the notebook's plain launch (the notebook's claim; no
  eval run had such a GPU — if it OOMs there, remove the plain branch and re-run).
- GPU under ~125 GiB and ≥ ~128 GiB host RAM available (`MemAvailable` of /proc/meminfo = `free -g`
  `available`, capped by the cgroup ceiling `/sys/fs/cgroup/memory.max` inside a container — /proc/meminfo
  is the host's): bf16 + `cpu_offload_config` (eval-verified on 251 GiB and 1 TB hosts; on a 125.6 GiB host
  the host peak reached 126,136 of 128,664 MiB at 158 s and the OOM killer SIGKILLed the server at 170 s).
  128 GiB is a lower bound, not a verified sufficient figure: no run had between 128 and 250 GiB available,
  and the only sub-128 GiB run was killed after a ≥ 123 GiB peak sample.
- less host RAM: the Hugging Face `fp8` branch with the `embed_tokens` patch (eval-verified on an
  80 GB H100 with 125.6 GiB host RAM; FP8 needs an SM 89+ GPU — Ada/Hopper/Blackwell; Text2Image-4Step
  only — `nvidia/Cosmos3-Super-Image2Video-4Step` also has an `fp8` branch, 220a1b2d today, but no eval
  run served it, and the offload launch was exercised only with the t2i config).

### 7.2 Skill launch script (not in the notebook) — measure, pick ONE route, start exactly one server (SKILL.md Variant 6)

Run it as a script (`bash variant6-serve.sh`, not pasted line by line: `exit 1` would close an
interactive shell and every launch block needs the Variant 5 Route A venv, `CUDA_HOME`,
`TRTLLM_ROOT` and the guardrail switch in the same shell — one eval run lost `CUDA_HOME` between
tool calls and `tensorrt_llm` failed to import; Route B's devel container has both and skips the
`source` line):

```bash
# Variant 6 launch script — measure, pick ONE route, start exactly one trtllm-serve
source "${TRTLLM_VENV:?Route A venv path (Route B devel container: skip this line)}/bin/activate"; export CUDA_HOME="$(python -c 'import nvidia,os;print(os.path.join(nvidia.__path__[0],"cu13"))')"
export TRTLLM_ROOT="${TRTLLM_ROOT:-$PWD}"; export TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1   # smoke runs: skip the gated guardrail (request bodies also send use_guardrails: false)
CFG="$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-t2i-1gpu.yaml"
test -f "$CFG" || { echo "TRTLLM_ROOT=$TRTLLM_ROOT does not look like a TensorRT-LLM checkout"; exit 1; }
# the GPU must be empty: a server from an earlier attempt (still loading or orphaned) makes this launch OOM
GPU_SEL=${CUDA_VISIBLE_DEVICES:+-i ${CUDA_VISIBLE_DEVICES%%,*}}   # the GPU this server will use (first visible one; GPU 0 when unset)
USED=$(nvidia-smi $GPU_SEL --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ -n "$USED" ] || { echo "nvidia-smi failed — no usable GPU/driver"; exit 1; }
[ "$USED" -lt 1024 ] || { echo "GPU already in use ($USED MiB) — stop the previous server first (teardown below) and check nvidia-smi --query-compute-apps"; exit 1; }
GPU_MIB=$(nvidia-smi $GPU_SEL --query-gpu=memory.total --format=csv,noheader,nounits | head -1)   # 81559 on an H100 80 GB, 95830 on an H100 NVL 94 GB
AVAIL=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)                      # host RAM available in whole GiB (= free -g "available"; no procps needed)
LIM=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
[ "$LIM" = max ] || [ $((LIM / 1073741824)) -ge "$AVAIL" ] || AVAIL=$((LIM / 1073741824))   # inside a container the cgroup ceiling is what the loader gets
MODEL=nvidia/Cosmos3-Super-Text2Image-4Step
if [ "$GPU_MIB" -ge 128000 ]; then
  ROUTE=plain      # ≥ ~125 GiB GPU: the notebook's launch — weights resident, shipped YAML
elif [ "$AVAIL" -ge 128 ]; then
  ROUTE=offload    # bf16 + CPU offload: a prototype-status TensorRT-LLM option (visual_gen/args.py CpuOffloadConfig) that stages
                   # reasoner -> generator through a GPU arena and keeps the rest in pinned host RAM; leave CUDA graphs off (default)
  cp "$CFG" cosmos3-t2i-1gpu-offload.yaml
  printf 'cpu_offload_config:\n  enable: true\n  pin_memory: true\n' >> cosmos3-t2i-1gpu-offload.yaml
  CFG="$PWD/cosmos3-t2i-1gpu-offload.yaml"
else
  ROUTE=fp8        # the Hugging Face fp8 branch (revision fp8 e6e06ab9, 69.0 GB / 64.3 GiB, ModelOpt static per-tensor FP8) with the shipped YAML, no offload.
                   # Unpatched it fails at load — "Static quantization (FP8) is configured for module 'language_model.embed_tokens', but the
                   # checkpoint provides a torch.bfloat16 weight without the expected scale tensor(s) ['weight_scale']" — because the branch's two
                   # `ignore` lists (top-level in hf_quant_config.json, quantization_config.ignore in transformer/config.json) lack embed_tokens.
                   # Workaround until the branch lists it itself: an overlay directory of symlinks plus patched copies of those two JSON files
                   # (`ignore` += "*embed_tokens*"; weights untouched). The image then comes from the FP8 (ModelOpt) variant, not the bf16 main revision.
  export FP8_SRC="${FP8_SRC:-${HF_HOME:-$HOME/.cache/huggingface}/local/Cosmos3-Super-Text2Image-4Step-fp8}"   # 69 GB on the large disk, not inside the checkout
  case "$FP8_SRC" in /*) ;; *) FP8_SRC="$PWD/$FP8_SRC" ;; esac   # cp -s makes relative links only inside the cwd — the source must be absolute
  export FP8_DST="${FP8_DST:-${FP8_SRC}-patched}"
  uvx --no-config hf@latest download nvidia/Cosmos3-Super-Text2Image-4Step --revision fp8 --local-dir "$FP8_SRC" \
    || { echo "fp8 download failed"; exit 1; }   # 153 s in the eval run
  rm -rf "$FP8_DST"; mkdir -p "$FP8_DST" && cp -rs "$FP8_SRC"/. "$FP8_DST"/ \
    || { echo "overlay failed (FP8_SRC must be an absolute path to the downloaded branch)"; exit 1; }   # one symlink per checkpoint file, no weights copied
  python - "$FP8_DST" <<'PY' || { echo "embed_tokens patch failed"; exit 1; }
import json, os, sys
dst = sys.argv[1]
for rel, parents in (("hf_quant_config.json", ()), ("transformer/config.json", ("quantization_config",))):
    path = os.path.join(dst, rel)
    with open(path) as f:
        cfg = json.load(f)
    node = cfg
    for key in parents:
        node = node[key]
    ignore = node.setdefault("ignore", [])
    if "*embed_tokens*" not in ignore:
        ignore.append("*embed_tokens*")
    os.remove(path)                       # drop the symlink first — writing through it would edit the branch's file
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(rel, "ignore:", ignore)
PY
  MODEL="$FP8_DST"
fi
echo "route=$ROUTE gpu=${GPU_MIB} MiB host_available=${AVAIL} GiB model=$MODEL config=$CFG"
setsid trtllm-serve "$MODEL" --visual_gen_args "$CFG" --port 8000 > serve.log 2>&1 &   # the notebook's command with the route's model/config
echo $! > serve.pid
for _ in $(seq 180); do   # the notebook's check_trtllm_server deadline: 1800 s, every 10 s
  curl -sf http://localhost:8000/health >/dev/null && break
  kill -0 "$(cat serve.pid)" 2>/dev/null || { tail -50 serve.log; exit 1; }
  sleep 10
done
curl -sf http://localhost:8000/health >/dev/null || { echo "no /health after 1800 s"; tail -50 serve.log; exit 1; }
echo "ready: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1) in use"
```

### 7.3 The `plain` route and the notebook's command (SKILL.md Variant 6)

The `plain` route issues the notebook's command verbatim (`trtllm-serve nvidia/Cosmos3-Super-Text2Image-4Step --visual_gen_args .../cosmos3-t2i-1gpu.yaml --port 8000`, distilled-trt-llm.md § 2 — reference only; never run it by hand on a GPU under 128,000 MiB).

### 7.4 Measured — eval runs 2026-10-21/24, not notebook figures (SKILL.md Variant 6)

Measured (eval runs 2026-10-21/24, not notebook figures): `offload` route on one H100 80 GB with
≥ 128 GiB host RAM (TensorRT-LLM 1.3.0rc26 — the video route is unaffected by the rc26
image-route bug): `/health` after ~250 s, 70,271 MiB GPU peak, 116 GiB pinned host RAM,
1024x1024 t2i in ~7 s; on the H100 NVL (1 TB host) the same route came up in ~4.5–5 min (260 s and 301 s measured; 70,911 MiB in use). `fp8` route
on one H100 80 GB with 125.6 GiB host RAM (TensorRT-LLM 1.3.0rc27, two runs): `/health` after
~60 s (incl. the 7.2 s 1024x1024x1 warm-up), 70,587–70,983 MiB GPU peak of 81,559, host RAM
~6.7 GB, 1024x1024 t2i in 2.5 s, image visually correct (the robot arm draping blue satin on a
mannequin). Both routes still print `Distilled Cosmos3 checkpoint: fixed 4-step schedule [...]`
at startup. The fp8 weights are served as published; only the two JSON files differ from the branch.

### 7.5 Still unverified — not routes (SKILL.md Variant 6)

Still unverified — not routes: dynamic `quant_config: {quant_algo: FP8_BLOCK_SCALES, dynamic: true}`
on the bf16 weights serves at ~70 GB but returned pure noise for this checkpoint (do not use
it); the FP8 + cpu_offload + skip_warmup profile (~41 GB GPU) passed only a pixel-provenance
check, not a visual one. The Hugging Face `fp8` branch, by contrast, IS verified — only with
the `embed_tokens` patch above; unpatched it does not load.

### 7.6 Image-route fallback when the one-frame video route answers HTTP 500 (SKILL.md Variant 6)

If `POST /v1/videos/generations` answers HTTP 500 `Video generation failed`
although the log shows the 4 denoising steps and the decode completed (seen once, `fp8` route, 2026-10-24; the
server-side one-frame video packaging failed), post the same prompt to the native image route instead —
`POST /v1/images/generations` (needs 1.3.0rc27 — on rc26 this route answers the NameError 500 of Variant 5) with the body
`{"prompt": "<compact JSON of the asset>", "size": "1024x1024", "format": "png", "response_format": "b64_json",
"max_sequence_length": 4096, "seed": 0, "extra_params": {"use_resolution_template": false, "use_duration_template": false,
"use_guardrails": false, "enable_audio": false, "output_type": "image"}}` — `t2i_distilled`'s `extra_params` plus
`output_type`; no `num_inference_steps` / `guidance_scale` / `use_system_prompt`, no `seconds` / `fps` / `num_frames`
(the run recorded size, format, response_format and output_type) — HTTP 200 with a 1024x1024 PNG on the 4-step `fp8`
server in that run; decode `data[0].b64_json`, skip the ffmpeg step, and record in `result_summary.md` that the video
route answered 500 and this documented fallback was used.

### 7.7 Launch mechanism and teardown escalation (SKILL.md Variant 6)

Launch from a script with `setsid … &`: a server started inside a backgrounded subshell
`( … ) &` dies when the agent's tool kills that command at its timeout (one eval run,
2026-10-24, lost the offload server this way after 60–120 s; `nohup … & disown` also survived
it) — `setsid` puts the server in its own session and process group, out of that kill's reach.
If the process group is still alive 30 s after `kill -INT -- "-$PID"` (`kill -0 -- "-$PID"`),
escalate with `kill -TERM -- "-$PID"`, then `kill -KILL -- "-$PID"` (one eval run needed it),
and end with the leftover check (`nvidia-smi --query-gpu=memory.used` back to 0 MiB;
`--query-compute-apps` and `pgrep` name survivors to kill by PID): in two eval runs the
recorded PID's group was not the whole server, and the GPU stayed occupied until the survivors
were killed by PID.
