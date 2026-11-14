<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3-Nano-Sim-Bimanual with vLLM-Omni

Serve `nvidia/Cosmos3-Nano-Sim-Bimanual` with [vLLM-Omni](https://github.com/vllm-project/vllm-omni) and generate a unified-action example and a mixed-domain example through its synchronous Videos API. Both use seed 2026 and 30 FPS.

The model predicts video from the supplied actions; it does not choose robot actions.

> **Skill note (not in the source):** this file is the cookbook notebook `run_bimanual_with_vllm_omni.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/simulation/bimanual/run_bimanual_with_vllm_omni.ipynb`, cosmos-framework `main` 45fecab) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced as `python`, outputs omitted (the committed notebook has none). Edits are limited to three kinds: link targets that pointed at files this migration moved were re-pointed; path anchors in code that the move invalidated were replaced, each marked in place with a `Skill note (not in the source)` comment; and location text that named the retired cookbook folder (default output folders, where to run from) now names this skill's folder, each marked by a skill note. The changes are recorded in `docs/cookbook_to_skill_migration.md`.

## 1. Prerequisites

**Server host:** Linux with one NVIDIA GPU, Docker with NVIDIA Container Toolkit, and Hugging Face access to `nvidia/Cosmos3-Nano-Sim-Bimanual`. Log in on the host so the mounted cache includes your credentials.

**Notebook client:** Python 3.10 or newer, FFmpeg/FFprobe, and Hugging Face access to the model.

## 2. Install and start the server

All Simulation vLLM-Omni notebooks use the `vllm/vllm-omni:cosmos3` Docker image. Use a Linux GPU server with Docker and the NVIDIA Container Toolkit. Export `HF_TOKEN` with read access to the model. Install the Hugging Face CLI on the host with `pip install huggingface_hub` if `hf` is unavailable, then pull the image:

```bash
docker pull vllm/vllm-omni:cosmos3
```

Download the selected checkpoint snapshot, then serve its local directory and leave the container running while using the notebook:

```bash
export COSMOS3_MODEL_REVISION="${COSMOS3_MODEL_REVISION:-main}"
export COSMOS3_CHECKPOINT_DIR="${COSMOS3_CHECKPOINT_DIR:-$PWD/checkpoints/Cosmos3-Nano-Sim-Bimanual-$COSMOS3_MODEL_REVISION}"
hf download nvidia/Cosmos3-Nano-Sim-Bimanual \
  --revision "$COSMOS3_MODEL_REVISION" --local-dir "$COSMOS3_CHECKPOINT_DIR"

docker run --rm --gpus '"device=0"' --ipc=host \
  -e HF_TOKEN \
  -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -v "$COSMOS3_CHECKPOINT_DIR:/model:ro" \
  -p 127.0.0.1:8000:8000 \
  -w /app/vllm-omni \
  vllm/vllm-omni:cosmos3 \
  vllm serve /model \
    --served-model-name nvidia/Cosmos3-Nano-Sim-Bimanual \
    --omni --model-class-name Cosmos3NanoSimBimanualPipeline \
    --deploy-config vllm_omni/deploy/cosmos3_nano_sim_bimanual.yaml \
    --stage-overrides '{"0":{"engine_backend":"default","enforce_eager":false,"diffusion_compile_granularity":"regional","diffusion_compile_dynamic":true,"vae_fast_path":"channels_last","model_config":{"guardrails":false,"use_cuda_graphs":true,"kv_cache_inference_size":64,"attention_sink_size":8}}}' \
    --host 0.0.0.0 --port 8000 --init-timeout 1800
```

The container reuses the host Hugging Face cache and listens on all container interfaces. Docker publishes port 8000 on the host loopback interface, matching `http://127.0.0.1:8000`. For a remote notebook, forward the port with `ssh -L 8000:127.0.0.1:8000 <server-host>`, or publish it on a reachable interface with `-p 8000:8000` and set `VLLM_BASE_URL`. The working directory selects the deployment YAML files bundled in the image.

Use the same `COSMOS3_MODEL_REVISION` in the server shell and notebook client. Restart the server when changing checkpoint revisions.

This model-validation command disables guardrails through `"guardrails":false` in the stage `model_config`. Apply the deployment's guardrail policy in other applications. Use the port-forwarding instructions above when the notebook runs on another host.

## 3. Set up the client

The notebook client needs Python 3.10 or newer, FFmpeg/FFprobe, and Hugging Face read access. Install its lightweight dependencies in the active notebook environment:

```bash
pip install httpx huggingface_hub numpy orjson pillow ipython
```

Set `VLLM_BASE_URL` when the server is not local. Set `VLLM_API_KEY` if the server uses one. `COSMOS3_OUTPUT_DIR` selects the output directory. `COSMOS3_MODEL_REVISION` selects the same checkpoint revision as the server; set it before starting the notebook kernel, or edit `MODEL_REVISION` in the next cell.

```python
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote
from uuid import uuid4
import json
import os
import shutil
import subprocess
import time

import httpx
import numpy as np
import orjson
from huggingface_hub import hf_hub_download
from IPython.display import Video, display
from PIL import Image

MODEL_ID = "nvidia/Cosmos3-Nano-Sim-Bimanual"
MODEL_REVISION = os.environ.get("COSMOS3_MODEL_REVISION", "main")
EXAMPLE_REVISION = os.environ.get("COSMOS3_EXAMPLE_REVISION", "main")
CASES = [
    {"manifest": "examples/inference/sim_bimanual_unifiedv1_hf.jsonl", "action": "agibot_001_normalized_unifiedv1.json"},
    {"manifest": "examples/inference/sim_bimanual_mixed_action_hf.jsonl", "action": "agibot_mixed_028_normalized_mixed_action.json"},
]
VLLM_OMNI_IMAGE = "vllm/vllm-omni:cosmos3"
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY")
HEADERS = {"Authorization": f"Bearer {VLLM_API_KEY}"} if VLLM_API_KEY else {}
HTTP_TIMEOUT = 1860
OUTPUT_DIR = Path(os.environ.get("COSMOS3_OUTPUT_DIR", "outputs/vllm_omni/bimanual")).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Server:", VLLM_BASE_URL, "(with API key)" if VLLM_API_KEY else "")
print("Model:", MODEL_ID, "@", MODEL_REVISION)
print("vLLM-Omni image:", VLLM_OMNI_IMAGE)
print("Outputs:", OUTPUT_DIR)
```

```python
def download_hosted_asset(url: str) -> Path:
    prefix = f"https://huggingface.co/{MODEL_ID}/resolve/main/"
    if not url.startswith(prefix):
        raise ValueError(f"Expected a published {prefix} URL, got {url!r}")
    filename = unquote(url.removeprefix(prefix))
    return Path(hf_hub_download(repo_id=MODEL_ID, filename=filename, revision=EXAMPLE_REVISION))


def validate_actions(row: dict, sidecar: dict, metadata: dict) -> tuple[np.ndarray, list[int]]:
    contract = metadata["conditioning"]["input_contract"]
    count = row["num_frames"] - 1
    if count <= 0 or count != row["action_chunk_size"]:
        raise ValueError("action_chunk_size must equal num_frames - 1 and be positive")
    if count % metadata["temporal_compression_factor"]:
        raise ValueError("Action count must align with the checkpoint's temporal compression factor")
    if row["raw_action_dim"] != contract["action_dim"] or contract["action_dim"] != 59:
        raise ValueError("Expected 59D action rows matching the checkpoint")
    if sidecar["action_space"] != contract["action_space"]:
        raise ValueError("Action space does not match the checkpoint")
    actions = np.asarray(sidecar["action"], dtype=np.float32)
    if actions.shape != (count, contract["action_dim"]) or not np.isfinite(actions).all():
        raise ValueError(f"Expected {count} finite 59D actions, got {actions.shape}")
    names = sidecar["domain_names"]
    if isinstance(names, str):
        names = [names] * count
    if len(names) != count:
        raise ValueError("Supply one domain name per action, or one scalar domain name")
    embodiments = metadata["conditioning"]["embodiments"]
    unknown = sorted(set(names) - embodiments.keys())
    if unknown:
        raise ValueError(f"Unknown action domains: {unknown}")
    if row["num_steps"] != len(metadata["fixed_step_sampler_config"]["t_list"]):
        raise ValueError("num_steps must match the checkpoint's saved sampling schedule")
    if row["format_prompt_as_json"] is not False:
        raise ValueError("The released checkpoint requires a plain-text prompt")
    if not np.isfinite(row["fps"]) or row["fps"] <= 0:
        raise ValueError("fps must be positive and finite")
    return actions, [embodiments[name]["domain_id"] for name in names]


def expected_video_info(row: dict, image: Image.Image, metadata: dict) -> dict:
    alignment = metadata["vae_spatial_compression_factor"]
    return {
        "width": image.width // alignment * alignment,
        "height": image.height // alignment * alignment,
        "fps": Fraction(str(row["fps"])), "frames": row["num_frames"],
    }


def wait_for_model(client: httpx.Client, model: str, timeout: float = 1800) -> list[str]:
    deadline = time.monotonic() + timeout
    while True:
        try:
            response = client.get("/v1/models", timeout=30)
            if response.status_code < 500:
                response.raise_for_status()
                served = [item["id"] for item in response.json().get("data", []) if item.get("id")]
                if model not in served:
                    raise ValueError(f"Expected {model!r}; server exposes {served}")
                return served
        except httpx.TransportError:
            pass
        if time.monotonic() + 5 > deadline:
            raise TimeoutError(f"No ready server at {client.base_url} after {timeout:.0f}s")
        time.sleep(5)


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "fps": Fraction(stream["avg_frame_rate"]), "frames": int(stream["nb_read_frames"]),
    }


for executable in ("ffmpeg", "ffprobe"):
    if not shutil.which(executable):
        raise RuntimeError(f"Install {executable} and add it to PATH")
```

## 4. Download and validate the published examples

Download the selected initial images, plain-text prompts, and normalized 59D action sidecars. Frame count, FPS, and domain routing come from each manifest row; the server pads actions to 64 channels.

```python
checkpoint_path = Path(hf_hub_download(repo_id=MODEL_ID, filename="checkpoint.json", revision=MODEL_REVISION))
metadata = json.loads(checkpoint_path.read_text())["cosmos3_nano_sim_bimanual"]
transformer_config_path = Path(hf_hub_download(repo_id=MODEL_ID, filename="transformer/config.json", revision=MODEL_REVISION))
transformer_config = json.loads(transformer_config_path.read_text())
text_cache_max_len = transformer_config["cosmos3_nano_sim_bimanual"]["text_cache_max_len"]
if text_cache_max_len < 1024:
    raise ValueError("Expected a checkpoint with text_cache_max_len >= 1024.")
conditioning = metadata["conditioning"]
schedule = metadata["fixed_step_sampler_config"]["t_list"]
if conditioning["schema_version"] != 5 or conditioning["input_contract"]["action_space"] != "normalized_unified_v1":
    raise ValueError("Expected the released schema-5 normalized_unified_v1 checkpoint")
samples = []
for case in CASES:
    manifest_path = Path(hf_hub_download(repo_id=MODEL_ID, filename=case["manifest"], revision=EXAMPLE_REVISION))
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    row = next(item for item in rows if item["action_path"].endswith("/" + case["action"]))
    image = Image.open(download_hosted_asset(row["vision_path"])).convert("RGB")
    sidecar = json.loads(download_hosted_asset(row["action_path"]).read_text())
    actions, domain_ids = validate_actions(row, sidecar, metadata)
    samples.append({"row": row, "image": image, "sidecar": sidecar, "actions": actions, "domain_ids": domain_ids})
    print(f"{row['name']}: {image.width}x{image.height}, {row['num_frames']} frames, seed {row['seed']}")
    display(image)
print("Caption cache capacity:", text_cache_max_len, "tokens")
print("Four-step schedule:", schedule)
print(f"Prepared {len(samples)} examples")
```

## 5. Generate the videos

Each request uploads its selected initial image and complete action trajectory. Frame count and FPS follow the manifest, and each rollout uses a fresh session.

```python
outputs = []
with httpx.Client(base_url=VLLM_BASE_URL, headers=HEADERS, timeout=HTTP_TIMEOUT) as client:
    print("Server ready; serving", wait_for_model(client, MODEL_ID))
    for sample in samples:
        row, image, sidecar = sample["row"], sample["image"], sample["sidecar"]
        extra_params = {
            "session_id": uuid4().hex, "reset": True, "close_session": True,
            "action_space": sidecar["action_space"], "action": sample["actions"],
            "domain_id": sample["domain_ids"], "resolution": int(row["resolution"]),
        }
        extra_json = orjson.dumps(extra_params, option=orjson.OPT_SERIALIZE_NUMPY)
        if len(extra_json) > 1024 * 1024:
            raise ValueError("Complete action payload exceeds the server's 1 MiB multipart-field limit; use a server configured for larger form fields.")
        decoded_actions = np.asarray(json.loads(extra_json)["action"], dtype=np.float32)
        if not np.array_equal(decoded_actions, sample["actions"]):
            raise ValueError("Action serialization must preserve every float32 value")
        form = {
            "model": MODEL_ID, "prompt": row["prompt"],
            "num_frames": str(row["num_frames"]), "fps": str(row["fps"]),
            "seed": str(row["seed"]), "extra_params": extra_json.decode("utf-8"),
        }
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        files = {"input_reference": ("conditioning.png", image_bytes.getvalue(), "image/png")}
        sample_dir = OUTPUT_DIR / row["name"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        video_path = sample_dir / f"{row['name']}.mp4"
        (sample_dir / "request.json").write_text(json.dumps(form, indent=2) + "\n")
        started = time.monotonic()
        response = client.post("/v1/videos/sync", data=form, files=files, headers={"Accept": "video/mp4"})
        if response.is_error:
            (sample_dir / "response_error.txt").write_text(response.text)
            response.raise_for_status()
        if not response.headers.get("content-type", "").startswith("video/mp4"):
            raise ValueError(f"Expected video/mp4, got {response.headers.get('content-type')!r}")
        video_path.write_bytes(response.content)
        elapsed_seconds = round(time.monotonic() - started, 1)
        outputs.append({**sample, "sample_dir": sample_dir, "video_path": video_path, "elapsed_seconds": elapsed_seconds})
        print(f"Saved {video_path} in {elapsed_seconds:.1f}s")
```

## 6. Validate and view

Fully decode each result and match its frame count and FPS to the selected manifest row. Published inputs use the model's 480p preprocessing contract: spatial dimensions are aligned to the checkpoint's VAE factor rather than required to be 640×352. Inspect early, middle and late portions of longer rollouts for drift, identity changes and action response.

```python
for output in outputs:
    row, video_path = output["row"], output["video_path"]
    info = probe_video(video_path)
    expected = expected_video_info(row, output["image"], metadata)
    if info != expected:
        raise ValueError(f"Unexpected video metadata for {row['name']}: {info}; expected {expected}")
    decode = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(video_path), "-f", "null", "-"],
        capture_output=True, text=True,
    )
    if decode.returncode or decode.stderr.strip():
        raise RuntimeError(f"Full decode failed: {decode.stderr[:1000]}")
    receipt = {
        "name": row["name"], "status": "success", "video": str(video_path),
        "elapsed_seconds": output["elapsed_seconds"], "frames": info["frames"],
        "model_revision": MODEL_REVISION, "text_cache_max_len": text_cache_max_len,
        "fps": float(info["fps"]), "width": info["width"], "height": info["height"],
    }
    (output["sample_dir"] / "sample_outputs.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    display(Video(filename=str(video_path), embed=True, width=640))
```

Inspect the complete rollout for frozen frames, discontinuities, and disagreement with the supplied actions. In the default mixed-domain example, inspect the transitions around 10 and 20 seconds. For other inputs, derive transition times from their action rows and FPS. Use a new output directory when changing the checkpoint, inputs, or server settings. The [SGLang notebook](sglang.md) runs the published mixed-action example through SGLang's asynchronous Videos API.
