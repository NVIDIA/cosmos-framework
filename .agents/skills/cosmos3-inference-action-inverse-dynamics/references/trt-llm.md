<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Action Inverse Dynamics with TensorRT-LLM

Recover an autonomous-driving action trajectory from the checked-in observation clip. The
`av` domain preset selects the 60-step, 9D trajectory shape and the trained 480p/10 fps recipe.

> **Skill note (not in the source):** this file is the cookbook notebook `run_id_with_trt_llm.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/generator/action/run_id_with_trt_llm.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

## Start the TensorRT-LLM server

Follow the shared [TensorRT-LLM setup](../../cookbooks/cosmos3/README.md#tensorrt-llm-generator), then run this
from the TensorRT-LLM checkout. This notebook is a client; keep the server in a separate terminal.

```bash
export TRTLLM_ROOT="${TRTLLM_ROOT:-$PWD}"
trtllm-serve nvidia/Cosmos3-Nano \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-nano-1gpu.yaml" \
  --port 8000
```

Install client-side decoding dependencies in this notebook kernel if needed:

```bash
pip install requests safetensors torch imageio imageio-ffmpeg pillow
```

```python
from pathlib import Path
import os


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "cookbooks" / "cosmos3").is_dir():
            return path
    return start


COSMOS_ROOT = find_repo_root(Path.cwd().resolve())
ACTION_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3" / "generator" / "action"
OUTPUT_ROOT = Path(
    os.environ.get("COSMOS3_TRTLLM_ACTION_OUTPUT_ROOT", ACTION_ROOT / "outputs" / "notebooks" / "trt_llm")
).resolve()
TRTLLM_BASE_URL = os.environ.get("COSMOS3_TRTLLM_BASE_URL", "http://localhost:8000").rstrip("/")
TRTLLM_API_KEY = os.environ.get("COSMOS3_TRTLLM_API_KEY", "tensorrt_llm")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("COSMOS_ROOT:", COSMOS_ROOT)
print("ACTION_ROOT:", ACTION_ROOT)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("TRTLLM_BASE_URL:", TRTLLM_BASE_URL)
```

## TensorRT-LLM response contract

The notebooks use the blocking `POST /v1/videos/sync` route. Every Action request sets
`format=safetensors` and `response_format=file`; the response is `application/octet-stream` with named `video`, `action`,
and `frame_rate` tensors. This differs from vLLM-Omni's top-level action metadata.

TensorRT-LLM also implements the asynchronous `POST /v1/videos` route. It returns HTTP 202,
can be polled through `GET /v1/videos/{id}`, and serves the same tensor payload from
`GET /v1/videos/{id}/content`.

```python
import json
import mimetypes
import time

import imageio.v3 as iio
import requests
from IPython.display import Video, display
from safetensors.torch import load as load_safetensors


def server_root_url() -> str:
    return TRTLLM_BASE_URL[:-3] if TRTLLM_BASE_URL.endswith("/v1") else TRTLLM_BASE_URL


def video_api_url() -> str:
    root = TRTLLM_BASE_URL if TRTLLM_BASE_URL.endswith("/v1") else f"{TRTLLM_BASE_URL}/v1"
    return f"{root}/videos/sync"


def wait_for_server(timeout_s: int = 1800, interval_s: int = 10) -> None:
    deadline = time.time() + timeout_s
    url = f"{server_root_url()}/health"
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=10)
            if response.ok:
                print("TensorRT-LLM server is ready:", url)
                return
        except requests.RequestException as exc:
            print("waiting for TensorRT-LLM:", exc)
        time.sleep(interval_s)
    raise TimeoutError(f"TensorRT-LLM did not become ready at {url}")


def submit_action(prompt: str, reference_path: Path, extra_params: dict, output_path: Path) -> dict:
    """Call the synchronous API and decode its action/video tensor payload."""
    reference_path = reference_path.resolve()
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    media_type = mimetypes.guess_type(reference_path.name)[0] or "application/octet-stream"
    form = {
        "prompt": prompt,
        "format": "safetensors",
        "response_format": "file",
        "seed": "0",
        "extra_params": json.dumps(extra_params, separators=(",", ":")),
    }
    headers = {"Accept": "application/octet-stream"}
    if TRTLLM_API_KEY:
        headers["Authorization"] = f"Bearer {TRTLLM_API_KEY}"
    with reference_path.open("rb") as reference_file:
        response = requests.post(
            video_api_url(),
            data=form,
            files={"video_reference": (reference_path.name, reference_file, media_type)},
            headers=headers,
            timeout=3600,
        )
    if not response.ok:
        raise RuntimeError(f"TensorRT-LLM request failed ({response.status_code}): {response.text}")
    content_type = response.headers.get("content-type", "")
    if "application/octet-stream" not in content_type:
        raise RuntimeError(f"Expected a tensor payload, got {content_type!r}")
    output_path.write_bytes(response.content)
    payload = load_safetensors(response.content)
    missing = {"video", "action"} - payload.keys()
    if missing:
        raise RuntimeError(f"TensorRT-LLM action payload is missing {sorted(missing)}; got {sorted(payload)}")
    print("saved tensor payload:", output_path)
    print("video:", tuple(payload["video"].shape), payload["video"].dtype)
    print("action:", tuple(payload["action"].shape), payload["action"].dtype)
    return payload


def save_and_view_video(payload: dict, output_path: Path) -> Path:
    frames = payload["video"].detach().cpu().numpy()
    if frames.ndim == 5 and frames.shape[0] == 1:
        frames = frames[0]
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected video [T,H,W,3], got {frames.shape}")
    fps_tensor = payload.get("frame_rate")
    fps = float(fps_tensor.item()) if fps_tensor is not None else 24.0
    iio.imwrite(output_path, frames, fps=fps)
    print("saved rollout:", output_path, "fps:", fps)
    display(Video(str(output_path), embed=True))
    return output_path
```

## Prepare the checked-in AV observation

```python
video_path = ACTION_ROOT / "assets" / "videos" / "av_0.mp4"
if not video_path.exists():
    raise FileNotFoundError(video_path)

prompt = "You are an autonomous vehicle planning system."
extra_params = {
    "action_mode": "inverse_dynamics",
    "domain_name": "av",
    "use_guardrails": True,
}
print("observation:", video_path)
print(json.dumps(extra_params, indent=2))
display(Video(str(video_path), embed=True))
```

## Run inverse dynamics

```python
wait_for_server()
id_payload = submit_action(
    prompt,
    video_path,
    extra_params,
    OUTPUT_ROOT / "inverse_dynamics_av.safetensors",
)
```

## Inspect the reconstructed video and predicted trajectory

```python
save_and_view_video(id_payload, OUTPUT_ROOT / "inverse_dynamics_av.mp4")
id_action = id_payload["action"].detach().cpu().numpy()
print("predicted action shape:", id_action.shape)
print("first five predicted action rows:")
print(id_action[:5])
```

## Skill notes (not in the source) — install routes (from SKILL.md Variant 5)

### 1. "Follow the shared TensorRT-LLM setup": the setup guide's source build, quoted

The setup guide (`../../cookbooks/cosmos3/README.md`, "TensorRT-LLM Generator" — in the framework checkout:
`.agents/skills/cookbooks/cosmos3/README.md`): "Install TensorRT-LLM following its upstream documentation. To build
TensorRT-LLM from source, follow NVIDIA's Build from Source guide. This is the right path when you need a checkout that
contains a recent Cosmos3 VisualGen change before it is available in your installed package or release image."

```bash
apt-get update && apt-get -y install git git-lfs
git lfs install

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout cc63e66daf52644f1441e117bb7ba67d20566016
git submodule update --init --recursive
git lfs pull

# Pick a devel tag from the upstream build-from-source guide or NGC.
# Skill note (not in the source; eval 2026-11-08, two cells): a devel tag newer than the checkout fails at cmake configure with
#   `undefined symbol: ncclGetPeerDeviceLeInfo` (its torch links a newer NCCL than requirements.txt's `nvidia-nccl-cu13` pin;
#   seen with devel:1.4.0rc3 on this checkout) — verified remedy: comment out that pin in requirements.txt before build_wheel.py;
#   `-a 90-real` builds H100 only (~33 min on 256 cores; the clone with LFS took ~17 min — longer than a 10-minute tool cap, so background it).
docker pull nvcr.io/nvidia/tensorrt-llm/devel:<tag>
docker run --rm -it \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all \
  --volume "$PWD":"$PWD" \
  --workdir "$PWD" \
  nvcr.io/nvidia/tensorrt-llm/devel:<tag>

# Inside the container:
apt-get update && apt-get -y install ffmpeg
python3 scripts/build_wheel.py --use_ccache --skip_building_wheel --linking_install_binary
pip install -e .
```

"For Python-only changes, the upstream guide also documents `TRTLLM_USE_PRECOMPILED=1 pip install -e .` to reuse
precompiled binaries while installing the checkout in editable mode." Skill note (not in the source): do not — it resolves
an unpublished `+<sha>` version and, when forced to a published wheel, an ABI mismatch
(`../../cosmos3-inference-generate-media/references/trt-llm.md` § 7.5; eval 2026-11-08, one cell).

"Then install the Cosmos3 guardrail package in the same environment unless you explicitly disable guardrails before
starting the server:"

```bash
pip install cosmos_guardrail==0.3.2
# On headless servers without libGL.so.1, replace the OpenCV wheel pulled in by
# cosmos_guardrail with the matching headless build:
pip uninstall -y opencv-python
pip install opencv-python-headless==5.0.0.93
```

Skill note (not in the source): the smoke path disables guardrails (SKILL.md Variant 5, "Guardrails"), so the package is not
needed there.

### 2. The install this skill runs: Route A of the generate-media skill (skill note, not in the source)

With or without Docker (the § 1 source build is the fallback, not the smoke path), run the Route A block of
`../../cosmos3-inference-generate-media/references/trt-llm.md` § 7.2 as ONE shell call, verbatim: the `tensorrt_llm==1.3.0rc27` wheel
(`torch==2.13.0+cu130`, `torchvision==0.28.0+cu130`, OpenMPI and `ffmpeg` from apt), its `export CUDA_HOME=…/nvidia/cu13`
line and the pinned checkout `bca6761ab84fbcd58fc7f914eade7de48b32e35e` fetched by SHA as `TRTLLM_ROOT` (the checkout
supplies only `examples/visual_gen/configs/cosmos3-nano-1gpu.yaml`; § 7.3). Verified for this action route by the eval
2026-11-08 (one 80 GB GPU): the request answered in ~16 s with
`video (61, 480, 832, 3) uint8`, `action (60, 9) float32`; the whole cell took 8.7 min against 47–67 min
for the source-build attempts.

- `CUDA_HOME` must be set in the shell call that runs `trtllm-serve` (DeepGEMM asserts `_find_cuda_home()` at import, so
  the server exits before `/health` is up); the SKILL.md serve block re-exports § 7.2's line for that reason, and puts the
  venv first on `PATH` so the notebook's bare `trtllm-serve` is the venv's.
- "Host NVDEC/NVENC driver libraries available" (SKILL.md): the request's `video_reference` is decoded on the GPU
  (PyNvVideoCodec loads the driver's `libnvcuvid.so.1` and `libnvidia-encode.so.1` at the first request; `/health` answers
  without them), so they must load where `trtllm-serve` runs — check that `/sbin/ldconfig -p | grep -E 'nvcuvid|nvidia-encode'`
  lists both before the launch. A container gets them from the host driver only with
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` set when it is created (`docker run -e`, image ENV or pod spec);
  otherwise install the driver's codec packages at the exact `nvidia-smi` version (`libnvidia-decode-<branch>` /
  `libnvidia-encode-<branch>` from the NVIDIA CUDA apt repository, or copy the two libraries into a directory `ldconfig`
  scans, then run `ldconfig`) and restart the server. Missing: `POST /v1/videos/sync` answers HTTP 500 `Failed to load NVENC
  library` (eval 2026-11-08, two cells); libraries of another driver version: HTTP 400 `cuvidGetDecoderCaps returned error 100`.
- The client cells import `IPython.display`; the SKILL.md client line adds `ipython` (and `av` for the MP4 check of "Verify
  the output") to the notebook's list and creates the client venv with `uv venv --no-config`, so the framework checkout's
  `uv` pin does not apply to it.
