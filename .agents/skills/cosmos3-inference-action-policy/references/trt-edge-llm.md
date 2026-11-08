# Cosmos3-Edge Policy with TensorRT-Edge-LLM

This notebook runs **Cosmos3-Edge-Policy-DROID** with
[TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) (v0.11.0):
build the policy engines with the direct builder, then run
`cosmos3_policy_inference`.

Policy sits under the **Generator / Action** cookbook surface (action chunks),
not the Reasoner text path. For Edge multimodal reasoning on the same runtime,
see [`../../reasoner/run_with_trt_edge_llm.ipynb`](../../cookbooks/cosmos3/reasoner/run_with_trt_edge_llm.ipynb).

1. Points at a TensorRT-Edge-LLM v0.11.0 checkout built with
   `-DBUILD_EXPERIMENTAL_MODELS=ON` and `-DENABLE_CUTE_DSL=ALL`, and a Python
   environment with `pip install -e ".[builder]"`.
2. Builds `und-prefill`, `gen`, and `vae-encoder` with
   `tensorrt-edgellm-build` into `ENGINE_DIR/policy`. v0.11.0
   `tensorrt-edgellm-export --task policy` writes the UND input
   `attention_pos_id` and VAE `height` / `width` only. The v0.11.0 runtime
   binds `attention_position_ids` and reads `builder_config.content_height` /
   `content_width`. The direct builder writes that contract. DROID content is
   640×540 inside the 736×544 canvas.
3. Builds a 640×540 DROID `concat_view` from `assets/droid_lerobot_example`
   (same layout as the vLLM-Omni / SGLang policy notebooks) and writes
   `action.json`.

Complete the upstream
[Installation](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/installation.html)
and
[direct engine builder](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/direct-engine-builder.html)
guides first.

> **Skill note (not in the source) — x86 pre-flight for the v0.11.0 build (eval 2026-11-08, one H100, CUDA 13).** Do these before § 2, inside a CUDA development container (the C++ runtime build needs the CUDA toolkit and the TensorRT headers); each row is "symptom if skipped → what to run". Engines are version-locked: build and run them with the same TensorRT.

| Symptom if skipped (skill note, not in the source)                                                              | What to run                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tensorrt-edgellm-build`: `AttributeError: 'INetworkDefinition' object has no attribute 'add_rotary_embedding'` | TensorRT is older than the v0.11.0-qualified builds (`tensorrt_edgellm/_native/dependencies.py`: `tensorrt==10.16.0.72` for TensorRT 10 payloads other than DGX Spark, `==10.16.1.11` for DGX Spark, `==11.0.0.114` for TensorRT 11 x86; the NGC `pytorch:25.09` / `25.12` images ship 10.13 / 10.14). Simplest: build inside `nvcr.io/nvidia/pytorch:26.03-py3` (TensorRT 10.16.0.72 + CUDA 13.2 per the NGC support matrix — not run by this eval) or `:26.06-py3` (11.0.0.114). Eval-verified fallback in an older image: `pip install --extra-index-url https://pypi.nvidia.com tensorrt-cu13==10.16.0.72` (without NVIDIA's index pip resolves a stub `nvidia-cuda-runtime-cu13`) plus the matching 10.16.0.72 debs `libnvinfer10 libnvinfer-dev libnvinfer-headers-dev libnvinfer-headers-plugin-dev libnvinfer-plugin10 libnvinfer-plugin-dev libnvonnxparsers10` from the NVIDIA CUDA apt repository (`apt-get download` + `dpkg -x` into `/opt/TensorRT-10.16.0.72` when apt holds 10.14; without `libnvinfer-headers-plugin-dev` the C++ build stops at `fatal error: NvInferPlugin.h`). Configure with `-DTRT_PACKAGE_DIR=<dir holding lib/ and include/> -DCUDA_CTK_VERSION=<toolkit major.minor, e.g. 13.2> -DCUTE_DSL_ARTIFACT_TAG=sm_90` next to the two documented flags, and keep that `lib` dir on `LD_LIBRARY_PATH` for § 2 and § 4. |
| CMake configure: `Prebuilt CuTe DSL library not found: …/cuteDSLArtifact/x86_64/libcutedsl_x86_64.a`            | a source checkout ships no x86 CuTe DSL artifact. In the builder venv: `pip install "nvidia-cutlass-dsl[cu13]==4.7.0" cupy-cuda13x==13.6.0 && python kernelSrcs/build_cutedsl.py --gpu_arch sm_90` (4.7.0 exactly — the script rejects 4.8.0; the `[cu13]` extra, not the default cu12 libraries; `cupy-cuda12x==12.3.0` on CUDA 12; `--kernels` defaults to ALL and `--arch` auto-detects the host; it writes `cpp/kernels/cuteDSLArtifact/x86_64/sm_90/` offline, without probing a GPU), then add `-DCUTE_DSL_ARTIFACT_TAG=sm_90` to the configure next to `-DBUILD_EXPERIMENTAL_MODELS=ON -DENABLE_CUTE_DSL=ALL`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| § 2: `ModuleNotFoundError: No module named 'huggingface_hub'` / `huggingface-cli: command not found`            | `.[builder]` is an empty alias (base deps: `cuda-python`, `numpy`). In a fresh venv: `pip install -e ".[builder,export]"` (pulls `torch==2.13.0`, `transformers`, `huggingface-hub`). Inside an NGC pytorch container (torch present) install only `pip install "huggingface-hub>=1.5,<2"` — `.[export]` would replace the image's torch.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| § 3: `ModuleNotFoundError: No module named 'PIL'` / `'imageio_ffmpeg'`                                          | `pip install pillow imageio-ffmpeg` in the environment that runs § 3 (the NGC pytorch image has Pillow but not imageio-ffmpeg; a fresh venv has neither; the cell's IPython import is already guarded, no ipython needed)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |

> **Skill note (not in the source):** this file is the cookbook notebook `run_policy_with_trt_edge_llm.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/generator/action/run_policy_with_trt_edge_llm.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

## 1. Setup paths

`EDGELLM_ROOT` defaults to `~/TensorRT-Edge-LLM`, the same default as the
shared cookbook README. It must contain
`build/experimental_models/cosmos3/examples/` after a CMake build with
`-DBUILD_EXPERIMENTAL_MODELS=ON` and `-DENABLE_CUTE_DSL=ALL`.
`POLICY_CHECKPOINT` is either a local checkpoint directory or the Hugging Face
repo id `nvidia/Cosmos3-Edge-Policy-DROID`. Policy engines go under
`ENGINE_DIR/policy`, separate from the reasoner engines in
`ENGINE_DIR/reasoning`.

```python
from pathlib import Path
import os


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / ".agents" / "skills" / "cookbooks" / "cosmos3").is_dir():
            return candidate
        if (candidate / "cookbooks" / "cosmos3").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the cosmos-framework repository root")


def cookbook_root(repo: Path) -> Path:
    for relative in (Path(".agents/skills/cookbooks/cosmos3"), Path("cookbooks/cosmos3")):
        candidate = repo / relative
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(repo / ".agents/skills/cookbooks/cosmos3")


COSMOS_ROOT = find_repo_root(Path.cwd().resolve())
DROID_ASSET_ROOT = (
    cookbook_root(COSMOS_ROOT) / "generator" / "action" / "assets" / "droid_lerobot_example"
).resolve()
assert DROID_ASSET_ROOT.exists(), DROID_ASSET_ROOT

EDGELLM_ROOT = Path(
    os.environ.get("EDGELLM_ROOT", Path.home() / "TensorRT-Edge-LLM")
).expanduser().resolve()
ENGINE_DIR = Path(
    os.environ.get(
        "ENGINE_DIR",
        Path.home() / "tensorrt-edgellm-workspace" / "Cosmos3-Edge" / "engines",
    )
).expanduser().resolve()
POLICY_CHECKPOINT = os.environ.get(
    "POLICY_CHECKPOINT", "nvidia/Cosmos3-Edge-Policy-DROID"
)
POLICY_PROMPT = os.environ.get(
    "POLICY_PROMPT",
    "Pick up the object and place it in the target container.",
)
POLICY_STATE = os.environ.get(
    "POLICY_STATE", "0,0,0,0,0,0,0,0"
)
POLICY_ENGINE_DIR = ENGINE_DIR / "policy"
WORK_DIR = POLICY_ENGINE_DIR / "cookbook_work"
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = WORK_DIR / "action.json"

os.environ["COSMOS_ROOT"] = str(COSMOS_ROOT)
os.environ["EDGELLM_ROOT"] = str(EDGELLM_ROOT)
os.environ["ENGINE_DIR"] = str(ENGINE_DIR)
os.environ["POLICY_ENGINE_DIR"] = str(POLICY_ENGINE_DIR)
os.environ["POLICY_CHECKPOINT"] = POLICY_CHECKPOINT
os.environ["DROID_ASSET_ROOT"] = str(DROID_ASSET_ROOT)
os.environ["POLICY_PROMPT"] = POLICY_PROMPT
os.environ["POLICY_STATE"] = POLICY_STATE
os.environ["OUTPUT_JSON"] = str(OUTPUT_JSON)

print("cosmos root:", COSMOS_ROOT)
print("EDGELLM_ROOT:", EDGELLM_ROOT, "(exists:", EDGELLM_ROOT.exists(), ")")
print("ENGINE_DIR:", ENGINE_DIR)
print("POLICY_ENGINE_DIR:", POLICY_ENGINE_DIR)
print("POLICY_CHECKPOINT:", POLICY_CHECKPOINT)
print("DROID_ASSET_ROOT:", DROID_ASSET_ROOT)
print("prompt:", POLICY_PROMPT)
print("state:", POLICY_STATE)
```

## 2. Build the policy engines

Run this on the target GPU after `pip install -e ".[builder]"` in the
TensorRT-Edge-LLM v0.11.0 checkout. The command reads the checkpoint directly
and writes `und_prefill/`, `gen/`, `vae_encoder/`, `text_tokenizer/`, and
`embed_tokens.safetensors` under `POLICY_ENGINE_DIR`.

`--components` keeps the build on the policy towers. The checkpoint also
contains the reasoner configs, and `--components all` would build those too.
`--max-input-len 512` matches the v0.11.0 policy prompt budget. For DROID, the
builder sets `content_height=540` and `content_width=640` from
`checkpoint.json`. A repo id is downloaded with `huggingface-hub` (part of
`pip install -e ".[export]"`); a local checkpoint directory skips that download.

```bash
set -euo pipefail
: "${EDGELLM_ROOT:?run the setup cell first}"
: "${POLICY_ENGINE_DIR:?run the setup cell first}"
: "${POLICY_CHECKPOINT:?run the setup cell first}"

if [[ -f "$POLICY_CHECKPOINT/config.json" ]]; then
  MODEL_DIR="$POLICY_CHECKPOINT"
else
  MODEL_DIR="${CHECKPOINT_DIR:-$HOME/tensorrt-edgellm-workspace/Cosmos3-Edge/Cosmos3-Edge-Policy-DROID}"
  mkdir -p "$MODEL_DIR"
  POLICY_CHECKPOINT="$POLICY_CHECKPOINT" MODEL_DIR="$MODEL_DIR" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["POLICY_CHECKPOINT"],
    local_dir=os.environ["MODEL_DIR"],
)
PY
fi

cd "$EDGELLM_ROOT"
mkdir -p "$POLICY_ENGINE_DIR"
tensorrt-edgellm-build \
  --model-dir "$MODEL_DIR" \
  --engine-dir "$POLICY_ENGINE_DIR" \
  --components und-prefill,gen,vae-encoder \
  --max-input-len 512 \
  --max-batch-size 1

python3 - <<'PY'
import json
import os
from pathlib import Path

engine_dir = Path(os.environ["POLICY_ENGINE_DIR"])
vae = json.loads((engine_dir / "vae_encoder" / "config.json").read_text())
und = json.loads((engine_dir / "und_prefill" / "config.json").read_text())
builder = vae["builder_config"]
assert builder["content_height"] == 540, builder
assert builder["content_width"] == 640, builder
assert "attention_position_ids" in und["optimization_profile"]
assert (engine_dir / "embed_tokens.safetensors").is_file()
assert (engine_dir / "text_tokenizer" / "tokenizer.json").is_file()
print(
    "policy contract: content",
    f"{builder['content_width']}x{builder['content_height']}",
    "canvas",
    f"{builder['width']}x{builder['height']}",
)
PY

echo "Built policy engines under $POLICY_ENGINE_DIR"
```

## 3. Prepare a DROID observation

This cell extracts the first frame from each checked-in DROID camera video and
builds the 640×540 `concat_view` used by the vLLM-Omni / SGLang policy
notebooks (wrist on top, two exterior views below). Pass that composite to
`--image`. The v0.11.0 runtime fits this 640×540 concat into the 736×544 canvas using
`content_width` / `content_height`. Pass the DROID multiview, not a reasoner
caption asset.

Override `POLICY_PROMPT` in the setup cell if your instruction differs. The
upstream VLA guide's banana/bowl string is only a CLI placeholder.

```python
import subprocess

from PIL import Image

try:
    import imageio_ffmpeg
except ImportError as exc:
    raise RuntimeError(
        "Install imageio-ffmpeg in this notebook kernel: pip install imageio-ffmpeg"
    ) from exc

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
CAMERA_VIDEO_PATHS = {
    "observation/wrist_image_left": DROID_ASSET_ROOT
    / "videos"
    / "observation.image.wrist_image_left"
    / "chunk-000"
    / "file-000.mp4",
    "observation/exterior_image_1_left": DROID_ASSET_ROOT
    / "videos"
    / "observation.image.exterior_image_1_left"
    / "chunk-000"
    / "file-000.mp4",
    "observation/exterior_image_2_left": DROID_ASSET_ROOT
    / "videos"
    / "observation.image.exterior_image_2_left"
    / "chunk-000"
    / "file-000.mp4",
}
for key, video_path in CAMERA_VIDEO_PATHS.items():
    assert video_path.exists(), f"missing {key}: {video_path}"


def extract_first_frame(video_path: Path, out_path: Path) -> Path:
    if not out_path.exists() or out_path.stat().st_mtime < video_path.stat().st_mtime:
        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                str(out_path),
            ],
            check=True,
        )
    return out_path


frame_paths = {
    key: extract_first_frame(video_path, WORK_DIR / f"policy_{key.split('/')[-1]}.png")
    for key, video_path in CAMERA_VIDEO_PATHS.items()
}
frames = {key: Image.open(path).convert("RGB") for key, path in frame_paths.items()}

# Same geometry as run_policy_with_vllm_omni.ipynb / run_policy_with_sglang.ipynb:
# native 640×360 wrist on top; exteriors resized to 320×180 on the bottom row.
wrist = frames["observation/wrist_image_left"]
source_size = wrist.size
assert all(frame.size == source_size for frame in frames.values()), (
    "DROID camera frames must have matching dimensions",
    {key: frame.size for key, frame in frames.items()},
)
target_w = wrist.width
bottom_h, half_w = wrist.height // 2, wrist.width // 2
target_h = wrist.height + bottom_h
left = frames["observation/exterior_image_1_left"].resize(
    (half_w, bottom_h), Image.Resampling.BILINEAR
)
right = frames["observation/exterior_image_2_left"].resize(
    (half_w, bottom_h), Image.Resampling.BILINEAR
)
policy_image = Image.new("RGB", (target_w, target_h))
policy_image.paste(wrist, (0, 0))
policy_image.paste(left, (0, wrist.height))
policy_image.paste(right, (half_w, wrist.height))

OBSERVATION_IMAGE = WORK_DIR / "droid_policy_first_frame.png"
policy_image.save(OBSERVATION_IMAGE)
os.environ["OBSERVATION_IMAGE"] = str(OBSERVATION_IMAGE)

print("observation:", OBSERVATION_IMAGE, policy_image.size)
print("prompt:", POLICY_PROMPT)
try:
    from IPython.display import display

    display(policy_image)
except Exception:
    pass
```

## 4. Run the policy

`--image` is the 640×540 DROID concat from the previous cell. Pass
`--viewPoint concat_view` so the policy prompt matches that multiview layout
(the CLI default is `ego_view`). v0.11.0 `cosmos3_policy_inference` accepts
this flag. Policy-DROID engines are state-conditioned: `--state` is eight
comma-separated model-space values. The setup cell's `POLICY_STATE` is a
zero placeholder; replace it with the robot's current state. For an ordered
frame list, replace `--image`
with `--video frame_00.png,frame_01.png,frame_02.png`. The current policy
conditions on the most recent frame. Optional: `--steps` (denoise steps) and
`--seed`.

```bash
set -euo pipefail
: "${EDGELLM_ROOT:?run the setup cell first}"
: "${POLICY_ENGINE_DIR:?run the setup cell first}"
: "${OBSERVATION_IMAGE:?run the DROID observation cell first}"
: "${POLICY_PROMPT:?run the setup cell first}"
: "${POLICY_STATE:?run the setup cell first}"
: "${OUTPUT_JSON:?run the setup cell first}"

cd "$EDGELLM_ROOT"
./build/experimental_models/cosmos3/examples/cosmos3_policy_inference \
  --engineDir "$POLICY_ENGINE_DIR" \
  --image "$OBSERVATION_IMAGE" \
  --prompt "$POLICY_PROMPT" \
  --state "$POLICY_STATE" \
  --viewPoint concat_view \
  --output "$OUTPUT_JSON"

echo "Wrote $OUTPUT_JSON"
python3 - <<'PY'
import json
import os
from pathlib import Path
data = json.loads(Path(os.environ["OUTPUT_JSON"]).read_text())
print(json.dumps(data, indent=2)[:4000])
PY
```

## 5. Next steps

- Override `POLICY_PROMPT` in the setup cell, or rebuild the concat with your
  own DROID camera videos.
- Multimodal reasoning on TensorRT-Edge-LLM:
  [`../../reasoner/run_with_trt_edge_llm.ipynb`](../../cookbooks/cosmos3/reasoner/run_with_trt_edge_llm.ipynb).
- OpenAI-compatible policy serving on workstation GPUs:
  [`run_policy_with_vllm_omni.ipynb`](./vllm-omni.md).
- Cosmos Framework policy server:
  [`run_policy_with_cosmos_framework.md`](./cosmos-framework.md).
