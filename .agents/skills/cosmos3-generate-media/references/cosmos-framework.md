# Cosmos Framework (torchrun) — distilled from the cookbook notebook

Scope: the native PyTorch backend — the README quickstart, the notebook's environment variables, framework checkout, `uv` install and GPU check, and the per-use-case `torchrun -m cosmos_framework.scripts.inference` runs for Cosmos3-Nano, Nano FP8, Cosmos3-Super, the Super 4-Step students and Cosmos3-Edge; the payload helpers, asset sets and sampling tables this notebook shares with the other backends are referenced from payloads.md.

## Quickstart (cookbook README)

### Quickstart

Set up the environment: [Cosmos Framework setup](../../cookbooks/cosmos3/README.md#cosmos-framework).
Activate the framework venv created during setup (`packages/cosmos3/.venv` at the
cookbook root, `.agents/skills`), or call its `torchrun` by path. The notebook builds a full inference
payload JSON (not the raw prompt asset alone); build one the same way, then run
from this folder:

```bash
python3 - <<'PY'
import json
from pathlib import Path

prompt = json.dumps(
    json.load(open("assets/prompts/text2video/robot_kitchen.json")),
    ensure_ascii=True,
    separators=(",", ":"),
)
negative = json.dumps(
    json.load(open("assets/negative_prompts/text2video/neg_prompt.json")),
    ensure_ascii=True,
    separators=(",", ":"),
)
payload = {
    "model_mode": "text2video",
    "name": "robot_kitchen",
    "prompt": prompt,
    "negative_prompt": negative,
    "enable_sound": False,
    "num_steps": 35,
    "guidance": 6.0,
    "shift": 10.0,
    "fps": 24,
    "num_frames": 189,
    "resolution": "720",
    "aspect_ratio": "16,9",
    "seed": 0,
}
Path("/tmp/cosmos3_t2v_payload.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

torchrun --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=throughput \
  -i /tmp/cosmos3_t2v_payload.json \
  -o /tmp/cosmos3_t2v_framework \
  --checkpoint-path Cosmos3-Nano \
  --seed=0
```

To run **Cosmos3-Super** instead, set `--checkpoint-path Cosmos3-Super` and use
more GPUs via `--nproc-per-node`.

To run **Cosmos3-Edge** instead, set `--checkpoint-path Cosmos3-Edge`. Edge has
no audio modules, so keep `"enable_sound": False` in the payload. Edge supports
only 256p/480p, so use Edge's generation settings in the payload:
`"resolution": "480"`, `"num_frames": 121`, and `"fps": 24`.

### Notebook walkthrough

`run_with_cosmos_framework.ipynb` is the full
tutorial for the native PyTorch backend: it covers every use case — text-to-image,
text-to-video, image-to-video, with audio on or off — and includes the detailed,
environment-aware setup and visualization for each generation. It also includes
four-GPU inference examples for `nvidia/Cosmos3-Super-Text2Image-4Step` and
`nvidia/Cosmos3-Super-Image2Video-4Step`.

### Distillation training recipe

[`distill/README.md`](../../cookbooks/cosmos3/generator/audiovisual/distill/README.md) documents the short T2I and I2V DMD2
training, resume, and student-only export workflow. The first supported topology
is exactly 8 GB200 nodes with 4 GPUs per node. This is an integration smoke
recipe, not a production reproduction recipe.

## Cosmos3 Generator Audiovisual with Cosmos Framework (notebook)

Licence header: see payloads.md § 2.1.

This notebook runs Cosmos3 audiovisual generation through the native Cosmos Framework PyTorch entrypoint:

```bash
python -m cosmos_framework.scripts.inference
```

Run the use cases top-to-bottom: Cosmos3-Nano, full Cosmos3-Super, the published four-step distilled T2I/I2V students, and Cosmos3-Edge. Each section uses the matching checkpoint explicitly. Cosmos3-Edge has no audio modules, so its examples run without audio.

## 1. Prerequisites

See payloads.md § 2.2 (this notebook's machine, Hugging Face access and disk-space sentence) and § 2.4 (this notebook's **Headless servers** and **uv version** notes).

## 2. Configure Paths and Environment

The defaults are relative to this `cosmos` checkout and use the CUDA 13 or 12.8 dependency group depending on the CUDA version installed on your system (`cu130-train` or `cu128-train`):

```bash
export COSMOS3_REPO=/path/to/cosmos-framework
export COSMOS3_UV_GROUP=cu130-train
export UV_PROJECT_ENVIRONMENT=/path/to/large/uv/venvs/cosmos3-audiovisual
export COSMOS3_NUM_GPUS=4
export HF_HOME=/path/to/large/huggingface/cache
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

Paths (said once for this file): the notebook's `find_repo_root()`, `COSMOS_ROOT = find_repo_root(Path.cwd().resolve())`, `COSMOS3_AUDIOVISUAL_ROOT` and `COSMOS3_AUDIOVISUAL_OUTPUT_ROOT` lines are replaced by the configuration block in payloads.md § 4.1 — run it before the cell below; it sets `COSMOS3_AUDIOVISUAL_ROOT` and `COSMOS_ROOT` to this skill's directory (which holds `assets/`) and exports `COSMOS3_AUDIOVISUAL_OUTPUT_ROOT` from the plain output directory `OUT=outputs`, so payloads and media land under `$OUT/<backend>/...` instead of the notebook's `outputs/notebooks`.

```python
from pathlib import Path
import os
import socket


def free_local_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def default_framework_repo(root: Path) -> Path:
    candidates = [
        root / "packages" / "cosmos-framework",
        root / "packages" / "cosmos3",
    ]
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "cosmos_framework").exists():
            return candidate
    return candidates[0]


COSMOS3_REPO = Path(os.environ.get("COSMOS3_REPO", default_framework_repo(COSMOS_ROOT))).resolve()
COSMOS3_GIT_URL = os.environ.get("COSMOS3_GIT_URL", "git@github.com:NVIDIA/cosmos-framework.git")
COSMOS3_UV_GROUP = os.environ.get("COSMOS3_UV_GROUP", "cu130-train")
COSMOS3_UV_ENV = Path(os.environ.get("UV_PROJECT_ENVIRONMENT", COSMOS3_REPO / ".venv")).resolve()
COSMOS3_NUM_GPUS = os.environ.get("COSMOS3_NUM_GPUS", "4")

os.environ["COSMOS3_REPO"] = str(COSMOS3_REPO)
os.environ["COSMOS3_GIT_URL"] = COSMOS3_GIT_URL
os.environ["COSMOS3_UV_GROUP"] = COSMOS3_UV_GROUP
os.environ["COSMOS3_UV_ENV"] = str(COSMOS3_UV_ENV)
os.environ.setdefault("UV_PROJECT_ENVIRONMENT", str(COSMOS3_UV_ENV))
os.environ["COSMOS3_NUM_GPUS"] = COSMOS3_NUM_GPUS
os.environ.setdefault("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv"))
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
os.environ.setdefault("COSMOS3_MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("COSMOS3_TEXT_MASTER_PORT", free_local_port())
os.environ.setdefault("COSMOS3_IMAGE_MASTER_PORT", free_local_port())
```

## 3. Clone or Reuse Cosmos Framework

```bash
set -euo pipefail

mkdir -p "$(dirname "$COSMOS3_REPO")"

if [ -f "$COSMOS3_REPO/pyproject.toml" ] && [ -d "$COSMOS3_REPO/cosmos_framework" ]; then
  echo "Using existing framework checkout: $COSMOS3_REPO"
elif [ -e "$COSMOS3_REPO" ]; then
  echo "COSMOS3_REPO exists but is not a Cosmos Framework checkout: $COSMOS3_REPO"
  echo "Set COSMOS3_REPO to the staged framework checkout, for example packages/cosmos-framework."
  exit 1
else
  echo "Cloning $COSMOS3_GIT_URL into $COSMOS3_REPO"
  git clone "$COSMOS3_GIT_URL" "$COSMOS3_REPO"
fi

cd "$COSMOS3_REPO"
git status --short --branch
git remote -v
```

## 4. Install Native PyTorch Dependencies

Installs framework dependencies with `cu130-train` by default.

```bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

export GIT_LFS_SKIP_SMUDGE=1
cd "$COSMOS3_REPO"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$COSMOS3_UV_ENV}"
echo "Using UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT"
uv sync --all-extras --group="$COSMOS3_UV_GROUP"
if [ ! -x "$COSMOS3_UV_ENV/bin/python" ]; then
  echo "uv sync completed, but expected Python is missing: $COSMOS3_UV_ENV/bin/python"
  echo "Check UV_PROJECT_ENVIRONMENT above; uv created the environment somewhere else or the install failed."
  exit 1
fi
```

Skill note (not in the notebook): `command -v uv` is not enough — the uv version must equal `tool.uv.required-version` of the checkout (`==0.11.29` in its `pyproject.toml`), which uv applies from any cwd under the checkout; any other uv aborts `uv sync` with exit 2 (`error: Required uv version ==0.11.29 does not match the running version …`). `uv self update 0.11.29` only works for a standalone-installer uv (it refuses for pip/apt/vendored uv: `Self-update is only available for uv binaries installed via the standalone installation scripts`). SKILL.md Variant 2 has the preflight: install the pinned uv side-by-side (`curl -LsSf "https://astral.sh/uv/0.11.29/install.sh" | env UV_INSTALL_DIR="$HOME/.local/uv-0.11.29" INSTALLER_NO_MODIFY_PATH=1 sh`, then put that directory first on `PATH`), or leave the `uv` on PATH alone and run `GIT_LFS_SKIP_SMUDGE=1 uvx --from 'uv==0.11.29' uv sync --all-extras --group="$COSMOS3_UV_GROUP"` from the checkout.

## 5. Verify GPU and Python Environment

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
if [ ! -x "$COSMOS3_UV_ENV/bin/python" ]; then
  echo "Missing $COSMOS3_UV_ENV/bin/python"
  echo "Run the Install Native PyTorch Dependencies cell first, or set UV_PROJECT_ENVIRONMENT/COSMOS3_UV_ENV to the uv environment path."
  exit 1
fi
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$COSMOS3_UV_ENV/bin/python" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(f"device {index}:", torch.cuda.get_device_name(index))
PY
```

## 6. Preview Available Inputs

The preview cell is Jupyter plumbing and is not migrated; the assets it listed and the rules it encoded are in payloads.md § 3 (asset catalogue) and § 3.5 (preview rules).

## 7. Define Asset Sets, Payload Helpers, and Viewer Helpers

Everything this cell defines is shared content and lives in payloads.md; this notebook's versions are identical to the `run_with_cosmos_framework.ipynb` blocks and rows there (no differing lines):

- `IMAGE_EXTENSIONS`, `asset_path`, `compact_json_file`, `resolve_payload_path` — see payloads.md § 9.1.
- `FIXED_SAMPLING` — see payloads.md § 6.1 (§ 6.2 for how a use case's `sampling` override is spread on top of it).
- `ASSET_SETS` (thirteen use cases, with its two leading comments) — see payloads.md § 8.7 (source, verbatim) and §§ 8.1–8.3 (matrix).
- `payload_dimensions` — see payloads.md § 7 (this notebook's two-row version).
- `create_payload(use_case, *, backend)` — see payloads.md § 9.2 (this notebook's version, de-Jupyterised there); the environment variables it exports are in § 4.3 and its payload/output naming in § 4.2.
- `display_video` and `view_run` are the inline viewer (Jupyter plumbing, not migrated); the output discovery rule `view_run` applied is the framework row of payloads.md § 4.2.

## Use Cases

Run each use case top-to-bottom: create the JSON payload, run inference, then view the generated media. The published four-step students use four-GPU sharding for memory and are smoke examples, not production-quality reproduction claims.

The per-use-case "Create Payload / Run / View Results" cells are collapsed into the one table below (every `ASSET_SETS` use case, in notebook order): Create Payload is the row's `create_payload(...)` call (payloads.md § 9.2; payload/output naming § 4.2, exported `_INPUT`/`_OUTPUT` variables § 4.3), Run is the `torchrun` command of the row's run shape — standard (the Run block under "Nano: Text to Image"), FP8 (under "Nano FP8: Text to Image") or four-step (under "Super 4-Step: Text to Image" and "Super 4-Step: Image to Video") — with the row's `--master-port`, `-i`/`-o`, `--checkpoint-path` and trailing flags, and View Results is the media `cosmos_framework.scripts.inference` writes into the `-o` directory (discovery rule: payloads.md § 4.2).

| Use case (`ASSET_SETS` key) | Notebook section                    | Create Payload                                           | `model` (`ASSET_SETS`)            | `--checkpoint-path`                                                                                                     | `model_mode`  | Prompt asset                                               | Image asset (`vision_path`)                        | `enable_sound` | `sampling` override (on top of `FIXED_SAMPLING`)                                                                                         | `--master-port`                | Run shape; env prefix; flags after `--checkpoint-path`                                                                                                                                                                      | `-i` / `-o`                                                                                  | Payload file → output dir (under `$OUT`)                                                   |
| --------------------------- | ----------------------------------- | -------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------- | ---------------------------------------------------------- | -------------------------------------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `t2i`                       | Nano: Text to Image                 | `create_payload("t2i", backend="pytorch")`               | `Cosmos3-Nano`                    | `"Cosmos3-Nano"`                                                                                                        | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                  | `False`        | —                                                                                                                                        | `"$COSMOS3_TEXT_MASTER_PORT"`  | standard; `--seed=0`                                                                                                                                                                                                        | `"$COSMOS3_PYTORCH_T2I_INPUT"` / `"$COSMOS3_PYTORCH_T2I_OUTPUT"`                             | `pytorch/payloads/t2i/t2i.json` → `pytorch/t2i/`                                           |
| `t2i`                       | Nano FP8: Text to Image             | `create_payload("t2i", backend="pytorch_fp8")`           | `Cosmos3-Nano`                    | `"$FP8_CHECKPOINT"` = `$COSMOS3_REPO/checkpoints/Cosmos3-Nano-FP8` (`nvidia/Cosmos3-Nano` downloaded at revision `fp8`) | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                  | `False`        | —                                                                                                                                        | `"$COSMOS3_TEXT_MASTER_PORT"`  | FP8 (`uvx hf@latest download`, then torchrun); `--seed=0`                                                                                                                                                                   | `"$COSMOS3_PYTORCH_FP8_T2I_INPUT"` / `"$COSMOS3_PYTORCH_FP8_T2I_OUTPUT"`                     | `pytorch_fp8/payloads/t2i/t2i.json` → `pytorch_fp8/t2i/`                                   |
| `t2i_super`                 | Super: Text to Image                | `create_payload("t2i_super", backend="pytorch")`         | `Cosmos3-Super`                   | `"Cosmos3-Super"`                                                                                                       | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                  | `False`        | —                                                                                                                                        | `"$COSMOS3_TEXT_MASTER_PORT"`  | standard; `--seed=0`                                                                                                                                                                                                        | `"$COSMOS3_PYTORCH_T2I_SUPER_INPUT"` / `"$COSMOS3_PYTORCH_T2I_SUPER_OUTPUT"`                 | `pytorch/payloads/t2i_super/t2i_super.json` → `pytorch/t2i_super/`                         |
| `t2i_distilled`             | Super 4-Step: Text to Image         | `create_payload("t2i_distilled", backend="pytorch")`     | `Cosmos3-Super-Text2Image-4Step`  | `"Cosmos3-Super-Text2Image-4Step"`                                                                                      | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                  | `False`        | `{"num_steps": 50, "guidance": 1.0, "shift": 3.0, "fps": 30, "num_frames": 1, "resolution": "768", "aspect_ratio": "1,1", "seed": 1}`    | `"$COSMOS3_TEXT_MASTER_PORT"`  | four-step; `COSMOS_TRAINING=1 COSMOS_INTERNAL=0`; `--dp-shard-size=4 --dp-replicate-size=1 --cp-size=1 --cfgp-size=1 --resolution=768 --aspect-ratio=1,1 --fps=30 --num-frames=1 --guidance=1.0 --seed=1 --no-guardrails`   | `"$COSMOS3_PYTORCH_T2I_DISTILLED_INPUT"` / `"$COSMOS3_PYTORCH_T2I_DISTILLED_OUTPUT"`         | `pytorch/payloads/t2i_distilled/t2i_distilled.json` → `pytorch/t2i_distilled/`             |
| `t2v_nano_noaudio`          | Nano: Text to Video Without Audio   | `create_payload("t2v_nano_noaudio", backend="pytorch")`  | `Cosmos3-Nano`                    | `"Cosmos3-Nano"`                                                                                                        | `text2video`  | `assets/prompts/text2video/robot_kitchen.json`             | —                                                  | `False`        | —                                                                                                                                        | `"$COSMOS3_TEXT_MASTER_PORT"`  | standard; `--seed=0`                                                                                                                                                                                                        | `"$COSMOS3_PYTORCH_T2V_NANO_NOAUDIO_INPUT"` / `"$COSMOS3_PYTORCH_T2V_NANO_NOAUDIO_OUTPUT"`   | `pytorch/payloads/t2v_nano_noaudio/t2v_nano_noaudio.json` → `pytorch/t2v_nano_noaudio/`    |
| `t2vs`                      | Nano: Text to Video with Audio      | `create_payload("t2vs", backend="pytorch")`              | `Cosmos3-Nano`                    | `"Cosmos3-Nano"`                                                                                                        | `text2video`  | `assets/prompts/text2video/robot_pouring_water_audio.json` | —                                                  | `True`         | —                                                                                                                                        | `"$COSMOS3_TEXT_MASTER_PORT"`  | standard; `--seed=0`                                                                                                                                                                                                        | `"$COSMOS3_PYTORCH_T2VS_INPUT"` / `"$COSMOS3_PYTORCH_T2VS_OUTPUT"`                           | `pytorch/payloads/t2vs/t2vs.json` → `pytorch/t2vs/`                                        |
| `i2v_nano_noaudio`          | Nano: Image to Video Without Audio  | `create_payload("i2v_nano_noaudio", backend="pytorch")`  | `Cosmos3-Nano`                    | `"Cosmos3-Nano"`                                                                                                        | `image2video` | `assets/prompts/image2video/car_driving.json`              | `assets/images/image2video/car_driving.jpg`        | `False`        | —                                                                                                                                        | `"$COSMOS3_IMAGE_MASTER_PORT"` | standard; `--seed=0`                                                                                                                                                                                                        | `"$COSMOS3_PYTORCH_I2V_NANO_NOAUDIO_INPUT"` / `"$COSMOS3_PYTORCH_I2V_NANO_NOAUDIO_OUTPUT"`   | `pytorch/payloads/i2v_nano_noaudio/i2v_nano_noaudio.json` → `pytorch/i2v_nano_noaudio/`    |
| `i2vs`                      | Nano: Image to Video with Audio     | `create_payload("i2vs", backend="pytorch")`              | `Cosmos3-Nano`                    | `"Cosmos3-Nano"`                                                                                                        | `image2video` | `assets/prompts/image2video/coastal_road_audio.json`       | `assets/images/image2video/coastal_road_audio.jpg` | `True`         | —                                                                                                                                        | `"$COSMOS3_IMAGE_MASTER_PORT"` | standard; `--seed=0`                                                                                                                                                                                                        | `"$COSMOS3_PYTORCH_I2VS_INPUT"` / `"$COSMOS3_PYTORCH_I2VS_OUTPUT"`                           | `pytorch/payloads/i2vs/i2vs.json` → `pytorch/i2vs/`                                        |
| `t2v_super_noaudio`         | Super: Text to Video Without Audio  | `create_payload("t2v_super_noaudio", backend="pytorch")` | `Cosmos3-Super`                   | `"Cosmos3-Super"`                                                                                                       | `text2video`  | `assets/prompts/text2video/robot_kitchen.json`             | —                                                  | `False`        | —                                                                                                                                        | `"$COSMOS3_TEXT_MASTER_PORT"`  | standard; `--seed=0`                                                                                                                                                                                                        | `"$COSMOS3_PYTORCH_T2V_SUPER_NOAUDIO_INPUT"` / `"$COSMOS3_PYTORCH_T2V_SUPER_NOAUDIO_OUTPUT"` | `pytorch/payloads/t2v_super_noaudio/t2v_super_noaudio.json` → `pytorch/t2v_super_noaudio/` |
| `i2v_super_noaudio`         | Super: Image to Video Without Audio | `create_payload("i2v_super_noaudio", backend="pytorch")` | `Cosmos3-Super`                   | `"Cosmos3-Super"`                                                                                                       | `image2video` | `assets/prompts/image2video/car_driving.json`              | `assets/images/image2video/car_driving.jpg`        | `False`        | —                                                                                                                                        | `"$COSMOS3_IMAGE_MASTER_PORT"` | standard; `--seed=0`                                                                                                                                                                                                        | `"$COSMOS3_PYTORCH_I2V_SUPER_NOAUDIO_INPUT"` / `"$COSMOS3_PYTORCH_I2V_SUPER_NOAUDIO_OUTPUT"` | `pytorch/payloads/i2v_super_noaudio/i2v_super_noaudio.json` → `pytorch/i2v_super_noaudio/` |
| `i2v_distilled`             | Super 4-Step: Image to Video        | `create_payload("i2v_distilled", backend="pytorch")`     | `Cosmos3-Super-Image2Video-4Step` | `"Cosmos3-Super-Image2Video-4Step"`                                                                                     | `image2video` | `assets/prompts/image2video/car_driving.json`              | `assets/images/image2video/car_driving.jpg`        | `False`        | `{"num_steps": 35, "guidance": 1.0, "shift": 10.0, "fps": 30, "num_frames": 121, "resolution": "480", "aspect_ratio": "4,3", "seed": 1}` | `"$COSMOS3_IMAGE_MASTER_PORT"` | four-step; `COSMOS_TRAINING=1 COSMOS_INTERNAL=0`; `--dp-shard-size=4 --dp-replicate-size=1 --cp-size=1 --cfgp-size=1 --resolution=480 --aspect-ratio=4,3 --fps=30 --num-frames=121 --guidance=1.0 --seed=1 --no-guardrails` | `"$COSMOS3_PYTORCH_I2V_DISTILLED_INPUT"` / `"$COSMOS3_PYTORCH_I2V_DISTILLED_OUTPUT"`         | `pytorch/payloads/i2v_distilled/i2v_distilled.json` → `pytorch/i2v_distilled/`             |
| `t2i_edge`                  | —                                   | —                                                        | `Cosmos3-Edge`                    | —                                                                                                                       | `text2image`  | `assets/prompts/text2image/robot_draping.json`             | —                                                  | `False`        | `{"resolution": "480"}`                                                                                                                  | —                              | —                                                                                                                                                                                                                           | —                                                                                            | —                                                                                          |
| `t2v_edge_noaudio`          | —                                   | —                                                        | `Cosmos3-Edge`                    | —                                                                                                                       | `text2video`  | `assets/prompts/text2video/robot_kitchen.json`             | —                                                  | `False`        | `{"resolution": "480", "num_frames": 121, "fps": 24}`                                                                                    | —                              | —                                                                                                                                                                                                                           | —                                                                                            | —                                                                                          |
| `i2v_edge_noaudio`          | Edge: Image to Video Without Audio  | `create_payload("i2v_edge_noaudio", backend="pytorch")`  | `Cosmos3-Edge`                    | `"Cosmos3-Edge"`                                                                                                        | `image2video` | `assets/prompts/image2video/car_driving_edge.json`         | `assets/images/image2video/car_driving.jpg`        | `False`        | `{"resolution": "480", "num_frames": 121, "fps": 24, "num_steps": 20, "guidance": 5.0, "shift": 8.0}`                                    | `"$COSMOS3_IMAGE_MASTER_PORT"` | standard; `--seed=0`                                                                                                                                                                                                        | `"$COSMOS3_PYTORCH_I2V_EDGE_NOAUDIO_INPUT"` / `"$COSMOS3_PYTORCH_I2V_EDGE_NOAUDIO_OUTPUT"`   | `pytorch/payloads/i2v_edge_noaudio/i2v_edge_noaudio.json` → `pytorch/i2v_edge_noaudio/`    |

## Nano: Text to Image

Nano text-to-image generation using a structured JSON prompt.

### Run

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" LD_LIBRARY_PATH= \
"$COSMOS3_UV_ENV/bin/torchrun" \
  --nproc-per-node="$COSMOS3_NUM_GPUS" \
  --master-addr="$COSMOS3_MASTER_ADDR" \
  --master-port="$COSMOS3_TEXT_MASTER_PORT" \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=throughput \
  -i "$COSMOS3_PYTORCH_T2I_INPUT" \
  -o "$COSMOS3_PYTORCH_T2I_OUTPUT" \
  --checkpoint-path "Cosmos3-Nano" \
  --seed=0
```

## Nano FP8: Text to Image

FP8 checkpoints are published in the same Hugging Face repositories as the standard checkpoints, under the `fp8` revision rather than in separate FP8 repositories. This example downloads `nvidia/Cosmos3-Nano` at revision `fp8`. To use another released FP8 checkpoint, change `FP8_MODEL_ID` and `FP8_CHECKPOINT` below; keep `FP8_REVISION=fp8`.

### Run

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
FP8_MODEL_ID="nvidia/Cosmos3-Nano"
FP8_REVISION="fp8"
FP8_CHECKPOINT="$COSMOS3_REPO/checkpoints/Cosmos3-Nano-FP8"
uvx hf@latest download "$FP8_MODEL_ID" \
  --revision "$FP8_REVISION" \
  --local-dir "$FP8_CHECKPOINT"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" LD_LIBRARY_PATH= \
"$COSMOS3_UV_ENV/bin/torchrun" \
  --nproc-per-node="$COSMOS3_NUM_GPUS" \
  --master-addr="$COSMOS3_MASTER_ADDR" \
  --master-port="$COSMOS3_TEXT_MASTER_PORT" \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=throughput \
  -i "$COSMOS3_PYTORCH_FP8_T2I_INPUT" \
  -o "$COSMOS3_PYTORCH_FP8_T2I_OUTPUT" \
  --checkpoint-path "$FP8_CHECKPOINT" \
  --seed=0
```

## Super: Text to Image

Super text-to-image generation using the same structured JSON prompt.

## Super 4-Step: Text to Image

Run the published `nvidia/Cosmos3-Super-Text2Image-4Step` student on four GPUs. The model uses its fixed-step SDE schedule; `dp_shard_size=4` is the validated memory layout for the 32B checkpoint. Guardrails are disabled only for this controlled smoke example.

### Run

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
COSMOS_TRAINING=1 COSMOS_INTERNAL=0 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" LD_LIBRARY_PATH= \
"$COSMOS3_UV_ENV/bin/torchrun" \
  --nproc-per-node="$COSMOS3_NUM_GPUS" \
  --master-addr="$COSMOS3_MASTER_ADDR" \
  --master-port="$COSMOS3_TEXT_MASTER_PORT" \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=throughput \
  -i "$COSMOS3_PYTORCH_T2I_DISTILLED_INPUT" \
  -o "$COSMOS3_PYTORCH_T2I_DISTILLED_OUTPUT" \
  --checkpoint-path "Cosmos3-Super-Text2Image-4Step" \
  --dp-shard-size=4 --dp-replicate-size=1 \
  --cp-size=1 --cfgp-size=1 \
  --resolution=768 --aspect-ratio=1,1 \
  --fps=30 --num-frames=1 \
  --guidance=1.0 --seed=1 --no-guardrails
```

## Nano: Text to Video Without Audio

Nano text-to-video generation with audio disabled.

## Nano: Text to Video with Audio

Nano text-to-video generation with generated audio.

## Nano: Image to Video Without Audio

Nano image-to-video generation using its paired image asset, with audio disabled.

## Nano: Image to Video with Audio

Nano image-to-video generation using its paired image asset and generated audio.

## Super: Text to Video Without Audio

Super text-to-video generation with audio disabled.

## Super: Image to Video Without Audio

Super image-to-video generation using its paired image asset, with audio disabled.

## Super 4-Step: Image to Video

Run the published `nvidia/Cosmos3-Super-Image2Video-4Step` student on four GPUs using the existing public car-driving image and prompt. The model uses its fixed-step SDE schedule; `dp_shard_size=4` is the validated memory layout for the 32B checkpoint. Guardrails are disabled only for this controlled smoke example.

### Run

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
COSMOS_TRAINING=1 COSMOS_INTERNAL=0 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" LD_LIBRARY_PATH= \
"$COSMOS3_UV_ENV/bin/torchrun" \
  --nproc-per-node="$COSMOS3_NUM_GPUS" \
  --master-addr="$COSMOS3_MASTER_ADDR" \
  --master-port="$COSMOS3_IMAGE_MASTER_PORT" \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=throughput \
  -i "$COSMOS3_PYTORCH_I2V_DISTILLED_INPUT" \
  -o "$COSMOS3_PYTORCH_I2V_DISTILLED_OUTPUT" \
  --checkpoint-path "Cosmos3-Super-Image2Video-4Step" \
  --dp-shard-size=4 --dp-replicate-size=1 \
  --cp-size=1 --cfgp-size=1 \
  --resolution=480 --aspect-ratio=4,3 \
  --fps=30 --num-frames=121 \
  --guidance=1.0 --seed=1 --no-guardrails
```

## Edge: Image to Video Without Audio

Edge image-to-video generation using its paired image asset, with audio disabled (Edge has no audio modules).

## 8. Skill notes (not in the notebook) — `uv sync` alternative and run settings (from SKILL.md Variant 2)

Skill-authored text moved here from SKILL.md on 2026-10-27 (layout change only; wording unchanged). Nothing in this section is notebook text; SKILL.md keeps a one-line pointer at each former location.

### 8.1 Install-free `uv sync` alternative (SKILL.md Variant 2)

Install-free alternative used by the framework docs, in place of the `uv sync` line:
`GIT_LFS_SKIP_SMUDGE=1 uvx --from "uv==$REQ_UV" uv sync --all-extras --group="$COSMOS3_UV_GROUP"`.

### 8.2 `--checkpoint-path` names, output discovery and the Edge / Super / 4-Step run settings (SKILL.md Variant 2)

`--checkpoint-path` takes the short names the notebook and README pass —
`Cosmos3-Edge`, `Cosmos3-Nano`, `Cosmos3-Super`, `Cosmos3-Super-Text2Image-4Step`,
`Cosmos3-Super-Image2Video-4Step` — or a local checkpoint directory (the notebook's
Nano FP8 example). `view_run` finds the result under the `-o` directory as `*.mp4`
(ignoring `*_preview.mp4` / `*_browser.mp4`) or `*.png` / `*.jpg` / `*.jpeg`. Edge runs
single-GPU (`--nproc-per-node=1`) but supports only 256p/480p, has no audio modules
(keep `"enable_sound": false`), and uses `"num_frames": 121` at `"fps": 24`. The 4-Step
students run on four GPUs with `--dp-shard-size=4` here; Super: "use more GPUs via `--nproc-per-node`".
