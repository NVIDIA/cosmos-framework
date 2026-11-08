<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Transfer with Cosmos Framework

This notebook runs Cosmos3 **video transfer** inference through the native Cosmos Framework PyTorch entrypoint:

```bash
python -m cosmos_framework.scripts.inference
```

Transfer generates a target clip from a caption (`prompt.json`) and one or more spatial control videos. Supported controls:

- **edge** — Canny edge map (`control_edge.mp4`)
- **blur** — blurred reference (`control_blur.mp4`)
- **depth** — depth map (`control_depth.mp4`)
- **seg** — segmentation map (`control_seg.mp4`)
- **wsm** — world-scenario map (`control_wsm.mp4`)
- **multi-control** — two or more hints blended in a single pass (§19)

Run Cosmos3-Nano single-control examples (§9–§13), Cosmos3-Super single-control examples (§14–§18), or the multi-control example (§19, Nano).

For the OpenAI-compatible vLLM-Omni transfer path, see [`run_video_transfer_with_vllm_omni.ipynb`](vllm-omni.md). This notebook focuses on the Cosmos Framework path.

| Model         | Launcher               | Parallelism | GPUs |
| ------------- | ---------------------- | ----------- | ---- |
| Cosmos3-Nano  | `python` (single GPU)  | `latency`   | 1    |
| Cosmos3-Super | `torchrun` (multi-GPU) | `latency`   | 4+   |

> **GPU required.** Run on a host where §3 (`nvidia-smi`) and §7 (`cuda available: True`) both pass.

**Self-contained setup:** everything needed (system packages, clone, Python venv) is in §2–§7 — no external bootstrap scripts required.

Workflow: §2 configure → §3 GPU check → §4 system packages → §5 clone framework → §6 install → §7 verify → §8 review specs → §9–§13 Nano single-control → §14–§18 Super single-control → §19 multi-control (Nano).

> **Skill note (not in the source):** this file is the cookbook notebook `run_video_transfer_with_cosmos_framework.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/generator/transfer/run_video_transfer_with_cosmos_framework.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

## 1. Prerequisites

1. Linux machine with an NVIDIA GPU.
2. A [Hugging Face account](https://huggingface.co) with access to the Cosmos3 model repos — paste your token into **§2 below**.
3. `git` available on PATH (§5 clones Cosmos Framework when missing).
4. Outbound internet for `git clone`, PyPI (`uv sync` in §6), and HF checkpoint downloads.

Everything else — system packages, framework clone, Python venv — is installed automatically by §4–§6.

## 2. Configure

**Edit the cell directly below** — it is the only cell you need to change before running the notebook top to bottom.

| Setting                        | What it controls                                                        |
| ------------------------------ | ----------------------------------------------------------------------- |
| `HF_TOKEN`                     | Hugging Face token for downloading model weights                        |
| `COSMOS3_CACHE_ROOT`           | Path for uv + HF caches (leave `""` to use default under this cookbook) |
| `COSMOS3_TRANSFER_OUTPUT_ROOT` | Where generated videos are saved (leave `""` for default)               |

```python
# ── Edit here, then run all cells ──────────────────────────────────────────

# Hugging Face token — required to download Cosmos3 weights.
# Get yours at https://huggingface.co/settings/tokens
HF_TOKEN = ""  # e.g. "hf_xxxxxxxxxxxxxxxxxxxxxxxx"

# Cache root for uv and Hugging Face downloads.
# Set to a large disk, e.g. "/path/to/cache". Leave "" for default.
COSMOS3_CACHE_ROOT = ""

# Output directory for generated videos. Leave "" for default.
COSMOS3_TRANSFER_OUTPUT_ROOT = ""

# ── Push to environment (do not edit below this line) ───────────────────────
import os
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
if COSMOS3_CACHE_ROOT:
    os.environ["COSMOS3_CACHE_ROOT"] = COSMOS3_CACHE_ROOT
if COSMOS3_TRANSFER_OUTPUT_ROOT:
    os.environ["COSMOS3_TRANSFER_OUTPUT_ROOT"] = COSMOS3_TRANSFER_OUTPUT_ROOT
print(f"cache: {COSMOS3_CACHE_ROOT or '(default)'}")
print(f"HF_TOKEN: {'set' if HF_TOKEN else 'not set — using existing hf login session'}")
```

## 3. Confirm GPU access

Run this **before** install (§6) or inference (§9+). If it fails, fix GPU allocation or driver setup before continuing.

```python
from pathlib import Path
import json
import os
import platform
import socket


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "cookbooks" / "cosmos3").is_dir():
            return path
    return start


def free_local_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def default_framework_repo(root: Path) -> Path:
    for candidate in (root / "packages" / "cosmos-framework", root / "packages" / "cosmos3"):
        if (candidate / "pyproject.toml").exists() and (candidate / "cosmos_framework").exists():
            return candidate
    return root / "packages" / "cosmos3"


COSMOS_ROOT = find_repo_root(Path.cwd().resolve())
COSMOS3_TRANSFER_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3" / "generator" / "transfer"
COSMOS3_REPO = Path(os.environ.get("COSMOS3_REPO", default_framework_repo(COSMOS_ROOT))).resolve()
COSMOS3_GIT_URL = os.environ.get(
    "COSMOS3_GIT_URL",
    "https://github.com/NVIDIA/cosmos-framework.git",
)
def default_uv_group() -> str:
    # cu128-train attention wheels are x86_64-only; aarch64 hosts need cu130-train for natten.
    if platform.machine() == "aarch64":
        return "cu130-train"
    return "cu128-train"


COSMOS3_UV_GROUP = os.environ.get("COSMOS3_UV_GROUP", default_uv_group())
COSMOS3_TRANSFER_OUTPUT_ROOT = Path(
    os.environ.get(
        "COSMOS3_TRANSFER_OUTPUT_ROOT",
        COSMOS3_TRANSFER_ROOT / "outputs" / "notebooks",
    )
).resolve()
COSMOS3_SPECS_DIR = COSMOS3_TRANSFER_ROOT / "specs"
TRANSFER_CONTROLS = ("edge", "blur", "depth", "seg", "wsm")


def _detect_gpu_count() -> str:
    """Count visible GPUs via nvidia-smi; fall back to 4."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=10, stderr=subprocess.DEVNULL,
        ).decode()
        count = len([l for l in out.strip().splitlines() if l])
        if count > 0:
            return str(count)
    except Exception:
        pass
    return "4"


COSMOS3_NUM_GPUS = os.environ.get("COSMOS3_NUM_GPUS") or _detect_gpu_count()

os.environ["COSMOS_ROOT"] = str(COSMOS_ROOT)
os.environ["COSMOS3_TRANSFER_ROOT"] = str(COSMOS3_TRANSFER_ROOT)
os.environ["COSMOS3_REPO"] = str(COSMOS3_REPO)
os.environ["COSMOS3_GIT_URL"] = COSMOS3_GIT_URL
os.environ["COSMOS3_UV_GROUP"] = COSMOS3_UV_GROUP
os.environ["COSMOS3_TRANSFER_OUTPUT_ROOT"] = str(COSMOS3_TRANSFER_OUTPUT_ROOT)
os.environ["COSMOS3_NUM_GPUS"] = COSMOS3_NUM_GPUS


def default_cache_path(name: str) -> str:
    root = os.environ.get("COSMOS3_CACHE_ROOT")
    if root:
        return str((Path(root).expanduser() / name).resolve())
    return str((COSMOS3_TRANSFER_ROOT / ".cache" / name).resolve())


os.environ["UV_CACHE_DIR"] = os.environ.get("COSMOS3_UV_CACHE_DIR", default_cache_path("uv"))
os.environ["HF_HOME"] = os.environ.get("COSMOS3_HF_HOME", default_cache_path("huggingface"))
# NGC PyTorch images: clear bundled libtorch from LD_LIBRARY_PATH before inference.
os.environ.pop("LD_LIBRARY_PATH", None)
# Default CUDA_VISIBLE_DEVICES to all detected GPUs so both Nano and Super cells work.
_all_gpus = ",".join(str(i) for i in range(int(COSMOS3_NUM_GPUS)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", _all_gpus)
os.environ.setdefault("COSMOS3_MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("COSMOS3_MASTER_PORT", free_local_port())

print("cosmos root:", COSMOS_ROOT)
print("transfer cookbook:", COSMOS3_TRANSFER_ROOT)
print("framework:", COSMOS3_REPO)
print("controls:", ", ".join(TRANSFER_CONTROLS))
print("output root:", COSMOS3_TRANSFER_OUTPUT_ROOT)
print("UV_CACHE_DIR:", os.environ["UV_CACHE_DIR"])
print("HF_HOME:", os.environ["HF_HOME"])
print("COSMOS3_UV_GROUP:", os.environ["COSMOS3_UV_GROUP"])
print("COSMOS3_NUM_GPUS:", os.environ["COSMOS3_NUM_GPUS"])
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
```

```bash
set -euo pipefail
echo "hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Run on a GPU host (see §1)."
  exit 1
fi
nvidia-smi -L
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${GPU_COUNT:-0}" -lt 1 ]; then
  echo "ERROR: no GPUs visible. Allocate a GPU or set CUDA_VISIBLE_DEVICES."
  exit 1
fi
echo "OK: ${GPU_COUNT} GPU(s) visible on $(hostname)"
```

## 4. Install system packages (Linux)

Framework guardrails and previews need **ffmpeg**, **git-lfs**, and graphics libraries (`libxcb1`, `libgl1`, …). On hosts with `apt-get` (NGC PyTorch container, many training images), run the next cell to install them.

If `apt-get` is unavailable, install the same packages with your OS package manager — see [Cosmos3 cookbooks README — System packages](../../cookbooks/cosmos3/README.md#system-packages-required-for-framework-guardrails).

```bash
set -euo pipefail

PACKAGES=(curl ffmpeg git-lfs libgl1 libglib2.0-0 libx11-dev libxcb1 tree wget)

if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  echo "Installing system packages via apt-get..."
  apt-get update -qq
  apt-get install -y --no-install-recommends "${PACKAGES[@]}"
  echo "OK: apt packages installed"
else
  echo "NOTE: apt-get not found on this host."
  echo "If guardrails or OpenCV fail with libxcb.so.1, install manually:"
  echo "  ${PACKAGES[*]}"
  echo "See .agents/skills/cookbooks/cosmos3/README.md (Framework guardrails)."
fi

for cmd in git ffmpeg; do
  command -v "$cmd" >/dev/null || { echo "ERROR: missing $cmd"; exit 1; }
done
if command -v git-lfs >/dev/null 2>&1; then
  git lfs install --skip-repo 2>/dev/null || true
  echo "git-lfs: OK"
else
  echo "WARN: git-lfs not in PATH (uv sync may still work with GIT_LFS_SKIP_SMUDGE=1)"
fi
echo "System package check complete."
```

## 5. Clone Cosmos Framework

Clones `COSMOS3_GIT_URL` into `COSMOS3_REPO` when the tree is not already present.

```bash
set -euo pipefail

mkdir -p "$(dirname "$COSMOS3_REPO")"

if [ -d "$COSMOS3_REPO/.git" ]; then
  echo "Using existing framework checkout: $COSMOS3_REPO"
elif [ -f "$COSMOS3_REPO/pyproject.toml" ] && [ -d "$COSMOS3_REPO/cosmos_framework" ]; then
  echo "Using existing framework tree (no .git): $COSMOS3_REPO"
else
  echo "Cloning $COSMOS3_GIT_URL into $COSMOS3_REPO"
  git clone "$COSMOS3_GIT_URL" "$COSMOS3_REPO"
fi

cd "$COSMOS3_REPO"
if [ -d .git ]; then
  git status --short --branch
  git remote -v
fi
```

## 6. Install Cosmos Framework Dependencies

Installs the Framework-required `uv 0.11.29` if it is missing or a different version is selected, then runs `uv sync` to create `packages/cosmos3/.venv`. Uses `COSMOS3_UV_GROUP` from §2.

**Skip:** if `.venv` already imports `cosmos_framework` with CUDA available, the next cell skips `uv sync` (fast re-runs). Set `COSMOS3_FORCE_UV_SYNC=1` to force a full re-sync.

For `jupyter execute` on a GPU node, set `COSMOS3_UV_CACHE_DIR` / `COSMOS3_HF_HOME` (or `COSMOS3_CACHE_ROOT`) in §2 first.

If you change `COSMOS3_UV_GROUP`, **re-run this cell** before inference.

```bash
set -euo pipefail
unset LD_LIBRARY_PATH

UV_REQUIRED_VERSION=0.11.29
uv_version=""
if command -v uv >/dev/null 2>&1; then
  uv_version=$(uv --version | awk '{print $2}')
fi
if [ "$uv_version" != "$UV_REQUIRED_VERSION" ]; then
  echo "Installing uv $UV_REQUIRED_VERSION..."
  curl -LsSf https://astral.sh/uv/0.11.29/install.sh | sh
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env" 2>/dev/null || export PATH="${HOME}/.local/bin:${PATH}"
  hash -r
fi
uv_version=$(uv --version | awk '{print $2}')
if [ "$uv_version" != "$UV_REQUIRED_VERSION" ]; then
  echo "Expected uv $UV_REQUIRED_VERSION, found $uv_version at $(command -v uv)"
  exit 1
fi
uv --version

export GIT_LFS_SKIP_SMUDGE=1
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"
cd "$COSMOS3_REPO"
export UV_CACHE_DIR="${UV_CACHE_DIR:?set paths in §2 (run the configure cell first)}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$COSMOS3_REPO/.venv}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
echo "UV_CACHE_DIR=$UV_CACHE_DIR"
echo "COSMOS3_UV_GROUP=$COSMOS3_UV_GROUP"
echo "UV_HTTP_TIMEOUT=$UV_HTTP_TIMEOUT"

if [ -z "${COSMOS3_FORCE_UV_SYNC:-}" ] && [ -x ".venv/bin/python" ]; then
  if env -u LD_LIBRARY_PATH .venv/bin/python -c \
      'import cosmos_framework, torch; assert torch.cuda.is_available()'; then
    echo "venv ready at $COSMOS3_REPO/.venv — skipping uv sync (set COSMOS3_FORCE_UV_SYNC=1 to re-sync)"
    uv pip install imageio imageio-ffmpeg
    env -u LD_LIBRARY_PATH .venv/bin/python -c \
      'import cosmos_framework, torch; print("venv OK (skipped sync)")'
    exit 0
  fi
  echo "Existing .venv failed CUDA/framework check — running full uv sync..."
fi

attempt=1
max_attempts=3
until uv sync --all-extras --group="$COSMOS3_UV_GROUP"; do
  if [ "$attempt" -ge "$max_attempts" ]; then
    echo "ERROR: uv sync failed after $max_attempts attempts (PyPI timeout or network)."
    exit 1
  fi
  echo "uv sync attempt $attempt failed; retrying in 30s..."
  attempt=$((attempt + 1))
  sleep 30
done

uv pip install imageio imageio-ffmpeg

env -u LD_LIBRARY_PATH .venv/bin/python -c \
  'import cosmos_framework, torch; assert torch.cuda.is_available(); print("venv OK")'
echo "Install complete: $COSMOS3_REPO/.venv"
```

## 6.5. Authenticate with Hugging Face

Downloads Cosmos3 weights from Hugging Face during first inference. This cell uses `HF_TOKEN` from §2 if set, or falls back to an existing `hf login` session.

```bash
set -euo pipefail

HF_BIN="$COSMOS3_REPO/.venv/bin/huggingface-cli"
[ -x "$HF_BIN" ] || HF_BIN=$(command -v huggingface-cli 2>/dev/null || echo "")

if [ -n "${HF_TOKEN:-}" ]; then
  echo "Logging in with HF_TOKEN..."
  if [ -n "$HF_BIN" ]; then
    "$HF_BIN" login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
  fi
  echo "HF_TOKEN: set"
else
  echo "HF_TOKEN not set — checking for existing login session..."
  if [ -n "$HF_BIN" ] && "$HF_BIN" whoami >/dev/null 2>&1; then
    echo "Already logged in as: $(\"$HF_BIN\" whoami 2>/dev/null | head -1)"
  else
    echo "WARNING: No HF_TOKEN and no active login session."
    echo "Inference will fail when downloading weights. Fix by either:"
    echo "  1. Setting HF_TOKEN in §2 and re-running from the top, or"
    echo "  2. Running: uvx hf@latest auth login"
  fi
fi
```

## 7. Verify GPU Environment

```python
import subprocess

verify_code = r'''
import sys
import torch
print("uv group (env):", __import__("os").environ.get("COSMOS3_UV_GROUP", "?"))
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
else:
    print("FIX: set COSMOS3_UV_GROUP in §2 (cu130-train or cu128-train), re-run §6 install, then this cell.")
    sys.exit(1)
'''
result = subprocess.run(
    [str(COSMOS3_REPO / ".venv" / "bin" / "python"), "-c", verify_code],
    cwd=str(COSMOS3_REPO),
    env=os.environ.copy(),
)
if result.returncode != 0:
    raise RuntimeError(
        "CUDA not available. Pass §3 first, then re-run §6 install with the correct COSMOS3_UV_GROUP."
    )
```

> **Skill note (not in the source):** § 7 does not exercise Transformer Engine. If the § 9 run dies at import with `RuntimeError: cudart shared object not found` (toolkit-free host, `cu128-train` wheels — eval 2026-11-08: 2 of 4 first runs, while the two runs on `cu130-train` passed), set `COSMOS3_UV_GROUP=cu130-train` (CUDA 13 driver; the setup guide's default), re-run § 6 with `COSMOS3_FORCE_UV_SYNC=1`, then § 7 and § 9. The loader mechanism is inferred from the agents' traces, not verified against the Transformer Engine source; the fix rests on that 4-of-4 correlation plus the setup guide's driver-to-tag rule. Keep `unset LD_LIBRARY_PATH` (source text in every cell; both passing runs kept it).

## 8. Input Specs and Preview Helpers

Checked-in [`specs/<control>.json`](../specs) are model-agnostic — the same spec runs with Nano and Super.

Inference writes videos to:

```text
<COSMOS3_TRANSFER_OUTPUT_ROOT>/<model>/transfer_<control>/vision.mp4
```

For example:

```text
outputs/notebooks/Cosmos3-Nano/transfer_edge/vision.mp4
outputs/notebooks/Cosmos3-Super/transfer_edge/vision.mp4
```

```python
missing = [c for c in TRANSFER_CONTROLS if not (COSMOS3_SPECS_DIR / f"{c}.json").is_file()]
if missing:
    raise FileNotFoundError(f"missing checked-in specs for {missing} under {COSMOS3_SPECS_DIR}")
print("Using specs:", ", ".join(f"{c}.json" for c in TRANSFER_CONTROLS))
```

```python
from preview_helpers import load_transfer_spec, resolve_spec_path

for control in TRANSFER_CONTROLS:
    spec = load_transfer_spec(control)
    block = spec[control]
    print(
        f"{control}: frames={spec.get('num_frames')} fps={spec.get('fps')} "
        f"guidance={spec.get('guidance')} control_guidance={spec.get('control_guidance')} "
        f"control={resolve_spec_path(block['control_path'])}"
    )
```

## Nano Inference

Run cells §9–§13 for **Cosmos3-Nano** (single GPU, `latency` preset). Outputs go to `<output_root>/Cosmos3-Nano/transfer_<control>/vision.mp4`.

## 9. Nano: Edge (Canny) Transfer

Precomputed edge control (`control_edge.mp4`) + caption. Output:

```text
<COSMOS3_TRANSFER_OUTPUT_ROOT>/Cosmos3-Nano/transfer_edge/vision.mp4
```

```bash
set -euo pipefail
unset LD_LIBRARY_PATH
CONTROL=edge
SPEC="$COSMOS3_TRANSFER_ROOT/specs/${CONTROL}.json"
OUT_DIR="$COSMOS3_TRANSFER_OUTPUT_ROOT/Cosmos3-Nano"
mkdir -p "$OUT_DIR"
echo "control=$CONTROL model=Cosmos3-Nano spec=$SPEC output=$OUT_DIR"
cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$SPEC" \
  -o "$OUT_DIR" \
  --checkpoint-path Cosmos3-Nano \
  --seed 2026
```

### Preview Edge (Canny) (Nano)

```python
import os
import sys
from pathlib import Path

_root = Path.cwd()
if not (_root / "preview_helpers.py").is_file():
    for p in [_root, *_root.parents]:
        cand = p / "cookbooks" / "cosmos3" / "generator" / "transfer"
        if (cand / "preview_helpers.py").is_file():
            _root = cand
            break
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from preview_helpers import preview_transfer

preview_transfer("edge", model="Cosmos3-Nano")
```

## 10. Nano: Blur Transfer

Blurred-reference control (`control_blur.mp4`) + caption. Output:

```text
<COSMOS3_TRANSFER_OUTPUT_ROOT>/Cosmos3-Nano/transfer_blur/vision.mp4
```

```bash
set -euo pipefail
unset LD_LIBRARY_PATH
CONTROL=blur
SPEC="$COSMOS3_TRANSFER_ROOT/specs/${CONTROL}.json"
OUT_DIR="$COSMOS3_TRANSFER_OUTPUT_ROOT/Cosmos3-Nano"
mkdir -p "$OUT_DIR"
echo "control=$CONTROL model=Cosmos3-Nano spec=$SPEC output=$OUT_DIR"
cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$SPEC" \
  -o "$OUT_DIR" \
  --checkpoint-path Cosmos3-Nano \
  --seed 2026
```

### Preview Blur (Nano)

```python
import os
import sys
from pathlib import Path

_root = Path.cwd()
if not (_root / "preview_helpers.py").is_file():
    for p in [_root, *_root.parents]:
        cand = p / "cookbooks" / "cosmos3" / "generator" / "transfer"
        if (cand / "preview_helpers.py").is_file():
            _root = cand
            break
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from preview_helpers import preview_transfer

preview_transfer("blur", model="Cosmos3-Nano")
```

## 11. Nano: Depth Transfer

Depth map control (`control_depth.mp4`) + caption. Output:

```text
<COSMOS3_TRANSFER_OUTPUT_ROOT>/Cosmos3-Nano/transfer_depth/vision.mp4
```

```bash
set -euo pipefail
unset LD_LIBRARY_PATH
CONTROL=depth
SPEC="$COSMOS3_TRANSFER_ROOT/specs/${CONTROL}.json"
OUT_DIR="$COSMOS3_TRANSFER_OUTPUT_ROOT/Cosmos3-Nano"
mkdir -p "$OUT_DIR"
echo "control=$CONTROL model=Cosmos3-Nano spec=$SPEC output=$OUT_DIR"
cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$SPEC" \
  -o "$OUT_DIR" \
  --checkpoint-path Cosmos3-Nano \
  --seed 2026
```

### Preview Depth (Nano)

```python
import os
import sys
from pathlib import Path

_root = Path.cwd()
if not (_root / "preview_helpers.py").is_file():
    for p in [_root, *_root.parents]:
        cand = p / "cookbooks" / "cosmos3" / "generator" / "transfer"
        if (cand / "preview_helpers.py").is_file():
            _root = cand
            break
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from preview_helpers import preview_transfer

preview_transfer("depth", model="Cosmos3-Nano")
```

## 12. Nano: Segmentation Transfer

Segmentation map control (`control_seg.mp4`) + caption. Output:

```text
<COSMOS3_TRANSFER_OUTPUT_ROOT>/Cosmos3-Nano/transfer_seg/vision.mp4
```

```bash
set -euo pipefail
unset LD_LIBRARY_PATH
CONTROL=seg
SPEC="$COSMOS3_TRANSFER_ROOT/specs/${CONTROL}.json"
OUT_DIR="$COSMOS3_TRANSFER_OUTPUT_ROOT/Cosmos3-Nano"
mkdir -p "$OUT_DIR"
echo "control=$CONTROL model=Cosmos3-Nano spec=$SPEC output=$OUT_DIR"
cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$SPEC" \
  -o "$OUT_DIR" \
  --checkpoint-path Cosmos3-Nano \
  --seed 2026
```

### Preview Segmentation (Nano)

```python
import os
import sys
from pathlib import Path

_root = Path.cwd()
if not (_root / "preview_helpers.py").is_file():
    for p in [_root, *_root.parents]:
        cand = p / "cookbooks" / "cosmos3" / "generator" / "transfer"
        if (cand / "preview_helpers.py").is_file():
            _root = cand
            break
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from preview_helpers import preview_transfer

preview_transfer("seg", model="Cosmos3-Nano")
```

## 13. Nano: WSM Transfer

World-scenario map control (`control_wsm.mp4`) + caption. Output:

```text
<COSMOS3_TRANSFER_OUTPUT_ROOT>/Cosmos3-Nano/transfer_wsm/vision.mp4
```

```bash
set -euo pipefail
unset LD_LIBRARY_PATH
CONTROL=wsm
SPEC="$COSMOS3_TRANSFER_ROOT/specs/${CONTROL}.json"
OUT_DIR="$COSMOS3_TRANSFER_OUTPUT_ROOT/Cosmos3-Nano"
mkdir -p "$OUT_DIR"
echo "control=$CONTROL model=Cosmos3-Nano spec=$SPEC output=$OUT_DIR"
cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$SPEC" \
  -o "$OUT_DIR" \
  --checkpoint-path Cosmos3-Nano \
  --seed 2026
```

### Preview WSM (Nano)

```python
import os
import sys
from pathlib import Path

_root = Path.cwd()
if not (_root / "preview_helpers.py").is_file():
    for p in [_root, *_root.parents]:
        cand = p / "cookbooks" / "cosmos3" / "generator" / "transfer"
        if (cand / "preview_helpers.py").is_file():
            _root = cand
            break
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from preview_helpers import preview_transfer

preview_transfer("wsm", model="Cosmos3-Nano")
```

## Super Inference

Run cells §14–§18 for **Cosmos3-Super** (multi-GPU, 32B, `latency` preset). Requires `COSMOS3_NUM_GPUS` GPUs (auto-detected). Outputs go to `<output_root>/Cosmos3-Super/transfer_<control>/vision.mp4`.

## 14. Super: Edge (Canny) Transfer

Precomputed edge control (`control_edge.mp4`) + caption. Output:

```text
<COSMOS3_TRANSFER_OUTPUT_ROOT>/Cosmos3-Super/transfer_edge/vision.mp4
```

```bash
set -euo pipefail
unset LD_LIBRARY_PATH
CONTROL=edge
SPEC="$COSMOS3_TRANSFER_ROOT/specs/${CONTROL}.json"
OUT_DIR="$COSMOS3_TRANSFER_OUTPUT_ROOT/Cosmos3-Super"
mkdir -p "$OUT_DIR"
echo "control=$CONTROL model=Cosmos3-Super spec=$SPEC output=$OUT_DIR"
cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
.venv/bin/torchrun \
  --nproc-per-node="${COSMOS3_NUM_GPUS}" \
  --master-addr="${COSMOS3_MASTER_ADDR}" \
  --master-port="${COSMOS3_MASTER_PORT}" \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$SPEC" \
  -o "$OUT_DIR" \
  --checkpoint-path Cosmos3-Super \
  --seed 2026
```

### Preview Edge (Canny) (Super)

```python
import os
import sys
from pathlib import Path

_root = Path.cwd()
if not (_root / "preview_helpers.py").is_file():
    for p in [_root, *_root.parents]:
        cand = p / "cookbooks" / "cosmos3" / "generator" / "transfer"
        if (cand / "preview_helpers.py").is_file():
            _root = cand
            break
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from preview_helpers import preview_transfer

preview_transfer("edge", model="Cosmos3-Super")
```

## 15. Super: Blur Transfer

Blurred-reference control (`control_blur.mp4`) + caption. Output:

```text
<COSMOS3_TRANSFER_OUTPUT_ROOT>/Cosmos3-Super/transfer_blur/vision.mp4
```

```bash
set -euo pipefail
unset LD_LIBRARY_PATH
CONTROL=blur
SPEC="$COSMOS3_TRANSFER_ROOT/specs/${CONTROL}.json"
OUT_DIR="$COSMOS3_TRANSFER_OUTPUT_ROOT/Cosmos3-Super"
mkdir -p "$OUT_DIR"
echo "control=$CONTROL model=Cosmos3-Super spec=$SPEC output=$OUT_DIR"
cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
.venv/bin/torchrun \
  --nproc-per-node="${COSMOS3_NUM_GPUS}" \
  --master-addr="${COSMOS3_MASTER_ADDR}" \
  --master-port="${COSMOS3_MASTER_PORT}" \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$SPEC" \
  -o "$OUT_DIR" \
  --checkpoint-path Cosmos3-Super \
  --seed 2026
```

### Preview Blur (Super)

```python
import os
import sys
from pathlib import Path

_root = Path.cwd()
if not (_root / "preview_helpers.py").is_file():
    for p in [_root, *_root.parents]:
        cand = p / "cookbooks" / "cosmos3" / "generator" / "transfer"
        if (cand / "preview_helpers.py").is_file():
            _root = cand
            break
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from preview_helpers import preview_transfer

preview_transfer("blur", model="Cosmos3-Super")
```

## 16. Super: Depth Transfer

Depth map control (`control_depth.mp4`) + caption. Output:

```text
<COSMOS3_TRANSFER_OUTPUT_ROOT>/Cosmos3-Super/transfer_depth/vision.mp4
```

```bash
set -euo pipefail
unset LD_LIBRARY_PATH
CONTROL=depth
SPEC="$COSMOS3_TRANSFER_ROOT/specs/${CONTROL}.json"
OUT_DIR="$COSMOS3_TRANSFER_OUTPUT_ROOT/Cosmos3-Super"
mkdir -p "$OUT_DIR"
echo "control=$CONTROL model=Cosmos3-Super spec=$SPEC output=$OUT_DIR"
cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
.venv/bin/torchrun \
  --nproc-per-node="${COSMOS3_NUM_GPUS}" \
  --master-addr="${COSMOS3_MASTER_ADDR}" \
  --master-port="${COSMOS3_MASTER_PORT}" \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$SPEC" \
  -o "$OUT_DIR" \
  --checkpoint-path Cosmos3-Super \
  --seed 2026
```

### Preview Depth (Super)

```python
import os
import sys
from pathlib import Path

_root = Path.cwd()
if not (_root / "preview_helpers.py").is_file():
    for p in [_root, *_root.parents]:
        cand = p / "cookbooks" / "cosmos3" / "generator" / "transfer"
        if (cand / "preview_helpers.py").is_file():
            _root = cand
            break
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from preview_helpers import preview_transfer

preview_transfer("depth", model="Cosmos3-Super")
```

## 17. Super: Segmentation Transfer

Segmentation map control (`control_seg.mp4`) + caption. Output:

```text
<COSMOS3_TRANSFER_OUTPUT_ROOT>/Cosmos3-Super/transfer_seg/vision.mp4
```

```bash
set -euo pipefail
unset LD_LIBRARY_PATH
CONTROL=seg
SPEC="$COSMOS3_TRANSFER_ROOT/specs/${CONTROL}.json"
OUT_DIR="$COSMOS3_TRANSFER_OUTPUT_ROOT/Cosmos3-Super"
mkdir -p "$OUT_DIR"
echo "control=$CONTROL model=Cosmos3-Super spec=$SPEC output=$OUT_DIR"
cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
.venv/bin/torchrun \
  --nproc-per-node="${COSMOS3_NUM_GPUS}" \
  --master-addr="${COSMOS3_MASTER_ADDR}" \
  --master-port="${COSMOS3_MASTER_PORT}" \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$SPEC" \
  -o "$OUT_DIR" \
  --checkpoint-path Cosmos3-Super \
  --seed 2026
```

### Preview Segmentation (Super)

```python
import os
import sys
from pathlib import Path

_root = Path.cwd()
if not (_root / "preview_helpers.py").is_file():
    for p in [_root, *_root.parents]:
        cand = p / "cookbooks" / "cosmos3" / "generator" / "transfer"
        if (cand / "preview_helpers.py").is_file():
            _root = cand
            break
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from preview_helpers import preview_transfer

preview_transfer("seg", model="Cosmos3-Super")
```

## 18. Super: WSM Transfer

World-scenario map control (`control_wsm.mp4`) + caption. Output:

```text
<COSMOS3_TRANSFER_OUTPUT_ROOT>/Cosmos3-Super/transfer_wsm/vision.mp4
```

```bash
set -euo pipefail
unset LD_LIBRARY_PATH
CONTROL=wsm
SPEC="$COSMOS3_TRANSFER_ROOT/specs/${CONTROL}.json"
OUT_DIR="$COSMOS3_TRANSFER_OUTPUT_ROOT/Cosmos3-Super"
mkdir -p "$OUT_DIR"
echo "control=$CONTROL model=Cosmos3-Super spec=$SPEC output=$OUT_DIR"
cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
.venv/bin/torchrun \
  --nproc-per-node="${COSMOS3_NUM_GPUS}" \
  --master-addr="${COSMOS3_MASTER_ADDR}" \
  --master-port="${COSMOS3_MASTER_PORT}" \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$SPEC" \
  -o "$OUT_DIR" \
  --checkpoint-path Cosmos3-Super \
  --seed 2026
```

### Preview WSM (Super)

```python
import os
import sys
from pathlib import Path

_root = Path.cwd()
if not (_root / "preview_helpers.py").is_file():
    for p in [_root, *_root.parents]:
        cand = p / "cookbooks" / "cosmos3" / "generator" / "transfer"
        if (cand / "preview_helpers.py").is_file():
            _root = cand
            break
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from preview_helpers import preview_transfer

preview_transfer("wsm", model="Cosmos3-Super")
```

## 19. Multi-Control Transfer (Nano)

Combines **edge** (Canny) and **blur** controls simultaneously from a single source video,
computed on-the-fly via `vision_path`. No pre-computed control files are needed.

Spec: [`specs/multi_control.json`](../specs/multi_control.json).
Output:

```text
<COSMOS3_TRANSFER_OUTPUT_ROOT>/Cosmos3-Nano/transfer_multi_control/vision.mp4
```

The framework also saves the intermediate control signals alongside the output:
`control_edge.mp4` and `control_blur.mp4`.

```bash
set -euo pipefail
unset LD_LIBRARY_PATH
SPEC="$COSMOS3_TRANSFER_ROOT/specs/multi_control.json"
OUT_DIR="$COSMOS3_TRANSFER_OUTPUT_ROOT/Cosmos3-Nano"
mkdir -p "$OUT_DIR"
echo "spec=$SPEC output=$OUT_DIR model=Cosmos3-Nano"
cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$SPEC" \
  -o "$OUT_DIR" \
  --checkpoint-path Cosmos3-Nano \
  --seed 2026
```

### Preview multi_control

```python
import sys
from pathlib import Path

_root = Path.cwd()
if not (_root / "preview_helpers.py").is_file():
    for p in [_root, *_root.parents]:
        cand = p / "cookbooks" / "cosmos3" / "generator" / "transfer"
        if (cand / "preview_helpers.py").is_file():
            _root = cand
            break
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from preview_helpers import preview_multi_control

preview_multi_control()
```
