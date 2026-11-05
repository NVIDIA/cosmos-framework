# π0.5 Joint PPO on LIBERO-Spatial

Start with the four-GPU container and Posttrain environment from
[the general run guide](../howto/run-posttrain.md). Use Python 3.12
for Joint PPO because the pinned vLLM-Omni shared-memory runtime is not yet
compatible with Python 3.13.

Create a dedicated Python 3.12 environment when the general environment selected
Python 3.13:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-posttrain-joint"
uv sync --frozen --no-default-groups --python 3.12
source .venv-posttrain-joint/bin/activate
export POSTTRAIN_PY="$PWD/.venv-posttrain-joint/bin/python"
```

Install the Joint PPO model-side closure and pinned vLLM-Omni commit:

```bash
uv sync --project docker/posttrain/environment --frozen \
  --group gr00t-cu130-aarch64 --active --inexact
uv pip install --python "$POSTTRAIN_PY" --no-deps \
  'git+https://github.com/vllm-project/vllm-omni@2d0b1d8a29ff686698dbf0a123ca15be20fb0a8f'
```

The published checkpoint requires Hugging Face access. Request access to the
repository if needed, then export a read token before downloading it and the
public tokenizer:

```bash
export HF_TOKEN="<your-read-token>"
export POSTTRAIN_RESOURCE_ROOT="$PWD/artifacts"
mkdir -p "$POSTTRAIN_RESOURCE_ROOT/checkpoints/pi05_libero" \
  "$POSTTRAIN_RESOURCE_ROOT/tokenizers"

uvx hf@latest download RLinf/RLinf-Pi05-LIBERO-SFT \
  --revision 45ccfcc4e28634f1576ebf78cab0fbe2fd82432d \
  --local-dir "$POSTTRAIN_RESOURCE_ROOT/checkpoints/pi05_libero"
curl --fail --location \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model \
  --output "$POSTTRAIN_RESOURCE_ROOT/tokenizers/paligemma_tokenizer.model"
```

Create the isolated simulator environment from the checked-in lock. The lock is
a complete qualified closure and intentionally contains a metadata conflict, so
install its exact rows without dependency re-resolution. `--no-config` keeps the
repository's `pyproject.toml` settings out of this install: its
`override-dependencies` would replace the lock's `numpy==1.26.4` with numpy 2.x,
and the simulator refuses to start on any version that differs from the lock.

```bash
export POSTTRAIN_SIM_VENV="$PWD/.venv-libero"
uv venv --python 3.12 "$POSTTRAIN_SIM_VENV"
uv pip install --no-config --python "$POSTTRAIN_SIM_VENV/bin/python" --no-deps \
  -r cosmos_framework/posttrain/embodied/sims/libero/requirements.lock

git clone --depth 1 https://github.com/RLinf/LIBERO.git artifacts/RLinf-LIBERO
"$POSTTRAIN_SIM_VENV/bin/libero-download-assets" --link \
  "$PWD/artifacts/RLinf-LIBERO/libero/libero/assets"
```

Install the simulator system libraries in the container:

```bash
apt-get update
apt-get install -y --no-install-recommends libgl1 libegl1 libgles2 libglib2.0-0
```

Create the simulator config and LIBERO `shared_500` manifests:

```bash
export LIBERO_MANIFEST_ROOT="$PWD/artifacts/libero/manifests"
export LIBERO_CONFIG_PATH="$PWD/artifacts/libero/config"
mkdir -p "$LIBERO_MANIFEST_ROOT" "$LIBERO_CONFIG_PATH"
SIM_SITE="$($POSTTRAIN_SIM_VENV/bin/python -c \
  'import sysconfig; print(sysconfig.get_path("purelib"))')"

cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF
benchmark_root: $SIM_SITE/libero/libero
bddl_files: $SIM_SITE/libero/libero/bddl_files
init_states: $SIM_SITE/libero/libero/init_files
assets: $PWD/artifacts/RLinf-LIBERO/libero/libero/assets
datasets: $PWD/artifacts/libero/datasets
EOF

"$POSTTRAIN_PY" - <<'PY'
from pathlib import Path
from cosmos_framework.posttrain.embodied.sims.libero.protocol import fixed_manifest

root = Path("artifacts/libero/manifests")
for split in ("train", "eval"):
    fixed_manifest(split, "libero_spatial", "shared_500").write(
        root / f"libero_spatial-shared_500-{split}.json"
    )
PY
```

Export the runtime paths:

```bash
export POSTTRAIN_SIM_PYTHON="$POSTTRAIN_SIM_VENV/bin/python"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="$PWD/artifacts/libero/egl-nvidia.json"
printf '%s\n' \
  '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
  > "$__EGL_VENDOR_LIBRARY_FILENAMES"
export RAY_TMPDIR=/tmp/posttrain-ray
export B1K_SIM_IPC_DIR=/tmp/posttrain-sim-ipc
export B1K_SHM_DIR=/tmp/posttrain-sim-shm
mkdir -p "$RAY_TMPDIR" "$B1K_SIM_IPC_DIR" "$B1K_SHM_DIR"
```

Validate sources and start training:

```bash
"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.joint_ppo \
  --config cosmos_framework/posttrain/configs/pi05_libero_spatial_public_4gpu.py \
  --validate-manifest

"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.joint_ppo \
  --config cosmos_framework/posttrain/configs/pi05_libero_spatial_public_4gpu.py
```

Before its first training step, this recipe evaluates the policy on all 500 cases of the
`shared_500` eval manifest. On the reference host that takes about 40 minutes, during which the
main log prints nothing after its `setup progress` lines; the first metric line is
`step 0/12000 val/success_rate ...`. The evaluation repeats every 480 steps and after the last
step, and the recipe runs for 12000 steps. For a short check, run five steps without evaluation:

```bash
"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.joint_ppo \
  --config cosmos_framework/posttrain/configs/pi05_libero_spatial_public_4gpu.py -- \
  rl.max_steps=5 rl.validation.enabled=false
```

See the [run guide Q&A](../howto/run-posttrain-qa.md#π05-prints-nothing-for-about-40-minutes-after-it-starts-is-it-hung)
for how to tell the evaluation is progressing and how to change its schedule.
