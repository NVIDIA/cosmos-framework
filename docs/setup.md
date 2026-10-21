# Setup Guide

> **Skill:** `.agents/skills/cosmos3-setup/SKILL.md`

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [System Requirements](#system-requirements)
  - [Recommended Base Image](#recommended-base-image)
- [Installation](#installation)
  - [Quickstart: From the Recommended Base Image](#quickstart-from-the-recommended-base-image)
  - [Docker Container](#docker-container)
  - [Advanced](#advanced)
    - [Virtual Environment](#virtual-environment)
    - [CUDA Variants](#cuda-variants)
- [Environment Variables](#environment-variables)
- [Verify the GPU and media runtime](#verify-the-gpu-and-media-runtime)
- [Downloading Base Checkpoints](#downloading-base-checkpoints)
- [Troubleshooting](#troubleshooting)
  - [PyTorch Import Issue](#pytorch-import-issue)
  - [Dependency Issue](#dependency-issue)
  - [Python Issue](#python-issue)
  - [CUDA Issue](#cuda-issue)
    - [TorchCodec Cannot Find CUDA 12 NPP](#torchcodec-cannot-find-cuda-12-npp)

______________________________________________________________________

<!--TOC-->

## System Requirements

- NVIDIA GPUs with Ampere architecture (RTX 30 Series, A100) or newer — Hopper (H100) or Blackwell (B200) recommended for full training throughput
- NVIDIA driver compatible with [CUDA version](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [NVIDIA CUDA >=12.8](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#ubuntu)
- Linux x86-64/aarch64
- glibc >=2.35 (e.g. Ubuntu >=22.04)
- Python >=3.10
- Multi-node training additionally requires a working NCCL setup (IB/RoCE recommended) and a shared filesystem visible to all ranks for checkpoint I/O
- Free disk: ~150 GiB recommended for a first-run inference or training workflow (Hugging Face cache ~90 GiB, uv cache ~20 GiB, run outputs ~30 GiB). See [FAQ → Expected disk footprint](./faq.md#q-how-much-disk-space-do-i-need) for the breakdown and how to relocate caches.

<details><summary><b>Recommended Base Image</b></summary>

### Recommended Base Image

For CUDA 13 builds, the [NVIDIA NGC PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) is the recommended starting point — it bundles PyTorch + CUDA 13 + cuDNN + NCCL tuned for NVIDIA hardware, plus Apex, TransformerEngine, and Megatron utilities that training infra users commonly need.

```dockerfile
FROM nvcr.io/nvidia/pytorch:26.06-py3
```

For CUDA 12.8 builds, pin to an earlier NGC tag (e.g. `nvcr.io/nvidia/pytorch:25.06-py3`) that still ships CUDA 12.

</details>

## Installation

If you encounter issues, see [Troubleshooting](#troubleshooting).

Clone the repository:

```bash
git clone git@github.com:NVIDIA/cosmos-framework.git
cd cosmos-framework
```

The two supported install paths are the recommended base image and the Docker container. For other paths (standalone venv, custom torch/cuda) see [Advanced](#advanced).

<details open><summary><b>Quickstart: From the Recommended Base Image</b></summary>

### Quickstart: From the Recommended Base Image

If you started from the [recommended base image](#recommended-base-image) (`nvcr.io/nvidia/pytorch:26.06-py3`), the following commands set up the full environment in one go. Run them **from the root of this repository** (i.e. inside the `cosmos-framework/` directory you just cloned):

```shell
apt-get update
apt-get install -y --no-install-recommends curl ffmpeg git-lfs libgl1 libglib2.0-0 libx11-dev tree wget

curl -LsSf https://astral.sh/uv/0.11.29/install.sh | sh
source $HOME/.local/bin/env

# CUDA 13.0 (recommended); for CUDA 12.8 use `--group=cu128-train`
uv sync --all-extras --group=cu130-train
source .venv/bin/activate
```

Check the [GPU and media runtime](#verify-the-gpu-and-media-runtime) before
downloading weights. Do not clear `LD_LIBRARY_PATH` unconditionally: an older
host driver may depend on the container's CUDA compatibility libraries.

</details>

<details><summary><b>Docker Container</b></summary>

### Docker Container

Please make sure you have access to Docker on your machine and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed.

Build the container:

```bash
image_tag=$(docker build -q .)
```

Run the container:

```bash
docker run -it --runtime=nvidia --ipc=host --rm \
  -v .:/workspace -v /workspace/.venv \
  -v /root/.cache:/root/.cache \
  -e HF_TOKEN="$HF_TOKEN" \
  $image_tag
```

For multi-node training, also bind-mount your shared dataset and checkpoint directories so all ranks see the same filesystem.

Optional arguments:

- `--ipc=host`: Use host system's shared memory, since parallel torchrun consumes a large amount of shared memory. If not allowed by security policy, increase `--shm-size` ([documentation](https://docs.docker.com/engine/containers/run/#runtime-constraints-on-resources)).
- `-v /root/.cache:/root/.cache`: Mount host cache to avoid re-downloading cache entries.

If you get `docker: Error response from daemon: unknown or invalid runtime name: nvidia`, you need to [configure docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker):

```shell
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

See [docker/README.md](../docker/README.md) for additional images and build options.

</details>

<details><summary><b>Advanced</b></summary>

### Advanced

Use these paths only when the recommended base image or Docker container are not viable for your environment.

<details><summary><b>Virtual Environment</b></summary>

#### Virtual Environment

Install system dependencies:

```shell
sudo apt-get install -y --no-install-recommends curl ffmpeg git-lfs libgl1 libglib2.0-0 libx11-dev tree wget
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```shell
curl -LsSf https://astral.sh/uv/0.11.29/install.sh | sh
source $HOME/.local/bin/env
```

Install the package using one of the following methods:

<details open><summary><b>UV Sync: fully reproducible environment</b></summary>

Choose the dependency group that matches your CUDA toolkit (see [CUDA Variants](#cuda-variants)):

```shell
# CUDA 13.0 (recommended)
uv sync --all-extras --group=cu130-train
# Or, for CUDA 12.8:
# uv sync --all-extras --group=cu128-train
source .venv/bin/activate
```

</details>

<details><summary><b>UV Pip: virtual environment</b></summary>

```shell
# Create virtual environment (skip if using an existing environment)
uv venv --clear && source .venv/bin/activate

uv pip install -r pyproject.toml --all-extras --group=cu130-train
uv pip install -e .
```

</details>

<details><summary><b>UV Pip: system environment</b></summary>

```shell
uv pip install --system --break-system-packages -r pyproject.toml --all-extras --group=cu130-train
```

</details>

<details><summary><b>Custom torch/cuda versions</b></summary>

```shell
cuda_name=cu130
torch_name=torch210

# 1. Create and activate the virtual environment
uv venv --clear && source .venv/bin/activate

# 2. Install the desired torch/cuda versions
uv pip install "torch==2.10.0" "torchvision" --torch-backend=$cuda_name

# 3. Install the package with desired extras
uv pip install -r pyproject.toml --all-extras --group=cu130-train

# 4. Install one of the following attention backends:
# * Blackwell
uv pip install "natten==0.21.6.dev6+$cuda_name.$torch_name" -f https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0/natten
# * Hopper
uv pip install "flash-attn-3-nv==1.0.3+$cuda_name.$torch_name" -f https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0/flash-attn-3-nv
# * Ada/Ampere
uv pip install "flash-attn==2.7.4.post1+$cuda_name.$torch_name" -f https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0/flash-attn
```

If there is no attention backend wheel for your torch/cuda versions, you can build one using [cosmos-dependencies](https://github.com/nvidia-cosmos/cosmos-dependencies).

</details>

Optional package extras:

- `train`: Training infrastructure (FSDP, parallelism, checkpointing, datasets)

#### CUDA Variants

This repository is **training-focused**, so the `*-train` dependency groups are the supported install path. Inference-only groups exist for evaluating trained checkpoints in-tree but are not required for training.

| CUDA Version                | Training (recommended) | Notes                                                                                                                                    |
| --------------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **CUDA 13.0 (recommended)** | `--group=cu130-train`  | [NVIDIA Driver](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) |
| CUDA 12.8                   | `--group=cu128-train`  | [NVIDIA Driver](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) |

CUDA 13 normally requires driver 580 or newer. With an older driver, such as
570, use the CUDA 12.8 group unless your platform supplies a supported
[CUDA forward-compatibility setup](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).
If that setup requires a `cuda-compat` library directory on `LD_LIBRARY_PATH`,
preserve it. Installing a new CUDA toolkit alone does not upgrade the host driver.
The CUDA version displayed by `nvidia-smi` is the driver's supported version,
not the version installed in the Python environment; exact equality with
`torch.version.cuda` is not required.

</details>

</details>

## Environment Variables

Export the following before downloading checkpoints or launching training. See [environment_variables.md](environment_variables.md) for the full reference.

| Variable                 | Purpose                                                                                                                          |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| `HF_TOKEN`               | Hugging Face access token for gated model/dataset downloads. Alternative to `uvx hf auth login`.                                 |
| `HF_HOME`                | Cache directory for Hugging Face models and datasets. Recommend ≥ 1 TB free.                                                     |
| `IMAGINAIRE_OUTPUT_ROOT` | Output root for training DCP checkpoints and logs. Recommend ≥ 1 TB free.                                                        |
| `UV_CACHE_DIR`           | Cache directory for `uv`-managed dependencies.                                                                                   |
| `LD_LIBRARY_PATH`        | Preserve required driver/compatibility libraries; remove conflicting library directories only when diagnosing an import failure. |

## Verify the GPU and media runtime

Run these checks inside the same container and allocation that will run the model:

```bash
uv --version
nvidia-smi
ffmpeg -version
python - <<'PY'
import cv2
import torch
from cosmos_framework.utils.generator.torchcodec_dependencies import load_torchcodec

print(f"torch={torch.__version__}, CUDA build={torch.version.cuda}, OpenCV={cv2.__version__}")
torch.cuda.init()
torch.ones(1, device="cuda").add_(1)
torch.cuda.synchronize()
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"TorchCodec: {load_torchcodec().__version__}")
PY
```

The selected `uv` executable must match `tool.uv.required-version` in
`pyproject.toml` (currently `0.11.29`). If another version is already installed,
run `uv self update 0.11.29` or reinstall that exact version and check `command -v uv`.
An import-only check does not establish that CUDA initializes or video decoding works.
After downloading an example video, verify decoding too:

```bash
export CHECK_VIDEO=/path/to/one/example.mp4
python - <<'PY'
import os
from cosmos_framework.utils.generator.torchcodec_dependencies import load_torchcodec

codec = load_torchcodec()
decoder = codec.decoders.VideoDecoder(os.environ["CHECK_VIDEO"], device="cpu")
print(decoder[0].shape)
PY
```

The apt packages above supply FFmpeg shared libraries and OpenCV's OpenGL/GLib
runtime dependencies. A Python wheel installation does not replace these system
libraries. See [CUDA troubleshooting](#cuda-issue) for TorchCodec's NPP loading.

## Downloading Base Checkpoints

Training in this repo typically starts from a pretrained base checkpoint that you fine-tune or post-train. The recommended source is the Hugging Face Hub.

1. Get a [Hugging Face Access Token](https://huggingface.co/settings/tokens) with `Read` permission.
2. Authenticate using **either** mechanism (they are equivalent — pick one, do not set both with different tokens):
   - **`HF_TOKEN` environment variable** — preferred for Docker and non-interactive shells. Export it once and any `huggingface_hub` call (CLI or library) picks it up.
   - **`uvx hf auth login`** — preferred for local interactive use. Writes the token to `~/.cache/huggingface/token`, persisted across sessions (and across Docker runs if you bind-mount `/root/.cache`).
3. Accept the license for any gated model you intend to use (e.g. the [NVIDIA Open Model License Agreement](https://huggingface.co/nvidia/Cosmos-Guardrail1) where applicable).
4. Test access:

   ```shell
   uvx hf@latest download --repo-type model nvidia/Cosmos-Guardrail1 \
     --revision d6d4bfa899a71454a700907664f3e88f503950cf --include "README.md"
   ```

If you encounter issues:

1. Check which account is active with `hf auth whoami`. If `HF_TOKEN` is set, it takes precedence over the saved login; keep these credentials consistent. Keep token values out of logs and support requests.
2. Check that your [token](https://huggingface.co/settings/tokens) has sufficient permissions.

Checkpoints are downloaded on demand during training and evaluation. To change the cache location, set [`HF_HOME`](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome). See [training.md](training.md) for [DCP conversion](training.md#step-2--prepare-checkpoint) and [Hugging Face safetensors export](training.md#export-checkpoint-to-hugging-face-safetensors).

## Troubleshooting

### PyTorch Import Issue

Errors:

- `ImportError: cannot import name '_functionalization' from 'torch._C'`

An inherited NGC library directory can conflict with the environment's PyTorch.
Inspect `LD_LIBRARY_PATH` and remove the conflicting entries. If the host driver
already supports the selected CUDA build and no compatibility directory is needed,
test a single process with an empty path:

```shell
LD_LIBRARY_PATH= python -c 'import torch; torch.cuda.init(); print(torch.cuda.get_device_name())'
```

Otherwise retain the platform's `cuda-compat` directory while removing conflicting
cuDNN/NCCL/PyTorch directories. Do not add an unconditional path reset to your
shell startup or Dockerfile. Re-run the GPU check after any change.

If this doesn't fix the issue, try [reinstalling venv](#dependency-issue).

### Dependency Issue

Errors:

- `ModuleNotFoundError: No module named <module_name>`

Reinstall venv:

```shell
uv sync --all-extras --group=cu130-train --reinstall
source .venv/bin/activate
```

If this doesn't fix the issue, try [reinstalling uv](#python-issue).

### Python Issue

Errors:

- `fatal error: Python.h: No such file or directory`

Reinstall uv and venv:

```shell
curl -LsSf https://astral.sh/uv/0.11.29/install.sh | sh
uv python install --reinstall
rm -rf .venv
uv sync --all-extras --group=cu130-train --reinstall
source .venv/bin/activate
```

### CUDA Issue

For TorchCodec's CUDA builds, NPP is required even for CPU video decoding. The
`cu128` and `cu130` groups install the matching NPP wheel. Framework's common
video reader and inference conditioning decoder load that wheel before importing
TorchCodec, so an empty `LD_LIBRARY_PATH` does not require a system NPP install.
Standalone TorchCodec callers can use the same preparation:

```python
from cosmos_framework.utils.generator.torchcodec_dependencies import load_torchcodec

torchcodec = load_torchcodec()
```

Install FFmpeg shared libraries that TorchCodec can load, as well as the
`ffmpeg` executable.

- `OSError: <lib_name>: cannot open shared object file: No such file or directory`

Check which library is missing before installing a toolkit. Missing FFmpeg,
OpenGL or GLib libraries require the system packages above. CUDA initialization
errors require checking the host driver and [CUDA variant](#cuda-variants),
including any compatibility-library path. If a system CUDA toolkit library is
actually missing, install the toolkit matching the environment's CUDA build:

```shell
sudo apt-get install -y --no-install-recommends "cuda-toolkit-${CUDA_TOOLKIT_VERSION:?set a version such as 13-0}"
```

Alternatively, use the [Docker container](#docker-container).

#### TorchCodec Cannot Find CUDA 12 NPP

With `cu128` or `cu128-train`, `import torchcodec` can fail with
`libnppicc.so.12: cannot open shared object file` when the host only exposes
another CUDA major version. The environment already installs `nvidia-npp-cu12`,
but its wheel directory is not necessarily on the system linker's search path.
Do not substitute a CUDA 13 library for a CUDA 12 library.

For a session-local workaround, use the matching libraries from the active
environment. This also works when a launch script clears `LD_LIBRARY_PATH`:

```shell
source .venv/bin/activate
COSMOS_NPP_LIBDIR="$(python -c 'from importlib.metadata import distribution; print(distribution("nvidia-npp-cu12").locate_file("nvidia/npp/lib"))')"
export LD_PRELOAD="$COSMOS_NPP_LIBDIR/libnppc.so.12:$COSMOS_NPP_LIBDIR/libnppicc.so.12"
LD_LIBRARY_PATH= python -c 'import torch; import torchcodec; print(torchcodec.__version__)'
```

Apply this only to the CUDA 12 environment that needs it. After the run, unset
`LD_PRELOAD` and `COSMOS_NPP_LIBDIR` (or restore their previous values if set).
This does not install or replace system CUDA libraries. A matching CUDA 12
container or toolkit, as described above, is the alternative.
