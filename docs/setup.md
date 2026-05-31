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
- [Downloading Base Checkpoints](#downloading-base-checkpoints)
- [Troubleshooting](#troubleshooting)
  - [PyTorch Import Issue](#pytorch-import-issue)
  - [Dependency Issue](#dependency-issue)
  - [Python Issue](#python-issue)
  - [CUDA Issue](#cuda-issue)

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
FROM nvcr.io/nvidia/pytorch:25.09-py3
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

If you started from the [recommended base image](#recommended-base-image) (`nvcr.io/nvidia/pytorch:25.09-py3`), the following commands set up the full environment in one go. Run them **from the root of this repository** (i.e. inside the `Cosmos/` directory you just cloned):

```shell
apt-get update
apt-get install -y --no-install-recommends curl ffmpeg git-lfs libx11-dev tree wget

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# CUDA 13.0 (recommended); for CUDA 12.8 use `--group=cu128-train`
uv sync --all-extras --group=cu130-train
source .venv/bin/activate && export LD_LIBRARY_PATH=
```

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
sudo apt-get install -y --no-install-recommends curl ffmpeg git-lfs libx11-dev tree wget
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
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
source .venv/bin/activate && export LD_LIBRARY_PATH=
```

</details>

<details><summary><b>UV Pip: virtual environment</b></summary>

```shell
# Create virtual environment (skip if using an existing environment)
uv venv --clear && source .venv/bin/activate && export LD_LIBRARY_PATH=

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
- `eval`: Evaluation harnesses for trained checkpoints

#### CUDA Variants

This repository is **training-focused**, so the `*-train` dependency groups are the supported install path. Inference-only groups exist for evaluating trained checkpoints in-tree but are not required for training.

| CUDA Version                | Training (recommended) | Notes                                                                                                                                    |
| --------------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **CUDA 13.0 (recommended)** | `--group=cu130-train`  | [NVIDIA Driver](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) |
| CUDA 12.8                   | `--group=cu128-train`  | [NVIDIA Driver](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) |

</details>

</details>

## Environment Variables

Export the following before downloading checkpoints or launching training. See [environment_variables.md](environment_variables.md) for the full reference.

| Variable                 | Purpose                                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------------------------ |
| `HF_TOKEN`               | Hugging Face access token for gated model/dataset downloads. Alternative to `uvx hf auth login`.       |
| `HF_HOME`                | Cache directory for Hugging Face models and datasets. Recommend ≥ 1 TB free.                           |
| `IMAGINAIRE_OUTPUT_ROOT` | Output root for training DCP checkpoints and logs. Recommend ≥ 1 TB free.                              |
| `UV_CACHE_DIR`           | Cache directory for `uv`-managed dependencies.                                                         |
| `LD_LIBRARY_PATH=`       | Clear (set to empty) after sourcing the venv to avoid host library bleed-through into PyTorch imports. |

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

1. Check that you don't have conflicting [environment variables](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables) — e.g. an `HF_TOKEN` set to a different token than the one cached by `hf auth login`: `printenv | grep HF_`.
2. Check that your [token](https://huggingface.co/settings/tokens) has sufficient permissions.

Checkpoints are downloaded on demand during training and evaluation. To change the cache location, set [`HF_HOME`](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome). See [training.md](training.md) for [DCP conversion](training.md#step-2--prepare-checkpoint) and [Hugging Face safetensors export](training.md#export-checkpoint-to-hugging-face-safetensors).

## Troubleshooting

### PyTorch Import Issue

Errors:

- `ImportError: cannot import name '_functionalization' from 'torch._C'`

Clear the library path in your current shell:

```shell
export LD_LIBRARY_PATH=
```

This applies to the current session only. To persist, add the line to your `Dockerfile` or `~/.bashrc`.

If this doesn't fix the issue, try [reinstalling venv](#dependency-issue).

### Dependency Issue

Errors:

- `ModuleNotFoundError: No module named <module_name>`

Reinstall venv:

```shell
uv sync --all-extras --group=cu130-train --reinstall
source .venv/bin/activate && export LD_LIBRARY_PATH=
```

If this doesn't fix the issue, try [reinstalling uv](#python-issue).

### Python Issue

Errors:

- `fatal error: Python.h: No such file or directory`

Reinstall uv and venv:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install --reinstall
rm -rf .venv
uv sync --all-extras --group=cu130-train --reinstall
source .venv/bin/activate && export LD_LIBRARY_PATH=
```

### CUDA Issue

- `OSError: <lib_name>: cannot open shared object file: No such file or directory`

Ensure you have [CUDA installed](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#ubuntu). The major version must match between the system and virtual environment CUDA versions.

```shell
sudo apt-get install -y --no-install-recommends cuda-toolkit-<cuda_major_version>
```

Alternatively, use the [Docker container](#docker-container).
