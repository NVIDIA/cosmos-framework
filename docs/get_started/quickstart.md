# Quickstart

Install Cosmos-Framework with `uv`, sign in to Hugging Face, and generate a video from a text prompt with Cosmos3-Nano on one GPU.

## Requirements

| Item   | Requirement                                                                                                                                             |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GPU    | One NVIDIA GPU, Ampere or newer. Cosmos3-Nano's weights alone take about 32 GB of GPU memory.                                                           |
| Driver | The CUDA 13.0 build needs driver 580 or newer. With an older driver, such as 570, use the CUDA 12.8 group ([CUDA variants](../setup.md#cuda-variants)). |
| OS     | Linux x86-64 or aarch64, glibc 2.35 or newer (for example, Ubuntu 22.04 or newer).                                                                      |
| Python | 3.13, pinned in `.python-version`.                                                                                                                      |
| uv     | Exactly 0.11.29, pinned in `pyproject.toml`.                                                                                                            |
| Disk   | About 150 GiB free: Hugging Face cache ~90 GiB, uv cache ~20 GiB, outputs ~30 GiB.                                                                      |

## Install

```shell
git clone https://github.com/NVIDIA/cosmos-framework.git
cd cosmos-framework
sudo apt-get install -y --no-install-recommends curl ffmpeg git-lfs libgl1 libglib2.0-0 libx11-dev tree wget

curl -LsSf https://astral.sh/uv/0.11.29/install.sh | sh
source $HOME/.local/bin/env

# CUDA 13.0 (recommended)
uv sync --all-extras --group=cu130-train
# Or, for CUDA 12.8:
# uv sync --all-extras --group=cu128-train
source .venv/bin/activate
```

Then run the [GPU and media checks](../setup.md#verify-the-gpu-and-media-runtime). To install in a container instead, see [the NGC base image](../setup.md#quickstart-from-the-recommended-base-image) or [Docker container](../setup.md#docker-container).

## Sign in to Hugging Face

The first run downloads Cosmos3-Nano and the guardrail models from Hugging Face.

- Create an [access token](https://huggingface.co/settings/tokens) with `Read` permission.
- Sign in with `uvx hf auth login`, or export `HF_TOKEN` (preferred in Docker and non-interactive shells). Use one, not both with different tokens.
- Accept the license of any gated model you use.

Test access:

```shell
uvx hf@latest download --repo-type model nvidia/Cosmos-Guardrail1 \
  --revision d6d4bfa899a71454a700907664f3e88f503950cf --include "README.md"
```

To keep the cache on a larger disk, set `HF_HOME` before the first run.

## Generate your first video

From the repository root, with the environment active:

```shell
python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i "inputs/omni/t2v.json" \
    -o outputs/omni_nano \
    --checkpoint-path Cosmos3-Nano \
    --seed=0
```

| Flag                           | Meaning                                                                                                                                                              |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--parallelism-preset=latency` | The default preset. It only changes the layout on multiple GPUs ([parallelism arguments](../inference.md#parallelism-arguments)).                                    |
| `-i`                           | Sample file. `t2v.json` sets `model_mode` to `text2video`, the sample `name` and the prompt; other fields use [built-in defaults](../inference.md#sample-arguments). |
| `-o`                           | Output directory (required).                                                                                                                                         |
| `--checkpoint-path`            | A model name such as `Cosmos3-Nano`, downloaded from Hugging Face, or a local checkpoint directory.                                                                  |
| `--seed`                       | Random seed for reproducibility.                                                                                                                                     |

## Check the output

Each sample writes to a folder named after its `name`, `t2v` here:

```text
outputs/omni_nano/
├── console.log
├── debug.log
├── setup_args.json           # resolved setup
└── t2v/
    ├── sample_args.json      # resolved sample arguments
    ├── sample_outputs.json   # generation status
    └── vision.mp4            # generated video
```

Running the command again overwrites matching outputs. Pass a new `-o` directory to keep earlier results.

## Troubleshooting

- Out of GPU memory: see [Torch CUDA out of memory](../inference.md#torch-cuda-out-of-memory-error), or switch to [Cosmos3-Edge](../inference.md#cosmos3-edge), which fits on one GPU and supports every mode except audio.
- Download, install or import errors: see [Downloading base checkpoints](../setup.md#downloading-base-checkpoints) and [Setup troubleshooting](../setup.md#troubleshooting).

## Next steps

- [Install](../setup.md): other install paths, CUDA variants and environment variables.
- [Run inference](../inference.md): every mode, multi-GPU launches and sample arguments.
- [Models](../models/README.md): which checkpoint fits which task.
- [Post-training](../post_training/choose_path.md): choose between SFT, distillation and RL.
- [Deploy](../deploy/serving.md): choose a serving path.
