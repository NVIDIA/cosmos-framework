<p align="center">
    <img src="https://github.com/user-attachments/assets/28f2d612-bbd6-44a3-8795-833d05e9f05f" width="274" alt="NVIDIA Cosmos"/>
</p>

<p align="center">🤗 <a href="https://huggingface.co/collections/nvidia-cosmos-ea/cosmos3-ea">Hugging Face</a></p>

# Cosmos-Framework

**Cosmos-Framework** is an end-to-end framework for training and serving world foundation models, including the **Cosmos3** model family. Everything lives in a single top-level [`cosmos_framework/`](./cosmos_framework) Python package:

- **Training** — distributed FSDP / TP / CP / PP trainer, native DCP checkpoints with HuggingFace `safetensors` import/export, JSONL / WebDataset / LeRobot dataset adapters. Entry point: `cosmos_framework.scripts.train`. See [`docs/training.md`](./docs/training.md).
- **Inference** — Diffusers / Transformers / vLLM backends with offline batch generation and online serving (Ray + Gradio). Entry point: `cosmos_framework.scripts.inference`. Ecosystem-facing shim libraries (lightweight standalone wrappers for downstream projects) live under [`packages/`](./packages).

## Documentation

- [Gallery](./docs/gallery.md)
- [Quickstart](#setup)
- [Setup](./docs/setup.md)
- [Training (Supervised Fine-Tuning)](./docs/training.md)
  - [JSONL Dataset](./docs/dataset_jsonl.md)
  - [Action Policy Closed-Loop Evaluation on LIBERO](./docs/action_policy_closed_loop_eval.md)
- [Prompting](./docs/prompting.md)
- [Inference](./docs/inference.md)
- Reference
  - [Code Structure](./docs/code_structure.md)
  - [Environment Variables](./docs/environment_variables.md)
  - [FAQ](./docs/faq.md)
  - [AGENTS.md](./AGENTS.md)

## Overview

**Cosmos3** is a world foundation model that unifies understanding and generation within a single Mixture-of-Transformer (MoT) architecture. Two tightly coupled towers—a **Reasoner** (vision-language model) and a **Generator** (world simulator)—share latent representations so that structured perception directly grounds realistic, temporally consistent simulation.

<p align="center"><img width="930" height="545" alt="Image" src="https://github.com/user-attachments/assets/81ec0329-a425-4a62-a18b-da0a66672e1f" /></p>

One model, many capabilities:

| Input Modality          | Output Modality | Application           | Status       |
| ----------------------- | --------------- | --------------------- | ------------ |
| Video \| Text           | Video           | Video Generator       | ✅           |
| Video \| Text           | Text            | Vision Language Model | ✅           |
| Action \| Video \| Text | Video           | World Model           | ✅           |
| Video \| Text           | Video & Action  | Policy Model          | ✅           |

## Setup

For more details and alternative installation methods, see [Setup](./docs/setup.md#installation). Before installing, make sure your machine meets the [System Requirements](./docs/setup.md#system-requirements). If you want a curated PyTorch + CUDA environment, start from the [recommended NVIDIA NGC base image](./docs/setup.md#recommended-base-image).

Install system dependencies:

```shell
sudo apt-get install -y --no-install-recommends curl ffmpeg git-lfs libx11-dev tree wget
```

Install the package with `uv` (pick the dependency group that matches your CUDA toolkit — see [CUDA Variants](./docs/setup.md#cuda-variants)):

```shell
# CUDA 13.0 (recommended)
uv sync --all-extras --group=cu130-train
# Or, for CUDA 12.8:
# uv sync --all-extras --group=cu128-train
source .venv/bin/activate && export LD_LIBRARY_PATH=
```

If you are starting from the recommended NGC image (`nvcr.io/nvidia/pytorch:25.09-py3`), see the [one-shot quickstart](./docs/setup.md#quickstart-from-the-recommended-base-image).

## Training

For the full guide (data preparation, base-checkpoint conversion, parallelism strategies, mixed precision, resuming), see [Training](./docs/training.md). A minimal single-GPU training launch looks like:

```shell
python -m cosmos_framework.scripts.train --sft-toml=examples/toml/sft_config/<recipe>.toml
```

## Prompting

See [Prompting](./docs/prompting.md).

## Inference

See [Inference](./docs/inference.md) for the full guide — launch commands, supported modes, parallelism presets, and troubleshooting.

Quick single-GPU launch:

```shell
python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i "inputs/omni/t2v.json" \
    -o outputs/omni_nano \
    --checkpoint-path Cosmos3-Nano \
    --seed=0
```

## Reference

| Topic                                                        | What it covers                                                                                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| [Setup](./docs/setup.md)                                     | Hardware/software prerequisites, `uv` install paths, CUDA variants, Docker base image, and base-checkpoint downloading.  |
| [Code Structure](./docs/code_structure.md)                   | Repository layout and a per-subpackage tour of `cosmos_framework/` — where each concern lives and where to add new code. |
| [Training](./docs/training.md)                               | Launching single-GPU, multi-GPU, and multi-node runs; parallelism strategies; mixed precision; resuming.                 |
| [Inference (from a trained checkpoint)](./docs/inference.md) | Loading a trained checkpoint into one of the inference backends.                                                         |
| [FAQ](./docs/faq.md)                                         | Troubleshooting (OOM, NCCL hangs, slow training), environment variables, and common pitfalls.                            |
