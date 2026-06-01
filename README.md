<p align="center">
    <img src="https://github.com/user-attachments/assets/28f2d612-bbd6-44a3-8795-833d05e9f05f" width="274" alt="NVIDIA Cosmos"/>
</p>

<p align="center">
    <a href="https://github.com/NVIDIA/Cosmos">NVIDIA Cosmos</a> |
    🤗 <a href="https://huggingface.co/collections/nvidia/cosmos3">Cosmos 3 </a>
</p>

<p align="center">
    Part of the <a href="https://github.com/NVIDIA/Cosmos">NVIDIA Cosmos</a> project family — the training and serving framework repository.
</p>

# Cosmos-Framework

**Cosmos-Framework** is an end-to-end framework for training and serving world models, including the **Cosmos3** model family. Everything lives in a single top-level [`cosmos_framework/`](./cosmos_framework) Python package:

- **Training** — distributed FSDP / TP / CP / PP trainer, native DCP checkpoints with HuggingFace `safetensors` import/export, JSONL / WebDataset / LeRobot dataset adapters. Entry point: `cosmos_framework.scripts.train`. See [`docs/training.md`](./docs/training.md).
- **Inference** — Diffusers / Transformers / vLLM backends with offline batch generation and online serving (Ray + Gradio). Entry point: `cosmos_framework.scripts.inference`. Ecosystem-facing shim libraries (lightweight standalone wrappers for downstream projects) live under [`packages/`](./packages).

## Cosmos 3

**Cosmos 3** is our newest model family [[Report]](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) [[Website]](https://research.nvidia.com/labs/cosmos-lab/cosmos3/). It is a suite of omnimodal world models designed to jointly process and generate language, images, video, audio, and action sequences within a unified Mixture-of-Transformers architecture. By supporting highly flexible input-output configurations, it seamlessly unifies critical modalities for Physical AI — effectively subsuming vision-language models, video generators, world simulators, and world-action models into a single framework. For a guided experience to test out Cosmos3, please visit [[Cosmos]](https://github.com/nvidia/cosmos).

## Framework Documentation

- [Quickstart](#setup)
- [Setup](./docs/setup.md)
- [Training (Supervised Fine-Tuning)](./docs/training.md)
  - [JSONL Dataset](./docs/dataset_jsonl.md)
- [Inference](./docs/inference.md)
- Reference
  - [Code Structure](./docs/code_structure.md)
  - [Environment Variables](./docs/environment_variables.md)
  - [FAQ](./docs/faq.md)
  - [AGENTS.md](./AGENTS.md)

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

For the full guide (data preparation, base-checkpoint conversion, parallelism strategies, mixed precision, resuming), see [Training](./docs/training.md). The number of GPUs required depends on the recipe; the shipped recipes under [`examples/`](./examples/README.md) are 8-GPU configurations (tested on 8× H100 80 GB) launched via their paired launch shells, e.g.:

```shell
bash examples/launch_sft_vision_nano.sh
```

Users may adjust the GPU count to match their model and underlying hardware architecture — tune `NPROC_PER_NODE` and the parallelism degrees (DP/CP/FSDP shard) in the recipe accordingly.

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
| [Training](./docs/training.md)                               | Launching multi-GPU and multi-node runs; parallelism strategies; mixed precision; resuming.                              |
| [Inference (from a trained checkpoint)](./docs/inference.md) | Loading a trained checkpoint into one of the inference backends.                                                         |
| [FAQ](./docs/faq.md)                                         | Troubleshooting (OOM, NCCL hangs, slow training), environment variables, and common pitfalls.                            |
