# FAQ

> **Skills:** `.agents/skills/cosmos3-setup/SKILL.md` · `.agents/skills/cosmos3-inference/SKILL.md`

A catch-all collection of frequently asked questions, tips, and troubleshooting for the Cosmos3 package. Can't find what you need? Check [setup.md](./setup.md) for installation issues or [inference.md](./inference.md) for inference details.

To add a new entry, append it under the most relevant section — or under [Miscellaneous](#miscellaneous) if nothing fits.

---

## Table of Contents

- [Setup and Installation](#setup-and-installation)
- [Configuration and Defaults](#configuration-and-defaults)
- [Inference](#inference)
- [Training](#training)
- [Tips and Tricks](#tips-and-tricks)
- [Miscellaneous](#miscellaneous)

---

## Setup and Installation

### Q: I get `ImportError: cannot import name '_functionalization' from 'torch._C'` inside an NGC container

Clear the library path before running anything:

```shell
export LD_LIBRARY_PATH=''
```

This is needed because the NGC PyTorch container ships its own libraries that conflict with the venv-installed versions. See [setup.md#pytorch-import-issue](./setup.md#pytorch-import-issue).

### Q: `ModuleNotFoundError: No module named 'cosmos_framework'`

Make sure you installed the package:

```shell
uv sync --all-extras --group=cu130
source .venv/bin/activate
```

If already installed, try `--reinstall` to force a clean state.

### Q: Which CUDA version should I use?

CUDA 13.0 is recommended. CUDA 12.8 is also supported. The major version must match between your system CUDA and the installed PyTorch wheels. Check with:

```shell
nvidia-smi                                    # system CUDA
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA
```

### Q: I get a CUDA / NVIDIA driver mismatch error (e.g. `CUDA error: no kernel image is available for execution on the device`, `libcudart.so.* cannot open`, `The NVIDIA driver on your system is too old`)

The installed PyTorch CUDA wheels do not match your system's NVIDIA driver. Check the driver's reported CUDA version with `nvidia-smi`, then delete `.venv/` and `uv sync` against the matching group (`cu130-train` if the driver supports CUDA 13.x; `cu128-train` if it supports CUDA 12.8):

```shell
nvidia-smi                                    # check the "CUDA Version" field (top right)
rm -rf .venv
uv sync --all-extras --group=cu130-train --reinstall   # or --group=cu128-train
source .venv/bin/activate && export LD_LIBRARY_PATH=
```

Use the inference-only groups (`cu130` / `cu128`) instead if you don't need the training-only dependencies.

### Q: How do I download model checkpoints?

Checkpoints are downloaded automatically from Hugging Face during inference. You need:

1. A [Hugging Face token](https://huggingface.co/settings/tokens) with Read permission
2. Accepted [NVIDIA Open Model License Agreement](https://huggingface.co/nvidia/Cosmos-Guardrail1)
3. `HF_TOKEN` environment variable set, or `uvx hf auth login`

Control the download location with `HF_HOME` (default: `~/.cache/huggingface`). If downloads fail, the commands are printed to the console — run them manually to debug. See [setup.md#downloading-base-checkpoints](./setup.md#downloading-base-checkpoints).

### Q: `fatal error: Python.h: No such file or directory`

Reinstall uv and the venv from scratch:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install --reinstall
rm -rf .venv
uv sync --all-extras --group=cu130 --reinstall
source .venv/bin/activate
```

### Q: How much disk space do I need?

Plan for ~150 GiB free before the first run. A successful first-run inference or training workflow typically consumes:

- **Hugging Face cache** (`$HF_HOME`, default `~/.cache/huggingface`): ~90 GiB — base checkpoints (e.g. Cosmos3-Nano, Wan2.2 VAE), tokenizers, and any dataset snapshots pulled by training recipes.
- **uv cache** (`$UV_CACHE_DIR`, default `~/.cache/uv`): ~20 GiB — wheels for torch/CUDA dependencies across the install groups (`cu130-train`, `cu128-train`, etc.).
- **Run outputs** (`$IMAGINAIRE_OUTPUT_ROOT`, training, or your `-o` output dir, inference): ~30 GiB per run — config snapshots, DCP checkpoints saved every `save_freq` iterations, callback outputs, optional wandb files.

Actual sizes scale with the model tier (Cosmos3-Super is larger than Cosmos3-Nano), the dataset, and how many checkpoints you keep. To relocate any of these off the system disk, set the corresponding env var before installation/run (e.g. `export HF_HOME=/data/hf`, `export UV_CACHE_DIR=/data/uv`, `export IMAGINAIRE_OUTPUT_ROOT=/data/cosmos-runs`).

---

## Configuration and Defaults

### Q: Where are the default inference parameters (guidance, shift, num_steps, etc.)?

Per-modality defaults live in JSON files under `cosmos_framework/inference/defaults/<mode>/sample_args.json`:

| Mode          | Default file                                                       |
| ------------- | ------------------------------------------------------------------ |
| `text2image`  | `cosmos_framework/inference/defaults/text2image/sample_args.json`  |
| `text2video`  | `cosmos_framework/inference/defaults/text2video/sample_args.json`  |
| `image2video` | `cosmos_framework/inference/defaults/image2video/sample_args.json` |

Action and image/video-to-video modes have parallel files under `cosmos_framework/inference/defaults/{image2image,video2video,forward_dynamics,inverse_dynamics,policy}/sample_args.json`.

See [AGENTS.md](../AGENTS.md) for the full config defaults chain.

### Q: How do I override a default parameter?

From most temporary to most permanent:

1. **CLI flag**: `--shift 5.0` (per-run, applies to all samples)
2. **Sample argument file**: set the field in your input JSON (per-sample)
3. **Custom defaults file**: pass `"defaults_file": "my_defaults.json"` in your sample argument file (see [inference.md#custom-defaults](./inference.md#custom-defaults))
4. **Built-in default**: edit `cosmos_framework/inference/defaults/<mode>/sample_args.json` (permanent change)

Fields set in the sample argument file take precedence over defaults. CLI flags override both.

### Q: What is the `shift` parameter and which value should I use?

`shift` controls the time-shift in the UniPC diffusion sampler. Higher values produce more detail but can introduce artifacts. Recommended values:

| Model               | Recommended shift |
| ------------------- | ----------------- |
| Cosmos3-Nano (8B)   | `10.0` (default)  |
| Cosmos3-Super (32B) | `5.0`             |

### Q: How do I add a new parameter to the inference pipeline?

1. Add the field to `SamplingArgs` and `SamplingOverrides` in `cosmos_framework/inference/args.py`
2. Add its default to each `cosmos_framework/inference/defaults/<mode>/sample_args.json`
3. Wire it through `OmniSampleOverrides.build_sample()` in `cosmos_framework/inference/args.py`

### Q: What does `defaults_file` do?

It lets you supply a custom JSON file of default values instead of the built-in presets. The format is the same as the files in `cosmos_framework/inference/defaults/`. Fields in your sample argument file still take precedence over the custom defaults. See [inference.md#custom-defaults](./inference.md#custom-defaults).

---

## Inference

### Q: How much GPU memory do the models need?

| Model               | GPU Memory |
| ------------------- | ---------- |
| Cosmos3-Nano (8B)   | 32 GB      |
| Cosmos3-Super (32B) | 128 GB     |

### Q: I get `torch.cuda.OutOfMemoryError` during inference

Try these in order:

1. **Reduce allocator fragmentation** — usually the cheapest fix:

   ```shell
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

2. **Increase `--dp-shard-size`** to shard model weights across more GPUs via FSDP. Inference auto-picks a value that fits the model at ~75% device memory (see `_get_dp_shard_size` in `cosmos_framework/inference/args.py`); passing a larger explicit value drops per-GPU memory at the cost of more all-gather traffic. Requires multi-GPU.
3. **Lower `--device-memory-utilization`** (default `0.75`). The auto-`dp_shard_size` formula is `ceil(model_memory / device_memory / utilization)`, so passing e.g. `--device-memory-utilization=0.5` forces auto-mode to pick a larger `dp_shard_size` and leaves more per-GPU headroom for activations / KV cache. Requires multi-GPU.
4. **Add `--offload-guardrail-models`** to move the text and video guardrail models to CPU. Frees the GPU memory they would otherwise hold for the full run, at the cost of some extra latency when guardrails are invoked.

See [inference.md#torch-cuda-out-of-memory-error](./inference.md#torch-cuda-out-of-memory-error) for the full troubleshooting section.

### Q: What is the difference between `latency` and `throughput` parallelism presets?

| Preset       | What it does                        | When to use                 |
| ------------ | ----------------------------------- | --------------------------- |
| `latency`    | Spreads each sample across all GPUs | Interactive / real-time use |
| `throughput` | One sample per GPU in parallel      | Large batch jobs            |

### Q: How do I generate images instead of videos?

Use a text2image input file:

```shell
python -m cosmos_framework.scripts.inference -i inputs/omni/t2i.json -o outputs/ --checkpoint-path Cosmos3-Nano
```

The modality is determined by the input JSON (`num_frames=1` for images), not by a separate flag. See `inputs/omni/t2i.json` for the format.

### Q: How many frames can I generate?

Depends on resolution:

| Resolution | Max frames |
| ---------- | ---------- |
| 256p       | 400        |
| 480p       | 300        |
| 720p       | 200        |

Default is 189 frames at 24 FPS (~7.9 seconds).

### Q: What input formats does image-to-video support?

Provide a `vision_path` pointing to an image (`.jpg`, `.jpeg`, `.png`) or a URL. See `inputs/omni/i2v.json` for the format.

### Q: How do I run online inference with Ray?

Install serve dependencies and start the server:

```shell
uv pip install -e ".[serve]"
python -m cosmos_framework.inference.ray.serve --parallelism-preset=latency -o outputs/ray_serve --checkpoint-path Cosmos3-Nano
```

Then submit requests via curl, the submit CLI, or the Gradio UI.

### Q: How do I fix a torch.compile error during inference?

Add a command-line argument `--no-use-torch-compile`

OR

Delete the torchinductor cache under the /tmp directory, `rm -rf /tmp/torchinductor_*`

---

### Q: I get `torch.distributed.DistNetworkError: ... port: 29500 ... EADDRINUSE, address already in use`

`torchrun` defaults its rendezvous to port `29500`. The error means that port is already taken on the node — usually because another `torchrun` job (yours or someone else's on a shared node) is still using it.

Pass a different free port with `--master-port`, placed **before** `-m` (it is a `torchrun` argument, not an inference argument):

```shell
torchrun --nproc-per-node=8 --master-port=29501 -m cosmos_framework.scripts.inference \
  --parallelism-preset=throughput \
  -i "inputs/omni/t2i.json" \
  -o outputs/omni_t2i \
  --checkpoint-path Cosmos3-Super-Text2Image \
  --seed=0
```

Any free port works (e.g. `29501`, `29510`); give each concurrent job on the same node a distinct port. Alternatively, `--rdzv-endpoint=localhost:0` lets `torchrun` auto-pick a free port.

---

## Training

### Q: I get `torch.cuda.OutOfMemoryError` during training (SFT)

Knobs are in the recipe TOML under `[model]`, `[model.parallelism]`, and `[dataloader_train]`. Try in order:

1. **Reduce allocator fragmentation** — usually the cheapest fix:

   ```shell
   export PYTORCH_ALLOC_CONF=expandable_segments:True
   ```

   (The `_super` launch shells already export this — see `examples/launch_sft_*_super.sh`.)

2. **Enable activation checkpointing** in `[model.activation_checkpointing]`:
   - `mode = "full"` — checkpoint every transformer block (largest memory savings, trades extra recompute for memory).
   - `mode = "selective"` — per-op SAC, MoT only (smaller savings, smaller overhead). Falls back to no checkpointing on the VLM path.

3. **Raise `[model.parallelism].data_parallel_shard_degree`** to shard weights/optimizer state across more ranks via FSDP. Runtime invariant (from `cosmos_framework/utils/vfm/parallelism.py:50-52`): `data_parallel_replicate_degree × data_parallel_shard_degree == WORLD_SIZE` always holds — `context_parallel_shard_degree` and `cfg_parallel_shard_degree` are *overlay* axes that share dp rank slots, not separate mesh dims. Use `-1` to let `data_parallel_shard_degree` auto-fill from `torchrun` world size.

4. **Raise `[model.parallelism].context_parallel_shard_degree`** to split the sequence dimension across ranks. Helpful when activations (not weights) drive the OOM — long videos, high resolution.

5. **Lower `[dataloader_train].max_samples_per_batch`** to cap samples per micro-batch. `None` lets the packer's token budget decide; setting an explicit small number trades throughput for headroom.

6. **Enable LoRA on a Cosmos3-Nano recipe.** Nano recipes are full-finetune by default (`lora_enabled = false`); setting `[model].lora_enabled = true` trains low-rank adapters instead of the full weights, dropping optimizer-state memory substantially. The `_super` recipes (e.g. `vision_sft_super`) are already LoRA-only, so this lever doesn't apply there.

See [docs/training.md](./training.md) for the full SFT setup and TOML reference (`[model.activation_checkpointing]`, `[model.parallelism]`, `[dataloader_train]` sections).

---

## Tips and Tricks

### Seed reproducibility

Always pass `--seed` when comparing runs. Without it, a random seed is used each time.

### Prompt upsampling

Short prompts produce worse results. Use the built-in prompt upsampler with a vLLM-served Qwen3 model:

```shell
python -m cosmos_framework.scripts.upsample_prompts -i "inputs/omni/*.json" -o outputs/upsample_prompts
```

### Batch inference resume

The inference script automatically skips samples whose output files already exist. If a run is interrupted, re-run the same command to resume.

### Generate all modalities at once

```shell
python -m cosmos_framework.scripts.inference -i "inputs/omni/*.json" -o outputs/ --checkpoint-path Cosmos3-Nano --seed=0
```

### CLI help

All available flags and their current defaults:

```shell
python -m cosmos_framework.scripts.inference --help
```

---

## Miscellaneous

*This section is a catch-all for tips that don't fit elsewhere. Add new entries freely.*

### Q: What are the example scripts in `examples/` for?

They illustrate how the inference logic works under the hood — `examples/inference.py` shows the low-level model API and `examples/inference_pipeline.py` shows the pipeline API. For production use, prefer `python -m cosmos_framework.scripts.inference`.

### Q: Where are the Ray Serve config files?

`cosmos_framework/inference/ray/configs/latency.yaml` and `cosmos_framework/inference/ray/configs/throughput.yaml`. These configure the Ray Serve deployment with different parallelism strategies.
