# Inference

> **Skill:** `.agents/skills/cosmos3-inference/SKILL.md`

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Quick Start](#quick-start)
  - [Single-GPU](#single-gpu)
  - [Multi-GPU](#multi-gpu)
- [Models](#models)
- [Modes](#modes)
- [Parallelism Arguments](#parallelism-arguments)
- [Sample Arguments](#sample-arguments)
  - [Text](#text)
  - [Vision (Image/Video)](#vision-imagevideo)
  - [Action](#action)
    - [Action Configuration](#action-configuration)
  - [Custom Defaults](#custom-defaults)
- [Schema Reference](#schema-reference)
- [Troubleshooting](#troubleshooting)
  - [Checkpoint Issue](#checkpoint-issue)
  - [Torch CUDA Out of Memory Error](#torch-cuda-out-of-memory-error)
  - [NCCL Issue](#nccl-issue)
    - [NCCL Plugin Issue](#nccl-plugin-issue)

______________________________________________________________________

<!--TOC-->

Prerequisites:

- [Setup](../README.md#setup)
- [Environment Variables](./environment_variables.md)

Arguments:

- `-i`, `--input-files`: Path to the sample argument file(s) (JSON, JSONL, YAML). Accepts quoted glob patterns (e.g. `"inputs/*.json"`).
- `-o`, `--output-dir`: Output directory.

Outputs:

- `<sample_name>/`
  - `sample_args.json`: Sample arguments.
  - `sample_outputs.json`: Generation status, action (if enabled).
  - `vision.jpg`, `vision.mp4`: Vision output (if enabled).

To see all available arguments:

```shell
python -m cosmos_framework.scripts.inference --help
```

## Quick Start

### Single-GPU

Use `python -m` directly. Suitable for `--parallelism-preset=latency` on a single GPU, or for quick experimentation:

```shell
python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i "inputs/omni/t2v.json" \
    -o outputs/omni_nano \
    --checkpoint-path Cosmos3-Nano \
    --seed=0
```

### Multi-GPU

Use `torchrun --nproc-per-node=N` when launching across multiple GPUs (N > 1). Both parallelism presets (`latency` and `throughput`) work with any GPU count, but throughput typically scales with N for batch generation:

```shell
torchrun --nproc-per-node=8 -m cosmos_framework.scripts.inference \
    --parallelism-preset=throughput \
    -i "inputs/omni/*.json" \
    -o outputs/omni_nano \
    --checkpoint-path Cosmos3-Nano \
    --seed=0
```

**Note:** The progress bar only prints on rank 0.

## Models

| Model         | Arguments                         | Modes                                          |
| ------------- | --------------------------------- | ---------------------------------------------- |
| Cosmos3-Nano  | `--checkpoint-path=Cosmos3-Nano`  | All                                            |
| Cosmos3-Super | `--checkpoint-path=Cosmos3-Super` | `text2image`, `text2video`, `image2video`      |

## Modes

`model_mode` selects the generation modality. The table below lists every supported mode with its required sample fields and a paired example file.

| `model_mode`       | Inputs                                     | Outputs                                              | Required sample fields  | Example                                                                                               |
| ------------------ | ------------------------------------------ | ---------------------------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------- |
| `text2image`       | text prompt                                | `vision.jpg`                                         | `prompt`                | [`inputs/omni/t2i.json`](../inputs/omni/t2i.json)                                                     |
| `text2video`       | text prompt                                | `vision.mp4`                                         | `prompt`                | [`inputs/omni/t2v.json`](../inputs/omni/t2v.json)                                                     |
| `image2image`      | text prompt + image                        | `vision.jpg`                                         | `prompt`, `vision_path` | [`inputs/omni/i2i.json`](../inputs/omni/i2i.json)                                                     |
| `image2video`      | text prompt + image                        | `vision.mp4`                                         | `prompt`, `vision_path` | [`inputs/omni/i2v.json`](../inputs/omni/i2v.json)                                                     |
| `video2video`      | text prompt + video                        | `vision.mp4`                                         | `prompt`, `vision_path` | [`inputs/omni/v2v.json`](../inputs/omni/v2v.json)                                                     |
| `forward_dynamics` | observation image/video + prompt + actions | future visual rollout in `vision.jpg` / `vision.mp4` | `action_path`           | [`inputs/omni/action_forward_dynamics_robot.json`](../inputs/omni/action_forward_dynamics_robot.json) |
| `inverse_dynamics` | observation video + prompt                 | predicted action sequence in `sample_outputs.json`   | `raw_action_dim`        | [`inputs/omni/action_inverse_dynamics_av.json`](../inputs/omni/action_inverse_dynamics_av.json)       |
| `policy`           | current observation image/video + prompt   | predicted action sequence + any visual output        | `raw_action_dim`        | [`inputs/omni/action_policy_robot.json`](../inputs/omni/action_policy_robot.json)                     |

Set `enable_sound: true` on a `text2video` sample (see [`inputs/omni/t2av.json`](../inputs/omni/t2av.json)) to also generate audio. To run every example in one batch, use `-i "inputs/omni/*.json"`.

## Parallelism Arguments

Both presets work with one or more GPUs.

- `--parallelism-preset`
  - `latency`: Minimize wall-clock per sample by maximizing **context parallelism** — each sample is split across all visible GPUs. Used for real-time jobs.
  - `throughput`: Maximize samples per second by maximizing **batch size** — no context parallelism; defaults to `batch_size = num_gpu`. Used for batch jobs.
- `--max-num-seqs`: Batch size per GPU (overrides the default for `throughput`).

## Sample Arguments

Sample arguments are read from multiple sources (in priority order):

- CLI overrides (e.g. `--model-mode=text2video`): Overrides for all samples.
- Input files (e.g. `--input-files "inputs/omni/*t2i*.json"`): Single sample per input.
- Defaults: `cosmos_framework/inference/defaults/<model_mode>`: Defaults for all samples.

For debugging, the full set of sample arguments is saved to `<output_dir>/<sample_name>/sample_args.json`.

Common arguments:

- `model_mode`: Generation modality. See [Modes](#modes) above for all options.
- `seed`: Random seed for reproducibility.

**Note:** Condition file paths are relative to the input file.

### Text

- `prompt`: Inline text prompt.

### Vision (Image/Video)

Common arguments:

- `fps`: Condition and output frames per second.
- `resolution` (`"256"`, `"480"`, `"720"`): Condition and output resolution (height in pixels).
- `aspect_ratio` (`1,1`, `4,3`, `"3,4`, `16,9`, `9,16`): Condition and output aspect ratio. Defaults to `16,9`.

Condition arguments:

- `vision_path`: Path to an image or video file (local path or URL).

Generation arguments:

- `num_frames`: Number of output frames. `1` = image; `≥24` = video.

Outputs `vision.jpg` or `vision.mp4` depending on `num_frames`.

### Action

Common arguments:

- `action_chunk_size`: Number of action steps in the chunk. The action media loader reads at most `action_chunk_size + 1` observation frames.
- `domain_name`: Domain name passed to the action domain registry, such as `libero` or `av`.

Condition arguments:

- `action_path`: JSON action sequence. Required for `forward_dynamics`; each row is one action step and each column is one raw action dimension.
- `image_size`: Action input resize bucket. The value is passed as the action media resolution bucket; examples use `256` for LIBERO and `480` for AV.

Generation arguments:

- `raw_action_dim`: Raw action width to return for generated actions. Required for `inverse_dynamics` and `policy`.

The action output is written to `sample_outputs.json`.

See the [Modes](#modes) table above for the action mode inputs/outputs and example files.

#### Action Configuration

The action sample fields control input preprocessing and action tensor shape. `action_chunk_size` should match the chunk length used by the checkpoint. `image_size` should match the action training/evaluation resolution bucket. `domain_name` must be compatible with the checkpoint's action domain registry.

Action tensors are padded to `model.config.max_action_dim` before generation. Set it with `--experiment_overrides "[model.config.max_action_dim=<D>]"` when the checkpoint config does not already define the desired padded width. Use a value greater than or equal to the raw action width in `action_path` or `raw_action_dim`.

### Custom Defaults

To use your own default values instead of the built-in presets, pass a JSON file via the `defaults_file` field in your sample arguments:

```json
{
    "defaults_file": "my_defaults.json",
    "prompt": "..."
}
```

The custom defaults file has the same format as the built-in presets. Fields you set explicitly in the sample argument file still take precedence over the custom defaults file.

## Schema Reference

The `schemas/` directory contains auto-generated reference files listing every available argument with types, constraints, and descriptions. These files are the authoritative reference for field names and valid values.

- [`OmniSetupOverrides.yaml`](../schemas/OmniSetupOverrides.yaml): All setup/CLI arguments with default values and inline comments.
- [`OmniSetupOverrides.schema.json`](../schemas/OmniSetupOverrides.schema.json): JSON Schema with types, enums, and validation constraints.
- [`OmniSampleOverrides.yaml`](../schemas/OmniSampleOverrides.yaml): All sample arguments with default values and inline comments.
- [`OmniSampleOverrides.schema.json`](../schemas/OmniSampleOverrides.schema.json): JSON Schema with types, enums, and validation constraints.

## Troubleshooting

### Checkpoint Issue

If you encounter failures downloading checkpoints, refer to [Downloading Base Checkpoints](./setup.md#downloading-base-checkpoints).

Checkpoint download commands are printed to the console. You can run them manually to debug issues.

### Torch CUDA Out of Memory Error

Error: `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X MiB`

[Optimize memory allocation](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf):

```shell
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### NCCL Issue

Error:

```shell
[rank0]:[W415 18:57:09.249883195 ProcessGroupNCCL.cpp:5138] Guessing device ID based on global rank. This can cause a hang if rank to GPU mapping is heterogeneous. You can specify device_id in init_process_group()

Fatal Python error: Segmentation fault
```

Re-run with debugging enabled:

```shell
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1
```

#### NCCL Plugin Issue

Error:

```shell
NCCL INFO Failed to initialize NET plugin Libfabric

Fatal Python error: Segmentation fault
```

Fix:

```shell
export NCCL_NET_PLUGIN=none
```
