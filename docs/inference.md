# Inference

> **Skill:** `.agents/skills/cosmos3-inference/SKILL.md`

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Quick Start](#quick-start)
  - [Single-GPU](#single-gpu)
  - [Multi-GPU](#multi-gpu)
    - [Cosmos3-Nano](#cosmos3-nano)
    - [Cosmos3-Super](#cosmos3-super)
- [Models](#models)
- [Modes](#modes)
- [Parallelism Arguments](#parallelism-arguments)
- [Sample Arguments](#sample-arguments)
  - [Text](#text)
  - [Vision (Image/Video)](#vision-imagevideo)
  - [Action](#action)
  - [Custom Defaults](#custom-defaults)
- [Guardrails](#guardrails)
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
- [FAQ](./faq.md) — troubleshooting (OOM, NCCL hangs), defaults, common pitfalls.

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

**Note:** Cosmos3-Super (32B) does not fit on a single 80 GB H100 — see [Cosmos3-Super](#cosmos3-super) for the multi-GPU recipes.

### Multi-GPU

Use `torchrun --nproc-per-node=N` when launching across multiple GPUs (N > 1). By default the model weights are sharded (FSDP) across all N GPUs, so any model fits. The `throughput` preset runs that single sharded replica over a batch; the `latency` preset additionally needs `--dp-shard-size=1` on multiple GPUs so the ranks are free for context parallelism (see [Parallelism Arguments](#parallelism-arguments)).

#### Cosmos3-Nano

```shell
torchrun --nproc-per-node=8 -m cosmos_framework.scripts.inference \
    --parallelism-preset=throughput \
    -i "inputs/omni/*.json" \
    -o outputs/omni_nano \
    --checkpoint-path Cosmos3-Nano \
    --seed=0
```

**Note:** The progress bar only prints on rank 0.

**Note:** With the default full-GPU sharding, this same command also runs Cosmos3-Super (32B) on 8×80 GB H100 — the weights are sharded (FSDP) across all 8 GPUs. See [Cosmos3-Super](#cosmos3-super) for the explicit-axis variants and the 4-GPU recipe.

#### Cosmos3-Super

Cosmos3-Super (32B) must be sharded across multiple GPUs to fit in 80 GB H100 memory. The default already shards the model across every visible GPU (FSDP), so the `throughput` preset fits it directly; the commands below pin the axes explicitly (pure FSDP, no context- or CFG-parallelism overlay) and add the 4-GPU recipe.

**4 GPUs:**

```shell
torchrun --nproc-per-node=4 -m cosmos_framework.scripts.inference \
    --parallelism-preset=throughput \
    --dp-shard-size=4 --dp-replicate-size=1 \
    --cp-size=1 --cfgp-size=1 \
    -i "inputs/omni/*.json" \
    -o outputs/omni_super \
    --checkpoint-path Cosmos3-Super \
    --seed=0
```

**8 GPUs:**

```shell
torchrun --nproc-per-node=8 -m cosmos_framework.scripts.inference \
    --parallelism-preset=throughput \
    --dp-shard-size=8 --dp-replicate-size=1 \
    --cp-size=1 --cfgp-size=1 \
    -i "inputs/omni/*.json" \
    -o outputs/omni_super \
    --checkpoint-path Cosmos3-Super \
    --seed=0
```

The four `--{dp,cp,cfgp}-*-size` flags override the auto-selected values from `--parallelism-preset`. Super supports `text2image`, `text2video`, and `image2video` (see [Models](#models)).

## Models

| Model         | Arguments                         | Modes                                          |
| ------------- | --------------------------------- | ---------------------------------------------- |
| Cosmos3-Nano  | `--checkpoint-path=Cosmos3-Nano`  | All                                            |
| Cosmos3-Super | `--checkpoint-path=Cosmos3-Super` | `text2image`, `text2video`, `image2video`      |

## Modes

`model_mode` selects the generation modality. The table below lists every supported mode with its required sample fields and a paired example file.

| `model_mode`       | Inputs                                     | Outputs                                                                                    | Required sample fields                      | Example                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------ | ------------------------------------------ | ------------------------------------------------------------------------------------------ | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `text2image`       | text prompt                                | `vision.jpg`                                                                               | `prompt`                                    | [`inputs/omni/t2i.json`](../inputs/omni/t2i.json)                                                                                                                                                                                                                                                                                                                                                                        |
| `text2video`       | text prompt                                | `vision.mp4`                                                                               | `prompt`                                    | [`inputs/omni/t2v.json`](../inputs/omni/t2v.json)                                                                                                                                                                                                                                                                                                                                                                        |
| `image2video`      | text prompt + image                        | `vision.mp4`                                                                               | `prompt`, `vision_path`                     | [`inputs/omni/i2v.json`](../inputs/omni/i2v.json)                                                                                                                                                                                                                                                                                                                                                                        |
| `video2video`      | text prompt + video                        | `vision.mp4`                                                                               | `prompt`, `vision_path`                     | [`inputs/omni/v2v.json`](../inputs/omni/v2v.json)                                                                                                                                                                                                                                                                                                                                                                        |
| `forward_dynamics` | observation image/video + prompt + actions | future visual rollout in `vision.mp4`                                                      | `domain_name`, `vision_path`, `action_path` | [`inputs/omni/action_forward_dynamics_av.json`](../inputs/omni/action_forward_dynamics_av.json), [`inputs/omni/action_forward_dynamics_camera.json`](../inputs/omni/action_forward_dynamics_camera.json), [`inputs/omni/action_forward_dynamics_robot.json`](../inputs/omni/action_forward_dynamics_robot.json), [`inputs/omni/action_forward_dynamics_batch.jsonl`](../inputs/omni/action_forward_dynamics_batch.jsonl) |
| `inverse_dynamics` | observation video + prompt                 | predicted action sequence in `sample_outputs.json`                                         | `domain_name`, `vision_path`                | [`inputs/omni/action_inverse_dynamics_av.json`](../inputs/omni/action_inverse_dynamics_av.json), [`inputs/omni/action_inverse_dynamics_robot.json`](../inputs/omni/action_inverse_dynamics_robot.json), [`inputs/omni/action_inverse_dynamics_batch.jsonl`](../inputs/omni/action_inverse_dynamics_batch.jsonl)                                                                                                          |
| `policy`           | observation image/video + prompt           | predicted action sequence in `sample_outputs.json` + future visual rollout in `vision.mp4` | `domain_name`, `vision_path`                | [`inputs/omni/action_policy_av.json`](../inputs/omni/action_policy_av.json), [`inputs/omni/action_policy_robot.json`](../inputs/omni/action_policy_robot.json), [`inputs/omni/action_policy_batch.jsonl`](../inputs/omni/action_policy_batch.jsonl)                                                                                                                                                                      |

Set `enable_sound: true` on a `text2video` sample (see [`inputs/omni/t2vs.json`](../inputs/omni/t2vs.json)) to also generate audio. To run every example in one batch, use `-i "inputs/omni/*.json"`.

## Parallelism Arguments

By default the model weights are sharded (FSDP) across **all** visible GPUs (`dp_shard_size = WORLD_SIZE`, `dp_replicate_size = 1`), so any model fits regardless of size. Override any axis with the `--dp-shard-size` / `--dp-replicate-size` / `--cp-size` / `--cfgp-size` flags.

- `--parallelism-preset`
  - `latency`: Minimize wall-clock per sample by splitting each sample across GPUs with **context parallelism**. On multiple GPUs, also pass `--dp-shard-size=1` so the ranks are used for context/CFG parallelism instead of weight sharding. Used for real-time jobs.
  - `throughput`: No context parallelism (`cp=cfgp=1`); the model is sharded across all GPUs and a single replica processes the batch. Used for batch jobs.
- `--dp-shard-size`: Number of ranks the model is sharded over (FSDP). Defaults to all ranks (`WORLD_SIZE`).
- `--max-num-seqs`: Maximum number of samples batched together per replica.

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

- `num_frames`: Number of output frames. `1` = image; `≥24` = video. Default 189; resolution-dependent max — see [FAQ § How many frames can I generate?](./faq.md#q-how-many-frames-can-i-generate).

Outputs `vision.jpg` or `vision.mp4` depending on `num_frames`.

### Action

Common arguments:

- `action_chunk_size`: Number of action steps in the chunk. The action media loader reads at most `action_chunk_size + 1` observation frames.
- `domain_name`: Domain name passed to the action domain registry, such as `bridge_orig_lerobot`, `camera_pose`, or `av`.
- `view_point`: Viewpoint description injected into the action prompt, such as `ego_view`.

Condition arguments:

- `action_path`: JSON action sequence. Required for `forward_dynamics`; each row is one action step and each column is one raw action dimension.
- `image_size`: Action input resize bucket. The value is passed as the action media resolution bucket; examples use `256` for LIBERO and `480` for AV.

The action output is written to `sample_outputs.json`.

See the [Modes](#modes) table above for the action mode inputs/outputs and example files.

### Custom Defaults

To use your own default values instead of the built-in presets, pass a JSON file via the `defaults_file` field in your sample arguments:

```json
{
    "defaults_file": "my_defaults.json",
    "prompt": "..."
}
```

The custom defaults file has the same format as the built-in presets. Fields you set explicitly in the sample argument file still take precedence over the custom defaults file.

## Guardrails

Inference ships with guardrails enabled by default, sourced from [nvidia/Cosmos-Guardrail1](https://huggingface.co/nvidia/Cosmos-Guardrail1). Active filters: text blocklist (better-profanity + fuzzy match), text safety classifier ([Qwen/Qwen3Guard-Gen-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B)), video content-safety classifier, and RetinaFace face-blur post-processor. Pass `--no-guardrails` to disable, or `--offload-guardrail-models` to keep them on CPU between calls (saves GPU memory, adds latency).

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

If that's not enough, see [FAQ § OOM during inference](./faq.md#q-i-get-torchcudaoutofmemoryerror-during-inference) for the full ladder (`--dp-shard-size`, `--device-memory-utilization`, `--offload-guardrail-models`).

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
