<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: OpenMDW-1.1 -->

# DROID Action-Policy Post-Training — `Cosmos3-Nano-Policy-DROID`

> **STATUS: recipe ships in this package.** The registered experiment, the DROID action
> dataset class (`joint_pos` 8D + `use_state`), and the EMA warm-start fix land here.
> To run it you supply two external inputs — a prepared **DROID LeRobot v3.0** dataset and
> a **DCP base checkpoint** converted from `nvidia/Cosmos3-Nano` (see
> [Inputs you provide](#inputs-you-provide)). Validated end-to-end on H200: 1 node / 8 GPU
> and 2 nodes / 16 ranks (HSDP).

Fine-tune `Cosmos3-Nano` (the 8B MoT) into an action policy on the **DROID LeRobot** dataset,
reproducing `Cosmos3-Nano-Policy-DROID`. The policy is initialized from **`nvidia/Cosmos3-Nano`**
(public Hugging Face repo) and trained with absolute joint-position actions + proprioceptive
state at 480p.

______________________________________________________________________

## Inputs you provide

This package ships the training stack — the registered `action_policy_droid_nano` experiment,
the DROID action dataset class with the recipe knobs (`action_space=joint_pos`, `use_state`,
`concat_view`), and the EMA warm-start in `checkpoint/dcp.py`. Two inputs are external and must
be provided per environment:

1. **Prepared DROID LeRobot v3.0 dataset** — the LeRobot v2.0→v3.0 conversion + success
   filtering is run out-of-band (not yet in this repo). Point `DROID_ROOT` at the resulting
   `…/droid_lerobot/success` directory (must contain `meta/info.json`).
2. **DCP base checkpoint** — convert `nvidia/Cosmos3-Nano` to DCP and point
   `BASE_CHECKPOINT_PATH` at it (see [Full reproduction](#full-reproduction)). Action heads are
   not loaded from it (they init fresh).

## Dataset — DROID LeRobot

To be released.

## Recipe

| knob              | value                                                               |
| ----------------- | ------------------------------------------------------------------- |
| init              | `nvidia/Cosmos3-Nano` (public Hugging Face repo)                    |
| action space      | `joint_pos` (absolute joint position, 8-D incl. gripper)            |
| state             | `use_state=true` (proprioception; valid only with `joint_pos`)      |
| resolution        | `480`                                                               |
| viewpoint / video | `concat_view` / `video_mode=null`                                   |
| chunk length      | `32` (tokenizer `encode_exact_durations=[33]`)                      |
| lr                | `2e-4`                                                              |
| samples/rank      | `32` (H200-safe; 64 OOMs at 480p). global batch = `32 × world_size` |
| eval              | disabled for the reproduction run                                   |

## Full reproduction

The OSS flow mirrors the other recipes (see [docs/training.md](./training.md)):

```shell
# Step 1: prepare DROID LeRobot v3.0 success split -> $DATASET_PATH (see "Inputs you provide")

# Step 2: convert the base checkpoint -> $BASE_CHECKPOINT_PATH
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path Cosmos3-Nano \
  -o $BASE_CHECKPOINT_PATH 

# Step 3: launch. The TOML selects the experiment + scalars; the dataset/action
# knobs come from the registered experiment.
export DATASET_PATH=/path/to/dataset/success
export BASE_CHECKPOINT_PATH=/path/to/base_checkpoint
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export NPROC_PER_NODE=8
bash examples/launch_sft_action_policy_droid.sh
```

The recipe TOML (`examples/toml/sft_config/action_policy_droid_repro.toml`) sets the scalar
knobs (`max_iter`, `save_iter`, `grad_clip`, parallelism, wandb); the dataset/action knobs
(`joint_pos`, `use_state`, `concat_view`, 480p, chunk 32, count-based batch) live in the
registered `action_policy_droid_nano` experiment per the schema's design. For multi-node HSDP,
set `model.parallelism.data_parallel_replicate_degree = <num_nodes>` (intra-node shard stays 8).

## Smoke reproduction

Config/import/data sanity without burning a full run: small node count + a handful of iters via
`--config-overrides "trainer.max_iter=10" "checkpoint.save_iter=10"` (and a small
`data_parallel_shard_degree`). Use this to validate the recipe composes and the dataset opens
before any large allocation.

## Checkpoints

- Saved every `save_iter` iters (1000 in the validated run) to the object store, at
  `<bucket>/<project>/<group>/<job.name>/checkpoints/iter_<N>/`.
- The run is **resumable** from the latest checkpoint (re-launch with the same `job.name`).
- Export to HF safetensors via `cosmos_framework.scripts.export_model` (see [docs/training.md](./training.md)).

## Non-goals

- **Closed-loop / action evaluation is out of scope** for this reproduction pass (training
  reproduction only), unless explicitly expanded.
