# Cosmos3 DROID Action-Policy Post-Training

This example post-trains [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) into a DROID action policy on the [Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID) dataset, reproducing [Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID). The model predicts absolute joint-position actions conditioned on proprioceptive state and video observations at a resolution of 480p (`640×360`). Experiment `action_policy_droid_nano`, launcher `launch_sft_action_policy_droid_nano.sh`.

Two external inputs are required: (1) a pre-downloaded Cosmos3-DROID dataset in LeRobotDataset v3.0 format, and (2) a DCP base checkpoint converted from the chosen base model (Cosmos3-Nano or Cosmos3-Edge).

The recipe runs multi-node via HSDP (single node / 8 GPUs and beyond).

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Prerequisites](#prerequisites)
- [Inputs You Provide](#inputs-you-provide)
- [Recipe](#recipe)
- [Full Reproduction](#full-reproduction)
- [Checkpoints](#checkpoints)

______________________________________________________________________

<!--TOC-->

## Prerequisites

- [Setup](../README.md#setup) — clone the repo, install the training extras (`uv sync --all-extras --group=cu130-train`), and activate the environment.
- [Environment Variables](./environment_variables.md) — set environment variables.
- [FAQ](./faq.md) — troubleshooting (OOM during SFT, defaults) and common pitfalls.

The runnable artifacts (TOML recipe, paired launch shell) live in [`examples/`](../examples); all commands below run from the repo root with the environment activated.

## Inputs You Provide

This package ships the training stack — the registered `action_policy_droid_nano`
experiment, the DROID action dataset class with the recipe knobs
(`action_space=joint_pos`, `use_state`, `concat_view`), and the EMA warm-start in
`checkpoint/dcp.py`. Two inputs are external and must be provided per environment:

1. **[Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID) dataset (in LeRobotDataset v3.0 format)** — pre-download the
   dataset into a directory named `droid_plus_lerobot_640x360_20260412` (the loader resolves the
   dataset schema from the directory name) and point `DROID_ROOT` at that directory — the parent
   containing `success/`, not `.../success` itself.
2. **DCP base checkpoint** — convert [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) to DCP and point
   `BASE_CHECKPOINT_PATH` at it (see [Full Reproduction](#full-reproduction)). Action heads are
   not loaded from it (they init fresh).

## Recipe

| knob              | value                                                                                                                                              |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| init              | `Cosmos3-Nano` (public Hugging Face repo)                                                                                                          |
| action space      | `joint_pos` (absolute joint position, 8-D incl. gripper)                                                                                           |
| state             | `use_state=true` (proprioception; valid only with `joint_pos`)                                                                                     |
| task mode         | `policy` (single-task; the `joint` multi-task default is avoided)                                                                                  |
| resolution        | `480`                                                                                                                                              |
| viewpoint / video | `concat_view` / `video_mode=null`                                                                                                                  |
| chunk length      | `32` (tokenizer `encode_exact_durations=[33]`)                                                                                                     |
| sequence packing  | `max_num_tokens_after_packing=-1` (full vision sequence per step)                                                                                  |
| shuffle           | episode-shuffle stream (decorrelates the per-step global batch)                                                                                    |
| window filter     | [keep_ranges_1_0_1.json](https://huggingface.co/KarlP/droid/blob/main/keep_ranges_1_0_1.json) (`KarlP/droid`) — trains the curated ≈74% window set |
| lr                | `2e-4`                                                                                                                                             |
| global batch      | `8192` = `max_samples_per_batch` × world size × `grad_accum_iter` (reduce the first / raise the last to fit GPU memory)                            |
| eval              | disabled for the reproduction run                                                                                                                  |

## Full Reproduction

The OSS flow mirrors the other recipes (see [docs/training.md](./training.md)):

```shell
# Step 1: download Cosmos3-DROID into a directory named droid_plus_lerobot_640x360_20260412
# (the loader resolves the dataset schema from the directory name), then point
# DATASET_PATH at that parent directory — the one containing success/, not .../success.
hf download nvidia/Cosmos3-DROID --repo-type dataset \
  --local-dir /path/to/droid_plus_lerobot_640x360_20260412
export DATASET_PATH=/path/to/droid_plus_lerobot_640x360_20260412

# Step 2: convert the base checkpoint -> $BASE_CHECKPOINT_PATH
export BASE_CHECKPOINT_PATH=/path/to/base_checkpoint
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path Cosmos3-Nano \
  -o $BASE_CHECKPOINT_PATH

# Step 3: download the keep_ranges_1_0_1.json window filter (drops idle/non-task frames -> trains
# the curated ~74% window set, matching the released model).
export FILTER_DIR=/path/to/droid_filters
hf download KarlP/droid keep_ranges_1_0_1.json --local-dir $FILTER_DIR

# URI-keyed filters require their dataset prefix, supplied by you.
# Set DROID_FILTER_SOURCE_PREFIX as described below before continuing.
: "${DROID_FILTER_SOURCE_PREFIX:?set the filter URI prefix before the complete episode ID}"

# Step 4: launch. The TOML selects the experiment + scalars; the dataset/action
# knobs come from the registered experiment.
: "${BASE_CHECKPOINT_PATH:?set BASE_CHECKPOINT_PATH before Step 2}"
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export NPROC_PER_NODE=8
# Enable the keep_ranges_1_0_1.json filter via EXTRA_TAIL_OVERRIDES (space-separated Hydra
# overrides; an exported string survives `bash <wrapper>`). $FILTER_DIR is expanded HERE,
# at export time — it must already be set in this shell, or filter_dict_path silently
# becomes /keep_ranges_1_0_1.json and the dataset fails with FileNotFoundError.
: "${FILTER_DIR:?set FILTER_DIR (Step 3) before composing EXTRA_TAIL_OVERRIDES}"
export EXTRA_TAIL_OVERRIDES=" \
  dataloader_train.dataloader.datasets.droid.dataset.use_filter_dict=True \
  dataloader_train.dataloader.datasets.droid.dataset.filter_dict_path=$FILTER_DIR/keep_ranges_1_0_1.json \
"
bash examples/launch_sft_action_policy_droid_nano.sh
```

The recipe TOML ([`action_policy_droid_nano.toml`](../examples/toml/sft_config/action_policy_droid_nano.toml)) sets the scalar
knobs (`max_iter`, `save_iter`, `grad_clip`, parallelism, wandb); the dataset/action knobs
(`joint_pos`, `use_state`, `concat_view`, 480p, chunk 32, count-based batch) live in the
registered `action_policy_droid_nano` experiment per the schema's design. For multi-node HSDP,
set `model.config.parallelism.data_parallel_replicate_degree = <num_nodes>` (intra-node shard stays 8).

The **keep_ranges_1_0_1.json filter** maps each DROID trajectory key to a list of `[start, end]` frame
ranges; only windows whose start falls inside a kept range are trained on (episodes absent from
the dict are dropped). To train on the full window set instead, leave `EXTRA_TAIL_OVERRIDES` unset.

For URI-keyed filters, compare a key in your filter JSON with an `episode_id` in your dataset
metadata. Set `DROID_FILTER_SOURCE_PREFIX` to the URI prefix **before the complete episode ID**;
do not drop date or other directory components that belong to the ID. For example, if the key is
`s3://example-bucket/data/day-1/episode-1/recordings/MP4--s3://example-bucket/data/day-1/episode-1/trajectory.h5`
and the episode ID is `day-1/episode-1`, set:

```shell
export DROID_FILTER_SOURCE_PREFIX=s3://example-bucket/data
```

Replace that example with the prefix in your own file; it is not an address to download from.
Both the `examples/` launcher and the cookbook launcher forward this variable as
`dataloader_train.dataloader.datasets.droid.dataset.filter_dict_source_prefix`.
For a direct training command, pass that dataset override yourself. Filters keyed by relative
episode IDs need no prefix; omit the prefix check above for those files.

The cookbook launcher enables filtering by default. To disable it there, set
`EXTRA_TAIL_OVERRIDES="dataloader_train.dataloader.datasets.droid.dataset.use_filter_dict=False"`.

## Checkpoints

- Saved every `save_iter` iters (1000 in the validated run) to the object store, at
  `<bucket>/<project>/<group>/<job.name>/checkpoints/iter_<N>/`.
- The run is **resumable** from the latest checkpoint (re-launch with the same `job.name`).
- Export to HF safetensors via `cosmos_framework.scripts.export_model` (see [docs/training.md](./training.md)).
