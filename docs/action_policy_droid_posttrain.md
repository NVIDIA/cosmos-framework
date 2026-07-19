# Cosmos3 DROID Action-Policy Post-Training

This example post-trains a Cosmos3 base model into a DROID action policy on the [Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID) dataset. The model predicts absolute joint-position actions conditioned on proprioceptive state and video observations at a resolution of 480p (`640×360`). Two base-model tiers are covered:

- **Nano** — reproduces [Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID), post-trained from [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) (16B Mixture-of-Transformers). Experiment `action_policy_droid_nano`, launcher `launch_sft_action_policy_droid_nano.sh`, lr 2e-4.
- **Edge** — post-trained from [Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge) (4B Mixture-of-Transformers with a Nemotron-2B-Dense-VL reasoner). Experiment `action_policy_droid_edge`, launcher `launch_sft_action_policy_droid_edge.sh`, lr 5e-4 (linear decay to 0.4x), idle-frame prompt conditioning, and the generator qk-norm fix (`k_norm_und_for_gen`) kept training.

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

This package ships the training stack — the registered `action_policy_droid_nano` and
`action_policy_droid_edge` experiments, the DROID action dataset class with the recipe knobs
(`action_space=joint_pos`, `use_state`, `concat_view`), and the EMA warm-start in
`checkpoint/dcp.py`. Two inputs are external and must be provided per environment:

1. **[Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID) dataset (in LeRobotDataset v3.0 format)** — pre-download the
   dataset and point `DROID_ROOT` at the resulting `…/Cosmos3-DROID/success` directory (must
   contain `meta/info.json`).
2. **DCP base checkpoint** — convert the chosen base model
   ([Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) or
   [Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge)) to DCP and point
   `BASE_CHECKPOINT_PATH` at it (see [Full Reproduction](#full-reproduction)). Action heads are
   not loaded from it (they init fresh).

## Recipe

| knob              | value                                                                                                                                              |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| init              | `Cosmos3-Nano` or `Cosmos3-Edge` (public Hugging Face repos)                                                                                       |
| action space      | `joint_pos` (absolute joint position, 8-D incl. gripper)                                                                                           |
| state             | `use_state=true` (proprioception; valid only with `joint_pos`)                                                                                     |
| task mode         | `policy` (single-task; the `joint` multi-task default is avoided)                                                                                  |
| resolution        | `480`                                                                                                                                              |
| viewpoint / video | `concat_view` / `video_mode=null`                                                                                                                  |
| chunk length      | `32` (tokenizer `encode_exact_durations=[33]`)                                                                                                     |
| sequence packing  | `max_num_tokens_after_packing=-1` (full vision sequence per step)                                                                                  |
| shuffle           | episode-shuffle stream (decorrelates the per-step global batch)                                                                                    |
| window filter     | [keep_ranges_1_0_1.json](https://huggingface.co/KarlP/droid/blob/main/keep_ranges_1_0_1.json) (`KarlP/droid`) — trains the curated ≈74% window set |
| lr                | `2e-4` (Nano); `5e-4`, linear decay to 0.4x (Edge)                                                                                                 |
| global batch      | `8192` = `max_samples_per_batch` × world size × `grad_accum_iter` (reduce the first / raise the last to fit GPU memory)                            |
| eval              | disabled for the reproduction run                                                                                                                  |

## Full Reproduction

The OSS flow mirrors the other recipes (see [docs/training.md](./training.md)):

```shell
# Step 1: prepare Cosmos3-DROID success split -> $DATASET_PATH (see "Inputs You Provide")

# Step 2: convert the base checkpoint -> $BASE_CHECKPOINT_PATH
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path Cosmos3-Nano \
  -o $BASE_CHECKPOINT_PATH
# (Edge: --checkpoint-path Cosmos3-Edge)

# Step 3: download the keep_ranges_1_0_1.json window filter (drops idle/non-task frames -> trains
# the curated ~74% window set, matching the released model).
hf download KarlP/droid keep_ranges_1_0_1.json --local-dir $FILTER_DIR

# Step 4: launch. The TOML selects the experiment + scalars; the dataset/action
# knobs come from the registered experiment.
export DATASET_PATH=/path/to/dataset/success
export BASE_CHECKPOINT_PATH=/path/to/base_checkpoint
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export NPROC_PER_NODE=8
# Enable the keep_ranges_1_0_1.json filter via EXTRA_TAIL_OVERRIDES (space-separated Hydra
# overrides; an exported string survives `bash <wrapper>`).
export EXTRA_TAIL_OVERRIDES=" \
  dataloader_train.dataloader.datasets.droid.dataset.use_filter_dict=True \
  dataloader_train.dataloader.datasets.droid.dataset.filter_dict_path=$FILTER_DIR/keep_ranges_1_0_1.json \
"
bash examples/launch_sft_action_policy_droid_nano.sh
# (Edge: bash examples/launch_sft_action_policy_droid_edge.sh)
```

The recipe TOMLs ([`action_policy_droid_nano.toml`](../examples/toml/sft_config/action_policy_droid_nano.toml),
[`action_policy_droid_edge.toml`](../examples/toml/sft_config/action_policy_droid_edge.toml)) set the scalar
knobs (`max_iter`, `save_iter`, `grad_clip`, parallelism, wandb); the dataset/action knobs
(`joint_pos`, `use_state`, `concat_view`, 480p, chunk 32, count-based batch) live in the
registered experiments per the schema's design. For multi-node HSDP,
set `model.config.parallelism.data_parallel_replicate_degree = <num_nodes>` (intra-node shard stays 8).

The **keep_ranges_1_0_1.json filter** maps each DROID trajectory key to a list of `[start, end]` frame
ranges; only windows whose start falls inside a kept range are trained on (episodes absent from
the dict are dropped). To train on the full window set instead, leave `EXTRA_TAIL_OVERRIDES` unset.

## Checkpoints

- Saved every `save_iter` iters (1000 in the validated run) to the object store, at
  `<bucket>/<project>/<group>/<job.name>/checkpoints/iter_<N>/`.
- The run is **resumable** from the latest checkpoint (re-launch with the same `job.name`).
- Export to HF safetensors via `cosmos_framework.scripts.export_model` (see [docs/training.md](./training.md)).
