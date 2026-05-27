# Post-Training (Supervised Fine-Tuning)

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Step 1 - Prepare data and config](#step-1---prepare-data-and-config)
- [Step 2 — Prepare checkpoint](#step-2--prepare-checkpoint)
- [Step 3 — Run training](#step-3--run-training)
  - [Generator Post-Training](#generator-post-training)
    - [Option A (recommended): the paired launch shell](#option-a-recommended-the-paired-launch-shell)
      - [Overriding the defaults](#overriding-the-defaults)
    - [Option B: raw `torchrun`](#option-b-raw-torchrun)
- [Outputs](#outputs)
- [Export checkpoint to Hugging Face safetensors](#export-checkpoint-to-hugging-face-safetensors)
- [Evaluation](#evaluation)
- [Config](#config)
  - [Common Hydra tail overrides](#common-hydra-tail-overrides)

______________________________________________________________________

<!--TOC-->

Fine-tune a pre-trained Cosmos3 model on your own dataset using supervised fine-tuning (SFT). Tested on 8× H100 (80 GB).

Prerequisites:

- [Setup](../README.md#setup)
- [Environment Variables](./environment_variables.md)

The runnable artifacts (TOML recipes, paired launch shells, inference helpers) live in [`examples/`](../examples/README.md).

## Step 1 - Prepare data and config

Some datasets are license gated — visit the repository page and accept any terms, and authenticate with `uvx hf@latest auth login` (or set `HF_TOKEN`).

The per-recipe download commands below write to `examples/data/<dataset>/` and `examples/checkpoints/wan22_vae/Wan2.2_VAE.pth`, which match the launcher's default `$DATASET_PATH` and `$WAN_VAE_PATH`. See [Step 3 → Option A](#option-a-recommended-the-paired-launch-shell) for how to override these defaults if you'd rather keep data on a different filesystem.

Select one of the following recipes:

<details open><summary><b>Action Forward Dynamics with Bridge LeRobot v3 (Cosmos3-Nano)</b></summary>

Fine-tune action forward dynamics on [nvidia/bridge_lerobot_v3](https://huggingface.co/datasets/nvidia/bridge_lerobot_v3).

Launch shell: `examples/launch_sft_action_fdm_nano.sh`

```shell
BASE_CHECKPOINT_NAME=Cosmos3-Nano

# Defaults match the launcher (see Step 3 → Option A to override).
uvx hf@latest download --repo-type dataset nvidia/bridge_lerobot_v3 \
    --local-dir examples/data/bridge_lerobot_v3 --quiet
uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
    --local-dir examples/checkpoints/wan22_vae --quiet
```

</details>

<details><summary><b>Action Forward Dynamics LoRA (Cosmos3-Super)</b></summary>

LoRA SFT for action forward dynamics on Qwen3-VL-32B MoT (Cosmos3-Super), on the same Bridge dataset as **Action Forward Dynamics with Bridge LeRobot v3** above. Step 2 must convert the Cosmos3-Super checkpoint, not Cosmos3-Nano.

Launch shell: `examples/launch_sft_action_fdm_super.sh`

```shell
BASE_CHECKPOINT_NAME=Cosmos3-Super

# Defaults match the launcher (see Step 3 → Option A to override).
uvx hf@latest download --repo-type dataset nvidia/bridge_lerobot_v3 \
    --local-dir examples/data/bridge_lerobot_v3 --quiet
uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
    --local-dir examples/checkpoints/wan22_vae --quiet
```

</details>

<details><summary><b>Action Policy with LIBERO LeRobot Dataset (Cosmos3-Nano)</b></summary>

Fine-tune action policy on [nvidia/LIBERO_LeRobot_v3](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3/) routed through `DataPackerDataLoader` (token-budget packing, `max_batch_size=256`). Trains in policy mode with concatenated `agentview` and wrist camera observations, frame-wise relative actions, 6D rotations, and quantile-rotation action normalization. The launcher expects `$DATASET_PATH` to contain the `libero_10/`, `libero_object/`, `libero_spatial/`, `libero_goal/` short-name subdirs directly — the HF snapshot already ships in that shape, so no symlinking is required.

Launch shell: `examples/launch_sft_action_policy_nano.sh`

```shell
BASE_CHECKPOINT_NAME=Cosmos3-Nano

# Defaults match the launcher (see Step 3 → Option A to override).
uvx hf@latest download --repo-type dataset nvidia/LIBERO_LeRobot_v3 \
    --revision ddc1edeb6e51e2b7d4d2ba7a1433daaecd37aa64 \
    --local-dir examples/data/LIBERO_LeRobot_v3 --quiet
uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
    --local-dir examples/checkpoints/wan22_vae --quiet
```

For evaluation, see [LIBERO Closed-Loop Evaluation](./action_policy_closed_loop_eval.md).

</details>

<details><summary><b>Action Policy LoRA (Cosmos3-Super)</b></summary>

LoRA SFT for action policy on Qwen3-VL-32B MoT (Cosmos3-Super), on the same LIBERO dataset as **Action Policy with LIBERO LeRobot Dataset** above. Same `$DATASET_PATH` layout requirement (`libero_10/`, `libero_object/`, `libero_spatial/`, `libero_goal/` subdirs). Step 2 must convert the Cosmos3-Super checkpoint, not Cosmos3-Nano.

Launch shell: `examples/launch_sft_action_policy_super.sh`

```shell
BASE_CHECKPOINT_NAME=Cosmos3-Super

# Defaults match the launcher (see Step 3 → Option A to override).
uvx hf@latest download --repo-type dataset nvidia/LIBERO_LeRobot_v3 \
    --revision ddc1edeb6e51e2b7d4d2ba7a1433daaecd37aa64 \
    --local-dir examples/data/LIBERO_LeRobot_v3 --quiet
uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
    --local-dir examples/checkpoints/wan22_vae --quiet
```

For evaluation, see [LIBERO Closed-Loop Evaluation](./action_policy_closed_loop_eval.md).

</details>

<details><summary><b>Vision SFT (Cosmos3-Nano)</b></summary>

T2V/I2V/V2V SFT on [nvidia/bridge-v2-subset-synthetic-captions](https://huggingface.co/datasets/nvidia/bridge-v2-subset-synthetic-captions/tree/main). `$DATASET_PATH` should be the directory containing `train/video_dataset_file.jsonl`.

Launch shell: `examples/launch_sft_vision_nano.sh`

```shell
BASE_CHECKPOINT_NAME=Cosmos3-Nano

# Defaults match the launcher (see Step 3 → Option A to override).
uvx hf@latest download --repo-type dataset nvidia/bridge-v2-subset-synthetic-captions \
    --revision 46468e12ac0dd36901e9e3240d4fc7620942b5d7 \
    --local-dir examples/data/bridge-v2-subset-synthetic-captions --quiet
uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
    --local-dir examples/checkpoints/wan22_vae --quiet
```

</details>

<details><summary><b>Vision SFT LoRA (Cosmos3-Super)</b></summary>

LoRA SFT on Qwen3-VL-32B MoT (Cosmos3-Super), on the same Bridge dataset as **Vision SFT (Cosmos3-Nano)**. Step 2 must convert the Cosmos3-Super checkpoint, not Cosmos3-Nano.

Launch shell: `examples/launch_sft_vision_super.sh`

```shell
BASE_CHECKPOINT_NAME=Cosmos3-Super

# Defaults match the launcher (see Step 3 → Option A to override).
uvx hf@latest download --repo-type dataset nvidia/bridge-v2-subset-synthetic-captions \
    --revision 46468e12ac0dd36901e9e3240d4fc7620942b5d7 \
    --local-dir examples/data/bridge-v2-subset-synthetic-captions --quiet
uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
    --local-dir examples/checkpoints/wan22_vae --quiet
```

</details>

<details><summary><b>Reasoner Alignment SFT with LLaVA-OneVision (vfm-vlm)</b></summary>

Alignment SFT for the Reasoner variant on the [lmms-lab/LLaVA-OneVision-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data) dataset (streamed from HF Hub). Skips Step 2: the backbone is `Qwen/Qwen3-VL-8B-Instruct` (set by the parent experiment's `vlm_policy=qwen3_vl_8b_instruct` default) and is fetched from the HF Hub by the model downloader at startup — no DCP conversion needed and no env-var plumbing required.

Launch shell: `examples/launch_sft_llava_ov.sh`

```shell
# No required env vars. The first launch will populate the HF Hub cache under
# $HF_HOME (defaults to /tmp/hf_cache inside the wrapper); subsequent launches
# reuse the cached snapshot.
#
# (optional) HF_TOKEN raises HF Hub rate limits for the streamed dataset
# revision lookup — useful if you're running 8-rank fan-out from a single IP:
# export HF_TOKEN=hf_...
```

</details>

## Step 2 — Prepare checkpoint

Convert the base checkpoint to [PyTorch Distributed Checkpoint (DCP)](https://pytorch.org/docs/stable/distributed.checkpoint.html) format. `cosmos_framework.scripts.convert_model_to_dcp` ships in the unified `cosmos_framework/` package, so this step runs from the repo root (with the environment activated per [Setup](./setup.md)).

Set `BASE_CHECKPOINT_NAME` to the value from the recipe block you picked in Step 1 (`Cosmos3-Nano` or `Cosmos3-Super`):

```shell
BASE_CHECKPOINT_NAME=Cosmos3-Nano   # or Cosmos3-Super — match the recipe in Step 1

# Default output dir matches the launcher (see Step 3 → Option A to override).
python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o examples/checkpoints/$BASE_CHECKPOINT_NAME \
  --checkpoint-path $BASE_CHECKPOINT_NAME
```

`$BASE_CHECKPOINT_NAME` (e.g. `Cosmos3-Nano`, `Cosmos3-Super`) is a registered name in the checkpoint catalog; the converter downloads the matching repo from the Hugging Face Hub and writes the DCP into `examples/checkpoints/$BASE_CHECKPOINT_NAME`.

**Reasoner recipe:** Skip this step — the Reasoner alignment SFT loads `Qwen/Qwen3-VL-8B-Instruct` from the HF Hub at startup (no DCP conversion, no env vars).

## Step 3 — Run training

**Weights & Biases (optional):** every recipe TOML defaults to `job.wandb_mode = "disabled"`. To log a run to W&B, flip that field to `"online"` in the TOML and export `WANDB_API_KEY` in your environment before launching.

### Generator Post-Training

#### Option A (recommended): the paired launch shell

Each recipe ships as a `examples/toml/sft_config/<recipe>.toml` (validated against the pydantic schema at [`cosmos_framework/configs/toml_config/sft_config.py`](../cosmos_framework/configs/toml_config/sft_config.py)) paired with `examples/launch_sft_<recipe>.sh`; the full catalog is indexed in [`examples/README.md`](../examples/README.md). Each `.sh` sources [`examples/_sft_launcher_common.sh`](../examples/_sft_launcher_common.sh) and forwards into `cosmos_framework.scripts.train --sft-toml=<recipe-toml>`. From the repo root, run the launch shell paired with the recipe you set up in Step 1. The wrapper resolves `DATASET_PATH`, `BASE_CHECKPOINT_PATH`, and `WAN_VAE_PATH` from the default locations under `examples/` (populated by Step 1 + Step 2), so no env-var setup is required (see [below](#overriding-the-defaults) to override):

```shell
# from the repo root, after Step 1 + Step 2:
bash examples/launch_sft_vision_nano.sh
```

Each launcher's default paths come from the `DATASET_PATH` + `BASE_CHECKPOINT_PATH` defaults declared at the top of its `.sh` (each uses `: "${VAR:=…}"` so any value you `export` in the shell before launching wins over the default):

| Launch shell                          | Default $DATASET_PATH (under examples/data/)             | Default $BASE_CHECKPOINT_PATH (under examples/checkpoints/) |
| ------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------- |
| `launch_sft_action_fdm_nano.sh`       | `bridge_lerobot_v3`                                      | `Cosmos3-Nano`                                              |
| `launch_sft_action_fdm_super.sh`      | `bridge_lerobot_v3`                                      | `Cosmos3-Super`                                             |
| `launch_sft_action_policy_nano.sh`    | `LIBERO_LeRobot_v3`                                      | `Cosmos3-Nano`                                              |
| `launch_sft_action_policy_super.sh`   | `LIBERO_LeRobot_v3`                                      | `Cosmos3-Super`                                             |
| `launch_sft_vision_nano.sh`           | `bridge-v2-subset-synthetic-captions/sft_dataset_bridge` | `Cosmos3-Nano`                                              |
| `launch_sft_vision_super.sh`          | `bridge-v2-subset-synthetic-captions/sft_dataset_bridge` | `Cosmos3-Super`                                             |
| `launch_sft_llava_ov.sh`              | (none; dataset streams from HF Hub)                      | (none; backbone fetched at startup)                         |

`WAN_VAE_PATH` defaults to `examples/checkpoints/wan22_vae/Wan2.2_VAE.pth` for every non-reasoner recipe.

##### Overriding the defaults

If you'd rather put data or checkpoints on a different filesystem (e.g. a faster SSD or shared mount), download to your chosen path in Step 1 / convert the DCP to your chosen path in Step 2, then export the matching env var(s) before launching:

```shell
# Example: data on /scratch, base DCP on /nfs/ckpts.
export DATASET_PATH=/scratch/bridge-v2-subset-synthetic-captions/sft_dataset_bridge
export BASE_CHECKPOINT_PATH=/nfs/ckpts/Cosmos3-Nano
export WAN_VAE_PATH=/nfs/ckpts/wan22_vae/Wan2.2_VAE.pth
bash examples/launch_sft_vision_nano.sh
```

Each env var falls back to its default if unset, so you only need to export the ones you're moving. The downloads / `convert_model_to_dcp` commands in Step 1 + Step 2 just need their `--local-dir` / `-o` argument pointed at the same path you export here. `.gitignore` excludes `examples/data/` and `examples/checkpoints/` so the multi-GB downloads aren't tracked when you keep the defaults.

#### Option B: raw `torchrun`

If you'd rather not use the paired launch shell, invoke `torchrun` directly with the recipe's TOML. Unlike Option A, **raw `torchrun` does not auto-resolve `DATASET_PATH` / `BASE_CHECKPOINT_PATH` / `WAN_VAE_PATH` from `examples/`** — they have to come from your shell:

- `BASE_CHECKPOINT_PATH` and `WAN_VAE_PATH` are read via `${oc.env:BASE_CHECKPOINT_PATH}` / `${oc.env:WAN_VAE_PATH}` at the TOML's `[checkpoint].load_path` / `[model.tokenizer].vae_path` keys.
- `DATASET_PATH` is read via `${oc.env:DATASET_PATH}` inside the experiment-SKU Python (e.g. `cosmos_framework/configs/base/experiment/sft/<recipe>.py`), not in the TOML.

You have two options to fill them in (pick either, not both):

1. **Export them in the shell before `torchrun`** (whether they point at the default `examples/...` paths from Step 1+2 or your own overrides) — shown below.
2. **Edit the TOML by hand** — open `examples/toml/sft_config/<recipe>.toml` and replace the `${oc.env:BASE_CHECKPOINT_PATH}` / `${oc.env:WAN_VAE_PATH}` placeholders with literal paths. Useful if you want a self-contained TOML you can hand to a colleague or commit alongside an experiment record. (Hand-editing won't help for `DATASET_PATH` — that's resolved out of the experiment Python, so you must still export it.)

Run from the repo root (the directory containing `pyproject.toml` and `examples/`); the snippet uses `$PWD` to absolutize the relative paths.

```shell
# This example uses the vision_sft_nano recipe end-to-end (same recipe as
# Option A). To switch recipes, swap TOML_FILE + DATASET_PATH per the table in
# Option A, and Cosmos3-Nano → Cosmos3-Super on the LoRA / super recipes.
TOML_FILE="examples/toml/sft_config/vision_sft_nano.toml"

# Match the launcher's defaults — or substitute your own paths.
export DATASET_PATH="$PWD/examples/data/bridge-v2-subset-synthetic-captions/sft_dataset_bridge"
export BASE_CHECKPOINT_PATH="$PWD/examples/checkpoints/Cosmos3-Nano"
export WAN_VAE_PATH="$PWD/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth"

IMAGINAIRE_OUTPUT_ROOT=outputs/train PYTHONPATH=. \
torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \
    --sft-toml=$TOML_FILE
```

To resume from the latest in-progress checkpoint, point `BASE_CHECKPOINT_PATH` at the run's `checkpoints/iter_<N>/` directory under `$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/` (see [Outputs](#outputs) below for the full layout).

## Outputs

Outputs land under `$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/`:

1. `config.yaml`, `config.pkl`: Finalized resolved config (YAML for inspection, pickle for re-instantiation).
1. `launch_info.yaml`, `job_env.yaml`: Job metadata and captured launch environment.
1. `checkpoints/`:
    1. `latest_checkpoint.txt`: Pointer file containing the latest checkpoint directory name (e.g. `iter_000000200`).
    1. `iter_<iter>/`: DCP checkpoint saved every `[train.ckpt].save_freq` iterations (zero-padded 9-digit, e.g. `iter_000000200/`):
        1. `model/`: model weights (sharded `.distcp`).
        1. `optim/`: optimizer state.
        1. `scheduler/`: LR scheduler state.
        1. `trainer/`: training state — includes the `iteration` counter and per-rank `rng_state_<i>` (numpy + random + torch + torch_cuda).
        1. `dataloader/`: optional per-rank pickle shards (`rank_<i>.pkl`) — only present for dataloaders that implement `has_state()`.
1. `<callback_name>/`: Callback outputs, one directory per registered callback (e.g. `DeviceMonitor/`, `EveryNDrawSample/`, `norm_monitor/`).
1. `wandb/`, `wandb_id.txt`: Wandb run files — only present when `[job].wandb_mode` is `online` or `offline`.

The shorthand `$RUN_DIR` used in the rest of this page refers to `$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>`. For example, with `IMAGINAIRE_OUTPUT_ROOT=outputs/train` and the `vision_sft_nano` recipe, `$RUN_DIR` is `outputs/train/cosmos3/sft/vision_sft_nano`.

## Export checkpoint to Hugging Face safetensors

Export the DCP checkpoint produced in Step 3 to a Hugging Face safetensors checkpoint:

```shell
RUN_DIR=$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>

CHECKPOINT_ITER=$(cat $RUN_DIR/checkpoints/latest_checkpoint.txt)
CHECKPOINT_PATH=$RUN_DIR/checkpoints/$CHECKPOINT_ITER

python -m cosmos_framework.scripts.export_model \
  --checkpoint-path $CHECKPOINT_PATH \
  --config-file $RUN_DIR/config.yaml \
  -o $RUN_DIR/model
```

The exported safetensors land at `$RUN_DIR/model` and can be used in [Inference](../README.md#inference) commands by passing `--checkpoint-path $RUN_DIR/model`.

## Evaluation

**Supported modalities:** Forward Dynamics, Inverse Dynamics, Policy.

Run inference on held-out dataset split and compare to ground truth:

```shell
torchrun --nproc-per-node=8 -m cosmos_framework.scripts.eval \
    -o outputs/train_eval \
    --checkpoint-path outputs/train/model \
    --dataset.config-file outputs/train/config.yaml
```

Arguments:

- `--dataset.model-mode`: Which modality to evaluate.
- `--dataset.num-samples N`: Maximum number of samples to evaluate.

Outputs:

- `metrics_aggregate.json`: Aggregate metrics.
- `<dataset>/<mode>/<id>/metrics.json`: Per sample metrics.

Metrics:

- Vision: Peak Signal-to-Noise Ratio (PSNR)
- Action: Mean Squared Error (MSE)

## Config

The recipe TOML is parsed against the pydantic schema [`SFTExperimentConfig`](../cosmos_framework/configs/toml_config/sft_config.py) at load time. Every top-level key listed below maps to a sub-model in that file; unknown keys raise a `ValidationError` before training starts (`extra="forbid"` on every sub-model). Values may use OmegaConf env interpolation `${oc.env:NAME}` — the recipe TOMLs use this for `BASE_CHECKPOINT_PATH` (`[checkpoint].load_path`) and `WAN_VAE_PATH` (`[model.tokenizer].vae_path`). `DATASET_PATH` is consumed the same way but inside the experiment-SKU Python (`cosmos_framework/configs/base/experiment/sft/<recipe>.py`), not in the TOML.

For the full field-by-field reference (every section, every default, every VFM/VLM applicability note, the `"???"` MISSING sentinel, env interpolation, the VFM↔VLM path-remap table, and how to extend the schema), see [SFT Structured-TOML Config Reference](./sft_config.md).

The commonly tuned knobs:

1. `[job]`
    1. `task` — `"vfm"` (generator recipes) or `"vlm"` (Reasoner alignment). Picks the base config: `cosmos_framework/configs/base/config.py` vs `…/vlm/config.py`. Also drives `PATH_REMAPS` in `toml_config_helper.py`.
    1. `experiment` — Registered experiment SKU name (e.g. `action_fdm_sft_nano`). Each SKU is a Python file under `cosmos_framework/configs/base/experiment/sft/` that wires up dataloader, model variant, and recipe-specific defaults.
    1. `project`, `group`, `name` — Components of the run output dir `$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/`. Also flow to W&B as the project / group / run name.
    1. `wandb_mode` — `"online"` (logs to W&B; `WANDB_API_KEY` must be set), `"offline"` (logs locally, sync later with `wandb sync`), or `"disabled"`.
1. `[model]`
    1. `max_num_tokens_after_packing` — VFM token-packing target. `-1` disables the cap. VFM only; VLM uses `data_setting.max_tokens` (tail override).
    1. `joint_attn_implementation` — VFM attention layout: `"two_way"` / `"three_way"` (NATTEN) / `"flex"`.
    1. `attn_implementation` — VLM attention impl: `"cosmos"` / `"flash_attention_2"` / `"sdpa"` / `"eager"`. VLM only.
    1. `lora_enabled`, `lora_rank`, `lora_alpha`, `lora_target_modules` — LoRA adapter knobs for the generation pathway. Used by SUPER-tier recipes; NANO-tier leaves `lora_enabled=false`. VFM only.
1. `[model.ema]`
    1. `enabled`, `rate`, `iteration_shift` — Exponential moving average of generation-pathway weights. Full fine-tunes typically enable it; LoRA recipes leave it off.
1. `[model.parallelism]`
    1. `data_parallel_shard_degree` — FSDP shard degree. `data_parallel_shard_degree × data_parallel_replicate_degree × context_parallel_shard_degree` must equal `WORLD_SIZE`. `-1` autoselects from torchrun world size.
    1. `data_parallel_replicate_degree` — HSDP replicate degree (outer replicate loop over the shard topology).
    1. `context_parallel_shard_degree` — Context-parallel shard degree. `>1` splits the sequence dim across ranks (used by super-tier configs: DP=4, CP=2 → 8 GPUs).
    1. `cfg_parallel_shard_degree` — Classifier-free-guidance shard degree. Almost always `1` for SFT.
    1. `use_torch_compile`, `compile_dynamic` — Toggle `torch.compile`. `compile_dynamic=True` recompiles per shape (required for `compile_tokenizer` callback's progressive warmup).
    1. `precision` — Compute dtype for forward/backward: `"bfloat16"` / `"float16"` / `"float32"`. Master weights stay fp32 separately.
1. `[model.activation_checkpointing]`
    1. `mode` — `"none"` / `"selective"` (per-op SAC, MoT-only) / `"full"` (per-block checkpointing).
    1. `save_ops_regex` — Regex patterns for ops to keep saved under `mode="selective"`.
    1. `preserve_rng_state`, `determinism_check` — Recompute determinism plumbing.
1. `[model.tokenizer]`
    1. `vae_path` — Wan2.2 VAE `.pth` path. Recipe TOMLs use `"${oc.env:WAN_VAE_PATH}"`. VFM only.
1. `[optimizer]`
    1. `lr` — Base learning rate.
    1. `betas`, `eps`, `fused`, `weight_decay` — Standard AdamW knobs. `eps` is VFM-only.
    1. `keys_to_select` — Substring allowlist for trainable params. Empty list = train everything; `["lora_"]` = adapter-only fine-tune.
1. `[optimizer.lr_multipliers]`
    1. Inline table of `<substring> = <multiplier>` pairs (e.g. `action_modality_embed = 5.0`).
1. `[scheduler]`
    1. `cycle_lengths`, `warm_up_steps` — Cycle length and warmup duration (lists, one entry per cycle), in optimizer steps.
    1. `f_max`, `f_min`, `f_start` — LR multipliers at peak / trough / step-0 (ratios of `optimizer.lr`).
    1. `verbosity_interval` — Scheduler-side LR log frequency. VFM only.
1. `[trainer]`
    1. `max_iter` — Total optimizer steps.
    1. `grad_accum_iter` — Micro-batches per optimizer step. Effective global batch = `grad_accum_iter × per-rank batch × world_size`.
    1. `logging_iter` — Console / W&B scalar log frequency.
    1. `distributed_parallelism` — `"fsdp"` is the only supported value.
1. `[trainer.callbacks.compile_tokenizer]`
    1. `enabled`, `compile_after_iterations`, `warmup_resolutions` — Lazy `torch.compile` of the VAE tokenizer. VFM only.
1. `[trainer.callbacks.grad_clip]`
    1. `clip_norm` — Max global L2 norm of the gradient (steps with larger norm are rescaled).
    1. `force_finite` — Replace NaN/Inf grads with zero (default `true` on VFM, `false` on VLM).
1. `[checkpoint]`
    1. `load_path` — Base DCP checkpoint directory to resume from (Step 2 output, or a prior run's `checkpoints/iter_<N>/`). Recipe TOMLs use `"${oc.env:BASE_CHECKPOINT_PATH}"`.
    1. `save_iter` — Save a new DCP checkpoint every N optimizer steps.
    1. `keys_to_skip_loading` — Substring blocklist applied at load time. Used to mask EMA / LoRA / action tensors when warm-starting from a checkpoint that doesn't have them yet.
1. `[dataloader_train]` — Top-level scalars only; the dataloader's class (LazyCall) and pipeline wiring (datasets, packers, …) stay in the experiment Python.
    1. `max_samples_per_batch` — Per-micro-batch sample cap (remapped to `max_batch_size` on the VLM packer). `null` / omitted = no per-count cap.
    1. `max_sequence_length` — Per-packed-sequence token cap (remapped to `max_tokens` on the VLM packer).
    1. `seed` — Dataloader RNG seed (VFM only).

### Common Hydra tail overrides

These knobs aren't part of the pydantic schema today; pass them as trailing `key.path=value` positionals after `--` (the `cosmos_framework.scripts.train` flow forwards them through OmegaConf):

- `dataloader_train.dataloaders.action_data.dataloader.batch_size` — Inner action_data sub-loader prefetch batch (per-worker memory knob, **not** the effective training batch). Used by `launch_sft_action_*_super.sh`.
- `model.config.policy.backbone.model_name` — VLM backbone HF identifier (e.g. `Qwen/Qwen3-VL-8B-Instruct`). Used by `launch_sft_llava_ov.sh`.
- `data_setting.max_tokens` — VLM token-packing cap (the VLM analogue of `[model].max_num_tokens_after_packing`). Used by `launch_sft_llava_ov.sh`.

The launchers wire these via `TAIL_OVERRIDES=(…)`; the helper appends `-- "${TAIL_OVERRIDES[@]}"` after the `--sft-toml=` argument.
