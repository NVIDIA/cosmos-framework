# Native QLoRA (experimental)

Native QLoRA trains LoRA adapters on top of a frozen base whose language-model linear layers are
stored in 4-bit NF4 ([bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)).
Qwen3-VL uses the existing Posttrain SFT entrypoint; Alpamayo-R1 and GR00T N1.7 use the new
`action_sft` entrypoint. Both save adapter-only checkpoints that resume, evaluate and export with
the base model. The aim is to reduce training memory while keeping task quality close to BF16 LoRA.

Supported scope:

| Model                             | Entry point                        | Recipe                               | Adapted modules                                                                                    |
| --------------------------------- | ---------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------- |
| Qwen3-VL-8B (VLM SFT)             | `posttrain.entrypoints.sft`        | `configs/sft_qwen3_vl_8b_qlora.py`   | NF4 language model with LoRA on attention and MLP projections                                      |
| Alpamayo-R1-10B (trajectory flow) | `posttrain.entrypoints.action_sft` | `configs/sft_alpamayo_qlora.py`      | NF4 Qwen3-VL backbone with LoRA, plus the trajectory expert                                        |
| GR00T N1.7 (LIBERO flow BC)       | `posttrain.entrypoints.action_sft` | `configs/gr00t_n1d7_libero_qlora.py` | NF4 Cosmos-Reason2-2B backbone with LoRA, plus the action head or optional DiT-attention head LoRA |

Recipe paths are relative to `cosmos_framework/posttrain/`; entry points are modules under
`cosmos_framework.`. Each recipe runs on one GPU (`torchrun --nproc-per-node=1`). The recipes were
tested on an A100 80GB; other GPUs with compute capability 8.0 or newer (native BF16) are
accepted with a warning, and older GPUs such as V100 or T4 are refused. Multi-GPU sharding of the NF4 base is not supported, and
validation refuses it, along with any silent fallback to a dense base.

## Install

NF4 needs the optional `qlora` dependency group (bitsandbytes 0.49.2) of the Posttrain
environment. Run the commands in this guide from the repository root:

```bash
uv sync --project docker/posttrain/environment --frozen --python 3.12 \
    --no-default-groups --group sft --group cu128-runtime --group qlora
source docker/posttrain/environment/.venv/bin/activate
```

Alpamayo-R1 and GR00T N1.7 run in the separate
[native action runtime](../../docker/posttrain/native_action/README.md) instead (Transformers
4.57.6, with pinned Isaac-GR00T and Alpamayo-R1 sources and bitsandbytes). The installer needs
uv 0.11.29 and writes an activation script:

```bash
python docker/posttrain/native_action/install.py --uv /path/to/uv-0.11.29 --venv /path/to/venv
source /path/to/venv/native-action-env.sh
```

Every QLoRA entry point enables strict PyTorch determinism, which requires
`CUBLAS_WORKSPACE_CONFIG=:4096:8` in the launch environment.

## Qwen3-VL-8B SFT

`SFT_DATA_ROOT` must contain `train.jsonl` and `validation.jsonl` in the
[Posttrain SFT conversation format](../posttrain/guides/sft.md). Stage the pinned base model,
then train:

```bash
export SFT_MODEL_PATH=/path/to/base SFT_DATA_ROOT=/path/to/data
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=17
python -m cosmos_framework.posttrain.scripts.prepare_qlora_source --destination "$SFT_MODEL_PATH"

torchrun --standalone --nproc-per-node=1 -m cosmos_framework.posttrain.entrypoints.sft \
    --config cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_qlora.py -- \
    job.name=qwen3_vl_qlora model.model_config.adapter.rank=32 model.model_config.adapter.alpha=64 \
    model.model_config.activation_checkpointing=none \
    rl.max_steps=300 rl.optimizer.lr=1e-4 rl.micro_batch_size=1 rl.grad_accum=8
```

`configs/sft_qwen3_vl_8b_lora.py` is the BF16 LoRA recipe; it takes the same overrides.
Add `--validate-rows` to check the data contract without training.

Optional settings (all default off):

- `model.model_config.quantized_linear_execution=checkpointed_dequant`: dequantizes NF4 weights
  inside activation checkpointing. It requires `model.model_config.activation_checkpointing=full`
  and the default adapter initialization, and does not support excluded layers.
- `model.model_config.sft_logits_mode=supervised`: computes logits only on supervised tokens,
  which lowers peak memory for long sequences.
- `model.model_config.quantized_base.excluded_language_layers=[...]` keeps the listed language
  layers in BF16.
- `model.model_config.adapter.initialization=nf4_residual_svd`: initializes the adapters from
  the NF4 quantization residual (not supported with excluded layers). Prepare the factors once with
  `torchrun --standalone --nproc-per-node=1 -m cosmos_framework.posttrain.scripts.prepare_qlora_initialization --model "$SFT_MODEL_PATH" --output /path/to/factors --rank 32 --alpha 64 --seed 42`
  (they must match `adapter.rank`, `adapter.alpha` and `adapter.initialization_seed`, which defaults to 42), then set `model.model_config.adapter.initialization_artifact=/path/to/factors/manifest.json`
  (absolute path) and `model.model_config.adapter.initialization_artifact_sha256` to its SHA-256 (`sha256sum /path/to/factors/manifest.json`).
- `COSMOS_SFT_PROFILE_DIR` (with `COSMOS_SFT_PROFILE_{WAIT,WARMUP,ACTIVE}`) writes a bounded
  PyTorch profiler trace per rank.
- `COSMOS_QLORA_METRICS_FILE` appends one JSON line per training phase with host memory
  (RSS, peak RSS, PSS) and CUDA allocator statistics.

## Alpamayo-R1-10B trajectory flow

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=17 PYTORCH_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc-per-node=1 -m cosmos_framework.posttrain.entrypoints.action_sft \
    --config cosmos_framework/posttrain/configs/sft_alpamayo_qlora.py -- \
    model.checkpoint_path=/path/to/Alpamayo-R1-10B model.processor_path=/path/to/Qwen3-VL-8B-processor \
    data.train_path=/path/to/train.jsonl data.validation_path=/path/to/validation.jsonl \
    model.backbone_adaptation.mode=nf4_lora rl.max_steps=512 rl.evaluation_every=128 \
    '++rl.optimizer.lr_multipliers={expert:0.1,action_in_proj:0.1,action_out_proj:0.1}'
```

Set `model.backbone_adaptation.mode=bf16_lora` for BF16 LoRA. Evaluation reports
open-loop ADE and FDE at 6.4 s.

Each JSONL row is one clip. Media paths are relative to the JSONL file; the row must be valid
JSON without NaN or infinity. Converting PhysicalAI-AV clips to this format is up to you.

| Key                                 | Shape or type    | Constraint                                                                                                                                                                           |
| ----------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `id`, `episode_id`                  | string           | Non-empty; `id` unique. Training and validation share no `id`, episode or media.                                                                                                     |
| `actions`                           | 64 × 3 floats    | Future positions in the vehicle frame at t0, 0.1 s apart                                                                                                                             |
| `media`                             | camera → 4 paths | The four cameras below, four distinct frames each, oldest first                                                                                                                      |
| `observation.camera_order`          | list             | `camera_cross_left_120fov`, `camera_front_wide_120fov`, `camera_cross_right_120fov`, `camera_front_tele_30fov`                                                                       |
| `observation.camera_indices`        | list             | `[0, 1, 2, 6]`                                                                                                                                                                       |
| `observation.frame_timestamps_us`   | 4 × 4 ints       | Non-decreasing per camera; frame j at or before `t0_us + (j − 3) × 100000`                                                                                                           |
| `observation.t0_us`                 | int              | Timestamp of the current frame                                                                                                                                                       |
| `observation.coordinate_frame`      | string           | `"vehicle_at_t0"`                                                                                                                                                                    |
| `observation.trajectory_dt_seconds` | float            | `0.1`                                                                                                                                                                                |
| `observation.ego_history_xyz`       | 16 × 3           | Last row is the origin                                                                                                                                                               |
| `observation.ego_history_rot`       | 16 × 3 × 3       | Last entry is the identity                                                                                                                                                           |
| `observation.ego_future_rot`        | 64 × 3 × 3       | Finite                                                                                                                                                                               |
| `observation.calibration`           | object           | Per camera: `camera_intrinsics` (`width`, `height`, `cx`, `cy`, `bw_poly_0`–`4`, `fw_poly_0`–`4`) and `sensor_extrinsics` (unit quaternion `qx`, `qy`, `qz`, `qw` and `x`, `y`, `z`) |

## GR00T N1.7 LIBERO

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=17
torchrun --standalone --nproc-per-node=1 -m cosmos_framework.posttrain.entrypoints.action_sft \
    --config cosmos_framework/posttrain/configs/gr00t_n1d7_libero_qlora.py -- \
    model.checkpoint_path=/path/to/GR00T-N1.7-LIBERO/libero_spatial \
    model.backbone_adaptation.backbone_path=/path/to/Cosmos-Reason2-2B \
    data.train_path=/path/to/train.jsonl data.validation_path=/path/to/validation.jsonl \
    model.backbone_adaptation.mode=nf4_lora rl.max_steps=200
```

To train DiT-attention LoRA adapters instead of the full 1.6B-parameter action head, add
`model.backbone_adaptation.train_action_head=false` and
`model.backbone_adaptation.action_head_adapter_profile=gr00t_n1d7_dit_attention_lora_v1`.
Head adapters use the same rank, alpha and dropout as the backbone adapters. Most of the GR00T
memory saving comes from this option.

Each JSONL row is one training window:

```json
{"id": "s0-e3-t40", "episode_id": "s0-e3",
 "observation": {"state": [0.1, 0.0, 0.9, 3.1, 0.0, 0.0, 0.03, -0.03], "prompt": "pick up the black bowl"},
 "actions": [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], "... 10 to 16 rows of 7 LIBERO actions"],
 "media": {"image": "frames/s0-e3/40.png", "wrist_image": "frames/s0-e3/40_wrist.png"}}
```

`state` is the 8-value LIBERO robot state, the gripper action lies in [-1, 1], and media paths
are relative to the JSONL file. Converting LIBERO demonstrations is up to you. For both models,
`action_sft --validate-data` checks the files and that training and validation share no `id`,
episode or image without building a model.

Closed-loop LIBERO-Spatial rollouts run the policy against a separate LIBERO simulator
environment (built as in the [π0.5 LIBERO guide](../posttrain/guides/joint-ppo-pi05-public.md)):

```bash
echo '[{"suite": "libero_spatial", "task_id": 0, "init_state_id": 0}]' > /path/to/cases.json
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 -m cosmos_framework.posttrain.entrypoints.gr00t_rollout \
    --config cosmos_framework/posttrain/configs/gr00t_n1d7_libero_qlora.py \
    --checkpoint /path/to/checkpoint --sim-python /path/to/libero-venv/bin/python \
    --cases /path/to/cases.json --seed 17 --output /path/to/new/rollout-dir -- <training overrides>
```

Use `--base` instead of `--checkpoint` to roll out the untrained base (with a `*_frozen` mode),
and add `--dense` or `--unmerged` for an exported policy. The policy executes eight actions per
call: pass `model.execution_horizon=8` if training used another value.

## Checkpoints, evaluation and export

- Checkpoints contain only the trainable adapter (and head) tensors, plus metadata that binds the
  base model, quantization, adapter and training contract.
- Resume refuses a change to anything that alters the training trajectory: training data content,
  model, adapter, quantization, optimizer, LR schedule, seed, batching, processor, execution
  policy or software environment. Moving the validation file, changing the validation cadence, or
  raising `rl.max_steps` is allowed when the LR schedule does not depend on it (the recipes use a
  constant schedule or set `cycle_steps`).
- Reloading a checkpoint for evaluation or export checks only the model definition: the base
  model, adapter rank, alpha and targets, quantization, adapter compute dtype and the task or
  processor contract. Activation checkpointing, logits mode, compilation, TF32, hardware, library
  versions, seeds, dropout and the adapter initializer may differ.
- Digests of the frozen base are recorded when the model is built. The full base is re-hashed at
  dense export and at the end of action SFT; set `COSMOS_VERIFY_FROZEN_BASE_EVERY_CHECKPOINT=1` to
  re-hash it at every checkpoint as well.
- `COSMOS_TRAINING_CODE_REVISION` and `COSMOS_TRAINING_WHEEL_SHA256`, if set, are saved with
  checkpoints and exports as build information; they are never compared.
- `torchrun --standalone --nproc-per-node=1 -m cosmos_framework.posttrain.entrypoints.action_eval
  --config <recipe> --checkpoint <dir> --output <new dir> -- <same overrides as training>`
  evaluates a saved action checkpoint in a fresh process. Use `--base` instead of `--checkpoint`
  to score the untrained base (with a `*_frozen` mode), and add `--dense` or `--unmerged` to load
  an exported policy.
- `export_adapter_dense` (Qwen3-VL) writes the effective model, with adapters merged into the
  dequantized base, as a new dense Hugging Face checkpoint. `export_family_dense` (Alpamayo-R1,
  GR00T N1.7) writes a standalone policy with its task heads, either merged
  (`--merged-backbone-dtype`) or with the base and adapters as separate safetensors shards
  (`--unmerged`). Both run under `torchrun` with one process, write to a new directory, and take
  training overrides as repeated `--override key=value`:

```bash
torchrun --standalone --nproc-per-node=1 -m cosmos_framework.posttrain.checkpoint.tools.export_adapter_dense \
    --config cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_qlora.py \
    --checkpoint /path/to/checkpoint --output /path/to/new/export --output-dtype bfloat16 \
    --override model.model_config.adapter.rank=32 --override model.model_config.adapter.alpha=64
torchrun --standalone --nproc-per-node=1 -m cosmos_framework.posttrain.checkpoint.tools.export_family_dense \
    --config cosmos_framework/posttrain/configs/gr00t_n1d7_libero_qlora.py \
    --checkpoint /path/to/checkpoint --output /path/to/new/export --unmerged \
    --override model.checkpoint_path=/path/to/GR00T-N1.7-LIBERO/libero_spatial \
    --override model.backbone_adaptation.backbone_path=/path/to/Cosmos-Reason2-2B \
    --override model.backbone_adaptation.mode=nf4_lora
```

## Measured results

Each measurement compares NF4 LoRA with BF16 LoRA on the same recipe, data and seeds, on one A100
80GB. These are development measurements, not benchmark claims.

| Model                                            | Peak GPU memory, NF4 / BF16 | Time per update vs BF16 | Quality vs BF16 LoRA                                |
| ------------------------------------------------ | --------------------------- | ----------------------- | --------------------------------------------------- |
| Qwen3-VL-8B (video multiple-choice SFT, rank 32) | 10.0 / 19.5 GiB             | +5.8%                   | Validation loss within 1% (0.8% lower)              |
| Alpamayo-R1-10B (512 updates, 3 seeds)           | 67.1 / 76.2 GiB             | +2–3%                   | ADE +0.7%, FDE +0.6% (lower is better)              |
| GR00T N1.7 (DiT-attention head LoRA)             | 11.9 / 31.5 GiB\*           | about equal             | 390/400 vs 391/400 LIBERO-Spatial rollout successes |

\* The GR00T baseline is BF16 LoRA with the full action head trained. Most of the saving comes
from training head adapters instead of the full head. NF4 alone saves 1.1 GiB over the same
recipe in BF16.

Limitations:

- **Single GPU only:** the NF4 base is not sharded across GPUs.
- **Small speed cost:** NF4 is slightly slower per update than BF16 LoRA; its benefit is memory.
- **Limited quality evidence:** the results come from small development sets: Qwen3-VL one
  video multiple-choice task, Alpamayo-R1 three seeds on 256 clips (open-loop metrics only), and
  GR00T N1.7 one seed in simulation.
- **Recipe sensitivity:** in the Alpamayo and GR00T recipes above, neither NF4 nor BF16 LoRA
  clearly improved over the frozen base. QLoRA was within noise of LoRA in these runs, but these recipes are not tuned for
  task gains.
