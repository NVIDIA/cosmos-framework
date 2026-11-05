# Run native Framework Stage 0–3 with a local Transfer bundle

Use Framework's native recipes and local loader. The argument helper binds local
assets and checkpoint parents to `configs/simulation/local.py`. The adapter
preserves Stage 0's 480/720 streams and applies each stage's transform overrides,
so the same bundle supports all four stages.

## Prepare local assets

Complete [Framework setup](../../shared/setup.md), including native
libraries and the CUDA toolkit, on every compute node. Export `FRAMEWORK_DIR`
and `SIM_TRANSFER_SKILL` as absolute locations. All model, data and output paths
must resolve identically on every rank.
For the example run, [download example2](../../shared/hf-sample.md) and
[prepare its native training bundle](../../shared/local-dataset.md) to set
`SIM_TRANSFER_TRAINING_BUNDLE`.
Follow [Nano conversion](nano-conversion.md) to download the model and video
VAE from Hugging Face and set `NANO_HF_DIR`, `VIDEO_VAE_PATH` and
`BASE_CKPT_ROOT`. Set `IMAGINAIRE_OUTPUT_ROOT` to a writable output directory.
Run Bash from the Framework root:

```bash
set -euo pipefail
export FRAMEWORK_DIR="${FRAMEWORK_DIR:?Set the absolute Framework checkout root}"
: "${SIM_TRANSFER_SKILL:?Set the absolute skill bundle location}"
cd "$FRAMEWORK_DIR"
source .venv/bin/activate
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1 COSMOS_SMOKE=0
export COSMOS_DEVICE=cuda WANDB_MODE=disabled
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export TRANSFER_BUNDLE_PATH="${SIM_TRANSFER_TRAINING_BUNDLE:?Prepare the example2 training bundle first}"
export NANO_HF_DIR="${NANO_HF_DIR:?Complete the HF Nano download and conversion first}"
export VIDEO_VAE_PATH="${VIDEO_VAE_PATH:?Complete the HF video VAE download first}"
export AUDIO_VAE_PATH="$NANO_HF_DIR/sound_tokenizer/avae_48k_noncausal_25hz_64ch.ckpt"
export AUDIO_VAE_CONFIG_PATH="$NANO_HF_DIR/sound_tokenizer/avae_48k_noncausal_25hz_64ch.json"
export BASE_CKPT_ROOT="${BASE_CKPT_ROOT:?Set the converted base DCP root}"
export TRANSFER_QA_BASE_MODEL_ONLY=1
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:?Set the writable output directory shared by all ranks}"
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"
RUN_ID="local_example2_$(date -u +%Y%m%dT%H%M%SZ)"
STAGE_HELPER="$SIM_TRANSFER_SKILL/post-training/scripts/qa_transfer_stage.py"
ARGS_FILE="$IMAGINAIRE_OUTPUT_ROOT/${RUN_ID}.args"
```

`NANO_HF_DIR/text_tokenizer/` supplies the text tokenizer. Pinned
[Nano conversion](nano-conversion.md) derives the audio assets under
`sound_tokenizer/`. `BASE_CKPT_ROOT` contains the native DCP `model/` directory.
`TRANSFER_BUNDLE_PATH` contains the `manifest.json` and `samples.json` created
from example2 by the native bundle builder. For your own data, use the same
[custom-bundle preparation](../../shared/local-dataset.md#prepare-your-own-examples)
contract. Inspect aligned RGB/control/caption examples before training.

`TRANSFER_QA_BASE_MODEL_ONLY=1` skips only `net_ema.*` when converted Nano contains
`net.*` alone. Regular-weight loading stays strict. EMA age zero lets the first
optimizer update initialize EMA from updated regular weights. Native same-job
resume ignores this warm-start skip and restores full state. Use `0` when the
base already has EMA; retain the chosen flag across the restart.

The helper passes these explicit local overrides:

| Field                                                                                         | Local binding                             |
| --------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `model.config.tokenizer.bucket_name`, `object_store_credential_path_pretrained`               | Empty strings                             |
| `model.config.tokenizer.vae_path`                                                             | `VIDEO_VAE_PATH`                          |
| `model.config.vlm_config.tokenizer.config_variant`                                            | `hf`                                      |
| `model.config.vlm_config.tokenizer.pretrained_model_name`                                     | `NANO_HF_DIR/text_tokenizer`              |
| `model.config.vlm_config.pretrained_weights.enabled`                                          | `False`; DCP supplies weights             |
| Stage 0 `model.config.sound_tokenizer.bucket_name`, `object_store_credential_path_pretrained` | Empty strings                             |
| Stage 0 `model.config.sound_tokenizer.avae_path`, `avae_config_path`                          | `AUDIO_VAE_PATH`, `AUDIO_VAE_CONFIG_PATH` |
| `checkpoint.load_from_object_store.enabled`, `save_to_object_store.enabled`                   | `False`                                   |
| Every native local child dataset's `bundle_path`                                              | `TRANSFER_BUNDLE_PATH`                    |

## Plan training resources

For the example2 configuration, the resource-planning reference is eight H200
GPUs with roughly 140 GB of device memory each and approximately 126 GB per
GPU at the Stage 3 peak. Size your allocation for the resolved clip lengths,
batch packing and parallelism. A different GPU memory budget requires a
matching recipe and memory plan.

Full-state checkpoints include optimizer state. For this recipe, allow roughly
170 GB per Stage 0–2 checkpoint and 280 GB per Stage 3 checkpoint, plus space for model
assets and temporary writes. Keeping both checkpoints from all four stages
requires roughly 1.6 TB of output space. Keep the preceding complete checkpoint
until its replacement has finished writing.

The short walkthrough below selects synchronous checkpoint saving to avoid
the additional shared, pinned-memory staging allocation. It still needs host
RAM for training and checkpoint serialization. Before enabling asynchronous
saves, provision `/dev/shm` for the combined staging allocations of all ranks
on each host and leave room for other shared-memory users. See
[checkpoint memory](checkpoint-and-resume.md#checkpoint-memory) for the
capacity check and supported options.

## Run and resume each stage

Use the eight-GPU allocation described above. On two four-GPU nodes, replace the
`TRAIN=(torchrun ...)` assignment below with the
[multi-node array](multinode-training.md)
for that allocation. Each call must start all ranks and return when
training finishes. Keep the output directory shared between nodes.

```bash
TRAIN=(torchrun --standalone --nproc_per_node=8 -m cosmos_framework.scripts.train_experiment)
for stage in 0 1 2 3; do
  python "$STAGE_HELPER" --stage "$stage" > "$ARGS_FILE"
  mapfile -t STAGE_ARGS < "$ARGS_FILE"
  ARGS=(
    "${STAGE_ARGS[@]}"
    job.project=cosmos3_sim_transfer
    job.group=local
    job.wandb_mode=disabled
    "job.name=${RUN_ID}_stage${stage}"
    checkpoint.save_iter=6
    checkpoint.dcp_async_mode_enabled=False
    trainer.logging_iter=1
  )
  python -m cosmos_framework.scripts.train_experiment --dryrun "${ARGS[@]}" trainer.max_iter=6
  "${TRAIN[@]}" "${ARGS[@]}" trainer.max_iter=6
  "${TRAIN[@]}" "${ARGS[@]}" trainer.max_iter=12
  stage_checkpoint="$IMAGINAIRE_OUTPUT_ROOT/cosmos3_sim_transfer/local/${RUN_ID}_stage${stage}/checkpoints/iter_000000012"
  test -f "$stage_checkpoint/model/.metadata"
  case "$stage" in
    0) export STAGE0_CKPT_ROOT="$stage_checkpoint" ;;
    1) export STAGE1_CKPT_ROOT="$stage_checkpoint" ;;
    2) export STAGE2_CKPT_ROOT="$stage_checkpoint" ;;
  esac
done
```

Run this loop from one controlling shell. Each stage saves after six iterations,
then resumes to a total of 12 iterations. To adjust a stage individually, run its
loop body separately and inspect `config.yaml` after `--dryrun`. The dry run
writes the resolved configuration without constructing weights or loaders.
Choose longer durations and parent checkpoints using held-out results on your
dataset; the recipe defaults are listed in the [method reference](canonical-recipe.md).

| Stage | Native experiment                 | Parents                                                                              |
| ----- | --------------------------------- | ------------------------------------------------------------------------------------ |
| 0     | `sim_local_transfer_stage0_bidir` | `BASE_CKPT_ROOT` regular weights                                                     |
| 1     | `sim_local_transfer_stage1_tf`    | `STAGE0_CKPT_ROOT` EMA to regular                                                    |
| 2     | `sim_local_transfer_stage2_tfdcm` | `STAGE1_CKPT_ROOT` EMA to student and frozen teacher                                 |
| 3     | `sim_local_transfer`              | `STAGE2_CKPT_ROOT/model` student EMA; `STAGE0_CKPT_ROOT/model` teacher and score EMA |

Each new stage starts fresh optimization state. Same-run latest checkpoints
override the warm-start initializer on restart. Keep run names, output root,
bundle contents, topology and parents unchanged between iterations 6 and 12.
Use a new run identity when switching to another training dataset.

## Native recipe and resume behavior

The resolved native configuration preserves these settings:

- Stage 0: separate 480/720 child streams, weights 4/2, frame ceilings 201/149,
  74,000-token packing, dropout 0.1 and post-decode direct resize. Native
  `preserve_transfer_streams=True` retains these stream settings.
- Stages 1–2: 480 tier, text dropout 0.1; Stage 3: 480 tier, dropout 0. Keep
  dense-caption probability 0.07, equal four-control weights and latent chunk 1.
- Stage 2 and Stage 3: teacher CFG 3 with the native negative prompt and preserved
  metadata. Stage 3: equal rollout lengths, one packed pair per rank, CP=1,
  replay block 1 and full-state saving. The native recipe defaults to asynchronous
  saves; this short walkthrough overrides only the save mode to synchronous.
- The `local_simulation_resume` callback saves the joint packer and child-loader
  cursors with the checkpoint and restores them when the same run restarts.
- Stage 3 checkpoints contain student and score-model weights, student EMA,
  and both optimizer/scheduler states. Wait for asynchronous writes to complete
  before using or moving a checkpoint.

The six-iteration example includes tokenizer compilation, which starts after
three iterations, and a complete Stage 3 student/score-model update cycle.
At each optimizer boundary, the trainer logs `distillation optimizer phase`
with the same one-based iteration number as checkpoints and iteration-speed
logs. `net` identifies the student and `fake_score` identifies the score model.
Across the twelve iterations in this walkthrough, student phases occur at
iterations 1 and 7; the other ten iterations run the score-model phase.
The saved optimizer and scheduler states retain each phase's step count
when training resumes.
Keep parents accessible: Stage 2/3 construct them before restoring the child
checkpoint. See [checkpoint operations](checkpoint-and-resume.md) for restart
requirements.

Finish with [inference](../../inference/SKILL.md).
Select useful parents through held-out review before extending training.

## Framework training entry points

The [local Stage 0–3 walkthrough](#run-and-resume-each-stage)
above supplies executable commands, local assets, six fresh plus six resumed
updates, and handoffs from your own selected EMA checkpoints. It runs
`cosmos_framework.scripts.train_experiment` with `cosmos_framework/configs/simulation/local.py`.

Framework owns these recipe factories under
`$FRAMEWORK_DIR/cosmos_framework/configs/simulation/common/`:

| Stage | Factory                                                              | Local experiment                  |
| ----- | -------------------------------------------------------------------- | --------------------------------- |
| 0     | `transfer_stage0_profiles.make_transfer_bidirectional_experiment`    | `sim_local_transfer_stage0_bidir` |
| 1     | `transfer_profiles.make_transfer_teacher_forcing_experiment`         | `sim_local_transfer_stage1_tf`    |
| 2     | `transfer_profiles.make_transfer_teacher_forcing_dcm_experiment`     | `sim_local_transfer_stage2_tfdcm` |
| 3     | `transfer_stage3_profiles.make_transfer_self_forcing_dmd_experiment` | `sim_local_transfer`              |

The local config uses
`cosmos_framework.data.generator.local_simulation.config.make_local_experiment_config`.
Stage 0 preserves the native 480/720 streams and overrides caption dropout,
resize policy and minimum frames. Stages 1–2 override dropout to 0.1. Stage 3
uses the matching 480/zero-dropout bundle policy. The native local loader
captures resume state.

For your own data, prepare a local bundle with the same
[data contract](../../shared/data-contract.md). Before launching, inspect the
resolved configuration for your resource paths, selected parents and output
identity. Use the [multi-node launcher](multinode-training.md)
when the eight GPU processes span two nodes.
