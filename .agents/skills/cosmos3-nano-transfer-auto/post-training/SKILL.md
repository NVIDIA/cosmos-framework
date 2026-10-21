---
name: cosmos3-nano-transfer-auto-post-training
description: Fine-tune the released Cosmos3-Nano-Transfer-Auto model on paired driving data in Cosmos Framework, or run the four-phase training curriculum from Cosmos3-Nano.
---

# Post-train Cosmos3-Nano-Transfer-Auto

Fine-tune the released Transfer-Auto model on your paired recordings to turn
camera WSM and HD-map controls into matching camera video and LiDAR.
The Quickstart starts from released weights. To train from Cosmos3-Nano,
follow the four-phase curriculum below.

This guide uses **Cosmos Framework**. You supply your own recordings and controls;
no access to the original training dataset is required.

## Choose your starting point

| Goal                          | Start with                        | Follow                                                                      |
| ----------------------------- | --------------------------------- | --------------------------------------------------------------------------- |
| Adapt to your data            | Released Transfer-Auto checkpoint | [Fine-tuning Quickstart](#quickstart-fine-tune-the-released-model)          |
| Reproduce the training phases | Cosmos3-Nano checkpoint           | [Four-phase curriculum](#reproduce-the-four-training-phases)                |
| Generate a driving scene      | Released Transfer-Auto checkpoint | [Inference](../inference/SKILL.md)                                          |
| Build a streaming simulator   | Your Transfer-Auto checkpoint     | [Sim-Auto distillation](../../cosmos3-nano-sim-auto/post-training/SKILL.md) |

## Set up the environment

Follow the [Framework setup guide](../../../../docs/setup.md)
for installation in your selected checkout, then apply the additional
[Auto requirements](../shared/framework-runtime.md). Reuse a working Framework environment.

The commands use the native
[four-phase recipes](../../../../cosmos_framework/configs/simulation/common/transfer_auto_recipes.py)
and [registrations](../../../../cosmos_framework/configs/simulation/experiment/auto/transfer_auto.py)
in this checkout: `transfer_auto_phase1`–`phase4`.

## Quickstart: fine-tune the released model

### Activate the Framework environment

Run from the Framework checkout with the prepared environment. The same
entrypoint is used by the four-phase curriculum:

```bash
cd "${FRAMEWORK_DIR:?set the absolute Framework checkout root}"
source .venv/bin/activate
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1
entry=(
  --config cosmos_framework/configs/simulation/config.py
  --experiment-module cosmos_framework.configs.simulation.experiment.auto.transfer_auto
)
```

Read the [data contract](references/data-contract.md) for sensor meaning and the
[parameter guide](references/parameter-guide.md) before changing training settings.
Keep user assets outside the repository. The
[reference scene](../shared/reference-scene.md) provides a pinned HF download,
the `daytime_scene` assets and a shared native training index. Use it as a format
reference, then prepare a diverse corpus of your own recordings.

### Prepare the released model

Set `TRANSFER_AUTO_HF` to a local snapshot directory, `TRANSFER_AUTO_REVISION`
to its selected immutable revision, and `TRANSFER_AUTO_DCP` to a new conversion
directory. Obtain the matching RGB VAE and text tokenizer from the
[model assets guide](../shared/framework-runtime.md#model-snapshots).
Set `VIDEO_VAE` to the matched Wan2.2 VAE file and `LOCAL_QWEN_TOKENIZER`
to the matching local text tokenizer directory for data preparation.

```bash
hf download nvidia/Cosmos3-Nano-Transfer-Auto \
  --revision "${TRANSFER_AUTO_REVISION:?set the chosen snapshot revision}" \
  --local-dir "${TRANSFER_AUTO_HF:?set a local snapshot directory}"
python -m cosmos_framework.scripts.prepare_sim_auto_checkpoint \
  --checkpoint-path "$TRANSFER_AUTO_HF" \
  --output-path "${TRANSFER_AUTO_DCP:?set a new weights-only DCP directory}"
export COSMOS_TRANSFER_AUTO_VIDEO_VAE="${VIDEO_VAE:?set the matched RGB VAE file}"
export COSMOS_TRANSFER_AUTO_LIDAR_VAE="$TRANSFER_AUTO_HF/lidar_vae/diffusion_pytorch_model.safetensors"
```

Despite its name, this preparation tool converts the bidirectional Auto weights
and checks the trained LiDAR heads and sensor table. It does not distill the model.

### Prepare a local training bundle

Store scene descriptions in the caption files, without model-facing headers.
The native training transforms add the camera-rig/current-view direction and FOV
description and the task-specific control instruction during formatting and
tokenization. You do not need to add either manually; see
[automatic prefixes](../shared/caption-policy.md#caption-text-and-automatic-prefixes).

1. Prepare synchronized RGB and WSM, camera-local target-grounded captions,
   measured LiDAR and HD-map range controls. Prefer
   [timestamped captions](references/data-contract.md#training-captions) to match
   the training format; paragraph captions remain supported.
2. Check camera identity and alignment with
   [input preparation](../inference/references/input-preparation.md). Do not use
   fictional inference appearance assignments as training labels.
3. Create the [JSONL training index](references/data-contract.md#local-training-index),
   or use `$AUTO_SAMPLE_ROOT/training-index.jsonl` from the reference scene.
   That index selects paragraph captions; follow
   [caption selection](../shared/reference-scene.md#choose-a-caption-style)
   to select the timestamped alternative before packaging.
   Set `AUTO_INDEX`, `LOCAL_QWEN_TOKENIZER` and a new `AUTO_BUNDLE` directory.
4. Package the selected physical cameras. This list matches the seven-camera example:

```bash
python -m cosmos_framework.data.generator.local_simulation.auto_bundle \
  --index "${AUTO_INDEX:?set your training index}" \
  --tokenizer "${LOCAL_QWEN_TOKENIZER:?set the local text tokenizer}" \
  --output "${AUTO_BUNDLE:?set a new bundle directory}" \
  --camera-keys \
    camera_front_wide_120fov camera_cross_right_120fov \
    camera_rear_right_70fov camera_rear_tele_30fov \
    camera_rear_left_70fov camera_cross_left_120fov camera_front_tele_30fov
export COSMOS_TRANSFER_AUTO_LOCAL_BUNDLE="$AUTO_BUNDLE"
```

The builder copies media and tokenizer resources. It derives eligible camera
subsets from your available cameras; this changes exposure relative to an
eleven-camera corpus. Supply both resolution strata used by the recipe, or
remove the absent stream before constructing the loader. For this 720p/300-frame
example, use the [bounded sample settings](../shared/reference-scene.md#short-training-check).
Inspect a decoded sample and actual formatted batch before accepting the source.

### Configure the fine-tuning run

Choose a unique `AUTO_RUN_NAME` and export your shared `IMAGINAIRE_OUTPUT_ROOT`.
Resolve a new run using the final-model recipe with weights-only initialization:

```bash
settings=(
  experiment=transfer_auto_phase4
  "job.name=${AUTO_RUN_NAME:?set a fresh run name}"
  "checkpoint.load_path=$TRANSFER_AUTO_DCP"
  checkpoint.load_training_state=False
  'checkpoint.keys_not_to_resume=[]'
  'checkpoint.keys_to_skip_loading=[net_ema.]'
  model.config.ema.iteration_shift=0
)
python -m cosmos_framework.scripts.train_experiment \
  "${entry[@]}" --dryrun -- "${settings[@]}"
```

This configuration defaults to 50,000 joint-training steps. See the
[parameter guide](references/parameter-guide.md) to adjust the learning rate,
schedule, dropout and training duration. This starts a new run from the model
weights, rather than resuming the original training. Regular weights load
strictly; the new run starts a fresh EMA history.

### Launch and monitor

For either path, append allocation-matched topology to the selected `settings`.
Set `NNODES`, `GPUS_PER_NODE`, `NODE_RANK`, `MASTER_ADDR`,
`CONTEXT_PARALLEL` and `FSDP_SHARDS` from the allocation:

```bash
torchrun --nnodes="$NNODES" --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$NODE_RANK" --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:29500" \
  -m cosmos_framework.scripts.train_experiment "${entry[@]}" -- \
  "${settings[@]}" \
  model_parallel.context_parallel_size="$CONTEXT_PARALLEL" \
  model.config.parallelism.context_parallel_shard_degree="$CONTEXT_PARALLEL" \
  model.config.parallelism.data_parallel_shard_degree="$FSDP_SHARDS"
```

Resolve again with these same topology overrides before launch. Reference CP/FSDP
degrees are 32/32; smaller layouts require memory/batch validation, not just fewer
launcher processes. Begin with a separate bounded smoke, then train/save/reload.
For the reference scene on four H200 GPUs, start with the explicit
[short-check topology and synchronous save settings](../shared/reference-scene.md#short-training-check).
Read the [checkpoint memory requirements](../shared/framework-runtime.md#checkpoint-memory)
before enabling asynchronous saves.
Inspect actual task/resolution exposure, finite RGB/LiDAR losses, LR, updated
parameters and complete checkpoints. Shortening `trainer.max_iter` does not
require shortening the scheduler cycle.

For same-phase resume, restore complete model/EMA, optimizer, scheduler, iteration
and supported loader state. Reuse that run's output identity; do not apply fresh
warm-start resets again. The CLI's stop iteration is an absolute endpoint.

## Export and check inference

Export from the **saved resolved training config**, with the chosen weight
variant and matching tokenizers:

```bash
python -m cosmos_framework.scripts.export_model \
  --checkpoint-path "${AUTO_CHECKPOINT:?set a completed iteration root}" \
  --config-file "${AUTO_CONFIG:?set the saved run config.yaml}" \
  --use-ema-weights --no-vit \
  --student-only-checkpoint-metadata --bundle-lidar-vae \
  --per-view-captions on \
  --per-view-caption-task av_joint_camera_lidar_transfer \
  -o "${AUTO_EXPORT:?set a new export directory}"
```

This is the joint final-model export, not a separate-sensor intermediate export.
`--no-vit` creates a generation-only artifact. Keep tokenizer assets complete and
remove machine-specific paths from export metadata before sharing. Use the
[inference skill](../inference/SKILL.md) on a fixed held-out scene.

## Reproduce the four training phases

Follow the [curriculum and handoffs](references/canonical-recipe.md):

1. Cosmos3-Nano → separate RGB/LiDAR: 70k updates using single-camera RGB
   clips of up to 397 frames (400 requested) and separate 21-sweep LiDAR clips.
2. Separate-sensor checkpoint → short joint RGB/LiDAR: 61 RGB frames aligned
   with 21 LiDAR sweeps.
3. Short-joint checkpoint → long joint RGB/LiDAR: up to 397 RGB frames aligned
   with 133 sweeps, continuing to joint iteration 22,000.
4. Continue long joint training to joint iteration 50,000 with the same model,
   attention, dropout and schedule.

Separate-sensor tasks have different durations because they are independent
samples. Only the joint phases require matching RGB/LiDAR time windows.
Phases 2–4 use deduplicated cross-view attention over the preceding 0.4 seconds,
including the current time. Phases 3 and 4 both use 10% whole-LiDAR dropout.

## References

- [Framework environment and model assets](../shared/framework-runtime.md)
- [Training data contract and bundle layout](references/data-contract.md)
- [Parameter guide](references/parameter-guide.md)
- [Shared Auto reference scene](../shared/reference-scene.md)
- [Camera and WSM input checks](../inference/references/input-preparation.md)
- [Four-phase training recipe and checkpoint handoffs](references/canonical-recipe.md)
