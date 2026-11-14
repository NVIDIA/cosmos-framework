---
name: cosmos3-post-train-simulation-auto
description: >
  Post-train the Cosmos3 Auto simulation models on your own paired driving data in Cosmos Framework: fine-tune
  Cosmos3-Nano-Transfer-Auto from the released checkpoint, or run its four-phase curriculum from Cosmos3-Nano, and
  distill a Transfer-Auto checkpoint into a Cosmos3-Nano-Sim-Auto student in three stages (TF, TF-dCM, SF-DMD), with
  the native train_experiment launcher under torchrun. Covers the training bundle, checkpoint handoffs, resume and
  export. Use when the user asks "fine-tune Transfer-Auto on my rig or data", "post-train Transfer-Auto", "reproduce
  the Transfer-Auto training phases", "distill Sim-Auto", "train my own Sim-Auto student" or "export my Transfer-Auto
  or Sim-Auto checkpoint". Not for inference with the released checkpoints (cosmos3-inference-simulation-auto),
  Sim-Bimanual or Sim-Transfer post-training (start at cosmos3-post-train-simulation), or generator and reasoner SFT
  (cosmos3-post-training).
---

# Cosmos3 Simulation Post-Training: Auto (Transfer-Auto and Sim-Auto)

The Auto block of Cosmos3 Simulation post-training, in Cosmos Framework and on your own paired driving recordings, in two parts: fine-tune Cosmos3-Nano-Transfer-Auto, then distill a Transfer-Auto checkpoint into a Cosmos3-Nano-Sim-Auto student. The second part starts from the first: a Sim-Auto student of your own is distilled from the released Transfer-Auto checkpoint or from your adapted Transfer-Auto export, not from plain Cosmos3-Nano.

This skill replaces the Transfer-Auto and Sim-Auto post-training skills. Their two bodies (cosmos-framework `main` 45fecab) are the sections [Fine-tune Transfer-Auto](#fine-tune-transfer-auto) and [Distill Sim-Auto](#distill-sim-auto) below, word for word apart from what is listed here: the two section titles stand in for the old titles "Post-train Cosmos3-Nano-Transfer-Auto" and "Post-train Cosmos3-Nano-Sim-Auto"; the headings below them are one level down, and the two that the bodies shared ("Choose your starting point", "Set up the environment") carry the model name; link targets are re-pointed; each old description is the paragraph that opens its section; five blocks moved to [`references/transfer-auto-training-notes.md`](references/transfer-auto-training-notes.md) and [`references/sim-auto-training-notes.md`](references/sim-auto-training-notes.md) behind one-line pointers; and the two "References" lists are the opening words of their rows in [Reference files](#reference-files). The row's entry skill is `../cosmos3-post-train-simulation/SKILL.md`.

## When to use this skill

- Use to fine-tune Cosmos3-Nano-Transfer-Auto on your own paired driving data from the released checkpoint, to run its four-phase curriculum from Cosmos3-Nano, or to distill a Transfer-Auto checkpoint into a Cosmos3-Nano-Sim-Auto student through TF, TF-dCM and SF-DMD, in Cosmos Framework.
- Not for inference with the released checkpoints, see `../cosmos3-inference-simulation-auto/SKILL.md`; not for generator or reasoner supervised fine-tuning, see `../cosmos3-post-training/SKILL.md`; not for Sim-Bimanual or Sim-Transfer post-training, see `../cosmos3-post-train-simulation-bimanual/SKILL.md` and `../cosmos3-post-train-simulation-transfer/SKILL.md`.

## Starting points

| Goal                                        | Start with                                                                                     | Follow                                                                                                                                                                 |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Try a released model before training        | Released Transfer-Auto or Sim-Auto checkpoint; no training needed                              | Auto inference skill: [Transfer-Auto](../cosmos3-inference-simulation-auto/SKILL.md#transfer-auto), [Sim-Auto](../cosmos3-inference-simulation-auto/SKILL.md#sim-auto) |
| Adapt Transfer-Auto to your data            | Released Transfer-Auto checkpoint                                                              | [Fine-tune Transfer-Auto](#fine-tune-transfer-auto): [Fine-tuning Quickstart](#quickstart-fine-tune-the-released-model)                                                |
| Reproduce the Transfer-Auto training phases | Cosmos3-Nano checkpoint                                                                        | [Fine-tune Transfer-Auto](#fine-tune-transfer-auto): [Four-phase curriculum](#reproduce-the-four-training-phases)                                                      |
| Train your own Sim-Auto student             | Compatible Transfer-Auto checkpoint (released, or your adapted export), not plain Cosmos3-Nano | [Distill Sim-Auto](#distill-sim-auto): [Stage 1 Quickstart](#quickstart-train-stage-1)                                                                                 |

## Prerequisites

Framework checkout, `uv` and checkpoint download: `../cosmos3-setup/SKILL.md` and the repository [setup guide](../../../docs/setup.md). Then what the sources require: the additional [Auto requirements](../cosmos3-inference-simulation-auto/references/framework-runtime.md) (environment and model snapshots, shared with inference), their training half in [`references/runtime.md`](references/runtime.md) (multi-GPU allocation and output root, training memory, checkpoint memory, and [Training components](references/runtime.md#training-components), which says where the matched RGB VAE and the text tokenizer come from: "the recipes expect a Wan2.2 VAE `.pth`, distributed in `Wan-AI/Wan2.2-TI2V-5B`", and the text tokenizer directory is "normally `text_tokenizer/` in the snapshot"), and each part's own list: [Set up the environment (Transfer-Auto)](#set-up-the-environment-transfer-auto), and for Sim-Auto [What you need](#what-you-need) and [Set up the environment (Sim-Auto)](#set-up-the-environment-sim-auto).

## Fine-tune Transfer-Auto

Fine-tune the released Cosmos3-Nano-Transfer-Auto model on paired driving data in Cosmos Framework, or run the four-phase training curriculum from Cosmos3-Nano.

Fine-tune the released Transfer-Auto model on your paired recordings to turn
camera WSM and HD-map controls into matching camera video and LiDAR.
The Quickstart starts from released weights. To train from Cosmos3-Nano,
follow the four-phase curriculum below.

This guide uses **Cosmos Framework**. You supply your own recordings and controls;
no access to the original training dataset is required.

### Choose your starting point (Transfer-Auto)

| Goal                          | Start with                        | Follow                                                                   |
| ----------------------------- | --------------------------------- | ------------------------------------------------------------------------ |
| Adapt to your data            | Released Transfer-Auto checkpoint | [Fine-tuning Quickstart](#quickstart-fine-tune-the-released-model)       |
| Reproduce the training phases | Cosmos3-Nano checkpoint           | [Four-phase curriculum](#reproduce-the-four-training-phases)             |
| Generate a driving scene      | Released Transfer-Auto checkpoint | [Inference](../cosmos3-inference-simulation-auto/SKILL.md#transfer-auto) |
| Build a streaming simulator   | Your Transfer-Auto checkpoint     | [Sim-Auto distillation](#distill-sim-auto)                               |

### Set up the environment (Transfer-Auto)

Follow the [Framework setup guide](../../../docs/setup.md)
for installation in your selected checkout, then apply the additional
[Auto requirements](../cosmos3-inference-simulation-auto/references/framework-runtime.md). Reuse a working Framework environment.

The commands use the native
[four-phase recipes](../../../cosmos_framework/configs/simulation/common/transfer_auto_recipes.py)
and [registrations](../../../cosmos_framework/configs/simulation/experiment/auto/transfer_auto.py)
in this checkout: `transfer_auto_phase1`–`phase4`.

### Quickstart: fine-tune the released model

#### Activate the Framework environment

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
[parameter guide](references/transfer-auto-parameters.md) before changing training settings.
Keep user assets outside the repository. The
[reference scene](../cosmos3-inference-simulation-auto/references/reference-scene.md) provides an HF download,
the `daytime_scene` assets and a shared native training index. Use it as a format
reference, then prepare a diverse corpus of your own recordings.

#### Prepare the released model

Set `TRANSFER_AUTO_HF` to a local snapshot directory, `TRANSFER_AUTO_REVISION`
to its selected immutable revision, and `TRANSFER_AUTO_DCP` to a new conversion
directory. Obtain the matching RGB VAE and text tokenizer from the
[model assets guide](../cosmos3-inference-simulation-auto/references/framework-runtime.md#model-snapshots).
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

#### Prepare a local training bundle

Store scene descriptions in the caption files, without model-facing headers.
The native training transforms add the camera-rig/current-view direction and FOV
description and the task-specific control instruction during formatting and
tokenization. You do not need to add either manually; see
[automatic prefixes](../cosmos3-inference-simulation-auto/references/caption-policy.md#caption-text-and-automatic-prefixes).

1. Prepare synchronized RGB and WSM, camera-local target-grounded captions,
   measured LiDAR and HD-map range controls. Prefer
   [timestamped captions](references/data-contract.md#training-captions) to match
   the training format; paragraph captions remain supported.
2. Check camera identity and alignment with
   [input preparation](../cosmos3-inference-simulation-auto/references/input-preparation.md). Do not use
   fictional inference appearance assignments as training labels.
3. Create the [JSONL training index](references/data-contract.md#local-training-index),
   or use `$AUTO_SAMPLE_ROOT/training-index.jsonl` from the reference scene.
   That index selects paragraph captions; follow
   [caption selection](../cosmos3-inference-simulation-auto/references/reference-scene.md#choose-a-caption-style)
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

What the builder copies and derives, the two resolution strata, the bounded settings for the 720p/300-frame example and the sample inspection before you accept the source: [After packaging the bundle](references/transfer-auto-training-notes.md#after-packaging-the-bundle).

#### Configure the fine-tuning run

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
[parameter guide](references/transfer-auto-parameters.md) to adjust the learning rate,
schedule, dropout and training duration. This starts a new run from the model
weights, rather than resuming the original training. Regular weights load
strictly; the new run starts a fresh EMA history.

#### Launch and monitor

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
[short-check topology and synchronous save settings](references/training-bundle-and-short-check.md#short-training-check).
Read the [checkpoint memory requirements](references/runtime.md#checkpoint-memory)
before enabling asynchronous saves.
Inspect actual task/resolution exposure, finite RGB/LiDAR losses, LR, updated
parameters and complete checkpoints. Shortening `trainer.max_iter` does not
require shortening the scheduler cycle.

For same-phase resume, restore complete model/EMA, optimizer, scheduler, iteration
and supported loader state. Reuse that run's output identity; do not apply fresh
warm-start resets again. The CLI's stop iteration is an absolute endpoint.

### Export and check inference

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
[inference skill](../cosmos3-inference-simulation-auto/SKILL.md#transfer-auto) on a fixed held-out scene.

### Reproduce the four training phases

The outline of the four phases (what each one trains, its frame and sweep windows, the cross-view attention window and the LiDAR dropout) is in [Reproduce the four training phases](references/transfer-auto-training-notes.md#reproduce-the-four-training-phases); the phase commands, checkpoint handoffs and settings are in the [four-phase recipe](references/transfer-auto-recipe.md).

## Distill Sim-Auto

Distill Cosmos3-Nano-Transfer-Auto into Cosmos3-Nano-Sim-Auto using customer RGB, WSM, captions and LiDAR in Cosmos Framework. Covers TF, TF-dCM, SF-DMD, checkpoint handoffs, resume and student export.

Start from **Transfer-Auto**, including a Transfer-Auto model adapted to your
camera rig or domain. Do not start this workflow from plain Cosmos3-Nano or
directly fine-tune the released distilled Sim-Auto student with an SFT objective.
See the [model and checkpoint contract](references/sim-auto-model-contract.md).

### Choose your starting point (Sim-Auto)

| Goal                      | Start with                                                   | Follow                                                              |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------- |
| Generate now              | Released Sim-Auto student; no training needed                | [Inference](../cosmos3-inference-simulation-auto/SKILL.md#sim-auto) |
| Train your simulator      | Compatible Transfer-Auto checkpoint, not plain Cosmos3-Nano  | [Stage 1 Quickstart](#quickstart-train-stage-1)                     |
| Adapt to your data or rig | Post-train Transfer-Auto first, then distill that checkpoint | [Transfer-Auto](#fine-tune-transfer-auto)                           |

Ordinary SFT of the final distilled student is not the workflow documented here.

### What you need

- The complete Transfer-Auto snapshot and matched text, RGB and LiDAR tokenizers.
- Your own paired recordings, controls and captions in a validated local bundle.
- The [shared environment](../cosmos3-inference-simulation-auto/references/framework-runtime.md).
- An allocation sized for the selected topology, and new output directories.

The data instructions (the reference scene as a format example, the bundle from [Transfer-Auto data preparation](#prepare-a-local-training-bundle), the [Sim-Auto data settings](references/sim-auto-data.md) with the camera-map overrides, the short settings for the 720p scene, caption files) and the table of the three stages (what each stage learns, where its student starts, its teacher): [Data instructions](references/sim-auto-training-notes.md#data-instructions) and [Three stages](references/sim-auto-training-notes.md#three-stages).

### Set up the environment (Sim-Auto)

Follow the [Framework setup guide](../../../docs/setup.md)
for installation in your selected checkout, then check the additional
[Auto requirements](../cosmos3-inference-simulation-auto/references/framework-runtime.md).
Reuse a working Framework environment.

### Quickstart: train Stage 1

#### 1. Activate the Framework environment

Set `FRAMEWORK_DIR` to your Framework checkout. Run the commands below from its
root with the prepared environment:

```bash
cd "${FRAMEWORK_DIR:?set the absolute Framework checkout root}"
source .venv/bin/activate
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1
```

Use the same environment and shared asset paths on every training node.

#### 2. Prepare the Transfer-Auto initializer and local data

Set `TRANSFER_AUTO_HF` to your complete compatible Transfer-Auto export if you
have already adapted the model, and skip the download below. Otherwise, set
`TRANSFER_AUTO_REVISION` to the selected immutable released-model revision and
`TRANSFER_AUTO_HF` to its local snapshot directory, then download the model:

```bash
hf download nvidia/Cosmos3-Nano-Transfer-Auto \
  --revision "${TRANSFER_AUTO_REVISION:?set the chosen snapshot revision}" \
  --local-dir "${TRANSFER_AUTO_HF:?set a local snapshot directory}"
```

Prepare `AUTO_BUNDLE` with the data instructions above. Set `VIDEO_VAE` to the
matched Wan2.2 VAE file, `SIM_AUTO_STAGE0_CHECKPOINT` to a new conversion directory,
`IMAGINAIRE_OUTPUT_ROOT` to shared training storage and `SIM_AUTO_RUN_GROUP` to a
new name for this chain. Bind those resources and convert the initializer:

```bash
: "${AUTO_BUNDLE:?set the validated local training bundle}"
: "${VIDEO_VAE:?set the matching Wan2.2_VAE.pth file}"
: "${SIM_AUTO_STAGE0_CHECKPOINT:?set a new DCP conversion directory}"
: "${IMAGINAIRE_OUTPUT_ROOT:?set a shared training-output root}"
: "${SIM_AUTO_RUN_GROUP:?set a unique group for this training chain}"
export COSMOS_SIM_AUTO_LOCAL_BUNDLE="$AUTO_BUNDLE"
export RGB_VAE="$VIDEO_VAE"
export LIDAR_VAE="$TRANSFER_AUTO_HF/lidar_vae/diffusion_pytorch_model.safetensors"
export QWEN_TOKENIZER="$AUTO_BUNDLE/resources/tokenizer"
export SIM_AUTO_STAGE0_CHECKPOINT IMAGINAIRE_OUTPUT_ROOT SIM_AUTO_RUN_GROUP

python -m cosmos_framework.scripts.prepare_sim_auto_checkpoint \
  --checkpoint-path "$TRANSFER_AUTO_HF" \
  --output-path "$SIM_AUTO_STAGE0_CHECKPOINT"
```

The conversion produces a weights-only iteration root containing `model/`, not
optimizer or trainer state. Keep the LiDAR tokenizer separate. See
[checkpoint preparation](references/sim-auto-checkpoint-preparation.md) for the asset
checks and fresh-EMA handling.

#### 3. Select and resolve Stage 1

Read the [recipe](references/sim-auto-recipe.md) and
[parameter guide](references/sim-auto-parameters.md) before changing its settings.
First preserve the generic bundle's camera mappings at every stage:

```bash
bundle_overrides=()
for resolution in 480 720; do
  dataset="dataloader_train.dataloaders.cam_lidar_joint_${resolution}p_separate_view_text.dataloader.dataset.transform_overrides"
  bundle_overrides+=(
    "~${dataset}.camera_keys"
    "~${dataset}.camera_caption_key_mapping"
    "~${dataset}.camera_control_key_mapping"
    "~${dataset}.camera_sampler"
  )
done
```

Select Stage 1 for a new chain. The same block selects Stage 2 or 3 after you
bind its accepted predecessors in step 5:

```bash
STAGE=1
case "$STAGE" in
  1) module=stage1_tf; experiment=sim_auto_joint_stage1_tf_all ;;
  2) module=stage2_tfdcm; experiment=sim_auto_joint_stage2_tfdcm_all ;;
  3) module=stage3_sfdmd; experiment=sim_auto_joint_stage3_sfdmd_all ;;
  *) echo "STAGE must be 1, 2 or 3" >&2; exit 2 ;;
esac
entry=(
  --config cosmos_framework/configs/simulation/config.py
  --experiment-module "cosmos_framework.configs.simulation.experiment.auto.$module"
)
settings=(
  "experiment=$experiment"
  "job.group=$SIM_AUTO_RUN_GROUP"
  "job.name=stage${STAGE}_seed0"
  "${bundle_overrides[@]}"
)
if [[ "$STAGE" == 1 ]]; then
  settings+=('checkpoint.keys_to_skip_loading=[net_ema.]')
fi
```

The EMA skip applies only to fresh converted initialization. For same-stage
resume, replace it with `checkpoint.keys_to_skip_loading=[]`.

#### 4. Check the configuration and launch training

Set `NNODES`, `GPUS_PER_NODE`, `NODE_RANK` and `MASTER_ADDR` from the allocation,
then set `CONTEXT_PARALLEL` and `FSDP_SHARDS` for that layout. Reference CP/FSDP
degrees are 32/64; the [runtime guide](references/sim-auto-runtime.md) describes the
reference allocations. For a short first run, apply the existing
[smoke settings](references/sim-auto-smoke.md) before resolving and launching; do not
enable `COSMOS_SMOKE`, which can change architecture. The 720p reference scene
also needs its [short-window settings](references/training-bundle-and-short-check.md#short-training-check).

```bash
topology=(
  "model_parallel.context_parallel_size=${CONTEXT_PARALLEL:?set CP}"
  "model.config.parallelism.context_parallel_shard_degree=$CONTEXT_PARALLEL"
  "model.config.parallelism.data_parallel_shard_degree=${FSDP_SHARDS:?set FSDP shards}"
)
python -m cosmos_framework.scripts.train_experiment \
  "${entry[@]}" --dryrun -- "${settings[@]}" "${topology[@]}"
```

Inspect the saved configuration for the selected initializer, camera mappings,
VAE/text assets, data windows and topology. Then run the same arguments on each
allocated node with its own node rank:

```bash
torchrun --nnodes="${NNODES:?}" --nproc-per-node="${GPUS_PER_NODE:?}" \
  --node-rank="${NODE_RANK:?}" --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR:?}:29500" \
  -m cosmos_framework.scripts.train_experiment "${entry[@]}" -- \
  "${settings[@]}" "${topology[@]}"
```

Outputs are under
`$IMAGINAIRE_OUTPUT_ROOT/cosmos3_interactive/$SIM_AUTO_RUN_GROUP/stageN_seed0`.
Check finite sensor losses, LR and completed saves, then perform the
[fresh-process resume check](references/sim-auto-smoke.md). A dry run only resolves
configuration; it does not load weights or train.

### Continue training and export

After accepting a Stage 1 checkpoint, set `SIM_AUTO_STAGE1_CHECKPOINT` to its
exact iteration root. Change `STAGE=1` to `STAGE=2` in step 3 and repeat the
configuration and launch steps. After
accepting Stage 2, bind its `model/` and the original Transfer-Auto teacher:

```bash
export SIM_AUTO_STAGE2_CHECKPOINT_MODEL="${SIM_AUTO_STAGE2_CHECKPOINT:?set the accepted Stage 2 iteration root}/model"
export SIM_AUTO_STAGE3_TEACHER_MODEL="$SIM_AUTO_STAGE0_CHECKPOINT/model"
```

Change that selection to `STAGE=3` and repeat steps 3 and 4. Check both student and critic
updates in `phase_metrics.jsonl`, not just total iteration progress. Follow
[checkpoint acceptance and export](references/sim-auto-checkpoints.md#export-the-student)
to export the **regular Stage 3 student**, then run
[inference](../cosmos3-inference-simulation-auto/SKILL.md#sim-auto) on held-out controls.

The reference budget of 4000 trainer iterations per stage, the rule for weights between stages, what to monitor and what to preserve: [Training budget, stage weights, monitoring and records](references/sim-auto-training-notes.md#training-budget-stage-weights-monitoring-and-records).

## Run inference with your checkpoint

Try a released checkpoint, or check your own export, with the Cosmos Framework sections of the Auto inference skill, `../cosmos3-inference-simulation-auto/SKILL.md`: [Transfer-Auto](../cosmos3-inference-simulation-auto/SKILL.md#transfer-auto) for the export of [Export and check inference](#export-and-check-inference), [Sim-Auto](../cosmos3-inference-simulation-auto/SKILL.md#sim-auto) for the regular Stage 3 student of [Continue training and export](#continue-training-and-export).

## Troubleshooting

Training, checkpoint, resume and distillation symptoms of both models: [`references/troubleshooting.md`](references/troubleshooting.md) ([Transfer-Auto](references/troubleshooting.md#troubleshoot-cosmos3-nano-transfer-auto), [Sim-Auto](references/troubleshooting.md#troubleshoot-cosmos3-nano-sim-auto)). Input, caption and generation symptoms, and how to collect one failing example: [Auto inference troubleshooting](../cosmos3-inference-simulation-auto/references/troubleshooting.md). Environment, CUDA, import and container failures: `../cosmos3-env-troubleshoot/SKILL.md`.

## Reference files

Every file under `references/`, then the three files of the Auto inference skill that the two old "References" lists name.

| File                                                                   | When to read it                                                                                                                                                                                      |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `references/transfer-auto-recipe.md`                                   | Four-phase training recipe and checkpoint handoffs: phase-to-config mapping, the `settings` of each phase, what each transition restores and resets, scientific settings, objective and conditioning |
| `references/transfer-auto-parameters.md`                               | Parameter guide (Transfer-Auto): keeping the model and data aligned, the settings to review with their reference values                                                                              |
| `references/data-contract.md`                                          | Training data contract and bundle layout: source files, the local training index, training captions, time and spatial alignment, the model-facing batch contract                                     |
| `references/training-bundle-and-short-check.md`                        | Training half of the reference-scene page: the scene's training index and bundle, the short training check for Transfer-Auto fine-tuning and for Sim-Auto Stage 1                                    |
| `references/runtime.md`                                                | Training half of the environment page: allocation and output root, training memory, checkpoint memory, training components                                                                           |
| `references/transfer-auto-training-notes.md`                           | Moved out of the Transfer-Auto section: the notes after packaging the bundle, the outline of the four training phases                                                                                |
| `references/sim-auto-recipe.md`                                        | Three-stage training recipe: stage modules and experiments, material values, objectives and conditioning, acceptance                                                                                 |
| `references/sim-auto-parameters.md`                                    | Parameter guide (Sim-Auto): keeping the model and data aligned, the settings to review with their reference values                                                                                   |
| `references/sim-auto-checkpoint-preparation.md`                        | Checkpoint and tokenizer preparation: binding the assets, converting the Transfer-Auto initializer, the fresh EMA                                                                                    |
| `references/sim-auto-checkpoints.md`                                   | Checkpoint handoffs, resume and student export: accepting a checkpoint, initializing the next stage, same-stage resume, the export command                                                           |
| `references/sim-auto-runtime.md`                                       | Stage configuration and training launch: binding predecessors, resolving a stage, allocation and launch, what to observe                                                                             |
| `references/sim-auto-smoke.md`                                         | Short training, save and resume checks: the reduced source, the smoke settings, the order of the checks                                                                                              |
| `references/sim-auto-data.md`                                          | Training data and camera mappings: the bundle camera-map overrides, the stage settings overlaid on the bundle, source coverage                                                                       |
| `references/sim-auto-model-contract.md`                                | Model and checkpoint contract: the checkpoint handoffs from Transfer-Auto to inference, trainable parameters                                                                                         |
| `references/sim-auto-training-notes.md`                                | Moved out of the Sim-Auto section: the data instructions, the table of the three stages, training budget, stage weights, monitoring and records                                                      |
| `references/troubleshooting.md`                                        | Training, checkpoint, resume and distillation symptoms and first checks for both models                                                                                                              |
| `../cosmos3-inference-simulation-auto/references/framework-runtime.md` | Framework environment and model assets, in the Auto inference skill: the environment, model snapshots, execution modes                                                                               |
| `../cosmos3-inference-simulation-auto/references/reference-scene.md`   | Shared Auto reference scene, in the Auto inference skill: download and checksum, files, caption styles                                                                                               |
| `../cosmos3-inference-simulation-auto/references/input-preparation.md` | Camera and WSM input checks, in the Auto inference skill                                                                                                                                             |

## Related skills

| Skill                                           | When to use                                                                                                     |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `../cosmos3-post-train-simulation/SKILL.md`     | Entry skill of the Simulation post-training row: routes to the Auto, Sim-Bimanual and Sim-Transfer block skills |
| `../cosmos3-inference-simulation-auto/SKILL.md` | Inference with the released Transfer-Auto and Sim-Auto checkpoints, or with your own export                     |
| `../cosmos3-setup/SKILL.md`                     | Framework checkout, `uv` and checkpoint download                                                                |
| `../cosmos3-env-troubleshoot/SKILL.md`          | Environment, CUDA, import and container failures                                                                |
| `../cosmos3-post-training/SKILL.md`             | Generator and reasoner supervised fine-tuning                                                                   |
