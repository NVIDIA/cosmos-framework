---
name: cosmos3-nano-sim-transfer-post-training
description: Prepare aligned spatial controls, RGB targets and captions for Cosmos3-Nano-Sim-Transfer; run native Cosmos Framework Stage 0–3 training with local assets; resume full-state checkpoints; pass selected EMA to inference.
---

# Post-train Cosmos3-Nano-Sim-Transfer

Cosmos3-Nano-Sim-Transfer generates RGB video from a spatial-control clip and a
caption. Controls can be edge, blur, depth or segmentation. Train on aligned
control/RGB clips and captions with Cosmos Framework. Start at Stage 0 with
compatible base regular weights, then pass your selected EMA outputs through
teacher forcing, four-step distillation and self-forcing.

## Choose your starting point

| Goal                         | Start with                                                      | Follow                                                            |
| ---------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------- |
| Start a new training chain   | Compatible base regular weights and a prepared local bundle     | [Stage 0 Quickstart](#quickstart-train-stage-0)                   |
| Continue your training chain | Your completed stage checkpoints and saved configuration        | [Continue training](#continue-training)                           |
| Generate RGB video           | Matched release checkpoint or your selected Stage 3 student EMA | [Inference](../../cosmos3-inference-simulation-transfer/SKILL.md) |

## Set up the environment

Export `FRAMEWORK_DIR` as the absolute Framework checkout root and
`SIM_TRANSFER_SKILL` as the absolute location of this skill bundle. Follow
the [Sim-Transfer setup guide](../../cosmos3-inference-simulation-transfer/references/setup.md) to prepare
Python 3.13 and CUDA in the Framework checkout containing this skill. Run
training commands from `FRAMEWORK_DIR`.

> **Skill note (not in the source):** the setup guide sets `SIM_TRANSFER_SKILL` to the Sim-Transfer inference skill folder, which holds the sample-preparation script. The commands of this skill address its own scripts from `FRAMEWORK_DIR`.

## Quickstart: train Stage 0

### Activate the Framework environment

The native entry point is `cosmos_framework.scripts.train_experiment`; the
native local config is `cosmos_framework/configs/simulation/local.py`.

Activate the prepared environment from the Framework root:

```bash
export FRAMEWORK_DIR="${FRAMEWORK_DIR:?Set the absolute Framework checkout root}"
: "${SIM_TRANSFER_SKILL:?Set the absolute skill bundle location}"
cd "$FRAMEWORK_DIR"
source .venv/bin/activate
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1 COSMOS_SMOKE=0
export COSMOS_DEVICE=cuda WANDB_MODE=disabled
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
```

### Prepare the training data

1. Split complete recordings into train/validation before constructing windows.
   Preserve camera, source FPS, dimensions, frame counts and sample identity.
2. Prepare aligned RGB, depth and segmentation. Edge and blur are derived from
   resized RGB. Use equal weights for the four controls.
3. Review each structured caption and its nonempty `temporal_caption` against
   first, middle and last RGB frames. Retained captions use the same-window
   93% structured / 7% dense mixture. See [caption preparation](../../cosmos3-inference-simulation-transfer/references/caption-guide.md).
4. Follow the [data contract](../shared/data-contract.md) and
   [local bundle layout](../shared/local-dataset.md). Inspect aligned decoded
   controls, RGB and packed tensors. Normalize pixels to `pixels / 127.5 - 1`;
   a 121-frame clip becomes 31 temporal latents. Reject aspect distortion above 3%.
5. Bind local video VAE, Nano text tokenizer and completed DCP roots with the
   [local training walkthrough](references/local-training.md). Stage 0 also
   initializes its inherited audio tokenizer and needs both audio assets.

Use **[Hugging Face example2](../../cosmos3-inference-simulation-transfer/references/hf-sample.md)** for this workflow.
Download its RGB, depth, segmentation, short prompt and structured caption, then
follow [bundle preparation](../shared/local-dataset.md) to create the training
manifest and add the tokenizer.

### Select a stage

| Stage                    | Native local experiment           | Initial weights                                                            |
| ------------------------ | --------------------------------- | -------------------------------------------------------------------------- |
| 0, bidirectional teacher | `sim_local_transfer_stage0_bidir` | Compatible base regular weights                                            |
| 1, teacher forcing       | `sim_local_transfer_stage1_tf`    | Your Stage 0 EMA                                                           |
| 2, TF-dCM                | `sim_local_transfer_stage2_tfdcm` | Your Stage 1 EMA for student and frozen teacher                            |
| 3, SF-DMD                | `sim_local_transfer`              | Your Stage 2 EMA for student; your Stage 0 EMA for teacher and score model |

### 1. Bind local resources

Complete [bundle preparation](../shared/local-dataset.md) and
[Nano conversion](references/nano-conversion.md) first. These workflows set
`SIM_TRANSFER_TRAINING_BUNDLE`, `NANO_HF_DIR`, `VIDEO_VAE_PATH` and
`BASE_CKPT_ROOT`. Stage 0 also uses the audio assets derived during conversion.
Set `IMAGINAIRE_OUTPUT_ROOT` to a writable directory. All model, data and output
paths must resolve identically on every training rank.

```bash
set -euo pipefail
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
# Skill note (not in the source): SIM_TRANSFER_SKILL is now the Sim-Transfer inference skill folder, so this script is addressed from FRAMEWORK_DIR. The source path was "$SIM_TRANSFER_SKILL/post-training/scripts/qa_transfer_stage.py".
STAGE_HELPER="$FRAMEWORK_DIR/.agents/skills/cosmos3-nano-sim-transfer/post-training/scripts/qa_transfer_stage.py"
ARGS_FILE="$IMAGINAIRE_OUTPUT_ROOT/${RUN_ID}.args"
```

Use `TRANSFER_QA_BASE_MODEL_ONLY=1` only when the converted base lacks EMA;
use `0` when it already contains EMA. Regular-weight loading remains strict.
The native same-job resume restores full state. Keep `RUN_ID` and resource
bindings unchanged while resuming; generating a new ID starts a different run.

### 2. Resolve Stage 0

The argument helper binds local assets and checkpoints to the native recipe.
This example saves every six iterations and resumes to iteration 12. Select
Stage 0 for a new chain and capture its arguments:

```bash
stage=0
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
python -m cosmos_framework.scripts.train_experiment \
  --dryrun "${ARGS[@]}" trainer.max_iter=6
```

Inspect `config.yaml` under
`$IMAGINAIRE_OUTPUT_ROOT/cosmos3_sim_transfer/local/${RUN_ID}_stage0/` for your
local assets, base checkpoint and stage settings. A dry run does not construct
weights or loaders. This short walkthrough uses synchronous saving to avoid
the additional shared, pinned-memory staging allocation. It preserves the
full-state checkpoint format. See [checkpoint memory](references/checkpoint-and-resume.md#checkpoint-memory)
when enabling asynchronous saving.

### 3. Train, save and resume

Use an existing eight-GPU allocation sized for the
[training resource requirements](references/local-training.md#plan-training-resources).
For two four-GPU nodes, use the
[multi-node launcher](references/multinode-training.md)
instead of the single-node `TRAIN` array below.

```bash
TRAIN=(torchrun --standalone --nproc_per_node=8 -m cosmos_framework.scripts.train_experiment)
"${TRAIN[@]}" "${ARGS[@]}" trainer.max_iter=6
"${TRAIN[@]}" "${ARGS[@]}" trainer.max_iter=12
stage_checkpoint="$IMAGINAIRE_OUTPUT_ROOT/cosmos3_sim_transfer/local/${RUN_ID}_stage${stage}/checkpoints/iter_000000012"
test -f "$stage_checkpoint/model/.metadata"
```

The second launch resumes the same run from iteration 6 and continues to the
absolute endpoint 12. `trainer.max_iter` is the total iteration limit, including
restored iterations. The full-state checkpoint includes weights, optimizer,
scheduler, trainer and local-loader state. Wait for checkpoint writes to finish
before using the output in the next stage.

## Continue training

After Stage 0 succeeds, export its exact output:

```bash
export STAGE0_CKPT_ROOT="$stage_checkpoint"
```

Change `stage=0` to `stage=1` in step 2 and repeat argument preparation and the
two launches, keeping the chain's `RUN_ID`. After Stage 1, export
`STAGE1_CKPT_ROOT="$stage_checkpoint"`; after Stage 2, export
`STAGE2_CKPT_ROOT="$stage_checkpoint"`. Select stages 2 and 3 in the same way.
The [complete walkthrough](references/local-training.md) provides a loop for
all four stages. Each stage uses a new run name and fresh optimization state;
same-stage resume retains the run name and output root.

The example uses six fresh and six resumed iterations per stage. For your
dataset, choose training durations and checkpoints using held-out outputs.
The [method reference](references/canonical-recipe.md) lists the full recipe
defaults, and the [parameter guide](references/parameter-guide.md) explains
settings you can tune.

Stage 0 uses mixed 480/720 streams. Stages 1–3 use 480. Text dropout is 0.1 in
Stages 0–2 and 0 in Stage 3. These stage settings override the bundle's transform
policy. The native recipes use eight GPU processes and one latent per causal
chunk. Stages 2–3 use teacher guidance 3. Stage 3 uses equal-length rollouts,
alternates five score-model updates with one student update, and saves both
optimizer states.

### Resume and hand off

Keep the same run identity, bundle, topology and parent artifacts when resuming.
See [checkpoint and resume](references/checkpoint-and-resume.md) for full-state
requirements and explicit checkpoint selection.

Pass your selected Stage 3 student EMA and resolved configuration to the
[inference Skill](#run-inference-from-your-trained-checkpoint), together with
the tokenizer/VAE, preprocessing, caption policy and sampling settings. Retain
the full checkpoint for further training.

## Run inference from your trained checkpoint

> **Skill note (not in the source):** this section is the Stage 3 DCP route of the Sim-Transfer inference skill, moved here because its block reads variables of the training shell. In the source it follows the Hugging Face download block of [1. Select the matched inference checkpoint](../../cosmos3-inference-simulation-transfer/SKILL.md#1-select-the-matched-inference-checkpoint), which defines `SIM_TRANSFER_RESOURCE_OVERRIDES` ("the HF resource overrides") and the checkpoint argument array that the block below replaces, and it precedes the generation commands of that skill.

For a **Stage 3 DCP produced by the training walkthrough**, replace only the
checkpoint argument array below. Keep the HF resource overrides:

```bash
: "${SIM_TRANSFER_DCP_ROOT:?Select a completed Stage 3 checkpoint from your training run}"
export COSMOS_SIM_TRANSFER_STAGE0_CHECKPOINT="${STAGE0_CKPT_ROOT:?Use the Stage 0 parent from the training walkthrough}"
export COSMOS_SIM_TRANSFER_STAGE2_CHECKPOINT="${STAGE2_CKPT_ROOT:?Use the Stage 2 parent from the training walkthrough}"
export COSMOS_SIM_TRANSFER_LOCAL_BUNDLE="${SIM_TRANSFER_TRAINING_BUNDLE:?Use the prepared example2 training bundle}"
SIM_TRANSFER_CHECKPOINT_ARGS=(
  --checkpoint-path "$SIM_TRANSFER_DCP_ROOT"
  --config-file cosmos_framework/configs/simulation/local.py
  --experiment sim_local_transfer
  --credential-path ""
  --use-ema-weights
)
```

Framework resolves the recipe before loading a native checkpoint, so retain
its parent-checkpoint and prepared-bundle bindings. Inference then skips training
loaders, parent initialization and teacher/score-model construction. The native
route selects EMA from the checkpoint's `model/` child. Earlier stages use
their own [inference profiles](references/canonical-recipe.md#check-intermediate-training-models).

## References

- [Framework environment setup](../../cosmos3-inference-simulation-transfer/references/setup.md)
- [Control video and caption contract](../shared/data-contract.md)
- [Caption preparation and review](../../cosmos3-inference-simulation-transfer/references/caption-guide.md)
- [Local training bundle preparation](../shared/local-dataset.md)
- [Complete Stage 0–3 training walkthrough](references/local-training.md)
- [Hugging Face examples](../../cosmos3-inference-simulation-transfer/references/hf-sample.md)
- [Nano checkpoint conversion](references/nano-conversion.md)
- [Two-node training launcher](references/multinode-training.md)
- [Training method and stage settings](references/canonical-recipe.md)
- [Parameter guide](references/parameter-guide.md)
- [Checkpoint handoffs and resume](references/checkpoint-and-resume.md)
