---
name: cosmos3-nano-sim-auto-post-training
description: Distill Cosmos3-Nano-Transfer-Auto into Cosmos3-Nano-Sim-Auto using customer RGB, WSM, captions and LiDAR in Cosmos Framework. Covers TF, TF-dCM, SF-DMD, checkpoint handoffs, resume and student export.
---

# Post-train Cosmos3-Nano-Sim-Auto

Start from **Transfer-Auto**, including a Transfer-Auto model adapted to your
camera rig or domain. Do not start this workflow from plain Cosmos3-Nano or
directly fine-tune the released distilled Sim-Auto student with an SFT objective.
See the [model and checkpoint contract](../shared/model-contract.md).

## Choose your starting point

| Goal                      | Start with                                                   | Follow                                                                   |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------ |
| Generate now              | Released Sim-Auto student; no training needed                | [Inference](../inference/SKILL.md)                                       |
| Train your simulator      | Compatible Transfer-Auto checkpoint, not plain Cosmos3-Nano  | [Stage 1 Quickstart](#quickstart-train-stage-1)                          |
| Adapt to your data or rig | Post-train Transfer-Auto first, then distill that checkpoint | [Transfer-Auto](../../cosmos3-nano-transfer-auto/post-training/SKILL.md) |

Ordinary SFT of the final distilled student is not the workflow documented here.

## What you need

- The complete Transfer-Auto snapshot and matched text, RGB and LiDAR tokenizers.
- Your own paired recordings, controls and captions in a validated local bundle.
- The [shared environment](../../cosmos3-nano-transfer-auto/shared/framework-runtime.md).
- An allocation sized for the selected topology, and new output directories.

The shared [reference scene](../../cosmos3-nano-transfer-auto/shared/reference-scene.md)
provides an HF download of `daytime_scene` and its native training index.
It is a format example, not a training corpus. Use
[Transfer-Auto data preparation](../../cosmos3-nano-transfer-auto/post-training/SKILL.md#prepare-a-local-training-bundle)
to create the same local format. Then apply the
[Sim-Auto data settings](../shared/data-contract.md), including the camera-map
overrides required for the generic bundle builder. For this 720p/300-frame scene,
apply the [short training settings](../../cosmos3-nano-transfer-auto/shared/reference-scene.md#short-training-check)
as well; the full TF window does not fit.
An inference request alone is not training data.

Keep caption files as scene descriptions only. The native training transforms
add the camera-rig/current-view direction and FOV description and the joint
WSM/HD-map task instruction automatically during formatting and tokenization.
Do not duplicate either in your labels; see
[automatic prefixes](../../cosmos3-nano-transfer-auto/shared/caption-policy.md#caption-text-and-automatic-prefixes).

## Three stages

| Stage                                             | What it learns                                       | Student starts from | Teacher                              |
| ------------------------------------------------- | ---------------------------------------------------- | ------------------- | ------------------------------------ |
| 1: teacher forcing (TF)                           | Generate causally using clean preceding observations | Transfer-Auto       | None                                 |
| 2: teacher-forcing diffusion consistency (TF-dCM) | Reduce generation to four denoising steps            | Accepted Stage 1    | Frozen Stage 1                       |
| 3: self-forcing distribution matching (SF-DMD)    | Generate using its own preceding predictions         | Accepted Stage 2    | Original bidirectional Transfer-Auto |

Stage 3 also trains a fake-score network initialized from the original Transfer-Auto
teacher. That network is not part of the inference artifact.

## Set up the environment

Follow the [Framework setup guide](../../../../docs/setup.md)
for installation in your selected checkout, then check the additional
[Auto requirements](../../cosmos3-nano-transfer-auto/shared/framework-runtime.md).
Reuse a working Framework environment.

## Quickstart: train Stage 1

### 1. Activate the Framework environment

Set `FRAMEWORK_DIR` to your Framework checkout. Run the commands below from its
root with the prepared environment:

```bash
cd "${FRAMEWORK_DIR:?set the absolute Framework checkout root}"
source .venv/bin/activate
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1
```

Use the same environment and shared asset paths on every training node.

### 2. Prepare the Transfer-Auto initializer and local data

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
[checkpoint preparation](references/checkpoint-preparation.md) for the asset
checks and fresh-EMA handling.

### 3. Select and resolve Stage 1

Read the [recipe](references/canonical-recipe.md) and
[parameter guide](references/parameter-guide.md) before changing its settings.
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

### 4. Check the configuration and launch training

Set `NNODES`, `GPUS_PER_NODE`, `NODE_RANK` and `MASTER_ADDR` from the allocation,
then set `CONTEXT_PARALLEL` and `FSDP_SHARDS` for that layout. Reference CP/FSDP
degrees are 32/64; the [runtime guide](references/runtime.md) describes the
reference allocations. For a short first run, apply the existing
[smoke settings](references/smoke.md) before resolving and launching; do not
enable `COSMOS_SMOKE`, which can change architecture. The 720p reference scene
also needs its [short-window settings](../../cosmos3-nano-transfer-auto/shared/reference-scene.md#short-training-check).

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
[fresh-process resume check](references/smoke.md). A dry run only resolves
configuration; it does not load weights or train.

## Continue training and export

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
[checkpoint acceptance and export](references/checkpoints.md#export-the-student)
to export the **regular Stage 3 student**, then run
[inference](../inference/SKILL.md) on held-out controls.

The reference budget is 4000 **total trainer iterations per stage**. Stage 3 alternates
one student update with five critic updates: 4000 iterations give 667 student and
3333 critic updates. Evaluate the trained student on held-out scenes.

Use regular-network weights between stages; do not substitute EMA. Each new
stage starts new optimizer, scheduler and iteration state. Same-stage resume
restores that stage's complete state instead. See the
[checkpoint guide](references/checkpoints.md).

Training uses paired RGB/LiDAR dropout and per-camera captions. Inspect both
present- and absent-LiDAR batches. Monitor actual losses and Stage 3
`phase_metrics.jsonl`, not just whether a process is alive.

Preserve accepted checkpoints, resolved configs, runtime/source versions, data
manifest and test results. Do not upload confidential data or credentials in an
MR, logs or model export.

## References

- [Model and checkpoint contract](../shared/model-contract.md)
- [Framework environment and model assets](../../cosmos3-nano-transfer-auto/shared/framework-runtime.md)
- [Shared Auto reference scene](../../cosmos3-nano-transfer-auto/shared/reference-scene.md)
- [Training data and camera mappings](../shared/data-contract.md)
- [Checkpoint and tokenizer preparation](references/checkpoint-preparation.md)
- [Three-stage training recipe](references/canonical-recipe.md)
- [Parameter guide](references/parameter-guide.md)
- [Stage configuration and training launch](references/runtime.md)
- [Short training, save and resume checks](references/smoke.md)
- [Checkpoint handoffs, resume and student export](references/checkpoints.md)
