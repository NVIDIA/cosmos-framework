# Resolve and launch a Sim-Auto stage

Use the [shared environment](../../../cosmos3-nano-transfer-auto/shared/framework-runtime.md)
and [prepared assets](checkpoint-preparation.md). Run from the Framework repository
root using the native Python experiment entry point, not generic RGB SFT TOML recipes.

## Bind accepted predecessors

Export the predecessor variables needed by the selected stage so the training
process can resolve them:

- TF: `SIM_AUTO_STAGE0_CHECKPOINT` is the converted Transfer-Auto iteration root.
- TF-dCM: `SIM_AUTO_STAGE1_CHECKPOINT` is the accepted TF iteration root.
- SF-DMD: `SIM_AUTO_STAGE2_CHECKPOINT_MODEL` is accepted TF-dCM's `model/`;
  `SIM_AUTO_STAGE3_TEACHER_MODEL="$SIM_AUTO_STAGE0_CHECKPOINT/model"`.

Never substitute an unrelated checkpoint based on its iteration number.

## Resolve the selected stage

Set `STAGE` to 1, 2 or 3. First define `bundle_overrides` from the
[data contract](../../shared/data-contract.md#preserve-your-bundles-camera-mappings).
Then:

```bash
case "${STAGE:?choose 1, 2 or 3}" in
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
python -m cosmos_framework.scripts.train_experiment \
  "${entry[@]}" --dryrun -- "${settings[@]}"
```

The EMA exception is for **fresh converted initialization only**. On same-stage
resume replace it with `checkpoint.keys_to_skip_loading=[]`. Use a new output
identity for a new run; reusing one can select its existing latest checkpoint.

Inspect the resolved model, teacher/student paths, local mappings, VAE/text assets,
optimizer groups, scheduler, stopping point and data windows. Save that config.

## Allocate and launch

Reference layouts are 64/128/64 nodes respectively, four GPUs per node,
CP32/FSDP64. These are reference scales, not minimum hardware requirements.
Smaller allocations need topology, packing and memory validation.

Append the same allocation-matched topology to dryrun and launch:

```bash
topology=(
  "model_parallel.context_parallel_size=${CONTEXT_PARALLEL:?set CP}"
  "model.config.parallelism.context_parallel_shard_degree=$CONTEXT_PARALLEL"
  "model.config.parallelism.data_parallel_shard_degree=${FSDP_SHARDS:?set FSDP shards}"
)
python -m cosmos_framework.scripts.train_experiment \
  "${entry[@]}" --dryrun -- "${settings[@]}" "${topology[@]}"
torchrun --nnodes="${NNODES:?}" --nproc-per-node="${GPUS_PER_NODE:?}" \
  --node-rank="${NODE_RANK:?}" --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR:?}:29500" \
  -m cosmos_framework.scripts.train_experiment "${entry[@]}" -- \
  "${settings[@]}" "${topology[@]}"
```

Run on each allocated node with its own node rank. Verify the device mesh is
supported and divides world size. Independent packed batches per global iteration
are `world_size / CP × grad_accum_iter`; samples/tokens per pack vary. Do not
equate a CP rank with an independent batch or silently scale LR.

## Observe and finish

Outputs are under
`$IMAGINAIRE_OUTPUT_ROOT/cosmos3_interactive/$SIM_AUTO_RUN_GROUP/stageN_seed0`.
Check finite sensor losses, LR, checkpoint completion and phase counters.
Stage 3's native callback writes `phase_metrics.jsonl`; inspect both student and
critic updates. No Skill-local callback or extra import path is needed.

Confirm checkpoint saving completes before using it for a handoff or export.
