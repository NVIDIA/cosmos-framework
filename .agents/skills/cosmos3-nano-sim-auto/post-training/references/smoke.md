# Bounded train/save/resume validation

Use these short runs to check your data, runtime and checkpoint handoffs before
committing to full training. They check loading, parameter updates and save/resume;
evaluate model quality after full training on your corpus. Do not enable
`COSMOS_SMOKE`, which can change architecture.

## Prepare a reduced source

Build a separate short **customer-owned** subset with the front-wide camera,
paired LiDAR and controls. Select only that camera in the bundle builder. Inspect
decoded samples and formatted batches, including present- and absent-LiDAR layouts.

Apply all four mapping removals from the [data contract](../../shared/data-contract.md)
so the reduced bundle's sampler is used. Use the [launcher](runtime.md) with
CP4/FSDP4 on a supported four-GPU node. Append these settings:

```bash
settings+=(
  job.wandb_mode=disabled '~trainer.callbacks.wandb'
  trainer.callbacks.norm_monitor.log_stat_wandb=false
  model.config.max_num_tokens_after_packing=100000
  dataloader_train.max_sequence_length=100000
  dataloader_train.max_samples_per_batch=1
  model.config.compile.enabled=false
  trainer.logging_iter=1
  checkpoint.dcp_async_mode_enabled=false
  checkpoint.load_from_object_store.enabled=false
  checkpoint.save_to_object_store.enabled=false
  "trainer.max_iter=${SMOKE_STOP:?set 2 for TF/TF-dCM or 6 for SF-DMD}"
  "checkpoint.save_iter=$SMOKE_STOP"
)
if [[ "$STAGE" == 3 ]]; then
  settings+=('~trainer.callbacks.wandb_10x' model.config.fake_score_compile.enabled=true)
else
  settings+=('~trainer.callbacks.wandb_2x' '~trainer.callbacks.wandb_val')
fi
for resolution in 480 720; do
  stream="dataloader_train.dataloaders.cam_lidar_joint_${resolution}p_separate_view_text"
  ratio=0
  if [[ "$resolution" == 480 ]]; then ratio=1; fi
  settings+=(
    "$stream.ratio=$ratio"
    "$stream.dataloader.dataset.transform_overrides.num_video_frames=24"
    "$stream.dataloader.dataset.transform_overrides.joint_aligned_num_video_frames=21"
  )
done
```

Resolve these settings before launch. This uses 21 aligned RGB frames, 480p and
one camera. Fake-score compilation stays enabled in Stage 3.

## Execute and accept in order

1. TF: two updates, synchronous save; check finite loss and strict initialization.
2. TF-dCM: initialize from accepted TF; two updates and save.
3. SF-DMD: initialize from accepted TF-dCM and original Transfer-Auto;
   **six updates** and save. This includes one student and five critic updates.
4. Resume each same-stage run with full state and the same identity: absolute
   endpoints 3, 3 and 12. Clear the fresh EMA skip.
5. Verify resumed iteration, optimizer/scheduler counters, loader state and finite
   losses. The second SF-DMD interval must include the next student update.

Check the [checkpoint inventory](checkpoints.md#accept-a-checkpoint), load the saved
trainer iteration and inspect each rank's loader state. Retain logs, configs and
phase metrics.

After the short check, train with the intended cameras, resolutions and horizons,
and evaluate held-out scenes.
