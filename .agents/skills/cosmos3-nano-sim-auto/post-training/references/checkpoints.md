# Stage handoffs, resume and inference export

## Accept a checkpoint

Record its exact directory, source/runtime version, resolved config, data manifest
and completed iteration. Confirm finite losses and intended trainable groups.

A complete same-stage checkpoint includes:

- Model and EMA state, trainer iteration and supported RNG state.
- TF/TF-dCM: `optim/` and `scheduler/`.
- SF-DMD: `optim_net/`, `optim_fake_score/`, `scheduler_net/`,
  `scheduler_fake_score/`.
- Loader state for every training rank.
- Completed DCP metadata/shards and a matching `latest_checkpoint.txt`.

Wait for asynchronous saves. Metadata presence does not replace loading the state
and checking the saved iteration.

## Initialize the next stage

| Next stage | Bind                                                         | Restore                                                                     |
| ---------- | ------------------------------------------------------------ | --------------------------------------------------------------------------- |
| TF         | `SIM_AUTO_STAGE0_CHECKPOINT`: Transfer-Auto iteration root   | Regular model only; new EMA/optimizer/scheduler/counters                    |
| TF-dCM     | `SIM_AUTO_STAGE1_CHECKPOINT`: accepted TF iteration root     | Regular student and teacher; new stage state                                |
| SF-DMD     | `SIM_AUTO_STAGE2_CHECKPOINT_MODEL`: accepted TF-dCM `model/` | Regular student; original Transfer-Auto teacher/fake score; new stage state |

For SF-DMD also set `SIM_AUTO_STAGE3_TEACHER_MODEL` to the original Transfer-Auto
`model/`, not the causal TF teacher. Inspect the profiles' resolved roles.

## Resume the same stage

Reuse that run's identity and complete checkpoint. Clear the fresh weights-only
exception with `checkpoint.keys_to_skip_loading=[]`. Preserve model/EMA,
optimizer(s), scheduler(s), trainer and loader state. Check the first resumed
iteration and phase. `trainer.max_iter` is an absolute endpoint, not additional steps.

Do not delete the latest pointer or reapply warm-start resets to repair an
unexpected resume. Confirm the artifact actually selected. Changed-topology/data
resume needs separate validation and does not promise identical sample replay.

## Export the student

Use an accepted Stage 3 checkpoint and its **saved resolved config**.
Run the export on a Hopper (sm90) or Blackwell (sm100/sm103) GPU with
FlashAttention-4. The causal student's maskless replay requires `COSMOS_DEVICE=cuda`;
the exporter otherwise defaults to CPU.

Set `STAGE3_RUN`, `SIM_AUTO_STAGE3_CHECKPOINT` (iteration root) and a new
`SIM_AUTO_STUDENT_HF`:

```bash
python -m cosmos_framework.scripts.prepare_student_config \
  --config-file "$STAGE3_RUN/config.yaml" \
  --output-config "$STAGE3_RUN/student/config.yaml" \
  --report "$STAGE3_RUN/student/report.json"
COSMOS_DEVICE=cuda python -m cosmos_framework.scripts.export_model \
  --checkpoint-path "$SIM_AUTO_STAGE3_CHECKPOINT" \
  --config-file "$STAGE3_RUN/student/config.yaml" \
  --no-use-ema-weights --no-vit \
  --student-only-checkpoint-metadata --bundle-lidar-vae \
  --per-view-captions on \
  --per-view-caption-task av_joint_camera_lidar_transfer \
  -o "$SIM_AUTO_STUDENT_HF" \
  --experiment-overrides \
  model.config.rolling_kv_cache_chunks=2 \
  model.config.rolling_kv_sink_chunks=1 \
  model.config.rolling_prompt_policy=segmented
```

Review inserted defaults in the preparation report. Export only the **regular
student**, not teacher/fake-score weights or an implicit EMA substitution.
The prepared model-only config has no dataloaders, so explicit per-view caption
flags are required. The explicit rolling settings are also required: the training
config does not supply the streaming profile.

`--no-vit` selects generation-only export; `--bundle-lidar-vae` includes matched
LiDAR weights/config. Remove machine-specific paths from export metadata before sharing.
Preserve needed tokenizer assets and provenance separately.

Run [inference](../../inference/SKILL.md) on fixed held-out controls.
Check per-camera outputs, joint LiDAR, physical frame IDs and caption transitions.
Successful conversion is not inference parity or quality validation.
