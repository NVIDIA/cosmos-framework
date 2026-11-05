# Three-stage causal distillation recipe

The native Framework stage factories implement the following recipe. Resolve the
actual config before launch.

| Stage  | Module under `cosmos_framework.configs.simulation.experiment.auto` | Experiment                        |
| ------ | ------------------------------------------------------------------ | --------------------------------- |
| TF     | `stage1_tf`                                                        | `sim_auto_joint_stage1_tf_all`    |
| TF-dCM | `stage2_tfdcm`                                                     | `sim_auto_joint_stage2_tfdcm_all` |
| SF-DMD | `stage3_sfdmd`                                                     | `sim_auto_joint_stage3_sfdmd_all` |

Keep these registered selector names unchanged. They read your prepared data with
the [customer bundle and mapping overrides](../../shared/data-contract.md).
Read [handoffs](checkpoints.md) and [launch instructions](runtime.md) together.

## Material values

| Setting                              | TF                   | TF-dCM                 | SF-DMD                         |
| ------------------------------------ | -------------------- | ---------------------- | ------------------------------ |
| Total trainer iterations             | 4000                 | 4000                   | 4000                           |
| Reference nodes × GPUs/node          | 64 × 4               | 128 × 4                | 64 × 4                         |
| CP / FSDP shard                      | 32 / 64              | 32 / 64                | 32 / 64                        |
| Independent packed batches/iteration | 8                    | 16                     | 8                              |
| Gradient accumulation                | 1                    | 1                      | 1                              |
| Packed sample limit                  | Variable             | Variable               | 1                              |
| Model/loader token ceiling           | 1,600,000            | 1,600,000              | 1,600,000                      |
| Requested camera frames              | 400                  | 400                    | 96, exact aligned 93           |
| LR                                   | 1e-5                 | 2e-6                   | Student 1e-6; fake score 2e-7  |
| Adam betas                           | 0.9, 0.999           | 0, 0.999               | 0, 0.999                       |
| Weight decay / epsilon               | 0.01 / 1e-8          | Same                   | Same                           |
| Scheduler / warmup                   | Constant / 0         | Constant / 0           | Independent constant / 0       |
| RGB / LiDAR loss scales              | 1 / 2                | 100 / 200              | 100 / 200                      |
| Scoring-time distribution            | waver                | waver                  | uniform                        |
| Student compilation                  | Enabled              | Enabled                | Disabled                       |
| Teacher / fake-score compilation     | N/A                  | Teacher follows method | Teacher disabled; fake enabled |
| Activation checkpointing             | Full, no CPU offload | Same                   | Same                           |
| Checkpoint interval / async          | 100 / enabled        | Same                   | Same                           |

All stages use trainer seed 0, loader seed 42, bf16 compute, float32 FSDP master
weights, bf16 reduction, correct CP gradients, EMA rate 0.1/shift 0, regular-weight
handoffs and strict loading. Loss reduction is `local_item_mean`, not global
sample averaging. TF/TF-dCM have `grad_clip=False`; SF-DMD uses cap 1.0.

Start with 4000 total trainer iterations per stage and evaluate held-out quality.
Keep total trainer iterations distinct from optimizer steps.

## Objectives and conditioning

TF learns causal generation from clean history. TF-dCM freezes the accepted TF
teacher, uses 48 teacher timesteps with skipping interval 1, detached target
endpoint, teacher guidance 1.5 and a metadata-preserving negative prompt. The
student learns the four-step schedule.

SF-DMD freezes the original bidirectional Transfer-Auto teacher, initializes fake
score from that teacher, and initializes the student from TF-dCM. It uses replayed
causal rollouts, block size 1, equal rollout lengths, up to four simulation steps
and one backward-gradient step. Preserve the supplied attention/history contract.

With `student_update_freq=6`, indices 0,6,... update the student; other iterations
update the critic. At 4000 total iterations this gives **667 student and 3333
critic updates** from fresh counters. Inspect the native `phase_metrics.jsonl`.

Both resolution streams read your validated bundle: 480×832 with ratio 0.6 and
720×1280 with ratio 0.4. These are expected iteration shares, not clip/token/pixel
fractions. Packing makes sample exposure variable. Both streams use caption
dropout 0.2, per-view dropout 0.2, paired whole-LiDAR dropout 0.1 and prefix
probabilities 0.7/0.2/0.1 for 0/1/2 RGB latents. Do not independently drop LiDAR
targets and controls or desynchronize their initial conditions.

## Acceptance

Use the [bounded smoke](smoke.md), then evaluate the intended full setup and
held-out inference. Preserve runtime/source versions, resolved config, data
manifest, regular-weight handoffs, optimizer counters and complete checkpoints.
