# Four-phase Transfer-Auto curriculum

Start from **Cosmos3-Nano**, then train separate sensors, short joint clips,
long joint clips, and a long-joint continuation using your own synchronized data.

The initial separate-sensor phase uses single-camera RGB clips of up to
397 frames at 30 FPS and separate LiDAR clips of 21 sweeps at 10 Hz. The RGB
loader requests 400 source frames and trims to the tokenizer-compatible
`1 + 4k` length. These independent tasks do not need matching durations.
Keep this setting for the 70k-update phase; 61 RGB frames / 21 sweeps is the
aligned window for the subsequent short-joint phase.

## Phase-to-config mapping

All four phases use the same paired customer-data bundle.

| Phase                      | Framework selector     | Task / horizon                                                                             | Endpoint          |
| -------------------------- | ---------------------- | ------------------------------------------------------------------------------------------ | ----------------- |
| 1. Separate sensors        | `transfer_auto_phase1` | 30% single-camera WSM→RGB, up to 397 frames; 70% HD-map→LiDAR, 21 sweeps; separate samples | 70k               |
| 2. Short joint             | `transfer_auto_phase2` | Paired multiview RGB/LiDAR; 61 frames / 21 sweeps                                          | 10k new           |
| 3. Long joint              | `transfer_auto_phase3` | Up to 397 frames / 133 sweeps                                                              | Joint counter 22k |
| 4. Long-joint continuation | `transfer_auto_phase4` | Same task, attention and 10% whole-LiDAR dropout as phase 3                                | Joint counter 50k |

All joint phases use the released deduplicated attention with a preceding
0.4-second cross-view window, including the current time. Phase 1 uses
single-view/two-way attention. Phase 4 retains phase 3's training state and
extends the same schedule; it does not introduce a new attention policy.

## Configure the phases

Use the entry point and launcher in [post-training](../SKILL.md).
Set `COSMOS_TRANSFER_AUTO_LOCAL_BUNDLE`, the matched RGB/LiDAR VAE paths and
a new output identity. Start with the immutable Cosmos3-Nano snapshot:

```bash
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path "${NANO_HF:?set the Cosmos3-Nano snapshot}" \
  -o "${NANO_DCP:?set a new conversion directory}"
export COSMOS_TRANSFER_AUTO_NANO_DCP="$NANO_DCP"
```

This creates weights-only initialization. Phase 1 initializes only new LiDAR
projections and the sensor embedding table, plus fresh optimizer/scheduler/EMA.

For each phase, set `settings` before the dry run and launch:

```bash
# Phase 1: retain 400 requested RGB frames (up to 397 usable) and 21 LiDAR sweeps.
settings=(experiment=transfer_auto_phase1 "job.name=$AUTO_RUN_NAME")

# Phase 2: load the actual phase-1 full checkpoint directly.
export COSMOS_TRANSFER_AUTO_PHASE1_CHECKPOINT="$SEPARATE_SENSOR_CHECKPOINT"
settings=(
  experiment=transfer_auto_phase2
  "job.name=$AUTO_RUN_NAME"
  "model.config.ema.iteration_shift=$SEPARATE_SENSOR_UPDATES"
)

# Phase 3: continue the joint counter, EMA and optimizer.
export COSMOS_TRANSFER_AUTO_PHASE2_CHECKPOINT="$SHORT_JOINT_CHECKPOINT"
settings=(
  experiment=transfer_auto_phase3
  "job.name=$AUTO_RUN_NAME"
  "model.config.ema.iteration_shift=$SEPARATE_SENSOR_UPDATES"
)

# Phase 4: continue from joint iteration 22000.
export COSMOS_TRANSFER_AUTO_PHASE3_CHECKPOINT="$LONG_JOINT_CHECKPOINT"
settings=(
  experiment=transfer_auto_phase4
  "job.name=$AUTO_RUN_NAME"
  "model.config.ema.iteration_shift=$SEPARATE_SENSOR_UPDATES"
)
```

Execute each block separately after its predecessor completes. Give each phase
a distinct `AUTO_RUN_NAME`. `SEPARATE_SENSOR_UPDATES` is the actual EMA history
inherited from phase 1 (70,000 for the stated budget and the factory default).
When shortening phase 1, override this offset in **all** joint phases. Verify
handoffs with a train/save/reload check before a full training run.

| Transition              | Restore                                              | Reset                                                               |
| ----------------------- | ---------------------------------------------------- | ------------------------------------------------------------------- |
| Cosmos3-Nano → separate | Compatible regular model weights                     | Optimizer, schedule, counter and EMA; initialize new sensor modules |
| Separate → short joint  | Model, EMA and Adam moments                          | Data, scheduler and trainer counter; retain actual EMA age          |
| Short → long joint      | Model, EMA, Adam, schedule and counter               | Data iterator only                                                  |
| Phase 3 → phase 4       | Same complete state                                  | Data iterator only                                                  |
| Same-phase resume       | Complete checkpoint including supported loader state | No phase-transition resets                                          |

## Scientific settings

| Setting                           | Separate (phase 1)                              | Short joint (phase 2)        | Long joint (phases 3 and 4) |
| --------------------------------- | ----------------------------------------------- | ---------------------------- | --------------------------- |
| Resolution exposure               | RGB 15% 480p + 15% 720p; LiDAR 70%              | 60% 480p / 40% 720p          | Same                        |
| Caption policy                    | Full RGB body; opening paragraph for LiDAR-only | Opening paragraph per camera | Full per-camera body        |
| Packing ceiling                   | 240k                                            | 300k                         | 1.5M                        |
| RGB / LiDAR loss scale            | 1 / 1                                           | 1 / 2                        | 1 / 2                       |
| LiDAR streaming context / chunk   | 21 / 21                                         | 21 / 21                      | 21 / 20                     |
| Sample / per-view caption dropout | 0.1 / 0                                         | 0.1 / 0.1                    | 0.1 / 0.1                   |
| Whole-LiDAR dropout               | 0                                               | 0                            | 0.1                         |

Ratios refer to logical samples, not equal pixels/tokens/work. Keep independent
camera captions, physical IDs, clocks and the [data contract](data-contract.md).
For the complete eleven-camera bundle, the reference rig distribution is
60% seven-camera / 30% eleven-camera / 10% other eligible subsets at 480p;
720p uses 90% six-of-seven / 10% other eligible subsets. Smaller rigs renormalize
eligible choices and are a data adaptation.

There is no independent whole-RGB modality dropout: cameras are selected by this
rig/subset policy. Per-view caption dropout removes description bodies while
retaining camera headers; sample-level CFG dropout blanks every view's caption
together. The task system prompt remains. View-completion and control dropout
are disabled.

Use fused AdamW with base LR `1e-4`, betas `(0.9,0.99)`, epsilon `1e-8`,
weight decay `0.05` and gradient clipping `1.0`. Train generator/time/video
projections, LiDAR projections and the sensor table; keep the reference text and
other parameters frozen. Module selection is `moe_gen`, `time_embedder`,
`vae2llm`, `llm2vae`, `lidar2llm`, `llm2lidar`, `rig_view_embed`.

The scheduler is **LambdaLinear**, not cosine:

| Schedule              | Start LR | Peak | Floor | Warmup | Cycle |
| --------------------- | -------- | ---- | ----- | ------ | ----- |
| Separate              | 1e-10    | 8e-5 | 1e-5  | 2000   | 70000 |
| Shared joint schedule | 1e-5     | 4e-5 | 1e-5  | 1000   | 50000 |

Warmup and decay are linear; cycles include warmup. Joint phases share the saved
scheduler counter. The final 50k endpoint is absolute, not another 50k updates.
A shortened smoke changes the stop/save interval, not the scientific schedule.

EMA uses power-law profile `rate=0.1`, not fixed decay 0.1. For rate `s`, let
`g` be the largest real root of
`g^3+7g^2+(16-s^-2)g+12-s^-2=0`. With
`i=iteration+iteration_shift`, beta is 0 for i<1, otherwise
`(1-1/(i+1))^(g+1)`. Retain actual age when loading EMA; a fresh direct fine-tune
starts a new history. Do not add the offset again at the long-phase handoff.

Use BF16 compute/reduction, FP32 masters and selective activation checkpointing
(joint phases save marked operations). Reference context/FSDP degrees are 32/32.
At 512 ranks and accumulation 1, world-size / CP gives 16 data replicas; FSDP is
not another divisor. Verify actual packed clips per update on adapted hardware.

## Objective and conditioning

Use rectified-flow velocity prediction, `waver` video time sampling and uniform
loss weighting. Training shift is 5 at 480p, 10 at 720p; LiDAR-only uses 5.
Transfer samples mix T2V/I2V 50/50 with controls present throughout. Joint I2V
conditions one initial camera latent and the aligned measured LiDAR prefix.
Whole-clip LiDAR conditioning and view completion are off; control and prefix
dropout are zero. Whole-LiDAR dropout removes both target and control together.

Keep `local_item_mean` and `normalize_loss_by_active=False`: conditioned
elements have zero error but remain in the mean denominator. Inspect actual
batches, masks, losses and restored state before accepting an adapted recipe.

Use the [parameter guide](parameter-guide.md) for adaptation and
[post-training](../SKILL.md#export-and-check-inference) for the final export.
