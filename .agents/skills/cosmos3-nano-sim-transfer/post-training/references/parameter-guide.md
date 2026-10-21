# Adapt Cosmos3-Nano-Sim-Transfer training to your task

Start with the [Stage 0–3 walkthrough](local-training.md). The
[method reference](canonical-recipe.md) lists the full defaults for each stage.
Use the settings below to plan changes while preserving checkpoint and data
compatibility.

## Preserve the model and data interfaces

| Setting                  | Requirement                                                                                                            | Why it matters                                                          |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Input and target         | One aligned edge, blur, depth or segmentation video paired with RGB and a caption from the same interval               | Preserves timing, camera geometry and the meaning of each control       |
| Latent geometry          | Temporal compression 4, spatial compression 16, 48 channels, spatial patch size 2                                      | Matches the tokenizer, packing layout and checkpoint shapes             |
| Causal chunk             | One latent time per chunk in Stages 1–3 and causal inference                                                           | Keeps training and inference history boundaries consistent              |
| Attention visibility     | Stage 0 attends across the whole window. Stages 1–3 see controls through the current time and RGB strictly in the past | Prevents future RGB from entering the causal student's history          |
| Parent weights           | Base regular weights → Stage 0 EMA → Stage 1 EMA → Stage 2 EMA → final student EMA                                     | Initializes each stage with the model that supplies its training target |
| Stage 3 batch layout     | One pair per rank, context-parallel degree 1, equal rollout lengths and synchronized prefix lengths                    | Aligns distributed forward/backward schedules                           |
| Stage 3 trainable models | Student and score model have separate optimizers; teacher is frozen                                                    | Implements distribution-matching training                               |

A new stage starts fresh optimizer, scheduler, trainer and data state.
Same-stage resume restores the full checkpoint. See the
[checkpoint guide](checkpoint-and-resume.md) for the exact parent bindings.

## Choose data and conditioning

Prepare held-out controls and prompts before tuning. Inspect representative
RGB/control pairs for timing, geometry, depth polarity, segmentation palette
and caption agreement. Use the [data contract](../../shared/data-contract.md)
when adding a new dataset.

| Setting                  | Recipe value                                                                                     | Practical effect                                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| Resolution and lengths   | Stage 0: 480 tier up to 201 frames, 720 up to 149, minimum 5. Stages 1–3: 480 tier, 9–201 frames | Longer clips and larger canvases increase memory and computation. RGB lengths follow `1+4n`             |
| Stage 0 stream selection | 480:720 weights 4:2                                                                              | Produces an expected 2:1 stream-selection frequency; packed example counts vary with token cost         |
| Local sample selection   | Longest eligible caption window per clip, grouped by latent length                               | Keeps rollout lengths aligned across ranks                                                              |
| Control mixture          | Equal edge, blur, depth and segmentation weights                                                 | Balances exposure to the four control types                                                             |
| Caption mixture          | Retained captions: 93% structured, 7% same-window dense prose                                    | Preserves structured scene detail while teaching dense prompts                                          |
| Text dropout             | 0.1 in Stages 0–2; 0 in Stage 3                                                                  | Trains the earlier stages' unconditional text branch                                                    |
| RGB prefix               | 0, 1 or 2 latent frames with probabilities 0.7, 0.2 and 0.1                                      | Supplies known RGB history. Prefix positions are excluded from generation loss                          |
| Teacher guidance         | 3 in Stages 2–3, with the configured negative prompt and clip metadata                           | Controls the teacher signal distilled into the student                                                  |
| Noise shift              | Stage 0: 5 at 480, 10 at 720. Stages 1–3: 5                                                      | Changes the training-noise distribution; the distilled four-step sampling schedule is stored separately |

Change data and conditioning settings together when they share a meaning—for
example, update caption intervals when changing temporal sampling. Compare
control adherence, prompt meaning, appearance and motion across the full clip.

## Tune optimization and execution

| Setting                  | Recipe value                                                                                                                     | Adjustment guidance                                                                                                           |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Learning rate            | Stage 0 `1e-4` with schedule multiplier up to 0.4; Stage 1 `1e-5`; Stage 2 `2e-6`; Stage 3 student `1e-6`, score model `2e-7`    | Begin at these values. Compare finite losses, gradients and held-out outputs after changing effective batch or update cadence |
| Adam and schedule        | Stage-specific betas, decay and schedules in the [method reference](canonical-recipe.md)                                         | Treat learning rate, batch size and schedule duration as coupled settings                                                     |
| Stage 3 update ratio     | Five score-model updates per student update; both loss multipliers 1                                                             | Count both optimizers separately when comparing training duration                                                             |
| Trainable modules        | Stage 0 selects ten parameter groups; Stages 1–3 select the generation expert, time embedding and video input/output projections | Changing the selected modules changes which model capabilities can adapt                                                      |
| Parallelism              | Parameter-shard degree 8, context parallelism 1, accumulation 1                                                                  | World size must be divisible by the chosen shard degree. Keep Stage 3 context parallelism at 1                                |
| Packing                  | Stage 0: 74,000-token budget. Stages 1–3: one emitted pair per rank                                                              | Measure emitted examples and tokens per update when changing topology or accumulation                                         |
| Activation checkpointing | Stage 0 selective; Stages 1–3 full                                                                                               | Trades recomputation for activation memory                                                                                    |
| Compilation and replay   | Stage 0 compiles the language component. Stages 1–3 disable model compilation. Stage 3 replays one latent at a time              | Changes memory and execution cost; replay preserves history for the student gradient                                          |
| Precision                | BF16 compute, FP32 master weights; Stage 0 BF16 gradient reduction, Stages 1–3 FP32 reduction                                    | Retain FP32 master and optimizer states for continued training                                                                |
| Stage 3 history cache    | Unquantized, window 51 latent times, first 1 retained as a sink                                                                  | Bounds training history. The default short inference recipe uses full history with no sink                                    |
| EMA                      | Power EMA rate 0.1, iteration shift 0                                                                                            | Select student EMA for stage handoffs and final inference                                                                     |
| Seeds                    | Trainer 0, loader 42                                                                                                             | Keep fixed for comparisons and same-run data continuation                                                                     |
| Saves and logging        | Stage-specific intervals in the [method reference](canonical-recipe.md)                                                          | Choose intervals for recovery and monitoring needs; wait for asynchronous writes to complete                                  |
| Local loader             | Zero workers, prefetch 1                                                                                                         | Increase workers only after measuring throughput and host-memory use; preserve loader settings when resuming                  |

For Stages 1–3 at context-parallel degree 1:

```text
examples per update = one emitted pair × data-parallel world size × accumulation
```

Data-parallel world size includes parameter-shard and replication ranks. Stage 0
uses variable token packing, so measure its emitted batch directly. If changing
context parallelism in Stages 0–2, count distinct examples per update in the
new layout. Changing accumulation also changes data draws per saved iteration;
start a new run for a comparison with different batch settings.

Choose checkpoints using held-out outputs, then follow the
[checkpoint guide](checkpoint-and-resume.md) to resume or hand the model to
inference.
