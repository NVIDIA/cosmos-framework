# Four-stage reference recipe

## Cosmos Framework

Use the native recipes in [the Framework Bimanual builder](../../../../../cosmos_framework/configs/simulation/common/bimanual_profiles.py).
The [post-training tutorial](../SKILL.md) provides setup, HF-to-DCP preparation,
local data bindings, launch commands and handoffs. The four TOMLs live in
`examples/toml/sft_config/`:

| Stage | TOML                             | Reference budget               | Initialization                                                         |
| ----- | -------------------------------- | ------------------------------ | ---------------------------------------------------------------------- |
| 0     | `sim_bimanual_stage0_bidir.toml` | 35,000                         | Cosmos3-Nano regular weights, fresh action modules and native EMA copy |
| 1     | `sim_bimanual_stage1_tf.toml`    | 8,000                          | Stage 0 checkpoint root                                                |
| 2     | `sim_bimanual_stage2_tfdcm.toml` | 4,000                          | Stage 1 for student and frozen teacher                                 |
| 3     | `sim_bimanual_stage3_sfdmd.toml` | 1,500 total trainer iterations | Stage 2 student; Stage 0 reference/critic                              |

Launch with `cosmos_framework.scripts.train --sft-toml`. These TOMLs select the
native local loader and disable W&B. The model layout uses FSDP shard degree 8
and CP degree 4. Keep the caller's distributed placement consistent with it.
Stage 2 teacher guidance is 3; the current Framework Stage 3 has teacher guidance
1 and action-conditioned guidance 1. Its saved student schedule is four-step
SDE, with one student update per six trainer iterations and five critic updates.
These are the values selected by the Framework builder.

Fresh Stage 0 uses the built-in `InitializeEmaFromRegular`; same-run resume
preserves the saved EMA. Follow [Framework checkpoint handoffs](checkpoint-and-resume.md#cosmos-framework).
The released schema-5 inference example is also four-step; use its own saved
ladder rather than assuming it is identical to a user-trained export.

## Shared values

- action generation enabled. Sound generation disabled.
- `max_action_dim=64`, with 59 semantic values and five padded values.
- `num_embodiment_domains=32`, `state_ch=48`, and `state_t=300`.
- spatial compression 16 and temporal compression 4.
- AdamW-family optimization with a constant learning-rate multiplier after zero
  warmup.

The reference parallel degrees are data-parallel sharding 8 and context
parallelism 4. Placement is chosen by the user. Any placement must satisfy the
parallel divisibility and memory requirements and must preserve or explicitly
record the effective global batch.

## Stage 0 — representation adaptation

| Setting                           | Reference value                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------------- |
| attention                         | noncausal, two-way                                                                    |
| data window                       | User-selected aligned robot window: `T+1` observations and `T` actions                |
| packed-token budget               | 400,000                                                                               |
| activation checkpointing          | full                                                                                  |
| maximum iterations / save cadence | 35,000 / 100                                                                          |
| optimizer                         | LR `1e-5`, betas `(0.9, 0.99)`, weight decay `0.05`                                   |
| action-module LR multipliers      | 5 for `action2llm` and `action_modality_embed`                                        |
| flow objective                    | `train_time_weight="uniform"`, `train_time_video_distribution="waver"`, loss scale 10 |

Warm-start from the `Cosmos3-Nano` weights with fresh optimizer, scheduler,
trainer, random-number, and dataloader state. Initialize action-specific
input/output modules as defined by the stage config. Set
`checkpoint.load_training_state=False`, `keys_not_to_resume=[]`, and
`keys_to_skip_loading=["net_ema.", "action2llm", "llm2action",
"action_modality_embed"]`. After the strict regular-weight load, initialize the
fresh EMA tree from the loaded regular tree with the built-in
`InitializeEmaFromRegular` callback described in the
[checkpoint contract](checkpoint-and-resume.md). The converted Cosmos3-Nano checkpoint
does not provide EMA keys.

## Stage 1 — causal prediction

| Setting                           | Reference value                                      |
| --------------------------------- | ---------------------------------------------------- |
| attention                         | causal, three-way, clean-history conditioning        |
| predicted chunk                   | 2 latent frames                                      |
| data window                       | User-selected aligned robot window                   |
| packed-token budget               | 500,000                                              |
| activation checkpointing          | full                                                 |
| maximum iterations / save cadence | 8,000 / 100                                          |
| optimizer                         | LR `1e-5`, betas `(0.9, 0.999)`, weight decay `0.01` |

Warm-start weights from the Stage 0 checkpoint produced by the user's run.
Start fresh optimizer, scheduler, trainer, random-number, and dataloader state.
Set `checkpoint.load_training_state=False`,
`keys_not_to_resume=[]`, and `keys_to_skip_loading=[]`.

## Stage 2 — fixed-step distillation

| Setting                           | Reference value                                                                        |
| --------------------------------- | -------------------------------------------------------------------------------------- |
| attention and chunk               | causal, three-way. 2 latent frames                                                     |
| distillation ladder               | 48 total steps. Skip interval 1                                                        |
| frozen-reference guidance         | 3.0                                                                                    |
| sampler                           | shift-5, fixed four-step SDE schedule                                                  |
| flow objective                    | `train_time_weight="uniform"`, `train_time_video_distribution="waver"`, loss scale 100 |
| data window                       | Stage 1's user-selected aligned robot-window policy                                    |
| activation checkpointing          | selective                                                                              |
| maximum iterations / save cadence | 4,000 / 100                                                                            |
| optimizer                         | LR `2e-6`, betas `(0, 0.999)`, weight decay `0.01`                                     |

Use the exact Stage 1 checkpoint produced by the user's run as the weight-only
warm start for both trainable initialization and the frozen reference. Set both
the top-level checkpoint load path and
`model.config.teacher_forcing_dcm_teacher_load_from` to that artifact.
Set `checkpoint.load_training_state=False`, `keys_not_to_resume=[]`, and
`keys_to_skip_loading=[]` so training state starts fresh.

## Stage 3 — rollout distillation

| Setting                                | Reference value                                                                            |
| -------------------------------------- | ------------------------------------------------------------------------------------------ |
| attention and chunk                    | causal, three-way. 2 latent frames                                                         |
| data window                            | User-selected aligned robot window                                                         |
| activation checkpointing               | selective                                                                                  |
| rollout schedule                       | step counts `[4]`, one matching shift-5 SDE schedule, maximum-mode selection               |
| replay                                 | block 1. Conditioned frames excluded. One backward-gradient step                           |
| teacher guidance, no actions in batch  | `teacher_guidance=1.0`                                                                     |
| teacher guidance, any actions in batch | `action_conditioned_teacher_guidance=1.0`                                                  |
| update cadence                         | `student_update_freq=6`: one generation and five critic updates per six trainer iterations |
| flow objective                         | uniform time, uniform video distribution, scale 10                                         |
| maximum iterations / save cadence      | 1,500 / 100                                                                                |
| generation optimizer                   | LR `1e-6`, betas `(0, 0.999)`, weight decay `0.01`                                         |
| critic optimizer                       | LR `2e-7`, betas `(0, 0.999)`, weight decay `0.01`                                         |

Warm-start the generation weights from the user's Stage 2 checkpoint. Initialize
the frozen reference and critic from the user's Stage 0 checkpoint. Export the
Stage 3 generation weights for inference. Set
`checkpoint.load_training_state=False`. Stage 3 uses
`model.config.student_load_from` and `model.config.teacher_load_from` rather
than a top-level warm-start path.

Iterations divisible by six are generation/student updates; the other five
iterations in each cycle update the critic.

## Runtime requirements

Keep `LocalSimulationResumeCallback` enabled for same-run resume. Preserve the
bundle, topology, parent checkpoints, run name, and output root when restarting
a stage.

Stage 0 and Stage 1 enable tokenizer AOT compilation after three iterations.
Every rank loads the compiled packages, so set `IMAGINAIRE_OUTPUT_ROOT` to an
absolute, writable path visible at the same location on every node. Use a
unique `RUN_NAME` for each stage.

The native loader uses seed 42. Stage 0 packs samples to a token budget;
Stages 1–3 set `max_samples_per_batch=1`. Compute effective global batch from
the emitted sample count, data-parallel world size, and gradient accumulation;
do not count context-parallel copies as distinct samples.

## Stage 3 four-step schedule

The native Stage 3 builder retains `sf_simulation_steps_per_frame=[4]` and one
matching backward schedule. Bind the user's Stage 2 student and Stage 0
reference/critic through the environment inputs in the
[checkpoint contract](checkpoint-and-resume.md#cosmos-framework).
Changing the inference CLI's `--num-steps` does not rewrite the saved schedule.

## Runtime defaults to record

The checked-in source profiles supply these defaults. Save their resolved
values with each run, including user adaptations.

| Contract                      | Source-backed value or collection step                                                                                                                                                                                                                                     |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Trainable key selection       | Stage 0-2 and both Stage 3 optimizer groups select `moe_gen`, `time_embedder`, `vae2llm`, `llm2vae`, `action2llm`, and `action_modality_embed`                                                                                                                             |
| Fresh modules                 | Stage 0 initializes `action2llm`, `llm2action`, and `action_modality_embed`. `llm2action` is omitted from the forward-dynamics optimizer                                                                                                                                   |
| Frozen modules                | Stage 2 frozen teacher comes from Stage 1. Stage 3 frozen reference comes from Stage 0. Stage 3 critic also starts from Stage 0 and is optimized separately                                                                                                                |
| Data source                   | The public local recipe accepts one robot source. If a user bundle contains multiple supported robot strata, their captured relative weights are normalized within the robot data                                                                                          |
| Data exposure                 | Token packing and source lengths make samples per optimizer step variable. Record actual emitted counts by source                                                                                                                                                          |
| Microbatch and accumulation   | Native local Stage 0 packs by token budget; Stages 1–3 set `max_samples_per_batch=1`. Record the composed accumulation and emitted batch, rather than equating a raw record with an optimizer batch                                                                        |
| Global-batch arithmetic       | Count each distinct sample once across data-parallel replicas and accumulation. For a constant packed count `m`, use `m * runtime_data_parallel_world_size * trainer.grad_accum_iter`. Count context-parallel copies once and retain actual emitted counts when `m` varies |
| EMA                           | Base `ema=power` enables EMA with rate `0.10` and iteration shift `0`. Resolve the stage's final config and retain its generation-net variant explicitly                                                                                                                   |
| Seeds                         | The native local loader is constructed with seed `42`. Save the composed trainer seed, loader seed, bundle identity and shuffle state separately                                                                                                                           |
| Precision                     | Base FSDP defaults set master dtype `float32` and reduction dtype `bfloat16`. Save the final per-stage precision/autocast/scaler settings from composition rather than assuming every model type uses the same override                                                    |
| Scheduler                     | Stage 0-2 set `f_max=[1.0]`, `f_min=[1.0]`, `warm_up_steps=[0]`, and cycle length equal to that stage's `max_iter`. Stage 3 sets `scheduler={}` and uses its DMD optimizer groups                                                                                          |
| Gradient clipping and scalers | Stage callbacks set clip norm `1.0`. Root grad-scaler default is disabled. Preserve any composed model-specific settings                                                                                                                                                   |
| Source artifacts              | Keep dataset versions, source-entry ratios, caption selection, statistics digest, tokenizer identity, and split definition in the resolved run config; exclude credentials and access tokens                                                                               |
| Checkpoint destination        | Caller-owned writable storage configured through the selected checkpointer. Record the root, iteration, contents, and selected weight variant                                                                                                                              |
| Hardware/topology             | The worked command launches eight processes with FSDP shard degree 8 and CP degree 4. Confirm memory fit and the resolved layout in the target allocation; eight processes are not a universal memory minimum                                                              |

Save the fully resolved configuration with each run. Use the
[checkpoint contract](checkpoint-and-resume.md) for the full-state versus
weight-only boundary.
