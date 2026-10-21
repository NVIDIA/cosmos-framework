# Checkpoint, optimizer, scheduler, and EMA handoffs

## Cosmos Framework

Use the native TOMLs and commands in [post-training](../SKILL.md). Framework's
[Bimanual builder](../../../../../cosmos_framework/configs/simulation/common/bimanual_profiles.py)
owns the stage checkpoint wiring:

| Stage | Required environment inputs                                                      | Load behavior                                                                                          |
| ----- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| 0     | `COSMOS_SIM_BIMANUAL_GA_CHECKPOINT`                                              | Weight-only Cosmos3-Nano initialization; strict regular weights, fresh action modules, native EMA copy |
| 1     | `COSMOS_SIM_BIMANUAL_STAGE0_CHECKPOINT`                                          | Stage 0 root; new optimization state, preserve both weight trees                                       |
| 2     | `COSMOS_SIM_BIMANUAL_STAGE1_CHECKPOINT`                                          | Stage 1 root; initialize student and frozen teacher                                                    |
| 3     | `COSMOS_SIM_BIMANUAL_STAGE0_CHECKPOINT`, `COSMOS_SIM_BIMANUAL_STAGE2_CHECKPOINT` | Stage 0 reference/critic and Stage 2 student                                                           |

Supply completed iteration roots, not `model/`; the builder appends that child
for nested sources. Each new stage has `load_training_state=False`. Same-run
auto-resume takes precedence and restores model/EMA, optimizers, schedulers,
iteration and compatible loader state. Keep the run name, output root, bundle,
topology and parent checkpoints unchanged when testing resume.

Only fresh Stage 0 skips `net_ema.` and initializes EMA through the built-in
`InitializeEmaFromRegular` callback. Do not reinitialize EMA during resume. The [export command](../SKILL.md#export-and-check-inference)
uses the saved config and records the chosen Stage 3 weight variant.

## Choose the operation deliberately

- **Weight-only warm start:** load the stage's declared model inputs into a new
  run. Start fresh optimizer, scheduler, trainer, RNG and loader state.
- **Same-run resume:** restore model/EMA, both Stage 3 optimizer groups when
  applicable, schedulers, counters and compatible loader state.
- **Portable export:** export the Stage 3 student from its saved config,
  explicitly selecting regular or EMA weights. Teachers and critic are not
  inference components.

Do not use same-run resume across a stage boundary.

## Fresh four-stage chain

```text
Cosmos3-Nano -> Stage 0
Stage 0 -> Stage 1
Stage 1 -> Stage 2 student and frozen teacher
Stage 2 -> Stage 3 student
Stage 0 -> Stage 3 frozen reference and critic
Stage 3 student export -> inference
```

Use the four native TOMLs in the [main tutorial](../SKILL.md#continue-training).
No user-authored registration wrapper is needed. Stage 0's skip list is
`["net_ema.", "action2llm", "llm2action", "action_modality_embed"]`:
regular Cosmos3-Nano keys load strictly, the new action modules remain freshly initialized,
and `InitializeEmaFromRegular` copies the resulting regular tree into EMA.
Stages 1 and 2 have empty skip lists. Stage 2's frozen teacher uses Stage 1;
Stage 3 uses its separate student/reference inputs from the table above.
Inspect the resolved paths and `checkpoint.load_training_state=False` before
starting a new stage.

## Resume compatibility

Keep stage, run name, output root, model structure, optimizer, scheduler,
preprocessing, bundle contents, split, seed, rank topology and parent checkpoints
unchanged for a same-stage continuation. The native auto-resume path takes
precedence over the warm-start source and restores saved EMA; it must not copy
regular weights over that EMA again.

The native `LocalSimulationResumeCallback` saves consumed local-stream progress.
Check the restored iteration and loader state in a fresh process. A placement,
population, seed or preprocessing change needs a new run unless compatible
restoration is demonstrated. Keep parent checkpoints accessible: distillation
constructs its reference models before restoring the child.

Wait for all ranks and asynchronous checkpoint writers to finish before
restarting. Confirm that the resumed process restores the saved iteration,
optimizer, scheduler, EMA, and loader state.

Before relaxing a model load, classify each missing key as an intentionally
fresh action module, a training-only teacher/critic, omitted training state, or
an actual mismatch. Stop on an unexplained mismatch rather than disabling
strict loading.

## EMA and inference selection

EMA is training state and must survive same-run resume. A new stage uses the
weight trees selected by its native builder, not an inferred EMA-to-regular
conversion. The [student export](../SKILL.md#export-and-check-inference) uses
`--use-ema-weights`; record that choice with the exported artifact.

The canonical Stage 3 student export and released schema-5 inference model both
use four-step SDE schedules. Each export owns its exact ladder; follow its
matched [inference instructions](../../inference/SKILL.md).
