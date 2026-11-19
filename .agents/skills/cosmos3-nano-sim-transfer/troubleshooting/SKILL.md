---
name: cosmos3-nano-sim-transfer-troubleshooting
description: Diagnose Cosmos3-Nano-Sim-Transfer setup, checkpoint loading, media alignment, training-state and output failures in an existing workflow.
---

# Troubleshoot Cosmos3-Nano-Sim-Transfer

> **Skill note (not in the source):** the opening guidance, the Runtime, Model and runtime, Media and data and Inference sections of this skill, their failure-table rows and the support-request paragraph that closed "Training state" are in the Sim-Transfer inference skill's [troubleshooting reference](../../cosmos3-inference-simulation-transfer/references/troubleshooting.md). This file keeps "Training state".

| Symptom                                 | Reference                         |
| --------------------------------------- | --------------------------------- |
| Wrong restart state or training failure | [Training state](#training-state) |

## Training state

Choose full-state resume or a new weight-only start using
[checkpoint operations](../post-training/references/checkpoint-and-resume.md).
`load_ema_to_reg=True requires load_training_state=False` means an EMA warm
start was combined with optimizer continuation. Use EMA with fresh state for
a new run, or restore regular weights and their optimizer for resume.

If a new run resumes an old iteration, its existing run identity may have
selected a latest checkpoint. Use a new run name for an intended warm start.
Model-only exports do not restore optimizer or scheduler progress. For a
local Stage 0 data-cursor issue, inspect the composed callbacks for
`local_simulation_resume` and the corresponding saved state.

The [four-stage recipe](../post-training/references/local-training.md#framework-training-entry-points)
defines Stage 2 parent bindings and separate Stage 3 student/teacher paths.
Keep parent weights available across restart because construction loads them
first. If loss becomes nonfinite, capture the first failing operation and
compare its input, precision, masks, parents and optimization with the
[reference recipe](../post-training/references/canonical-recipe.md).
Stage 3 alternates one student update and five score-model updates in each
six-iteration cycle; its teacher remains frozen.

## References

- [Failure playbook](references/failure-playbook.md)
- [Checkpoint operations and resume](../post-training/references/checkpoint-and-resume.md)
- [Stage configuration and training launch](../post-training/references/local-training.md)
- [Reference training recipe](../post-training/references/canonical-recipe.md)
