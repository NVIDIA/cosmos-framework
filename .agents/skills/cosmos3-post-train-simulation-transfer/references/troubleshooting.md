# Troubleshoot Cosmos3-Nano-Sim-Transfer

> **Skill note (not in the source):** this file holds the training-state part of the Sim-Transfer troubleshooting sub-skill, moved verbatim (cosmos-framework `main` 45fecab): the "Wrong restart state or training failure" symptom row, the section "Training state" and the "Failure playbook" entry and the three training entries of "References" of its SKILL.md, then the opening sentence and the "Training state" table of its failure playbook under "Common failures". Changes: link targets were re-pointed; the links between the two source files are in-file anchors (the playbook's link to the troubleshooting skill lands on "Training state", the part of that skill this file holds); the two playbook headings dropped one level ("Common failures" and its "Training state"). The opening guidance, the Runtime, Model and runtime, Media and data and Inference sections, the "Runtime, input and inference" failure rows and the support-request paragraph that closed "Training state" in the source are in the Sim-Transfer inference skill's [troubleshooting reference](../../cosmos3-inference-simulation-transfer/references/troubleshooting.md).

| Symptom                                 | Reference                         |
| --------------------------------------- | --------------------------------- |
| Wrong restart state or training failure | [Training state](#training-state) |

## Training state

Choose full-state resume or a new weight-only start using
[checkpoint operations](checkpoint-and-resume.md).
`load_ema_to_reg=True requires load_training_state=False` means an EMA warm
start was combined with optimizer continuation. Use EMA with fresh state for
a new run, or restore regular weights and their optimizer for resume.

If a new run resumes an old iteration, its existing run identity may have
selected a latest checkpoint. Use a new run name for an intended warm start.
Model-only exports do not restore optimizer or scheduler progress. For a
local Stage 0 data-cursor issue, inspect the composed callbacks for
`local_simulation_resume` and the corresponding saved state.

The [four-stage recipe](local-training.md#framework-training-entry-points)
defines Stage 2 parent bindings and separate Stage 3 student/teacher paths.
Keep parent weights available across restart because construction loads them
first. If loss becomes nonfinite, capture the first failing operation and
compare its input, precision, masks, parents and optimization with the
[reference recipe](canonical-recipe.md).
Stage 3 alternates one student update and five score-model updates in each
six-iteration cycle; its teacher remains frozen.

## References

- [Failure playbook](#common-failures)
- [Checkpoint operations and resume](checkpoint-and-resume.md)
- [Stage configuration and training launch](local-training.md)
- [Reference training recipe](canonical-recipe.md)

## Common failures

Use the [troubleshooting skill](#training-state) to locate the failing operation.

### Training state

| Symptom                                                   | Cause                                                              | Correction                                                                                                                    |
| --------------------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `load_ema_to_reg=True requires load_training_state=False` | EMA warm start combined with optimizer continuation                | Use EMA with fresh state for a new run, or regular weights with matching full state for resume                                |
| New run starts at an old iteration                        | Existing identity selected its latest checkpoint                   | Choose a new run name for a warm start and preserve the old run                                                               |
| Local Stage 0 loses its data cursor                       | Missing `local_simulation_resume` callback or saved callback state | Follow [local training](local-training.md) and restore the callback state                                                     |
| Restart loses optimizer or learning-rate progress         | Model-only export used as a resumable checkpoint                   | Resume from a completed full-state DCP                                                                                        |
| Nonfinite loss or stalled score training                  | Input, precision, objective or optimization mismatch               | Capture the first failing operation and compare the relevant reference settings; Stage 3 needs both student and score updates |
