# Common failures

Use the [troubleshooting skill](../SKILL.md) to locate the failing operation.

> **Skill note (not in the source):** the "Runtime, input and inference" rows are in the Sim-Transfer inference skill's [troubleshooting reference](../../../cosmos3-inference-simulation-transfer/references/troubleshooting.md#runtime-input-and-inference).

## Training state

| Symptom                                                   | Cause                                                              | Correction                                                                                                                    |
| --------------------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `load_ema_to_reg=True requires load_training_state=False` | EMA warm start combined with optimizer continuation                | Use EMA with fresh state for a new run, or regular weights with matching full state for resume                                |
| New run starts at an old iteration                        | Existing identity selected its latest checkpoint                   | Choose a new run name for a warm start and preserve the old run                                                               |
| Local Stage 0 loses its data cursor                       | Missing `local_simulation_resume` callback or saved callback state | Follow [local training](../../post-training/references/local-training.md) and restore the callback state                      |
| Restart loses optimizer or learning-rate progress         | Model-only export used as a resumable checkpoint                   | Resume from a completed full-state DCP                                                                                        |
| Nonfinite loss or stalled score training                  | Input, precision, objective or optimization mismatch               | Capture the first failing operation and compare the relevant reference settings; Stage 3 needs both student and score updates |
