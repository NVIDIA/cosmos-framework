---
name: cosmos3-nano-sim-auto-troubleshooting
description: Diagnose Cosmos3-Nano-Sim-Auto input, causal inference, distillation, checkpoint and export failures in Cosmos Framework. Check camera mapping, captions, rolling metadata and stage-state handoffs without silently changing the recipe.
---

# Troubleshoot Cosmos3-Nano-Sim-Auto

> **Skill note (not in the source):** the inference rows of the table below (side/rear geometry; short output or caption transition; missing metadata / HF load rejected; CFG or sampler behavior; LiDAR) and the stage-neutral text around it (what to collect for one failing example; preserving inputs, applying the smallest correction and reporting) moved to [Auto inference troubleshooting](../../cosmos3-inference-simulation-auto/references/troubleshooting.md#sim-auto); the training rows and the reader-error rule stay here.

| Symptom                                        | First checks                                                                                                                              |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Missing training camera/caption/control fields | [Bundle mappings](../shared/data-contract.md#preserve-your-bundles-camera-mappings) at every stage; inspect decoded and formatted samples |
| Wrong teacher or resume iteration              | [Checkpoints](../post-training/references/checkpoints.md): new stage versus complete same-stage resume                                    |
| OOM / compilation / distributed error          | Verify setup/mesh, then an authorized [smoke](../post-training/references/smoke.md); preserve architecture/objective                      |
| Only one SF-DMD phase appears                  | Six iterations include one student and five critic updates; inspect phase metrics and both optimizer histories                            |

For reader errors, let bounded retries finish and check training progress. Stop
the affected run on a fatal failure, exhausted retries, invalid data or its time
limit; leave independent healthy jobs running and allow saves/exports to finish.

## References

- [Shared camera and WSM input checks](../../cosmos3-inference-simulation-auto/references/input-preparation.md)
- [Sim-Auto input and request settings](../../cosmos3-inference-simulation-auto/references/sim-auto-inputs.md)
- [Checkpoint handoffs, resume and student export](../post-training/references/checkpoints.md)
- [Training data and camera mappings](../shared/data-contract.md)
- [Short training, save and resume checks](../post-training/references/smoke.md)
