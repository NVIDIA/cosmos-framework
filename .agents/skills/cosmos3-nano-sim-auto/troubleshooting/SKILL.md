---
name: cosmos3-nano-sim-auto-troubleshooting
description: Diagnose Cosmos3-Nano-Sim-Auto input, causal inference, distillation, checkpoint and export failures in Cosmos Framework. Check camera mapping, captions, rolling metadata and stage-state handoffs without silently changing the recipe.
---

# Troubleshoot Cosmos3-Nano-Sim-Auto

Collect one failing request or training step, runtime/source revision, resolved
config, artifact identity, logs and relevant outputs. Keep confidential inputs and
credentials out of shared reports.

| Symptom                                        | First checks                                                                                                                                                                     |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Side/rear geometry mismatch                    | Shared [input preparation](../../cosmos3-nano-transfer-auto/inference/references/input-preparation.md): physical camera mapping, synchronized WSM, crop and actual per-view text |
| Short output or caption transition failure     | [Input settings](../inference/references/input-preparation.md#request-fields): total horizon, source coverage, sidecar bounds and first-chunk selection                          |
| Missing metadata / HF load rejected            | [Export](../post-training/references/checkpoints.md#export-the-student): regular student, rolling profile, per-view text and matched tokenizers                                  |
| Unexpected CFG or sampler behavior             | Fixed four-step student; use the guidance-1 recipe, not bidirectional RF step/shift settings                                                                                     |
| Missing or misaligned LiDAR                    | HD-map controls versus measured prefix, 30/10 Hz clocks, source IDs; causal measured prefix is at most one sweep                                                                 |
| Missing training camera/caption/control fields | [Bundle mappings](../shared/data-contract.md#preserve-your-bundles-camera-mappings) at every stage; inspect decoded and formatted samples                                        |
| Wrong teacher or resume iteration              | [Checkpoints](../post-training/references/checkpoints.md): new stage versus complete same-stage resume                                                                           |
| OOM / compilation / distributed error          | Verify setup/mesh, then an authorized [smoke](../post-training/references/smoke.md); preserve architecture/objective                                                             |
| Only one SF-DMD phase appears                  | Six iterations include one student and five critic updates; inspect phase metrics and both optimizer histories                                                                   |

Preserve original inputs and the model's checkpoint/attention settings while
isolating the issue. Apply the smallest correction and check the affected result.
For reader errors, let bounded retries finish and check training progress. Stop
the affected run on a fatal failure, exhausted retries, invalid data or its time
limit; leave independent healthy jobs running and allow saves/exports to finish.
Report observations, the confirmed cause or remaining hypothesis, the correction
and checks performed.

## References

- [Shared camera and WSM input checks](../../cosmos3-nano-transfer-auto/inference/references/input-preparation.md)
- [Sim-Auto input and request settings](../inference/references/input-preparation.md)
- [Checkpoint handoffs, resume and student export](../post-training/references/checkpoints.md)
- [Training data and camera mappings](../shared/data-contract.md)
- [Short training, save and resume checks](../post-training/references/smoke.md)
