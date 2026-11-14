---
name: cosmos3-nano-transfer-auto-troubleshooting
description: Diagnose cosmos3-nano-transfer-auto failures, including distorted side/rear views, WSM alignment, caption mismatch, unexpected frame counts, missing LiDAR, out-of-memory errors, and checkpoint or resume failures. Use for focused checks and recovery, not automatic input remapping, retraining, causal Sim-Auto debugging, or quality certification.
---

# Troubleshoot Cosmos3-Nano-Transfer-Auto

> **Skill note (not in the source):** the inference checks of this skill (video timing or dimension problems, side/rear failures and the inference sections of the symptom guide) and its two stage-neutral sections ("Collect one failing example" and "Diagnose, correct, verify") moved to [Auto inference troubleshooting](../../cosmos3-inference-simulation-auto/references/troubleshooting.md#transfer-auto); training, checkpoint and resume symptoms stay here, in the symptom guide.

`cosmos3-nano-transfer-auto` generates driving videos and optional LiDAR from layout
controls and per-camera captions. Start with one failing example and choose the
matching section of the [symptom guide](references/symptom-guide.md).
Use the [shared model contract](../../cosmos3-inference-simulation-auto/references/model-contract.md) for compatibility checks;
this model does not use the causal Sim-Auto inference workflow.

## References

- [Symptom guide and checks](references/symptom-guide.md)
- [Model contract](../../cosmos3-inference-simulation-auto/references/model-contract.md)
