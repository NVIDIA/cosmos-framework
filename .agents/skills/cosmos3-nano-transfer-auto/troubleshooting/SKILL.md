---
name: cosmos3-nano-transfer-auto-troubleshooting
description: Diagnose cosmos3-nano-transfer-auto failures, including distorted side/rear views, WSM alignment, caption mismatch, unexpected frame counts, missing LiDAR, out-of-memory errors, and checkpoint or resume failures. Use for focused checks and recovery, not automatic input remapping, retraining, causal Sim-Auto debugging, or quality certification.
---

# Troubleshoot Cosmos3-Nano-Transfer-Auto

`cosmos3-nano-transfer-auto` generates driving videos and optional LiDAR from layout
controls and per-camera captions. Start with one failing example and choose the
matching section of the [symptom guide](references/symptom-guide.md).
Use the [shared model contract](../shared/model-contract.md) for compatibility checks;
this model does not use the causal Sim-Auto inference workflow.

## Collect one failing example

Collect the exact request/command, resolved config, checkpoint and weight variant,
per-view text actually consumed, camera-key→control mapping, selected time span,
one failing output with controls, and the error/log excerpt. Keep credentials and
confidential source locations out of shared diagnostics.

For a video timing or dimension problem, first compare the actual files:

```bash
: "${TRANSFER_AUTO_CONTROL:?Set an existing local WSM video path}"
: "${TRANSFER_AUTO_RGB:?Set its corresponding local generated RGB video path}"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration \
  -of json "$TRANSFER_AUTO_CONTROL"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration \
  -of json "$TRANSFER_AUTO_RGB"
```

Inputs are the two actual files; outputs are metadata reports. Success means both
probe and their timing/dimensions agree with the resolved request after documented
preprocessing. If `nb_frames` is missing, count decoded frames. Inspect synchronized
frames to check view identity and event alignment.

## Diagnose, correct, verify

Use the symptom guide to check likely causes, propose the smallest correction,
and verify the affected inputs and outputs afterward. Preserve the original
manifest, captions and outputs.

For side/rear failures, verify the camera mapping, controls and actual captions,
then compare a reference input with the failing scene, changing one suspected
factor at a time.

Return observations, hypotheses, the checks actually run, proposed corrections,
and remaining uncertainty. Pass a corrected request to the
[inference Skill](../inference/SKILL.md), or a state/topology issue to
[post-training](../post-training/SKILL.md).

## References

- [Symptom guide and checks](references/symptom-guide.md)
- [Model contract](../shared/model-contract.md)
