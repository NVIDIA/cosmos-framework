# Start here and sampling notes for Auto inference

> **Skill note (not in the source):** these three blocks are from the bodies of the Transfer-Auto and Sim-Auto inference skills that this skill replaces (cosmos-framework `main` 45fecab), moved here word for word because SKILL.md has a size budget; only link targets were re-pointed, and the headings are new or carry the model name ("Start here" twice; "Sampling (Transfer-Auto)" stands over the four sampling paragraphs of "Sampling and output checks"). The rest of both bodies is in [SKILL.md](../SKILL.md). "The command" in the sampling notes is the [Transfer-Auto run block](../SKILL.md#run-transfer-auto).

## Start here (Transfer-Auto)

1. **Try the reference scene first.** Use its complete request and the released
   model to establish a baseline before substituting your own inputs.
2. **Choose and match your cameras.** Follow the
   [rig workflow](input-preparation.md#choose-reference-rig-or-custom-rig-generation), then use the
   [Camera Rig Explorer](../scripts/wsm/docs/camera-rig/guide.md) and
   [mapping guide](input-preparation.md#2-map-physical-cameras-to-learned-identities)
   to choose the closest supported identities for your rig. Keep each camera's
   RGB, WSM and caption together; `camera_key` selects its learned view embedding.
3. **Prepare camera-local captions.** Describe the requested scene in the
   [training-aligned caption style](caption-policy.md#keep-training-and-inference-captions-aligned),
   then check synchronization and agreement across views.
4. **Generate, inspect, then tune.** The reference recipe uses **720p**, guidance
   **3**, shift **10**, **35 steps** and the bundled [negative prompt](../assets/negative-prompt.txt).
   Read [what these settings do](input-and-execution.md#sampling)
   before changing them; inspect every camera, not only the front view.

## Sampling (Transfer-Auto)

The reference request generates 297 frames at 720p/30 FPS. For a short first
check, change the command to `--resolution 480` and prepare a 61-frame interval.
Match caption spans and control coverage to the selected interval:
caption-driven Transfer-Auto output length can differ from a top-level override.

Both 480p and 720p are supported. Use `aspect_ratio: "16,9"` for the
training-matched 832×480 or 1280×720 sizes; see
[resolution and aspect ratio](input-and-execution.md#resolution-and-aspect-ratio).
Transfer-Auto trains on approximately 13.3-second chunks and supports longer
RGB sequences through chunkwise continuation. Read the
[duration guide](input-and-execution.md#generation-length-and-long-rollouts)
before extending a request, especially for joint RGB/LiDAR.

For 10-FPS RGB-only inputs, follow the
[recommended inference configuration](input-and-execution.md#recommended-inference-configuration-for-10-fps-inputs):
keep the input frames unchanged and pass `--output-fps 10`. Model conditioning
defaults to 30 FPS; an explicit different `fps` is rejected. Generated RGB and
WSM previews are saved at 10 FPS without resampling. Joint RGB/LiDAR retiming is
unsupported; keep its output FPS at 30.

For separate per-view captions, route the supplied
[negative text](../assets/negative-prompt.txt) through `per_view_negative_prompt`,
not `negative_prompt`. Preserve the user's explicit settings in comparisons.
The [sampling guide](input-and-execution.md#sampling) explains the
guidance choices and joint LiDAR timing.

## Start here (Sim-Auto)

- **New rig:** compare it with the reference cameras in
  [Camera Rig Explorer](../scripts/wsm/docs/camera-rig/guide.md), then follow the
  shared [camera-to-embedding mapping](input-preparation.md#2-map-physical-cameras-to-learned-identities).
  Map each camera to the closest supported identity and keep RGB, WSM and caption
  assignments consistent. For domain/rig adaptation, post-train Transfer-Auto
  first and then [distill it](../../cosmos3-nano-sim-auto/post-training/SKILL.md).
- **Captions:** use the same
  [camera-local, training-aligned style](caption-policy.md#keep-training-and-inference-captions-aligned)
  as Transfer-Auto, checking appearance and motion across views and time.
- **Sampling:** use guidance **1**, control guidance **1** and the checkpoint's
  **four-step** schedule. Read [why Sim-Auto settings differ](input-and-execution.md#sim-auto-uses-a-distilled-schedule);
  guidance 1 still uses captions and controls, but a negative prompt adds no CFG effect.
- **Timing:** use aligned 30-FPS cameras for the reference recipe; joint Sim-Auto
  requires that camera rate. Read the [FPS guide](input-and-execution.md#source-timing-model-fps-and-playback-fps)
  before using lower-rate recordings. Changing playback speed does not correct
  model timing or resample controls.
- **Size and duration:** both 480p and 720p are supported; use the
  [training-matched landscape preset](input-and-execution.md#resolution-and-aspect-ratio).
  Sim-Auto supports long rolling generation, with qualitatively observed
  multi-minute rollouts. Read the [duration guide](input-and-execution.md#generation-length-and-long-rollouts)
  for quality expectations and full-horizon input requirements.
