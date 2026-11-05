# Using the scenario-to-RDS-HQ skill

Use this skill with a coding agent to convert driving scenarios into RDS-HQ and
render WSM 4.0.1 control videos for Transfer-Auto or Sim-Auto (`version=v4`).

You supply scene geometry, tracks, calibrated cameras and poses; the agent
converts and checks them. An RGB video or already-rendered WSM MP4 alone cannot
provide the raw scene needed for conversion or camera retargeting. Keep your
confidential inputs outside the repository. The renderer runs locally with
cloud upload disabled.

## Quick start

Use a coding agent with the [skill](../skills/scenario-to-rds-hq/SKILL.md).
If it is not installed in the agent's catalog, include that file path in your
request. The agent reads the detailed references during conversion.

Provide this repository, a representative clip, and a new output directory.
A custom format definition or existing reader for your own scenarios helps but
is not mandatory: include class and enum meanings, axes and units, transform
direction, timestamp units, and camera calibration with distortion. RGB is optional.

From the standalone `wsm` directory, use either prompt:

**Minimal:**

> Use the `scenario-to-rds-hq` skill to convert `<source-path>` into RDS-HQ
> under `<output-path>`.

**With extra details:**

> Use the `scenario-to-rds-hq` skill to convert `<source-path>` into RDS-HQ
> under `<output-path>`. Use any supplied format definition or reader to interpret
> the source. Validate the conversion and render WSM 4.0.1 videos with
> `version=v4`, 30 FPS and a first 297-frame window when source coverage permits.
> Save the conversion script, class mapping report, validation results, and visual
> review beside the output.

Validate one short clip and camera before expanding. If format documentation is
unavailable, have the agent inspect the source and record unresolved assumptions.
Rendering also needs the [GPU/EGL environment](../README.md#install); conversion
and archive checks can run without it. Use the actual source timing: the renderer
samples an existing pose timeline and cannot upsample 10 Hz poses to 30 Hz.
Any required timeline resampling belongs in the conversion, with tracks, poses
and signal states kept aligned. Do not change metadata alone to claim 30 FPS.

## Camera retargeting

By default, the skill finishes by retargeting the converted scene to the NVIDIA
Hyperion 8 reference rig: seven F-theta cameras at 1924×1084 and four F-theta
fisheyes at 1720×1080. The skill uses the rig's intrinsics and combines its
extrinsics with the vehicle poses to write new camera poses in a separate
dataset. Scene annotations and timing stay unchanged.

Inspect the rig in the offline [Camera Rig Explorer](camera-rig/guide.md).
Retargeting can reduce geometry mismatch with the released models, but cannot
create missing annotations or guarantee quality on an unfamiliar scene.

If you plan to post-train with your own data, skip Hyperion retargeting and use
your own camera rig.

To keep the original cameras, add: **“Keep the source rig; skip Hyperion
retargeting.”** The agent may need the source vehicle-frame definition or camera
extrinsics to establish the vehicle poses.
Do not pair retargeted controls with unchanged source-camera RGB for training
or initial-image conditioning; the viewpoints no longer match.

## Continue to Auto inference

In a full Framework checkout, the
[rendering notebook](../../generate_wsm.ipynb)
connects the steps below. If you installed only the standalone renderer, keep
the rendered controls and complete this handoff in a full Framework environment.
The notebook accepts F-theta calibration, supported Auto camera keys and 30 Hz
poses. For other cameras or controls without the raw archives, use the
[custom-rig handoff](#custom-rigs-and-already-rendered-controls) below.

1. **Render and inspect the selected views.** Use the 30 FPS / 297-frame example
   after preparing sufficient synchronized source coverage. Retain the actual
   camera keys, timestamps and calibration. The notebook does not need captions
   to render. Its default `RESOLUTION_SPEC="resize_1280x720"` matches the README;
   select `source` for unresized controls. Match the selected cameras and render
   profile before reusing existing output.
2. **Prepare captions from those rendered views.** Use the existing
   [per-view captioning guide](../../../../../../cosmos3-nano-transfer-auto/inference/references/caption-guide.md).
   Supply the desired time of day, weather, road condition and geographic setting;
   WSM colors do not establish those properties. Prefer timestamped captions,
   and keep shared objects' appearance and motion consistent across cameras.
   After retargeting, caption the new views rather than reusing source-view text.
   Request one `<camera_key>.json` sidecar per camera, not only `captions.json`.
   Existing captions can be reviewed without silently rewriting them.
3. **Export the native request.** The notebook checks caption coverage and writes
   a `request.json` with matching control and caption paths. It retains source
   captions and writes interval-specific copies beside the request. Framework
   adds camera/FOV and task prefixes automatically; leave them out of the bodies.
4. **Generate RGB.** In the native Transfer-Auto or Sim-Auto notebook, set
   `CUSTOM_REQUEST` to the exported file and `MODE = "text2video"`. The notebooks
   apply the appropriate model-specific sampling settings.

Initial RGB must match the rendered views and start time. Joint LiDAR generation
requires separately prepared, aligned HD-map range-map controls. Camera rendering
does not produce those controls; never borrow them from an unrelated sample.
For post-training captions, describe the actual recorded RGB targets using the
[training data guide](../../../../../../cosmos3-nano-transfer-auto/post-training/references/data-contract.md#training-captions)
instead of assigning hypothetical appearance to a future generated scene.

### Custom rigs and already-rendered controls

The standalone renderer supports both `camera_type=pinhole` and
`camera_type=ftheta`, with the camera names from your source calibration. Keep
those names and geometry intact; the notebook's narrower convenience path is
not a restriction of the renderer.

After rendering, use the existing
[Auto input-preparation guide](../../../../../../cosmos3-nano-transfer-auto/inference/references/input-preparation.md#2-map-physical-cameras-to-learned-identities)
to select the closest supported model identity for each physical camera, with
the [Camera Rig Explorer](camera-rig/guide.md) and your actual calibration.
Use the guide's [native request schema](../../../../../../cosmos3-nano-transfer-auto/inference/references/input-and-execution.md#camera-request):
`camera_key` selects the model identity, while `control_path` and `caption_path`
can retain your original filenames. No file renaming or new adapter is needed.
For example, a verified front-wide match may use
`camera_key="camera_front_wide_120fov"` with `control_path="wsm/front_camera.mp4"`.
Do not assign that key merely because the filename says "front".

Prepare camera-local captions with the same guide and selected time interval.
For controls-only generation, use `model_mode="text2video"`, omit `vision_path`,
disable `condition_video_as_image`, and set `num_conditional_frames=0`.
Pass the request as `CUSTOM_REQUEST` with `MODE="text2video"` in either native
Auto notebook. This also works for already-rendered controls without RDS-HQ
archives. Inspect the effective crop, timing and per-view text before generation.

An approximate camera-key match does not change the source projection or guarantee
quality. Prefer reference-rig retargeting when scene coverage permits; for paired
domain adaptation retain the recorded rig and follow the post-training guide.

## Visual review

Watch the rendered videos and check that they provide a recognizable, abstract
representation of the driving scene. Look for the expected road layout, actors,
signs, and signals, with consistent positions, motion, and state changes over time.

Use the [WSM4 guide](wsm4.md#visual-conventions) to interpret what each color,
symbol, and line style represents.

When RGB is available, compare it with the original camera renders at matching
times, before retargeting. Hyperion views have different camera positions and
projection; review them against the source scenario and annotations. RGB remains
optional.

## Pay attention to class mapping

Review the class mapping report and notes saved with the conversion. They should
list source and target classes, evidence, assignment methods, and counts.

Focus on classes that were merged, inferred, defaulted, or omitted. Check
questionable mappings against the source format definition and output
annotations, including the meaning of numeric enums. Investigate unexplained
`Other` mappings and missing components. If these notes are missing, ask the
agent to provide them.

## Optional reading

- [WSM4 rendering](wsm4.md): start here when reviewing the output to understand
  its visual style, colors, and symbols.
- [The skill](../skills/scenario-to-rds-hq/SKILL.md): details of the conversion
  and validation workflow.
- [RDS-HQ data format](rds-hq.md): consult as needed for supported classes,
  coordinates, camera models, timing, and required fields.
