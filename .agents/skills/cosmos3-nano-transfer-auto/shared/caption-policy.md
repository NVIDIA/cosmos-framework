# Caption preparation for cosmos3-nano-transfer-auto

## Keep training and inference captions aligned

Use the same camera-local descriptive style, level of detail and time convention
when preparing post-training labels and inference prompts. This reduces the gap
between training text and inference prompts.

- Prefer **timestamped descriptions** of the scene and its changes. Plain dense
  paragraphs also work, but timestamped captions better match the model's training
  style and have empirically produced better inference results. Follow the
  [caption guide](../inference/references/caption-guide.md) for formatting; retain
  the native per-camera sidecar schema and synchronized frame bounds.
- Ground object layout and motion in synchronized WSM. For inference, take the
  requested **time of day, weather, road condition and geographic setting** from
  the user or scenario metadata; WSM class colors cannot supply those facts.
  For training, captions must describe the actual paired RGB/recording instead.
- Check all views together: the same actor must have compatible appearance and
  physical motion, and scene conditions must agree. Different screen-space motion,
  reflections and visibility are normal across cameras; do not copy front-view
  prose to every camera or require every view to show the same landmarks.
- Store scene-description bodies only. Keep the recipe's camera/task formatting:
  Framework supplies the rig/current-camera context and WSM task instruction.
  Manually duplicating those prefixes changes the model-facing text.

Changing the requested weather for inference is valid. Changing only a training
caption to rainy night while leaving a sunny RGB target unchanged creates
contradictory supervision. Inspect control/target alignment before packaging
training data, and use a held-out scene to compare the resulting model.

## Author or review captions

For inference prompts, follow the bundled [caption guide](../inference/references/caption-guide.md)
for generation or upsampling.

Pass synchronized WSM media, verified physical view mapping/renderer semantics,
the selected interval, the four requested properties with their sources, existing
caption bodies when relevant, and the output directory. Defaults are long,
timestamped captions, 300–400 body words per view across all blocks, for all
supplied cameras. Actual WSM inspection is required; extra visual review is not.

For a check-only request, use
[input-preparation guide](../inference/references/input-preparation.md).
Report contradictions and suggested corrections without invoking a rewrite.
An existing `captions.json` is a camera-to-body mapping, not an inference request.
Check the release reader and resolved text before packaging sidecars.

For training labels, follow [target-grounded captions](../post-training/references/data-contract.md#training-captions)
instead. Plausible appearance assignment for future generated RGB must not become
invented descriptions of unchanged real RGB training targets.

## Caption text and automatic prefixes

Save only the camera-local scene description in each sidecar's `caption` field,
with timestamps when using the recommended format. Framework adds two distinct
pieces of model-facing context in the standard per-view Auto workflows:

- **Camera-rig and current-view description:** the selected cameras, their
  directions, lens types and nominal fields of view, followed by the identity
  of the camera whose scene description follows. This comes from the selected
  camera keys and configured camera attributes, not visual detection.
- **WSM-control task instruction:** a system prompt explaining that WSM controls
  determine the controlled objects and road layout, while captions describe
  appearance and unconstrained background details. The joint camera/LiDAR task
  also describes the HD-map control and camera–LiDAR alignment.

Training adds these through the native caption formatter and text-tokenizer
transforms; inference adds them through its caption loader and model-tokenization
path. The recipe and inference route select the appropriate task template;
single-sensor training phases use their own formatting. Duration, FPS and
resolution metadata are also supplied by the configured formatters.

Do not copy these headers or system instructions into the caption files, and do
not use a top-level scene prompt to replace per-camera bodies. Keep the supplied
recipe/model caption settings. For a custom rig, verify camera-key mapping and
calibration: an automatically added nominal FOV is not proof that the physical
camera matches it.
