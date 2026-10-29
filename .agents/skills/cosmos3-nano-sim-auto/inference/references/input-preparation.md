# Prepare Sim-Auto inputs

Use the shared
[input-preparation guide](../../../cosmos3-nano-transfer-auto/inference/references/input-preparation.md)
for camera mapping, renderer semantics, synchronization and review of supplied
captions. For new captions it invokes
[av-per-view-caption](../../../cosmos3-nano-transfer-auto/inference/references/caption-guide.md).
Return the checked media inventory and per-view caption sidecars here.

That guide's physical checks apply to both models. Its Transfer-Auto guidance,
step count and execution command do **not** apply to Sim-Auto.

## Apply the causal rules

- Use one initial RGB image per camera, or the first frame of each RGB video,
  for the documented image-to-video example. Keep conditioning consistent across
  views; do not use this path for view completion.
- Keep explicit physical `camera_key` values and `caption_path` per view,
  with `multiview.load_caption_from_data=true`.
- Use caption bodies without duplicate camera/FPS/resolution headers. Cover the
  requested interval from frame zero. Crop/rebase all media together when
  selecting a different interval.
- Set `num_frames` to the rollout horizon. Small latent-chunk scheduling comes
  from the checkpoint and does not need a separate chunk-size flag.
- For joint generation, include `lidar.control_path` with prepared HD-map range
  data. An optional measured LiDAR prefix is at most one initial sweep on this
  causal path; longer prefixes are unsupported.
- Set guidance and control guidance to 1 and normalization false for the
  documented distilled example. Do not copy Transfer-Auto's CFG recipe.

Use `use_first_chunk_only=false` when the custom caption timeline has multiple
outer chunks; keep `first_caption_paragraph_only=false` to retain all paragraphs.
Supply sufficient controls and caption coverage rather than relying on terminal
padding. Existing compatible captions need no rewrite.

## Package the request

Follow the [request fields](#request-fields) and
hand the request to [inference](../SKILL.md). Record physical
camera IDs, selected bounds, caption files and unresolved alignment checks.

For training, use
[target-grounded captions](../../../cosmos3-nano-transfer-auto/post-training/references/data-contract.md#training-captions)
and the [Sim training-data contract](../../shared/data-contract.md), not a WSM-only
appearance-generation prompt as a label for recorded RGB.

## Request fields

Start from the shared [Framework camera request](../../../cosmos3-nano-transfer-auto/inference/references/input-and-execution.md).
Keep the camera entries, physical IDs, initial images, `wsm: {}`, per-camera
caption paths and `multiview.load_caption_from_data=true`.

For the provided scene, use `$AUTO_SAMPLE_ROOT/input-sim-auto.jsonl` directly:
it requests 720p and one caption chunk covering all 300 source frames. Its
`use_first_chunk_only: true` does not truncate that single interval. Do not add
`joint_generation` to the JSONL; the LiDAR block selects joint output.

When adapting the generic 480p camera example for Sim-Auto:

- Keep `model_mode: "image2video"`, `resolution: "480"`, `fps: 30`,
  `num_frames: 297` and `seed: 0`.
- Set `guidance: 1.0`, `control_guidance: 1.0`, `normalize_cfg: false`.
- Remove the bidirectional `num_steps` and `shift` entries. The student has its
  own fixed four-step schedule.
- Keep `condition_video_as_image: true` when conditioning from RGB videos.
- Use `use_first_chunk_only: false` and provide caption coverage through the
  complete horizon.
- Add `lidar: {"control_path": "lidar/hdmap_rangemap.tar"}` for joint output,
  or omit the LiDAR block entirely for camera-only output.

The shared guide documents the caption JSON shape and exclusive end bounds.
A 21-frame first check needs matching media and caption spans; 297 frames needs
full-length controls and captions. Do not change only a filename or FPS tag to
make mismatched recordings appear synchronized.

For RGB-only 10-FPS inputs, keep synchronized frames unchanged and set top-level
`output_fps: 10` (CLI `--output-fps 10`). Model conditioning defaults to 30 FPS;
explicit different `fps` values are rejected. This retimes saved RGB/WSM videos,
not model inputs or caption timestamps. Follow the shared
[timing guide](../../../cosmos3-nano-transfer-auto/inference/references/input-and-execution.md#source-timing-model-fps-and-playback-fps).
Joint RGB/LiDAR must retain 30-FPS output; joint retiming is not supported.

Optional measured LiDAR uses `lidar.condition_path` and a single initial sweep;
longer target prefixes and view completion are not supported by this causal
workflow. HD-map controls are not measured initial observations.

## Camera-local captions

Use the shared [AV caption guide](../../../cosmos3-nano-transfer-auto/inference/references/caption-guide.md)
and [input schema](../../../cosmos3-nano-transfer-auto/inference/references/input-and-execution.md).
Do not create a second caption policy for the distilled model.

Each physical camera consumes its own caption sidecar. Shared weather, time of
day, road surface and geographic setting remain consistent, but objects, geometry
and motion must be local to each view. Training descriptions match recorded
targets; inference appearance comes from the requested scenario.

The [shared sample](../../../cosmos3-nano-transfer-auto/shared/reference-scene.md#choose-a-caption-style)
contains paragraph-style `captions/` and the recommended
`captions_with_timestamp/` alternative. Keep the same actor's appearance and
physical motion consistent across cameras and time; perspective can change its
screen direction and when it is visible.

Keep outer frame bounds synchronized across views. For multiple outer chunks,
use `use_first_chunk_only=false` to retain the full timeline. The sample has
only one outer chunk, so `true` still retains all timestamped paragraphs;
`first_caption_paragraph_only` must be `false` to keep them all. Text timestamps
do not schedule rolling prompt transitions. Apply the [Sim-Auto settings](#request-fields).
Inspect saved caption/rolling metadata and transitions as well as videos.
