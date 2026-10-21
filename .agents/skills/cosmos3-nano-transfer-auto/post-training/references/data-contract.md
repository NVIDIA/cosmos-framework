# Prepare custom training data

Use your own synchronized driving recordings and rendered controls. The data
contract is about contents, timing and sensor meaning; it does not require a
particular storage service or collection system.

## Source files

An illustrative local organization is:

```text
dataset/
  train/scene-0001/
    rgb/<camera_key>.mp4
    wsm/<camera_key>.mp4
    captions/<camera_key>.json
    lidar/target/
    lidar/control/
    timestamps.json
    calibration.json
  validation/scene-0002/...
```

The folders organize your source files; the index below is the native builder's
input. The Framework builder packages existing media and tokenizer resources.
It does not render WSM, generate captions or synthesize measured LiDAR.

To render controls from your scenarios, follow the [WSM README](../../../cookbooks/cosmos3/simulation/auto/wsm/README.md) and
[conversion skill](../../../cookbooks/cosmos3/simulation/auto/wsm/skills/scenario-to-rds-hq/SKILL.md), with the
[renderer checks](../../inference/references/wsm-conventions.md#renderer-evidence-and-geometry).
You need map geometry, object tracks, ego poses, signal states and calibrated
sensor projections/timing. RGB and measured LiDAR recordings alone do not supply
those annotations. Prepare HD-map range controls separately on the matched LiDAR
rays; do not reuse camera-control MP4s as numerical LiDAR controls.

## Local training index

Create a JSONL file with one scene interval per row. Paths are relative to that
file. Here is a one-camera example; add video/WSM/caption entries for each selected
camera, retaining its physical key:

```json
{"id":"drive1-clip1","stratum":"480","assets":{"video_camera_front_wide_120fov":"train/scene-0001/rgb/camera_front_wide_120fov.mp4","wsm_camera_front_wide_120fov":"train/scene-0001/wsm/camera_front_wide_120fov.mp4","caption_camera_front_wide_120fov":"train/scene-0001/captions/camera_front_wide_120fov.json","lidar/lidar_rangemap":"train/scene-0001/lidar/target/lidar_rangemap.tar","lidar/hdmap_rangemap":"train/scene-0001/lidar/control/hdmap_rangemap.tar"}}
```

Keep `id` consistent with the clip key inside the range archives. The `stratum`
is `480` or `720`; default training recipes sample both, so supply both or
explicitly disable the absent stream.

Each caption file contains:

```json
{"caption_structured":{"chunk_0_400":{"start_frame":0,"end_frame":400,"caption":"Description grounded in this camera's recorded RGB and recording metadata."}}}
```

End bounds are exclusive. Replace the example interval and text with actual
source coverage. Preserve full text; stage formatters choose the appropriate
window or opening paragraph. Include front-wide caption metadata for LiDAR-only
training even when that camera is not in the selected RGB subset.

Range `.tar` files use the sparse format consumed by
`cosmos_framework.model.generator.tokenizers.lidar.preprocessing.load_prepared_lidar_video_3ch`.
For each sweep, include these archive members:

```text
<clip-id>.<source-frame-id>.lidar_row.npz
<clip-id>.<source-frame-id>.lidar_col.npz
<clip-id>.<source-frame-id>.lidar_range.npz
<clip-id>.<source-frame-id>.lidar_intensity.npz
```

Each NPZ stores one 1-D array under `arr_0`; arrays have equal lengths.
Rows are integer beam indices in [0,128), columns in [0,3600), range is in
meters. Resolve duplicate rays by nearest return before packaging. Measured
intensity is floating-point [0,1] or integer [0,255]. HD-map controls omit the
intensity member; occupied rays receive unit intensity in preprocessing.
Use a consistent clip ID matching the index, and actual camera-clock source IDs
(0,3,6,... for aligned regular 30/10 Hz data). Do not renumber missing sweeps
as if they were present. Use the projection described below, not arbitrary image
indices. The [reference scene](../../shared/reference-scene.md) shows the concrete
filenames and native index, with an HF download and checksum instructions.

Run `cosmos_framework.data.generator.local_simulation.auto_bundle` as shown in
the [post-training skill](../SKILL.md#prepare-a-local-training-bundle). It writes
`manifest.json`, `samples.json`, `assets/` and `resources/tokenizer/` into a
new directory. Inspect synchronized RGB/WSM overlays and a decoded training batch
after packaging.

| Source                      | Contents                                                                            | Role                                            |
| --------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------- |
| RGB per camera              | Real video for the selected scene interval                                          | Target; optionally its initial clean frame      |
| WSM per camera              | Layout rendered with that camera's effective projection, crop and time              | Full clean control, not appearance ground truth |
| Per-view captions           | Camera-local bodies with matching sample-local spans                                | Separate text sequence for each camera          |
| HD-map range control        | Geometry projected to the expected LiDAR rays, with unit intensity on occupied rays | Full clean LiDAR control                        |
| Measured LiDAR range target | Measured metric ranges, normalized intensity and validity at known sweep IDs        | Target; optionally an initial clean sweep       |

The documented `auto_bundle` builder requires **complete paired rows**: both
range items and RGB/WSM/captions for every selected camera, plus the front-wide
caption even if that camera is not selected. Separate-sensor training selects
RGB-only or LiDAR-only tasks from these paired rows; it does not permit missing
members at packaging time.
Keep licenses/consent and scene/drive-level split identities with your dataset.
Never split adjacent overlapping clips from the same drive across train and validation.

Use WSM v4.0.1 conventions (renderer selector `version=v4`), including signal
state/front/back, lane markings and projection. Inspect synchronized control/RGB overlays using
the same crop and resize. The
[input-checking guide](../../inference/references/input-preparation.md)
covers camera geometry and renderer checks for both training and inference.

## Training captions

Prefer timestamped descriptions of each camera's scene and its progression.
This is the format used in our training data; matching it reduces the
caption-format gap during post-training. Single-paragraph captions also work.
The [reference scene](../../shared/reference-scene.md#choose-a-caption-style)
provides both `captions/` and `captions_with_timestamp/`, with instructions for
selecting the latter in the training index before building the bundle.

Keep timestamped paragraphs inside the existing `caption` string and preserve
the outer `start_frame` / exclusive `end_frame` bounds. Use real clip-local times,
with paragraph breaks at meaningful changes rather than a mandatory equal split.
The sample's timestamped paragraphs belong to one outer chunk, not separate training samples.
Preserve the full source text; the selected recipe applies its full-caption or
opening-paragraph policy.

Training text must describe the **recorded scene**, not an appearance you would
like the model to generate later. For RGB-containing samples, use this source order:

1. Inspect synchronized target RGB across the selected cameras and interval for
   visible appearance, weather, lighting, road surface and motion. Use recording
   metadata for facts such as geographic setting that pixels cannot establish.
2. Compare WSM with those targets for camera identity, geometry, actors and timing.
   A control/target disagreement needs investigation; do not rewrite the caption
   to conceal it or fabricate missing control objects.
3. Preserve supported existing descriptions; write camera-local bodies and use
   consistent identities, appearance and physical motion across views and time.
   The same actor's color/type and trajectory must agree on a shared clock;
   opposite screen motion or different visibility intervals can be correct for
   different camera directions. Do not assign new paint, clothing, background
   or weather to unchanged targets. If a property is unobservable and metadata is
   missing, leave it unspecified and report the limitation rather than guessing.
4. Check the effective caption against target RGB at the start/end and narrated
   changes, including extra frames for ambiguous events. Apply the stage's full-body
   or opening-paragraph policy only after this target-grounding check.

For example, real blue cars on a dry daytime road must not acquire white paint
and rainy-night captions solely from a requested augmentation. A new weather or
appearance condition requires corresponding target data, not just edited labels.

Keep the [caption style aligned with inference](../../shared/caption-policy.md#keep-training-and-inference-captions-aligned):
prefer timestamped, camera-local descriptions and consistent appearances/motion
across views. The grounding rules above still apply to every training target.

The bundled `av-per-view-caption` guide is for **inference prompt authoring**:
it can assign plausible appearance from WSM when future RGB will be generated.
Do not invoke that WSM-only workflow as a training-label generator. If a required
RGB target cannot be inspected, retain supplied captions as unverified; do not
certify new captions as suitable supervision.

**LiDAR-only tasks in the separate-sensor phase:** the formatter uses aligned
front-camera caption metadata without loading camera video for that task. The
bundle still contains the required camera members. Verify caption provenance,
scene identity and interval against the source recording and range pair; apply
the stage's opening-paragraph policy. LiDAR alone cannot verify paint, lighting
or weather. Use the paired recording/metadata rather than inventing those properties.

## Camera identities

When adapting to your own rig, keep its calibrated viewpoints and render WSM
aligned with the recorded RGB, including the same crop and projection. Skip
reference-rig retargeting for these training pairs: it changes control viewpoints
without transforming RGB targets. See the
[reference-rig versus custom-rig workflow](../../inference/references/input-preparation.md#choose-reference-rig-or-custom-rig-generation).

For your own rig, use [Camera Rig Explorer](../../../cookbooks/cosmos3/simulation/auto/wsm/docs/camera-rig/guide.md)
and the [nearest-camera mapping workflow](../../inference/references/input-preparation.md#use-camera-rig-explorer-for-your-own-rig)
before building the training index. Keep that mapping consistent with inference.
Source filenames can remain your own; associate their paths with the chosen
supported camera keys in the index. The native bundle builder accepts supported,
unique keys; it does not create new view embeddings for arbitrary sensor names.

Keep these learned identities even when dropping or reordering views:

| ID  | Camera key                    |
| --- | ----------------------------- |
| 0   | `camera_front_wide_120fov`    |
| 1   | `camera_cross_right_120fov`   |
| 2   | `camera_rear_right_70fov`     |
| 3   | `camera_rear_tele_30fov`      |
| 4   | `camera_rear_left_70fov`      |
| 5   | `camera_cross_left_120fov`    |
| 6   | `camera_front_tele_30fov`     |
| 7   | `camera_front_fisheye_200fov` |
| 8   | `camera_left_fisheye_200fov`  |
| 9   | `camera_right_fisheye_200fov` |
| 10  | `camera_rear_fisheye_200fov`  |

LiDAR uses sensor identity 11. An index within a selected subset is not the learned
sensor ID. For example, front-wide/rear-left/front-tele retain IDs 0/4/6.
Nominal camera names are not full calibration. Compare direction, FOV and lens
projection before assigning a custom camera. Render controls with the same
calibration as the recorded RGB.

## Time and spatial alignment

- Use nominal 30 FPS cameras and 10 Hz LiDAR for the reference recipe.
- Select one interval across cameras, controls, captions and LiDAR using actual
  timestamps/source IDs, not just equal file durations.
- Decode RGB and WSM with identical frame selection and resize/center-crop.
  Apply calibration changes consistently when rendering/cropping.
- Trim RGB to `1 + 4k` frames for its tokenizer. The short joint stage uses
  61 RGB frames / 21 endpoint-inclusive sweeps; a complete long-stage window
  reaches 397 RGB frames / 133 sweeps after decoding up to 400 RGB frames.
- Handle missing/irregular sweeps explicitly. A regular tensor index does not
  preserve an omitted source timestamp. Reject insufficient/unsynchronized
  samples or use a documented, validated alignment policy.

Use clip-local caption times. A 61-frame clip at 30 FPS has playback duration
61/30 seconds and a final frame timestamp of 60/30; those are different quantities.
Store enough information to recover the selected source interval.

## Model-facing batch contract

These are native loader output requirements, not a replacement dataset-file schema.
Verify them in the actual batch before accepting a new dataset.

1. **Vision:** `video=[WSM_control, RGB_target]`, each `[3,V*T,H,W]`,
   uint8 before model normalization. Flatten camera-major: all times of one camera,
   then the next. Keep selected view IDs and per-view text in the same order.
2. **LiDAR:** `lidar=[HDmap_range_control, LiDAR_target]`, each
   `[3,T_lidar,128,1808]`. Channels are metric range, intensity in [0,1],
   and binary validity. The range projection pools 3600 native horizontal bins
   to 1800 semantic rays, then circular-pads to 1808. Padding is not extra rays.
   Use the supplied Pandar128 beam-elevation calibration, azimuth +180° to −180°
   with the endpoint excluded, x-forward/y-left/z-up coordinates, and nearest-return
   selection. The tokenizer's affine normalization uses a fixed 5–100 m span;
   invalid range is zero with a separate validity mask. Do not substitute arbitrary
   128-row images or a different range scale. A different sensor needs a validated
   projection adapter, not a shape-only resize; vertical resampling is not supported
   by the reference projection. Preserve the tokenizer's filtering/validity rules.
3. **Video encoding:** normalize RGB/WSM to [-1,1], encode cameras independently;
   temporal/spatial compression 4/16, 48 latent channels, model patch size 2.
   **LiDAR encoding:** no temporal downsampling, spatial compression 16,
   128 latent channels, model patch size 1. Use the supplied tokenizers.
4. **Text:** retain full selected camera-local bodies for the final stage.
   Let the runtime add camera/rig and duration/resolution wrappers once.
   Collate caption lists per sample and preserve independent text boundaries.
   The short-joint phase's shortened-caption policy is described in the recipe.
5. **Conditioning:** controls are clean throughout. Target prefixes are clean only
   for I2V. At 30/10 Hz, one conditioned camera latent maps to one initial measured
   LiDAR sweep. Zero probability of *whole-clip* LiDAR conditioning does not
   disable this temporal prefix.
6. **Dropout/loss:** sample-level and per-view caption dropouts each use probability
   0.1 in joint training; both long-joint phases drop both LiDAR items together
   on 10% of samples. It does not drop only the prefix. Controls otherwise stay.
   Clean elements have zero prediction error but remain in the configured
   local-item mean denominator; do not switch to active-element-only reduction.

Inspect how the installed loader handles corruption, short clips and missing
captions. Count rejected/retried samples and check that intended source mixtures
are still achieved. For one real sample, retain decoded frames, selected IDs/times,
captions, range units/masks, formatted shapes, conditioning masks and packed lengths.
Verify a retained-LiDAR case and its dropout case separately.

Use these checks to verify a new dataset before starting training.
