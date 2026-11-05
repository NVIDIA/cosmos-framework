---
name: scenario-to-rds-hq
description: Convert a driving scenario or dataset in an unfamiliar format into RDS-HQ for WSM4 rendering, with final Hyperion camera retargeting by default. Use for source-schema discovery, coordinate and camera conversion, semantic class reduction, missing-field inference, and validation of converted scenarios.
---

# Prepare and render WSM controls

Produce an RDS-HQ dataset that preserves the source scene and loads in the reference
renderer, together with a reproducible adapter and a concise record of mappings,
inferences, omissions, and validation. Use judgment to bridge differences between
formats; the source need not already resemble RDS-HQ.

## What the user supplies and receives

- **Inputs:** scene geometry, object tracks, camera calibration and poses, with
  source-format details when available. RGB is optional evidence; RGB or WSM
  videos alone do not contain the scene needed for camera retargeting.
- **Outputs:** converted scene archives, the reusable source converter, rendered
  camera controls when requested, and a concise mapping/validation report.
- **Next step:** caption the final rendered views and prepare an Auto request
  using the [inference handoff](../../docs/scenario-to-rds-hq.md#continue-to-auto-inference).
  Conversion or rendering alone does not authorize model inference or uploads.

Keep confidential source assets outside the checkout. Rendering requires the
[standalone GPU/EGL environment](../../README.md#install); structural conversion
checks do not. For the released Auto models use WSM 4.0.1, selected by `version=v4`.

First convert and validate the scene with its source cameras. Then retarget it
to the bundled NVIDIA Hyperion 8 rig: seven cameras at 1924×1084 and four
fisheyes at 1720×1080. This final step is enabled by default; skip it when the
user asks to keep the source rig or disable retargeting, or plans to post-train
with their own data.
Read [Hyperion retargeting](references/hyperion-retargeting.md)
before source discovery so the adapter retains the required vehicle poses.

## Example request

> Use the scenario-to-rds-hq skill to convert the clip at `<source-path>` into
> RDS-HQ under `<output-path>`, validate it, and render WSM4 videos.

For an Auto inference example, request 30 FPS and an initial 297-frame window
when the source covers it. Check the actual prepared timeline before rendering:
the pose sampler uses an integer stride, not upsampling. Prepare any needed
resampling in the conversion with aligned actors, poses and signals; do not
relabel 10 Hz observations as 30 Hz. Keep explicit user timing choices and explain
how they differ from the model's reference 30 FPS inputs.

If the skill is not installed in the agent's skill catalog, include the path to
this `SKILL.md` in the request. Discover the source format from the supplied files;
ask for missing information only when it affects the conversion and cannot be
established from the source or available readers.

## Read the target contract

Resolve paths relative to this skill's directory, not the working directory.
Read the [RDS-HQ format reference](../../docs/rds-hq.md) before choosing an
intermediate representation. Use [WSM4](../../docs/wsm4.md) for visual meaning
and rendering commands. Inspect the relevant implementation when a contract
detail affects the result:

- [Input loading](../../cosmos_auto/wm/cli.py) and
  [sample decoding](../../cosmos_auto/wm/wds_utils.py): archive keys, camera
  selection, frame indices, resizing, and defaults.
- [Camera models](../../cosmos_auto/wm/camera): pose axes, projection, polynomial
  direction, and pixel transforms.
- [Objects](../../cosmos_auto/wm/bbox_utils.py): class aliases, box origins,
  interpolation, and stationary-track stabilization.
- [Map and signals](../../cosmos_auto/wm/minimap_utils.py),
  [lanelines](../../cosmos_auto/wm/laneline_utils.py), and
  [boundaries](../../cosmos_auto/wm/road_boundary_utils.py): consumed attributes,
  vertex order, facing, special classes, and fallbacks.
- [Palette and class configurations](../../cosmos_auto/wm/color_scheme) and
  [render configuration](../../cosmos_auto/wm/conf): the actual supported classes
  and the settings needed for the source's cameras and timeline.

Treat code as the authority on current behavior. Add missing, verified target
schema details to the format reference when editing that reference is in scope.
Keep source-specific recipes and conversion procedures out of the reference.
Check projection helpers against known rays and analytical geometry. If a helper
disagrees with those checks, isolate the implementation issue rather than changing
correct source calibration to compensate; report the validation limit.

## Discover the source

Inspect the actual payloads, metadata, and available readers before writing the
adapter. File names and familiar-looking arrays are clues, not proof of units,
axes, transforms, enums, or timestamp meaning. Sample multiple frames and feature
classes, including missing values and the clip's endpoints.

Establish these facts with evidence from schemas, producers, readers, calibration,
or cross-frame consistency:

- The requested scenario interval versus the full extent of surrounding files.
- Timestamp units, clock domains, source rates, camera offsets, and track gaps.
- Geometry units; handedness and axes; world, vehicle, and sensor frames;
  transform direction; quaternion order; and rotation composition order.
- Camera models and available views, calibration image sizes, crop/resize
  history, extrinsics, and any calibration corrections.
- Vehicle pose origin and its offset from the rear-axle ground projection,
  needed to attach the Hyperion rig correctly.
- Static map observations versus persistent map entities, object IDs, class
  definitions, and signal geometry/state associations.

Select a compact, source-specific adapter after discovery. Reuse a reader only
after checking its assumptions: a renderer's approximate enum map or an older
reader is not an authoritative source taxonomy. Inspect only relevant fields;
do not copy recordings or unrelated source metadata into the RDS-HQ tars.

## Resolve semantics and incomplete data

Read [mapping and inference decisions](references/mapping-and-inference.md).
Use many-to-one semantic mapping into the target's small vocabulary. Prefer an
explicit source label, then a documented parent class, then evidence from
geometry, topology, appearance, and temporal behavior. Choose a useful supported
class when the evidence warrants it instead of defaulting every unfamiliar label
to Other.

Check that a source object belongs in the target before assigning a class.
`Others` covers potentially moving objects such as protruding objects, strollers,
and animals. Exclude standalone static props such as cones and construction
objects from RDS-HQ and record their omission. A stopped or parked vehicle still
belongs to its actor class. Set `object_is_moving=true` in every record of a track
that moves at any point in the clip, including its stationary portions. Use
`false` only for a track that stays stationary throughout the clip. The renderer's
permissive fallback for unknown class strings does not expand the format's
semantic scope.

When information is missing, distinguish a recoverable attribute from an
unobserved feature. Derive transforms, dimensions, orientation, motion, or broad
classes from available evidence. Use a consistent, physically plausible prior
when a minor missing attribute needs one, and record its basis and affected
count. For a physical road boundary whose subtype remains unresolved, use `curb`
as the nominal default and record the assignment as `defaulted`. Preserve a
known or supported barrier, wall, seam, or other type. For other components, use
their documented priors or unknown/Other values when subtype evidence is weak.
An unavailable map component is a valid empty layer, not fabricated geometry.

Do not guess numeric enum meanings from their values. Do not infer a green signal
from vehicle movement, or convert an unobserved signal into an observed off
signal. Geometry, camera projection, or clock ambiguity that would substantially
change the scene deserves a focused question when local evidence cannot resolve
it. Continue independent work while the question is pending.

Keep a small conversion report outside the tars. For each source category or
missing-field decision, record target value, evidence, method (`direct`,
`mapped`, `inferred`, `defaulted`, or `omitted`), and count. Preserve useful extra
source details there when they cannot be expressed in the rendering contract.
The report should expose information loss without copying whole source records.

## Build geometry and time consistently

Choose a shared timeline and one fixed world frame, anchored at the first frame's
rear-axle ground projection. Apply the same world rebase to cameras, objects, and
map vertices. The initial vehicle transform is identity; the initial camera
transform retains the mounting offset and orientation.

When retargeting is enabled, save `vehicle_to_world.npy` with shape `[T,4,4]`
beside the native conversion, outside its `pose` archives. Use metres, vehicle
FLU axes, the rear-axle ground origin, and the annotations' world frame and
timestamps. Recover vehicle poses from camera poses only with verified source
extrinsics; see the [transform chain](references/hyperion-retargeting.md#vehicle-poses).

Write down and check the transform chain before applying it. Camera poses map
OpenCV axes to world axes; source camera axes may differ from vehicle axes. A
change of the camera's local basis multiplies on the appropriate side of its
extrinsic, not both sides automatically. Check forward-facing test rays and an
object in front of the vehicle to catch inverted or mirrored transforms.

Use the calibrated projection model. Respect polynomial direction and units;
zero-pad a lower-order polynomial only when it is the same polynomial model.
Apply known image crop and resize operations to the calibration. Fit a different
model only when necessary, measuring reprojection error across the field of view
and retaining the fit residuals. Check available cameras against actual data;
a calibration entry alone does not prove that a synchronized view is available.
Record whether the target image plane matches raw or undistorted RGB. When
exporting ideal pinhole intrinsics from a distorted source camera, undistort the
comparison RGB with the source coefficients and the same output intrinsics.
Keep this processing explicit for downstream RGB/control pairs.

Match annotations and camera poses by timestamps before assigning pose-frame
indices. Every camera needs a pose at every index from `0` through `T-1` on that
shared timeline. Resample source poses within their observed coverage, or use
consecutive indices at the source rate and set `input_fps_spec.input_pose_fps`
accordingly. The renderer does not fill missing pose files.

Use rigid rotation interpolation and linear translation interpolation within
supported intervals. Avoid interpolating an object across disappearance,
an ID reuse, or a long gap. Full-rate object records, including empty frames,
make track presence explicit and avoid unintended renderer extrapolation.
Sparse archives must include records at the first and last pose frames, using
empty dictionaries where no actors are present. If source coverage ends early,
shorten the shared timeline or infer an endpoint only from defensible evidence;
do not silently hold the last observed actors or assume that the scene is empty.
Signal states use the same original frame indices; hold observations only for a
bounded interval justified by the source update rate, then use `unknown`.

Transform repeated static observations into the fixed world frame before
consolidating them. Confirm that IDs are meaningful; an all-zero ID field is not
a track identifier. Select or fuse consistent observations, separate disconnected
geometry, and remove duplicate representations without merging nearby distinct
features. Different IDs can denote the same re-identified feature, but closeness
alone is insufficient: check observation times, dimensions, orientation, and
semantic compatibility before merging. A static RDS-HQ map contains no per-frame
visibility or confidence; describe material loss from a changing or incomplete
source map in the report.

Follow the documented cuboid ordering and face orientation. Preserve box centers,
physical dimensions, elevated roads, lane-line direction, and arrow travel
direction. Never use arbitrary point order from an unordered box corner set.

## Write and validate

Use [rds_hq_io.py](scripts/rds_hq_io.py) for deterministic archive writing and a
structural check, or an equivalent writer after inspecting that helper's API.
Its `write_clip` API takes full-rate camera poses and one object dictionary per
pose frame; its validator also accepts sparse object archives with original
pose-frame indices and both boundary records. The helper uses NumPy and Shapely,
included in the package's `wm` installation. Use numeric arrays/scalars and the
lowercase camera names specified by the format reference.
Keep semantic inference in the adapter; the writer should not silently invent
missing layers or attributes. Write all required layers, using `{"labels":[]}`
when a component is absent. Traffic light state keys are string indices into the
geometry's `labels` list, such as `"0"`; each record also carries the matching
`feature_id`. Write exactly `T` states per light, indexed by original pose frame,
and include a basic-color mapping for every state used. Fill unobserved frames
with `unknown`. An empty traffic light layer has empty state and mapping
dictionaries.
Reject self-intersecting or degenerate XY polygon footprints. Review the
validator's `nonplanar_polygons` report against the source road surface: preserve
surveyed elevation when supported, or split into suitable planar pieces when
required. Use `require_planar_polygons=True` in the writer and
`--require-planar-polygons` in the validator for strict planarity; do not silently
flatten geometry to make validation pass.

Save the adapter and report with the conversion outputs. Use an explicit output
root, preserve the source, and avoid overwriting an existing converted clip.
Validate staged archives before publishing them so a failed conversion can be
corrected and retried without leaving a partially valid dataset.
The skill's own files contain reusable guidance and helpers, not generated
datasets, source-format details, or environment-specific launch paths.

Run the structural validator:

```bash
python <skill-directory>/scripts/rds_hq_io.py \
    <rds-hq-root> <clip-id> --camera-type ftheta
```

Then load the archives with the package's actual `get_sample` reader and camera
models. Check more than file existence: frame and track coverage, finite rigid
poses, positive dimensions, class counts, geometrical extent, signal alignment,
and a few forward projections. Compare these counts with source coverage so an
unexpectedly empty or heavily reduced scene is investigated before rendering.

## Validate the source-camera scene and refine

When rendering is authorized, start with one representative camera and a complete
short clip, using `version=v4`. Include close objects and camera crossings of
road polygons when present; a few initial frames can miss projection failures.
Choose camera names, input FPS, and chunk length to match the converted clip
rather than relying on incompatible defaults. Reuse a verified rendering
environment when available; check GPU/EGL in the actual execution environment
before rendering. Inspect the current
runtime rather than assuming a previous environment or launch script still applies.
After a failure, diagnose and retry this small case before expanding.
Use the standard WSM4 colors. Changing the control-video color scheme requires
retraining the model on controls rendered with that scheme.

Confirm job completion and decode the video to verify resolution, frame rate,
frame count, and nonempty content. Then follow the
[RGB and render review](references/visual-review.md). When RGB exists, save and
inspect comparisons with the source-camera render for every requested view.
Use the WSM4 guide to check component meaning and appearance. Include early, middle,
late, and event frames. Review signs as well as actors and road geometry.
Record reviewed components, source IDs for suspicious features, and unresolved
issues. Passing archive and video checks does not establish semantic or
geometrical correctness.

Fix the adapter, recheck the affected invariant, and rerender the representative
case. Once it passes, validate the remaining requested views or duration. Report
the output location, represented components, consequential assumptions, and
observed validation results. Distinguish unavailable source components from
conversion failures and do not claim support for arbitrary formats based on a
single successful example.

## Retarget to Hyperion and validate the final views

When retargeting is enabled, run [retarget_hyperion.py](scripts/retarget_hyperion.py)
after the native conversion passes validation. Follow
[Hyperion retargeting](references/hyperion-retargeting.md) for the command and
checks. Use the bundled calibration, saved vehicle poses, and a separate output
root. If a required vehicle transform is unknown, ask for it before retargeting.

Load all eleven cameras through the actual reader and F-theta model. Verify
`vehicle_to_world[t] @ camera_to_vehicle`, the prepared intrinsics at each
camera's resolution, unchanged scene archives, and the original pose indices
and sample rate.
When rendering is authorized, use `version=v4 camera_type=ftheta
camera_spec=internal_11cam resolution_spec=source`, with FPS and duration matched
to the clip. Validate one representative view before reviewing all eleven.

Keep source RGB comparisons with the native render. Review Hyperion views
against the scene geometry, and save the retargeting report and visual findings,
including any gaps in source coverage. If retargeting is disabled, deliver the
validated native conversion and record the opt-out.

## Hand off the final rendered views

For requested Auto input preparation in a full Framework checkout, follow
[render → caption → inference](../../docs/scenario-to-rds-hq.md#continue-to-auto-inference).
Render before asking for caption files. Invoke the existing per-view captioning
guide with synchronized final-view controls, verified camera identities and the
four user/scenario properties; do not create a second caption policy here.
Request camera-local JSON sidecars covering the selected frame interval. Review
provided captions without rewriting them unless asked. Retargeted views need
new visibility checks, not copied source-camera narratives.

Use the notebook's native request export for F-theta scenes with supported Auto
camera keys and 30 Hz poses. Match its selected cameras, interval and resolution
profile to any reused renders. For pinhole cameras, custom source names or
controls without raw archives, use the documented
[custom-rig handoff](../../docs/scenario-to-rds-hq.md#custom-rigs-and-already-rendered-controls)
and existing native request schema. Verify each physical-camera-to-model-key
mapping explicitly; preserve source filenames and geometry. Do not introduce a
new inference adapter or dataloader, or rename a camera to conceal a rig mismatch.
Retargeting does not transform recorded RGB or LiDAR: image conditioning needs
matching initial images, and joint generation needs aligned numerical HD-map
range-map controls prepared separately. If only the standalone renderer is
available, deliver its outputs and point to the full Framework handoff without
claiming model inference has run.
