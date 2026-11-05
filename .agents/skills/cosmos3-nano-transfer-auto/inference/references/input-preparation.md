# Check Transfer-Auto inputs

Align camera identities, controls, captions and timing with the selected model.

## Options and scope

- “Show me the options and defaults.” Explain the workflow without requesting
  media or starting work.
- “Check my request, controls, calibration and renderer settings. Do not generate.”
  Inspect and recommend; do not rewrite or launch.
- “Prepare captions and requests for these scenes in a new folder.” Author and
  package only the requested outputs, preserving original inputs.

Default to checking. Read the actual manifest fields and the installed runtime's
reader; an inventory is not a universal request schema. Return prepared requests
to [cosmos3-nano-transfer-auto-inference](../SKILL.md) for generation.
Do not switch runtime/checkpoint or invoke an external paid caption service silently.

## 1. Establish the actual run

Read the request, selected weights/model version and runtime version, plus resolved
arguments/logs when supplied. Record views, controls, selected interval and scenario
properties. Distinguish physical source channels from model camera aliases.

Determine no-initial-RGB versus initial-RGB conditioning from actual fields, not
the request filename. Inspect the text each camera consumes; a scene-level prompt
copied into a results export does not establish the effective per-view conditioning.
Do not compare I2V against WSM-only generation and attribute the difference to captions.

## 2. Map physical cameras to learned identities

### Choose reference-rig or custom-rig generation

The released Auto models are optimized for the reference camera rig. You can
try your own rig, but expect quality to degrade as camera placement, orientation,
FOV, lens projection and overlap differ from the training geometry.

Choose based on the camera viewpoints you need in the generated output:

- **Generate from the reference rig's viewpoints.** Follow the bundled
  [scenario conversion skill](../../../cookbooks/cosmos3/simulation/auto/wsm/skills/scenario-to-rds-hq/SKILL.md)
  and its [Hyperion retargeting guide](../../../cookbooks/cosmos3/simulation/auto/wsm/skills/scenario-to-rds-hq/references/hyperion-retargeting.md).
  The provided `retarget_hyperion.py` helper uses validated RDS-HQ scene
  annotations and vehicle poses to write camera poses and intrinsics for the
  bundled eleven-camera Hyperion rig. Then render WSM with that calibration,
  use the corresponding camera keys, and prepare captions for the rendered views.
  This matches the rendered views to the released model's camera geometry.
- **Generate from your own rig's viewpoints.** Keep your actual calibration and
  render WSM for those cameras. Use the [Camera Rig Explorer workflow](#use-camera-rig-explorer-for-your-own-rig)
  below to choose the closest supported identities, considering direction, FOV,
  projection, position and overlap. Inspect the generated result in every view. If quality remains insufficient after
  checking alignment and inputs, [post-train Transfer-Auto](../../post-training/SKILL.md#quickstart-fine-tune-the-released-model)
  on your paired recordings. For streaming generation, then
  [distill the adapted Transfer-Auto checkpoint into Sim-Auto](../../../cosmos3-nano-sim-auto/post-training/SKILL.md).
  Ordinary SFT of the released distilled student is not the recommended adaptation path.

Retargeting requires scene geometry, annotations and verified vehicle poses;
RGB or WSM MP4s alone are insufficient. It prepares geometry for re-rendering,
not a transformation of recorded RGB. Keep native and retargeted data separate.
**Do not pair retargeted WSM with unchanged custom-rig RGB** as initial conditioning
or training supervision: their viewpoints no longer match. Initial RGB must
match the selected output cameras and time. For custom-rig post-training, skip
reference-rig retargeting and retain aligned native-camera RGB/WSM pairs. The
camera retargeting tool does not adapt LiDAR; joint generation still requires
the supported LiDAR geometry and synchronized controls.

### Preserve camera identity

The checkpoint learned a separate view embedding for each supported camera key.
Framework looks up that ID from `camera_key`; it does not infer camera orientation
from pixels, captions, filenames or the camera's position in a list. A correctly
named file can still contain the wrong view.

Use the model's [camera identity table](../../post-training/references/data-contract.md#camera-identities)
and the user's calibration or verified renderer mapping. Compare optical-axis
direction, then effective FOV and lens projection after crop/resize. Nominal
wide/telephoto labels do not establish pinhole versus F-theta calibration.
Filenames are not evidence of physical orientation.

Consider all supported identities and the rig as a whole, not just the usual seven.
Flag large direction/FOV differences, fisheye-versus-pinhole mismatch, uncertain
overlap and duplicate assignments. Use unique supported camera keys. Record
geometry differences so they can guide retargeting or post-training.

| Source camera / control | Verified direction / FOV / projection | Model camera key | Learned ID | Difference / uncertainty |
| ----------------------- | ------------------------------------- | ---------------- | ---------- | ------------------------ |

IDs come from the identity table, not list positions. For example,
front-wide/rear-left/front-tele retain 0/4/6, not 0/1/2. Preserve the association
between control, RGB, caption and output when reordering or dropping views.
For Framework requests, preserve the explicit `multiview.views` list.
Do not add a canonical view-count override when preserving a custom subset.

A broad rear camera is not automatically a rear-tele camera, and a fisheye alias
does not reproject its pixels. Recommend a mapping for check-only requests; apply
it only when input preparation is requested, retaining source-channel provenance.

### Use Camera Rig Explorer for your own rig

Open the offline [Camera Rig Explorer](../../../cookbooks/cosmos3/simulation/auto/wsm/docs/camera-rig/guide.md)
to inspect reference camera positions, viewing directions, coverage and intrinsics
in synchronized 2D/3D views. Open `.agents/skills/cookbooks/cosmos3/simulation/auto/wsm/docs/camera-rig/index.html` locally with its
supporting files kept alongside it. Nothing needs to be uploaded.

1. Inspect both reference calibrations and compare your cameras' optical axes,
   effective FOV, projection, positions and overlap. You can import your rig using
   the viewer's [geometry JSON contract](../../../cookbooks/cosmos3/simulation/auto/wsm/docs/camera-rig/guide.md#import-another-calibration);
   it requires prepared rays/sections, not an arbitrary raw calibration file.
2. Assign each customer camera the closest supported `camera_key` as a starting
   point for inference and Transfer-Auto post-training. Record the mapping in the
   table above, including mismatches. The viewer helps you compare geometry; it
   does not choose model IDs or modify an inference request.
3. Preserve that assignment across training and inference. Render WSM using your
   actual camera calibration and keep matched RGB, WSM and caption paths together.
   Use the identity table for embedding IDs, not the viewer's display numbers.
4. Test a short synchronized scene and inspect every view and overlap.
   For substantial rig differences, adapt
   [Transfer-Auto](../../post-training/SKILL.md) on paired data first; if you need
   streaming generation, then [distill the adapted model into Sim-Auto](../../../cosmos3-nano-sim-auto/post-training/SKILL.md).

## 3. Check controls and synchronization

Compare the actual renderer version/config and class conversion against the
model's expected [WSM conventions](wsm-conventions.md). Shared colors
or an MP4 encoder tag do not identify the correct renderer version.

Probe resolution, FPS/PTS, frame count, interval, offsets and decode errors.
Actually inspect synchronized overview frames, including start/end, then targeted
frames/crops for important ambiguities. Check corresponding events against the
source timestamps to verify synchronization. If frames cannot be viewed, report
visual inspection as incomplete.

Record the actual source sampling rate, requested model `fps`, and saved-video
playback rate separately. The reference uses 30-FPS cameras; lower-rate inputs
need an explicit quality check. Follow the [FPS troubleshooting guide](input-and-execution.md#source-timing-model-fps-and-playback-fps)
for 10-FPS data. An FPS flag or container-header edit does not resample controls;
joint generation also requires aligned LiDAR clocks and source sweep IDs.

Check class meanings, lane/crossing/boundary conventions, signal state/front/back
and projection. Unknown signals are not red, green or off. A truck-category box
does not prove a truck subtype; class colors are not paint. Do not invent actors
or signals to compensate for uncertain upstream labels.

Compare source RGB and WSM to check camera alignment and investigate missing
objects or detection errors. For caption authoring, follow the source rules below.

## 4. Delegate caption authoring

For new/upsampled captions, read and follow the bundled
[AV caption guide](caption-guide.md), including its caption policy.
Pass synchronized media/timing, renderer semantics, physical directions and chosen
output keys, mapping uncertainties, four properties with sources, and compatible
existing text unless excluded. Do not replace physical geometry with an approximate alias.

Defaults are **long timestamped captions, 300–400 body words per view across all
blocks, all supplied views**. Paragraphs, shorter lengths and view subsets are
options. Extra visual review is off by default; WSM inspection during authoring
remains required. Do not add a mandatory proposal/review loop.

For existing captions in a check-only request, report semantic contradictions with
view/time evidence and suggested corrections; do not invoke authoring or rewrite
the original. Check time of day, weather, road condition and geographic setting
by meaning: “night” with sunlit buildings conflicts, while a coastal rear view
need not show the sea.

Check the captions together on a shared timeline: the same actor must retain
compatible color, type and physical motion across views and time. For example,
do not describe one car as blue in one view and white in another, or ego as
stopped and driving at the same time. Different screen-space motion, reflections
and visibility can be correct because cameras face different directions; do not
force identical wording, visibility times or paragraph boundaries.

## 5. Check packaging and effective sampling settings

Follow the [input/execution contract](input-and-execution.md):

- Keep bodies separate from camera/rig wrappers; insert headers once. Inspect
  actual effective text rather than silently broadcasting the front prompt.
- Check key coverage, order, sidecar bounds, truncation, tokenizer limits and
  effective duration. Caption loading can determine the generated frame span.
- For a front-only caption comparison, retain sidecars with the same frame bounds
  for every selected view and use `"caption": ""` for the other views. Inspect the
  effective text; JSON `null` and the string `"None"` are not empty captions.
- A caption-key-to-body JSON is not a runnable inference manifest. Preserve
  controls, initial images and all non-caption settings in a caption comparison.
- Report weights/variant, conditioning mode, view IDs, resolution, FPS/frames,
  seed, steps, shift, guidance and actual positive/negative text.

For the undistilled Transfer-Auto model, guidance 3 is a starting point; 1.5 is a
lower-guidance comparison option, not a universal default. Preserve explicit
settings when reproducing a user's run. Do not apply these settings to a causal
or distilled model.

Use the bundled [negative prompt](../assets/negative-prompt.txt)
when requested. In separate-per-view caption mode, use the runtime's
`per_view_negative_prompt` field; the ordinary negative field is not equivalent.
Check whether its wording conflicts with intentional fog, rain or lens effects
before recommending a change. Other runtime interfaces require their own
resolved-field check; do not copy unsupported flags.

## 6. Make results inspectable

For authorized generation, save compressed viewing copies and **50% WSM / 50%
generated-RGB blends** per camera, preserving originals. Use the actual controls
fed to the model, same interval/crop/resize. Do not stretch or misalign inputs
just to make a blend. Reuse available tooling; a new framework is unnecessary.

Describe failures by camera and time, separating observations from hypotheses.
Compare source controls with the saved model input before blaming mapping,
captions or model generalization. Inspect every selected view and confirm its
camera identity against the calibration.

Propose the smallest matched comparison: fixed checkpoint, conditioning, controls,
seed and all settings except the suspected cause. Launch only when requested.
Finish with checked/proposed/applied mappings, effective settings, output paths
and remaining uncertainty. Distinguish input checks, completed generation and
visually inspected quality.
