# RGB and render review

Use the [WSM4 guide](../../../docs/wsm4.md) to interpret the final rendered
components. RGB is optional. When available, compare it with the render as
described below. When absent, validate the converted data and inspect the render
alone; record that alignment and appearance could not be checked against source
images. Complete source RGB comparisons using the native camera conversion
before the default Hyperion retargeting step.

## Match the comparison

Use the same camera, image projection, crop, resolution, and capture time. Keep
raw RGB identifiable separately from undistorted or resized previews. Record
the source location and processing used to produce comparison images or videos.

Check camera exposure times and poses against the annotation timeline. A shared
frame pose may differ from the pose at image capture, and rolling shutter can
leave spatially varying errors. Correct supported timing differences or report
their size and the resulting limit on alignment. An algebraic projection check
after changing coordinate frames does not measure alignment with real RGB.
Measure projection and timing errors when correspondences permit, and separate
source annotation error from conversion error.

## Inspect represented components

Inspect frames from the final videos for every requested view. When RGB is
available, save side-by-side images or overlays. Include early, middle, and late
frames, plus events such as crossing a crosswalk, passing a close object, or a
signal changing state. Use another view or time when a feature is too small or
occluded to identify.

- Check lane colors and styles, physical boundary types, map elevation, and
  crosswalk placement. Investigate unexplained all-Other mappings and missing
  components. Source coverage may be incomplete even when the RGB shows them.
- Compare actor positions, dimensions, headings, and track continuity. Cuboids
  represent full extents, so their outlines need not match visible silhouettes.
- Include signs in overlays. Check height, facing, dimensions, source class, and
  consolidation across frames. Separate real sign panels from supporting
  structures and unrelated props; trace suspicious boxes to source IDs.
- Check road-arrow direction and observed signal state against the WSM4 guide.
  Distinguish a missing annotation from a conversion or rendering failure.

Do not change the source calibration to hide a renderer defect. If geometry
explodes or fills the image when crossing behind a camera, isolate the case and
check clipping before perspective division. Validate a renderer correction on
the complete representative clip before continuing the batch.

## Review retargeted views

For retargeted output, inspect all eleven Hyperion views using the same frame
and component checks. Check mounting height, direction, field of view, and
continuity against the vehicle trajectory and world geometry. Source RGB does
not share these camera poses or intrinsics, so an overlay with Hyperion output
is not an alignment check. Keep native RGB comparisons separate and identify
missing source coverage exposed by the new views.

## Save a review record

Record the reviewed clips, cameras, frame indices, represented components,
comparison paths, and findings. For a questionable feature, include its source
ID, the evidence for the chosen mapping, and any remaining uncertainty. Mark
unreviewed components explicitly rather than treating them as passed.

Report archive validity, decoded-video checks, geometric alignment, and semantic
review separately. Resolve conversion errors, rerender, and review the affected
frames again. State any source omissions, priors, and timing limitations that
remain in the delivered output.
