# WSM4 rendering

World Scenario Map (WSM) is a visual representation used in control videos for
Cosmos3-Nano-Transfer-Auto and Cosmos3-Nano-Sim-Auto. **WSM4** improves on WSM V3, used in Cosmos Transfer 2.5,
with finer control over scene features.

The reference renderer, `cosmos-auto-wm`, takes a scenario in
[RDS-HQ format](rds-hq.md) and projects its polylines, polygons, and cuboids into
a control video using camera extrinsics and intrinsics. Colors and symbols
encode the scene features in each camera view.

> **Use the WSM version the model was trained on.** Compatibility between V3 and
> WSM4 controls is not guaranteed. The released Auto models use WSM 4.0.1
> conventions, selected by `version=v4`. The separate `v4.1` selector adds
> occupancy voxels and is not another spelling of 4.0.1.

The palettes and annotated frames below are included in this repository. This
guide explains their visual conventions and how to reproduce the representation.

- [What's new in WSM4](#whats-new-in-wsm4)
- [Render one clip](#render-one-clip)
- [Visual conventions](#visual-conventions)
- [Annotated examples](#annotated-examples)
- [Cameras, frames, and output files](#cameras-frames-and-output-files)
- [Validation and troubleshooting](#validation-and-troubleshooting)

## What's new in WSM4

Compared with WSM V3:

- **Traffic signs** have distinct colors for stop, yield, do not enter, one-way,
  and pedestrian-crossing signs. Left- and right-facing one-way signs use
  different colors. The backs of recognized signs also have a distinct color.
- **Traffic lights** distinguish red, green, yellow, red-yellow, and stateless
  indications; solid, left-arrow, right-arrow, and up-arrow signals; and
  pedestrian stop/go states. Front and back faces are also distinguished.
- **Road markings** include seven directional arrow shapes. STOP and KEEP CLEAR
  markings appear as colored areas without lettering.
- **Road boundaries** use six color groups: barrier/fence, curb, road boundary
  (seam), obstacle bars (guide posts), wall, and other.
- **Dynamic-object boxes** follow the distortion of F-theta lenses when projected
  into the camera view.

Dynamic objects, lanelines, crosswalks, waitlines, and poles retain their V3 colors
and styles.

For WSM4, the reference renderer reads traffic-light states from
`3d_traffic_lights_lightnet_status` instead of the V3 component,
`3d_traffic_lights_status`. The other RDS-HQ component formats are unchanged.
See the [RDS-HQ guide](rds-hq.md) for the input format.

## Render one clip

From the `wsm` package directory, install into an activated Python
environment:

```bash
pip install .
```

Rendering requires an NVIDIA GPU, OpenGL/EGL libraries (`libGL.so.1`,
`libEGL.so.1`), and FFmpeg with H.264 encoding. For containers, enable graphics
support with `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`.
See the [installation instructions](../README.md#install) for system packages,
including the headers needed when installing on Linux ARM.

Run this example from the same package directory to render the
[bundled RDS-HQ clip](../examples/rds_hq/README.md):

```bash
RDS_HQ_ROOT=examples/rds_hq
WSM_OUTPUT_ROOT=example_data/wsm4-output

cosmos-auto-wm \
    input_root="$RDS_HQ_ROOT" \
    output_root="$WSM_OUTPUT_ROOT" \
    version=v4 s3_spec=no_upload \
    camera_spec.camera_list='[camera_front_wide_120fov]' \
    input_fps_spec.input_pose_fps=30 \
    resolution_spec=resize_1280x720 \
    output_frame_spec.target_render_fps=30 \
    output_frame_spec.target_chunk_frame=297 \
    output_frame_spec.overlap_frame=0 output_frame_spec.max_chunk=1
```

The example writes one 1280×720 MP4 for the front-wide camera with 297 frames
at 30 FPS (9.9 seconds). The bundled clip includes seven calibrated F-theta
cameras. Replace the camera-list override with `camera_spec=internal_7cam` to
render all seven. Use a fresh output directory for each run.

For your own RDS-HQ data, change `RDS_HQ_ROOT`. The renderer processes all clips
under `pose/` in that directory. Set `camera_type` to `ftheta` or `pinhole` to
match the calibration, and select cameras by name with
`camera_spec.camera_list='[camera_front]'`. Set the input pose rate as described
in [Coordinates and time](rds-hq.md#coordinates-and-time). For other scenario
formats, use the [scenario-to-RDS-HQ skill](../skills/scenario-to-rds-hq/SKILL.md)
to create the input data.

Run `cosmos-auto-wm --help` to see the available options. Add `--cfg job --resolve`
to a render command to display the settings it will use without rendering.

## Visual conventions

**Changing the control-video color scheme requires retraining the model on
control videos rendered with the new scheme.** Use the same colors for training
and inference.

Colors below are configured RGB values before rasterization and lossy MP4 encoding.
All palette charts use the same dark background and light labels; their swatches
show the configured colors. Chart backgrounds and labels are presentation aids.

### Traffic signs

Signs use flat rectangular faces. Category is conveyed by color, without printed
text or the real sign's silhouette. The palette is
[`config_color_sign.json`](../cosmos_auto/wm/color_scheme/config_color_sign.json).

![Traffic-sign palette showing category colors, RGB values, and the dark-gray back face.](images/wsm4/palette-traffic-signs.png)

*Figure 1. Traffic-sign colors.*

| Category                       | Color     | RGB              |
| ------------------------------ | --------- | ---------------- |
| Stop                           | Red       | `(220, 20, 20)`  |
| Yield                          | Orange    | `(255, 140, 0)`  |
| Do not enter                   | Magenta   | `(200, 0, 100)`  |
| One-way, direction unspecified | Blue      | `(40, 110, 220)` |
| One-way left                   | Cyan      | `(0, 180, 230)`  |
| One-way right                  | Dark blue | `(30, 60, 200)`  |
| Pedestrian crossing            | Yellow    | `(255, 220, 0)`  |
| Back face                      | Dark gray | `(40, 40, 40)`   |

Signs outside these categories use the generic color `(8, 2, 255)`. Recognized
types, such as speed-limit signs, still have dark-gray backs. Unrecognized signs
use the generic color on both sides. See the [RDS-HQ sign classes](rds-hq.md#3d_traffic_signs)
for input labels.

### Traffic lights

The signal housing is a cuboid. Its front shows the state; sides and back stay
dark gray.

![Traffic-light palette showing solid states, left/up/right triangles, pedestrian and cyclist X/circle symbols, unknown, and off.](images/wsm4/palette-traffic-lights.png)

*Figure 2. Traffic-light colors and state symbols.*

Triangles indicate left, straight-ahead, or right directions. For pedestrian and
cyclist signals, red uses an X; green and yellow use circles in the corresponding
color. The white pedestrian go signal (`white_vru`) appears as a white circle.

Red-yellow states appear orange. A light observed with no active signal
(`stateless`) appears near-black; unknown states appear gray. See the
[traffic light states](rds-hq.md#traffic-light-states) for input fields and definitions.

The [traffic-light palette](../cosmos_auto/wm/color_scheme/config_color_traffic_light.json)
defines red `(240, 50, 50)`, yellow `(255, 200, 0)`, green `(130, 240, 132)`,
red-yellow `(208, 136, 72)`, unknown `(150, 150, 150)`, stateless `(20, 20, 20)`,
front housing `(105, 105, 105)`, and sides/back `(60, 60, 60)`.

### Road markings and boundaries

Seven painted arrow shapes are supported: `forward`, `left`, `right`,
`left_forward`, `right_forward`, `merge_left`, and `merge_right`. Each is fitted
and oriented to its annotated footprint and uses near-white `(244, 244, 244)`.
STOP areas use red `(255, 0, 0)`; KEEP CLEAR areas use green `(0, 255, 32)`.
Both remain filled polygons without lettering. `ROI_POLYGON_TEXT_PAINT` and
unrecognized markings use the generic mint-green `(20, 254, 185)` fill. TEXT PAINT
fills the supplied polygons without interpreting or generating text.

Arrow symbols require a supported class and a
[four-corner footprint](rds-hq.md#3d_road_markings). A five-corner footprint, for
example, is filled as a solid colored area instead of an arrow. Unsupported arrow
classes also use a solid fill. Polygons with no usable area are skipped.

KEEP CLEAR and TEXT PAINT footprints are subdivided into smaller triangles before
projection to better follow lens distortion. See the
[road-marking palette](../cosmos_auto/wm/color_scheme/config_color_road_marking.json)
and [arrow shapes](../cosmos_auto/wm/assets/road_marking_arrows/arrow_shape.json).

![Road-marking palette showing seven arrow shapes and the STOP, KEEP CLEAR, and other painted-area colors.](images/wsm4/palette-road-markings.png)

*Figure 3. Directional arrows and filled road markings.*

Road boundaries are solid polylines, grouped by `boundary_type`. This includes
subtypes whose names describe painted dots or dashes; boundary subtype changes
color, not the stroke pattern:

![Road-boundary palette showing the six subtype groups and their RGB values.](images/wsm4/palette-road-boundaries.png)

*Figure 4. Road-boundary colors.*

| Group                     | Included types                      | RGB               |
| ------------------------- | ----------------------------------- | ----------------- |
| Barrier/fence             | `barrier`, `fence`                  | `(253, 1, 232)`   |
| Curb                      | `curb`, `low_curb`, `tall_curb`     | `(207, 255, 191)` |
| Road boundary/seam        | `road_boundary`                     | `(0, 159, 96)`    |
| Obstacle bars/guide posts | `obstacle_bars`                     | `(255, 207, 159)` |
| Wall                      | `wall`                              | `(0, 128, 159)`   |
| Other                     | Remaining, absent, or unknown types | `(239, 191, 255)` |

The [boundary palette](../cosmos_auto/wm/color_scheme/config_color_road_boundary.json)
and [`road_boundary_utils.py`](../cosmos_auto/wm/road_boundary_utils.py) define the
exact grouping.

### Lanelines

Lane-line geometry uses the supplied color/style pair. White is `(255, 255, 255)`,
yellow is `(255, 255, 0)`, and an unrecognized pair uses `(181, 164, 71)`.
The [RDS-HQ laneline definitions](rds-hq.md#3d_lanelines) describe the annotated
features.

The exact settings are in the
[lane geometry/color configuration](../cosmos_auto/wm/color_scheme/config_color_geometry_laneline.json).

### Objects and other map features

Object boxes retain V3's five [object categories](rds-hq.md#object-tracks): Car,
Truck, Pedestrian, Cyclist, and Others. Filled cuboids have light-gray edges and
a dark-front/light-back category gradient. The gradient follows the object's front-to-back axis, independent of
camera distance. Edges use `(200, 200, 200)`. Box projection follows F-theta
distortion. For stationary tracks, the renderer averages dimensions and stabilizes
poses across the clip. Boxes that contain the camera center are suppressed.

![Five object-category gradients from dark front to light back, plus color swatches for poles, waitlines, and crosswalks.](images/wsm4/palette-objects-and-map-features.png)

*Figure 5. Object gradients and remaining map-feature colors.*

| Object category | Front RGB      | Back RGB          |
| --------------- | -------------- | ----------------- |
| Car             | `(0, 46, 136)` | `(126, 206, 255)` |
| Truck           | `(204, 55, 0)` | `(255, 192, 64)`  |
| Pedestrian      | `(148, 0, 62)` | `(255, 124, 171)` |
| Cyclist         | `(0, 80, 66)`  | `(102, 208, 198)` |
| Others          | `(53, 26, 20)` | `(166, 136, 125)` |

Crosswalk polygons are filled. Waitlines are drawn unless marked implicit.
Pole subtypes share one line style.

| Map feature | Geometry       | RGB              |
| ----------- | -------------- | ---------------- |
| Poles       | Polyline       | `(183, 69, 177)` |
| Waitline    | Polyline       | `(108, 179, 59)` |
| Crosswalks  | Filled polygon | `(139, 93, 255)` |

These features retain their V3 appearance. Their settings are in the
[box palette](../cosmos_auto/wm/color_scheme/config_color_bbox.json) and
[HD-map palette](../cosmos_auto/wm/color_scheme/config_color_hdmap.json).

## Annotated examples

![Annotated intersection with green and red solid signals, left-turn triangles, and the back of a traffic light.](images/wsm4/annotated-traffic-lights.png)

*Example A. Traffic-light states and orientation.*

![Annotated scene identifying a yield sign, pedestrian-crossing sign, sign back, curb, and other road marking.](images/wsm4/annotated-traffic-signs.png)

*Example B. Traffic-sign types.*

![Annotated road with a filled STOP marking, left-turn arrow, straight arrow, merge-left arrow, pole, and other road marking.](images/wsm4/annotated-road-markings.png)

*Example C. Road-marking types.*

![Annotated road identifying curb, road boundary, barrier or fence, solid and dashed lane lines, and the rear of a car.](images/wsm4/annotated-road-boundaries.png)

*Example D. Road-boundary types.*

## Cameras, frames, and output files

The default camera is front wide (`camera_spec=internal_1cam`). Use
`camera_spec=internal_7cam` for all seven cameras in the bundled clip. Every
selected camera needs poses and intrinsics. See the
[camera profiles](../cosmos_auto/wm/conf/camera_spec) for other rigs.

The Auto quickstart above explicitly selects one 297-frame chunk at 30 FPS.
Without these overrides, the general-purpose defaults produce up to three
complete chunks at 1280×720 and 10 FPS, with 100 frames per chunk and 45 frames
of overlap. Use
[`resolution_spec`](../cosmos_auto/wm/conf/resolution_spec) and
[`output_frame_spec`](../cosmos_auto/wm/conf/output_frame_spec) to choose other
output profiles.

Set `input_fps_spec.input_pose_fps` to the actual prepared pose rate. The sampler
selects pose frames at an integer stride; it does not interpolate a 10 Hz pose
timeline into 30 Hz. Prepare a synchronized, appropriately resampled timeline
first if needed. The pose rate must be at least, and an integer multiple of,
`output_frame_spec.target_render_fps`. Do not relabel 10 FPS controls as 30 FPS
at inference: that changes the motion timing the model assumes.

Videos are saved locally, one MP4 per camera and chunk:

```text
<output_root>/world_scenario/<camera_type>_<camera_name>/<clip_id>_<chunk_index>.mp4
```

Chunk indices start at `0`. To rerender after changing settings, use a new output
root or `skip_resume=true`.

## Validation and troubleshooting

Confirm the videos play and match the selected cameras, resolution, frame rate,
and frame count. The quickstart produces one 1280×720 video per selected camera,
with 297 frames at 30 FPS. Inspect a few frames for geometry alignment, sign facing,
and signal timing.

| Symptom                                            | Check                                                                                                |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `libEGL.so.1 not loaded` or context creation fails | Check the GPU and native OpenGL/EGL libraries in the rendering environment.                          |
| Missing camera key / empty pose stack              | Ensure every camera in `camera_spec` has poses and intrinsics.                                       |
| All signals appear unknown                         | Check that the traffic-light state archive exists and its state sequences cover the rendered frames. |
| No MP4 for a short clip                            | Supply enough frames for the selected output profile, or use `allow_short_final_chunk=true`.         |
| Output unchanged after a config change             | Use a new output root or `skip_resume=true`.                                                         |
| Misaligned geometry                                | Check world/camera transforms, calibration, and frame timestamps.                                    |
