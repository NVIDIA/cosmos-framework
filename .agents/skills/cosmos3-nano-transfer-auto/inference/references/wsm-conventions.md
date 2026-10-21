# WSM control checks

Both Auto models use **WSM v4.0.1** layout controls. The renderer's configuration
selector is still `version=v4`; do not change it to `v4.0.1` or infer a renderer
version from a sample directory name. Use the inspection legend below,
matching camera calibration and verified renderer settings. WSM3 controls are not
a verified substitute. Do not certify a renderer build from colors alone.

WSM colors describe classes and signal states, not realistic paint, light, weather
or geographic setting. Lossy video compression changes exact pixel values.

| Feature                        | Expected meaning                                                                                                           |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| Dynamic objects                | Filled cuboids with light-gray edges and a category gradient along the object's front-to-back axis; not distance shading   |
| Signs                          | Flat category-colored faces without sign text or realistic silhouettes; recognized backs are dark gray                     |
| Signals                        | Front face conveys state; side/back faces stay dark gray; directional triangles and pedestrian/cyclist symbols matter      |
| Unknown / stateless signals    | Gray / near-black respectively; neither implies a known red or green                                                       |
| Road arrows                    | Supported directional shapes follow annotated footprints; unsupported or unsuitable footprints can become solid fills      |
| STOP / KEEP CLEAR areas        | Red / green filled areas without lettering                                                                                 |
| Boundaries                     | Color distinguishes barrier/fence, curb, seam, guide posts, wall and other; boundary subtype does not imply dashed strokes |
| Lane lines                     | Supplied color/style geometry, separate from road boundaries                                                               |
| Crosswalks / waitlines / poles | Filled polygons / lines / lines; use the map-feature colors below                                                          |

## Objects

Reference object-category front → back RGB:

| Class      | Front      | Back          |
| ---------- | ---------- | ------------- |
| Car        | (0,46,136) | (126,206,255) |
| Truck      | (204,55,0) | (255,192,64)  |
| Pedestrian | (148,0,62) | (255,124,171) |
| Cyclist    | (0,80,66)  | (102,208,198) |
| Other      | (53,26,20) | (166,136,125) |

Cuboid edges use (200,200,200). The gradient indicates object facing, not paint or
distance. Projection must follow the camera lens model; F-theta distortion matters.
The reference renderer stabilizes stationary tracks and averages their dimensions
across the clip. A box containing the camera center is suppressed. A coarse class
does not establish an exact vehicle subtype or a persistent identity.

## Signs

Signs are flat rectangular faces, not realistic sign silhouettes or printed text.

| Category                       | RGB          |
| ------------------------------ | ------------ |
| Stop                           | (220,20,20)  |
| Yield                          | (255,140,0)  |
| Do not enter                   | (200,0,100)  |
| One-way, direction unspecified | (40,110,220) |
| One-way left                   | (0,180,230)  |
| One-way right                  | (30,60,200)  |
| Pedestrian crossing            | (255,220,0)  |
| Recognized sign's back         | (40,40,40)   |
| Other / unrecognized           | (8,2,255)    |

Recognized types outside the named categories, such as speed-limit signs, use the
generic front color and dark back. Unrecognized signs use the generic color on
both sides. Do not infer a printed speed limit or other text from a generic face.

## Signals

| State or surface                             | RGB           |
| -------------------------------------------- | ------------- |
| Red                                          | (240,50,50)   |
| Yellow                                       | (255,200,0)   |
| Green                                        | (130,240,132) |
| Red-yellow                                   | (208,136,72)  |
| White pedestrian go (`white_vru`)            | (255,255,255) |
| Unknown state                                | (150,150,150) |
| Observed with no active signal (`stateless`) | (20,20,20)    |
| Front housing                                | (105,105,105) |
| Side / back housing                          | (60,60,60)    |

Triangles indicate left, straight-ahead or right. Pedestrian/cyclist red states
use an X; green and yellow use circles in that color. White pedestrian go is a
white circle. Unknown does not mean off, red or green. Different visible signals
can govern different approaches; a signal in the image is not necessarily for ego.

## Road markings and boundaries

Seven directional arrow classes are drawn near-white (244,244,244): `forward`,
`left`, `right`, `left_forward`, `right_forward`, `merge_left`, `merge_right`.
They need a supported class and four-corner footprint. Unsupported classes or
unsuitable footprints, such as five corners, become solid fills; degenerate areas
are skipped. Missing arrow detail is not evidence of a missing road marking.

| Painted area                      | RGB          | Meaning                                |
| --------------------------------- | ------------ | -------------------------------------- |
| STOP                              | (255,0,0)    | Filled polygon, no lettering           |
| KEEP CLEAR                        | (0,255,32)   | Filled polygon, no lettering           |
| TEXT PAINT / unrecognized marking | (20,254,185) | Generic fill; does not encode the text |

KEEP CLEAR and TEXT PAINT polygons are subdivided before projection to follow
lens distortion. Road boundaries are **solid polylines**, even when an input
subtype's name mentions dots or dashes:

| Boundary group              | Included types                     | RGB           |
| --------------------------- | ---------------------------------- | ------------- |
| Barrier / fence             | `barrier`, `fence`                 | (253,1,232)   |
| Curb                        | `curb`, `low_curb`, `tall_curb`    | (207,255,191) |
| Road boundary / seam        | `road_boundary`                    | (0,159,96)    |
| Obstacle bars / guide posts | `obstacle_bars`                    | (255,207,159) |
| Wall                        | `wall`                             | (0,128,159)   |
| Other                       | Remaining, absent or unknown types | (239,191,255) |

## Lane lines and other map features

Lane lines are distinct from road boundaries. Their supplied color/style pair
controls the stroke. White is (255,255,255), yellow (255,255,0), and an unrecognized
pair (181,164,71). The reference mapping is:

| Color/style pair            | Pattern                    | Relative thickness |
| --------------------------- | -------------------------- | ------------------ |
| `WHITE SOLID_SINGLE`        | `solid`                    | 1                  |
| `WHITE LONG_DASHED_SINGLE`  | `long_dashed`              | 1                  |
| `WHITE SHORT_DASHED_SINGLE` | `short_dashed`             | 1                  |
| `WHITE DOT_DASHED_SINGLE`   | `dot_dashed`               | 1                  |
| `WHITE OTHER`               | `solid`                    | 3                  |
| `WHITE SOLID_GROUP`         | `solid`, `solid`           | 1                  |
| `YELLOW DOT_SOLID_SINGLE`   | `dotted_1_9`               | 1                  |
| `YELLOW DOT_SOLID_GROUP`    | `dotted_1_9`, `dotted_1_9` | 1                  |
| `YELLOW DASHED_SOLID`       | `solid`, `long_dashed`     | 1                  |
| `YELLOW SOLID_DASHED`       | `long_dashed`, `solid`     | 1                  |
| `YELLOW SOLID_SINGLE`       | `solid`                    | 1                  |
| `YELLOW SOLID_GROUP`        | `solid`, `solid`           | 1                  |
| `YELLOW LONG_DASHED_SINGLE` | `long_dashed`              | 1                  |
| `YELLOW OTHER`              | `solid`                    | 3                  |
| `OTHER`                     | `solid`                    | 1                  |

Two listed patterns form a dual line, in renderer configuration order; do not
infer their image-left/image-right placement from the class name alone.

| Map feature | Geometry                               | RGB          |
| ----------- | -------------------------------------- | ------------ |
| Pole        | Polyline                               | (183,69,177) |
| Waitline    | Polyline, omitted when marked implicit | (108,179,59) |
| Crosswalk   | Filled polygon                         | (139,93,255) |

## Renderer evidence and geometry

Cosmos Framework includes an independently installable renderer under
`.agents/skills/cookbooks/cosmos3/simulation/auto/wsm/`.
Follow its [README](../../../cookbooks/cosmos3/simulation/auto/wsm/README.md) for its separate Python environment, GPU/EGL and FFmpeg
dependencies and license. Follow the [conversion skill](../../../cookbooks/cosmos3/simulation/auto/wsm/skills/scenario-to-rds-hq/SKILL.md) for
scenario conversion when needed. This caption/input bundle includes an inspection
legend, not a replacement renderer or a converter for an undocumented sensor format.
Inspect these renderer-package-relative resources:

- `docs/wsm4.md`: WSM v4.0.1 behavior and rendering instructions.
- `docs/rds-hq.md`: class conversion inputs, coordinates, calibration and clocks.
- `cosmos_auto/wm/color_scheme/config_color_*.json`: the palettes; lane geometry
  is in `config_color_geometry_laneline.json`.
- `cosmos_auto/wm/conf/`: selected version, projection, camera and output profiles.

From the installed renderer's `wsm/` directory, a local configuration check is:

```bash
cosmos-auto-wm \
  input_root="${RDS_HQ_ROOT:?set your converted local scenario}" \
  output_root="${WSM_OUTPUT:?set a new output directory}" \
  version=v4 \
  output_frame_spec.target_render_fps=30 \
  output_frame_spec.target_chunk_frame=400 \
  output_frame_spec.overlap_frame=0 output_frame_spec.max_chunk=1 \
  --cfg job --resolve
```

Select the matching camera rig, crop and resolution in that configuration before
rendering. After inspection, remove `--cfg job --resolve` to render into the
new directory. The 400-frame request is for training coverage; use the chosen
inference interval when preparing inference-only controls. The bundled renderer
scene has annotations and calibration but no recorded RGB/LiDAR, so it does not
replace the paired [daytime example](../../shared/reference-scene.md) or your
own training targets.

Record the renderer build and resolved settings, including `version=v4`. For its
RDS-HQ inputs, WSM v4.0.1 reads `3d_traffic_lights_lightnet_status`, not the older
`3d_traffic_lights_status`; check state coverage and timing, not just box presence.
The reference renderer's default output is 10 FPS, not the model's 30 FPS RGB
clock. Configure and verify the requested sampling interval; changing MP4 FPS
metadata alone does not synchronize or resample scene content.

Verify projection type, optical-axis direction, FOV, crop and timestamps for every
camera. F-theta and pinhole controls are not interchangeable by renaming files.
Inspect controls against source RGB when available, including start/end and ambiguous
events. Check that upstream classes were converted correctly: a coarse truck label
may have originated from construction equipment.

If renderer provenance or a needed class meaning is unknown, report that specific
uncertainty and avoid the affected claims. Do not repair controls, recolor palettes
or infer a version from the filename without authorization and evidence.
