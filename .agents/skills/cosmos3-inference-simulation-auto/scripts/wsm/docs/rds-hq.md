# RDS-HQ data format

RDS-HQ stores camera calibration, poses, object tracks, map geometry, and
traffic-light states. **RDS-HQ is the input; WSM4 is the rendered representation.**
This reference defines the data consumed by `cosmos_auto.wm`. See
[WSM4 rendering](wsm4.md) for visual conventions and commands.

For scenarios in another format, use the
[scenario-to-RDS-HQ conversion skill](../skills/scenario-to-rds-hq/SKILL.md).

The tables specify all fields needed for the default WSM4 map layers.
“Required” means required for a complete, unambiguous dataset, even where
a loader tolerates omission. Class definitions describe the annotated scene
features. Expand each component's class table for
the supported vocabulary and its meaning.

- [Files and required layers](#files-and-required-layers)
- [Coordinates and time](#coordinates-and-time)
- [Camera poses and calibration](#camera-poses-and-calibration)
- [Object tracks](#object-tracks)
- [Static map structure](#static-map-structure)
- [Map components](#map-components)
- [Traffic light states](#traffic-light-states)

The [bundled example clip](../examples/rds_hq/README.md) provides real archives to
inspect alongside these schemas: seven-camera calibration and poses, all five
object categories, all eight map components, and traffic-light states. The archives
retain only the fields used by WSM4; RGB video is not included.

## Files and required layers

Organize the dataset **by component, then clip**. Every component uses the same
clip ID. The conventional ID is `<session_id>_<start_us>_<end_us>`, with system
timestamps in microseconds and an exclusive end. The loader treats the ID as
opaque and does not infer frame counts from it. Use letters, digits, underscores,
and hyphens, without dots or path separators, for WebDataset-compatible names.

```text
rds-hq/
├── pose/<clip_id>.tar
├── pinhole_intrinsic/<clip_id>.tar     # or ftheta_intrinsic
├── all_object_info/<clip_id>.tar
├── 3d_lanelines/<clip_id>.tar
├── 3d_road_boundaries/<clip_id>.tar
├── 3d_crosswalks/<clip_id>.tar
├── 3d_wait_lines/<clip_id>.tar
├── 3d_poles/<clip_id>.tar
├── 3d_traffic_signs/<clip_id>.tar
├── 3d_road_markings/<clip_id>.tar
├── 3d_traffic_lights/<clip_id>.tar
└── 3d_traffic_lights_lightnet_status/<clip_id>.tar
```

Each uncompressed tar contains **one WebDataset sample**. Name every member
`<clip_id>.<decoded key>`, without a directory prefix. For example,
`session_0_21000000.000000.pose.camera_front_wide_120fov.npy` decodes to
`000000.pose.camera_front_wide_120fov.npy`. WebDataset adds `__key__`; do not write
it as a member. JSON is UTF-8. NPY arrays are numeric, without pickled objects.
JSON geometry and dimensions contain numbers, not numeric strings.
Frame IDs have six digits and begin at `000000`.

See [Coordinates and time](#coordinates-and-time) for coordinate conventions
and frame timing.

| Component directory                 | Decoded key(s)                   | Required content                                                                                                  |
| ----------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `pose`                              | `<frame>.pose.<camera>.npy`      | Camera-to-world pose per frame and camera                                                                         |
| `pinhole_intrinsic`                 | `pinhole_intrinsic.<camera>.npy` | Six values for `camera_type=pinhole`                                                                              |
| `ftheta_intrinsic`                  | `ftheta_intrinsic.<camera>.npy`  | Eleven or fourteen values for `camera_type=ftheta`; use the intrinsic component matching the selected camera type |
| `all_object_info`                   | `<frame>.all_object_info.json`   | Tracked boxes; `{}` for an empty frame                                                                            |
| `3d_lanelines`                      | `lanelines.json`                 | Painted lines and color/style classes                                                                             |
| `3d_road_boundaries`                | `road_boundaries.json`           | Road boundaries and subtypes                                                                                      |
| `3d_crosswalks`                     | `crosswalks.json`                | Crossing polygons                                                                                                 |
| `3d_wait_lines`                     | `wait_lines.json`                | Stop/wait lines and explicit/implicit flags                                                                       |
| `3d_poles`                          | `poles.json`                     | Pole polylines                                                                                                    |
| `3d_traffic_signs`                  | `traffic_signs.json`             | Sign cuboids and classes                                                                                          |
| `3d_road_markings`                  | `road_markings.json`             | Painted polygons and classes                                                                                      |
| `3d_traffic_lights`                 | `traffic_lights.json`            | Signal cuboids and stable feature IDs                                                                             |
| `3d_traffic_lights_lightnet_status` | `aggregated_states.json`         | Signal IDs, basic-state mapping, and frame sequences                                                              |

Keep all eight geometry tars even when a layer is empty. Write `{"labels":[]}`
inside the correct JSON member; an empty or missing tar is not an empty map.
Source RGB and `3d_lanes` are unnecessary for the default WSM4 render.

## Coordinates and time

Use one fixed local world frame for the entire clip. Its reference origin is the
**center of the ego vehicle's rear axle in the first frame, projected onto the
ground**, with axes aligned to the vehicle's FLU axes at that frame. Keep this
frame fixed as the vehicle moves. Camera poses, object poses, and map vertices
must all refer to this same world frame.

| Quantity        | Convention                                                                                                   |
| --------------- | ------------------------------------------------------------------------------------------------------------ |
| World geometry  | Metres, right-handed, z up; origin at the first frame's rear-axle ground projection                          |
| Vehicle frame   | Origin at the rear-axle ground projection; FLU: x forward, y left, z up                                      |
| Camera frame    | OpenCV: x right, y down, z forward                                                                           |
| Rigid transform | 4×4 `T = [[R,t],[0,0,0,1]]`; column vectors, `point_world = R @ point_local + t`                             |
| Rotation        | Orthonormal 3×3 matrix with determinant +1                                                                   |
| Camera pose     | Camera-to-world, including vehicle motion and camera extrinsics                                              |
| Object pose     | Object-to-world at the **geometric box center**; local x along length/heading, y along width, z along height |
| Map vertices    | Already in world coordinates; a `coordinate_frame` attribute does not transform them                         |

Camera poses satisfy `camera_to_world = vehicle_to_world @ camera_to_vehicle`.
The inverse is the world-to-camera transform.

With this convention, **`vehicle_to_world[0]` is expected to be the 4×4 identity
matrix**: identity rotation and zero translation. Consequently,
`camera_to_world[0] = camera_to_vehicle`; the first camera pose includes its
mounting offset and orientation and is generally not identity. Existing datasets
may retain a small initial vehicle-pose offset from the reference frame.

Object translation locates the geometric box center, not its bottom center or
rear axle. Rotation is stored as a matrix, not Euler angles or a quaternion.

By default, the renderer assumes camera poses and video frames are sampled at 30 Hz.
At this rate, frame `i` represents `i / 30` seconds from the clip's start.

**Every selected camera must have a pose for every frame `0..T-1`, with no gaps.**
The renderer stacks poses in filename order and does not interpolate missing
poses. For example, storing only `000000`, `000003`, `000006`, … places those
poses at consecutive array indices `0`, `1`, `2`, …; it does not preserve the
missing time steps. **Dropping pose files can silently misalign camera motion,
object tracks, and traffic-light states.**

Frame indices identify one shared sampling timeline, not independent frame
numbers from each camera recording. The renderer does not read a timestamp
array per camera. `input_fps_spec` is the renderer's configuration for source
sampling rates. Its `input_pose_fps` field specifies camera poses per second
(default `30`), so frame `i` represents `i / input_pose_fps` seconds.

For lower-rate source poses, either resample them onto a complete 30 Hz timeline
before export, or use consecutive frame IDs at the source rate and set
`input_fps_spec.input_pose_fps` to that rate in the render command (for example,
`input_fps_spec.input_pose_fps=10` for 10 Hz poses). Align object frame IDs,
traffic-light state sequences, and any camera video with the chosen timeline;
changing the rate setting alone does not resample or synchronize the data.

For a 30 Hz pose timeline, object observations at 10 Hz use keys `000000`,
`000003`, `000006`, …. Full-rate observations are also supported; `{}` denotes a
frame without objects. Sparse observations retain these pose-frame indices,
with stable IDs and records at **both frame `0` and frame `T-1`**, even when the
last pose does not fall on the regular object sampling interval. Use `{}` at a
boundary where no objects are present. The loader interpolates only IDs shared
by the bracketing observations; without boundary records, it copies the nearest
observation outside the recorded range and can extend tracks beyond their lifetime.

Traffic-light state arrays have `T` entries, indexed by the **original pose
frame**. At 10 FPS rendering from 30 Hz input, states are read at 0, 3, 6, …;
chunks do not restart those indices.

**Output FPS** is the frame rate of the rendered WSM4 video. It must be a
positive integer divisor of the input pose rate.

## Camera poses and calibration

### `pose`

Each `<frame>.pose.<camera>.npy` stores a 4×4 NumPy matrix describing one camera's
position and orientation at that frame. The matrix converts 3D points from the
camera's coordinates (x right, y down, z forward) into the clip's shared world
coordinates.

| Field        | Type/shape                 | Definition                                                                                                                 |
| ------------ | -------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Entire array | float32 or float64 `[4,4]` | Camera-to-world rigid transform                                                                                            |
| `[:3,:3]`    | `[3,3]`                    | Top-left block: camera orientation; its columns give the camera's right, down, and forward directions in world coordinates |
| `[:3,3]`     | `[3]`                      | First three entries of the last column: camera optical-center position in world coordinates, in metres                     |
| `[3,:]`      | `[4]`                      | Exactly `[0,0,0,1]`                                                                                                        |

Choose a lowercase name for each camera and use it consistently in its pose and
calibration keys. Use the same name in `camera_spec.camera_list`, the renderer
setting that selects which cameras to render. For a camera named `camera_front`
with pinhole calibration:

| Item                                   | Example                                  |
| -------------------------------------- | ---------------------------------------- |
| Decoded pose key for frame 0           | `000000.pose.camera_front.npy`           |
| Decoded calibration key                | `pinhole_intrinsic.camera_front.npy`     |
| Camera selection in the render command | `camera_spec.camera_list=[camera_front]` |

Select the projection model with `camera_type=pinhole` or `camera_type=ftheta`,
matching the calibration format described below. Each camera has one calibration
array, called its **intrinsics**, which stays constant throughout the clip.

### `pinhole_intrinsic`

Required NPY fields, in order: `[fx, fy, cx, cy, width, height]`.

| Field             | Unit   | Definition                                                    |
| ----------------- | ------ | ------------------------------------------------------------- |
| `fx`, `fy`        | Pixels | Positive horizontal/vertical focal lengths                    |
| `cx`, `cy`        | Pixels | Principal point in the calibrated image                       |
| `width`, `height` | Pixels | Positive integer-valued image dimensions in the numeric array |

Class: `pinhole`, selected with `camera_type=pinhole`. Projection is
`u = fx*X/Z + cx`, `v = fy*Y/Z + cy` for camera-space `Z > 0`. This model has no
distortion coefficients and describes an undistorted image.

### `ftheta_intrinsic`

Required NPY fields, in this exact order, with an optional three-value suffix:

```text
[cx, cy, width, height, p0, p1, p2, p3, p4, p5, is_bw_poly]
[cx, cy, width, height, p0, p1, p2, p3, p4, p5, is_bw_poly, c, d, e]
```

| Field             | Requirement       | Definition                                                       |
| ----------------- | ----------------- | ---------------------------------------------------------------- |
| `cx`, `cy`        | Required          | Principal point in pixels                                        |
| `width`, `height` | Required          | Positive integer-valued calibrated image dimensions              |
| `p0..p5`          | Required          | Polynomial coefficients in ascending power order                 |
| `is_bw_poly`      | Required          | `1`: radius-to-angle polynomial; `0`: angle-to-radius polynomial |
| `c`, `d`, `e`     | Optional together | Linear correction `A = [[c,d],[e,1]]`; defaults `[1,0,0]`        |

All six polynomial coefficient slots are present. Coefficients above the
polynomial's degree are zero.

Class: `ftheta`, selected with `camera_type=ftheta`. Let `theta` be the ray angle
from the optical axis in radians and `r` the pixel radius before linear correction.
Backward: `theta = p0 + p1*r + ... + p5*r^5`.
Forward: `r = p0 + p1*theta + ... + p5*theta^5`.
For a non-axial ray, the pixel is
`[cx,cy] + A @ (r * [X,Y] / sqrt(X²+Y²))`. Require `c - d*e != 0` and a calibrated
polynomial valid over the field of view.

These coefficients describe the specified F-theta polynomial, not another
distortion model. Intrinsics match the calibrated image resolution. The renderer subsequently applies its
configured resize/crop.

## Object tracks

Object tracks describe actors and other objects that can potentially move.
Standalone static objects such as cones and construction props are outside
RDS-HQ's object classes. A stopped or parked vehicle remains in its vehicle
class; observed motion is recorded separately by `object_is_moving`.

<details>
<summary>Object classes and accepted aliases</summary>

All canonical classes and recognized aliases:

| Class        | Definition                                                                             | Additional input aliases                                            |
| ------------ | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `Car`        | Passenger/light vehicle                                                                | `Vehicle`, `Automobile`, `Other_vehicle`                            |
| `Truck`      | Large/heavy vehicle, including buses, trailers, and rail vehicles                      | `Bus`, `Heavy_truck`, `Train_or_tram_car`, `Trolley_bus`, `Trailer` |
| `Pedestrian` | Person on foot                                                                         | `Person`                                                            |
| `Cyclist`    | Cycle/rider category                                                                   | `Rider`                                                             |
| `Others`     | Other potentially moving objects, including protruding objects, strollers, and animals | Unrecognized strings fall back to this class in the renderer        |

</details>

Each `<frame>.all_object_info.json` maps stable string track IDs to records:

```json
{
  "car-1": {
    "object_type": "Car",
    "object_is_moving": true,
    "object_lwh": [4.5, 1.8, 1.6],
    "object_to_world": [[1,0,0,18], [0,1,0,0], [0,0,1,0.8], [0,0,0,1]]
  }
}
```

| Required attribute | Type                   | Definition                                                                                                                                                                        |
| ------------------ | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dictionary key     | String                 | Stable track ID; never reuse for another object in the clip                                                                                                                       |
| `object_type`      | String                 | Case-sensitive class from the table above                                                                                                                                         |
| `object_is_moving` | Boolean                | `true` if the object moves in world coordinates at any point during the clip; `false` only if it remains stationary throughout. Use the same value in every record for that track |
| `object_lwh`       | Three positive numbers | Length, width, height in metres, along local x, y, z                                                                                                                              |
| `object_to_world`  | 4×4 numeric list       | Centered box pose in world coordinates                                                                                                                                            |

For example, a parked car that later pulls out has `object_is_moving=true` in
every record, including the initially parked frames. Its per-frame
`object_to_world` poses describe when it is stationary and when it moves.

The renderer uses `object_is_moving=false` to average box dimensions and stabilize
the track's world pose across frames, reducing annotation jitter. `true` bypasses
this stabilization. Switching the flag to `false` during separate stops could
cause the renderer to average those different stopping locations. Include the
flag in every object record; the renderer reads it directly.

Prefer canonical names; lowercase names are not automatic aliases. Omit an
absent object from that frame and use `{}` for an empty frame. Optional `aux_info`
is not required. Keep track identity, dimensions, and motion flags consistent
across observations. The renderer's fallback for unrecognized strings does not
make static objects part of `Others`.

## Static map structure

Each map component stores one set of fixed features for the entire clip. Record
each feature once, using world coordinates. All map components share these fields:

| Field                                                          | Type                                                | Definition                                                                                                      |
| -------------------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `labels`                                                       | List                                                | One entry per feature; `[]` when the component is empty                                                         |
| `labelData.shape3d`                                            | Object inside each label                            | The feature's geometry and attributes                                                                           |
| `shape3d.attributes`                                           | List of objects                                     | Named properties such as a boundary type or line color; omit or use `[]` when the component needs no attributes |
| `shape3d.polyline3d`, `shape3d.surface`, or `shape3d.cuboid3d` | Object                                              | The geometry type required by the component: a line, polygon, or box                                            |
| `vertices`                                                     | List of `[x,y,z]` points inside the geometry object | Finite numeric coordinates in metres, in the shared world frame                                                 |

This example describes one curb as a line between two points:

```json
{
  "labels": [{
    "labelData": {"shape3d": {
      "attributes": [{"name": "boundary_type", "text": "curb"}],
      "polyline3d": {"vertices": [[0,4,0], [30,4,0]]}
    }}
  }]
}
```

Each attribute has a `name` and a value in the format shown for that component.
Here, the `boundary_type` attribute has the text value `curb`. Keep attributes in
a list, even when there is only one.

Use `{"labels":[]}` for an empty component. Additional export metadata, such as
`labelId`, timestamps, and unit or version fields, is not required.

Polylines need at least two distinct vertices. For polygons, list at least three
vertices in order around the perimeter, without repeating the first vertex at
the end. The outline must enclose nonzero area and must not cross itself when
viewed from above (in the XY plane). Split disconnected regions and polygons
with holes into separate simple polygons. Preserve vertex elevations, especially
on stacked roads. Mildly non-planar surveyed road surfaces are acceptable.

### Sign and signal cuboid order

Each cuboid contains eight explicit vertices. Define local x as the outward front-face normal,
local y across the face, and local z upward. With half-dimensions
`a=thickness/2`, `b=width/2`, `c=height/2`, use this order:

| Index | Local coordinate | Index | Local coordinate |
| ----- | ---------------- | ----- | ---------------- |
| 0     | `[-a,-b,-c]`     | 4     | `[-a,-b,+c]`     |
| 1     | `[-a,+b,-c]`     | 5     | `[-a,+b,+c]`     |
| 2     | `[+a,+b,-c]`     | 6     | `[+a,+b,+c]`     |
| 3     | `[+a,-b,-c]`     | 7     | `[+a,-b,+c]`     |

Transform each vertex by `R @ local_vertex + center`. The front face is
`[2,3,7,6]`, the back `[0,1,5,4]`. Point local +x toward traffic that should see
the face. Center, dimensions, and rotation alone do not replace these eight vertices.

## Map components

The following sections describe each component's fields inside
`labelData.shape3d`, using the [static map structure](#static-map-structure) above.

### `3d_lanelines`

Painted lines dividing or bounding lanes.

| Required attribute     | Type/example                                                   | Definition                           |
| ---------------------- | -------------------------------------------------------------- | ------------------------------------ |
| `polyline3d.vertices`  | `[N,3]`, `N >= 2`                                              | Ordered line or line-pair centerline |
| `attributes`: `colors` | `{"name":"colors","enumsList":{"enumsList":["WHITE"]}}`        | Color of the whole line              |
| `attributes`: `styles` | `{"name":"styles","enumsList":{"enumsList":["SOLID_SINGLE"]}}` | Style of the whole line              |

The `attributes` list contains two records: one named `colors` and one named
`styles`. Each record stores its value as a one-element list of strings at
`enumsList.enumsList`, such as `["WHITE"]` or `["SOLID_SINGLE"]`. The renderer
reads only the first string; empty lists are invalid and extra strings are
ignored. If a line changes color or style, split it into separate labels at the
change.

<details>
<summary>Laneline classes and definitions</summary>

`SINGLE` describes one lane line; `GROUP` describes a double lane line represented
by one annotation. Color describes the physical marking. Long and short dashes
are relative marking styles, without fixed lengths in the data format.

| Color    | Style                 | Definition                                                                       |
| -------- | --------------------- | -------------------------------------------------------------------------------- |
| `WHITE`  | `SOLID_SINGLE`        | Single solid white lane line                                                     |
| `WHITE`  | `LONG_DASHED_SINGLE`  | Single white lane line with long dashed segments                                 |
| `WHITE`  | `SHORT_DASHED_SINGLE` | Single white lane line with short dashed segments                                |
| `WHITE`  | `DOT_DASHED_SINGLE`   | Single white lane line made up of dots and dashes                                |
| `WHITE`  | `OTHER`               | White lane marking with a style outside the listed classes                       |
| `WHITE`  | `SOLID_GROUP`         | Double solid white lane line                                                     |
| `YELLOW` | `DOT_SOLID_SINGLE`    | Single dotted yellow lane line                                                   |
| `YELLOW` | `DOT_SOLID_GROUP`     | Double dotted yellow lane line                                                   |
| `YELLOW` | `DASHED_SOLID`        | Double yellow lane line with dashed and solid components                         |
| `YELLOW` | `SOLID_DASHED`        | Double yellow lane line with solid and dashed components                         |
| `YELLOW` | `SOLID_SINGLE`        | Single solid yellow lane line                                                    |
| `YELLOW` | `SOLID_GROUP`         | Double solid yellow lane line                                                    |
| `YELLOW` | `LONG_DASHED_SINGLE`  | Single yellow lane line with long dashed segments                                |
| `YELLOW` | `OTHER`               | Yellow lane marking with a style outside the listed classes                      |
| `OTHER`  | `OTHER`               | Lane marking whose color and style are unspecified or outside the listed classes |

Keep polyline vertex order consistent. A double line uses one path representing
the pair. The [WSM4 laneline guide](wsm4.md#lanelines) describes its visual encoding.

</details>

### `3d_road_boundaries`

| Required attribute            | Type/example                             | Definition                                   |
| ----------------------------- | ---------------------------------------- | -------------------------------------------- |
| `polyline3d.vertices`         | `[N,3]`, `N >= 2`                        | Ordered boundary path                        |
| `attributes`: `boundary_type` | `{"name":"boundary_type","text":"curb"}` | Boundary type below, normalized to lowercase |

<details>
<summary>Road-boundary types and definitions</summary>

The six boundary types and their accepted input values:

| Type                        | Definition                                                                                                       | Input `boundary_type`                                                             |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Barrier/fence               | Protective barrier, guardrail, or fence along the road edge                                                      | `barrier`, `fence`                                                                |
| Curb                        | Raised edge separating the roadway from an adjoining surface                                                     | `curb`, `low_curb`, `tall_curb`                                                   |
| Road boundary (seam)        | Road edge or seam between the road surface and an adjoining surface, such as a transition from pavement to grass | `road_boundary`                                                                   |
| Obstacle bars (guide posts) | Row of bars or guide posts marking the road limit                                                                | `obstacle_bars`                                                                   |
| Wall                        | Wall defining the road edge                                                                                      | `wall`                                                                            |
| Other                       | Other painted or physical road limit, or an unspecified boundary                                                 | `other_marking`, `other_physical`, or any remaining value; also used when missing |

</details>

### `3d_crosswalks`

Required geometry: `surface.vertices`, an ordered `[N,3]` polygon with `N >= 3`.
Required semantic attributes: none; omit `attributes` or use `[]`.

Class: **crosswalk**, a marked area where pedestrians cross the roadway, with
separate polygons for separate painted regions. No subclass or stripe-style
attribute is required.

### `3d_wait_lines`

| Required attribute          | Type/example                          | Definition                                      |
| --------------------------- | ------------------------------------- | ----------------------------------------------- |
| `polyline3d.vertices`       | `[N,3]`, `N >= 2`                     | Ordered stop/wait line                          |
| `attributes`: `is_implicit` | `{"name":"is_implicit","flag":false}` | Whether the line is virtual rather than painted |

| Class value | Definition                                                      |
| ----------- | --------------------------------------------------------------- |
| `false`     | Explicit painted line marking where traffic should stop or wait |
| `true`      | Inferred stopping boundary with no corresponding painted line   |

A missing flag is treated as explicit. No additional `type` enum is required.

### `3d_poles`

Required geometry: `polyline3d.vertices`, `[N,3]` with `N >= 2`, along the pole,
typically bottom to top. Required semantic attributes: none; omit `attributes`
or use `[]`.

Class: **pole**, a slender upright support, such as a signpost, signal pole, or
streetlight pole. Its polyline follows the support's axis. No pole-radius or
`pole_type` attribute is required.

### `3d_traffic_signs`

| Required attribute         | Type/example                                                     | Definition                                                          |
| -------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------- |
| `cuboid3d.vertices`        | `[8,3]`                                                          | Sign cuboid in the [specified order](#sign-and-signal-cuboid-order) |
| `attributes`: `sign_class` | `{"name":"sign_class","text":"TRAFFIC_SIGN_REGULATORY_R1_STOP"}` | Physical sign class as seen facing it                               |

Keep recognized `sign_class` values even when they fall under Other below. For
example, a speed-limit sign should keep `TRAFFIC_SIGN_REGULATORY_R2_SPEED_LIMIT`.
Replacing it with `OTHER` loses the information the renderer uses to distinguish
the front and back.

<details>
<summary>Traffic-sign classes and definitions</summary>

The `sign_class` attribute identifies the physical sign. The loader strips
whitespace and uppercases the value.

| Class                | Input `sign_class`                                                                                     | Definition                                                                         |
| -------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| Stop                 | `TRAFFIC_SIGN_REGULATORY_R1_STOP`                                                                      | Sign instructing traffic to stop                                                   |
| Yield                | `TRAFFIC_SIGN_REGULATORY_R1_YIELD`                                                                     | Sign instructing traffic to give way                                               |
| Do not enter         | `TRAFFIC_SIGN_REGULATORY_R5_DO_NOT_ENTER`, `TRAFFIC_SIGN_GERMAN_BASICS_NO_ENTER`                       | Sign prohibiting entry from the direction it faces                                 |
| One-way left         | `TRAFFIC_SIGN_REGULATORY_R6_ONEWAY_ARROW_LEFT`, `TRAFFIC_SIGN_REGULATORY_R6_ONEWAY_RECTANGULAR_LEFT`   | Sign indicating one-way travel to the left when facing it                          |
| One-way right        | `TRAFFIC_SIGN_REGULATORY_R6_ONEWAY_ARROW_RIGHT`, `TRAFFIC_SIGN_REGULATORY_R6_ONEWAY_RECTANGULAR_RIGHT` | Sign indicating one-way travel to the right when facing it                         |
| One-way, unspecified | `TRAFFIC_SIGN_REGULATORY_R6_ONEWAY`                                                                    | One-way sign without an annotated left/right direction                             |
| Pedestrian crossing  | `TRAFFIC_SIGN_WARNING_W11_PEDESTRAIN_CROSSING`, `TRAFFIC_SIGN_WARNING_W11_PEDESTRIAN_CROSSING`         | Sign indicating a pedestrian crossing; source and corrected spellings are accepted |
| Other                | Any other string, or missing value                                                                     | Sign outside the named categories, or with an unspecified class                    |

Any string beginning with `TRAFFIC_SIGN_REGULATORY_R6_ONEWAY` identifies a one-way
sign. A final `_LEFT` or `_RIGHT` supplies its direction; other suffixes leave the
direction unspecified. No printed-text or numeric speed-limit field is required.

</details>

### `3d_road_markings`

Use a polygon around each painted road marking. This area is its **footprint**;
the vertices specify its position, size, and orientation on the road.

| Required attribute                 | Type/example                                                                 | Definition                                                                       |
| ---------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `surface.vertices`                 | `[N,3]` polygon                                                              | Boundary of the marking's footprint; arrows use the four corners described below |
| `attributes`: `road_marking_class` | `{"name":"road_marking_class","text":"ROI_POLYGON_DIRECTION_PAINT_FORWARD"}` | Class below                                                                      |

For an arrow, supply the four corners of the area enclosing the painted symbol,
rather than tracing the arrowhead and shaft. A rectangle is the usual choice.
Define front, rear, left, and right relative to the direction of travel along
the lane. Store the corners in this order: **front-right, front-left, rear-left,
rear-right**.

Top view, with lane travel toward the top:

```text
                   Lane travel
                        ^
                        |
front-left   q1 +----------------+ q0   front-right
               |                |
               |   arrow area   |
               |                |
rear-left    q2 +----------------+ q3   rear-right
```

For example, a footprint 4 m long and 2 m wide, on the ground at `z=0`, with
lane travel along world +x, has these corners:

| Order | Corner      | World coordinates `[x,y,z]` |
| ----- | ----------- | --------------------------- |
| `q0`  | Front-right | `[14,-1,0]`                 |
| `q1`  | Front-left  | `[14,1,0]`                  |
| `q2`  | Rear-left   | `[10,1,0]`                  |
| `q3`  | Rear-right  | `[10,-1,0]`                 |

Store them in `surface.vertices` as:

```json
[[14,-1,0], [14,1,0], [10,1,0], [10,-1,0]]
```

The edge from `q0` to `q1` is the front; the opposite edge is the rear. The
direction of travel runs from the rear edge's midpoint toward the front edge's
midpoint. Keep this orientation for every arrow class: a left-turn arrow
uses the same corner order, with `ROI_POLYGON_DIRECTION_PAINT_LEFT` specifying
the turn. The four corners must enclose a nonzero area; do not repeat `q0` as a
fifth vertex. Class names are uppercase, without padding whitespace.

<details>
<summary>Road-marking classes and definitions</summary>

Classes describe painted road features; road-area masks are not painted markings.

| Class                                       | Definition                                                                    |
| ------------------------------------------- | ----------------------------------------------------------------------------- |
| `ROI_POLYGON_DIRECTION_PAINT_FORWARD`       | Painted arrow indicating straight-ahead travel                                |
| `ROI_POLYGON_DIRECTION_PAINT_LEFT`          | Painted arrow indicating a left turn                                          |
| `ROI_POLYGON_DIRECTION_PAINT_RIGHT`         | Painted arrow indicating a right turn                                         |
| `ROI_POLYGON_DIRECTION_PAINT_LEFT_FORWARD`  | Painted arrow indicating straight-ahead or left-turn travel                   |
| `ROI_POLYGON_DIRECTION_PAINT_RIGHT_FORWARD` | Painted arrow indicating straight-ahead or right-turn travel                  |
| `ROI_POLYGON_DIRECTION_PAINT_MERGE_LEFT`    | Painted arrow indicating a merge to the left                                  |
| `ROI_POLYGON_DIRECTION_PAINT_MERGE_RIGHT`   | Painted arrow indicating a merge to the right                                 |
| `ROI_POLYGON_ROAD_MARKING_STOP`             | Painted STOP lettering identifying a stopping location                        |
| `ROI_POLYGON_KEEP_CLEAR`                    | Painted area identifying space that traffic should keep clear                 |
| `ROI_POLYGON_TEXT_PAINT`                    | Painted words, letters, or numbers without a more specific road-marking class |
| Any other string                            | Other painted road marking                                                    |

</details>

### `3d_traffic_lights`

Each label describes a **traffic-light housing**.

| Required attribute         | Type/example                              | Definition                                                      |
| -------------------------- | ----------------------------------------- | --------------------------------------------------------------- |
| `cuboid3d.vertices`        | `[8,3]`                                   | Housing in the [specified order](#sign-and-signal-cuboid-order) |
| `attributes`: `feature_id` | `{"name":"feature_id","text":"signal-1"}` | Unique stable signal ID matching the state record               |

The [traffic light state](#traffic-light-states) specifies the displayed
indication, including arrows and pedestrian signals.

The `feature_id` attribute must contain exactly the `name` and `text` fields
shown above. A clip can contain a single light. Use `{"labels":[]}` when no
lights are present.

## Traffic light states

`3d_traffic_lights_lightnet_status/<clip_id>.tar` contains
`<clip_id>.aggregated_states.json`, with one state for each traffic light at each
pose frame. The same state is used for that light in every camera view.

This example describes one light, `signal-1`, over three pose frames:

```json
{
  "state_to_legacy_render_state": {
    "unknown": "unknown",
    "red_solid": "red",
    "green_right_arrow": "green"
  },
  "traffic_light_states": {
    "0": {
      "feature_id": "signal-1",
      "state": ["red_solid", "unknown", "green_right_arrow"]
    }
  }
}
```

The file has two required fields:

- `state_to_legacy_render_state` maps each state used in the file to `red`,
  `yellow`, `green`, or `unknown`. The renderer uses this mapping for solid and arrow colors.
- `traffic_light_states` contains one record for every light in `3d_traffic_lights`.
  The key `"0"` refers to the first item in its `labels` list, `"1"` to the second,
  and so on.

Each light record has two required fields:

| Field        | Type            | Definition                                                                                                            |
| ------------ | --------------- | --------------------------------------------------------------------------------------------------------------------- |
| `feature_id` | String          | Must match the `feature_id` of the geometry label at this index. In the example, `labels[0]` must have ID `signal-1`. |
| `state`      | List of strings | Exactly `T` entries for a clip with `T` pose frames. Entry `i` is the light's state at pose frame `i`.                |

Use `unknown` for any frame without a reliable observation, keeping its place in
the list. Use `stateless` when the light is observed with no active indication.
If the geometry contains no lights, set both top-level fields to `{}`.

<details>
<summary>Traffic light states and definitions</summary>

State names describe the observed signal indication. “Up” means straight ahead;
VRU denotes vulnerable road users, such as pedestrians or cyclists. In directional
state names, the prefix gives the indication's color; `red_yellow` denotes
simultaneous red and yellow indications.

| Input state                                                                            | Definition                                                                      |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `red_solid`                                                                            | Red signal indication without a directional arrow                               |
| `yellow_solid`                                                                         | Yellow signal indication without a directional arrow                            |
| `green_solid`                                                                          | Green signal indication without a directional arrow                             |
| `red_yellow_solid`                                                                     | Simultaneous red and yellow indications without a directional arrow             |
| `red_left_arrow`, `yellow_left_arrow`, `green_left_arrow`, `red_yellow_left_arrow`     | Signal indication controlling left-turn travel                                  |
| `red_right_arrow`, `yellow_right_arrow`, `green_right_arrow`, `red_yellow_right_arrow` | Signal indication controlling right-turn travel                                 |
| `red_up_arrow`, `yellow_up_arrow`, `green_up_arrow`, `red_yellow_up_arrow`             | Signal indication controlling straight-ahead travel                             |
| `red_vru`                                                                              | Red signal indication for pedestrians or cyclists                               |
| `green_vru`                                                                            | Green signal indication for pedestrians or cyclists                             |
| `yellow_vru`                                                                           | Yellow signal indication for pedestrians or cyclists                            |
| `white_vru`                                                                            | White pedestrian go signal                                                      |
| `stateless`                                                                            | Observed signal with no active indication                                       |
| `unknown`                                                                              | No reliable state observation                                                   |
| Other source state                                                                     | Additional indication, such as a combined-direction, diagonal, or U-turn signal |

</details>
