# LiDAR range maps

`cosmos_auto.lidar` renders two range maps on the lattice of the ego vehicle's
spinning LiDAR, for RDS-HQ clips:

| Output                                  | Content                                                                    | Used as                                           |
| --------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------- |
| `hdmap_rangemap/<clip_id>.tar`          | HD-map layers + tracked object cuboids, ray-cast onto the LiDAR lattice    | **control signal** for the Cosmos LiDAR generator |
| `lidar_rangemap/<clip_id>.tar`          | the recorded LiDAR returns (range + intensity), motion compensation undone | ground truth / target                             |
| `lidar_rangemap_metadata/<clip_id>.npz` | per-sweep start/end poses and timestamps, provenance                       | bookkeeping                                       |
| `render/<clip_id>.mp4`                  | colour strips: measured range / HD-map range / intensity                   | eyeballing                                        |

The code is a port of the Cosmos-Drive-Dreams toolkit
(`convert_lidar_pointcloud_to_rangemap.py`, `utils/lidar_*.py`,
`utils/minimap_utils_for_lidar.py`, `utils/fill_in_road.py`) with two deliberate
changes described below. Output tars use the toolkit's sparse layout
(`<frame>.lidar_row.npz` uint8, `<frame>.lidar_col.npz` uint16,
`<frame>.lidar_range.npz` float16, optional `<frame>.lidar_intensity.npz`). Each
tar also contains `<clip_id>.lidar_metadata.json` with `format_version: 2`,
`n_rows` and `n_columns`, derived from the rendered arrays. Framework's updated
three-channel reader and the Auto cookbook converter support 128 x 1800 and
128 x 3600: 1800 columns are retained; 3600 columns are nearest-return pooled
when the model expects 1800 semantic columns. Model padding is unchanged.

Use updated readers for these archives. Files without metadata are interpreted
as legacy 128 x 3600; old 1800-column files without metadata must be regenerated.
Metadata does not change sensor calibration, units or azimuth conventions, and
does not make 1800-column inputs compatible with the legacy V0 tokenizer.

- [Pose sources](#pose-sources)
- [Sweep poses from RDS-HQ](#sweep-poses-from-rds-hq)
- [Lattice and sensor model](#lattice-and-sensor-model)
- [Pipeline](#pipeline)
- [Command line](#command-line)
- [Inputs](#inputs)
- [Validation](#validation)
- [Differences from the toolkit](#differences-from-the-toolkit)

## Quickstart: a scene with no measured LiDAR

Run from `wsm` after the [standalone installation](../README.md#install):

```bash
python examples/lidar_scene.py --output-root outputs/scene
cosmos-auto-lidar-rangemap --input-root outputs/scene --output-root outputs/lidar \
  --clip-id synthetic_drive --max-frames 67 --export-control --device cpu --no-vis
```

The example creates an explicit synthetic camera/LiDAR rig, all map layers,
tracked cuboids and 202 synchronized poses at 30 Hz. No recorded returns or
model weights are required. The last pose brackets the last sweep. The outputs
include `hdmap_rangemap/synthetic_drive.tar` and
`controls/synthetic_drive/lidar_control.safetensors`. The latter contains only
FP32 `frames[3,67,128,1800]`: metres, placeholder unit intensity and validity.
Empty rays have all three channels zero. Adjacent native azimuth columns are
pooled by nearest nonzero return. Runtime code owns padding, filtering and
normalization; do not preprocess this physical file into model space.

To visualize range-view strips, omit `--no-vis` after installing `'.[lidar-vis]'`.
For a point cloud, load a native sweep with `load_range_maps_from_tar`, then use
`range_map_to_points(torch.from_numpy(ranges), torch.from_numpy(model.ray_directions()))`;
points are in the FLU sensor frame. These are geometry samples, not measured returns.

Render camera WSM from **the same root and clip** (requires GPU/EGL):

```bash
cosmos-auto-wm input_root=outputs/scene output_root=outputs/camera version=v4 \
  camera_spec.camera_list='[camera_front_wide_120fov]' \
  input_fps_spec.input_pose_fps=30 output_frame_spec.target_render_fps=30 \
  output_frame_spec.target_chunk_frame=201 output_frame_spec.overlap_frame=0 \
  output_frame_spec.max_chunk=1 resolution_spec=resize_1280x720
```

The example is a deliberately simple one-camera synthetic scene, not a substitute
for the released multi-camera inference fixture. Change `--pole-x` or `--speed`
and write a fresh scene/output directory to see geometry or ego-motion change
both controls. Camera images alone cannot recover this 3D contract.

### Use with joint inference

For the minimal scene above, use the **native Framework** path below. It
supports joint generation without initial RGB; the joint I2V example in the
[Auto cookbook](../../auto.md) requires RGB for every requested camera (seven in the bundled sample). Do not supply
unrelated RGB or duplicate one view across the rig to satisfy that walkthrough.

From `wsm/`, set the generated scene and its synthetic caption sidecar:

```bash
export AUTO_RDS_HQ_ROOT="$PWD/outputs/scene"
export AUTO_RDS_HQ_CLIP=synthetic_drive
export AUTO_RDS_HQ_CAPTIONS="$AUTO_RDS_HQ_ROOT/captions"
export AUTO_WSM_OUTPUT_ROOT="$PWD/outputs/camera"
export WSM_PYTHON=/absolute/path/to/renderer/venv/bin/python
```

Open the [WSM notebook](../../generate_wsm.ipynb).
Before running its configuration cell, select:

```python
CAMERAS = ["camera_front_wide_120fov"]
NUM_FRAMES, FPS = 201, 30
GENERATE_LIDAR_CONTROL = True
RESOLUTION_SPEC = "resize_1280x720"
```

Run inspection, rendering, video checks, caption export and LiDAR export. If the
camera command above already produced these exact controls, skip only the camera
render cell. The exporter writes `request-joint.json` using this scene's camera
control, captions and 67 LiDAR sweeps. The generated caption defines a desired
daylight/dry-road appearance for the synthetic scene; it is not a recorded target.

In the [native inference notebook](../../run_auto_with_cosmos_framework.ipynb),
set `CUSTOM_REQUEST` to that file and **`MODE = "joint"`**. Joint mode retains
the LiDAR input and selects text-to-video when no view has RGB. Selecting
`MODE = "text2video"` is camera-only and removes LiDAR. Obtain the matching
checkpoint before running model download/generation. Check the prepared request
still contains `lidar`, `model_mode="text2video"`, no `vision_path` fields and
`condition_video_as_image=false`.

This one-camera synthetic scene checks input preparation and execution plumbing;
it is not a quality benchmark or a claim that its custom rig matches the model's
reference rig. Model output still needs validation with the actual checkpoint.

Prepare captions and the camera request using the existing
[render → caption → inference notebook](../../generate_wsm.ipynb).
Enable `GENERATE_LIDAR_CONTROL` to render the matching control and export
`request-joint.json` automatically, with sweep count and start-frame checks.
For a manually prepared request, add `lidar: {"control_path": "<matching control path>"}` to the native request. Set `num_frames=201`, `fps=30`.
Paths resolve relative to the request file. Native Framework emits LiDAR output
automatically; `lidar.return_output` is a serving-API field and is not accepted
in a native request. Use the exported safetensors or the
sparse tar directly with Framework; vLLM-Omni requires the physical safetensors.
Any `vision_path` must contain first-frame RGB from this same scene/camera;
omit it for supported text-to-video generation rather than borrow another scene.
Keep the caption, camera control and LiDAR interval starting at pose frame 0.

201 camera frames at 30 FPS require 67 LiDAR sweeps at 10 Hz, starting at
pose IDs `0,3,...,198`. Their end poses include frame 201, so supply at least
202 pose frames. A 201-pose input yields only 66 complete sweeps; never claim
that is a complete control for the 201-frame request. Source frame IDs and
start/end timestamps are saved in `lidar_rangemap_metadata/<clip>.npz`.

In a full inference environment, check the actual runtime loader before running
the joint inference command from the notebook:

```python
from pathlib import Path
from cosmos_framework.inference.lidar_transfer import load_lidar_transfer_control
from cosmos_framework.model.generator.tokenizers.lidar.range_projection import V1_TRANSFER_RANGE_PROJECTION
frames, _ = load_lidar_transfer_control(
    Path("outputs/lidar/controls/synthetic_drive/lidar_control.safetensors"),
    max_frames=67, range_projection=V1_TRANSFER_RANGE_PROJECTION,
)
assert tuple(frames.shape) == (3, 67, 128, 1800)
```

`lidar_control.safetensors` is an **input** derived from geometry. Generated
`lidar.safetensors` is an **output** with `range_m`, `intensity`, `frame_ids`;
it is not an interchangeable control file. Measured `lidar_raw` is optional
and only used with the explicit flags below. The provenance of an unrelated
supplied fixture cannot be inferred from tensor shape.

The renderer and loader checks do not establish model inference quality.
End-to-end joint inference additionally needs the released Transfer-Auto joint
checkpoint, its supported rig/captions and a compatible inference GPU environment.

## Pose sources

The HD-map range map is the control signal for LiDAR generation: the generator
is conditioned on the map and produces the LiDAR sweep. The toolkit builds that
map from the **recorded** sweep poses (every `lidar_raw` frame carries
`lidar_to_world` and `starting_timestamp`), which ties the control signal to the
sensor being generated.

`cosmos_auto.lidar` therefore defaults to `pose_source="rds_hq_pose"`: the sweep
poses are derived from the RDS-HQ ego trajectory and the rig calibration alone.
This lets the control signal be produced for clips that have no LiDAR, for novel
or edited trajectories, and at inference time where only the map and the ego
path are available, and it keeps the control independent of the sensor whose
output is being generated.

`pose_source="lidar_raw"` is retained for comparison with the original toolkit.
It is also always what the *measured* `lidar_rangemap` uses, since undoing the
motion compensation of recorded returns needs the recorded sweep poses.

## Sweep poses from RDS-HQ

```text
lidar_to_world(sweep i, start) = vehicle_to_world[f]        @ lidar_to_vehicle
lidar_to_world(sweep i, end)   = vehicle_to_world[f + s]    @ lidar_to_vehicle
f = i * s,  s = input_pose_fps / lidar_fps  (30 / 10 = 3)
```

| Quantity              | Source (first available)                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `vehicle_to_world[f]` | `vehicle_pose/<clip>.tar` (`<frame>.vehicle_pose.npy`); otherwise `pose/<clip>.tar` of `camera_name` combined with that camera's rig extrinsic, `camera_to_world[f] @ inv(camera_to_vehicle)`, which is exact in any world frame; only when `rig/<clip>.json` has no entry for the camera does it fall back to the RDS-HQ convention `vehicle_to_world[0] = I` (`camera_to_world[f] @ inv(camera_to_world[0])`), and it says so on stdout. On RDS-HQ exports all three agree to float precision. |
| `lidar_to_vehicle`    | `rig/<clip>.json` (`nominalSensor2Rig_FLU` rotation right-multiplied by `correction_sensor_R_FLU`, translation plus `correction_rig_T`); otherwise `sensor_collection/<clip>.txt` `sensor_to_vehicle_transform`; otherwise, only with `allow_default_extrinsic`, a nominal mount for the rig's top GT LiDAR.                                                                                                                                                                                     |
| sweep timestamps      | `ftheta_<camera>_timestmap/<clip>.timestamps` when present (the exporter's directory name), else `frame / input_pose_fps`. Only the start-to-end difference matters.                                                                                                                                                                                                                                                                                                                             |

Sweep `i` is keyed by frame `f` (`000000`, `000003`, ...), exactly like the
recorded `lidar_raw` sweeps, so HD-map and measured maps line up frame by frame.
The final frame without a successor is dropped, as in the toolkit.

Pose-source comparison on a 20-sweep RDS-HQ clip: sweep poses derived from the
RDS-HQ trajectory differ from the recorded `lidar_to_world` by 0.04-0.11 m along
the direction of travel and < 0.05 deg in rotation. The recorded pose
corresponds to a time ~10 ms earlier than the pose-frame timestamp. The
resulting HD-map maps agree to a median 2-3 cm on jointly occupied cells and
align with the measured LiDAR equally well from either pose source (median
|Δrange| 0.14-0.34 m under 40 m on most frames).

## Lattice and sensor model

`SpinningLidarModel` describes the range-map lattice: per-ring elevation and
azimuth offset, `n_columns` firing azimuths per revolution, spin rate. Loaded from

1. `sensor_collection/<clip>.txt` (`rotating_lidar_parameters.line_parameters`, 128 rings; proto3 text omits zero-valued fields),
2. `lidar_sensor_config/<clip>.json` (toolkit format: `row_elevations_rad`, `row_azimuth_offsets_rad`), or
3. the bundled Pandar128 default (`assets/row_offset_spinning_lidar_model_parameters.json`).

Conventions: FLU sensor frame, ring 0 is the top ring, the head spins clockwise,
column 0 looks backwards and the forward direction is the centre column. Relative
firing time of a direction is `column / (n_columns - 1)` where the column is the
nearest nominal azimuth after removing the ring's azimuth offset (the NCORE
`angles_to_columns_map` lookup, computed analytically).

## Pipeline

Per sweep, for the HD-map range map:

1. **World points.** Map layers (`3d_lanelines`, `3d_road_boundaries`, `3d_crosswalks`,
   `3d_road_markings`, `3d_wait_lines`, `3d_poles`, `3d_traffic_signs`, `3d_traffic_lights`)
   are sampled into dense points: polylines and cuboid edges as thin point
   cylinders (`radius`, 5 cm), polygons as a planar grid (`density`, 2 cm).
   Object cuboids from `all_object_info` are sampled per frame the same way.
   `--fill-in-road` adds an estimated drivable surface between the road
   boundaries (`road_surface.py`).
2. **Prefilter** (opt-in, `--prefilter`): project with the sweep-start pose, keep one
   point per occupied cell, then unproject each surviving cell along the ray
   through its lattice cell centre (ring elevation, nominal centre azimuth
   `pi - (k + 0.5) * 2pi / n_columns`; per-ring azimuth offsets are not applied,
   matching `points_to_range_map`) at the kept range, so every surviving cell
   re-projects to its own cell. Thins tens of millions of points to at most
   `n_rows * n_columns` (toolkit step 3.1), making the sweep loop several times
   faster. Thinning at a single pose is lossy once the ego moves during the
   sweep: on a clip driven at 11 m/s it drops 10-24% of the cells an exact
   render produces (the original toolkit has the same behaviour), so it is off
   by default and the control signal is rendered from all points.
3. **Rolling shutter.** `shutter.world_points_to_shutter_sensor_frame` finds each
   point's firing time by fixed-point iteration: start at the sweep-start pose,
   read the column, interpolate the forward sensor pose (`lidar_to_world`)
   between the sweep-start and sweep-end poses at that firing time (quaternion
   slerp for rotation, linear interpolation for translation), re-project; stop at
   a mean change of `1e-4` sweeps or after 10 iterations. The result is the
   points in the moving sensor frame, which is what a real sweep records.
4. **Projection.** `rangemap.points_to_range_map`: nearest return per cell via
   `torch.scatter_reduce(amin)`, ranges below 105 m.

For the measured map, `lidar_raw.uncompensated_sweep_points` undoes the published
motion compensation (returns are compensated to the sweep-start frame) with the
recorded start/end poses and each return's `column`, then projects.

## Command line

```bash
# HD-map control signal from RDS-HQ poses, one clip
cosmos-auto-lidar-rangemap --input-root /data/rds_hq --output-root /data/out --clip-id <clip_id>

# Also render the measured LiDAR range map and a comparison strip video
cosmos-auto-lidar-rangemap --input-root ... --output-root ... --clip-id <clip_id> --render-lidar

# Toolkit comparison: recorded sweep poses for the HD-map map
cosmos-auto-lidar-rangemap ... --pose-source lidar_raw --render-lidar

# Drivable-surface fill, every clip under <input_root>/pose
cosmos-auto-lidar-rangemap --input-root ... --output-root ... --fill-in-road
```

Useful flags: `--max-frames N` (debug), `--no-vis`, `--device cpu`,
`--sensor-model default|auto|<json>`, `--lidar-sensor-name`, `--density`,
`--radius`, `--prefilter`, `--allow-default-extrinsic`. Clips are processed
sequentially; wrap the CLI in your own job scheduler for fleets.

Wait lines flagged `is_implicit` in RDS-HQ (virtual stop lines with no paint) are not
rendered, like the world-scenario renderer. The Cosmos-Drive-Dreams toolkit draws them;
on an intersection-heavy clip they are about 2% of its HD-map cells, so expect that much
toolkit-only content at intersections when comparing.

Install the standalone WSM project (`python -m pip install -e .` from `wsm`).
Add `'.[lidar-vis]'` for range-view videos.
The default exact render needs a CUDA build of torch (not the `cpu` extra) to be
practical: a full 10 s clip (216 sweeps) takes about 40 s on an RTX 6000 Ada,
or about 64 s with `--render-lidar`, but about 13 s per sweep on a 32-thread
CPU. On CPU, `--prefilter` renders 20 sweeps in about 10 s.

## Inputs

Required under `input_root` (RDS-HQ layout, `<component>/<clip_id>.tar`):

| Component                                           | Needed for                                                              |
| --------------------------------------------------- | ----------------------------------------------------------------------- |
| `pose/` (or `vehicle_pose/`)                        | sweep poses (`rds_hq_pose`)                                             |
| `rig/<clip>.json` or `sensor_collection/<clip>.txt` | LiDAR extrinsic; `sensor_collection` also supplies the ring calibration |
| `3d_*` map layers, `all_object_info/`               | HD-map range map                                                        |
| `lidar_raw/`                                        | `--render-lidar`, `--pose-source lidar_raw`                             |
| `ftheta_<camera>_timestmap/<clip>.timestamps`       | optional, real sweep durations                                          |

## Validation

The historical toolkit comparisons below describe the source renderer. They are
not a claim of model inference validation for this standalone integration.

`tests/lidar/control_test.py` checks the new physical control export contract:
shape, dtype, nearest occupied return, empty rays and rejection of invalid ranges.
The notebook regressions cover joint conditioning and native request fields;
the F-theta scalar regression covers the camera-rendering exception.

Against the unmodified toolkit + NCORE output on a 20 s reference clip
(`--pose-source lidar_raw`, 40 sweeps): measured LiDAR maps are identical
(IoU 0.9999, every joint cell within one float16 step, intensity maximum absolute difference 0.012);
HD-map maps have IoU 0.976-0.993 with a median absolute range difference of 0 and p99 ≤ one
float16 step. The residual cells come from the analytic column lookup versus
NCORE's grid-sampled nearest-neighbour map.

The pose-source comparison in [Sweep poses from RDS-HQ](#sweep-poses-from-rds-hq)
quantifies how far the RDS-HQ-derived control signal is from the one built with
recorded poses.

## Differences from the toolkit

- No `ncore`, `torch_scatter`, `mediapy`, `scikit-image` or `scikit-spatial`
  dependencies; the sensor model, shutter solver, scatter-min, connected
  components (`scipy.ndimage.label`) and plane fit (SVD) are implemented here.
- `pose_source="rds_hq_pose"` (new, default) builds sweep poses from RDS-HQ ego
  poses and the rig extrinsic; `"lidar_raw"` keeps the toolkit behaviour.
- The static map is streamed one frame at a time (the toolkit materialised one
  copy per frame, ~80 GB on a 20 s clip).
- Missing `all_object_info` frames yield an empty box layer instead of a crash.
- The ring/column lattice comes from `sensor_collection` directly; the toolkit
  needed a separately generated `lidar_sensor_config` JSON.
