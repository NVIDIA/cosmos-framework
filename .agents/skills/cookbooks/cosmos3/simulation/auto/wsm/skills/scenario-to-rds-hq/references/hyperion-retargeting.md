# Hyperion retargeting

When retargeting is enabled, keep the native dataset and its RGB comparisons;
write the Hyperion dataset to a separate root.

## Saved rig

[hyperion8.json](../assets/hyperion8.json) contains eleven cameras:
front wide and telephoto, left and right cross views, left and right rear views,
rear telephoto, and front/left/rear/right fisheyes. Names match the renderer's
`camera_spec=internal_11cam` configuration.
The seven non-fisheye cameras use 1924×1084; the four fisheyes use 1720×1080.

Each camera has two fields:

| Field               | Meaning                                                                                                                            |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `ftheta_intrinsics` | Ready-to-write RDS-HQ array `[cx,cy,width,height,p0,p1,p2,p3,p4,p5,1,c,d,e]`; the polynomial maps pixel radius to angle in radians |
| `camera_to_vehicle` | 4×4 transform from OpenCV camera axes (right, down, forward) to vehicle FLU axes (forward, left, up), in metres                    |

The vehicle origin is the rear-axle centre projected onto the ground. Use the
saved calibration directly; no resizing or lens fitting is needed. The renderer
accepts both the example's 11-value intrinsics and this asset's 14-value form.
The extra `c,d,e` values account for differences in horizontal and vertical scaling.

## Vehicle poses

The adapter must save `vehicle_to_world.npy` with shape `[T,4,4]`, one pose for
each existing pose-frame index. Use the same fixed world frame, metres, and
timeline as the converted annotations. Normalize the source vehicle origin to
the rear-axle ground projection before composing the camera poses. For example,
if the source ego pose is measured at an IMU, apply the known rear-axle-to-IMU
transform as well. Do not assume those origins coincide.

When only camera poses are available, recover vehicle motion using verified
source extrinsics:

```text
vehicle_to_world[t] = source_camera_to_world[t] @ inverse(source_camera_to_vehicle)
hyperion_camera_to_world[t] = vehicle_to_world[t] @ hyperion_camera_to_vehicle
```

Both camera-to-vehicle transforms must use OpenCV camera axes and the same
vehicle convention. If source calibration or the vehicle-origin offset is
unknown, report the missing information and ask for it. A camera trajectory
alone does not establish a vehicle trajectory.

## Run and check

The helper uses the same NumPy and Shapely dependencies as `rds_hq_io.py`:

```bash
python <skill-directory>/scripts/retarget_hyperion.py \
    <native-rds-hq-root> <hyperion-rds-hq-root> <clip-id> \
    --vehicle-poses <vehicle_to_world.npy> \
    --source-camera-type pinhole
```

Set `--source-camera-type ftheta` when the native conversion uses F-theta.
The helper validates both datasets, copies the ten scene annotation archives
byte for byte, and writes new `pose` and `ftheta_intrinsic` archives. It refuses
to overwrite a clip. Source RGB and additional components remain in the native
output and are listed in the report.

Inspect `<hyperion-rds-hq-root>/retargeting/<clip-id>.json` for camera names,
frame counts, rig checksum, and annotation checksums. Load the archives through
the reference renderer and confirm the camera mounts move with the vehicle.
At an identity vehicle pose, each camera pose must equal its saved extrinsic.
A vehicle turn must rotate both the camera orientation and mounting offset.

For rendering, use `version=v4 camera_type=ftheta camera_spec=internal_11cam
resolution_spec=source`. Retargeting does not change FPS: set
`input_fps_spec.input_pose_fps` to the converted pose rate and choose the output
FPS and duration to fit the clip. No further calibration resizing is needed.
Follow the [visual review](visual-review.md) separately for native and Hyperion
views. Retargeting cannot create annotations outside the source's coverage or
turn source RGB into recordings from the new cameras.
