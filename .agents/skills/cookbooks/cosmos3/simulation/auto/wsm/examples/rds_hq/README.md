# RDS-HQ example clip

A real WSM4 input clip with seven F-theta cameras, all five object categories,
all eight map components, and LightNet traffic-light states. RGB and LiDAR are
not included.

Clip ID: `0d7766e8-25ed-11ef-8715-00044baf7cbc_1717889548967272_1717889568967272`.

Poses and signal states cover 649 frames at 30 Hz, ending at the final 10 Hz
object observation. Archives use `<component>/<clip_id>.tar` and are stored with Git LFS.
Fetch their contents before use.

See the [RDS-HQ format reference](../../docs/rds-hq.md) for field definitions.

## Render

Install the package and prepare the GPU/EGL environment as described in the
[WSM4 guide](../../docs/wsm4.md#render-one-clip). From the standalone `wsm` directory:

```bash
cosmos-auto-wm \
    input_root=examples/rds_hq \
    output_root=example_data/wsm4-output \
    version=v4 s3_spec=no_upload \
    camera_spec.camera_list='[camera_front_wide_120fov]' \
    input_fps_spec.input_pose_fps=30 \
    resolution_spec=resize_1280x720 \
    output_frame_spec.target_render_fps=30 \
    output_frame_spec.target_chunk_frame=297 \
    output_frame_spec.overlap_frame=0 output_frame_spec.max_chunk=1
```

This command renders one front-wide MP4 with 297 frames at 30 FPS and 1280×720.
Replace the explicit camera-list override with `camera_spec=internal_7cam` to
include all seven cameras. Keep the timing overrides: the general-purpose
renderer defaults are 10 FPS, not the Auto inference example's 30 FPS.

This is a renderer example, separate from the published seven-camera daytime
inference sample. To generate RGB from these controls, prepare captions for the
rendered views and use the [Auto handoff](../../docs/scenario-to-rds-hq.md#continue-to-auto-inference).
