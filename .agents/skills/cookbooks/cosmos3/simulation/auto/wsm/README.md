# WSM control-video renderer for Auto

Render driving scenarios in [RDS-HQ format](docs/rds-hq.md) into the WSM4 control
videos used by Cosmos3-Nano-Transfer-Auto and Cosmos3-Nano-Sim-Auto. Use the
WSM 4.0.1 style selected by `version=v4`. This directory is an independent Python
project: install it in its own environment and use its outputs with either
Cosmos Framework or vLLM-Omni.

## Choose your starting point

- **Try model inference:** use the [published daytime sample](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto/tree/fc00d08969cb1d3c69ac377f426951aeab05e113/assets/sample_data).
  Its WSM controls are already rendered; no renderer setup is needed.
- **Render your own scenario:** supply scene geometry, object tracks, calibration
  and poses. Follow the [conversion skill](skills/scenario-to-rds-hq/SKILL.md),
  then render, caption the resulting views and run Auto inference.
- **Check the renderer:** use the separate [raw RDS-HQ example](examples/rds_hq/README.md).
  It contains scene annotations and calibration, not RGB/LiDAR recordings or captions.

## Get the renderer

From your Cosmos Framework checkout, install Git LFS and fetch the renderer's
resources:

```bash
git lfs install
git lfs pull --include='.agents/skills/cookbooks/cosmos3/simulation/auto/wsm/**'
cd .agents/skills/cookbooks/cosmos3/simulation/auto/wsm
```

Preserve the directory contents when copying it
elsewhere; the bundled examples, configuration, palettes, and skill use relative
paths. A wheel contains the renderer and its runtime resources; the source
distribution also contains the documentation, scene example, skill, and tests.

## Install

Use Linux, Python 3.11 or newer, an NVIDIA GPU and driver, OpenGL/EGL
(`libGL.so.1`, `libEGL.so.1`), and FFmpeg with H.264 support. CPU tests and
configuration inspection do not require a GPU. In containers expose the GPU
and set `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`.

On Ubuntu/Debian, install the video and graphics runtime plus the development
headers needed to build ModernGL/glcontext on platforms without prebuilt wheels
(including Linux ARM):

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libgl1 libegl1 libx11-dev libgl-dev libegl-dev build-essential python3-dev
```

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
cosmos-auto-wm --help
```

Install a CUDA-enabled PyTorch build compatible with your GPU when your platform's
default PyTorch wheel is CPU-only. The renderer's dependencies are declared in
[pyproject.toml](pyproject.toml); the model inference environment is separate.
The `video` extra adds recorded-RGB resampling via Decord and the `pointcloud`
extra adds Open3D export. Neither is required for WSM4 control-video rendering.

### Check NVIDIA EGL in containers

The container runtime must expose both the NVIDIA EGL libraries and their GLVND
vendor configuration. `nvidia-smi` alone does not verify graphics support. Check
the renderer selected by EGL:

```bash
python -c 'import moderngl; ctx = moderngl.create_context(standalone=True, backend="egl", require=330); print(ctx.info["GL_VENDOR"], ctx.info["GL_RENDERER"]); ctx.release()'
```

The output should identify NVIDIA and your GPU. If `eglInitialize` fails and your
container exposes `libEGL_nvidia.so.0` but only has a Mesa vendor configuration,
configure GLVND explicitly before rendering:

```bash
cat > nvidia-egl.json <<'JSON'
{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}
JSON
export __EGL_VENDOR_LIBRARY_FILENAMES="$PWD/nvidia-egl.json"
```

Repeat the check. This configuration requires the NVIDIA driver libraries to be
present; it does not install or replace the host driver.

## Render the included scene

Run from this directory:

```bash
cosmos-auto-wm \
  input_root=examples/rds_hq \
  output_root=outputs/wsm4 \
  version=v4 s3_spec=no_upload \
  camera_spec.camera_list='[camera_front_wide_120fov]' \
  input_fps_spec.input_pose_fps=30 \
  resolution_spec=resize_1280x720 \
  output_frame_spec.target_render_fps=30 \
  output_frame_spec.target_chunk_frame=297 \
  output_frame_spec.overlap_frame=0 output_frame_spec.max_chunk=1
```

This renders one front-wide MP4 at **30 FPS, 297 frames (9.9 seconds), 1280×720**
under `outputs/wsm4/world_scenario/`. Expand `camera_spec.camera_list` with the
calibrated cameras in your scene. For the bundled scene, `camera_spec=internal_7cam`
selects all seven; remove the explicit camera-list override when using that profile.
The profile selects names; the actual calibration comes from the RDS-HQ data.

For a short rendering check, add:

```text
output_frame_spec.target_chunk_frame=5 output_frame_spec.max_chunk=1
```

Keep the explicit timing options for Auto inputs. The renderer's general-purpose
defaults otherwise select 10 FPS and overlapping 100-frame chunks. Its pose
sampler uses an integer stride, not temporal upsampling: the pose rate must be
at least the output rate and divisible by it. A 10 Hz source needs a correctly
resampled pose/annotation timeline before this 30 FPS example; changing a rate
label does not create intermediate observations. See [coordinates and time](docs/rds-hq.md#coordinates-and-time).

Use `--cfg job --resolve` to inspect a configuration before rendering. Input and
output directories are supplied by the caller. Local rendering needs no cloud
credentials. Optional upload requires an explicitly supplied destination and
credentials via the S3 configuration; no preconfigured cloud accounts are bundled.

## Use your own scenarios

Use the offline [Camera Rig Explorer](docs/camera-rig/guide.md) to inspect
calibrated camera positions and fields of view in 2D and 3D before conversion.

1. Convert scenario geometry, tracks, poses, camera calibration, and signal states
   to [RDS-HQ](docs/rds-hq.md). The [scenario conversion guide](docs/scenario-to-rds-hq.md)
   and [skill](skills/scenario-to-rds-hq/SKILL.md) describe the conversion workflow.
2. Choose reference-rig retargeting for generation, or keep your own rig for
   paired RGB/WSM post-training. Retargeting changes viewpoints; it does not
   transform recorded RGB or fill missing scene annotations.
3. Render with `version=v4`, selecting matching cameras and timing. See the
   [WSM4 guide](docs/wsm4.md) for the visual conventions and configuration options.
4. In a full Framework checkout, follow the
   [render → caption → inference notebook](../generate_wsm.ipynb).
   It renders before requiring captions, links the per-view captioning guide,
   and exports a native request for either Auto model. It accepts F-theta scenes
   with supported Auto camera keys at 30 Hz. To reuse this quickstart's output,
   select the same cameras and `RESOLUTION_SPEC="resize_1280x720"`, point to
   the existing output root and skip rendering. For unresized renders, select
   `RESOLUTION_SPEC="source"`. Pinhole cameras or custom names use the
   [custom-rig handoff](docs/scenario-to-rds-hq.md#custom-rigs-and-already-rendered-controls).

The standalone package produces camera controls and can also [render numeric LiDAR controls](docs/lidar-rangemap.md) from the same scene geometry, ego trajectory and LiDAR calibration. Captions, model weights and the
Framework inference environment are separate. Joint RGB/LiDAR generation also
requires aligned numerical HD-map range-map controls; use `cosmos-auto-lidar-rangemap`
for that separate rendering step. Do not borrow RGB or LiDAR from an unrelated example.

## Verify the standalone package

```bash
python -m pip install '.[dev]'
python tools/check_portability.py
python -m pytest -q --confcutdir=.
python -m build
python tools/check_portability.py dist/*.whl dist/*.tar.gz
```

Tests cover scene conversion, calibration, palettes, geometry, and frame selection.
GPU/EGL rendering is checked separately using the included scene. The audit also
checks resources inside archives and built distributions. Source version and
file hashes are recorded in [PROVENANCE.json](PROVENANCE.json); the LiDAR
source snapshot is recorded in [LIDAR_PROVENANCE.json](LIDAR_PROVENANCE.json).

## License

This component is licensed under the OpenMDW-1.1 license, the same license as
the repository; see [LICENSE](LICENSE).
