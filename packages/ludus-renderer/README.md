# Ludus Renderer

Ludus is a GPU-native F-theta renderer for autonomous-driving scenes. It renders
HD-map and obstacle primitives from clipgt / AV2 scenes directly on the GPU, with
either an OpenGL mesh-shader backend or a GL-free CUDA software rasterizer.

## Features

- **F-theta Camera Model**: Native support for fisheye lens distortion using polynomial projection
- **Mesh Shader Rendering**: High-performance GPU rendering using NVIDIA mesh shaders
- **CUDA Software Rasterizer**: GL-free rendering backend — no OpenGL/EGL/display dependencies required
- **Timestamped Rendering**: Efficient temporal queries for simulation playback
- **Adaptive Tessellation**: Automatic subdivision based on distortion error
- **MSAA**: 4x antialiasing via hardware MSAA (GL) or 2x supersampling (CUDA)
- **Mirror Augmentation**: Extend scenes by tiling reflected copies for longer driving sequences
- **GPU Spatial Culling**: Per-element AABB/sphere culling for large scenes

## Primitives

- **Polylines**: Thick line strips with configurable width and round caps
- **Polygons**: Filled polygons with pre-triangulation
- **Cubes**: Oriented bounding boxes with 9-DOF transform

## Requirements

**GL backend:**

- NVIDIA GPU with mesh shader support (Turing or later)
- OpenGL 4.6 with EGL

**CUDA backend:**

- NVIDIA GPU (Turing or later)
- No OpenGL/EGL/display dependencies

**Common:**

- CUDA 11+
- Python 3.10+
- A host C++ compiler for JIT extensions

## Installation

From the `cosmos-framework` repository root:

```bash
uv pip install -e packages/ludus-renderer
```

Install the matching runtime libraries for an existing CUDA 11 or CUDA 12
toolkit:

```bash
uv pip install -e 'packages/ludus-renderer[cuda11]'  # or cuda12
```

For a self-contained, pip-provided CUDA 13.0 build toolchain on Linux:

```bash
uv pip install -e 'packages/ludus-renderer[cuda13]'
```

The JIT extension loader first honors a compatible `CUDA_HOME` toolkit, which
supports CUDA 11+ environments and CUDA-enabled containers. The `cuda11` and
`cuda12` extras provide matching nvJPEG/nvCOMP runtime libraries; NVIDIA's
wheels for those releases do not include a complete `nvcc` compiler. The
`cuda13` extra provides a pip-only fallback using NVIDIA's CUDA 13.0 compiler,
nvJPEG, and nvCOMP wheels.

## Usage

```python
# GL backend
from ludus_renderer.torch import LudusTimestampedContext
ctx = LudusTimestampedContext(device="cuda")

# CUDA backend (no GL required)
from ludus_renderer.torch import LudusCudaTimestampedContext
ctx = LudusCudaTimestampedContext(device="cuda")
```

### Mirror Augmentation

Extend a scene by mirror-stitching it N times at load time. Two canonical tiles (original + single reflection) are placed alternately via rigid body transforms, producing an `[original]-[mirror]-[original]-[mirror]-...` pattern without rotational drift on curved roads.

```python
from ludus_renderer import load_clipgt_scene, mirror_augment_scene

scene = load_clipgt_scene("example_data/clipgts/clipgt-0300edb0-...", device=device)
extended = mirror_augment_scene(scene, n_mirrors=10, lookahead_m=50.0)
```

- `n_mirrors`: number of augmentation iterations (total segments = n_mirrors + 1)
- `lookahead_m`: distance (metres) beyond the ego endpoint to place the first mirror plane

GPU-side spatial culling ensures that rendering cost stays constant regardless of the augmented scene size. Culling is enabled by default (1.5x `depth_max`) and can be adjusted via:

```python
ctx.set_cull_radius(scale=1.5)  # 0 disables culling
```

## Tests

```bash
cd packages/ludus-renderer
uv run --group dev pytest tests
```

The tests need an NVIDIA GPU; they skip when CUDA or nvJPEG is unavailable.

## License

Ludus is released under [OpenMDW-1.1](../../LICENSE). The rasterizer core and
GL/CUDA utilities under `ludus_renderer/_cpp/cudaraster/`,
`ludus_renderer/_cpp/common/` and `ludus_renderer/_cpp/bindings/{torch_common.inl,torch_types.h}`
are derived from [nvdiffrast](https://github.com/NVlabs/nvdiffrast) and remain
under its [NVIDIA Source Code License](ludus_renderer/licenses/nvdiffrast/LICENSE.txt).
