# Environment requirements and troubleshooting

Use the selected Cosmos Framework environment from the [setup guide](setup.md).
Run package installation at `FRAMEWORK_DIR`, the Framework checkout root.
Nano conversion retains its separately pinned checkout and environment; see
[conversion](../post-training/references/nano-conversion.md).

Use Linux x86-64 or aarch64 with glibc 2.35 or newer. Generation and training
require an NVIDIA GPU from Ampere or later and a compatible driver. Use Python
**3.13**: the locked GB200/aarch64 CUDA 13.0 NVFP4 wheel requires it. The argument
and inference modules also need `typing.override`, available in Python 3.12+.

The native local data loader uses PyAV, TorchCodec, Transformers and FFmpeg
shared libraries. Install FFmpeg 8 for TorchCodec 0.10, including `ffmpeg` and
`ffprobe` for media preparation.
Download the text tokenizer from the [Nano model repository](https://huggingface.co/nvidia/Cosmos3-Nano/tree/e59a53c25979a090fa8706c9acc0c254a6e89b92/text_tokenizer)
as described in [bundle preparation](local-dataset.md).

`nvidia-smi` reports the driver's supported CUDA level; `torch.version.cuda`
reports PyTorch's runtime. The driver must support that runtime. The version
strings can differ without requiring a CUDA reinstall.

## Complete the CUDA 13.0 environment

Framework's `cu130-train` group already includes `nvidia-npp==13.0.0.50`.
The `cu128-train` group includes its matched CUDA 12 NPP package. Keep the lockfile
and selected group intact. The native libraries and compiler toolkit below
are still needed when the host does not supply them.

The setup command's `--all-extras` includes the `train` extra, which supplies
the locked `mujoco==3.10.0` used by native Simulation preprocessing. A selective
training install must include `--extra train` alongside the matched CUDA
training group. Keep that version rather than installing MuJoCo separately.
The `libero` group selects `mujoco==3.3.2` for a different workflow and is
explicitly incompatible with the `train` extra.

For hosts without `apt` or root access, use an available Conda installation to
create a native-library prefix. Choose a location visible at the same absolute
path on every training node and export it as `TRANSFER_NATIVE_PREFIX`. Leave
the Python packages in the activated uv environment. Do not activate this
native prefix over it.

```bash
set -euo pipefail
export TRANSFER_NATIVE_PREFIX="${TRANSFER_NATIVE_PREFIX:?Set the native-library prefix shared by all training nodes}"
conda create \
  --yes \
  --prefix "$TRANSFER_NATIVE_PREFIX" \
  --override-channels \
  --channel conda-forge \
  'ffmpeg>=8,<9' libgl xorg-libx11 \
  'cuda-nvcc=13.0' 'cuda-cudart-dev=13.0' 'cuda-crt=13.0' 'cuda-cccl=13.0'
case "$(uname -m)" in
  aarch64) TRANSFER_CUDA_TARGET="$TRANSFER_NATIVE_PREFIX/targets/sbsa-linux" ;;
  x86_64) TRANSFER_CUDA_TARGET="$TRANSFER_NATIVE_PREFIX/targets/x86_64-linux" ;;
  *) printf '%s\n' 'This toolkit layout supports Linux SBSA/aarch64 and x86-64.' >&2; exit 1 ;;
esac
export CUDA_HOME="$TRANSFER_NATIVE_PREFIX/cuda-toolkit"
mkdir -p "$CUDA_HOME"
ln -sfn "$TRANSFER_NATIVE_PREFIX/bin" "$CUDA_HOME/bin"
ln -sfn "$TRANSFER_CUDA_TARGET/include" "$CUDA_HOME/include"
ln -sfn "$TRANSFER_CUDA_TARGET/lib" "$CUDA_HOME/lib64"
ln -sfn "$TRANSFER_CUDA_TARGET/lib" "$CUDA_HOME/lib"
ln -sfn "$TRANSFER_NATIVE_PREFIX/nvvm" "$CUDA_HOME/nvvm"
ln -sfn "$TRANSFER_NATIVE_PREFIX/targets" "$CUDA_HOME/targets"
export PATH="$CUDA_HOME/bin:$PATH"
export TRANSFER_CUDA_LIB="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cu13/lib")')"
mkdir -p "$TRANSFER_NATIVE_PREFIX/runtime-links"
ln -sfn "$TRANSFER_CUDA_LIB/libcudart.so.13" \
  "$TRANSFER_NATIVE_PREFIX/runtime-links/libcudart.so"
export LD_LIBRARY_PATH="$TRANSFER_NATIVE_PREFIX/runtime-links:$TRANSFER_CUDA_LIB:$TRANSFER_NATIVE_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Conda-forge installs CUDA runtime, CRT and CCCL headers under
`targets/sbsa-linux/include` on GB200, and under `targets/x86_64-linux/include`
on x86-64. Its prefix root is not a conventional CUDA toolkit root. The
local `cuda-toolkit` directory gives PyTorch the expected `bin`, `include`
and `lib64` paths while retaining NVCC's `nvvm` and target directories. Use the
runtime libraries in the target directory; CUDA stub libraries cannot serve as
runtime libraries.

The local `libcudart.so` link satisfies the training utility's
`ctypes.CDLL("libcudart.so")` call when the runtime wheel supplies only the
versioned library. It points to the runtime selected by the Python environment
and does not modify system CUDA. NPP's `site-packages/nvidia/cu13/lib` must stay
on the library path for TorchCodec. Do not clear `LD_LIBRARY_PATH` after this
setup. Remove stale entries from a different PyTorch installation if they
conflict. Retain required real driver or compatibility libraries, never CUDA
stub libraries.

An existing CUDA 13.0 toolkit and compatible system video libraries can supply
these dependencies without a separate prefix. Use the toolkit version matched
to the selected PyTorch CUDA group.

## Tokenizer compilation

The training `compile_tokenizer` callback uses AOTInductor. Compute hosts need
`nvcc`, CUDA runtime/CRT/CCCL headers, a C++ compiler and a valid `CUDA_HOME`.
For the Conda installation above, point `CUDA_HOME` at `cuda-toolkit`.

The callback's `compile_after_iterations=3` compiles on the fourth step in each
process. Share `IMAGINAIRE_OUTPUT_ROOT` across nodes so ranks can exchange the
compiled packages; follow the [multi-node launch](../post-training/references/multinode-training.md).

## Diagnostics and common failures

Basic diagnostics:

```bash
python --version
uv --version
nvidia-smi \
  --query-gpu=name,driver_version,memory.total \
  --format=csv
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

| Symptom                                                                                        | Next step                                                                                                                                                                                                                                                    |
| ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ImportError: cannot import name 'override' from 'typing'` or a CPython 3.12 NVFP4 wheel error | Install this workflow with Python 3.13 and `uv sync --locked`.                                                                                                                                                                                               |
| `ModuleNotFoundError: No module named 'cosmos_framework'`                                      | Run from `FRAMEWORK_DIR`, activate its `.venv` and verify installation from that checkout's `pyproject.toml`.                                                                                                                                                |
| Missing `cosmos_framework.data.generator.local_simulation`                                     | Use the matched Framework revision containing `configs/simulation/local.py` and its native loader. Check the imported package location against `FRAMEWORK_DIR`.                                                                                              |
| Another `ModuleNotFoundError`                                                                  | Use the complete Framework training environment. Check the selected interpreter and the missing dependency.                                                                                                                                                  |
| `ImportError: cannot import name '_functionalization' from 'torch._C'` in an NGC container     | Check whether inherited `LD_LIBRARY_PATH` entries select libraries from a different PyTorch installation. Use the chosen environment's libraries while retaining any real CUDA driver/compatibility libraries it needs. Do not replace them with CUDA stubs. |
| `fatal error: Python.h: No such file or directory`                                             | Use the supported Python environment and its development headers when building a package from source.                                                                                                                                                        |
| Missing `libavdevice`, `libavcodec` or another FFmpeg shared library                           | Use the FFmpeg libraries compatible with the installed PyAV and TorchCodec versions. Both Python packages must be able to load the FFmpeg shared libraries.                                                                                                  |
| Missing `libnppicc.so.13`, `libGL.so.1`, `libX11.so.6` or `libcudart.so`                       | Apply the CUDA 13.0 environment setup above and preserve its library paths in the job.                                                                                                                                                                       |
| `CUDA_HOME` unset or `AOT compilation produced no loadable functions`                          | Check `nvcc`, the matching CUDA headers and C++ compiler on every node. Inspect the first compile error before retrying.                                                                                                                                     |
| Another `OSError: ... cannot open shared object file`                                          | Inspect the named missing library, installed runtime and driver compatibility before changing dependencies.                                                                                                                                                  |
| `CUDA available False`                                                                         | For generation or training, check the GPU allocation, device visibility and CUDA-enabled PyTorch installation. Caption API clients do not require a visible GPU.                                                                                             |
| `ffprobe: command not found`                                                                   | Install FFmpeg, including `ffprobe`, for media preparation.                                                                                                                                                                                                  |
| CUDA out of memory                                                                             | Use the model-specific GPU-memory, batch-size and clip-length requirements.                                                                                                                                                                                  |

For a support request, include the checkout version, Python/PyTorch versions,
environment or dependency group, relevant command and traceback. Include GPU
model and driver for a GPU failure. Replace private media paths with descriptive
placeholders and avoid dumping the complete process environment.
