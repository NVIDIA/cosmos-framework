# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""JIT compilation of C++/CUDA plugin for Ludus renderer."""

import logging
import os

import torch
import torch.utils.cpp_extension

from ludus_renderer._cuda_toolchain import configure_cuda_toolchain, find_nvidia_library

_cached_plugin = {}


def _get_plugin(gl=False):
    """Get or compile the C++/CUDA plugin.

    Args:
        gl: If True, compile the full GL+CUDA plugin (requires OpenGL).
            If False, compile a CUDA-only plugin (no OpenGL dependency).

    Returns:
        The compiled plugin module.
    """
    assert isinstance(gl, bool)

    # Return cached plugin if already loaded.
    if _cached_plugin.get(gl, None) is not None:
        return _cached_plugin[gl]

    # GL plugin is a superset of CUDA-only. If GL is already loaded and only
    # CUDA is requested, return the GL plugin to avoid duplicate type registration.
    if not gl and _cached_plugin.get(True, None) is not None:
        return _cached_plugin[True]

    # Make sure we can find the necessary compiler and library binaries.
    if os.name == "nt":
        lib_dir = os.path.dirname(__file__) + r"\..\lib"

        def find_cl_path():
            import glob

            def get_sort_key(x):
                x = x.split("\\")[3:]
                x[1] = {
                    "BuildTools": "~0",
                    "Community": "~1",
                    "Pro": "~2",
                    "Professional": "~3",
                    "Enterprise": "~4",
                }.get(x[1], x[1])
                return x

            vs_relative_path = r"\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64"
            paths = glob.glob(r"C:\Program Files" + vs_relative_path)
            paths += glob.glob(r"C:\Program Files (x86)" + vs_relative_path)
            if paths:
                return sorted(paths, key=get_sort_key)[-1]

        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
            os.environ["PATH"] += ";" + cl_path

    toolchain = configure_cuda_toolchain()

    # Compiler options.
    common_opts = ["-DNVDR_TORCH"]
    cc_opts = []
    include_paths = [str(toolchain.include_dir)]
    if os.name == "nt":
        cc_opts += ["/wd4067", "/wd4624"]

    # Linker options and source files depend on GL mode.
    ldflags = []
    if gl:
        if os.name == "posix":
            nvjpeg_include, nvjpeg_lib, nvjpeg_soname = find_nvidia_library(
                ("nvidia-nvjpeg", "nvidia-nvjpeg-cu12", "nvidia-nvjpeg-cu11"),
                "nvjpeg.h",
                ("libnvjpeg.so.13", "libnvjpeg.so.12", "libnvjpeg.so.11", "libnvjpeg.so"),
                toolchain,
            )
            include_paths.append(str(nvjpeg_include))
            ldflags = [
                "-lGL",
                "-lEGL",
                f"-L{nvjpeg_lib}",
                f"-l:{nvjpeg_soname}",
                f"-Wl,-rpath,{nvjpeg_lib}",
            ]
        elif os.name == "nt":
            libs = ["gdi32", "opengl32", "user32", "setgpu", "nvjpeg"]
            ldflags = ["/LIBPATH:" + lib_dir] + ["/DEFAULTLIB:" + x for x in libs]

        source_files = [
            "../_cpp/common/common.cpp",
            "../_cpp/common/glutil.cpp",
            "../_cpp/render/ludus_gl.cpp",
            "../_cpp/render/ludus_timestamped_gl.cpp",
            "../_cpp/render/ludus_jpeg.cu",
            "../_cpp/render/ludus_cuda.cu",
            "../_cpp/cudaraster/CudaRaster.cpp",
            "../_cpp/cudaraster/RasterImpl.cpp",
            "../_cpp/cudaraster/Buffer.cpp",
            "../_cpp/cudaraster/RasterImpl_kernel.cu",
            "../_cpp/bindings/torch_bindings_gl.cpp",
            "../_cpp/bindings/torch_rasterize_gl.cpp",
            "../_cpp/bindings/torch_rasterize_cuda.cpp",
        ]
    else:
        source_files = [
            "../_cpp/common/common.cpp",
            "../_cpp/render/ludus_cuda.cu",
            "../_cpp/render/ludus_jpeg.cu",
            "../_cpp/cudaraster/CudaRaster.cpp",
            "../_cpp/cudaraster/RasterImpl.cpp",
            "../_cpp/cudaraster/Buffer.cpp",
            "../_cpp/cudaraster/RasterImpl_kernel.cu",
            "../_cpp/bindings/torch_bindings_cuda.cpp",
            "../_cpp/bindings/torch_rasterize_cuda.cpp",
        ]

    # PyTorch detects the local GPU unless callers select an architecture for
    # headless compilation (for example, CI sets TORCH_CUDA_ARCH_LIST=8.9).

    # Warn about GLEW conflicts
    if gl and (os.name == "posix") and ("libGLEW" in os.environ.get("LD_PRELOAD", "")):
        logging.getLogger("ludus_renderer").warning(
            "Warning: libGLEW is being loaded via LD_PRELOAD, and will probably conflict with the OpenGL plugin"
        )

    # Check for stale lock files
    plugin_name = "ludus_renderer_plugin" + ("_gl" if gl else "_cuda")
    try:
        lock_fn = os.path.join(torch.utils.cpp_extension._get_build_directory(plugin_name, False), "lock")
        if os.path.exists(lock_fn):
            logging.getLogger("ludus_renderer").warning("Lock file exists in build directory: '%s'" % lock_fn)
    except Exception:
        pass

    # Speed up compilation on Windows
    if os.name == "nt":
        os.environ["VSCMD_SKIP_SENDTELEMETRY"] = "1"
        try:
            import distutils._msvccompiler
            import functools

            if not hasattr(distutils._msvccompiler._get_vc_env, "__wrapped__"):
                distutils._msvccompiler._get_vc_env = functools.lru_cache()(distutils._msvccompiler._get_vc_env)
        except Exception:
            pass

    # Compile and cache
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    _cached_plugin[gl] = torch.utils.cpp_extension.load(
        name=plugin_name,
        sources=source_paths,
        extra_include_paths=include_paths,
        extra_cflags=common_opts + cc_opts,
        extra_cuda_cflags=common_opts + ["-lineinfo"],
        extra_ldflags=ldflags,
        with_cuda=True,
        verbose=False,
    )

    return _cached_plugin[gl]


def _get_any_plugin():
    """Return whichever plugin variant is already loaded (prefer GL, fall back to CUDA)."""
    if _cached_plugin.get(True) is not None:
        return _cached_plugin[True]
    return _get_plugin(gl=False)


def get_log_level():
    """Get current log level.

    Returns:
        Current log level. See `set_log_level()` for possible values.
    """
    return _get_any_plugin().get_log_level()


def set_log_level(level):
    """Set log level.

    Log levels follow the convention on the C++ side of Torch:
        0 = Info,
        1 = Warning,
        2 = Error,
        3 = Fatal.
    The default log level is 1.

    Args:
        level: New log level as integer.
    """
    _get_any_plugin().set_log_level(level)
