# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
nvjpeg - GPU-accelerated JPEG encoding for PyTorch tensors.

This module provides hardware-accelerated JPEG encoding using NVIDIA's nvjpeg library.
It can encode GPU tensors directly to JPEG bytes without copying to CPU.

Example usage:
    import torch
    from ludus_renderer import nvjpeg

    # Create a batch of random images on GPU
    images = torch.randint(0, 255, (4, 3, 480, 640), dtype=torch.uint8, device='cuda')

    # Encode to JPEG
    jpeg_bytes_list = nvjpeg.encode(images, quality=85)

    # Save to files
    for i, jpeg_bytes in enumerate(jpeg_bytes_list):
        with open(f'image_{i}.jpg', 'wb') as f:
            f.write(jpeg_bytes)
"""

import os

import torch
import torch.utils.cpp_extension

from ludus_renderer._cuda_toolchain import configure_cuda_toolchain, find_nvidia_library

_cached_plugin = None


def _get_plugin():
    """Get or compile the nvjpeg plugin."""
    global _cached_plugin

    if _cached_plugin is not None:
        return _cached_plugin

    # Source files
    source_files = [
        "_cpp/nvjpeg/nvjpeg_encoder.cu",
        "_cpp/nvjpeg/nvjpeg_bindings.cpp",
    ]

    toolchain = configure_cuda_toolchain()

    # Compiler options
    common_opts = ["-DNVDR_TORCH"]
    cc_opts = []
    include_paths = [str(toolchain.include_dir)]

    # Linker options
    if os.name == "posix":
        nvjpeg_include, nvjpeg_lib, nvjpeg_soname = find_nvidia_library(
            ("nvidia-nvjpeg", "nvidia-nvjpeg-cu12", "nvidia-nvjpeg-cu11"),
            "nvjpeg.h",
            ("libnvjpeg.so.13", "libnvjpeg.so.12", "libnvjpeg.so.11", "libnvjpeg.so"),
            toolchain,
        )
        include_paths.append(str(nvjpeg_include))
        ldflags = [
            f"-L{nvjpeg_lib}",
            f"-l:{nvjpeg_soname}",
            f"-Wl,-rpath,{nvjpeg_lib}",
        ]
    elif os.name == "nt":
        ldflags = ["-lnvjpeg"]
    else:
        ldflags = []

    # PyTorch detects the local GPU unless callers select an architecture for
    # headless compilation (for example, CI sets TORCH_CUDA_ARCH_LIST=8.9).

    # Speed up compilation on Windows
    if os.name == "nt":
        os.environ["VSCMD_SKIP_SENDTELEMETRY"] = "1"
        cc_opts += ["/wd4067", "/wd4624"]

    # Compile
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    _cached_plugin = torch.utils.cpp_extension.load(
        name="nvjpeg_encoder_plugin",
        sources=source_paths,
        extra_include_paths=include_paths,
        extra_cflags=common_opts + cc_opts,
        extra_cuda_cflags=common_opts + ["-lineinfo"],
        extra_ldflags=ldflags,
        with_cuda=True,
        verbose=False,
    )

    return _cached_plugin


def is_available() -> bool:
    """Check if nvjpeg hardware encoder is available.

    Returns:
        True if nvjpeg is available and initialized successfully.
    """
    try:
        return _get_plugin().is_available()
    except Exception:
        return False


def encode(
    images: torch.Tensor,
    quality: int = 85,
    device_index: int | None = None,
) -> list[bytes]:
    """Encode GPU tensor to JPEG bytes.

    Args:
        images: GPU tensor of shape [B, 3, H, W] or [3, H, W], dtype uint8.
                Must be RGB format (not BGR). Images should be contiguous.
        quality: JPEG quality (1-100, default 85). Higher means better quality
                 but larger file size.
        device_index: CUDA device index for the encoder. If None (default), an encoder
                      for the tensor's device is used (lazy-created if needed).

    Returns:
        List of bytes objects, one per image in the batch.
        For a single image input [3, H, W], returns a list with one element.

    Raises:
        RuntimeError: If images are not on GPU, not uint8, or wrong shape.

    Example:
        >>> images = torch.randint(0, 255, (4, 3, 480, 640), dtype=torch.uint8, device='cuda')
        >>> jpegs = nvjpeg.encode(images)
        >>> len(jpegs)
        4
        >>> # Encode on a specific GPU:
        >>> images_gpu1 = images.to('cuda:1')
        >>> jpegs = nvjpeg.encode(images_gpu1, device_index=1)
    """
    dev = -1 if device_index is None else device_index
    return _get_plugin().encode(images, quality, dev)


def encode_single(
    image: torch.Tensor,
    quality: int = 85,
    device_index: int | None = None,
) -> bytes:
    """Encode a single GPU tensor to JPEG bytes.

    This is a convenience function for encoding a single image.

    Args:
        image: GPU tensor of shape [3, H, W], dtype uint8.
               Must be RGB format (not BGR). Image should be contiguous.
        quality: JPEG quality (1-100, default 85).
        device_index: CUDA device index for the encoder. If None (default), an encoder
                      for the tensor's device is used (lazy-created if needed).

    Returns:
        bytes object containing the JPEG data.

    Example:
        >>> image = torch.randint(0, 255, (3, 480, 640), dtype=torch.uint8, device='cuda')
        >>> jpeg = nvjpeg.encode_single(image)
        >>> type(jpeg)
        <class 'bytes'>
    """
    dev = -1 if device_index is None else device_index
    return _get_plugin().encode_single(image, quality, dev)
