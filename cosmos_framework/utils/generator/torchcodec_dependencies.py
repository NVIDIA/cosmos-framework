# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Discover wheel-provided NPP libraries before importing CUDA TorchCodec."""

import ctypes
import sys
from functools import cache
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
from types import ModuleType


@cache
def preload_npp(cuda_version: str | None) -> ctypes.CDLL | None:
    """Load NPP for the selected CUDA build without changing library search paths.

    CUDA TorchCodec links to NPP even for CPU decoding. Its Linux wheels do not
    locate the NPP wheel automatically. Prefer an already discoverable library;
    otherwise use the installed distribution's metadata, which supports both
    CUDA 12's nvidia/npp/lib and CUDA 13's nvidia/cu13/lib layout.
    """
    if sys.platform != "linux" or cuda_version is None:
        return None
    cuda_major = cuda_version.split(".", 1)[0]
    packages = {"12": "nvidia-npp-cu12", "13": "nvidia-npp"}
    if cuda_major not in packages:
        return None
    library = f"libnppicc.so.{cuda_major}"
    try:
        return ctypes.CDLL(library)
    except OSError:
        pass

    try:
        package = distribution(packages[cuda_major])
    except PackageNotFoundError:
        # Leave missing dependencies to TorchCodec's full loader diagnostics.
        return None
    for path in package.files or ():
        if path.name == library:
            # NPP's own $ORIGIN runpath resolves its sibling libnppc dependency.
            return ctypes.CDLL(str(package.locate_file(path)))
    return None


def load_torchcodec() -> ModuleType:
    """Prepare wheel-installed NPP in this process before importing TorchCodec.

    The cached preload keeps the native handle alive and is shared by both the
    common video reader and inference's direct decoder entry point.
    """
    import torch

    preload_npp(torch.version.cuda)
    return import_module("torchcodec")
