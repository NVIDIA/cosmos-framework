# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Opt-in UUID binding for LIBERO's distinct CUDA and EGL device namespaces.

robosuite 1.4.1 incorrectly compares those ordinals during import. Validate
persistent UUIDs, omit the EGL-only setting during vendor import, then restore
the validated ordinal before context creation. CUDA visibility never changes.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import os
import subprocess
import sys
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass

PROFILE_ENV = "COSMOS_LIBERO_RENDERER_PROFILE"
UUID_PROFILE = "egl_uuid_v1"
_lock = threading.RLock()
_binding: RendererBinding | None = None


@dataclass(frozen=True)
class RendererBinding:
    profile: str
    cuda_visible_devices: str
    gpu_uuid: str
    nvml_index: int
    egl_index: int
    driver: str


def _egl_devices() -> list[dict]:
    """Query metadata only, without initializing a display or CUDA runtime."""
    egl = ctypes.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    egl.eglGetProcAddress.restype = ctypes.c_void_p

    def function(name, result, *arguments):
        address = egl.eglGetProcAddress(name.encode())
        if not address:
            raise RuntimeError(f"{UUID_PROFILE} requires EGL function {name}")
        return ctypes.CFUNCTYPE(result, *arguments)(address)

    query = function(
        "eglQueryDevicesEXT", ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int)
    )
    string = function("eglQueryDeviceStringEXT", ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)
    binary = function(
        "eglQueryDeviceBinaryEXT",
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    )
    devices, count = (ctypes.c_void_p * 64)(), ctypes.c_int()
    if not query(64, devices, ctypes.byref(count)) or not 0 < count.value < 64:
        raise RuntimeError(f"{UUID_PROFILE} cannot enumerate a complete EGL device list")
    records = []
    for index, device in enumerate(devices[: count.value]):
        extensions = string(device, 0x3055)
        if not extensions or b"EGL_EXT_device_persistent_id" not in extensions.split():
            continue
        identifier, size = (ctypes.c_ubyte * 16)(), ctypes.c_int()
        driver = string(device, 0x335E)
        if not binary(device, 0x335C, 16, identifier, ctypes.byref(size)) or size.value != 16 or not driver:
            raise RuntimeError(f"{UUID_PROFILE} cannot read persistent EGL device identity")
        records.append(
            {
                "egl_index": index,
                "gpu_uuid": "GPU-" + str(uuid.UUID(bytes=bytes(identifier))),
                "driver": driver.decode(),
            }
        )
    return records


def resolve_renderer_binding() -> RendererBinding:
    """Resolve exactly one explicit full GPU UUID without assuming CUDA/NVML order."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.startswith("GPU-") or len(visible) != 40:
        raise ValueError(
            f"{UUID_PROFILE} requires one complete GPU UUID in CUDA_VISIBLE_DEVICES; numeric CUDA and NVML ordinals may differ"
        )
    if os.environ.get("MUJOCO_GL", "egl") != "egl" or os.environ.get("PYOPENGL_PLATFORM", "egl") != "egl":
        raise ValueError(f"{UUID_PROFILE} requires the EGL renderer")
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    devices = [tuple(part.strip() for part in line.split(",")) for line in rows.splitlines()]
    matched = [(index, identity) for index, identity in devices if visible == identity]
    if len(matched) != 1:
        raise ValueError(f"{UUID_PROFILE} cannot uniquely resolve CUDA_VISIBLE_DEVICES={visible!r}")
    index, identity = matched[0]
    selected = [row for row in _egl_devices() if row["gpu_uuid"] == identity]
    if len(selected) != 1 or selected[0]["driver"] != "nvidia":
        raise RuntimeError(f"{UUID_PROFILE} requires one NVIDIA EGL device matching {identity}")
    ordinal = selected[0]["egl_index"]
    requested = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    if requested is not None and requested != str(ordinal):
        raise ValueError(f"MUJOCO_EGL_DEVICE_ID={requested!r} differs from EGL index {ordinal} for {identity}")
    return RendererBinding(UUID_PROFILE, visible, identity, int(index), ordinal, "nvidia")


@contextmanager
def vendor_renderer_import():
    """Admit the renderer before vendor imports, including spawned workers."""
    global _binding
    profile = os.environ.get(PROFILE_ENV, "legacy")
    if profile == "legacy":
        if _binding is not None:
            raise RuntimeError("LIBERO renderer profile changed; start a fresh simulator process")
        yield
        return
    if profile != UUID_PROFILE:
        raise ValueError(f"unsupported LIBERO renderer profile {profile!r}")
    with _lock:
        if importlib.metadata.version("robosuite") != "1.4.1":
            raise RuntimeError(f"{UUID_PROFILE} requires robosuite 1.4.1")
        resolved = resolve_renderer_binding()
        if _binding is not None:
            if resolved != _binding:
                raise RuntimeError("LIBERO renderer device changed; start a fresh simulator process")
            yield
            return
        if "robosuite.utils.binding_utils" in sys.modules or "robosuite.renderers.context.egl_context" in sys.modules:
            raise RuntimeError(f"enable {PROFILE_ENV}={UUID_PROFILE} before importing the vendor renderer")
        original = os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
        try:
            yield
            if os.environ.get("CUDA_VISIBLE_DEVICES") != resolved.cuda_visible_devices:
                raise RuntimeError("vendor import changed CUDA visibility")
        except BaseException:
            if original is None:
                os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
            else:
                os.environ["MUJOCO_EGL_DEVICE_ID"] = original
            raise
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(resolved.egl_index)
        _binding = resolved


def renderer_identity() -> dict:
    """Actual admitted device, separate from physics and task protocol identity."""
    return asdict(_binding) if _binding is not None else {"profile": "legacy"}
