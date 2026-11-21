# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Resolve the explicitly owned physical GPU across Torch/NVML UUID spellings."""

from uuid import UUID


def canonical_gpu_uuid(value: str) -> str:
    """Torch may omit NVML's GPU- prefix; an ordinal is never an identity."""
    if not isinstance(value, str):
        raise ValueError("GPU monitoring requires the allocated physical GPU UUID")
    raw = value.removeprefix("GPU-")
    try:
        parsed = UUID(raw)
    except ValueError as error:
        raise ValueError("GPU monitoring requires a physical GPU UUID, not an ordinal or MIG identifier") from error
    if str(parsed) != raw.lower():
        raise ValueError("GPU monitoring requires a canonical hyphenated physical GPU UUID")
    return "GPU-" + str(parsed)


def resolve_gpu_handle(nvml, value: str):
    """Verify the UUID round trip without guessing a device index or fallback."""
    expected = canonical_gpu_uuid(value)
    handle = nvml.nvmlDeviceGetHandleByUUID(expected)
    actual = nvml.nvmlDeviceGetUUID(handle)
    if isinstance(actual, bytes):
        actual = actual.decode("ascii")
    if canonical_gpu_uuid(actual) != expected:
        raise ValueError("NVML returned a different GPU from the explicitly allocated UUID")
    return handle, expected
