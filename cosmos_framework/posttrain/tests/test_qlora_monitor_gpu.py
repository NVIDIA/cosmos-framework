# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Measurement identity checks; real NVML sampling remains an accelerator gate."""

from types import SimpleNamespace

import pytest

from tools.native_qlora_qualification.monitor_gpu import canonical_gpu_uuid, resolve_gpu_handle

UUID = "6cf26e37-902c-4b91-fe8c-a1f0f50c1e16"


@pytest.mark.parametrize("value", [UUID, "GPU-" + UUID, UUID.upper()])
def test_torch_and_nvml_uuid_spellings(value):
    assert canonical_gpu_uuid(value) == "GPU-" + UUID


@pytest.mark.parametrize("value", ["0", "", "all", "MIG-" + UUID, UUID.replace("-", ""), None])
def test_refuses_ordinal_fallback_or_unqualified_mig(value):
    with pytest.raises(ValueError, match="GPU|physical"):
        canonical_gpu_uuid(value)


@pytest.mark.parametrize("bytes_result", [False, True])
def test_exact_roundtrip(bytes_result):
    requested = []
    handle = object()

    def lookup(value):
        requested.append(value)
        return handle

    value = "GPU-" + UUID
    nvml = SimpleNamespace(
        nvmlDeviceGetHandleByUUID=lookup,
        nvmlDeviceGetUUID=lambda result: value.encode() if bytes_result else value,
    )
    assert resolve_gpu_handle(nvml, UUID) == (handle, value)
    assert requested == [value]


def test_different_gpu_is_rejected():
    nvml = SimpleNamespace(
        nvmlDeviceGetHandleByUUID=lambda value: object(),
        nvmlDeviceGetUUID=lambda handle: "GPU-00000000-0000-0000-0000-000000000000",
    )
    with pytest.raises(ValueError, match="different GPU"):
        resolve_gpu_handle(nvml, UUID)


def test_missing_device_never_falls_back():
    def unavailable(value):
        raise RuntimeError("NVML device unavailable")

    with pytest.raises(RuntimeError, match="unavailable"):
        resolve_gpu_handle(SimpleNamespace(nvmlDeviceGetHandleByUUID=unavailable), UUID)
