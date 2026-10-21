# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU architecture-admission checks for gapped replay, not generic Flash4."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from cosmos_framework.model.generator.mot import maskless_attention as attention


@pytest.mark.parametrize("capability", [(9, 0), (10, 0), (10, 3), (10, 9)])
def test_existing_architectures_do_not_acquire_the_sm120_dependency_requirement(capability):
    with (
        patch.object(torch.version, "hip", None),
        patch.object(torch.cuda, "get_device_capability", return_value=capability),
        patch.object(attention, "_flash_attn_interface", return_value=Mock()),
        patch.object(attention, "sm120_gapped_forward_unavailable_reason") as prerequisite,
    ):
        assert attention.gapped_replay_unavailable_reason(torch.device("cuda:0")) is None
        prerequisite.assert_not_called()


@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (9, 1), (11, 0), (12, 1), (13, 0)])
def test_other_architectures_stay_rejected_without_loading_fa4(capability):
    with (
        patch.object(torch.version, "hip", None),
        patch.object(torch.cuda, "get_device_capability", return_value=capability),
        patch.object(attention, "_flash_attn_interface") as loader,
    ):
        assert attention.gapped_replay_unavailable_reason(torch.device("cuda")) is not None
        loader.assert_not_called()


@pytest.mark.parametrize("prerequisite_reason", [None, "known-unsafe FA4 version"])
def test_sm120_always_checks_the_forward_prerequisite(prerequisite_reason):
    backend = Mock()
    with (
        patch.object(torch.version, "hip", None),
        patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
        patch.object(attention, "_flash_attn_interface", return_value=backend),
        patch.object(attention, "sm120_gapped_forward_unavailable_reason", return_value=prerequisite_reason) as check,
    ):
        assert attention.gapped_replay_unavailable_reason(torch.device("cuda")) == prerequisite_reason
        check.assert_called_once_with(backend)


def test_cpu_and_rocm_are_rejected_before_querying_cuda():
    with patch.object(torch.cuda, "get_device_capability") as capability:
        assert "CUDA device" in attention.gapped_replay_unavailable_reason(torch.device("cpu"))
        with patch.object(torch.version, "hip", "6.0"):
            assert "ROCm" in attention.gapped_replay_unavailable_reason(torch.device("cuda"))
        capability.assert_not_called()


def test_missing_optional_backend_is_reported():
    with (
        patch.object(torch.version, "hip", None),
        patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
        patch.object(attention, "_flash_attn_interface", side_effect=ImportError("not installed")),
    ):
        assert "could not be imported" in attention.gapped_replay_unavailable_reason(torch.device("cuda"))


def test_direct_sm120_forward_rejects_before_runtime_or_kernel_access():
    # Exercise the registered Python body without dispatching a CUDA custom op.
    q = SimpleNamespace(device=torch.device("cpu"))
    backend = Mock()
    with (
        patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
        patch.object(attention, "_flash_attn_interface", return_value=backend),
        patch.object(attention, "sm120_gapped_forward_unavailable_reason", return_value="unsafe FA4") as check,
        patch.object(attention, "_runtime") as runtime,
    ):
        with pytest.raises(RuntimeError, match="unsafe FA4"):
            attention._gapped_replay_forward._init_fn(q, None, None, None)
        check.assert_called_once_with(backend)
        runtime.assert_not_called()
        backend._flash_attn_fwd.assert_not_called()


@pytest.mark.parametrize("capability", [(9, 0), (10, 0), (12, 0)])
def test_direct_forward_reaches_existing_runtime_after_admission(capability):
    q = SimpleNamespace(device=torch.device("cpu"), shape=(1, 1, 32, 128))
    k = SimpleNamespace(shape=(1, 1, 8, 128))
    with (
        patch.object(torch.cuda, "get_device_capability", return_value=capability),
        patch.object(attention, "_flash_attn_interface", return_value=Mock()),
        patch.object(attention, "sm120_gapped_forward_unavailable_reason", return_value=None) as check,
        patch.object(attention, "_runtime", side_effect=RuntimeError("runtime sentinel")),
    ):
        with pytest.raises(RuntimeError, match="runtime sentinel"):
            attention._gapped_replay_forward._init_fn(q, k, None, None)
        assert check.call_count == int(capability == (12, 0))


def test_sm120_backward_rejects_before_loading_backend_or_runtime():
    q = SimpleNamespace(device=torch.device("cpu"))
    with (
        patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
        patch.object(attention, "_flash_attn_interface") as loader,
        patch.object(attention, "_runtime") as runtime,
    ):
        with pytest.raises(RuntimeError, match="inference-only"):
            attention._gapped_replay_backward._init_fn(q, None, None, None, None, None, None)
        loader.assert_not_called()
        runtime.assert_not_called()


@pytest.mark.parametrize("capability", [(9, 0), (10, 0)])
def test_existing_backward_keeps_its_backend_path(capability):
    q = SimpleNamespace(device=torch.device("cpu"))
    with (
        patch.object(torch.cuda, "get_device_capability", return_value=capability),
        patch.object(attention, "_flash_attn_interface", side_effect=RuntimeError("backend sentinel")),
    ):
        with pytest.raises(RuntimeError, match="backend sentinel"):
            attention._gapped_replay_backward._init_fn(q, None, None, None, None, None, None)
