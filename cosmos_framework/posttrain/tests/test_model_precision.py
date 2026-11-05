# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0 tests for `model/precision.py`, the two halves of `rl.mixed_precision_mode`: `compute_context` (what the forward
runs under) and `fsdp_param_dtype` (what the FSDP wrap delivers). CPU bf16 autocast is real, so both branches are
observed through matmul output dtypes, not mocked; the one CUDA-only fact (fp32 as an autocast target) is a GPU test."""

from __future__ import annotations

import pytest
import torch

from cosmos_framework.utils.helper_test import RunIf
from cosmos_framework.posttrain.model.precision import compute_context, fsdp_param_dtype


@pytest.mark.L0
@pytest.mark.CPU
def test_fsdp_mode_leaves_the_forward_alone() -> None:
    a = torch.ones(2, 2)
    with compute_context("fsdp", torch.bfloat16, "cpu"):
        assert not torch.is_autocast_enabled("cpu")
        assert (a @ a).dtype is torch.float32


@pytest.mark.L0
@pytest.mark.CPU
def test_autocast_mode_casts_matmuls_to_the_compute_dtype_and_is_reusable() -> None:
    a = torch.ones(2, 2)
    ctx = compute_context("autocast", torch.bfloat16, "cpu")
    for _ in range(2):  # callers build the context once and re-enter it every forward
        with ctx:
            assert torch.is_autocast_enabled("cpu")
            assert (a @ a).dtype is torch.bfloat16
        assert not torch.is_autocast_enabled("cpu")


@pytest.mark.L0
@pytest.mark.CPU
def test_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="mixed precision mode"):
        compute_context("amp", torch.bfloat16, "cpu")
    with pytest.raises(ValueError, match="mixed precision mode"):
        fsdp_param_dtype("amp", torch.bfloat16)


@pytest.mark.L0
@pytest.mark.CPU
def test_fsdp_param_dtype_is_the_compute_dtype_in_fsdp_mode_and_none_under_autocast() -> None:
    """`ShardPlan.param_dtype` is what the FSDP wrap DELIVERS: the compute dtype when the parameter cast is the
    delivery mechanism, `None` under autocast so the wrap casts neither parameters nor forward inputs (a plan that
    carried the storage dtype instead would make FSDP2 cast every wrapped block's inputs back to storage)."""
    assert fsdp_param_dtype("fsdp", torch.bfloat16) is torch.bfloat16
    assert fsdp_param_dtype("autocast", torch.bfloat16) is None


@pytest.mark.L0
@pytest.mark.GPU
@RunIf(min_gpus=1)
def test_cuda_autocast_honours_a_float32_compute_dtype() -> None:
    """`autocast + param_dtype="float32"` is accepted by the resolver, so it must be delivered: CUDA autocast has no
    target-dtype whitelist (the "target dtype is not supported, disabling autocast" branch belongs to the non-CUDA
    devices), so the listed ops upcast bf16 operands to fp32 instead of autocast silently switching itself off."""
    a = torch.ones(2, 2, device="cuda", dtype=torch.bfloat16)
    with compute_context("autocast", torch.float32, "cuda"):
        assert torch.is_autocast_enabled("cuda")
        assert (a @ a).dtype is torch.float32
    assert (a @ a).dtype is torch.bfloat16
