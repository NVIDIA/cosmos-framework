# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU checks for Flash4 architecture support and kernel restrictions."""

from functools import partial
from unittest.mock import patch

import pytest
import torch

from cosmos_framework.model.attention.flash4.checks import flash4_attention_check
from cosmos_framework.model.attention.flash4.meta import get_bwd_dtypes, get_fwd_dtypes
from cosmos_framework.model.attention.masks import CausalType

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("arch_tag", [80, 89, 90, 100, 103, 110, 120, 121])
def test_flash4_architecture_dtypes(arch_tag: int) -> None:
    expected = [torch.float16, torch.bfloat16] if arch_tag in (90, 100, 103, 110) else []
    assert get_fwd_dtypes(arch_tag) == expected
    assert get_bwd_dtypes(arch_tag) == expected


@pytest.mark.parametrize(
    "arch_tag,head_dim,head_dim_v,heads_kv,requires_grad,supported",
    [
        (90, 128, 128, 2, True, True),
        (90, 192, 128, 8, True, True),
        (90, 192, 128, 2, True, False),
        (90, 192, 128, 2, False, True),
        (90, 256, 256, 8, True, False),
        (90, 256, 256, 8, False, True),
        (100, 256, 256, 2, True, True),
        (103, 192, 128, 2, True, True),
        (90, 24, 24, 8, True, False),
        (90, 24, 24, 8, False, True),
    ],
)
def test_flash4_kernel_restrictions(
    arch_tag: int,
    head_dim: int,
    head_dim_v: int,
    heads_kv: int,
    requires_grad: bool,
    supported: bool,
) -> None:
    # Exercise production validation without importing or launching a CUDA kernel.
    with (
        patch("cosmos_framework.model.attention.flash4.checks.FLASH4_SUPPORTED", True),
        patch("cosmos_framework.model.attention.flash4.checks.get_arch_tag", return_value=arch_tag),
    ):
        assert (
            flash4_attention_check(
                query_shape=torch.Size((1, 128, 8, head_dim)),
                key_shape=torch.Size((1, 128, heads_kv, head_dim)),
                value_shape=torch.Size((1, 128, heads_kv, head_dim_v)),
                dtype=torch.bfloat16,
                device=torch.device("cuda"),
                requires_grad=requires_grad,
                is_causal=False,
                causal_type=CausalType.BottomRight,
                is_varlen=True,
            )
            is supported
        )


@pytest.mark.parametrize("arch_tag", [100, 103])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_varlen", [False, True])
@pytest.mark.parametrize(
    "head_dim,requires_grad,deterministic,supported",
    [(256, True, True, False), (256, True, False, True), (256, False, True, True), (128, True, True, True)],
)
def test_flash4_deterministic_backward_restrictions(
    arch_tag: int,
    dtype: torch.dtype,
    is_varlen: bool,
    head_dim: int,
    requires_grad: bool,
    deterministic: bool,
    supported: bool,
) -> None:
    with (
        patch("cosmos_framework.model.attention.flash4.checks.FLASH4_SUPPORTED", True),
        patch("cosmos_framework.model.attention.flash4.checks.get_arch_tag", return_value=arch_tag),
    ):
        check = partial(
            flash4_attention_check,
            query_shape=torch.Size((1, 128, 8, head_dim)),
            key_shape=torch.Size((1, 128, 8, head_dim)),
            value_shape=torch.Size((1, 128, 8, head_dim)),
            dtype=dtype,
            device=torch.device("cuda"),
            requires_grad=requires_grad,
            is_causal=False,
            causal_type=CausalType.BottomRight,
            is_varlen=is_varlen,
            deterministic=deterministic,
        )
        assert check() is supported
        if not supported:
            with pytest.raises(ValueError, match="does not support deterministic backward with head dim 256"):
                check(raise_error=True)
