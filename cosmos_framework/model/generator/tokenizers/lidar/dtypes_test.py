# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.model.generator.tokenizers.lidar.dtypes import as_torch_dtype

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_passes_through_a_real_dtype() -> None:
    assert as_torch_dtype(torch.bfloat16) is torch.bfloat16


@pytest.mark.parametrize("name", ["float32", "torch.float32"])
def test_accepts_the_serialized_string_form(name: str) -> None:
    """An exported config.json carries the dtype as a string, with or without the prefix."""
    assert as_torch_dtype(name) is torch.float32


def test_rejects_a_name_that_is_not_a_dtype() -> None:
    """torch.nn is a real attribute of torch but not a dtype; it must not slip through."""
    with pytest.raises(ValueError, match="Not a torch dtype name"):
        as_torch_dtype("nn")


def test_rejects_a_non_string_non_dtype() -> None:
    with pytest.raises(TypeError):
        as_torch_dtype(32)  # type: ignore[arg-type]


def test_torch_dtype_is_not_constructible_from_a_string() -> None:
    """Pins why this helper resolves the name by attribute lookup.

    cosmos_framework/model/generator/diffusion/rectified_flow.py does
    `torch.dtype(dtype) if isinstance(dtype, str) else dtype`, which cannot work --
    this asserts the reason so the idiom is not copied here.
    """
    with pytest.raises(TypeError, match="cannot create 'torch.dtype' instances"):
        torch.dtype("float32")  # type: ignore[call-arg]
