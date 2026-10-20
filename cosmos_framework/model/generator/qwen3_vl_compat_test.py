# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import torch

from cosmos_framework.model.generator.qwen3_vl_compat import (
    apply_qwen3_vl_patch_embed_compat,
    should_use_linear_patch_embed,
)


class _PatchEmbed(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 3
        self.temporal_patch_size = 2
        self.patch_size = 2
        self.embed_dim = 5
        self.proj = torch.nn.Conv3d(3, 5, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(x).view(-1, self.embed_dim)


def test_auto_selects_linear_only_on_sm80() -> None:
    assert should_use_linear_patch_embed("auto", device_capability=(8, 0))
    assert not should_use_linear_patch_embed("auto", device_capability=(9, 0))
    assert should_use_linear_patch_embed("linear", device_capability=(9, 0))
    assert not should_use_linear_patch_embed("conv3d", device_capability=(8, 0))


def test_linear_patch_is_algebraically_equivalent_and_preserves_keys() -> None:
    model = torch.nn.Sequential(_PatchEmbed())
    values = torch.randn(4, 3 * 2 * 2 * 2)
    expected = model(values)
    keys_before = set(model.state_dict())

    changed = apply_qwen3_vl_patch_embed_compat(
        model,
        model_type="qwen3_vl",
        mode="linear",
        device_capability=(8, 0),
    )

    assert changed is True
    torch.testing.assert_close(model(values), expected)
    assert set(model.state_dict()) == keys_before
