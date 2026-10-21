# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.inference.transfer_output import restore_transfer_output_size

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_restore_keeps_preview_panels_separate_and_preserves_frames() -> None:
    video = torch.zeros(3, 19, 4, 16)  # [C,T,Hm,Wm*2]
    video[:, :, :, 8:] = 1.0  # [C,T,Hm,Wm]
    restored = restore_transfer_output_size(video, original_hw=(6, 10), model_hw=(4, 8))  # [C,T,Hs,Ws*2]
    assert restored.shape == (3, 19, 6, 20)
    assert torch.equal(restored[:, :, :, :10], torch.zeros_like(restored[:, :, :, :10]))
    assert torch.allclose(restored[:, :, :, 10:], torch.ones_like(restored[:, :, :, 10:]))


def test_restore_retains_both_image_borders_and_temporal_order() -> None:
    video = torch.zeros(3, 3, 8, 8)  # [C,T,Hm,Wm]
    video[0, :, :, 0] = 1.0  # [T,Hm]
    video[1, :, :, -1] = 1.0  # [T,Hm]
    video[2, 1] = 0.5  # [Hm,Wm]
    video[2, 2] = 1.0  # [Hm,Wm]
    restored = restore_transfer_output_size(video, original_hw=(16, 16), model_hw=(8, 8))  # [C,T,Hs,Ws]
    assert restored[0, :, :, 0].min() > 0.9
    assert restored[1, :, :, -1].min() > 0.9
    assert restored[0, :, :, -1].max() == 0
    assert torch.allclose(restored[2].mean(dim=(1, 2)), torch.tensor([0.0, 0.5, 1.0]))
    assert restored.min() >= 0 and restored.max() <= 1


def test_restore_native_size_is_a_noop() -> None:
    video = torch.rand(3, 2, 4, 8)  # [C,T,Hm,Wm]
    assert restore_transfer_output_size(video, original_hw=(4, 8), model_hw=(4, 8)) is video


@pytest.mark.parametrize("source_hw,model_hw", [((0, 8), (4, 8)), ((4, 8), (3, 8)), ((4, 8), (4, 7))])
def test_restore_rejects_unknown_or_partial_panel_geometry(
    source_hw: tuple[int, int], model_hw: tuple[int, int]
) -> None:
    video = torch.rand(3, 2, 4, 8)  # [C,T,Hm,Wm]
    with pytest.raises(ValueError):
        restore_transfer_output_size(video, original_hw=source_hw, model_hw=model_hw)
