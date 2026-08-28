# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.data.generator.augmentors.transfer_control_transform import AddSelectedControlFromVideo

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("modality", ["edge", "blur"])
def test_selected_online_control_uses_existing_operation_and_cleans_private_keys(modality: str) -> None:
    video = torch.randint(0, 256, (3, 5, 24, 32), dtype=torch.uint8)  # [C,T,H,W]
    augmentor = AddSelectedControlFromVideo(input_keys=["video"], use_random=False)
    sample = {
        "video": video,
        "_selected_control_modality": modality,
        "_selected_rgb_stream_id": "cam",
    }
    output = augmentor(sample)
    assert output is not None
    assert output["control_input"].shape == video.shape
    assert "_selected_control_modality" not in output
    assert "_selected_rgb_stream_id" not in output


def test_selected_depth_uses_bilinear_resize_and_accepts_all_zero_control() -> None:
    video = torch.ones(3, 2, 5, 7)  # [C,T,H,W]
    depth = torch.zeros(3, 2, 2, 3)  # [C,T,h,w]
    augmentor = AddSelectedControlFromVideo(input_keys=["video"])
    sample = {
        "video": video,
        "depth": depth,
        "persisted_control": b"raw",
        "_persisted_control_meta": {"stream_type": "depth"},
        "_selected_control_modality": "depth",
    }
    output = augmentor(sample)
    assert output is not None
    assert output["control_input"].shape == video.shape
    assert torch.count_nonzero(output["control_input"]) == 0
    assert "depth" not in output
    assert "persisted_control" not in output
    assert "_persisted_control_meta" not in output


def test_selected_semantic_uses_nearest_neighbor_resize() -> None:
    video = torch.zeros(3, 1, 4, 4)  # [C,T,H,W]
    segmentation = torch.tensor(
        [[[[0, 10], [20, 30]]], [[[1, 11], [21, 31]]], [[[2, 12], [22, 32]]]], dtype=torch.uint8
    )  # [C,T,h,w]
    augmentor = AddSelectedControlFromVideo(input_keys=["video"])
    output = augmentor(
        {
            "video": video,
            "segmentation": segmentation,
            "_selected_control_modality": "seg",
        }
    )
    assert output is not None
    control = output["control_input"]  # [C,T,H,W]
    assert control.shape == video.shape
    assert torch.equal(control[:, :, :2, :2], segmentation[:, :, :1, :1].expand(-1, -1, 2, 2))


def test_selected_control_rejects_unknown_modality() -> None:
    augmentor = AddSelectedControlFromVideo(input_keys=["video"])
    video = torch.zeros(3, 1, 2, 2)  # [C,T,H,W]
    assert augmentor({"video": video, "_selected_control_modality": "wsm"}) is None
