# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Inference preserves physical rig identities when selecting camera views."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from cosmos_framework.inference import inference
from cosmos_framework.inference.lidar_transfer import build_joint_cam_lidar_batch
from cosmos_framework.inference.multiview import get_multiview_sample_data
from cosmos_framework.model.generator.utils.rig_view_embedding import vision_view_ids


@pytest.mark.parametrize("num_embeddings", [0, 12])
@pytest.mark.parametrize(
    "camera_keys,expected",
    [
        (["camera_left_fisheye_200fov"], [8]),
        (["camera_front_tele_30fov", "camera_cross_left_120fov"], [6, 5]),
        (["custom_camera"], None),
    ],
)
def test_rgb_batch_preserves_physical_ids(
    monkeypatch: pytest.MonkeyPatch, num_embeddings: int, camera_keys: list[str], expected: list[int] | None
) -> None:
    monkeypatch.setattr(inference, "_get_prompt_sample_data", lambda *args, **kwargs: {"ai_caption": [""]})
    args: Any = SimpleNamespace(
        multiview=SimpleNamespace(views=[SimpleNamespace(camera_key=key) for key in camera_keys], has_control=False),
        num_frames=3,
        vision_size=(4, 2),
        transfer_hints={},
        share_vision_temporal_positions=True,
        fps=30,
    )
    model: Any = SimpleNamespace(
        config=SimpleNamespace(diffusion_expert_config=SimpleNamespace(num_view_embeddings=num_embeddings))
    )
    if num_embeddings and expected is None:
        with pytest.raises(ValueError, match="known MADS camera keys"):
            get_multiview_sample_data(args, model, device="cpu", condition_frames=[None] * len(camera_keys))
        return
    batch = get_multiview_sample_data(args, model, device="cpu", condition_frames=[None] * len(camera_keys))
    if not num_embeddings:
        assert "view_indices_selection" not in batch
        return
    ids = vision_view_ids(
        batch, batch_size=1, item_counts=[1], views_per_item=[len(camera_keys)], num_embeddings=num_embeddings
    )
    assert ids[0].tolist() == expected
    assert batch["view_indices_selection"].dtype == torch.int64


def test_explicit_per_view_negative_is_formatted_for_each_output() -> None:
    args: Any = SimpleNamespace(
        fps=30,
        num_frames=297,
        aspect_ratio="16,9",
        native_prompt_upsampling=False,
        resolution_template=None,
        num_outputs=2,
        negative_prompt="unused global negative",
        per_view_negative_prompt="blurry road",
        autoregressive=False,
    )
    model: Any = SimpleNamespace(input_caption_key="ai_caption")
    out = inference._get_per_view_prompt_sample_data(
        args,
        model,
        ["front road", "rear road"],
        h=480,
        w=832,
        duration_template="{duration} seconds.",
    )
    assert len(out["ai_caption"]) == 2
    assert all(len(captions) == 2 for captions in out["ai_caption"])
    assert out["neg_ai_caption"][0] == out["neg_ai_caption"][1]
    assert "blurry road" in out["neg_ai_caption"][0]
    assert "9 seconds." in out["neg_ai_caption"][0]
    assert "unused global negative" not in out["neg_ai_caption"][0]


@pytest.mark.parametrize("camera_ids", [[8, 6], [8]])
def test_joint_batch_preserves_ids_for_both_control_and_target(camera_ids: list[int]) -> None:
    kwargs = dict(
        camera_control=[torch.zeros(3, 4, 4, 8), torch.zeros(3, 4, 4, 8)],
        lidar_control=torch.ones(3, 2, 4, 8),
        num_camera_frames=4,
        num_lidar_sweeps=2,
        camera_height=4,
        camera_width=8,
        camera_fps=30,
        camera_view_ids=camera_ids,
        prompt_key="ai_caption",
        prompt=["left camera", "front camera"],
        negative_prompt=None,
        device="cpu",
    )
    if len(camera_ids) != 2:
        with pytest.raises(ValueError, match="Expected 2 camera view IDs"):
            build_joint_cam_lidar_batch(**kwargs)
        return
    batch = build_joint_cam_lidar_batch(**kwargs)
    ids = vision_view_ids(batch, batch_size=1, item_counts=[2], views_per_item=[2, 2], num_embeddings=12)
    assert [value.tolist() for value in ids] == [[8, 6], [8, 6]]
