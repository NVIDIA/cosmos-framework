# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Check negative metadata against the actual camera-major target ownership."""

import json
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.distillation.self_forcing_dmd import SelfForcingDMDConfig
from cosmos_framework.model.generator.distillation.self_forcing_dmd_joint import JointSelfForcingDMDModel
from cosmos_framework.model.generator.sim_auto_models import SimAutoTeacherForcingDCMModel
from cosmos_framework.model.generator.utils.multiview_negative_prompt import multiview_negative_captions

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


def make_batch() -> dict:
    return {
        "video": [
            torch.empty(1, 3, 2 * 97, 16, 32),  # [1,C,V*T,H,W]
            torch.empty(1, 3, 2 * 97, 16, 32),  # [1,C,V*T,H,W]
            torch.empty(1, 3, 301, 32, 64),  # [1,C,T,H,W]
            torch.empty(1, 3, 301, 32, 64),  # [1,C,T,H,W]
        ],
        "num_vision_items_per_sample": [2, 2],
        "sample_n_views": [torch.tensor(2), torch.tensor(1)],  # list[[]]
        "num_video_frames_per_view": [torch.tensor(97), torch.tensor(301)],  # list[[]]
        "conditioning_fps": [torch.tensor(30.0), torch.tensor(29.97)],  # list[[]]
        "enable_per_camera_vae_encoding": [True, True],
    }


def test_camera_major_duration_and_mixed_sample_ownership() -> None:
    captions = multiview_negative_captions(make_batch(), 2, "flickering")
    assert "3.0 seconds" in captions[0]
    assert "6.0 seconds" not in captions[0]
    assert "10.0 seconds" in captions[1]
    assert "16" in captions[0] and "32" in captions[0]
    assert "32" in captions[1] and "64" in captions[1]


def test_structured_negative_metadata_matches_dataset_schema() -> None:
    captions = multiview_negative_captions(make_batch(), 2, '{"description": "flickering"}')
    first, second = [json.loads(value) for value in captions]
    assert first["description"] == "flickering"
    assert first["duration"] == "3s"
    assert first["fps"] == 30
    assert second["duration"] == "10s"
    assert second["fps"] == pytest.approx(29.97)
    assert first != second


def test_empty_caption_keeps_metadata_without_a_placeholder() -> None:
    captions = multiview_negative_captions(make_batch(), 2, "")
    assert all("SIM_AUTO_EMPTY_CAPTION" not in caption for caption in captions)
    assert "3.0 seconds" in captions[0]


@pytest.mark.parametrize(
    "key,value",
    [
        ("sample_n_views", [3, 1]),
        ("num_video_frames_per_view", [96, 301]),
        ("conditioning_fps", [0, 30]),
        ("conditioning_fps", [True, 30]),
        ("conditioning_fps", [float("nan"), 30]),
        ("sample_n_views", [2.5, 1]),
    ],
)
def test_inconsistent_metadata_is_rejected(key: str, value: list) -> None:
    batch = make_batch()
    batch[key] = value
    with pytest.raises(ValueError):
        multiview_negative_captions(batch, 2, "bad quality")


@pytest.mark.parametrize("model_type", [SimAutoTeacherForcingDCMModel, JointSelfForcingDMDModel])
def test_all_causal_paths_use_metadata(model_type: type) -> None:
    host = SimpleNamespace(
        input_video_key="video",
        config=SimpleNamespace(
            teacher_forcing_dcm_teacher_negative_prompt_keep_metadata=True,
            teacher_forcing_dcm_teacher_negative_prompt="flickering",
            teacher_negative_prompt_keep_metadata=True,
            teacher_negative_prompt_per_view_metadata=True,
            teacher_negative_prompt="flickering",
        ),
    )
    assert model_type._make_teacher_negative_captions(host, 2, make_batch()) == multiview_negative_captions(
        make_batch(), 2, "flickering"
    )


def test_joint_negative_metadata_is_opt_in() -> None:
    model = object.__new__(JointSelfForcingDMDModel)
    torch.nn.Module.__init__(model)
    model.config = SelfForcingDMDConfig(teacher_negative_prompt="flickering")
    model.input_video_key = "video"
    assert model._make_teacher_negative_captions(2, make_batch()) == ["flickering", "flickering"]
    model.config.teacher_negative_prompt_keep_metadata = True
    with pytest.raises(ValueError, match="single-view video batches"):
        model._make_teacher_negative_captions(2, make_batch())
    model.config.teacher_negative_prompt_per_view_metadata = True
    assert model._make_teacher_negative_captions(2, make_batch()) == multiview_negative_captions(
        make_batch(), 2, "flickering"
    )
