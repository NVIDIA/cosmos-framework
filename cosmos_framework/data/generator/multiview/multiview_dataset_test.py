# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


from __future__ import annotations

import pytest
import torch

from cosmos_framework.configs.simulation.common.auto_profiles import _dataset
from cosmos_framework.data.generator.augmentors.first_caption_paragraph import FirstCaptionParagraph
from cosmos_framework.data.generator.augmentors.text_tokenizer import TextTokenizerTransformForTransfer
from cosmos_framework.data.generator.multiview.camera_attributes import CAMERA_VIEW_MAPPING
from cosmos_framework.data.generator.multiview.local_pipeline import make_auto_augmentations
from cosmos_framework.data.generator.multiview.multiview_data_source import MADS_V1_400_ALL_CAPTION_KEY_MAPPING
from cosmos_framework.data.generator.multiview.transforms import (
    ExtractFramesAndCaptions,
    JointCamLidarToTrainingFormat,
)
from cosmos_framework.utils.lazy_config import instantiate


def _joint_cam_lidar_sample(*, camera_frames: int = 13, sweeps: int = 5) -> dict:
    """A camera clip on the 30 fps clock plus the sweeps that fall inside its window."""
    camera_frame_ids = list(range(camera_frames))
    sweep_ids = list(range(0, 3 * sweeps, 3))
    return {
        "__key__": "clip-0",
        "ai_caption": "front wide caption",
        "video": torch.ones((3, camera_frames, 720, 1280)),  # [C,T,H,W]
        "control_input": torch.zeros((3, camera_frames, 720, 1280)),  # [C,T,H,W]
        "hdmap_range": torch.zeros((1, sweeps, 128, 1024)),  # [C,T,H,W]
        "lidar_range": torch.ones((1, sweeps, 128, 1024)),  # [C,T,H,W]
        "frame_indices": torch.tensor(camera_frame_ids, dtype=torch.int64),
        "hdmap_frame_indices": torch.tensor(sweep_ids, dtype=torch.int64),
        "lidar_frame_indices": torch.tensor(sweep_ids, dtype=torch.int64),
        "fps": torch.tensor(30.0),
    }


@pytest.mark.parametrize("start_frame", [0, 400, 401])
@pytest.mark.parametrize(("camera_prefix", "lidar_prefix"), [(0, 0), (1, 1)])
def test_short_joint_window_preserves_both_sensors_and_paired_prefixes(
    start_frame: int, camera_prefix: int, lidar_prefix: int
) -> None:
    first_sweep = 3 * ((start_frame + 2) // 3)
    sweep_ids = torch.arange(first_sweep, start_frame + 65, 3)  # [T_lidar]
    camera_ids = torch.arange(start_frame, start_frame + 65)  # [65]
    camera = camera_ids.reshape(1, 65, 1, 1).repeat(3, 2, 4, 4).float()  # [C,V*T,H,W]
    lidar = sweep_ids.reshape(1, -1, 1, 1).repeat(3, 1, 4, 4).float()  # [C,T,H,W]
    sample = {
        "__key__": "short-joint",
        "video": camera,
        "control_input": camera.clone(),  # [C,V*T,H,W]
        "lidar_range": lidar,
        "hdmap_range": lidar.clone(),  # [C,T,H,W]
        "frame_indices": camera_ids,
        "lidar_frame_indices": sweep_ids,
        "hdmap_frame_indices": sweep_ids,
        "fps": torch.tensor(30.0),  # []
        "sample_n_views": torch.tensor(2),  # []
        "view_indices": torch.arange(2).repeat_interleave(65),  # [V*T]
        "view_indices_selection": torch.arange(2),  # [V]
    }
    result = JointCamLidarToTrainingFormat(
        camera_resolution_hw=(4, 4),
        max_camera_frames=65,
        fixed_camera_frames=61,
        conditioning_config={camera_prefix: 1.0},
    )(sample)
    assert result is not None
    for clip in result["video"]:
        assert clip.shape == (3, 122, 4, 4)
        views = clip.reshape(3, 2, 61, 4, 4)  # [C,V,T,H,W]
        assert views[0, :, 0, 0, 0].tolist() == [first_sweep, first_sweep]
        assert views[0, :, -1, 0, 0].tolist() == [first_sweep + 60, first_sweep + 60]
    assert all(clip.shape[1] == 21 for clip in result["lidar"])
    assert result["lidar_frame_indices"].tolist() == list(range(first_sweep, first_sweep + 61, 3))
    assert result["sequence_plan"].condition_frame_indexes_vision == list(range(camera_prefix))
    assert result["sequence_plan"].condition_frame_indexes_lidar == list(range(lidar_prefix))


def test_fixed_joint_window_rejects_short_coverage() -> None:
    sample = _joint_cam_lidar_sample()
    assert JointCamLidarToTrainingFormat(fixed_camera_frames=61)(sample) is None


def test_short_joint_options_reach_the_augmentors() -> None:
    config = dict(
        camera_keys=("camera_front_wide_120fov",),
        joint_cam_lidar_transfer=True,
        first_caption_paragraph_only=True,
        num_video_frames=65,
        joint_cam_lidar_fixed_camera_frames=61,
        conditioning_config={0: 0.5, 1: 0.5},
    )
    augmentations = make_auto_augmentations(config)
    extractor = next(item for item in augmentations if isinstance(item, ExtractFramesAndCaptions))
    assert extractor.first_caption_paragraph_only
    assert not any(isinstance(item, FirstCaptionParagraph) for item in augmentations)
    formatter = next(item for item in augmentations if isinstance(item, JointCamLidarToTrainingFormat))
    assert formatter.fixed_camera_frames == 61
    assert formatter.normalized_conditioning_config == {0: 0.5, 1: 0.5}


class _TokenizerStub:
    pass


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("resolution", ["480", "720"])
def test_sim_auto_profiles_keep_their_policy_in_the_shared_pipeline(stage: int, resolution: str) -> None:
    config = {key: instantiate(value) for key, value in _dataset(stage, resolution).transform_overrides.items()}
    config["resolution_hw"] = (480, 832) if resolution == "480" else (720, 1280)
    config["tokenizer_config"] = {"_target_": _TokenizerStub}
    pipeline = make_auto_augmentations(config)
    extractor = next(item for item in pipeline if isinstance(item, ExtractFramesAndCaptions))
    formatter = next(item for item in pipeline if isinstance(item, JointCamLidarToTrainingFormat))
    tokenizer = next(item for item in pipeline if isinstance(item, TextTokenizerTransformForTransfer))
    assert extractor.camera_view_mapping == CAMERA_VIEW_MAPPING
    assert extractor.camera_caption_key_mapping == MADS_V1_400_ALL_CAPTION_KEY_MAPPING
    assert extractor.separate_view_text_tokenization and extractor.add_view_prefix_to_caption
    assert extractor.per_view_caption_dropout_rate == 0.2
    assert formatter.exact_camera_frames == (93 if stage == 3 else None)
    assert formatter.normalized_conditioning_config == pytest.approx({0: 0.7, 1: 0.2, 2: 0.1})
    assert formatter.lidar_dropout_rate == 0.1
    assert tokenizer.tokenize_separately and tokenizer.cfg_dropout_rate == 0.2
