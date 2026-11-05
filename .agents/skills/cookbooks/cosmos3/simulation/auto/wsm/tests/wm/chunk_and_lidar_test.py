from unittest.mock import patch

import numpy as np
from cosmos_auto.wm import resume_utils
from cosmos_auto.wm.chunk_utils import get_expected_chunk_count, get_output_chunk_starts
from cosmos_auto.wm.cli import get_nearest_available_frame_id
from cosmos_auto.wm.config import CosmosAVConfig


def make_cfg() -> CosmosAVConfig:
    cfg = CosmosAVConfig(input_root="/input", output_root="/output")
    cfg.camera_spec.camera_list = ["front"]
    cfg.output_frame_spec.target_render_fps = 10
    cfg.output_frame_spec.target_chunk_frame = 100
    cfg.output_frame_spec.overlap_frame = 45
    cfg.output_frame_spec.max_chunk = 3
    return cfg


def make_pose_sample(frame_count: int) -> dict[str, np.ndarray]:
    return {f"{frame_id:06d}.pose.front.npy": np.eye(4) for frame_id in range(frame_count)}


def test_chunk_plan_allows_one_short_final_chunk():
    cfg = make_cfg()
    cfg.allow_short_final_chunk = True

    assert get_output_chunk_starts(50, cfg) == [0]


def test_resume_expected_chunks_follow_actual_clip_length():
    cfg = make_cfg()
    with patch("cosmos_auto.wm.chunk_utils.get_sample", return_value=make_pose_sample(300)):
        assert get_expected_chunk_count("clip", cfg) == 1
    with patch("cosmos_auto.wm.chunk_utils.get_sample", return_value=make_pose_sample(630)):
        assert get_expected_chunk_count("clip", cfg) == 3


def test_lidar_nearest_frame_uses_absolute_timeline_at_clip_end():
    lidar_frame_ids = list(range(0, 300, 3))

    assert get_nearest_available_frame_id(297, lidar_frame_ids) == 297
    assert get_nearest_available_frame_id(299, lidar_frame_ids) == 297


def test_resume_accepts_the_chunks_actually_emitted_for_short_clip():
    cfg = make_cfg()
    cfg.allow_short_final_chunk = True
    with patch("cosmos_auto.wm.chunk_utils.get_sample", return_value=make_pose_sample(150)):
        remaining = resume_utils.find_remaining_clip_list(cfg, ["clip"], ["clip_0"])

    assert remaining == []
