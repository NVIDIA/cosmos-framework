# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os

from cosmos_auto.wm.config import CosmosAVConfig
from cosmos_auto.wm.wds_utils import get_sample


def get_render_frame_ids(clip_id: str, cfg: CosmosAVConfig) -> list[int]:
    """Return absolute pose-frame IDs sampled for rendering."""
    target_render_fps = cfg.output_frame_spec.target_render_fps
    target_chunk_frame = cfg.output_frame_spec.target_chunk_frame
    overlap_frame = cfg.output_frame_spec.overlap_frame
    max_chunk = cfg.output_frame_spec.max_chunk

    input_pose_fps = cfg.input_fps_spec.input_pose_fps
    pose_file = os.path.join(cfg.input_root, "pose", f"{clip_id}.tar")
    pose_data = get_sample(pose_file)
    camera_key = f"pose.{cfg.camera_spec.camera_list[0]}.npy"
    frame_num = sum(camera_key in key for key in pose_data)
    render_frame_ids = list(range(0, frame_num, input_pose_fps // target_render_fps))

    if max_chunk > 0:
        expected_frame_num = target_chunk_frame + (target_chunk_frame - overlap_frame) * (max_chunk - 1)
        render_frame_ids = render_frame_ids[:expected_frame_num]

    return render_frame_ids


def get_output_chunk_starts(render_frame_count: int, cfg: CosmosAVConfig) -> list[int]:
    """Return chunk start offsets using the renderer's exact output policy."""
    target_chunk_frame = cfg.output_frame_spec.target_chunk_frame
    overlap_frame = cfg.output_frame_spec.overlap_frame
    chunk_stride = target_chunk_frame - overlap_frame
    if chunk_stride <= 0:
        raise ValueError("output_frame_spec.overlap_frame must be smaller than target_chunk_frame")

    chunk_starts = list(range(0, render_frame_count - target_chunk_frame + 1, chunk_stride))
    if not chunk_starts and cfg.allow_short_final_chunk and render_frame_count > 0:
        chunk_starts = [0]
    return chunk_starts


def get_expected_chunk_count(clip_id: str, cfg: CosmosAVConfig) -> int:
    """Return the number of chunks that rendering will actually emit for a clip."""
    return len(get_output_chunk_starts(len(get_render_frame_ids(clip_id, cfg)), cfg))
