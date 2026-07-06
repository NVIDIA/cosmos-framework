# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.model.generator.tokenizers.uniae.frame_math import (
    align_uniae_num_video_frames,
    ceil_uniae_num_video_frames,
)


def test_ceil_uniae_num_video_frames_preserves_valid_partial_tail() -> None:
    assert (
        ceil_uniae_num_video_frames(
            17,
            {"480": 16},
            pad_frames=1,
            temporal_compression_factor=4,
            resolution="480",
        )
        == 17
    )


def test_ceil_uniae_num_video_frames_uses_next_valid_partial_tail() -> None:
    assert (
        align_uniae_num_video_frames(
            24,
            {"480": 16},
            pad_frames=1,
            temporal_compression_factor=4,
            resolution="480",
        )
        == 21
    )
    assert (
        ceil_uniae_num_video_frames(
            24,
            {"480": 16},
            pad_frames=1,
            temporal_compression_factor=4,
            resolution="480",
        )
        == 25
    )
