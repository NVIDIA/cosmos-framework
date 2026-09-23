# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Video metadata must survive wrapper parsing in conversation order."""

import pytest
from PIL import Image

from cosmos_framework.data.generator.processors.base import maybe_parse_video_content

pytestmark = [pytest.mark.L1, pytest.mark.CPU]


def test_mixed_source_and_manual_video_metadata() -> None:
    frames = [Image.new("RGB", (32, 32))] * 3
    metadata = {"fps": 30.0, "total_num_frames": 300, "frames_indices": [30, 40, 50]}
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames, "fps": 4.0, "video_metadata": metadata},
                {"type": "video", "video": frames, "fps": 2.0},
            ],
        }
    ]
    count, fps, totals, indices = maybe_parse_video_content(messages)
    assert (count, fps, totals, indices) == (2, [30.0, 2.0], [300, 3], [[30, 40, 50], [0, 1, 2]])
    indices[0].append(50)
    assert metadata["frames_indices"] == [30, 40, 50]


def test_explicit_invalid_metadata_cannot_fall_back_to_fps() -> None:
    with pytest.raises(ValueError):
        maybe_parse_video_content(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": [Image.new("RGB", (32, 32))], "fps": 4.0, "video_metadata": None}
                    ],
                }
            ]
        )
