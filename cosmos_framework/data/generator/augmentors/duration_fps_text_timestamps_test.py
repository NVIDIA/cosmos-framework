# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Any

import pytest
import torch

from cosmos_framework.data.generator.augmentors.duration_fps_text_timestamps import DurationFPSTextTimeStamps


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("frames", "text_seconds", "json_seconds"), [(33, "1.1", "1.1"), (201, "6.7", "6.7"), (60, "2.0", "2")]
)
@pytest.mark.parametrize("explicit_per_view_count", [False, True])
def test_duration_metadata_retains_fractional_seconds(
    frames: int, text_seconds: str, json_seconds: str, explicit_per_view_count: bool
) -> None:
    video = torch.zeros((3, frames * (2 if explicit_per_view_count else 1), 2, 2), dtype=torch.uint8)  # [C,V*T,H,W]
    fps = torch.tensor(30.0)  # []
    data: dict[str, Any] = {
        "ai_caption": ["Front view.", {"caption": "Side view"}],
        "video": video,
        "conditioning_fps": fps,
    }
    args: dict[str, Any] = {"skip_on_error": False, "fractional_duration": True}
    if explicit_per_view_count:
        data["num_video_frames_per_view"] = torch.tensor(frames)  # []
        args["num_frames_key"] = "num_video_frames_per_view"

    assert DurationFPSTextTimeStamps(args=args)(data) is data
    assert data["ai_caption"] == [
        f"Front view. The video is {text_seconds} seconds long and is of 30 FPS.",
        {"caption": "Side view", "duration": f"{json_seconds}s", "fps": 30.0},
    ]
    assert data["video"] is video
    assert data["conditioning_fps"] is fps


@pytest.mark.L0
@pytest.mark.CPU
def test_duration_custom_template_and_disabled_augmentation() -> None:
    data = {"ai_caption": "Front view", "num_frames": 201, "conditioning_fps": 30}
    args = {
        "num_frames_key": "num_frames",
        "template": "Duration {duration:.2f}s at {fps:g} FPS.",
        "skip_on_error": False,
        "fractional_duration": True,
    }
    disabled = DurationFPSTextTimeStamps(args={**args, "enabled": False})
    assert disabled(data) is data
    assert data["ai_caption"] == "Front view"
    assert DurationFPSTextTimeStamps(args=args)(data) is data
    assert data["ai_caption"] == "Front view. Duration 6.70s at 30 FPS."


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("fractional_duration", [None, False, True])
def test_duration_preserves_default_integer_caption_contract(fractional_duration: bool | None) -> None:
    data = {"ai_caption": ["Action clip", {"caption": "Action clip"}], "num_frames": 23, "conditioning_fps": 8}
    args: dict[str, Any] = {"num_frames_key": "num_frames", "skip_on_error": False}
    if fractional_duration is not None:
        args["fractional_duration"] = fractional_duration

    assert DurationFPSTextTimeStamps(args=args)(data) is data
    text_duration, json_duration = ("2.9", "2.875") if fractional_duration else ("2.0", "2")
    assert data["ai_caption"] == [
        f"Action clip. The video is {text_duration} seconds long and is of 8 FPS.",
        {"caption": "Action clip", "duration": f"{json_duration}s", "fps": 8.0},
    ]
