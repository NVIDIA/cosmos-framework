# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for interleaved image resize augmentors."""

from __future__ import annotations

import pytest
from PIL import Image

from cosmos_framework.data.generator.augmentors.interleaved_image_transform import (
    InterleavedMediaResizeByMaxPixels,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_interleaved_media_resize_by_max_pixels_resizes_images_and_videos() -> None:
    transform = InterleavedMediaResizeByMaxPixels(max_pixels=1024 * 1024, padding_divisor=16)
    wide_image = Image.new("RGB", (1920, 1080), color="blue")
    small_image = Image.new("RGB", (640, 480), color="red")

    result = transform(
        {
            "media_list": {
                "image_0": wide_image,
                "image_1": small_image,
                "video_0": [wide_image.copy()],
            }
        }
    )

    assert result is not None
    resized_media = result["diffusion_media_list"]
    assert resized_media["image_0"].size == (1360, 768)
    assert resized_media["image_1"].size == (640, 480)
    assert resized_media["video_0"][0].size == (1360, 768)


def test_interleaved_media_resize_by_max_pixels_rejects_impossible_budget() -> None:
    transform = InterleavedMediaResizeByMaxPixels(max_pixels=255, padding_divisor=16)

    with pytest.raises(ValueError, match="too small"):
        transform({"media_list": {"image_0": Image.new("RGB", (64, 64))}})
