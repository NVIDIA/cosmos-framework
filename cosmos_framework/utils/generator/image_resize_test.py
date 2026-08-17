# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for Cosmos3 max-pixels image resize helpers."""

from __future__ import annotations

import pytest
from PIL import Image

from cosmos_framework.utils.generator.image_resize import (
    choose_aspect_nearest_shape,
    get_max_pixels_resized_size,
    resize_aspect_nearest_no_upscale,
    resize_pil_image,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_get_max_pixels_resized_size_downscales_wide_image() -> None:
    assert get_max_pixels_resized_size(1920, 1080, max_pixels=1024 * 1024, padding_constant=32) == (1344, 768)


def test_get_max_pixels_resized_size_preserves_square_at_budget() -> None:
    assert get_max_pixels_resized_size(1024, 1024, max_pixels=1024 * 1024, padding_constant=32) == (1024, 1024)


def test_get_max_pixels_resized_size_does_not_upscale_small_image() -> None:
    assert get_max_pixels_resized_size(640, 480, max_pixels=1024 * 1024, padding_constant=32) == (640, 480)


def test_get_max_pixels_resized_size_rounds_down_under_budget() -> None:
    assert get_max_pixels_resized_size(641, 481, max_pixels=1024 * 1024, padding_constant=32) == (640, 480)


def test_get_max_pixels_resized_size_rejects_impossible_budget() -> None:
    with pytest.raises(ValueError, match="too small"):
        get_max_pixels_resized_size(128, 128, max_pixels=1023, padding_constant=32)


def test_resize_pil_image_uses_computed_size() -> None:
    image = Image.new("RGB", (1920, 1080), (10, 20, 30))

    resized = resize_pil_image(image, max_pixels=1024 * 1024, padding_constant=32)

    assert resized.size == (1344, 768)


@pytest.mark.parametrize(
    ("source_size", "expected_size"),
    [
        ((1920, 1080), (768, 432)),
        ((1024, 1024), (960, 960)),
        ((640, 480), (640, 480)),
    ],
)
def test_choose_aspect_nearest_shape_matches_the_vision_protocol(
    source_size: tuple[int, int],
    expected_size: tuple[int, int],
) -> None:
    assert choose_aspect_nearest_shape(source_size, max_size=960, padding_constant=16) == expected_size


def test_resize_aspect_nearest_no_upscale_uses_the_selected_shape() -> None:
    image = Image.new("RGB", (1920, 1080), (10, 20, 30))

    resized = resize_aspect_nearest_no_upscale(image, max_size=960, padding_constant=16)

    assert resized.size == (768, 432)
