# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Image resizing helpers shared by Cosmos3 image-edit generation paths."""

from __future__ import annotations

import math

from PIL import Image

DEFAULT_MAX_PIXELS = 1024 * 1024
DEFAULT_PADDING_CONSTANT = 32
_RESOLUTION_768_SHAPES: tuple[tuple[int, int], ...] = (
    (1024, 1024),
    (1184, 880),
    (880, 1184),
    (1360, 768),
    (768, 1360),
)
_SUPPORTED_VISION_RESOLUTION_TIERS = frozenset({"256", "480", "720"})


def get_vision_data_resolution(spatial_shape: tuple[int, int]) -> str:
    """Determine the resolution string from spatial dimensions."""
    if spatial_shape in _RESOLUTION_768_SHAPES:
        return "768"

    min_dim = min(spatial_shape[0], spatial_shape[1])
    if min_dim <= 256:
        return "256"
    if min_dim <= 640:
        return "480"
    if min_dim <= 960:
        return "720"
    if min_dim <= 2048:
        return "768"
    raise ValueError(f"Unsupported resolution: {spatial_shape}")


def choose_aspect_nearest_shape(
    original_size: tuple[int, int],
    max_size: int,
    padding_constant: int,
) -> tuple[int, int]:
    """Choose the largest supported no-upscale shape with the closest aspect ratio."""
    original_width, original_height = original_size
    if original_width < 1 or original_height < 1:
        raise ValueError(f"Invalid source image size: {original_size}.")
    if max_size < 1 or padding_constant < 1:
        raise ValueError("max_size and padding_constant must be positive.")

    capped_long_side = min(max_size, max(original_width, original_height))
    legal_long_side = max(padding_constant, (capped_long_side // padding_constant) * padding_constant)
    target_aspect = original_width / original_height
    best: tuple[tuple[float, int], int, int] | None = None

    for width in range(padding_constant, legal_long_side + 1, padding_constant):
        for height in range(padding_constant, legal_long_side + 1, padding_constant):
            try:
                resolution_tier = get_vision_data_resolution((height, width))
            except ValueError:
                continue
            if resolution_tier not in _SUPPORTED_VISION_RESOLUTION_TIERS:
                continue
            score = (abs((width / height) / target_aspect - 1.0), -(width * height))
            if best is None or score < best[0]:
                best = (score, width, height)

    if best is None:
        raise ValueError(
            f"No I4-supported shape for source={original_size}, max_size={max_size}, "
            f"padding_constant={padding_constant}."
        )
    return best[1], best[2]


def resize_aspect_nearest_no_upscale(image: Image.Image, max_size: int, padding_constant: int) -> Image.Image:
    """Apply the Vision aspect-preserving, no-upscale input policy."""
    target_size = choose_aspect_nearest_shape(image.size, max_size, padding_constant)
    if target_size == image.size:
        return image
    return image.resize(target_size, resample=Image.Resampling.LANCZOS)


def get_max_pixels_resized_size(
    width: int,
    height: int,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    padding_constant: int = DEFAULT_PADDING_CONSTANT,
) -> tuple[int, int]:
    """Return an aspect-preserving size capped by max pixels and rounded down."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}.")
    if padding_constant <= 0:
        raise ValueError(f"padding_constant must be positive, got {padding_constant}.")
    if max_pixels < padding_constant * padding_constant:
        raise ValueError(
            f"max_pixels={max_pixels} is too small for padding_constant={padding_constant}; "
            f"minimum is {padding_constant * padding_constant}."
        )

    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    resized_width = int(width * scale)
    resized_height = int(height * scale)

    resized_width = (resized_width // padding_constant) * padding_constant
    resized_height = (resized_height // padding_constant) * padding_constant
    resized_width = max(resized_width, padding_constant)
    resized_height = max(resized_height, padding_constant)

    while resized_width * resized_height > max_pixels:
        if resized_width >= resized_height and resized_width > padding_constant:
            resized_width -= padding_constant
        elif resized_height > padding_constant:
            resized_height -= padding_constant
        else:
            break

    return resized_width, resized_height


def resize_pil_image(
    image: Image.Image,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    padding_constant: int = DEFAULT_PADDING_CONSTANT,
) -> Image.Image:
    """Resize a PIL image to a max-pixels budget while preserving aspect ratio."""
    resized_size = get_max_pixels_resized_size(
        width=image.size[0],
        height=image.size[1],
        max_pixels=max_pixels,
        padding_constant=padding_constant,
    )
    if resized_size == image.size:
        return image.copy()
    return image.resize(resized_size, Image.LANCZOS)
