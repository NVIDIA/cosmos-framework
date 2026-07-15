# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Image resizing helpers shared by Cosmos3 image-edit generation paths."""

from __future__ import annotations

import math

from PIL import Image

DEFAULT_MAX_PIXELS = 1024 * 1024
DEFAULT_PADDING_CONSTANT = 32


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
