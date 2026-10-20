# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Normalize explicit video pixel bounds against the active Qwen runtime."""

from __future__ import annotations

import threading
from typing import Any

_ATTESTED: set[tuple[int, int, int]] = set()
_ATTESTATION_LOCK = threading.Lock()


def normalize_video_pixel_bounds(
    element: dict[str, Any],
    image_patch_size: int,
    vision_process: Any,
) -> dict[str, Any]:
    """Preserve an explicit max while satisfying Qwen's pixel-bound invariant.

    Qwen derives a default per-frame minimum from its token constants and the
    caller's actual patch size.  An explicitly selected lower maximum is a
    valid throughput choice, but Qwen's default minimum would otherwise exceed
    it and fail before resizing.  Only that missing-minimum case is normalized:
    explicit bounds remain authoritative and contradictory explicit bounds
    fail loudly.
    """
    maximum = element.get("max_pixels")
    if maximum is None:
        return element
    maximum = int(maximum)
    if maximum < 1:
        raise ValueError(f"max_pixels must be positive, got {maximum}")
    # TOML-backed configuration can preserve numeric values as strings in the
    # message content. Qwen resumes with this same dictionary after the reader
    # returns, so canonicalize the shared object, not only the local value.
    element["max_pixels"] = maximum

    configured_minimum = element.get("min_pixels")
    if configured_minimum is not None:
        configured_minimum = int(configured_minimum)
        if configured_minimum < 1:
            raise ValueError(f"min_pixels must be positive, got {configured_minimum}")
        if maximum < configured_minimum:
            raise ValueError(
                f"explicit video pixel bounds are contradictory: max_pixels={maximum} < min_pixels={configured_minimum}"
            )
        element["min_pixels"] = configured_minimum
        return element

    merge_size = int(vision_process.SPATIAL_MERGE_SIZE)
    minimum_tokens = int(vision_process.VIDEO_MIN_TOKEN_NUM)
    factor = int(image_patch_size) * merge_size
    runtime_default_minimum = minimum_tokens * factor * factor
    if maximum >= runtime_default_minimum:
        return element

    # qwen-vl-utils calls the selected reader backend first and then resumes
    # inside its outer ``fetch_video`` with the same element dictionary.  The
    # normalized minimum must therefore be visible on that original object;
    # returning only a copy is too late and the outer resize would still use
    # its larger runtime default.
    element["min_pixels"] = maximum
    signature = (maximum, runtime_default_minimum, int(image_patch_size))
    if signature not in _ATTESTED:
        with _ATTESTATION_LOCK:
            if signature not in _ATTESTED:
                print(
                    "COSMOS_VIDEO_PIXEL_BOUNDS_NORMALIZED "
                    f"max_pixels={maximum} "
                    f"runtime_default_min_pixels={runtime_default_minimum} "
                    f"selected_min_pixels={maximum} "
                    f"image_patch_size={image_patch_size} "
                    "policy=preserve_explicit_max_pixels",
                    flush=True,
                )
                _ATTESTED.add(signature)
    return element
