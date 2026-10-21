# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared resolution-keyed constants for interactive configs."""

from collections.abc import Iterable, Mapping
from typing import TypeVar

MAX_480P_RESOLUTIONS: tuple[str, ...] = ("256", "480")
RECTIFIED_FLOW_SHIFT: dict[str, int] = {"256": 3, "480": 5, "720": 10}
TOKENIZER_ENCODE_CHUNK_FRAMES: dict[str, int] = {"256": 68, "480": 24, "720": 8}

_T = TypeVar("_T")


def select_resolution_map(values: Mapping[str, _T], resolutions: Iterable[str]) -> dict[str, _T]:
    """Return a resolution-keyed subset while preserving the requested order."""
    return {resolution: values[resolution] for resolution in resolutions}
