# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""LiDAR range-map tokenizer."""

from __future__ import annotations

from typing import Any

__all__ = [
    "LidarTokenizerV1Interface",
]


def __getattr__(name: str) -> Any:
    if name == "LidarTokenizerV1Interface":
        from cosmos_framework.model.generator.tokenizers.lidar.lidar_tokenizer_v1 import (
            LidarTokenizerV1Interface,
        )

        return LidarTokenizerV1Interface
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
