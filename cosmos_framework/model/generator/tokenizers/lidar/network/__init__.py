# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Checkpoint-compatible LiDAR TransformerVAE network."""

from __future__ import annotations

from typing import Any

__all__ = ["TransformerVAE"]


def __getattr__(name: str) -> Any:
    if name != "TransformerVAE":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from cosmos_framework.model.generator.tokenizers.lidar.network.transformer_vae import TransformerVAE

    return TransformerVAE
