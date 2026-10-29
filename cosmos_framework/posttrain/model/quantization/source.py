# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Model-side source verification; CPU preparation imports the lightweight utils module."""

from cosmos_framework.posttrain.utils.adapter_source import (
    SOURCE_MANIFEST,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    prepare_adapter_snapshot,
    prepare_sft_adapter_source,
    validate_adapter_source,
    verify_adapter_snapshot,
)

__all__ = [
    "SOURCE_MANIFEST",
    "SOURCE_REPOSITORY",
    "SOURCE_REVISION",
    "prepare_adapter_snapshot",
    "prepare_sft_adapter_source",
    "validate_adapter_source",
    "verify_adapter_snapshot",
]
