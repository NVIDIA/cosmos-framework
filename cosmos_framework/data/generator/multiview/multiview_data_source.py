# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Sim-Auto caption keys layered over the shared physical camera schema."""

from typing import Final

from cosmos_framework.data.generator.multiview.schema import MADS_CAMERAS as MADS_CAMERAS
from cosmos_framework.data.generator.multiview.schema import MADS_V1_CONTROL_KEY_MAPPING as MADS_V1_CONTROL_KEY_MAPPING

MADS_V1_400_ALL_CAPTION_KEY_MAPPING: Final[dict[str, str]] = {
    camera: f"captions_400_all_qwen35-27b-caption-v2-v1_{camera}" for camera in MADS_CAMERAS
}
