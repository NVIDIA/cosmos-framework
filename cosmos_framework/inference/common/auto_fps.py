# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Repository-owned camera FPS policy for Auto inference; no checkpoint edits required."""

from typing import Any

from cosmos_framework.inference.common.multiview_captions import JOINT_CAM_LIDAR_TASK, exported_caption_task


def auto_camera_fps(metadata: Any, *, repository: str | None = None) -> int | None:
    """Identify Auto by its existing task metadata, or an exact official HF repository ID.

    Both released Auto snapshots record the joint task, including when used for RGB-only
    generation. The plain multiview task covers camera-only Auto exports. A local directory
    name, view count, LiDAR tokenizer or base_fps alone does not identify an Auto checkpoint.
    This policy lives in inference code, not in the immutable checkpoint's config.json.
    """
    if exported_caption_task(metadata) in {JOINT_CAM_LIDAR_TASK, "av_multiview_transfer"} or repository in {
        "nvidia/Cosmos3-Nano-Transfer-Auto",
        "nvidia/Cosmos3-Nano-Sim-Auto",
    }:
        return 30
    return None
