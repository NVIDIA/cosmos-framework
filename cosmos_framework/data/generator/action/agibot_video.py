# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

import torch
import torchvision.transforms.functional as transforms_F

AGIBOT_NATIVE_VIDEO_SIZE_HW: tuple[int, int] = (480, 640)


AGIBOT_RESIZED_VIDEO_SIZE_HW: tuple[int, int] = (360, 640)


def resize_to_640x360(
    video_tchw: torch.Tensor,  # [T,C,H,W]
) -> torch.Tensor:  # returns [T,C,360,640]
    """Resize a video tensor to 640x360."""

    resized = transforms_F.resize(
        video_tchw,
        list(AGIBOT_RESIZED_VIDEO_SIZE_HW),
        interpolation=transforms_F.InterpolationMode.BICUBIC,
        antialias=True,
    )  # [T,C,360,640]
    if video_tchw.is_floating_point():
        resized = resized.clamp_(0.0, 1.0)
    return resized
