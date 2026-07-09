# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import numpy as np
import torch
from PIL import Image


def tensor_to_pil_images(video_tensor: torch.Tensor) -> list[Image.Image]:
    """Convert a video tensor of shape (C, T, H, W) or (T, C, H, W) into a list of PIL images.

    Args:
        video_tensor: Video tensor with shape (C, T, H, W) or (T, C, H, W).

    Returns:
        One PIL image per frame.
    """
    # (C, T, H, W) -> (T, C, H, W)
    if video_tensor.shape[0] == 3 and video_tensor.shape[1] > 3:
        video_tensor = video_tensor.permute(1, 0, 2, 3)

    # (T, C, H, W) -> (T, H, W, C) and detach to CPU numpy.
    video_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy()

    # PIL expects uint8 with values in [0, 255]; rescale floats accordingly.
    if video_np.dtype == np.float32 or video_np.dtype == np.float64:
        if video_np.max() <= 1.0:
            video_np = (video_np * 255).astype(np.uint8)
        else:
            video_np = video_np.astype(np.uint8)

    return [Image.fromarray(frame) for frame in video_np]
