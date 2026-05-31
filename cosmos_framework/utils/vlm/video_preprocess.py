# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import numpy as np
from PIL import Image


def tensor_to_pil_images(video_tensor):
    """
    Convert a video tensor of shape (C, T, H, W) or (T, C, H, W) to a list of PIL images.

    Args:
        video_tensor (torch.Tensor): Video tensor with shape (C, T, H, W) or (T, C, H, W)

    Returns:
        list[PIL.Image.Image]: List of PIL images
    """
    # Check tensor shape and convert if needed
    if video_tensor.shape[0] == 3 and video_tensor.shape[1] > 3:  # (C, T, H, W)
        # Convert to (T, C, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # [T,C,H,W]

    # Convert to numpy array with shape (T, H, W, C)
    video_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy()  # [T,H,W,C]

    # Ensure values are in the right range for PIL (0-255, uint8)
    if video_np.dtype == np.float32 or video_np.dtype == np.float64:
        if video_np.max() <= 1.0:
            video_np = (video_np * 255).astype(np.uint8)
        else:
            video_np = video_np.astype(np.uint8)

    # Convert each frame to a PIL image
    pil_images = [Image.fromarray(frame) for frame in video_np]

    return pil_images
