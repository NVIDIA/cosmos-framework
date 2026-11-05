# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Restore directly resized Transfer output to its source pixel geometry."""

import torch
import torch.nn.functional as F


def restore_transfer_output_size(
    video: torch.Tensor,  # [C,T,H_model,W_panels], floating-point pixels in [0,1]
    *,
    original_hw: tuple[int, int],
    model_hw: tuple[int, int],
) -> torch.Tensor:  # [C,T,H_source,W_source*panels]
    """Resize each RGB/preview panel separately, retaining all frames and the complete image.

    Interpolating the concatenated preview would blend pixels across panel boundaries.
    Processing a few frames at once bounds the interpolation workspace for large exports.
    This is the inverse geometry of direct resizing, not a way to undo center cropping.
    """
    if video.ndim != 4 or not video.is_floating_point():
        raise ValueError("Transfer output must be a floating-point [C,T,H,W] tensor.")
    source_h, source_w = original_hw
    model_h, model_w = model_hw
    if min(source_h, source_w, model_h, model_w) <= 0:
        raise ValueError("Transfer source and model dimensions must be positive.")
    if video.shape[2] != model_h or video.shape[3] % model_w:
        raise ValueError("Transfer output must contain complete panels at the model resolution.")
    if original_hw == model_hw:
        return video  # [C,T,H_model,W_panels]

    panel_count = video.shape[3] // model_w
    restored = video.new_empty((video.shape[0], video.shape[1], source_h, source_w * panel_count))  # [C,T,Hs,Ws*P]
    for panel_index in range(panel_count):
        for start in range(0, video.shape[1], 8):
            stop = min(start + 8, video.shape[1])
            frames = video[:, start:stop, :, panel_index * model_w : (panel_index + 1) * model_w]  # [C,Tc,Hm,Wm]
            frames = frames.permute(1, 0, 2, 3).float()  # [Tc,C,Hm,Wm]
            resized = F.interpolate(  # [Tc,C,Hs,Ws]
                frames, size=original_hw, mode="bicubic", align_corners=False, antialias=True
            ).clamp_(0, 1)
            restored[:, start:stop, :, panel_index * source_w : (panel_index + 1) * source_w] = resized.permute(
                1, 0, 2, 3
            )  # [C,Tc,Hs,Ws]
    return restored  # [C,T,Hs,Ws*P]
