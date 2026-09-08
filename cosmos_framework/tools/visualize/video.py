# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import IO, Any, Union

import numpy as np
import torch
from einops import rearrange
from PIL import Image as PILImage
from torch import Tensor

from cosmos_framework.utils.easy_io import easy_io


def save_video(grid, video_name, fps=30):
    # Remove ffmpegcv for license issue
    # Use imageio instead
    import imageio

    grid = (grid * 255).astype(np.uint8)
    grid = np.transpose(grid, (1, 2, 3, 0))
    imageio.mimsave(
        video_name,
        list(grid),
        format="mp4",
        fps=float(fps),
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
    )


def _to_writer_uint8(sample: Tensor, pattern: str) -> np.ndarray:
    """Lay a display tensor out as the contiguous uint8 array the writers consume.

    A uint8 sample already carries display levels, so it is only transposed. The float
    round trip it used to take -- up-converted to [0, 1] here, multiplied by 255 again a
    few lines later -- cost two copies of the whole clip, which is the difference between
    hundreds of megabytes and tens of gigabytes for a tiled multiview grid. That round
    trip also lost a level: 1/127.5 is not exact in binary, so truncating the reconstructed
    float darkened each pixel by up to one step.
    """
    if sample.dtype == torch.uint8:
        return np.ascontiguousarray(rearrange(sample.cpu().numpy(), pattern))
    return rearrange((sample.cpu().float().numpy() * 255), pattern).astype(np.uint8)


def save_img_or_video(
    sample: Tensor,  # [C,T,H,W] float in [0,1], or uint8 in [0,255]
    save_fp_wo_ext: Union[str, IO[Any]],
    fps: int = 24,
    quality=None,
    ffmpeg_params=None,
    **kwargs,
) -> None:
    """
    Save a tensor as an image or video file based on shape

        Args:
        sample (Tensor): Input tensor with shape (C, T, H, W), floating point in [0, 1]
            range or uint8 in [0, 255].
        save_fp_wo_ext (Union[str, IO[Any]]): File path without extension or file-like object.
        fps (int): Frames per second for video. Default is 24.
    """
    assert sample.ndim == 4, "Only support 4D tensor"
    assert isinstance(save_fp_wo_ext, str) or hasattr(save_fp_wo_ext, "write"), (
        "save_fp_wo_ext must be a string or file-like object"
    )

    if torch.is_floating_point(sample):
        sample = sample.clamp(0, 1)
    else:
        assert sample.dtype == torch.uint8, "Only support uint8 tensor"

    if ffmpeg_params is not None:
        kwargs["ffmpeg_params"] = ffmpeg_params

    if sample.shape[1] == 1:
        save_obj = PILImage.fromarray(
            _to_writer_uint8(sample, "c 1 h w -> h w c"),
            mode="RGB",
        )
        ext = ".jpg" if isinstance(save_fp_wo_ext, str) else ""
        easy_io.dump(
            save_obj,
            f"{save_fp_wo_ext}{ext}" if isinstance(save_fp_wo_ext, str) else save_fp_wo_ext,
            file_format="jpg",
            format="JPEG",
            quality=85 if quality is None else quality,
            **kwargs,
        )
    else:
        if quality is not None:
            kwargs["quality"] = quality
        save_obj = _to_writer_uint8(sample, "c t h w -> t h w c")
        ext = ".mp4" if isinstance(save_fp_wo_ext, str) else ""
        easy_io.dump(
            save_obj,
            f"{save_fp_wo_ext}{ext}" if isinstance(save_fp_wo_ext, str) else save_fp_wo_ext,
            file_format="mp4",
            format="mp4",
            fps=fps,
            **kwargs,
        )
