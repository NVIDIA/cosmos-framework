# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prepare frames the Framework has already decoded for the Qwen processor."""

from __future__ import annotations

from typing import Any

from cosmos_framework.inference.reasoner.video_pixel_bounds import normalize_video_pixel_bounds


def is_decoded_frames(video: Any) -> bool:
    """Return whether a video content value is a non-empty sequence of PIL frames."""
    from PIL import Image

    return (
        isinstance(video, (list, tuple)) and len(video) > 0 and all(isinstance(frame, Image.Image) for frame in video)
    )


def fetch_decoded_video(
    element: dict[str, Any],
    image_patch_size: int,
    vision_process: Any,
) -> tuple[Any, dict[str, Any]]:
    """Resize decoded frames once, on the model's grid, with the resize the video processor uses.

    The list-of-frames branch of ``qwen_vl_utils.fetch_video`` cannot be used for
    these frames. It hands the merged factor (patch size times merge size) to
    ``fetch_image`` as the patch size, where it is multiplied by the merge size a
    second time, so every frame is first rounded to twice the model's grid (64
    pixels for patch 16) before the video-level resize. A 480x288 clip is then
    evaluated at 512x256 although training, which resizes on the model's own
    32-pixel grid, sees 480x288. That branch also ignores the ``fps`` of the
    content item and stamps the frames at 2 fps.

    This function computes the size that ``fetch_video`` computes after one of
    its own readers returns, on the same grid and with the same pixel bounds,
    and records the sampled frame rate that the Framework decoder measured. The
    resize itself is the uint8 bicubic resize of the Hugging Face video
    processor, so an evaluation whose per-frame ``max_pixels`` times the frame
    count equals the training pixel budget feeds the model the pixels it was
    trained on. The frames are not padded: the video processor completes an odd
    count itself, which is also what training relies on.
    """
    import numpy as np
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as transforms_functional

    frames = element["video"]
    if not is_decoded_frames(frames):
        raise ValueError("fetch_decoded_video requires a non-empty sequence of PIL frames")
    sample_fps = element.get("fps")
    if sample_fps is None or float(sample_fps) <= 0:
        raise ValueError(f"Framework-decoded video frames require a positive fps, got {sample_fps}")
    sample_fps = float(sample_fps)

    element = normalize_video_pixel_bounds(element, image_patch_size, vision_process)
    video = torch.stack(
        [torch.from_numpy(np.array(vision_process.to_rgb(frame))).permute(2, 0, 1) for frame in frames]
    )  # [T,3,H,W]
    frame_count, _, height, width = video.shape

    factor = int(image_patch_size) * int(vision_process.SPATIAL_MERGE_SIZE)
    frame_factor = int(vision_process.FRAME_FACTOR)
    budget_frames = vision_process.ceil_by_factor(frame_count, frame_factor)
    min_pixels = element.get("min_pixels", vision_process.VIDEO_MIN_TOKEN_NUM * factor * factor)
    total_pixels = element.get("total_pixels", vision_process.MODEL_SEQ_LEN * factor * factor * 0.9)
    max_pixels = max(
        min(vision_process.VIDEO_MAX_TOKEN_NUM * factor * factor, total_pixels / budget_frames * frame_factor),
        int(min_pixels * 1.05),
    )
    max_pixels = min(element.get("max_pixels", max_pixels), max_pixels)
    resized_height, resized_width = vision_process.smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    video = transforms_functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()  # [T,3,H',W']
    metadata = {
        "fps": sample_fps,
        "frames_indices": list(range(frame_count)),
        "total_num_frames": frame_count,
    }
    return video, metadata


def process_vision_info(
    conversations: list[dict[str, Any]] | list[list[dict[str, Any]]],
    image_patch_size: int,
) -> tuple[list[Any] | None, list[Any] | None, dict[str, Any]]:
    """Mirror ``qwen_vl_utils.process_vision_info`` with metadata, routing decoded frames past its frame-list branch.

    Images and videos given as paths still go through qwen-vl-utils, including
    any reader the Framework registered there. The return value has the shape
    of ``process_vision_info(..., return_video_kwargs=True, return_video_metadata=True)``.
    """
    import qwen_vl_utils.vision_process as vision_process

    image_inputs: list[Any] = []
    video_inputs: list[Any] = []
    for element in vision_process.extract_vision_info(conversations):
        if "image" in element or "image_url" in element:
            image_inputs.append(vision_process.fetch_image(element, image_patch_size=image_patch_size))
        elif "video" in element:
            if is_decoded_frames(element["video"]):
                video_inputs.append(fetch_decoded_video(element, image_patch_size, vision_process))
            else:
                video_input, _ = vision_process.fetch_video(
                    element,
                    return_video_sample_fps=True,
                    image_patch_size=image_patch_size,
                    return_video_metadata=True,
                )
                video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    return image_inputs or None, video_inputs or None, {"do_sample_frames": False}
