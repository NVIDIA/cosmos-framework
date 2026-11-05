# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from collections.abc import Callable

import torch
import torchvision.transforms.v2 as T

DROID_IMAGE_CROP_RATIO = 0.95


def build_droid_image_augmentor(image_height: int, image_width: int) -> T.Compose:
    """Build the spatial and photometric augmentation recipe used by DROID."""
    return T.Compose(
        [
            T.RandomCrop((int(image_height * DROID_IMAGE_CROP_RATIO), int(image_width * DROID_IMAGE_CROP_RATIO))),
            T.Resize((image_height, image_width), antialias=True),
            T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
        ]
    )


def apply_synchronized_image_augmentation(
    videos: list[torch.Tensor],  # Each video: [T,C,H,W]
    augmentor: Callable[[torch.Tensor], torch.Tensor],
) -> list[torch.Tensor]:  # Each returned video: [T,C,H,W]
    """Apply one sampled augmentation consistently across every frame and view."""
    if not videos:
        raise ValueError("At least one video is required for image augmentation")

    reference_shape = videos[0].shape[1:]
    if any(video.ndim != 4 or video.shape[1:] != reference_shape for video in videos):
        shapes = [tuple(video.shape) for video in videos]
        raise ValueError(f"All videos must have matching [C,H,W] dimensions, got {shapes}")

    frame_counts = [video.shape[0] for video in videos]
    combined = torch.cat(videos, dim=0)  # [sum(T),C,H,W]
    augmented = augmentor(combined)  # [sum(T),C,H,W]
    return list(torch.split(augmented, frame_counts, dim=0))
