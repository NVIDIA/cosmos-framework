# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Local measured-LiDAR tar manifest adapter for tokenizer continuation."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from cosmos_framework.model.generator.tokenizers.lidar.preprocessing import (
    crop_range_video_width,
    enforce_tokenizer_clip_length,
    load_prepared_lidar_video_3ch,
)
from cosmos_framework.model.generator.tokenizers.lidar.range_projection import LidarRangeProjectionConfig


def read_manifest(path: Path) -> list[Path]:
    """Resolve one local tar path per line relative to the manifest directory."""
    samples = [(path.parent / line.strip()).resolve() for line in path.read_text().splitlines() if line.strip()]
    if not samples or len(samples) != len(set(samples)):
        raise ValueError(f"Manifest must contain unique, nonempty sample paths: {path}")
    for sample in samples:
        if not sample.is_file() or sample.suffix != ".tar":
            raise ValueError(f"Expected an existing local LiDAR tar: {sample}")
    return samples


class LidarTarDataset(Dataset):
    samples: list[Path]
    num_source_frames: int
    projection: LidarRangeProjectionConfig

    def __init__(
        self,
        samples: list[Path],
        num_source_frames: int = 8,
        projection: LidarRangeProjectionConfig = LidarRangeProjectionConfig(semantic_width=1800, model_width=1808),
    ) -> None:
        self.samples = samples
        self.num_source_frames = num_source_frames
        self.projection = projection

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        video, frame_ids = load_prepared_lidar_video_3ch(
            self.samples[index],
            min_source_frame_id=0,
            max_source_frame_id=2**63 - 1,
            max_frames_per_chunk=self.num_source_frames,
            prepend_first_frame=True,
            range_projection=self.projection,
        )  # [3,T,128,1808]
        video, frame_ids = enforce_tokenizer_clip_length(
            video,
            frame_ids,
            num_source_frames=self.num_source_frames,
        )  # video: [3,T,128,1808]; frame_ids: list[int]
        semantic = crop_range_video_width(video, semantic_width=self.projection.semantic_width)  # [3,T,128,1800]
        target = semantic[:2].contiguous()  # [2,T,128,1800]
        valid_mask = semantic[2:3].contiguous()  # [1,T,128,1800]
        return {"video": video, "target": target, "valid_mask": valid_mask}
