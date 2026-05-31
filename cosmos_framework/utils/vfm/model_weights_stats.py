# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class TrainingStats:
    """Data class to hold training statistics."""

    video_samples: int = 0
    image_samples: int = 0
    iterations: int = 0
    training_hours: float = 0.0


class WeightTrainingStat(nn.Module, ABC):
    """Abstract base class for tracking training statistics."""

    def __init__(self) -> None:
        super().__init__()
        self._initialize_tracking_buffers()

    def _initialize_tracking_buffers(self) -> None:
        """Initialize tracking buffers with default values."""
        tracking_buffers = {
            "accum_video_sample_counter": torch.tensor(0, dtype=torch.int64),
            "accum_image_sample_counter": torch.tensor(0, dtype=torch.int64),
            "accum_iteration": torch.tensor(0, dtype=torch.int64),
            "accum_train_in_hours": torch.tensor(0.0, dtype=torch.float32),
        }

        for name, tensor in tracking_buffers.items():
            self.register_buffer(name, tensor)

    def get_training_stats(self) -> TrainingStats:
        """Return current training statistics."""
        return TrainingStats(
            video_samples=self.accum_video_sample_counter.item(),
            image_samples=self.accum_image_sample_counter.item(),
            iterations=self.accum_iteration.item(),
            training_hours=self.accum_train_in_hours.item(),
        )

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass
