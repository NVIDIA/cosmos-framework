# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import attrs
import torch
import torch.nn as nn

from cosmos_framework.utils.config import make_freezable


@make_freezable
@attrs.define(slots=False)
class ModelConfig:
    input_size: int = 1152
    num_classes: int = 7


class SafetyClassifier(nn.Module):
    def __init__(self, input_size: int = 1024, num_classes: int = 2):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes),
            # Note: No activation function here; CrossEntropyLoss expects raw logits
        )

    def forward(self, x):
        return self.layers(x)


class VideoSafetyModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.network = SafetyClassifier(input_size=config.input_size, num_classes=self.num_classes)

    @torch.inference_mode()
    def forward(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logits = self.network(data_batch["data"].cuda())
        return {"logits": logits}
