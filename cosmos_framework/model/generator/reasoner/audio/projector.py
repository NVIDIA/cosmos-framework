# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared projector for audio understanding backends."""

import torch
from torch import nn


class AudioProjector(nn.Module):
    """Project audio features with the same MLP shape as the Qwen-VL merger."""

    def __init__(
        self,
        input_hidden_size: int,
        projection_hidden_size: int,
        out_hidden_size: int,
    ) -> None:
        super().__init__()
        self.input_hidden_size = input_hidden_size
        self.projection_hidden_size = projection_hidden_size
        self.out_hidden_size = out_hidden_size
        self.norm = nn.LayerNorm(input_hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(input_hidden_size, projection_hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(projection_hidden_size, out_hidden_size)

    def reset_parameters(self) -> None:
        """Initialize projector parameters, including after ``to_empty``."""
        self.norm.reset_parameters()
        self.linear_fc1.reset_parameters()
        self.linear_fc2.reset_parameters()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.linear_fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        return self.linear_fc2(hidden_states)


__all__ = ["AudioProjector"]
