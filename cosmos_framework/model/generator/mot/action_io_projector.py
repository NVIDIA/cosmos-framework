# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Configurable action input/output projectors."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from cosmos_framework.model.generator.mot.domain_aware_linear import DomainAwareLinear

ACTION_IO_PROJECTOR_DOMAIN_AWARE = "domain_aware"
ACTION_IO_PROJECTOR_SHARED_WEIGHT_BIAS = "shared_weight_bias"
ACTION_IO_PROJECTOR_SHARED_WEIGHT_NO_BIAS = "shared_weight_no_bias"
ACTION_IO_PROJECTOR_TYPES = (
    ACTION_IO_PROJECTOR_DOMAIN_AWARE,
    ACTION_IO_PROJECTOR_SHARED_WEIGHT_BIAS,
    ACTION_IO_PROJECTOR_SHARED_WEIGHT_NO_BIAS,
)


class SharedWeightBiasLinear(nn.Module):
    r"""Linear layer with one global weight and one global bias.

    This layer computes ``y = linear(x, K_shared, b_shared)`` for every action
    token type. ``type_id`` is accepted but intentionally ignored for interface
    parity with the other action I/O projectors.
    """

    def __init__(self, input_size: int, output_size: int, num_types: int) -> None:
        super().__init__()
        if input_size <= 0 or output_size <= 0 or num_types <= 0:
            raise ValueError(
                "SharedWeightBiasLinear dimensions must be positive, got "
                f"input_size={input_size}, output_size={output_size}, num_types={num_types}."
            )

        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.num_types = int(num_types)
        self.num_domains = self.num_types

        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))  # [O,I]
        self.bias = nn.Parameter(torch.empty(self.output_size))  # [O]
        self.initialize_action_parameters(1.0 / math.sqrt(self.input_size))

    def initialize_action_parameters(self, std: float) -> None:
        """Apply the shared Cosmos3 action-boundary initialization policy."""
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)  # [O,I]
        nn.init.zeros_(self.bias)  # [O]

    def forward(
        self, x: torch.Tensor, type_id: torch.LongTensor
    ) -> torch.Tensor:  # x: [B,I] or [B,T,I], returns [B,O] or [B,T,O]
        """Project rank-2 or rank-3 tokens without type-conditioned parameters."""
        if x.ndim not in (2, 3):
            raise ValueError(f"SharedWeightBiasLinear expected rank-2 or rank-3 input, got {tuple(x.shape)}.")
        return F.linear(x, self.weight, self.bias)  # [B,O] or [B,T,O]


class SharedWeightNoBiasLinear(nn.Module):
    r"""Linear layer with one global weight and no bias.

    This layer computes ``y = linear(x, K_shared)`` for every action token
    type. ``type_id`` is accepted but intentionally ignored for interface
    parity with the other action I/O projectors.
    """

    def __init__(self, input_size: int, output_size: int, num_types: int) -> None:
        super().__init__()
        if input_size <= 0 or output_size <= 0 or num_types <= 0:
            raise ValueError(
                "SharedWeightNoBiasLinear dimensions must be positive, got "
                f"input_size={input_size}, output_size={output_size}, num_types={num_types}."
            )

        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.num_types = int(num_types)
        self.num_domains = self.num_types

        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))  # [O,I]
        self.initialize_action_parameters(1.0 / math.sqrt(self.input_size))

    def initialize_action_parameters(self, std: float) -> None:
        """Apply the shared Cosmos3 action-boundary initialization policy."""
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)  # [O,I]

    def forward(
        self, x: torch.Tensor, type_id: torch.LongTensor
    ) -> torch.Tensor:  # x: [B,I] or [B,T,I], returns [B,O] or [B,T,O]
        """Project rank-2 or rank-3 tokens without bias or type-conditioned parameters."""
        if x.ndim not in (2, 3):
            raise ValueError(f"SharedWeightNoBiasLinear expected rank-2 or rank-3 input, got {tuple(x.shape)}.")
        return F.linear(x, self.weight)  # [B,O] or [B,T,O]


def build_action_io_projector(
    projector_type: str,
    input_size: int,
    output_size: int,
    num_types: int,
) -> DomainAwareLinear | SharedWeightBiasLinear | SharedWeightNoBiasLinear:
    """Build a paired action encoder/decoder projection implementation."""
    if projector_type == ACTION_IO_PROJECTOR_DOMAIN_AWARE:
        return DomainAwareLinear(input_size, output_size, num_types)
    if projector_type == ACTION_IO_PROJECTOR_SHARED_WEIGHT_BIAS:
        return SharedWeightBiasLinear(input_size, output_size, num_types)
    if projector_type == ACTION_IO_PROJECTOR_SHARED_WEIGHT_NO_BIAS:
        return SharedWeightNoBiasLinear(input_size, output_size, num_types)
    raise ValueError(
        f"Unsupported action_io_projector_type={projector_type!r}; expected one of {ACTION_IO_PROJECTOR_TYPES}."
    )
