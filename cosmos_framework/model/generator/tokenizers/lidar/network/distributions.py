# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The distribution modes to use for continuous image tokenizers."""

from enum import Enum

import torch


class IdentityDistribution(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, parameters: torch.Tensor, split: bool = False
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:  # parameters: [B,C,...], returns sample: [B,C,...]
        if not split:
            zero = parameters.new_zeros(1)  # [1]
            return parameters, (zero, zero)
        mean, logvar = parameters.chunk(2, dim=1)  # [B,C/2,...], [B,C/2,...]
        mean, logvar = mean.contiguous(), logvar.contiguous()  # [B,C/2,...], [B,C/2,...]
        zero = parameters.new_zeros(1)  # [1]
        return mean, (zero, zero)


class GaussianDistribution(torch.nn.Module):
    def __init__(self, min_logvar: float = -30.0, max_logvar: float = 20.0) -> None:
        super().__init__()
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

    def sample(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:  # [B,C,...], [B,C,...] -> [B,C,...]
        std = torch.exp(0.5 * logvar)  # [B,C,...]
        return mean + std * torch.randn_like(mean)  # [B,C,...]

    def forward(
        self, parameters: torch.Tensor, split: bool = False
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:  # parameters: [B,2C,...], returns sample: [B,C,...]
        del split
        mean, logvar = torch.chunk(parameters, 2, dim=1)  # [B,C,...], [B,C,...]
        mean, logvar = mean.contiguous(), logvar.contiguous()  # [B,C,...], [B,C,...]
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)  # [B,C,...]
        return self.sample(mean, logvar), (mean, logvar)


class ContinuousFormulation(Enum):
    VAE = GaussianDistribution
    AE = IdentityDistribution
