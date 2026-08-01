# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import math

import torch
from torch import nn
from transformers.activations import ACT2FN

from cosmos_framework.data.generator.sequence_packing import ModalityData


def has_noisy_tokens(modality_data: ModalityData | None) -> bool:
    """Check if a modality has valid noisy tokens for loss computation."""
    return (
        modality_data is not None
        and modality_data.tokens is not None
        and isinstance(modality_data.mse_loss_indexes, torch.Tensor)
        and modality_data.mse_loss_indexes.numel() > 0
    )


# --------------------------------------------------------
# TimestepEmbedder
# Reference:
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, bias: bool = True) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=bias),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=bias),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.hidden_size = hidden_size
        self._has_bias = bias
        frequencies = self._build_timestep_frequencies(frequency_embedding_size, max_period=10000)  # [D/2]
        self.register_buffer("_timestep_frequencies", frequencies, persistent=False)

    def _init_weights(self, buffer_device: torch.device | None = None) -> None:
        std = 1.0 / math.sqrt(self.frequency_embedding_size)
        torch.nn.init.trunc_normal_(self.mlp[0].weight, std=std, a=-3 * std, b=3 * std)
        if self._has_bias:
            torch.nn.init.zeros_(self.mlp[0].bias)

        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.mlp[2].weight, std=std, a=-3 * std, b=3 * std)
        if self._has_bias:
            torch.nn.init.zeros_(self.mlp[2].bias)

        frequencies = self._build_timestep_frequencies(
            self.frequency_embedding_size,
            max_period=10000,
            device=buffer_device,
        )  # [D/2]
        self.register_buffer("_timestep_frequencies", frequencies, persistent=False)

    @staticmethod
    def _build_timestep_frequencies(
        dim: int,
        max_period: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:  # [D/2]
        """Build device frequencies from the original CPU calculation."""
        half = dim // 2
        frequencies = torch.exp(  # [D/2]
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        if device is not None:
            frequencies = frequencies.to(device=device)  # [D/2]
        return frequencies

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor,  # [N]
        dim: int,
        max_period: int = 10000,
        frequencies: torch.Tensor | None = None,  # [D/2]
    ) -> torch.Tensor:  # [N,D]
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        if frequencies is None:
            frequencies = TimestepEmbedder._build_timestep_frequencies(dim, max_period, t.device)  # [D/2]
        args = t[:, None] * frequencies[None]  # [N,D/2]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [N,D]
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)  # [N,D+1]
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t: [N], returns [N,hidden_size]
        t_freq = self.timestep_embedding(  # [N,frequency_embedding_size]
            t,
            self.frequency_embedding_size,
            frequencies=self._timestep_frequencies,
        )
        t_emb = self.mlp(t_freq)  # [N,hidden_size]
        return t_emb


class MLPconnector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_act: str):
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)  # [N,out_dim]
        hidden_states = self.activation_fn(hidden_states)  # [N,out_dim]
        hidden_states = self.fc2(hidden_states)  # [N,out_dim]
        return hidden_states
