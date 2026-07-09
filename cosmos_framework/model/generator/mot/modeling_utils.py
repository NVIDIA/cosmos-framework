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

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.hidden_size = hidden_size

    def _init_weights(self):
        std = 1.0 / math.sqrt(self.frequency_embedding_size)
        torch.nn.init.trunc_normal_(self.mlp[0].weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.zeros_(self.mlp[0].bias)

        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.mlp[2].weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.zeros_(self.mlp[2].bias)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None] * freqs[None]  # [N,D/2]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [N,D]
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)  # [N,D+1]
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)  # [N,frequency_embedding_size]
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
