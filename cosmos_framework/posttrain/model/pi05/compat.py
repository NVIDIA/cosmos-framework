# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 Physical Intelligence
# SPDX-License-Identifier: OpenMDW-1.1
"""Local OpenPI Gemma extensions using unmodified Transformers components.

The arithmetic follows OpenPI's Transformers 4.53.2 replacements distributed
with RLinf ce4d821780d0e431d90f43d028fc93cac61f8fe0. Constructors compose
Transformers 4.57.6 attention/MLP/rotary modules explicitly: no package files,
AutoModel mappings or global classes are patched. Prefix caches are immutable
tensor pairs, so expert denoising cannot append to a previous denoising step.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import GemmaConfig
from transformers.models.gemma.modeling_gemma import (
    GemmaAttention,
    GemmaMLP,
    GemmaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

KVPair = tuple[torch.Tensor, torch.Tensor]


def additive_attention_mask(mask: torch.Tensor) -> torch.Tensor:
    """The reference finite FP32 attention bias, independent of a loader's dtype context."""
    return torch.zeros_like(mask, dtype=torch.float32)[:, None].masked_fill(~mask[:, None], -2.3819763e38)


class Pi05RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float, *, adaptive: bool) -> None:
        super().__init__()
        self.eps = eps
        self.dense = nn.Linear(width, 3 * width, dtype=torch.float32) if adaptive else None
        if not adaptive:
            self.weight = nn.Parameter(torch.zeros(width, dtype=torch.float32))

    def forward(
        self, hidden: torch.Tensor, condition: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dtype = hidden.dtype
        variance = torch.mean(torch.square(hidden.float()), dim=-1, keepdim=True)
        normalized = hidden * torch.rsqrt(variance + self.eps)
        if self.dense is None:
            return (normalized * (1.0 + self.weight.float())).to(dtype), None
        modulation = self.dense(condition).unsqueeze(1)
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        normalized = normalized * (1 + scale.float()) + shift.float()
        return normalized.to(dtype), gate.to(dtype)


def gated_residual(hidden: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    return hidden + residual if gate is None else hidden + residual * gate


class Pi05GemmaAttention(GemmaAttention):
    """Read-only prefix attention is a public forward for its BF16 FSDP group."""

    def forward(
        self,
        hidden: torch.Tensor,
        position_embeddings: KVPair,
        attention_mask: torch.Tensor,
        prefix: KVPair | None,
    ) -> tuple[torch.Tensor, KVPair]:
        shape = (*hidden.shape[:-1], -1, self.head_dim)
        query = self.q_proj(hidden).view(shape).transpose(1, 2)
        key = self.k_proj(hidden).view(shape).transpose(1, 2)
        value = self.v_proj(hidden).view(shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        if prefix is not None:
            key = torch.cat((prefix[0], key), dim=2)
            value = torch.cat((prefix[1], value), dim=2)
        output, _ = eager_attention_forward(self, query, key, value, attention_mask, self.scaling)
        output = output.reshape(*hidden.shape[:-1], -1).contiguous()
        return self.o_proj(output), (key, value)


class Pi05GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int, *, adaptive: bool) -> None:
        super().__init__()
        self.self_attn = Pi05GemmaAttention(config, layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = Pi05RMSNorm(config.hidden_size, config.rms_norm_eps, adaptive=adaptive)
        self.post_attention_layernorm = Pi05RMSNorm(config.hidden_size, config.rms_norm_eps, adaptive=adaptive)

    def forward(
        self,
        hidden: torch.Tensor,
        position_embeddings: KVPair,
        attention_mask: torch.Tensor,
        condition: torch.Tensor | None,
        prefix: KVPair | None,
    ) -> tuple[torch.Tensor, KVPair]:
        normalized, gate = self.input_layernorm(hidden, condition)
        output, cache = self.self_attn(normalized, position_embeddings, attention_mask, prefix)
        hidden = gated_residual(hidden, output, gate)
        normalized, gate = self.post_attention_layernorm(hidden, condition)
        return gated_residual(hidden, self.mlp(normalized), gate), cache


class Pi05GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig, *, adaptive: bool) -> None:
        super().__init__()
        self.config = config
        self.config._attn_implementation = "eager"
        self.embed_tokens = (
            None if adaptive else nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        )
        self.layers = nn.ModuleList(
            Pi05GemmaDecoderLayer(config, index, adaptive=adaptive) for index in range(config.num_hidden_layers)
        )
        self.norm = Pi05RMSNorm(config.hidden_size, config.rms_norm_eps, adaptive=adaptive)
        self.rotary_emb = GemmaRotaryEmbedding(config=config)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        condition: torch.Tensor | None = None,
        prefix_cache: tuple[KVPair, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[KVPair, ...]]:
        hidden = hidden.to(self.layers[0].self_attn.q_proj.weight.dtype)
        positions = self.rotary_emb(hidden, position_ids)
        cache = []
        for index, layer in enumerate(self.layers):
            prefix = None if prefix_cache is None else prefix_cache[index]
            hidden, kv = layer(hidden, positions, attention_mask, condition, prefix)
            cache.append(kv)
        hidden, _ = self.norm(hidden, condition)
        return hidden, tuple(cache)
