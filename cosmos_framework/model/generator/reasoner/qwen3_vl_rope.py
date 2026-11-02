# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shape-independent FP32 phases for Qwen3-VL long-context RoPE."""

from types import MethodType
from typing import cast

import torch
import torch.nn as nn
from transformers.modeling_rope_utils import dynamic_rope_update
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextRotaryEmbedding


@torch.no_grad()
@dynamic_rope_update
def _fp32_rotary_forward(
    self: Qwen3VLTextRotaryEmbedding | Qwen3VLMoeTextRotaryEmbedding, x: torch.Tensor, position_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Qwen3-VL uses separate temporal, height and width position axes.
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    device_type = x.device.type if x.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        # The native outer-product matmul may use TF32 despite autocast being off.
        # Long position IDs then lose precision differently for full sequences and
        # CP shards. Elementwise FP32 multiplication avoids TF32 without changing
        # the precision or performance policy for the model's other matmuls.
        # Native Qwen rotary modules register inv_freq as a tensor buffer.
        inv_freq = cast(torch.Tensor, self.inv_freq)
        frequencies = position_ids.float().unsqueeze(-1) * inv_freq.float()[None, None, None, :]
        frequencies = self.apply_interleaved_mrope(frequencies, self.mrope_section)
        phases = torch.cat((frequencies, frequencies), dim=-1)
        cos = phases.cos() * self.attention_scaling
        sin = phases.sin() * self.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def configure_qwen3_vl_fp32_rope(rotary_embedding: nn.Module) -> None:
    """Keep native MRoPE and dynamic updates, replacing only the phase outer product."""
    rotary_embedding.forward = MethodType(_fp32_rotary_forward, rotary_embedding)
