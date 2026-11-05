# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Qwen3.5-family dense SFT adaptation for Transformers 5.8.1 (including Qwen3.8).

Each rank executes image vision, video vision, then language blocks, even for text-only
inputs. Empty-slice sums retain zero-gradient edges without multiplying possibly nonfinite
features by zero. The image -> video input edge also orders BACKWARD: all video block
recomputations/reductions precede image ones on every rank. Merely adding two independent
dummy losses would not impose that order. No graph-bearing state is saved on the module.

Install on the backbone before FSDP wrapping. State-dict names and upstream block types
stay unchanged. This is a training adapter, not a new model or an MTP implementation.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from transformers.cache_utils import Cache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5ModelOutputWithPast
from transformers.utils.generic import can_return_tuple


def _dummy_pixels(model: Qwen3_5Model, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """One spatially merged token, using the actual vision patch geometry."""
    config = model.config.vision_config
    merge = config.spatial_merge_size
    patch = model.visual.patch_embed
    width = config.in_channels * patch.temporal_patch_size * patch.patch_size**2
    pixels = torch.zeros(merge**2, width, device=device, dtype=model.visual.dtype)  # [patches,patch_width]
    grid = torch.tensor([[1, merge, merge]], device=device, dtype=torch.long)  # [1,THW]
    return pixels, grid


@can_return_tuple
def _forward(
    self: Qwen3_5Model,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.Tensor | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    mm_token_type_ids: torch.Tensor | None = None,
    **kwargs: Any,
) -> Qwen3_5ModelOutputWithPast:
    """Upstream multimodal assembly, with unconditional, backward-ordered vision calls."""
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)  # [B,L,H]

    image_pixels, image_grid = (
        _dummy_pixels(self, inputs_embeds.device) if pixel_values is None else (pixel_values, image_grid_thw)
    )
    image_output = self.get_image_features(image_pixels, image_grid, return_dict=True)
    image_embeds = torch.cat(image_output.pooler_output, dim=0).to(inputs_embeds)  # [image_tokens,H]
    image_zero = image_embeds.reshape(-1)[:0].sum()  # scalar, graph-connected exact zero
    if pixel_values is not None:
        image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)  # [B,L,H]

    video_pixels, video_grid = (
        _dummy_pixels(self, inputs_embeds.device)
        if pixel_values_videos is None
        else (pixel_values_videos, video_grid_thw)
    )
    # Make the second call depend on the first even when neither modality has real data.
    video_pixels = video_pixels + image_zero.to(video_pixels)  # [video_patches,patch_width]
    video_output = self.get_video_features(video_pixels, video_grid, return_dict=True)
    video_embeds = torch.cat(video_output.pooler_output, dim=0).to(inputs_embeds)  # [video_tokens,H]
    if pixel_values_videos is not None:
        _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds, video_features=video_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)  # [B,L,H]
    # Attach after both scatters: replacing every input token must not sever the dummy graph.
    inputs_embeds = inputs_embeds + video_embeds.reshape(-1)[:0].sum()  # [B,L,H]

    if position_ids is None:
        # Only REAL grids reach M-RoPE; dummy features must not change positions or token counts.
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
        )
    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )
    return Qwen3_5ModelOutputWithPast(**outputs, rope_deltas=self.rope_deltas)


def patch_qwen3_5_forward(model: Qwen3_5Model) -> None:
    """Bind only this instance; call before ``HFModel.setup`` captures FSDP forwards."""
    model.forward = MethodType(_forward, model)
