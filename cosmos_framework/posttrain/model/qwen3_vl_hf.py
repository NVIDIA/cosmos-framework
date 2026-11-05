# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Transformers 5.8.1 dense Qwen3-VL assembly with ordered modality collectives.

Keep stock per-modality visual batches, deepstack assembly and positional/cache
computation. Missing modalities execute only dummy vision and contribute exact
empty-slice zeros. No graph-bearing cache survives a call. The image->video edge
orders both forward all-gathers and backward replay across different-media ranks.
Only the explicit new rounded-weight profile selects this path.
"""

from types import MethodType

import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast
from transformers.utils.generic import can_return_tuple

from cosmos_framework.posttrain.model.qwen3_5 import _dummy_pixels


@can_return_tuple
def _forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    mm_token_type_ids=None,
    **kwargs,
):
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
    image_pixels, image_grid = (
        _dummy_pixels(self, inputs_embeds.device) if pixel_values is None else (pixel_values, image_grid_thw)
    )
    image_output = self.get_image_features(image_pixels, image_grid, return_dict=True)
    image_embeds = torch.cat(image_output.pooler_output, dim=0).to(inputs_embeds)
    image_deep = image_output.deepstack_features
    image_zero = image_embeds.reshape(-1)[:0].sum()
    for feature in image_deep:
        image_zero = image_zero + feature.reshape(-1)[:0].sum().to(image_zero)
    image_mask = video_mask = None
    if pixel_values is not None:
        image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    video_pixels, video_grid = (
        _dummy_pixels(self, inputs_embeds.device)
        if pixel_values_videos is None
        else (pixel_values_videos, video_grid_thw)
    )
    video_pixels = video_pixels + image_zero.to(video_pixels)
    video_output = self.get_video_features(video_pixels, video_grid, return_dict=True)
    video_embeds = torch.cat(video_output.pooler_output, dim=0).to(inputs_embeds)
    video_deep = video_output.deepstack_features
    if pixel_values_videos is not None:
        _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds, video_features=video_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
    video_zero = video_embeds.reshape(-1)[:0].sum()
    for feature in video_deep:
        video_zero = video_zero + feature.reshape(-1)[:0].sum().to(video_zero)
    inputs_embeds = inputs_embeds + video_zero

    visual_pos_masks = deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        image_mask, video_mask = image_mask[..., 0], video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        for image, video in zip(image_deep, video_deep, strict=True):
            joint = image.new_zeros(visual_pos_masks.sum(), image.shape[-1])
            joint[image_mask[visual_pos_masks], :] = image
            joint[video_mask[visual_pos_masks], :] = video
            deepstack_visual_embeds.append(joint)
    elif image_mask is not None:
        visual_pos_masks, deepstack_visual_embeds = image_mask[..., 0], image_deep
    elif video_mask is not None:
        visual_pos_masks, deepstack_visual_embeds = video_mask[..., 0], video_deep
    if position_ids is None:
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
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
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )
    return Qwen3VLModelOutputWithPast(**outputs, rope_deltas=self.rope_deltas)


def patch_ordered_forward(model):
    model.forward = MethodType(_forward, model)
