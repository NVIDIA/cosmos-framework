# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""TF5 Qwen2/2.5-VL: ordered real/dummy vision calls before language FSDP.

All positional arithmetic and vision/language layers remain upstream-owned.
The image-to-video graph edge orders backward collectives, including checkpoint
replay. Only real video intervals reach Qwen2.5's temporal RoPE computation.
"""

from types import MethodType

import torch
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
    use_cache=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    rope_deltas=None,
    mm_token_type_ids=None,
    second_per_grid_ts=None,
    **kwargs,
):
    del rope_deltas  # Upstream maintains its own computed rope_deltas field.
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
    image_pixels, image_grid = (
        _dummy_pixels(self, inputs_embeds.device) if pixel_values is None else (pixel_values, image_grid_thw)
    )
    image = self.get_image_features(image_pixels, image_grid, return_dict=True)
    image_embeds = torch.cat(image.pooler_output, dim=0).to(inputs_embeds)
    image_zero = image_embeds.reshape(-1)[:0].sum()
    if pixel_values is not None:
        mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(mask, image_embeds)
    video_pixels, video_grid = (
        _dummy_pixels(self, inputs_embeds.device)
        if pixel_values_videos is None
        else (pixel_values_videos, video_grid_thw)
    )
    video = self.get_video_features(video_pixels + image_zero.to(video_pixels), video_grid, return_dict=True)
    video_embeds = torch.cat(video.pooler_output, dim=0).to(inputs_embeds)
    if pixel_values_videos is not None:
        _, mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(mask, video_embeds)
    inputs_embeds = inputs_embeds + video_embeds.reshape(-1)[:0].sum()
    if position_ids is None:
        temporal = {"second_per_grid_ts": second_per_grid_ts} if self.config.model_type == "qwen2_5_vl" else {}
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
            **temporal,
        )
    output = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )
    return self._posttrain_output_type(**output, rope_deltas=self.rope_deltas)


def patch_ordered_forward(model):
    import importlib

    module = importlib.import_module(type(model).__module__)
    model._posttrain_output_type = getattr(module, type(model).__name__ + "OutputWithPast")
    model.forward = MethodType(_forward, model)
