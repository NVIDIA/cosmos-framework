# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Image/video-conditioned prefill helpers for the Nemotron 3 Dense VL reasoner.

Mirrors ``reasoner/qwen3_vl/utils.py::prepare_multimodal_reasoner_inputs`` for
the Edge reasoner. The multimodal rope index (``get_rope_index``) and the
placeholder-mask helper are algorithmically identical to the Qwen3-VL path —
they depend only on ``model.config`` token ids + ``image_grid_thw`` /
``video_grid_thw`` — so they are reused directly. The Nemotron-specific parts
are:

* the vision tower is a SigLIP2 encoder (``causal_lm.visual`` is a
  :class:`NemotronSiglip2VisionEncoder`) whose ``get_image_features`` returns
  the projected ``[N_merged_patches, hidden]`` embeddings directly. SigLIP2 has
  no temporal attention, so a video is just its frames stacked along the grid's
  temporal axis (``video_grid_thw`` rows carry ``t>1``); the same
  ``get_image_features`` encodes them patch-by-patch, and
* there are **no deepstack** visual embeds (returned list is empty), so the
  shared ``_impl_reasoner_forward`` deepstack path is a no-op.

Image and video are mutually exclusive: exactly one of the
(``pixel_values``, ``image_grid_thw``) / (``pixel_values_videos``,
``video_grid_thw``) pairs is consumed per call.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import (
    get_placeholder_mask,
    get_rope_index,
)


def prepare_multimodal_reasoner_inputs(
    causal_lm: Any,
    input_ids: torch.Tensor,  # [B,T_prompt]
    pixel_values: torch.Tensor | None = None,  # [N_patches, C*patch*patch]
    image_grid_thw: torch.Tensor | None = None,  # [num_images,3]
    pixel_values_videos: torch.Tensor | None = None,  # [N_patches, C*patch*patch]
    video_grid_thw: torch.Tensor | None = None,  # [num_videos,3]
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Build the image/video-conditioned prefill inputs for the reasoner-only AR path.

    Returns ``(inputs_embeds, visual_pos_masks, deepstack_visual_embeds,
    position_ids, mrope_position_deltas)`` matching the contract consumed by
    :func:`unified_mot._impl_generate_reasoner_text`. ``deepstack_visual_embeds``
    is always ``[]`` for Nemotron (no deepstack tower).

    The video recipe mirrors the image recipe but routes the pre-processed
    ``pixel_values_videos`` / ``video_grid_thw`` through the shared SigLIP2
    ``get_image_features`` (temporal-agnostic), the video placeholder mask, and
    the ``video_grid_thw`` rope index.
    """
    is_video = pixel_values_videos is not None or video_grid_thw is not None
    if is_video and (pixel_values is not None or image_grid_thw is not None):
        raise ValueError(
            "prepare_multimodal_reasoner_inputs conditions on one medium at a time: "
            "pass the image pair OR the video pair, not both."
        )
    if is_video:
        if pixel_values_videos is None or video_grid_thw is None:
            raise ValueError(
                "prepare_multimodal_reasoner_inputs requires pixel_values_videos and video_grid_thw together."
            )
    elif pixel_values is None or image_grid_thw is None:
        raise ValueError("prepare_multimodal_reasoner_inputs requires pixel_values and image_grid_thw.")
    if not hasattr(causal_lm, "visual") or causal_lm.visual is None:
        raise ValueError("Nemotron reasoner has no vision tower (`causal_lm.visual`).")

    inputs_embeds = causal_lm.model.embed_tokens(input_ids).clone()  # [B,T_prompt,hidden]

    if is_video:
        pixel_values_videos = pixel_values_videos.to(device=inputs_embeds.device)
        video_grid_thw = video_grid_thw.to(device=inputs_embeds.device)
        # SigLIP2 has no temporal modeling: a video is its frames as extra grid
        # patches, so the image encoder handles it unchanged. Returns [N_merged_patches, hidden].
        video_embeds = causal_lm.visual.get_image_features(pixel_values_videos, video_grid_thw)
        video_embeds = video_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

        _image_mask, video_mask = get_placeholder_mask(
            causal_lm,
            input_ids,
            inputs_embeds=inputs_embeds,
            video_features=video_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)  # [B,T_prompt,hidden]
        visual_pos_masks = video_mask[..., 0]  # [B,T_prompt]
    else:
        pixel_values = pixel_values.to(device=inputs_embeds.device)
        image_grid_thw = image_grid_thw.to(device=inputs_embeds.device)

        # SigLIP2 encode -> spatial-merge -> project. Returns [N_merged_patches, hidden].
        image_embeds = causal_lm.visual.get_image_features(pixel_values, image_grid_thw)
        image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

        image_mask, _video_mask = get_placeholder_mask(
            causal_lm,
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)  # [B,T_prompt,hidden]
        visual_pos_masks = image_mask[..., 0]  # [B,T_prompt]

    deepstack_visual_embeds: list[torch.Tensor] = []

    position_ids, mrope_position_deltas = get_rope_index(
        causal_lm,
        input_ids=input_ids,
        image_grid_thw=None if is_video else image_grid_thw,
        video_grid_thw=video_grid_thw if is_video else None,
        attention_mask=attention_mask,
    )

    return inputs_embeds, visual_pos_masks, deepstack_visual_embeds, position_ids, mrope_position_deltas
