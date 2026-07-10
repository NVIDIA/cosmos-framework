# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Image-conditioned prefill helpers for the Nemotron 3 Dense VL reasoner.

Mirrors ``reasoner/qwen3_vl/utils.py::prepare_multimodal_reasoner_inputs`` for
the Edge reasoner. The multimodal rope index (``get_rope_index``) and the
placeholder-mask helper are algorithmically identical to the Qwen3-VL image
path — they depend only on ``model.config`` token ids + ``image_grid_thw`` — so
they are reused directly. The Nemotron-specific parts are:

* the vision tower is a SigLIP2 encoder (``causal_lm.visual`` is a
  :class:`NemotronSiglip2VisionEncoder`) whose ``get_image_features`` returns
  the projected ``[N_merged_patches, hidden]`` embeddings directly, and
* there are **no deepstack** visual embeds (returned list is empty), so the
  shared ``_impl_reasoner_forward`` deepstack path is a no-op.
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
    pixel_values_videos: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Build the image-conditioned prefill inputs for the reasoner-only AR path.

    Returns ``(inputs_embeds, visual_pos_masks, deepstack_visual_embeds,
    position_ids, mrope_position_deltas)`` matching the contract consumed by
    :func:`unified_mot._impl_generate_reasoner_text`. ``deepstack_visual_embeds``
    is always ``[]`` for Nemotron (no deepstack tower).
    """
    if pixel_values_videos is not None or video_grid_thw is not None:
        raise NotImplementedError(
            "Video-conditioned reasoner generation is not implemented for Nemotron 3 Dense VL."
        )
    if pixel_values is None or image_grid_thw is None:
        raise ValueError("prepare_multimodal_reasoner_inputs requires pixel_values and image_grid_thw.")
    if not hasattr(causal_lm, "visual") or causal_lm.visual is None:
        raise ValueError("Nemotron reasoner has no vision tower (`causal_lm.visual`).")

    inputs_embeds = causal_lm.model.embed_tokens(input_ids).clone()  # [B,T_prompt,hidden]
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
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        attention_mask=attention_mask,
    )

    return inputs_embeds, visual_pos_masks, deepstack_visual_embeds, position_ids, mrope_position_deltas
