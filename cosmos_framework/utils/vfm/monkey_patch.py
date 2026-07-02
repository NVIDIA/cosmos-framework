# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import importlib

import torch
import transformers
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from transformers.utils.import_utils import is_torchdynamo_compiling

from cosmos_framework.utils import log

_EXPECTED_TRANSFORMERS_VERSION_PREFIX = "4.57."


def patch_qwen3_vl_forward(model):
    """Monkey-patch a ``Qwen3VLModel`` **instance's** forward:
       **Single visual forward per batch**: Under FSDP, every rank must call
       ``self.visual(...)`` the same number of times each forward step so that
       collective all-gather operations stay in sync. Image and video inputs are
       encoded together in one visual call. When a batch contains only text, a
       lightweight dummy image (16x16 zeros) is pushed through the full ViT ->
       merger -> deepstack pipeline, then outputs are sliced to ``[0:0]`` so
       they carry ``grad_fn`` but contribute no features.
    Args:
        model: The ``Qwen3VLModel`` instance (i.e. ``model.model.model`` when
            the outer model is ``HFModel``).
    """
    if not transformers.__version__.startswith(_EXPECTED_TRANSFORMERS_VERSION_PREFIX):
        raise ValueError(f"monkey patching transformers version {transformers.__version__} is not supported")

    if not isinstance(model, Qwen3VLModel):
        raise ValueError(f"Trying to monkey patch a model that is not a Qwen3VLModel instance: {type(model)}")

    # Resolve the output dataclass from the actual runtime module
    model_module = importlib.import_module(type(model).__module__)
    Qwen3VLModelOutputWithPast = getattr(model_module, "Qwen3VLModelOutputWithPast")

    # Replaces Qwen3VLModel.forward from:
    #   transformers.models.qwen3_vl.modeling_qwen3_vl  (transformers v4.57.1)
    def patched_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        visual_pixel_values_list: list[torch.Tensor] = []
        visual_grid_thw_list: list[torch.Tensor] = []
        image_embed_len = 0
        video_embed_len = 0
        has_image = pixel_values is not None
        has_video = pixel_values_videos is not None

        if has_image:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw must be provided when pixel_values is provided")
            visual_pixel_values_list.append(pixel_values)
            visual_grid_thw_list.append(image_grid_thw)
            image_embed_len = int((image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).sum().item())

        if has_video:
            if video_grid_thw is None:
                raise ValueError("video_grid_thw must be provided when pixel_values_videos is provided")
            visual_pixel_values_list.append(pixel_values_videos)
            visual_grid_thw_list.append(video_grid_thw)
            video_embed_len = int((video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).sum().item())

        if has_image or has_video:
            visual_pixel_values = torch.cat(visual_pixel_values_list, dim=0).type(self.visual.dtype)  # [N_patch,D]
            visual_grid_thw = torch.cat(visual_grid_thw_list, dim=0)  # [N_media,3]
            visual_embeds, deepstack_visual_feature_lists = self.visual(
                visual_pixel_values, grid_thw=visual_grid_thw
            )  # visual_embeds: [N_visual,C]
            image_embeds = visual_embeds[:image_embed_len]  # [N_image_visual,C]
            video_embeds = visual_embeds[image_embed_len : image_embed_len + video_embed_len]  # [N_video_visual,C]
            deepstack_image_embeds = [
                deepstack_visual_embeds[:image_embed_len] for deepstack_visual_embeds in deepstack_visual_feature_lists
            ]  # each: [N_image_visual,C]
            deepstack_video_embeds = [
                deepstack_visual_embeds[image_embed_len : image_embed_len + video_embed_len]
                for deepstack_visual_embeds in deepstack_visual_feature_lists
            ]  # each: [N_video_visual,C]

            if has_image:
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)  # [N_image_visual,C]
                image_mask, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)  # [B,N_token,C]

            if has_video:
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)  # [N_video_visual,C]
                _, video_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)  # [B,N_token,C]

        # Dummy visual forward for text-only data
        else:
            dummy_h, dummy_w = 16, 16
            dummy_pixels = torch.zeros(
                dummy_h * dummy_w,
                self.visual.config.temporal_patch_size * self.visual.config.patch_size**2 * 3,
                device=inputs_embeds.device,
                dtype=self.visual.dtype,
            )  # [N_dummy_patch,D]
            dummy_thw = torch.tensor([[1, dummy_h, dummy_w]], device=inputs_embeds.device)  # [1,3]
            image_embeds, deepstack_image_embeds = self.visual(dummy_pixels, grid_thw=dummy_thw)
            image_embeds = image_embeds[0:0]  # [0,C]
            deepstack_image_embeds = [e[0:0] for e in deepstack_image_embeds]  # each: [0,C]

            # no-op to mask scatter empty embeddings into inputs to preserve computation graph
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)  # [0,C]
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)  # [B,N_token,C]

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

    # Replace the forward method
    model.forward = patched_forward.__get__(model, type(model))
    log.critical(f"Patched {type(model).__name__} instance forward with one visual call per forward")
