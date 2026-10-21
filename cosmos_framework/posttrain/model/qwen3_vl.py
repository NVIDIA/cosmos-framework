# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""i4-native Qwen3-VL policy adapter and shared family semantics.

The native model classes are imported lazily from ``projects.cosmos3``; this adapter never routes
through posttrain's generic ``HFModel``. This module owns Qwen construction and assembles explicit
lifecycle hooks for i4 activation checkpointing, FSDP, materialization, and checkpoint loading.
"""

from __future__ import annotations

import importlib
import re
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.utils.import_utils import is_torchdynamo_compiling

from cosmos_framework.posttrain.comm.weight_transfer import WeightWitnessSpec
from cosmos_framework.posttrain.model.protocol import VLMPolicyOutput
from cosmos_framework.posttrain.model.sft_source import register_native_sft_source

if TYPE_CHECKING:
    from cosmos_framework.posttrain.config.model import Qwen3VLModelConfig

_SUPPORTED_MODEL_TYPES: frozenset[str] = frozenset({"qwen3_vl", "qwen3_vl_moe"})
_NATIVE_BACKBONE_IDENTITIES: frozenset[tuple[str, str]] = frozenset(
    {
        ("cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl", "Qwen3VLModel"),
        ("cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe", "Qwen3VLMoeModel"),
    }
)
_NON_MODEL_BATCH_KEYS: frozenset[str] = frozenset(
    {
        "logprob_masks",
        "label_ids",
        "labels",
        "collated",
        "completion_text",
        "completion_token_count",
    }
)


# =============================================================================
# TEMPORARY I4 QWEN3-VL COMPATIBILITY BOUNDARY — DELETE WHEN UPSTREAM SUPPORTS IT
#
# The forward override below and the native config, causal-mask, RoPE, and padded-MoE adapters in
# this module and `qwen3_vl_moe.py` bridge the currently vendored i4 modeling snapshot. They must
# not become a second permanent model implementation. Remove them once i4 natively supports the
# same multimodal forward contract, installed Transformers version, and padded MoE policy batches.
# See `cosmos_framework/posttrain/docs/design/DESIGN.md#v5-native-qwen3-vl-compatibility-boundary`.
# =============================================================================
def override_qwen3_vl_forward(model: nn.Module) -> None:
    """Override one native ``Qwen3VLModel`` / ``Qwen3VLMoeModel`` instance's forward.

       **Single visual forward per batch**: Under FSDP, every rank must call
       ``self.visual(...)`` the same number of times each forward step so that
       collective all-gather operations stay in sync. Image and video inputs are
       encoded together in one visual call. When a batch contains only text, a
       lightweight dummy image (16x16 zeros) is pushed through the full ViT ->
       merger -> deepstack pipeline, then outputs are sliced to ``[0:0]`` so
       they carry ``grad_fn`` but contribute no features.

       Handles both the dense (``Qwen3VLModel``) and MoE (``Qwen3VLMoeModel``)
       backbones: their ``self.visual`` return signature, ``get_placeholder_mask``,
       ``get_rope_index`` and language-model call signature are identical for the
       purposes of this patch; only the output dataclass differs (resolved below).

    Args:
        model: The ``Qwen3VLModel`` / ``Qwen3VLMoeModel`` instance (i.e.
            ``net.model`` below the native conditional-generation module).
    """
    model_identity = (type(model).__module__, type(model).__name__)
    if model_identity not in _NATIVE_BACKBONE_IDENTITIES:
        raise ValueError(
            "Qwen3-VL forward override requires an exact i4-native Qwen3VLModel/Qwen3VLMoeModel, "
            f"got {type(model).__module__}.{type(model).__name__}"
        )
    if bool(getattr(model, "_posttrain_forward_override", False)):
        return

    # Resolve the output dataclass from the actual runtime module. Both dense and
    # MoE name it "<ModelClass>OutputWithPast" (Qwen3VLModel -> Qwen3VLModelOutputWithPast,
    # Qwen3VLMoeModel -> Qwen3VLMoeModelOutputWithPast) and both expose the only fields
    # this patch populates: (last_hidden_state, past_key_values, rope_deltas).
    model_module = importlib.import_module(type(model).__module__)
    output_cls = getattr(model_module, f"{type(model).__name__}OutputWithPast")

    # Replaces Qwen3VLModel/Qwen3VLMoeModel.forward from:
    #   transformers.models.qwen3_vl.modeling_qwen3_vl          (transformers v4.57.1)
    #   transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe  (transformers v4.57.1)
    def patched_forward(
        self: Any,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> Any:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)  # [B,N_token,C]

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
                )  # image_mask: [B,N_token,C]
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)  # [B,N_token,C]

            if has_video:
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)  # [N_video_visual,C]
                _, video_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )  # video_mask: [B,N_token,C]
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
            image_embeds, deepstack_image_embeds = self.visual(  # [N_dummy_visual,C], list[[N_dummy_visual,C]]
                dummy_pixels,
                grid_thw=dummy_thw,
            )
            image_embeds = image_embeds[0:0]  # [0,C]
            deepstack_image_embeds = [e[0:0] for e in deepstack_image_embeds]  # each: [0,C]

            # no-op to mask scatter empty embeddings into inputs to preserve computation graph
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)  # [0,C]
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )  # image_mask: [B,N_token,C]
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)  # [B,N_token,C]

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]  # [B,N_token]
            video_mask = video_mask[..., 0]  # [B,N_token]
            visual_pos_masks = image_mask | video_mask  # [B,N_token]
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]  # [N_visual]
            video_mask_joint = video_mask[visual_pos_masks]  # [N_visual]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(  # [N_visual,C]
                    img_embed.device
                )
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]  # [B,N_token]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]  # [B,N_token]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(  # [B,N_token]
                    attention_mask_tensor[:, 0], dim1=1, dim2=2
                )
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = (  # [B,N_token]
                        attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    )
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()  # [B,N_token]

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
                )  # [B,1] or scalar
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)  # [N_token]
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)  # [B,N_token]
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)  # [B,1]
                position_ids = position_ids.add(delta)  # [B,N_token]
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)  # [3,B,N_token]

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

        return output_cls(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

    # Replace the forward method
    model.forward = patched_forward.__get__(model, type(model))
    model._posttrain_forward_override = True


@dataclass(frozen=True)
class Qwen3VLBuildHooks:
    """Injectable native lifecycle operations, called in their load-bearing order."""

    model_factory: Callable[[Any], nn.Module] | None = None
    prepare_model: Callable[[nn.Module], None] | None = None
    parallelize: Callable[[nn.Module, Any], None] | None = None
    materialize: Callable[[nn.Module], None] | None = None
    load_weights: Callable[[nn.Module, str, str | None, Any], None] | None = None


def validate_qwen3_vl_parallelism(parallel_dims: Any | None) -> None:
    """Accept FSDP/HSDP inside one policy WorkerGroup; reject unsupported model axes.

    Both data-parallel axes consume distinct samples. FSDP2 owns gradient averaging over
    ``dp_replicate * dp_shard`` ranks; the framework must not reduce replica gradients again.
    """
    if parallel_dims is None:
        return
    for axis in ("tp", "ep", "cp", "pp", "cfgp"):
        value = int(getattr(parallel_dims, axis, 1))
        if value != 1:
            raise ValueError(f"Qwen3-VL policy requires {axis}=1, got {value}; only FSDP2 is supported")
    replicate = int(getattr(parallel_dims, "dp_replicate", 1))
    world_size = int(getattr(parallel_dims, "world_size", 1))
    shard = int(getattr(parallel_dims, "dp_shard", world_size))
    if world_size < 1 or replicate < 1 or shard < 1 or replicate * shard != world_size:
        raise ValueError(
            "Qwen3-VL policy requires positive dp_replicate and dp_shard whose product equals "
            "its WorkerGroup rank width, got "
            f"dp_replicate={replicate}, dp_shard={shard}, world_size={world_size}"
        )


def _native_qwen3_vl_classes(model_type: str) -> tuple[type[Any], type[nn.Module]]:
    """Resolve the vendored i4 config/model classes without eager family imports."""
    if model_type == "qwen3_vl":
        from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import Qwen3VLForConditionalGeneration

        return Qwen3VLConfig, Qwen3VLForConditionalGeneration
    if model_type == "qwen3_vl_moe":
        from cosmos_framework.posttrain.model.qwen3_vl_moe import native_qwen3_vl_moe_classes

        return native_qwen3_vl_moe_classes()
    raise ValueError(f"unsupported Qwen3-VL model_type {model_type!r}; expected {sorted(_SUPPORTED_MODEL_TYPES)}")


def _qwen3_vl_causal_mask(
    config: Any,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    past_key_values: Any | None,
    position_ids: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor | None:  # returns [B,1,N_q,N_kv] or None
    """Create the native Qwen3-VL eager mask without Transformers-version registry coupling."""
    del position_ids, kwargs
    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4:
        return attention_mask  # [B,1,N_q,N_kv]
    if getattr(config, "_attn_implementation", "eager") != "eager":
        # Native flash/i4 attention is causal internally. Policy batches are right padded, so valid
        # queries never attend a future padding key; padded query outputs are masked from the loss.
        return None
    from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import eager_mask

    if attention_mask is not None and attention_mask.ndim == 2:
        # Match native mask preprocessing: HF/SFT padding masks may contain integer 0/1 values.
        attention_mask = attention_mask.to(device=cache_position.device, dtype=torch.bool)  # [B,N_kv]
    if past_key_values is None:
        kv_length, kv_offset = input_embeds.shape[1], 0
    else:
        kv_length, kv_offset = past_key_values.get_mask_sizes(cache_position, 0)
    return eager_mask(
        batch_size=input_embeds.shape[0],
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        attention_mask=attention_mask,
        dtype=input_embeds.dtype,
    )  # [B,1,N_q,N_kv]


def _qwen3_vl_default_rope(
    config: Any,
    device: torch.device | str | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, float]:  # returns inv_freq [D/2], attention scaling
    """Restore the original RoPE initializer removed from Transformers 5's public registry."""
    del kwargs
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
    if dim < 2 or dim % 2:
        raise ValueError(f"Qwen3-VL rotary dimension must be a positive even integer, got {dim}")
    base = float(getattr(config, "rope_theta", 10_000.0))
    frequencies = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim  # [D/2]
    inv_freq = 1.0 / (base**frequencies)  # [D/2]
    return inv_freq, 1.0


def _override_native_attention_mask_compatibility(model_type: str) -> None:
    """Bind both native Qwen3-VL families to i4's version-stable causal-mask helper.

    The dense snapshot carries an incomplete local mask registry, while the MoE snapshot imports
    the Transformers helper whose keyword contract changed in Transformers 5. The adapter helper
    has the exact ``input_embeds`` contract both native decoders call and uses i4's eager-mask
    primitive without consulting the installed Transformers registry implementation.
    """
    if model_type == "qwen3_vl":
        from cosmos_framework.model.generator.reasoner.qwen3_vl import qwen3_vl as native_model
    elif model_type == "qwen3_vl_moe":
        from cosmos_framework.model.generator.reasoner.qwen3_vl_moe import qwen3_vl_moe as native_model
    else:
        raise ValueError(f"unsupported Qwen3-VL model_type {model_type!r}")
    native_model.create_causal_mask = _qwen3_vl_causal_mask
    rope_initializers = dict(native_model.ROPE_INIT_FUNCTIONS)
    rope_initializers.setdefault("default", _qwen3_vl_default_rope)
    native_model.ROPE_INIT_FUNCTIONS = rope_initializers


def _coerce_native_config(config_cls: type[Any], model_config: Any, attn_implementation: str) -> Any:
    """Convert an HF-compatible config/dict into the selected vendored config class."""
    source_num_experts: int | None = None
    if isinstance(model_config, config_cls):
        native_config = model_config
    else:
        if isinstance(model_config, Mapping):
            values = dict(model_config)
        else:
            to_dict = getattr(model_config, "to_dict", None)
            if not callable(to_dict):
                raise TypeError(
                    f"Qwen3-VL model_config must be {config_cls.__name__}, a mapping, or expose to_dict(); "
                    f"got {type(model_config).__name__}"
                )
            values = dict(to_dict())
            source_text_config = getattr(model_config, "text_config", None)
            raw_num_experts = getattr(source_text_config, "num_experts", None)
            if raw_num_experts is not None:
                if isinstance(raw_num_experts, bool) or not isinstance(raw_num_experts, int) or raw_num_experts < 1:
                    raise ValueError(f"Qwen3-VL-MoE num_experts must be a positive integer, got {raw_num_experts!r}")
                source_num_experts = raw_num_experts
                text_values = values.get("text_config")
                if not isinstance(text_values, Mapping):
                    raise ValueError("Qwen3-VL-MoE serialized config has no text_config mapping")
                # Transformers 5 exposes the canonical value as the live ``num_experts``
                # attribute but serializes only its ``num_local_experts`` alias. Passing that
                # lossy dictionary to the vendored config silently selects its 60-expert default,
                # which no longer matches either the 128-expert checkpoint or FSDP shard shapes.
                values["text_config"] = {**text_values, "num_experts": source_num_experts}
        from_dict = getattr(config_cls, "from_dict", None)
        native_config = from_dict(values) if callable(from_dict) else config_cls(**values)
    setattr(native_config, "_attn_implementation", attn_implementation)
    text_config = getattr(native_config, "text_config", None)
    vision_config = getattr(native_config, "vision_config", None)
    if source_num_experts is not None and getattr(text_config, "num_experts", None) != source_num_experts:
        raise ValueError(
            "native Qwen3-VL-MoE config changed the live expert count: "
            f"source={source_num_experts}, native={getattr(text_config, 'num_experts', None)!r}"
        )
    if text_config is not None:
        setattr(text_config, "_attn_implementation", attn_implementation)
    if vision_config is not None:
        setattr(vision_config, "_attn_implementation", attn_implementation)
    return native_config


# END TEMPORARY I4 QWEN3-VL COMPATIBILITY BOUNDARY


def _normalize_parameter_dtype(model: nn.Module, dtype: torch.dtype) -> None:
    """Normalize floating-point parameters while they are still cheap meta tensors."""
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point() and parameter.dtype != dtype:
                parameter.data = parameter.data.to(dtype=dtype)  # [*weight]


def _default_materialize(model: nn.Module) -> None:
    """Allocate meta tensors and co-locate existing buffers on the compute device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def materialize(tensor: torch.Tensor) -> torch.Tensor:  # tensor: [*], returns [*]
        if tensor.device.type == "meta":
            return torch.empty_like(tensor, device=device)  # [*]
        return tensor.to(device=device)  # [*]

    model._apply(materialize, recurse=True)


def _tie_embeddings(model: nn.Module, config: Any) -> None:
    """Bind the tied output head before loading a checkpoint that may omit it."""
    if not bool(getattr(config, "tie_word_embeddings", False)):
        return
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if not callable(get_input_embeddings):
        raise TypeError("tied Qwen3-VL model exposes no get_input_embeddings()")
    input_embeddings = get_input_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "weight"):
        raise TypeError("tied Qwen3-VL model has no input embedding weight")
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    output_embeddings = get_output_embeddings() if callable(get_output_embeddings) else None
    if output_embeddings is not None:
        output_embeddings.weight = input_embeddings.weight
        return
    tied_keys = tuple(getattr(model, "_tied_weights_keys", None) or ())
    if not tied_keys:
        raise ValueError("tie_word_embeddings=True but the native Qwen3-VL declares no output-weight path")
    for key in tied_keys:
        *module_path, attribute = key.split(".")
        target: Any = model
        for component in module_path:
            target = getattr(target, component, None)
            if target is None:
                raise ValueError(f"cannot resolve tied Qwen3-VL output path {key!r}")
        setattr(target, attribute, input_embeddings.weight)


def _default_load_weights(
    model: nn.Module,
    checkpoint_path: str,
    credential_path: str | None,
    parallel_dims: Any,
) -> None:
    """Use the i4-native dense/MoE safetensors loader."""
    from cosmos_framework.model.generator.utils.safetensors_loader import load_vlm_model

    load_vlm_model(
        model=model,
        checkpoint_path=checkpoint_path,
        credential_path=credential_path,
        parallel_dims=parallel_dims,
    )


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)  # [*]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


class Qwen3VLPolicy:
    """Posttrain capability adapter over an i4-native Qwen3-VL family module."""

    def __init__(self, net: nn.Module, model_type: str) -> None:
        if model_type not in _SUPPORTED_MODEL_TYPES:
            raise ValueError(f"unsupported Qwen3-VL model_type {model_type!r}")
        self.net = net
        config = getattr(net, "config", None)
        self.weight_mapper = Qwen3VLWeightMapper(
            model_type,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def validate_policy_batch(self, batch: dict[str, Any]) -> None:
        """Validate the model-input contract before any FSDP rank enters forward."""
        if "input_ids" not in batch:
            raise ValueError("Qwen3-VL policy batch is missing input_ids")
        input_ids = batch["input_ids"]
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError(
                f"Qwen3-VL input_ids must be a [B,T] tensor, got {type(input_ids).__name__} "
                f"with shape {getattr(input_ids, 'shape', None)}"
            )
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
                raise ValueError(
                    f"Qwen3-VL attention_mask must be [B,T], got {type(attention_mask).__name__} "
                    f"with shape {getattr(attention_mask, 'shape', None)}"
                )
            if tuple(attention_mask.shape) != tuple(input_ids.shape):
                raise ValueError(
                    f"Qwen3-VL attention_mask shape {tuple(attention_mask.shape)} does not match "
                    f"input_ids {tuple(input_ids.shape)}"
                )

    def prepare_policy_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Validate and move the complete batch before peers enter FSDP forward."""
        self.validate_policy_batch(batch)
        try:
            device = next(self.net.parameters()).device
        except StopIteration as error:
            raise ValueError("Qwen3-VL policy has no parameters") from error
        prepared = {
            key: _move_to_device(value, device) for key, value in batch.items() if key not in _NON_MODEL_BATCH_KEYS
        }
        return prepared

    def forward_policy_batch(self, batch: dict[str, Any]) -> VLMPolicyOutput:
        """Forward a batch already prepared by the rank-converged preflight."""
        self.validate_policy_batch(batch)
        input_ids = batch["input_ids"]  # [B,T]
        model_batch = {key: value for key, value in batch.items() if key not in _NON_MODEL_BATCH_KEYS}
        output = self.net(**model_batch, use_cache=False)  # output.logits: [B,T,V]
        # Do not validate post-forward rank-local output here. The OPD objective extracts and checks
        # logits inside its rank-converged finalization phase, so a malformed lane cannot strand a
        # peer before the next collective. ``cast`` changes only the static type, never the value.
        raw_logits = output if isinstance(output, torch.Tensor) else getattr(output, "logits", None)
        logits = cast(torch.Tensor, raw_logits)  # [B,T,V] (validated by the objective)
        return VLMPolicyOutput(logits=logits)


def build_qwen3_vl_policy(
    *,
    model_type: str,
    model_config: Any,
    checkpoint_path: str | None = None,
    credential_path: str | None = None,
    parallel_dims: Any | None = None,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "eager",
    hooks: Qwen3VLBuildHooks | None = None,
    initialize_on_meta: bool = True,
) -> Qwen3VLPolicy:
    """Build a native Qwen3-VL in explicit lifecycle order.

    Order: native config/model construction -> model preparation -> FSDP/AC ->
    materialization -> embedding tie -> checkpoint load. A distributed build must
    inject the i4 parallelization hook; silently running an unsharded copy on each
    rank is forbidden.
    """
    validate_qwen3_vl_parallelism(parallel_dims)
    hooks = hooks or Qwen3VLBuildHooks()
    config_cls, model_cls = _native_qwen3_vl_classes(model_type)
    _override_native_attention_mask_compatibility(model_type)
    native_config = _coerce_native_config(config_cls, model_config, attn_implementation)
    if initialize_on_meta and not checkpoint_path and hooks.load_weights is None:
        raise ValueError("meta-initialized Qwen3-VL requires checkpoint_path or an injected load_weights hook")
    if parallel_dims is not None and hooks.parallelize is None:
        raise ValueError("distributed Qwen3-VL build requires an injected i4 FSDP/activation-checkpointing hook")

    if initialize_on_meta:
        from accelerate import init_on_device

        init_context = init_on_device("meta", include_buffers=False)
    else:
        init_context = nullcontext()
    with init_context:
        factory = hooks.model_factory or model_cls
        net = factory(native_config)
    if not isinstance(net, nn.Module):
        raise TypeError(f"Qwen3-VL model factory returned {type(net).__name__}, expected torch.nn.Module")
    _normalize_parameter_dtype(net, dtype)

    if hooks.prepare_model is not None:
        hooks.prepare_model(net)
    if hooks.parallelize is not None:
        hooks.parallelize(net, parallel_dims)
    (hooks.materialize or _default_materialize)(net)
    _tie_embeddings(net, native_config)
    if checkpoint_path:
        (hooks.load_weights or _default_load_weights)(net, checkpoint_path, credential_path, parallel_dims)
    elif hooks.load_weights is not None:
        hooks.load_weights(net, "", credential_path, parallel_dims)
    if model_type == "qwen3_vl_moe":
        from cosmos_framework.posttrain.model.qwen3_vl_moe import Qwen3VLMoePolicy

        return Qwen3VLMoePolicy(net)
    return Qwen3VLPolicy(net, model_type)


def build_qwen3_vl_native_policy(
    model_id: str,
    cfg: "Qwen3VLModelConfig",
    pdims: Any = None,
) -> Qwen3VLPolicy:
    """Build from the generic native seam's single policy/tokenizer identity.

    ``checkpoint_path`` may still name a local or object-store mirror of the same weights. The
    structural model identity, processor, rollout engine, and native config all use ``model_id``.
    """
    if cfg.name_or_path and cfg.name_or_path != model_id:
        raise ValueError(
            "NativePolicyBuildSpec.model_id and Qwen3VLModelConfig.name_or_path must match, got "
            f"{model_id!r} and {cfg.name_or_path!r}"
        )
    cfg.name_or_path = model_id
    return build_qwen3_vl_model(cfg, pdims)


@register_native_sft_source(f"{__name__}.{build_qwen3_vl_native_policy.__name__}")
def qwen3_vl_sft_source(spec: Any) -> tuple[str, bool]:
    """The SFT seat of this family (declared here, not in the entrypoint).

    ``spec.model_id`` is the one model/tokenizer artifact the SFT packer and :func:`build_qwen3_vl_native_policy` share;
    its ``config.json`` must name a ``model_type`` this module builds (``qwen3_vl`` or ``qwen3_vl_moe``). The metadata
    read is the same local/public ``config.json`` read the packer builder performs; no model or processor is constructed.
    """
    from transformers import PretrainedConfig

    trust_remote_code = bool(getattr(spec.model_config, "trust_remote_code", False))
    model_type = PretrainedConfig.get_config_dict(spec.model_id, trust_remote_code=trust_remote_code)[0].get(
        "model_type"
    )
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"SFT native seat requires model_type in {sorted(_SUPPORTED_MODEL_TYPES)}, got {model_type!r} for "
            f"{spec.model_id!r}"
        )
    return spec.model_id, trust_remote_code


def build_qwen3_vl_native_value_model(
    model_id: str,
    cfg: "Qwen3VLModelConfig",
    pdims: Any = None,
) -> nn.Module:
    """Build a Qwen3-VL backbone and attach the shared scalar value head."""
    from cosmos_framework.posttrain.model.value import DenseValueModel

    return DenseValueModel(build_qwen3_vl_native_policy(model_id, cfg, pdims), pdims=pdims)


def build_qwen3_vl_model(cfg: "Qwen3VLModelConfig", pdims: Any = None) -> Qwen3VLPolicy:
    """Build an i4-native dense/MoE Qwen3-VL policy with model-owned AC and FSDP setup."""
    if not cfg.name_or_path:
        raise ValueError("Qwen3VLModelConfig.name_or_path must be non-empty")
    validate_qwen3_vl_parallelism(pdims)

    from transformers import AutoConfig

    from cosmos_framework.configs.base.defaults.activation_checkpointing import (
        ActivationCheckpointingConfig,
    )
    from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
    from cosmos_framework.model.generator.parallelize_vlm import apply_ac, apply_fsdp

    # One artifact owns both architecture semantics and rollout identity. A separate config path
    # would let RoPE/vision defaults drift while parameter names and shapes still appeared valid.
    model_config = AutoConfig.from_pretrained(cfg.name_or_path)
    resolved_model_type = getattr(model_config, "model_type", None)
    if resolved_model_type not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(
            "Qwen3VLModelConfig.name_or_path must resolve to model_type qwen3_vl|qwen3_vl_moe before "
            f"meta/FSDP construction, got {resolved_model_type!r}"
        )
    if resolved_model_type == "qwen3_vl_moe" and cfg.dtype != "bfloat16":
        raise ValueError(
            "Qwen3VLModelConfig.dtype must be 'bfloat16' for qwen3_vl_moe; the native grouped-mm "
            f"expert kernel does not support {cfg.dtype!r}"
        )
    if bool(getattr(model_config, "tie_word_embeddings", False)):
        raise ValueError(
            "native Qwen3-VL OPD requires tie_word_embeddings=False: publishing one tied policy tensor "
            "into the rollout engine's live embedding/head namespace is not supported"
        )
    activation_checkpointing = ActivationCheckpointingConfig(mode=cfg.activation_checkpointing)
    parallelism_config = ParallelismConfig(
        data_parallel_shard_degree=1 if pdims is None else int(pdims.dp_shard),
        data_parallel_replicate_degree=1 if pdims is None else int(pdims.dp_replicate),
        context_parallel_shard_degree=1,
        cfg_parallel_shard_degree=1,
        fsdp_master_dtype=cfg.dtype,
        fsdp_reduce_dtype=cfg.fsdp_reduce_dtype,
    )

    def prepare_native(net: nn.Module) -> None:
        from cosmos_framework.posttrain.model.qwen3_vl_moe import override_qwen3_vl_moe_for_padded_batches

        backbone = getattr(net, "model", None)
        if backbone is None:
            raise TypeError(f"native Qwen3-VL {type(net).__name__} exposes no `.model` VLM backbone")
        # Every policy rank enters the vision FSDP units once per forward. The temporary override
        # uses a graph-connected empty image for text-only shards and real media otherwise.
        override_qwen3_vl_forward(backbone)
        if resolved_model_type == "qwen3_vl_moe":
            overridden = override_qwen3_vl_moe_for_padded_batches(net)
            if overridden < 1:
                raise ValueError("qwen3_vl_moe model contains no native sparse-MoE block to adapt")

    def parallelize_native(net: nn.Module, parallel_dims: Any) -> None:
        # `parallelize_vlm` reads `.model`, the disabled audio flag and (since i4 97c344f2be8, the
        # Qwen3.5 caption port) `.hf_config.model_type` to decide whether vision blocks stay in the
        # root FSDP unit. Keep the shim here so the generic i4 module does not depend on posttrain's
        # policy wrapper; `hf_config` is the same AutoConfig the policy was built from, so the
        # qwen3_vl|qwen3_vl_moe branch keeps the pre-97c344f2be8 wrapping exactly.
        class NativeVLMShim:
            model: nn.Module = net
            sound_und: bool = False
            hf_config: Any = model_config

        shim = NativeVLMShim()
        slots = apply_ac(shim, activation_checkpointing)
        if parallel_dims is not None:
            apply_fsdp(shim, slots, parallel_dims, parallelism_config, cfg.dtype)

    return build_qwen3_vl_policy(
        model_type=resolved_model_type,
        model_config=model_config,
        checkpoint_path=cfg.checkpoint_path or cfg.name_or_path,
        credential_path=cfg.credential_path or None,
        parallel_dims=pdims,
        dtype=getattr(torch, cfg.dtype),
        attn_implementation=cfg.attn_impl,
        hooks=Qwen3VLBuildHooks(prepare_model=prepare_native, parallelize=parallelize_native),
        initialize_on_meta=True,
    )


_WRAPPER_PREFIXES: tuple[str, ...] = ("_orig_mod.", "_checkpoint_wrapped_module.")
_UNSUPPORTED_MOE_STATE_PARTS: tuple[str, ...] = (
    ".shared_expert.",
    ".gate_noise.",
    ".cosine_router.",
    ".expert_bias",
)
_EMBEDDING_NAME: str = "model.language_model.embed_tokens.weight"
_LM_HEAD_NAME: str = "lm_head.weight"
# vLLM's routed-expert parameters, addressed on the module the checkpoint calls ``experts``
# (checkpoint-container form). Which live module owns them differs by vLLM version; the rollout,
# which owns the engine, resolves that layout (roles/rollout/vllm_weights.VllmMoeLayout). This
# module therefore never names a nesting such as ``routed_experts``.
_ROUTED_EXPERT_GATE_UP_PARAMETER: str = "w13_weight"
_ROUTED_EXPERT_DOWN_PARAMETER: str = "w2_weight"


@dataclass(frozen=True)
class WeightTensorSpec:
    """One tensor placed on the policy-to-rollout wire.

    ``engine_targets`` names the vLLM parameter(s) this source contributes to,
    in checkpoint-container form: packed dense names are live names, and routed
    expert names sit on the ``experts`` container until the rollout resolves
    them onto its live layout. Several source tensors may intentionally share
    one packed destination, distinguished by ``packed_part``.
    """

    source_name: str
    wire_name: str
    shape: tuple[int, ...]
    dtype: str
    engine_targets: tuple[str, ...]
    layout: str = "identity"
    packed_part: str | None = None


def canonical_qwen3_vl_weight_name(name: str) -> str:
    """Strip compile/checkpoint-wrapper components from one state-dict name."""
    canonical = name
    for prefix in _WRAPPER_PREFIXES:
        canonical = canonical.replace(prefix, "")
    return canonical


def _net(model: Any) -> Any:
    return getattr(model, "net", model)


def _canonical_state_dict(model: Any) -> dict[str, tuple[str, Any]]:
    """Return canonical name -> (original name, tensor), rejecting collisions."""
    state = _net(model).state_dict()
    canonical: dict[str, tuple[str, Any]] = {}
    for source_name, tensor in state.items():
        wire_name = canonical_qwen3_vl_weight_name(source_name)
        existing = canonical.get(wire_name)
        if existing is not None:
            raise ValueError(
                "Qwen3-VL weight-name normalization collision: "
                f"{existing[0]!r} and {source_name!r} both map to {wire_name!r}"
            )
        canonical[wire_name] = (source_name, tensor)
    return canonical


def _engine_name(wire_name: str) -> str:
    """Map one HF/i4 wire name to vLLM's live parameter namespace."""
    if wire_name.startswith("model.language_model."):
        return wire_name.replace("model.language_model.", "language_model.model.", 1)
    if wire_name.startswith("model.visual."):
        return wire_name.replace("model.visual.", "visual.", 1)
    if wire_name == _LM_HEAD_NAME:
        return "language_model.lm_head.weight"
    raise ValueError(
        f"unsupported Qwen3-VL weight {wire_name!r}: expected model.language_model.*, model.visual.*, or lm_head.weight"
    )


def _semantic_layout(wire_name: str) -> tuple[tuple[str, ...], str, str | None]:
    """Return ``(engine_targets, layout, packed_part)`` for one wire tensor."""
    engine_name = _engine_name(wire_name)

    attention_match = re.search(r"\.self_attn\.(q_proj|k_proj|v_proj)(\.(?:weight|bias))$", engine_name)
    if attention_match is not None:
        part = attention_match.group(1).removesuffix("_proj")
        target = re.sub(r"\.(?:q|k|v)_proj(\.(?:weight|bias))$", r".qkv_proj\1", engine_name)
        return (target,), "packed_qkv", part

    mlp_match = re.search(r"\.mlp\.(gate_proj|up_proj)(\.weight)$", engine_name)
    if mlp_match is not None:
        part = mlp_match.group(1).removesuffix("_proj")
        target = re.sub(r"\.(?:gate|up)_proj(\.weight)$", r".gate_up_proj\1", engine_name)
        return (target,), "packed_gate_up", part

    if ".mlp.experts.gate_up_proj" in engine_name:
        return (
            (engine_name.replace("experts.gate_up_proj", f"experts.{_ROUTED_EXPERT_GATE_UP_PARAMETER}"),),
            "moe_gate_up_fused",
            None,
        )
    if ".mlp.experts.down_proj" in engine_name:
        return (
            (engine_name.replace("experts.down_proj", f"experts.{_ROUTED_EXPERT_DOWN_PARAMETER}"),),
            "moe_down_fused",
            None,
        )

    return (engine_name,), "identity", None


def _validate_moe_tensor(
    *,
    wire_name: str,
    shape: tuple[int, ...],
    model_type: str,
    num_experts: int | None,
) -> None:
    is_expert = ".mlp.experts." in wire_name
    if is_expert and model_type != "qwen3_vl_moe":
        raise ValueError(f"dense qwen3_vl policy unexpectedly exposes MoE tensor {wire_name!r}")
    if not is_expert:
        return
    if len(shape) != 3:
        raise ValueError(f"fused MoE tensor {wire_name!r} must be rank 3, got shape {shape}")
    if num_experts is not None and shape[0] != num_experts:
        raise ValueError(f"fused MoE tensor {wire_name!r} has {shape[0]} experts, but config declares {num_experts}")
    if wire_name.endswith("experts.gate_up_proj") and shape[2] % 2:
        raise ValueError(f"fused MoE gate/up tensor {wire_name!r} must have even final dimension 2F, got shape {shape}")


class Qwen3VLWeightMapper:
    """Build and validate the semantic native-policy -> vLLM manifest."""

    def __init__(self, model_type: str, *, tie_word_embeddings: bool | None = None) -> None:
        if model_type not in _SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"unsupported Qwen3-VL model_type {model_type!r}; expected {sorted(_SUPPORTED_MODEL_TYPES)}"
            )
        self.model_type = model_type
        self.tie_word_embeddings = tie_word_embeddings
        self._manifest_owner_ref: weakref.ReferenceType[Any] | None = None
        self._manifest_cache: tuple[WeightTensorSpec, ...] = ()
        self._specs_by_wire: dict[str, WeightTensorSpec] = {}
        self._export_state: dict[str, torch.Tensor] | None = None

    @contextmanager
    def export_session(self, model: Any) -> Iterator[None]:
        """Reuse detached local shards during one quiescent, producer-driven publication.

        Rebuilding the whole state dict per bucket creates DTensor wrappers and can trigger long
        CPU garbage collections while other ranks wait in all-gather. Full tensors remain
        bucket-scoped; these references only retain the model's existing shard storage.
        """
        state = _net(model).state_dict()  # values: local tensors or sharded DTensors, no gather
        self._export_state = state
        try:
            yield
        finally:
            self._export_state = None
            # Clear the dictionary too: a failed export's retained traceback may still refer to it.
            state.clear()

    def manifest(self, model: Any) -> list[WeightTensorSpec]:
        """Return a sorted metadata-only manifest without materializing FSDP tensors."""
        net = _net(model)
        cached_owner = self._manifest_owner_ref() if self._manifest_owner_ref is not None else None
        if cached_owner is net:
            return list(self._manifest_cache)
        state = _canonical_state_dict(net)
        config = getattr(net, "config", None)
        text_config = getattr(config, "text_config", None) or config
        num_experts_raw = getattr(text_config, "num_experts", None)
        num_experts = int(num_experts_raw) if num_experts_raw not in (None, 0) else None
        tied = (
            bool(getattr(config, "tie_word_embeddings", False))
            if self.tie_word_embeddings is None
            else bool(self.tie_word_embeddings)
        )

        if tied:
            missing_alias_endpoints = [name for name in (_EMBEDDING_NAME, _LM_HEAD_NAME) if name not in state]
            if missing_alias_endpoints:
                raise ValueError(
                    f"tied Qwen3-VL embeddings require both state-dict aliases; missing {missing_alias_endpoints}"
                )
            embed_shape = tuple(int(dim) for dim in state[_EMBEDDING_NAME][1].shape)
            head_shape = tuple(int(dim) for dim in state[_LM_HEAD_NAME][1].shape)
            if embed_shape != head_shape:
                raise ValueError(f"tied embedding/head shapes differ: {embed_shape} vs {head_shape}")

        specs: list[WeightTensorSpec] = []
        for wire_name, (original_name, tensor) in sorted(state.items()):
            if tied and wire_name == _LM_HEAD_NAME:
                continue
            if self.model_type == "qwen3_vl_moe" and any(part in wire_name for part in _UNSUPPORTED_MOE_STATE_PARTS):
                raise ValueError(
                    f"native MoE policy tensor {wire_name!r} has no stock vLLM destination; "
                    "disable shared/noisy/cosine/loss-free router extensions for OPD"
                )

            shape = tuple(int(dim) for dim in tensor.shape)
            _validate_moe_tensor(
                wire_name=wire_name,
                shape=shape,
                model_type=self.model_type,
                num_experts=num_experts,
            )
            engine_targets, layout, packed_part = _semantic_layout(wire_name)
            if tied and wire_name == _EMBEDDING_NAME:
                engine_targets = (*engine_targets, _engine_name(_LM_HEAD_NAME))
            specs.append(
                WeightTensorSpec(
                    source_name=original_name,
                    wire_name=wire_name,
                    shape=shape,
                    dtype=str(tensor.dtype),
                    engine_targets=engine_targets,
                    layout=layout,
                    packed_part=packed_part,
                )
            )

        self.validate_manifest(specs)
        self._manifest_owner_ref = weakref.ref(net)
        self._manifest_cache = tuple(specs)
        self._specs_by_wire = {spec.wire_name: spec for spec in specs}
        return list(self._manifest_cache)

    def validate_manifest(self, specs: Iterable[WeightTensorSpec]) -> None:
        """Reject duplicate names and incomplete/ambiguous packed destinations."""
        materialized = list(specs)
        source_names = [spec.source_name for spec in materialized]
        wire_names = [spec.wire_name for spec in materialized]
        if len(source_names) != len(set(source_names)):
            raise ValueError("Qwen3-VL weight manifest contains duplicate source names")
        if len(wire_names) != len(set(wire_names)):
            raise ValueError("Qwen3-VL weight manifest contains duplicate wire names")

        by_target: dict[str, list[WeightTensorSpec]] = {}
        for spec in materialized:
            for target in spec.engine_targets:
                by_target.setdefault(target, []).append(spec)
        for target, contributors in by_target.items():
            layouts = {spec.layout for spec in contributors}
            parts = {spec.packed_part for spec in contributors}
            if layouts == {"packed_qkv"}:
                if parts == {"q", "k", "v"} and len(contributors) == 3:
                    continue
                raise ValueError(
                    f"vLLM packed QKV destination {target!r} requires exactly q/k/v contributors, got "
                    f"{[(spec.wire_name, spec.packed_part) for spec in contributors]}"
                )
            if layouts == {"packed_gate_up"}:
                if parts == {"gate", "up"} and len(contributors) == 2:
                    continue
                raise ValueError(
                    f"vLLM packed gate/up destination {target!r} requires exactly gate/up contributors, got "
                    f"{[(spec.wire_name, spec.packed_part) for spec in contributors]}"
                )
            if len(contributors) == 1:
                continue
            # A tied alias intentionally gives one source two targets, never two
            # sources one target, so it does not enter this branch.
            raise ValueError(
                f"vLLM destination {target!r} has ambiguous contributors "
                f"{[(spec.wire_name, spec.layout, spec.packed_part) for spec in contributors]}"
            )

    def wire_manifest(self, model: Any) -> list[tuple[str, tuple[int, ...], str]]:
        """Project the rich semantic manifest onto the generic transport tuple shape."""
        return [(spec.wire_name, spec.shape, spec.dtype) for spec in self.manifest(model)]

    def destination_names(self, model: Any) -> tuple[str, ...]:
        """Return the deduplicated post-packing vLLM destinations in checkpoint-container form.

        Dense/packed names are exact live names. Routed-expert names address ``w13_weight``/``w2_weight``
        on the ``experts`` container; the rollout resolves them onto its engine's live layout before
        comparing against the engine namespace.
        """
        specs = self.manifest(model)
        return tuple(sorted({target for spec in specs for target in spec.engine_targets}))

    def witness_spec(self, model: Any) -> WeightWitnessSpec:
        """Select the first language q-projection to prove packed-QKV TP slicing."""
        candidates = [
            spec
            for spec in self.manifest(model)
            if re.fullmatch(
                r"model\.language_model\.layers\.\d+\.self_attn\.q_proj\.weight",
                spec.wire_name,
            )
            and len(spec.shape) == 2
            and spec.layout == "packed_qkv"
            and spec.packed_part == "q"
            and len(spec.engine_targets) == 1
        ]
        if not candidates:
            raise ValueError("Qwen3-VL weight manifest has no language q_proj.weight witness for packed QKV")
        selected = min(candidates, key=lambda spec: spec.wire_name)
        return WeightWitnessSpec(
            wire_name=selected.wire_name,
            destination_name=selected.engine_targets[0],
            shape=selected.shape,
            dtype=selected.dtype,
            layout=selected.layout,
            packed_part=str(selected.packed_part),
        )

    def export_bucket(
        self,
        model: Any,
        bucket: Iterable[WeightTensorSpec | tuple[str, tuple[int, ...], str]],
    ) -> dict[str, torch.Tensor]:
        """Materialize one generic wire bucket while preserving every source tensor unchanged."""
        self.manifest(model)
        selected: list[WeightTensorSpec] = []
        for entry in bucket:
            if isinstance(entry, WeightTensorSpec):
                wire_name, shape, dtype = entry.wire_name, entry.shape, entry.dtype
            else:
                wire_name, shape, dtype = entry
            spec = self._specs_by_wire.get(wire_name)
            if spec is None:
                raise ValueError(f"wire bucket names unknown mapped Qwen3-VL tensor {wire_name!r}")
            normalized_shape = tuple(int(dim) for dim in shape)
            if normalized_shape != spec.shape or dtype != spec.dtype:
                raise ValueError(
                    f"wire metadata for {wire_name!r} changed from {(spec.shape, spec.dtype)!r} "
                    f"to {(normalized_shape, dtype)!r}"
                )
            selected.append(spec)
        return self._export_specs(model, selected)

    def export_state(self, model: Any) -> dict[str, torch.Tensor]:
        """Materialize the complete mapped state for the legacy whole-model batch API."""
        return self._export_specs(model, self.manifest(model))

    def _export_specs(self, model: Any, bucket: Iterable[WeightTensorSpec]) -> dict[str, torch.Tensor]:
        """Materialize already-validated semantic specs from canonical source tensors."""
        # Direct bucket/whole-model calls outside a publication still read fresh source tensors.
        state = self._export_state if self._export_state is not None else _net(model).state_dict()  # values: shards
        out: dict[str, torch.Tensor] = {}
        for spec in bucket:
            tensor = state.get(spec.source_name)
            if tensor is None:
                raise ValueError(f"mapped source tensor {spec.source_name!r} disappeared before export")
            full_tensor = getattr(tensor, "full_tensor", None)
            full = full_tensor() if callable(full_tensor) else tensor  # [*weight]
            exported = full.detach()  # [*weight]
            if tuple(int(dim) for dim in exported.shape) != spec.shape:
                raise ValueError(
                    f"mapped tensor {spec.wire_name!r} shape changed from {spec.shape} to {tuple(exported.shape)}"
                )
            if str(exported.dtype) != spec.dtype:
                raise ValueError(
                    f"mapped tensor {spec.wire_name!r} dtype changed from {spec.dtype} to {exported.dtype}"
                )
            if spec.wire_name in out:
                raise ValueError(f"duplicate mapped bucket output {spec.wire_name!r}")
            out[spec.wire_name] = exported
        return out
