# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Framework-native Cosmos3-Edge VLM modeling.

Vendored/ported from the HF remote code ``modeling_nemotron_siglip2_h.py`` (old
``nvidia/Cosmos3-Edge``), byte-faithful for the dense path; Mamba/MoE, hybrid-cache
and generation paths are dropped (the shipped config is pure dense; training runs
with ``use_cache=False``). Logits are cast to fp32 to match the reference.

State-dict keys are contractual (safetensors loader, freeze regex
``model\\.visual\\.``, lr multipliers, HF export): ``model.visual.*``,
``model.projector.*``, ``model.language_model.{embeddings,layers.{0..55},norm_f}.*``,
``lm_head.weight``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import ModelOutput, logging

from cosmos_framework.model.generator.reasoner.cosmos3_edge.configuration_cosmos3_edge import (
    Cosmos3EdgeConfig,
    Cosmos3EdgeTextConfig,
)
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.nemotron_3_dense_vl import (
    MultiModalRotaryEmbedding,
    apply_rotary_pos_emb_partial,
)
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.vision_siglip2 import (
    PatchMerger,
    Siglip2VisionTransformer,
    eager_attention_forward,
    patch_merging_by_param,
)

logger = logging.get_logger(__name__)


class Cosmos3EdgeRMSNorm(nn.Module):
    """Port of NemotronHRMSNorm (fp32 compute, weight applied in fp32)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)


class Cosmos3EdgeTextAttention(nn.Module):
    """Port of NemotronHAttention (GQA + partial mRoPE), KV-cache path dropped."""

    def __init__(self, config: Cosmos3EdgeTextConfig, layer_idx: Optional[int] = None) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.head_dim * self.num_heads, self.hidden_size, bias=config.attention_bias)
        self.scaling = self.head_dim**-0.5
        self.sliding_window = getattr(config, "sliding_window", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packing_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb_partial(query_states, key_states, cos, sin)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if packing_args is not None:
            cu_seqlens = packing_args["cu_seqlens"]
            max_seqlen = packing_args["max_seqlen_in_batch"]
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Cosmos3EdgeMLP(nn.Module):
    """Port of NemotronHMLP."""

    def __init__(self, config: Cosmos3EdgeTextConfig, layer_idx: Optional[int] = None) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(x)))


class Cosmos3EdgeBlock(nn.Module):
    """Port of NemotronHBlock: shared pre-norm + attention/MLP mixer (dense pattern only)."""

    def __init__(self, config: Cosmos3EdgeTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = Cosmos3EdgeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.block_type = config.layers_block_type[layer_idx]
        if self.block_type == "attention":
            self.mixer = Cosmos3EdgeTextAttention(config, layer_idx=layer_idx)
        elif self.block_type == "mlp":
            self.mixer = Cosmos3EdgeMLP(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Invalid block type {self.block_type!r} at layer {layer_idx} (dense-only port)")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        packing_args: Optional[dict] = None,
    ) -> torch.Tensor:
        # The remote code pins the block to the CUDA default stream ("avoid NaN issues
        # when using multiple GPUs"); nullcontext off-CUDA so the port also runs on CPU.
        if hidden_states.device.type == "cuda":
            stream_ctx = torch.cuda.stream(torch.cuda.default_stream(hidden_states.device))
        else:
            stream_ctx = contextlib.nullcontext()
        with stream_ctx:
            residual = hidden_states
            hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

            if self.block_type == "attention":
                hidden_states, _ = self.mixer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    packing_args=packing_args,
                )
            else:
                hidden_states = self.mixer(hidden_states, padding_mask=padding_mask)

            hidden_states = residual + hidden_states
            return hidden_states


class Cosmos3EdgePreTrainedModel(PreTrainedModel):
    config_class = Cosmos3EdgeConfig
    base_model_prefix = "model"
    input_modalities = ["image", "text"]
    # Text block + vendored vision encoder layer: parallelize_vlm's FSDP block
    # collector matches these type names.
    _no_split_modules = ["Cosmos3EdgeBlock", "Siglip2EncoderLayer"]
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True


@dataclass
class Cosmos3EdgeModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class Cosmos3EdgeCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class Cosmos3EdgeTextModel(Cosmos3EdgePreTrainedModel):
    """Port of NemotronHModel: embeddings + 56 paired blocks + norm_f, mRoPE."""

    config_class = Cosmos3EdgeTextConfig

    def __init__(self, config: Cosmos3EdgeTextConfig) -> None:
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # 2 blocks per native paired layer: even index = attention, odd = MLP.
        self.layers = nn.ModuleList(
            [Cosmos3EdgeBlock(config, layer_idx=idx) for idx in range(2 * config.num_hidden_layers)]
        )
        # Old config.json pinned enable_rope=true; rope is unconditional here.
        self.rotary_emb = MultiModalRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.norm_f = Cosmos3EdgeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        # Legacy-checkpoint tolerance carried over from the remote code.
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,  # accepted for API parity; no cache support
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packing_args: Optional[dict] = None,
        **kwargs: Any,  # swallows visual_pos_masks etc., like the remote code
    ) -> Union[Tuple, Cosmos3EdgeModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        hidden_states = inputs_embeds

        if cache_position is None:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        # The hard coded `3` is for temporal, height and width (mRoPE).
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for mixer_block in self.layers:
            layer_mask = causal_mask if mixer_block.block_type == "attention" else None

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    mixer_block.__call__, hidden_states, layer_mask, position_embeddings, padding_mask, packing_args
                )
            else:
                hidden_states = mixer_block(
                    hidden_states,
                    attention_mask=layer_mask,
                    position_embeddings=position_embeddings,
                    padding_mask=padding_mask,
                    packing_args=packing_args,
                )

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return Cosmos3EdgeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Port of NemotronHModel._update_causal_mask (jamba lineage), cache-free.
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = cache_position[-1] + 1
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows (left padding), required by
            # SDPA's memory-efficient path. See pytorch/pytorch#110213.
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class Cosmos3EdgeModel(Cosmos3EdgePreTrainedModel):
    """Port of NemotronSiglip2Model: visual + projector + language_model."""

    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    # NOTE: explicit config_class, not a `config:` annotation — with
    # `from __future__ import annotations` PreTrainedModel.__init_subclass__ would
    # promote the (string) annotation to config_class and break Auto registration.
    config_class = Cosmos3EdgeConfig

    def __init__(self, config: Cosmos3EdgeConfig) -> None:
        super().__init__(config)
        # Vision pos_embed interpolation reads the merge size off the vision config.
        setattr(config.vision_config, "spatial_merge_size", config.projector_config.spatial_merge_size)
        self.visual = Siglip2VisionTransformer._from_config(config.vision_config)
        self.projector = PatchMerger(config.projector_config)
        self.language_model = Cosmos3EdgeTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Timestamp-based mRoPE (Qwen3VL-style, but every video frame is its own t=1 grid)."""
        # Videos are interleaved with per-frame timestamp tokens, so split grid rows per frame.
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        spatial_merge_size = self.config.projector_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None:
            if cu_seqlens is None:
                total_input_ids = input_ids
                if attention_mask is None:
                    total_attention_mask = torch.ones_like(total_input_ids)
                else:
                    total_attention_mask = attention_mask
                total_attention_mask = total_attention_mask.to(total_input_ids.device)
            else:
                varlen = cu_seqlens.cpu().tolist()
                total_input_ids = []
                total_attention_mask = []
                for start, end in zip(varlen[:-1], varlen[1:]):
                    total_input_ids.append(input_ids[:, start:end])
                    total_attention_mask.append(torch.ones_like(total_input_ids[-1]).to(input_ids.device))
            position_ids = []
            image_index, video_index = 0, 0
            for input_ids, attention_mask in zip(total_input_ids, total_attention_mask):
                input_ids = input_ids[attention_mask == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    # t_index is always 0 (llm_grid_t == 1; timestamps carry temporal info).
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids.append(llm_positions.to(input_ids.device))
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            if cu_seqlens is None:
                # Batching on second dim (batch) for batch seq
                position_ids = pad_sequence(position_ids, batch_first=False, padding_value=1)
            else:
                # Concat on last dim (seq_len) for packing seq
                position_ids = torch.cat(position_ids, dim=-1)
            return position_ids, mrope_position_deltas
        else:
            raise ValueError("input_ids is None")

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        num_image: int = 0,
    ):
        """Encode packed patches, 2x2-merge, project; split back into per-media features."""
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds, image_grid_thw = patch_merging_by_param(
            image_embeds, image_grid_thw, merge_size=self.projector.spatial_merge_size
        )
        image_embeds = image_embeds.view(-1, self.projector.spatial_merge_size**2, self.projector.input_hidden_size)
        projected_hidden_states = self.projector(image_embeds)
        split_sizes = image_grid_thw.prod(-1).tolist()
        image_embeds = torch.split(projected_hidden_states, split_sizes)
        image_embeddings = image_embeds[:num_image]
        video_embeddings = image_embeds[num_image:]
        return image_embeddings, video_embeddings

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        packing_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> Union[tuple, Cosmos3EdgeModelOutput]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Remote code hardcodes padding_mask=None (config/tokenizer pad-id mismatch upstream).
        padding_mask = None

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        skip_visual_embedding = False
        if pixel_values is None and pixel_values_videos is None:
            skip_visual_embedding = True
        elif pixel_values is None:
            final_pixel_value = pixel_values_videos
            final_thw = video_grid_thw
            num_image = 0
        elif pixel_values_videos is None:
            final_pixel_value = pixel_values
            final_thw = image_grid_thw
            num_image = image_grid_thw.shape[0]
        else:
            final_pixel_value = torch.cat([pixel_values, pixel_values_videos], dim=0)
            final_thw = torch.cat([image_grid_thw, video_grid_thw], dim=0)
            num_image = image_grid_thw.shape[0]

        if not skip_visual_embedding:
            image_embeds, video_embeds = self.get_image_features(final_pixel_value, final_thw, num_image)
        elif skip_visual_embedding and self.training:
            # Dummy forward to keep FSDP all-gather synchronised across ranks
            merge_size = self.projector.spatial_merge_size
            dummy_height = merge_size
            dummy_width = merge_size
            channels = self.visual.config.num_channels * self.visual.config.patch_size**2
            final_pixel_value = torch.zeros(
                dummy_height * dummy_width, channels, device=inputs_embeds.device, dtype=self.visual.dtype
            )
            final_thw = torch.tensor([[1, dummy_height, dummy_width]], device=inputs_embeds.device)
            num_image = 1
            image_embeds, video_embeds = self.get_image_features(final_pixel_value, final_thw, num_image)
            image_embeds = [image_embed[0:0] for image_embed in image_embeds]
        else:
            image_embeds = None
            video_embeds = None

        if image_embeds is not None and len(image_embeds) > 0:
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if video_embeds is not None and len(video_embeds) > 0:
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None and self.config.text_config.enable_mrope:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # No KV cache in this port: every forward is a prefill, so mRoPE position
            # ids are always recomputed (rope_deltas kept for parity/inspection).
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas

        # position_ids: [3, bsz, seq_len]; for visual tokens the dim-0 order is t, h, w.
        return self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            padding_mask=padding_mask,
            cache_position=cache_position,
            use_cache=use_cache,
            packing_args=packing_args,
            **kwargs,
        )


class Cosmos3EdgeForConditionalGeneration(Cosmos3EdgePreTrainedModel):
    """Port of NemotronSiglip2ForConditionCausalLM, training/eval forward only."""

    _tied_weights_keys = ["lm_head.weight"]
    config_class = Cosmos3EdgeConfig

    def __init__(self, config: Cosmos3EdgeConfig) -> None:
        super().__init__(config)
        self.model = Cosmos3EdgeModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.model.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def multi_modal_projector(self):
        return self.model.projector

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        valid_input_len: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Union[Tuple, Cosmos3EdgeCausalLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Sequence packing: valid_input_len drives the varlen (cu_seqlens) attention path.
        packing_args = None
        if valid_input_len is not None:
            batch_size = valid_input_len.shape[0]
            input_ids_list = []
            for i in range(batch_size):
                valid_len = valid_input_len[i].item()
                cur_input_ids = input_ids[i : i + 1, :valid_len].clone()
                input_ids_list.append(cur_input_ids)
            cu_seqlens = torch.cumsum(valid_input_len, dim=0).to(torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
            max_seqlen_in_batch = torch.max(valid_input_len).cpu().item()

            packing_args = {
                "cu_seqlens": cu_seqlens,
                "max_seqlen_in_batch": max_seqlen_in_batch,
            }
            input_ids = torch.cat(input_ids_list, dim=1)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            use_cache=use_cache,
            packing_args=packing_args,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Cosmos3EdgeCausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Cosmos3EdgeForConditionalGeneration",
    "Cosmos3EdgeModel",
    "Cosmos3EdgePreTrainedModel",
    "Cosmos3EdgeTextModel",
]
