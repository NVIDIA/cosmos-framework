# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""SigLIP2 vision encoder + PatchMerger for the Cosmos3-Edge (Nemotron 3 Dense VL) reasoner.

Vendored from the HF `nvidia/Cosmos3-Edge` remote code
(`modeling_nemotron_siglip2_h.py`): the custom grid_thw / cu_seqlens SigLIP2
vision transformer (naflex-style packed patches) and the Qwen3-style
`PatchMerger` projector, plus `patch_merging_by_param`. Kept byte-faithful so
outputs match the HF reference numerically.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Siglip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )

        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.FloatTensor`):
                Pixel values of shape (batch_size, max_num_patches, num_channels * patch_size * patch_size)
        """

        # Apply patch embeddings to already patchified pixel values
        patch_embeds = self.patch_embedding(pixel_values)

        return patch_embeds

class Siglip2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False
        self.num_key_value_groups = 1

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: total_pixel_value x hidden_size"""

        seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(seq_length, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        keys = keys.view(seq_length, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        values = values.view(seq_length, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        if self.config._attn_implementation == "flash_attention_2":
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                queries,
                keys,
                values,
                attention_mask=None,
                is_causal=self.is_causal,
                scaling=self.scale,
                dropout=0.0 if not self.training else self.dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (queries, keys, values)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scale,
                    dropout=0.0 if not self.training else self.dropout,
                    is_causal=self.is_causal,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output

class Siglip2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Siglip2EncoderLayer(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Siglip2Attention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class Siglip2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Siglip2EncoderLayer`].

    Args:
        config: Siglip2Config
    """

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Siglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.Tensor,
        **kwargs,
    ):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens,
                **kwargs,
            )

        return hidden_states

class Siglip2VisionTransformer(PreTrainedModel):
    config: Siglip2VisionConfig
    main_input_name = "pixel_values"
    base_model_prefix = "siglip_vit"
    supports_gradient_checkpointing = True

    _no_split_modules = [
        "Siglip2VisionEmbeddings",
        "Siglip2EncoderLayer",
        "Siglip2MultiheadAttentionPoolingHead",
    ]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    _can_record_outputs = {
        "hidden_states": Siglip2EncoderLayer,
        "attentions": Siglip2Attention,
    }
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.num_grid_per_side = self.embeddings.position_embedding_size
    
    def get_position_embedding(self, grid_thw: torch.Tensor) -> torch.Tensor:
        # prepare for interpolation
        positional_embedding = self.embeddings.position_embedding.weight.reshape(
            self.embeddings.position_embedding_size, self.embeddings.position_embedding_size, -1
        ).permute(2, 0, 1).unsqueeze(0)

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        embed_dim = self.embeddings.embed_dim
        # create a resized positional embedding of size (total_tokens, embed_size) to hold positional_embedding for all visual inputs
        resized_positional_embeddings = torch.empty((total_tokens, embed_dim), dtype=positional_embedding.dtype, device=grid_thw.device)
        offset = 0
        for t, height, width in grid_thw:
            resized_embeddings = F.interpolate(
                positional_embedding,
                size=(height.cpu().item(), width.cpu().item()),
                mode='bilinear',
                align_corners=False,
                antialias=True
            )
            resized_embeddings = resized_embeddings.reshape(embed_dim, -1).transpose(0, 1)
            
            num_spatial_tokens = height * width
            total_block_tokens = t * num_spatial_tokens

            resized_positional_embeddings[offset: offset + total_block_tokens] = resized_embeddings.repeat(t, 1)
            offset += total_block_tokens
        assert offset == resized_positional_embeddings.shape[0]
        return resized_positional_embeddings

    def get_position_embedding_fast_interpolation(self, grid_thw):
        print("wrong fast interpolation")
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = grid_thw.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.embeddings.position_embedding.weight.dtype, device=device)
        pos_embeds = self.embeddings.position_embedding(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thw: torch.LongTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        r"""
        spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
            Tensor containing the spatial dimensions (height, width) of the input images.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        hidden_states = self.embeddings(pixel_values)
        positional_embeddings = self.get_position_embedding(grid_thw)
        hidden_states = hidden_states + positional_embeddings

        # View pixel_value as packed multi-visual input, generate cu_seqlen her （migrate from qwen3-vl)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        hidden_states = self.encoder(
            inputs_embeds=hidden_states,
            cu_seqlens=cu_seqlens,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = self.post_layernorm(hidden_states)
        return last_hidden_state


class PatchMerger(nn.Module):

    def __init__(self, config: "ProjectorConfig", use_postshuffle_norm=False) -> None:
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.hidden_size = config.input_hidden_size * (self.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.input_hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, config.merger_intermedia)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(config.merger_intermedia, config.out_hidden_size)
        self.input_hidden_size = config.input_hidden_size
        self.out_hidden_size = config.out_hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x

    def init_weights(self):
        # init weight with he_normal
        # init bias with zero
        # init layernorm with standard gaussian distribution
        nn.init.kaiming_uniform_(self.linear_fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.linear_fc2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear_fc1.bias)
        nn.init.zeros_(self.linear_fc2.bias)
        self.norm.reset_parameters()



def patch_merging_by_param(image_embeds, image_grid_thw, merge_size=2):
    """
    image_embeds: [Total_Patches, C] -> 你的数据 [2008, 1152]
    image_grid_thw: [Num_Media, 3] -> 你的数据 [[1, 26, 38], [1, 34, 30]]
    merge_size: 来自 config 的参数，例如 2
    """
    new_embeds_list = []
    new_grid_thw_list = []
    curr_idx = 0

    C = image_embeds.shape[-1]

    for i in range(image_grid_thw.shape[0]):
        # 获取当前媒体的 T, H, W (这里的 H, W 是 patch 数量)
        t, h, w = image_grid_thw[i].tolist()
        num_patches = t * h * w

        # 1. 提取当前媒体特征 [T*H*W, C]
        media_seq = image_embeds[curr_idx : curr_idx + num_patches]
        curr_idx += num_patches

        # 2. 还原 3D 结构 [T, H, W, C]
        x = media_seq.view(t, h, w, C)

        # 3. 使用 einops 进行空间合并 (2x2 空间块合并)
        # 维度变换逻辑：
        # b=t, h=(h'/ms * ms), w=(w'/ms * ms)
        # -> [t, h/ms, ms, w/ms, ms, c] -> [t, h/ms, w/ms, (ms*ms*c)]
        # 注意：这里我们遵循 Qwen2-VL 的顺序：h1 w1 拼接在 C 之前
        x = rearrange(
            x, 
            't (h h1) (w w1) c -> t h w (h1 w1 c)', 
            h1=merge_size, 
            w1=merge_size
        )

        # 4. 展平回序列 [T * (H/ms) * (W/ms), C * ms^2]
        new_embeds_list.append(x.reshape(-1, x.shape[-1]))

        # 5. 更新 grid 信息: T 不变, H 和 W 缩减
        new_grid_thw_list.append([t, h // merge_size, w // merge_size])

    # 重新拼接所有媒体数据
    image_embeds_merged = torch.cat(new_embeds_list, dim=0)
    image_grid_thw_merged = torch.tensor(new_grid_thw_list, device=image_grid_thw.device)

    return image_embeds_merged, image_grid_thw_merged


class NemotronSiglip2VisionEncoder(nn.Module):
    """Vision tower: SigLIP2 transformer + PatchMerger. Mirrors HF get_image_features."""
    def __init__(self, vision_config: Siglip2VisionConfig, projector_config):
        super().__init__()
        self.spatial_merge_size = projector_config.spatial_merge_size
        setattr(vision_config, "spatial_merge_size", self.spatial_merge_size)
        self.visual = Siglip2VisionTransformer(vision_config)
        self.projector = PatchMerger(projector_config)

    @property
    def dtype(self):
        return self.visual.post_layernorm.weight.dtype

    def get_image_features(self, pixel_values, image_grid_thw):
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds, image_grid_thw = patch_merging_by_param(
            image_embeds, image_grid_thw, merge_size=self.projector.spatial_merge_size)
        image_embeds = image_embeds.view(-1, self.projector.spatial_merge_size**2, self.projector.input_hidden_size)
        projected = self.projector(image_embeds)
        return projected
