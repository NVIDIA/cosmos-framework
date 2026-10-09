# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Reusable same-instant RGB attention for multiview action sequences."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from cosmos_framework.model.attention import attention
from cosmos_framework.model.generator.mot.merge_bridge import MergeAttentionsBridge


@dataclass(frozen=True)
class MultiviewActionAttentionPlan:
    """Frame-major RGB gathers from camera-major [actions, RGB] supertokens."""

    rgb_indexes: torch.Tensor  # [N_RGB]; empty for a single camera
    offsets: torch.Tensor  # [T+1] int32
    tokens_per_instant: int
    gen_tokens: int


def build_multiview_action_attention_plan(
    vision_token_shapes: Sequence[tuple[int, int, int]],
    num_action_tokens: int,
    device: torch.device,
) -> MultiviewActionAttentionPlan:
    """Build one varlen segment per synchronized instant; actions never enter it.

    The plan only describes the packed action/RGB layout, so both teacher
    forcing and self-forcing callers can reuse it.
    """
    if not vision_token_shapes:
        raise ValueError("Multiview action attention requires at least one camera.")
    frames = vision_token_shapes[0][0]
    if any(shape[0] != frames for shape in vision_token_shapes):
        raise ValueError("Multiview action attention requires synchronized camera lengths.")
    if num_action_tokens < 1:
        raise ValueError("Multiview action attention requires RGB/action supertokens.")
    per_view: list[torch.Tensor] = []  # each [T,S_RGB_view]
    gen_tokens = 0
    for _, height, width in vision_token_shapes:
        spatial = height * width
        stride = num_action_tokens + spatial
        per_view.append(
            gen_tokens + torch.arange(frames)[:, None] * stride + num_action_tokens + torch.arange(spatial)[None, :]
        )  # [T,S_RGB_view]
        gen_tokens += frames * stride
    tokens_per_instant = sum(shape[1] * shape[2] for shape in vision_token_shapes)
    rgb_indexes = (
        torch.cat(per_view, dim=1).flatten().to(device=device)
        if len(vision_token_shapes) > 1
        else torch.empty(0, dtype=torch.long, device=device)
    )  # [N_RGB]
    offsets = torch.arange(frames + 1, dtype=torch.int32, device=device) * tokens_per_instant  # [T+1]
    return MultiviewActionAttentionPlan(rgb_indexes, offsets, tokens_per_instant, gen_tokens)


def same_instant_multiview_rgb_attention(
    query: torch.Tensor,  # [N_gen_padded,H,D]
    key: torch.Tensor,  # [N_gen_padded,H_kv,D]
    value: torch.Tensor,  # [N_gen_padded,H_kv,D]
    plan: MultiviewActionAttentionPlan,
) -> tuple[torch.Tensor, torch.Tensor]:  # [1,N_gen,H,D], [1,N_gen,H]
    """Attend live RGB at the same instant, retaining the merge backward contract.

    A view's own RGB participates in both this pass and its three-way pass,
    matching the bidirectional decomposition's overlap. Callers skip this
    component entirely for V=1. Clean history stays in the per-view TF path.
    """
    indexes = plan.rgb_indexes
    out, lse = attention(
        query[indexes].unsqueeze(0),  # [1,N_RGB,H,D]
        key[indexes].unsqueeze(0),  # [1,N_RGB,H_kv,D]
        value[indexes].unsqueeze(0),  # [1,N_RGB,H_kv,D]
        cumulative_seqlen_Q=plan.offsets,
        cumulative_seqlen_KV=plan.offsets,
        max_seqlen_Q=plan.tokens_per_instant,
        max_seqlen_KV=plan.tokens_per_instant,
        backend="natten",
        return_lse=True,
    )  # [1,N_RGB,H,D], [1,N_RGB,H]

    def scatter(
        output: torch.Tensor, normalizer: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:  # [1,N_RGB,H,D], [1,N_RGB,H] -> full GEN
        full_output = output.new_zeros(1, plan.gen_tokens, output.shape[2], output.shape[3])  # [1,N_gen,H,D]
        full_lse = normalizer.new_full(
            (1, plan.gen_tokens, normalizer.shape[2]), torch.finfo(normalizer.dtype).min
        )  # [1,N_gen,H]
        full_output[:, indexes] = output  # [1,N_gen,H,D]
        full_lse[:, indexes] = normalizer  # [1,N_gen,H]
        return full_output, full_lse

    def gather(
        output: torch.Tensor, normalizer: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:  # full GEN -> [1,N_RGB,H,D], [1,N_RGB,H]
        return output[:, indexes], normalizer[:, indexes]  # [1,N_RGB,H,D], [1,N_RGB,H]

    return MergeAttentionsBridge.apply(out, lse, scatter, gather)  # [1,N_gen,H,D], [1,N_gen,H]
