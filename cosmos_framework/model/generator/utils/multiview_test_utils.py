# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared sim-auto numerical oracles and tiny replay fixtures; no collected tests."""

from __future__ import annotations

from dataclasses import replace

import torch

from cosmos_framework.configs.base.defaults.replay_attention import TeacherForcingReplayPolicyConfig
from cosmos_framework.model.generator.mot.causal_flex_attention import (
    TeacherForcingFlexMetadata,
    build_teacher_forcing_multiview_flex_metadata,
)
from cosmos_framework.model.generator.mot.flex_attention import CaptionMaskItem, SensorMaskItem
from cosmos_framework.model.generator.mot.maskless_attention import ReplayMasklessPlan
from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb
from cosmos_framework.model.generator.utils.rolling_kv.rolling_prompt import (
    RollingPromptSchedule,
    RollingPromptSegment,
)
from cosmos_framework.model.generator.utils.rolling_kv.rolling_replay import (
    RollingReplayRequest,
    build_rolling_replay_metadata,
)

SPECIAL: dict[str, int] = {"bos_token_id": 1, "eos_token_id": 2, "start_of_generation": 3, "end_of_generation": 4}


def _gathered_weights(plan: ReplayMasklessPlan) -> dict[str, torch.Tensor]:
    """Expand each test-sized gathered pass to the (query, key) edge multiplicities it addresses."""
    result: dict[str, torch.Tensor] = {}
    for partition in plan.passes:
        if partition.name not in result:
            result[partition.name] = torch.zeros(plan.q_len, plan.kv_len, dtype=torch.int64)  # [Q,KV]
        weights = result[partition.name]  # [Q,KV]
        q_offsets = partition.q_offsets
        kv_offsets = partition.kv_offsets
        for index in range(len(q_offsets) - 1):
            q = partition.q_gather[q_offsets[index] : q_offsets[index + 1]].cpu()  # [Q_group]
            kv = partition.kv_gather[kv_offsets[index] : kv_offsets[index + 1]].cpu()  # [KV_group]
            weights[q[:, None], kv[None, :]] += 1  # [Q_group,KV_group]
    return result


def _pass_weights(plan: ReplayMasklessPlan) -> dict[str, torch.Tensor]:
    """Edge multiplicities per partition; ``same_view`` counts the gapped segments."""
    result = _gathered_weights(plan)
    weights = torch.zeros(plan.q_len, plan.kv_len, dtype=torch.int64)  # [Q,KV]
    for component in plan.gapped.components:
        for launch in component.launches:
            for segment in launch.segments:
                weights[
                    segment.q_row : segment.q_row + segment.q_len, segment.k_row : segment.k_row + segment.k_len
                ] += 1
    return {"same_view": weights, **result}


def _dense_reference(
    q: torch.Tensor,  # [1,Q,H,D]
    k: torch.Tensor,  # [1,KV,H_KV,D]
    v: torch.Tensor,  # [1,KV,H_KV,D]
    weights: torch.Tensor,  # [Q,KV] edge multiplicities
    real: torch.Tensor,  # [Q]
) -> torch.Tensor:  # [1,Q,H,D]
    """FP32 dense attention whose softmax counts each edge with its multiplicity."""
    k = k.repeat_interleave(q.shape[2] // k.shape[2], dim=2)  # [1,KV,H,D]
    v = v.repeat_interleave(q.shape[2] // v.shape[2], dim=2)  # [1,KV,H,D]
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / q.shape[-1] ** 0.5  # [1,H,Q,KV]
    scores = scores + weights.float().log()[None, None]  # [1,H,Q,KV]
    # Gapped plans give padding queries no keys at all; keep their zeroed rows and gradients finite.
    empty = ~weights.bool().any(-1)[None, None, :, None]  # [1,1,Q,1]
    probabilities = scores.masked_fill(empty, 0.0).softmax(-1).masked_fill(empty, 0.0)  # [1,H,Q,KV]
    output = torch.einsum("bhqk,bkhd->bqhd", probabilities, v.float())  # [1,Q,H,D]
    return torch.where(real[None, :, None, None], output, 0)  # [1,Q,H,D]


def _items(rgb_frames: int, lidar_frames: int, *, initial: bool) -> list[list[SensorMaskItem]]:
    result: list[SensorMaskItem] = []
    for count, views, view_offset, period, captions in (
        (rgb_frames, 2, 0, 4 / 30, "camera"),
        (lidar_frames, 1, 2, 0.1, "all_captions"),
    ):
        for control in (True, False):
            mask = torch.full((views * count,), float(control))  # [V*T]
            if initial and not control:
                mask[::count] = 1  # [V]
            result.append(
                SensorMaskItem(
                    token_shape=(views * count, 1, 2),
                    condition_mask=mask,
                    num_views=views,
                    view_offset=view_offset,
                    is_control=control,
                    seconds_per_frame=period,
                    caption_access=captions,
                )
            )
    return [result]


def _metadata(
    items: list[list[SensorMaskItem]],
    request: RollingReplayRequest | None,
    *,
    pass_kind: str,
    window: float | None,
    clean_causality: str = "chunk",
) -> TeacherForcingFlexMetadata:
    count = sum(item.latent_t * item.spatial_tokens for item in items[0])
    policy = TeacherForcingReplayPolicyConfig(
        control_visibility="causal",
        controls_read_strict_past_clean_rgb=True,
        clean_pass_causality=clean_causality,
        multiview_attention_scope="same_view",
        decomposed_temporal_window_seconds=window,
    )
    common = dict(
        seq_len=count + 2,
        full_q_offsets=torch.tensor([0, count], dtype=torch.int32),  # [2]
        sensor_mask_items=items,
        caption_mask_items=[[CaptionMaskItem(view_id=0, num_tokens=2), CaptionMaskItem(view_id=1, num_tokens=2)]],
        device=torch.device("cpu"),
        num_und=4,
        causal_offsets=torch.tensor([0, 4], dtype=torch.int32),  # [2]
    )
    if request is not None:
        metadata = build_rolling_replay_metadata(request=request, policy=policy, **common)
    else:
        clean_count = sum(
            int((~item.condition_mask.bool()).sum()) * item.spatial_tokens for item in items[0] if not item.is_control
        )
        metadata = build_teacher_forcing_multiview_flex_metadata(
            **common,
            pass_kind=pass_kind,
            frames_per_chunk=3,
            teacher_forcing_replay_policy=policy,
            clean_memory_seq_len=clean_count if pass_kind == "noisy" else 0,
        )
    policy.multiview_attention_scope = "decomposed"
    return replace(metadata, teacher_forcing_replay_policy=policy)


def _schedule(
    *, num_frames: int = 2000, break_frame: int = 600, sink: int = 2, before: int = 0, after: int = 0
) -> RollingPromptSchedule:
    return RollingPromptSchedule(
        fps=30,
        num_frames=num_frames,
        sink_tokens=sink,
        special_tokens=SPECIAL,
        context_before=before,
        context_after=after,
        views=(
            (
                RollingPromptSegment(0, 6, (11, 12, 13)),
                RollingPromptSegment(6, break_frame, (21, 22, 23, 24, 25)),
                RollingPromptSegment(break_frame, 2 * break_frame, (51, 52)),
                RollingPromptSegment(2 * break_frame, num_frames, (61,)),
            ),
            (
                RollingPromptSegment(0, 9, (31, 32, 33, 34)),
                RollingPromptSegment(9, break_frame, (41, 42, 43)),
                RollingPromptSegment(break_frame, 2 * break_frame, (71,)),
                RollingPromptSegment(2 * break_frame, num_frames, (81, 82)),
            ),
        ),
    )


def _rotary(
    device: torch.device = torch.device("cpu"), *, scaling: float = 1.0, head_dim: int = 128
) -> Qwen3VLTextRotaryEmbedding:
    config = Qwen3VLTextConfig(
        hidden_size=2 * head_dim,
        intermediate_size=4 * head_dim,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=head_dim,
        rope_theta=5000000.0,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [24, 20, 20] if head_dim == 128 else [head_dim // 2, 0, 0],
            "mrope_interleaved": True,
        },
    )
    rotary = Qwen3VLTextRotaryEmbedding(config).to(device)
    rotary.attention_scaling = scaling
    return rotary


def _native_rotate(
    rotary: Qwen3VLTextRotaryEmbedding,
    keys: torch.Tensor,  # [1,M,H,D]
    positions: torch.Tensor,  # [3,1,M]
) -> torch.Tensor:  # [1,M,H,D]
    cos, sin = rotary(keys, positions)  # each [1,M,D]
    _, keys_ = apply_rotary_pos_emb(keys, keys, cos, sin, unsqueeze_dim=2)  # each [1,M,H,D]
    return keys_  # [1,M,H,D]
