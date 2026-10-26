# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Real robot packs use the shared multiview attention metadata and configuration."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from cosmos_framework.configs.base.defaults.model_config import MultiviewActionConditioningConfig
from cosmos_framework.configs.base.defaults.multiview_attention import (
    CrossViewTemporalWindowConfig,
    MultiviewAttentionConfig,
    MultiviewAttentionMaskConfig,
    SameViewTemporalWindowConfig,
)
from cosmos_framework.model.generator.mot.attention import SplitInfo
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import (
    Cosmos3VFMNetwork,
    _multiview_sensor_mask_items,
)
from cosmos_framework.model.generator.mot.multiview_maskless_attention import (
    MultiviewMasklessPlan,
    cross_view_passes,
)
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.data.generator.sequence_packing.packers import (
    pack_multiview_action_conditioning,
    replicate_multiview_actions,
)
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence, SequencePlan


def _robot_pack(*, frames: tuple[int, ...] = (3,), num_views: int = 3) -> PackedSequence:
    grids = [(1, 2), (1, 1), (2, 1)][:num_views]
    views = [torch.zeros(1, 2, count, h, w) for count in frames for h, w in grids]  # list[[1,2,T,H,W]]
    actions = [
        torch.arange((count - 1) * 16, dtype=torch.float32).reshape((count - 1) * 4, 4) for count in frames
    ]  # list[[4*(T-1),4]]
    fps = torch.full((len(frames),), 20.0)  # [B]
    data = GenerationDataClean(
        batch_size=len(frames),
        is_image_batch=False,
        x0_tokens_vision=views,
        x0_tokens_action=actions,
        fps_vision=fps,
        num_vision_items_per_sample=[num_views] * len(frames),
    )
    data = replicate_multiview_actions(data, num_views=num_views, temporal_factor=4, action_dim=4)
    plans = [
        SequencePlan(
            has_text=True,
            has_vision=True,
            has_action=True,
            condition_frame_indexes_vision=[0],
            condition_frame_indexes_action=list(range((count - 1) * 4)),
        )
        for count in frames
    ]
    timesteps = torch.full((len(frames),), 0.5)  # [B]
    return pack_multiview_action_conditioning(
        plans,
        [[3, 4] for _ in frames],
        data,
        timesteps,
        config=MultiviewActionConditioningConfig(
            view_code_start=2, view_codes=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]][:num_views]
        ),
        special_tokens={"eos_token_id": 0, "start_of_generation": 1, "end_of_generation": 2},
        patch_spatial=1,
        temporal_factor=4,
        temporal_margin=15_000,
        base_fps=24.0,
    )


def _no_rigrope_heads(_device: torch.device) -> None:
    return None


def _prepare(
    packed: PackedSequence,
    *,
    mask: MultiviewAttentionMaskConfig | None = None,
) -> MultiviewMasklessPlan:
    # Only the pack identifies the robot layout. No network-level action switch or
    # temporal compression factor is available for the attention path to infer it from.
    network = SimpleNamespace(
        multiview_backend="maskless",
        natten_parameter_list=None,
        rigrope=None,
        _rigrope_head_masks=_no_rigrope_heads,
        config=SimpleNamespace(
            multiview_attention_config=MultiviewAttentionConfig(
                backend="maskless",
                mask=mask or MultiviewAttentionMaskConfig(attention_scope="decomposed"),
            )
        ),
    )
    attention_meta = SplitInfo(
        packed.split_lens, packed.attn_modes, packed.sample_lens, actual_len=packed.sequence_length
    )
    gen_tokens = sum(length for length, mode in zip(packed.split_lens, packed.attn_modes) if mode == "full")
    full_only_seq = torch.zeros(gen_tokens + 3, 1)  # [N_gen+3,1]
    Cosmos3VFMNetwork._prepare_multiview_attention(
        cast(Cosmos3VFMNetwork, network), packed, {"full_only_seq": full_only_seq}, attention_meta
    )
    assert attention_meta.flex_block_mask is None
    assert attention_meta.multiview_maskless is not None
    return attention_meta.multiview_maskless


def _cross_edges(plan: MultiviewMasklessPlan) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for cross_pass in cross_view_passes(plan):
        queries = cross_pass.gather.tolist()
        keys = cross_pass.kv_gather.tolist()
        for qs, qe, ks, ke in zip(
            cross_pass.offsets.tolist()[:-1],
            cross_pass.offsets.tolist()[1:],
            cross_pass.kv_offsets.tolist()[:-1],
            cross_pass.kv_offsets.tolist()[1:],
            strict=True,
        ):
            edges.update((query, key) for query in queries[qs:qe] for key in keys[ks:ke])
    return edges


@pytest.mark.L0
@pytest.mark.CPU
def test_robot_pack_uses_shared_items_and_preserves_ragged_full_rate_actions() -> None:
    packed = _robot_pack()
    assert packed.multiview_action_temporal_factor == 4
    assert packed.num_views_per_vision_item is None
    assert packed.action is not None
    assert packed.action.token_shapes == [(24,)]
    assert packed.action.tokens[0][:, 0].tolist() == list(range(0, 32, 4)) * 3
    items = _multiview_sensor_mask_items(packed)[0]
    assert [item.token_shape for item in items] == [(3, 1, 2), (3, 1, 1), (3, 2, 1)] + [(2, 1, 4)] * 3
    assert [item.view_offset for item in items] == [0, 1, 2, 0, 1, 2]
    assert [item.frame_offset for item in items] == [0, 0, 0, 1, 1, 1]
    assert all(not item.is_control and item.caption_access == "all_captions" for item in items)
    assert [item.seconds_per_frame for item in items] == pytest.approx([0.2] * 6)

    plan = _prepare(packed)
    assert plan.num_gen_tokens == 39
    assert plan.padded_gen_tokens == 42
    assert plan.is_control == (False,) * 6
    assert plan.same_view_offsets is not None and plan.same_view_gather is not None
    assert plan.same_view_offsets.tolist() == [0, 14, 25, 39, 42]
    assert plan.same_view_gather.tolist() == [
        *range(0, 6),
        *range(15, 23),
        *range(6, 9),
        *range(23, 31),
        *range(9, 15),
        *range(31, 42),
    ]
    # Frame zero contains RGB alone. Each later instant adds all four actions per
    # camera to its matching latent frame, retaining the camera-major payload order.
    frame_rows = [
        [0, 1, 6, 9, 10],
        [2, 3, 7, 11, 12, *range(15, 19), *range(23, 27), *range(31, 35)],
        [4, 5, 8, 13, 14, *range(19, 23), *range(27, 31), *range(35, 39)],
    ]
    camera_rows = [
        [*range(0, 6), *range(15, 23)],
        [*range(6, 9), *range(23, 31)],
        [*range(9, 15), *range(31, 39)],
    ]
    camera_by_row = {row: camera for camera, rows in enumerate(camera_rows) for row in rows}
    assert _cross_edges(plan) == {
        (query, key)
        for rows in frame_rows
        for query in rows
        for key in rows
        if camera_by_row[query] != camera_by_row[key]
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_robot_pack_forwards_temporal_window_first_frame_rule() -> None:
    packed = _robot_pack()
    plain = _prepare(
        packed,
        mask=MultiviewAttentionMaskConfig(
            attention_scope="decomposed", cross_view_temporal_window=CrossViewTemporalWindowConfig(bounds=(0.0, 0.0))
        ),
    )
    widened = _prepare(
        packed,
        mask=MultiviewAttentionMaskConfig(
            attention_scope="decomposed",
            cross_view_temporal_window=CrossViewTemporalWindowConfig(bounds=(0.0, 0.0), includes_first_frame=True),
        ),
    )
    plain_edges = _cross_edges(plain)
    widened_edges = _cross_edges(widened)
    assert plain_edges < widened_edges
    assert (4, 6) not in plain_edges and (4, 6) in widened_edges
    assert (15, 6) not in plain_edges and (15, 6) in widened_edges
    # Action[0] belongs to frame one, so adding the first observation never adds
    # that action to a later query whose zero-width time window excludes it.
    assert (4, 23) not in widened_edges
    assert (2, 23) in plain_edges
    assert (2, 15) not in widened_edges  # Own-view edges remain in the same-view pass.


@pytest.mark.L0
@pytest.mark.CPU
def test_robot_pack_rejects_unsupported_same_view_window() -> None:
    with pytest.raises(ValueError, match="frame offsets|neighborhood"):
        _prepare(
            _robot_pack(),
            mask=MultiviewAttentionMaskConfig(
                attention_scope="decomposed",
                same_view_temporal_window=SameViewTemporalWindowConfig(sensor_to_sensor_bounds=(-0.2, 0.0)),
            ),
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_single_camera_robot_pack_has_no_cross_view_pass() -> None:
    plan = _prepare(_robot_pack(num_views=1))
    assert plan.num_gen_tokens == 14
    assert plan.cross_view_empty
    assert plan.same_view_offsets is not None
    assert plan.same_view_offsets.tolist() == [0, 14, 17]
    assert plan.same_view_gather is None


@pytest.mark.L0
@pytest.mark.CPU
def test_robot_samples_keep_independent_clocks_and_action_blocks() -> None:
    plan = _prepare(_robot_pack(frames=(3, 4)))
    assert plan.num_gen_tokens == 95
    assert plan.items_per_sample == (6, 6)
    edges = _cross_edges(plan)
    assert all((query < 39) == (key < 39) for query, key in edges)
    assert (2, 23) in edges and (41, 71) in edges
