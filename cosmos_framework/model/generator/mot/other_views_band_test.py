# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``"other_views"`` band keys: each key counted once, over a tree of cross-view passes."""

import dataclasses
import math
from collections import Counter

import pytest
import torch
from torch._dynamo.testing import CompileCounter

from cosmos_framework.model.generator.mot import multiview_maskless_attention as dense
from cosmos_framework.model.generator.mot.camera_relative_pose import prope_pair_channels
from cosmos_framework.model.generator.mot.camera_relative_pose_test import _reference_attention, _reference_merge
from cosmos_framework.model.generator.mot.prope_cross_view_test import _pair_matrix, _rigid, _split_half_angles
from cosmos_framework.model.generator.mot.rigrope import apply_mrope_rotary, apply_rig_rotary, spread_over_even_pairs
from cosmos_framework.model.generator.utils.camera_relative_pose import invert_rigid_transform
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    get_causal_seq,
    get_full_only_seq,
    sequence_pack_from_packed_sequence,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_HEAD_DIM = 32
_FRAMES, _SPATIAL, _TEXT = 4, 2, 2
# (views, view axis, is control) per item.
_Item = tuple[int, int, bool]
# (sample, (view axis, view), frame, is control) per GEN token.
_Token = tuple[int, tuple[int, int], int, bool]


def _band(frame: int, radius: int, frame_zero: bool) -> set[int]:
    return {frame + offset for offset in range(-radius, radius + 1)} | ({0} if frame_zero else set())


def _layout(samples: list[list[_Item]]) -> list[_Token]:
    """Packed order: sample, item, view, frame, then spatial."""
    return [
        (sample, (axis, view), frame, control)
        for sample, items in enumerate(samples)
        for views, axis, control in items
        for view in range(views)
        for frame in range(_FRAMES)
        for _ in range(_SPATIAL)
    ]


def _build_plan(
    samples: list[list[_Item]],
    *,
    radius: int,
    frame_zero: bool,
    band_keys: str = "other_views",
    padded_gen_tokens: int | None = None,
) -> dense.MultiviewMasklessPlan:
    items = [item for sample_items in samples for item in sample_items]
    return dense.build_multiview_maskless_plan(
        [views for views, _, _ in items],
        [(views * _FRAMES, 1, _SPATIAL) for views, _, _ in items],
        device=torch.device("cpu"),
        items_per_sample=[len(sample_items) for sample_items in samples],
        is_control=[control for _, _, control in items],
        control_attends_sensor=True,
        view_axis=[axis for _, axis, _ in items],
        cross_view_band_radius=radius,
        include_frame_zero=frame_zero,
        cross_view_band_keys=band_keys,
        padded_gen_tokens=padded_gen_tokens,
    )


def _build_deduplicated_plan(
    samples: list[list[_Item]],
    *,
    radius: int,
    padded_gen_tokens: int | None = None,
) -> dense.MultiviewMasklessPlan:
    """The exact partition of ``deduplicate_cross_view``, windowed to ``radius`` frames of one second."""
    items = [item for sample_items in samples for item in sample_items]
    return dense.build_multiview_maskless_plan(
        [views for views, _, _ in items],
        [(views * _FRAMES, 1, _SPATIAL) for views, _, _ in items],
        device=torch.device("cpu"),
        seconds_per_frame=[1.0] * len(items),
        items_per_sample=[len(sample_items) for sample_items in samples],
        is_control=[control for _, _, control in items],
        control_attends_sensor=True,
        view_axis=[axis for _, axis, _ in items],
        deduplicate_cross_view=True,
        decomposed_temporal_window_seconds=(-float(radius), float(radius)) if radius else None,
        padded_gen_tokens=padded_gen_tokens,
    )


def _cross_pairs(plan: dense.MultiviewMasklessPlan) -> Counter[tuple[int, int]]:
    """How many times the cross passes together key each (query, key) pair."""
    pairs: Counter[tuple[int, int]] = Counter()
    for cross_pass in dense.cross_view_passes(plan):
        gather, kv_gather = cross_pass.gather.tolist(), cross_pass.kv_gather.tolist()
        # The scatter back writes each query's row once per pass.
        assert len(set(gather)) == len(gather)
        offsets, kv_offsets = cross_pass.offsets.tolist(), cross_pass.kv_offsets.tolist()
        assert len(offsets) == len(kv_offsets)
        assert max(torch.diff(cross_pass.offsets).tolist()) == cross_pass.max_len
        assert max(torch.diff(cross_pass.kv_offsets).tolist()) == cross_pass.kv_max_len
        for run in range(len(offsets) - 1):
            assert offsets[run + 1] > offsets[run] and kv_offsets[run + 1] > kv_offsets[run]
            for query in gather[offsets[run] : offsets[run + 1]]:
                for key in kv_gather[kv_offsets[run] : kv_offsets[run + 1]]:
                    pairs[(query, key)] += 1
    return pairs


def _expected_pairs(layout: list[_Token], radius: int, frame_zero: bool) -> Counter[tuple[int, int]]:
    """Every sensor key of another view group in the query's band, once; nothing for one-group samples."""
    groups: dict[int, set[tuple[int, int]]] = {}
    for sample, group, _, _ in layout:
        groups.setdefault(sample, set()).add(group)
    pairs: Counter[tuple[int, int]] = Counter()
    for query, (sample, group, frame, control) in enumerate(layout):
        if control or len(groups[sample]) < 2:
            continue
        band = _band(frame, radius, frame_zero)
        for key, (key_sample, key_group, key_frame, key_control) in enumerate(layout):
            if key_sample == sample and not key_control and key_group != group and key_frame in band:
                pairs[(query, key)] += 1
    return pairs


@pytest.mark.parametrize(
    "samples",
    [
        [[(2, 0, False)]],
        [[(3, 0, False)]],
        [[(1, 0, False)], [(4, 0, False)]],
        [[(5, 0, False)], [(1, 0, False)], [(3, 0, False)]],
        [[(7, 0, False)]],
        # A transfer sample's control tokens sit in its view groups but never in the cross pass.
        [[(3, 0, True), (3, 0, False)], [(1, 0, True), (1, 0, False)]],
        # A joint sample's range clip is a view group of its own.
        [[(2, 0, False), (1, 1, False)]],
    ],
)
@pytest.mark.parametrize(("radius", "frame_zero"), [(0, False), (1, True), (2, False), (5, True)])
def test_partition_keys_each_other_view_in_the_band_exactly_once(
    samples: list[list[_Item]], radius: int, frame_zero: bool
) -> None:
    plan = _build_plan(samples, radius=radius, frame_zero=frame_zero)
    layout = _layout(samples)
    expected = _expected_pairs(layout, radius, frame_zero)
    assert _cross_pairs(plan) == expected
    groups = max(len({group for s, group, _, _ in layout if s == sample}) for sample in range(len(samples)))
    assert len(dense.cross_view_passes(plan)) == math.ceil(math.log2(groups))


@pytest.mark.parametrize(
    "samples",
    [
        [[(3, 0, False)]],
        [[(1, 0, False)], [(4, 0, False)]],
        [[(5, 0, False)], [(1, 0, False)], [(3, 0, False)]],
        [[(3, 0, True), (3, 0, False)], [(1, 0, True), (1, 0, False)]],
    ],
)
@pytest.mark.parametrize("radius", [0, 1, 2])
def test_deduplicated_partition_keys_what_the_other_views_band_keys(samples: list[list[_Item]], radius: int) -> None:
    """``deduplicate_cross_view`` (windowed to ``radius`` frames) is the ``"other_views"`` band's key set."""
    plan = _build_deduplicated_plan(samples, radius=radius)
    assert plan.cross_view_partitions
    assert _cross_pairs(plan) == _expected_pairs(_layout(samples), radius, False)


@pytest.mark.parametrize(
    "overrides",
    [
        {"deduplicate_cross_view": True},
        {"decomposed_temporal_window_seconds": (-1.0, 1.0), "seconds_per_frame": [1.0]},
    ],
)
@pytest.mark.parametrize("band", [{"cross_view_band_radius": 1}, {"cross_view_band_keys": "other_views"}])
def test_band_refuses_the_deduplicated_and_windowed_cross_view(
    overrides: dict[str, object], band: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="set one or the other"):
        dense.build_multiview_maskless_plan(
            [2], [(2 * _FRAMES, 1, _SPATIAL)], device=torch.device("cpu"), **overrides, **band
        )


def test_tree_passes_gather_each_key_once_per_level_not_per_query_view() -> None:
    plan = _build_plan([[(7, 0, False)]], radius=1, frame_zero=False)
    aligned = _build_plan([[(7, 0, False)]], radius=1, frame_zero=False, band_keys="all_views")
    assert aligned.cross_view_kv_gather is not None
    keys = sum(cross_pass.kv_gather.numel() for cross_pass in dense.cross_view_passes(plan))
    # Three levels over seven views, each gathering every key at most once per band instant.
    assert keys <= 3 * aligned.cross_view_kv_gather.numel()


def test_all_single_view_batch_skips_the_cross_pass() -> None:
    plan = _build_plan([[(1, 0, False)], [(1, 0, True), (1, 0, False)]], radius=5, frame_zero=True)
    assert plan.cross_view_empty
    assert plan.cross_view_gather is None and plan.cross_view_kv_gather is None
    assert plan.cross_view_extra_passes == ()
    assert dense.cross_view_passes(plan) == []


@pytest.mark.parametrize("band_keys", ["other_views", "all_views"])
def test_two_groups_that_never_share_a_band_skip_the_cross_pass(band_keys: str) -> None:
    # Range sweeps ten camera frames apart land on instants 5, 15, 25 and 35, past the camera's
    # 0..3, so at radius zero no instant holds both groups.
    plan = dense.build_multiview_maskless_plan(
        [1, 1],
        [(_FRAMES, 1, _SPATIAL), (_FRAMES, 1, _SPATIAL)],
        device=torch.device("cpu"),
        seconds_per_frame=[1.0, 10.0],
        items_per_sample=[2],
        view_axis=[0, 1],
        cross_view_band_keys=band_keys,
    )
    # The aligned partition still runs each instant against itself, which only re-keys the
    # query's own view; the exact band has nothing to key.
    assert plan.cross_view_empty == (band_keys == "other_views")


def test_same_view_scope_refuses_other_views() -> None:
    with pytest.raises(ValueError, match="decomposed"):
        dense.build_multiview_maskless_plan(
            [2],
            [(2 * _FRAMES, 1, _SPATIAL)],
            device=torch.device("cpu"),
            attention_scope="same_view",
            cross_view_band_keys="other_views",
        )


def _packs(tensors: list[torch.Tensor], gen_lens: list[int]) -> list[SequencePack]:
    """One caption segment then one GEN segment per sample."""
    lengths = [_TEXT + gen_len for gen_len in gen_lens]
    starts = [sum(lengths[:sample]) for sample in range(len(lengths))]
    text = [start + index for start in starts for index in range(_TEXT)]
    gen = [start + _TEXT + index for start, gen_len in zip(starts, gen_lens) for index in range(gen_len)]
    return [
        sequence_pack_from_packed_sequence(
            tensor,
            attn_modes=["causal", "full"] * len(gen_lens),
            split_lens=[length for gen_len in gen_lens for length in (_TEXT, gen_len)],
            sample_lens=lengths,
            packed_und_token_indexes=torch.tensor(text),  # [N_text]
            packed_gen_token_indexes=torch.tensor(gen),  # [N_gen]
        )
        for tensor in tensors
    ]


def _inputs(
    views: tuple[int, ...], heads: int, kv_heads: int, *, requires_grad: bool = False
) -> tuple[list[torch.Tensor], list[SequencePack]]:
    gen_lens = [count * _FRAMES * _SPATIAL for count in views]
    length = sum(gen_lens) + _TEXT * len(views)
    tensors = [
        torch.randn(length, count, _HEAD_DIM, dtype=torch.float64, requires_grad=requires_grad)
        for count in (heads, kv_heads, kv_heads)
    ]  # each [N,H,D]
    return tensors, _packs(tensors, gen_lens)


def _encoded(
    plan: dense.MultiviewMasklessPlan, encoding: str, heads: tuple[torch.Tensor, torch.Tensor] | None
) -> dense.MultiviewMasklessPlan:
    """``"plain"`` leaves Q/K as given; the others carry mRoPE tables and their cross-view geometry.

    Every token takes random geometry, single-view samples included: they never enter the
    cross pass, so theirs must go unread.
    """
    if encoding == "plain":
        return plan
    padded = plan.padded_gen_tokens
    angles = _split_half_angles(padded, _HEAD_DIM)  # [N_gen,D]
    tables = {"mrope_cos": angles.cos(), "mrope_sin": angles.sin()}
    masks = {
        "rigrope_q_heads": None if heads is None else heads[0],
        "rigrope_k_heads": None if heads is None else heads[1],
    }
    if encoding == "rigrope":
        rig_angles = torch.randn(padded, _HEAD_DIM, dtype=torch.float64)  # [N_gen,D]
        return dataclasses.replace(plan, **tables, **masks, rigrope_cos=rig_angles.cos(), rigrope_sin=rig_angles.sin())
    if encoding == "rigrope_pairs":
        rig_angles = spread_over_even_pairs(torch.randn(padded, _HEAD_DIM // 2, dtype=torch.float64))  # [N_gen,D]
        return dataclasses.replace(
            plan,
            **tables,
            **masks,
            rigrope_cos=rig_angles.cos(),
            rigrope_sin=rig_angles.sin(),
            rigrope_pairs=prope_pair_channels(_HEAD_DIM, torch.device("cpu")),
        )
    poses = _rigid(padded)  # [N_gen,4,4]
    return dataclasses.replace(
        plan,
        **tables,
        **masks,
        prope_reference_to_camera=poses,
        prope_camera_to_reference=invert_rigid_transform(poses),
    )


def _reference(
    packs: list[SequencePack],
    plan: dense.MultiviewMasklessPlan,
    views: tuple[int, ...],
    *,
    radius: int,
    frame_zero: bool,
    encoding: str,
) -> torch.Tensor:  # [N_real,H*D]
    """One softmax per (query, head) over its own view at every frame, every other view of its
    sample in the band, and its captions -- each key once -- from the layout alone."""
    raw_q, raw_k, v = [get_full_only_seq(pack)[0] for pack in packs]  # [N_gen,H,D], [N_gen,H_kv,D]x2
    text_k, text_v = [get_causal_seq(pack)[0] for pack in packs[1:]]  # each [N_text,H_kv,D]
    if encoding == "plain":
        mrope_q, mrope_k = raw_q, raw_k
    else:
        assert plan.mrope_cos is not None and plan.mrope_sin is not None
        mrope_q = apply_mrope_rotary(raw_q, plan.mrope_cos, plan.mrope_sin)  # [N_gen,H,D]
        mrope_k = apply_mrope_rotary(raw_k, plan.mrope_cos, plan.mrope_sin)  # [N_gen,H_kv,D]
    pairs = (torch.arange(_HEAD_DIM) % (_HEAD_DIM // 2)) % 2 == 0  # [D]
    heads, kv_heads = raw_q.shape[1], raw_k.shape[1]
    group = heads // kv_heads
    layout = _layout([[(count, 0, False)] for count in views])
    rows = []
    for query, (sample, view, frame, _) in enumerate(layout):
        band = _band(frame, radius, frame_zero)
        same_view = [key for key, (s, w, _f, _c) in enumerate(layout) if s == sample and w == view]
        cross_view = [
            key
            for key, (s, w, f, _c) in enumerate(layout)
            if views[sample] > 1 and s == sample and w != view and f in band
        ]
        text = list(range(sample * _TEXT, (sample + 1) * _TEXT))
        head_rows = []
        for head in range(heads):
            kv = head // group
            geometry_head = encoding != "plain" and (plan.rigrope_q_heads is None or bool(plan.rigrope_q_heads[head]))
            cross_scores, cross_values = mrope_k[cross_view, kv] @ mrope_q[query, head], v[cross_view, kv]
            if geometry_head and encoding in ("rigrope", "rigrope_pairs") and cross_view:
                assert plan.rigrope_cos is not None and plan.rigrope_sin is not None
                rig_q = apply_rig_rotary(
                    raw_q[query : query + 1], plan.rigrope_cos[query : query + 1], plan.rigrope_sin[query : query + 1]
                )
                rig_k = apply_rig_rotary(raw_k[cross_view], plan.rigrope_cos[cross_view], plan.rigrope_sin[cross_view])
                if encoding == "rigrope_pairs":
                    # Geometry on the even pairs, mRoPE on the odd ones.
                    rig_q = torch.where(pairs, rig_q, mrope_q[query : query + 1])  # [1,H,D]
                    rig_k = torch.where(pairs, rig_k, mrope_k[cross_view])  # [N_cross,H_kv,D]
                cross_scores = rig_k[:, kv] @ rig_q[0, head]  # [N_cross]
            elif geometry_head and encoding == "prope" and cross_view:
                assert plan.prope_reference_to_camera is not None
                poses = plan.prope_reference_to_camera  # [N_gen,4,4]
                mixed_q = torch.where(pairs, raw_q[query, head], mrope_q[query, head])  # [D]
                scores, values = [], []
                for key in cross_view:
                    relative = _pair_matrix(poses[query] @ invert_rigid_transform(poses[key]), _HEAD_DIM)  # [D,D]
                    mixed_k = torch.where(pairs, raw_k[key, kv], mrope_k[key, kv])  # [D]
                    scores.append((relative @ mixed_k) @ mixed_q)
                    values.append(relative @ v[key, kv])  # [D]
                cross_scores, cross_values = torch.stack(scores), torch.stack(values)  # [N_cross], [N_cross,D]
            scores = torch.cat(
                [mrope_k[same_view, kv] @ mrope_q[query, head], cross_scores, text_k[text, kv] @ mrope_q[query, head]]
            ) / math.sqrt(_HEAD_DIM)  # [N_keys]
            values = torch.cat([v[same_view, kv], cross_values, text_v[text, kv]])  # [N_keys,D]
            head_rows.append(scores.softmax(0) @ values)  # [D]
        rows.append(torch.cat(head_rows))  # [H*D]
    return torch.stack(rows)  # [N_real,H*D]


@pytest.mark.parametrize("encoding", ["plain", "rigrope", "rigrope_pairs", "prope"])
@pytest.mark.parametrize("views", [(1, 3, 5), (4, 2)])
@pytest.mark.parametrize(("radius", "frame_zero"), [(1, True), (0, False)])
def test_tree_passes_match_one_softmax_forward_and_backward(
    monkeypatch: pytest.MonkeyPatch,
    encoding: str,
    views: tuple[int, ...],
    radius: int,
    frame_zero: bool,
) -> None:
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    torch.manual_seed(31)
    heads, kv_heads = 4, 2
    tensors, packs = _inputs(views, heads, kv_heads, requires_grad=True)
    plan = _build_plan(
        [[(count, 0, False)] for count in views],
        radius=radius,
        frame_zero=frame_zero,
        padded_gen_tokens=get_full_only_seq(packs[0])[0].shape[0],
    )
    assert len(dense.cross_view_passes(plan)) == math.ceil(math.log2(max(views)))
    geometry = _encoded(plan, encoding, dense.rigrope_local_head_masks(heads, kv_heads, 1))

    actual = dense.multiview_maskless_gen_attention(*packs, plan=geometry)  # [N_gen,H*D]
    expected = _reference(packs, geometry, views, radius=radius, frame_zero=frame_zero, encoding=encoding)
    actual = actual[: expected.shape[0]]
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    weight = torch.randn_like(actual)  # [N_real,H*D]
    actual_grads = torch.autograd.grad((actual * weight).sum(), tensors, retain_graph=True)
    expected_grads = torch.autograd.grad((expected * weight).sum(), tensors)
    # ``apply_rig_rotary`` rotates in fp32, so its gradient rounds there on either path.
    tolerance = 1e-6 if encoding.startswith("rigrope") else 1e-9
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("encoding", ["plain", "rigrope", "rigrope_pairs", "prope"])
@pytest.mark.parametrize("radius", [0, 1])
def test_deduplicated_partitions_match_one_softmax_forward_and_backward(
    monkeypatch: pytest.MonkeyPatch, encoding: str, radius: int
) -> None:
    """The cross-view geometry encodings run over ``deduplicate_cross_view``'s partitions too."""
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    torch.manual_seed(35)
    views = (1, 3, 5)
    heads, kv_heads = 4, 2
    tensors, packs = _inputs(views, heads, kv_heads, requires_grad=True)
    plan = _build_deduplicated_plan(
        [[(count, 0, False)] for count in views],
        radius=radius,
        padded_gen_tokens=get_full_only_seq(packs[0])[0].shape[0],
    )
    geometry = _encoded(plan, encoding, dense.rigrope_local_head_masks(heads, kv_heads, 1))

    actual = dense.multiview_maskless_gen_attention(*packs, plan=geometry)  # [N_gen,H*D]
    expected = _reference(packs, geometry, views, radius=radius, frame_zero=False, encoding=encoding)
    actual = actual[: expected.shape[0]]
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    weight = torch.randn_like(actual)  # [N_real,H*D]
    actual_grads = torch.autograd.grad((actual * weight).sum(), tensors, retain_graph=True)
    expected_grads = torch.autograd.grad((expected * weight).sum(), tensors)
    # ``apply_rig_rotary`` rotates in fp32, so its gradient rounds there on either path.
    tolerance = 1e-6 if encoding.startswith("rigrope") else 1e-9
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=tolerance, rtol=tolerance)


def _pretrained(
    packs: list[SequencePack], plan: dense.MultiviewMasklessPlan, sample: int, gen_start: int, gen_len: int
) -> torch.Tensor:  # [gen_len,H*D]
    """Plain full attention of one sample's GEN tokens over its captions and all its GEN tokens,
    under the plan's mRoPE when it carries one: the pretrained single-view attention."""
    raw_q, raw_k, v = [get_full_only_seq(pack)[0] for pack in packs]  # [N_gen,H,D], [N_gen,H_kv,D]x2
    text_k, text_v = [get_causal_seq(pack)[0] for pack in packs[1:]]  # each [N_text,H_kv,D]
    if plan.mrope_cos is not None:
        assert plan.mrope_sin is not None
        raw_q = apply_mrope_rotary(raw_q, plan.mrope_cos, plan.mrope_sin)  # [N_gen,H,D]
        raw_k = apply_mrope_rotary(raw_k, plan.mrope_cos, plan.mrope_sin)  # [N_gen,H_kv,D]
    rows, text = slice(gen_start, gen_start + gen_len), slice(sample * _TEXT, (sample + 1) * _TEXT)
    group = raw_q.shape[1] // raw_k.shape[1]
    keys = torch.cat([text_k[text], raw_k[rows]]).repeat_interleave(group, dim=1)  # [N_keys,H,D]
    values = torch.cat([text_v[text], v[rows]]).repeat_interleave(group, dim=1)  # [N_keys,H,D]
    out = torch.nn.functional.scaled_dot_product_attention(
        raw_q[rows].transpose(0, 1), keys.transpose(0, 1), values.transpose(0, 1)
    )  # [H,gen_len,D]
    return out.transpose(0, 1).reshape(gen_len, -1)  # [gen_len,H*D]


@pytest.mark.parametrize("encoding", ["rigrope", "rigrope_pairs", "prope"])
@pytest.mark.parametrize("band_keys", ["other_views", "all_views"])
def test_single_view_sample_in_a_mixed_batch_is_pretrained_mrope_attention(
    monkeypatch: pytest.MonkeyPatch, encoding: str, band_keys: str
) -> None:
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    torch.manual_seed(32)
    views = (3, 1, 2)
    _, packs = _inputs(views, 4, 2)
    plan = _build_plan(
        [[(count, 0, False)] for count in views],
        radius=5,
        frame_zero=True,
        band_keys=band_keys,
        padded_gen_tokens=get_full_only_seq(packs[0])[0].shape[0],
    )
    geometry = _encoded(plan, encoding, dense.rigrope_local_head_masks(4, 2, 1))
    actual = dense.multiview_maskless_gen_attention(*packs, plan=geometry)  # [N_gen,H*D]
    start, length = 3 * _FRAMES * _SPATIAL, _FRAMES * _SPATIAL
    expected = _pretrained(packs, geometry, 1, start, length)  # [F*S,H*D]
    torch.testing.assert_close(actual[start : start + length], expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("encoding", ["rigrope", "rigrope_pairs", "prope"])
def test_compiled_attention_does_not_recompile_on_the_sample_count(
    monkeypatch: pytest.MonkeyPatch, encoding: str
) -> None:
    """Packing puts a different number of samples on a rank every step; one graph must serve them all."""
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    # The toy packs hold as many caption tokens as ``sample_offsets`` entries, which duck sizing
    # would tie into one symbol; real packs carry far more caption tokens than samples.
    monkeypatch.setattr(torch.fx.experimental._config, "use_duck_shape", False)
    torch.manual_seed(34)
    counter = CompileCounter()
    compiled = torch.compile(dense.multiview_maskless_gen_attention, backend=counter, dynamic=True, fullgraph=True)
    # Same largest view count, hence the same number of tree passes, at every sample count.
    for views in [(3, 2), (3, 2, 2), (2, 3, 2, 2)]:
        _, packs = _inputs(views, 4, 2)
        plan = _build_plan(
            [[(count, 0, False)] for count in views],
            radius=1,
            frame_zero=True,
            padded_gen_tokens=get_full_only_seq(packs[0])[0].shape[0],
        )
        geometry = _encoded(plan, encoding, dense.rigrope_local_head_masks(4, 2, 1))
        expected = dense.multiview_maskless_gen_attention(*packs, plan=geometry)  # [N_gen,H*D]
        torch.testing.assert_close(compiled(*packs, plan=geometry), expected, atol=1e-12, rtol=1e-12)
    assert counter.frame_count == 1


def test_eager_attention_still_refuses_a_plan_for_another_sample_count() -> None:
    _, packs = _inputs((3, 2), 4, 2)
    plan = _build_plan([[(3, 0, False)], [(2, 0, False)], [(2, 0, False)]], radius=1, frame_zero=True)
    with pytest.raises(ValueError, match="samples but the pack holds"):
        dense.multiview_maskless_gen_attention(*packs, plan=plan)


def test_all_single_view_batch_is_pretrained_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no cross pass the network leaves mRoPE upstream, so Q/K arrive rotated, as in pretraining."""
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    torch.manual_seed(33)
    views = (1, 1)
    _, packs = _inputs(views, 4, 2)
    plan = _build_plan(
        [[(count, 0, False)] for count in views],
        radius=5,
        frame_zero=True,
        padded_gen_tokens=get_full_only_seq(packs[0])[0].shape[0],
    )
    assert plan.cross_view_empty
    actual = dense.multiview_maskless_gen_attention(*packs, plan=plan)  # [N_gen,H*D]
    length = _FRAMES * _SPATIAL
    for sample in range(len(views)):
        expected = _pretrained(packs, plan, sample, sample * length, length)  # [F*S,H*D]
        torch.testing.assert_close(actual[sample * length : (sample + 1) * length], expected, atol=1e-12, rtol=1e-12)
