# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RigRoPE under the maskless cross-view band, against an independent per-token reference."""

import dataclasses
import math

import pytest
import torch

from cosmos_framework.model.generator.mot import multiview_maskless_attention as dense
from cosmos_framework.model.generator.mot.camera_relative_pose_test import _reference_attention, _reference_merge
from cosmos_framework.model.generator.mot.rigrope import RopeRigPE, apply_mrope_rotary, apply_rig_rotary
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    get_causal_seq,
    get_full_only_seq,
    sequence_pack_from_packed_sequence,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_HEAD_DIM = 128
_VIEWS, _FRAMES, _SPATIAL, _TEXT = 2, 4, 2, 2
_TOKENS = _VIEWS * _FRAMES * _SPATIAL  # GEN tokens per sample
_SAMPLES = 2
_POSED, _UNPOSED = slice(0, _TOKENS), slice(_TOKENS, 2 * _TOKENS)


def _packs(tensors: list[torch.Tensor]) -> list[SequencePack]:
    lengths = [_TEXT + _TOKENS] * _SAMPLES
    text = [index for sample in range(_SAMPLES) for index in range(sample * lengths[0], sample * lengths[0] + _TEXT)]
    gen = [
        index for sample in range(_SAMPLES) for index in range(sample * lengths[0] + _TEXT, (sample + 1) * lengths[0])
    ]
    return [
        sequence_pack_from_packed_sequence(
            tensor,
            attn_modes=["causal", "full"] * _SAMPLES,
            split_lens=[_TEXT, _TOKENS] * _SAMPLES,
            sample_lens=lengths,
            packed_und_token_indexes=torch.tensor(text),  # [N_text]
            packed_gen_token_indexes=torch.tensor(gen),  # [N_gen]
        )
        for tensor in tensors
    ]


def _plan(
    radius: int, frame_zero: bool, packs: list[SequencePack], *, band_keys: str = "all_views"
) -> dense.MultiviewMasklessPlan:
    return dense.build_multiview_maskless_plan(
        [_VIEWS] * _SAMPLES,
        [(_VIEWS * _FRAMES, 1, _SPATIAL)] * _SAMPLES,
        device=torch.device("cpu"),
        items_per_sample=[1] * _SAMPLES,
        is_control=[False] * _SAMPLES,
        cross_view_band_radius=radius,
        include_frame_zero=frame_zero,
        cross_view_band_keys=band_keys,
        padded_gen_tokens=get_full_only_seq(packs[0])[0].shape[0],
    )


def _time_only_coords(plan: dense.MultiviewMasklessPlan) -> torch.Tensor:  # [N_gen,8]
    """A posed sample's random descriptors, then an unposed one's latent-frame times alone."""
    coords = torch.randn(plan.padded_gen_tokens, 8)  # [N_gen,8]
    coords[_UNPOSED] = 0.0
    coords[_UNPOSED, 7] = (torch.arange(_FRAMES) * 0.4).repeat_interleave(_SPATIAL).repeat(_VIEWS)  # [V*F*S]
    return coords


def _geometry(
    plan: dense.MultiviewMasklessPlan,
    coords: torch.Tensor,  # [N_gen,8]
    base_angles: torch.Tensor,  # [N_gen,D]
    *,
    valid: torch.Tensor | None,  # [N_gen] bool
    heads: tuple[torch.Tensor, torch.Tensor] | None,  # [H_local], [H_kv_local]
) -> dense.MultiviewMasklessPlan:
    angles = RopeRigPE(_HEAD_DIM)(coords).reshape(-1, _HEAD_DIM)  # [N_gen,D]
    return dataclasses.replace(
        plan,
        mrope_cos=base_angles.cos(),
        mrope_sin=base_angles.sin(),
        rigrope_cos=angles.cos(),
        rigrope_sin=angles.sin(),
        rigrope_valid=valid,
        rigrope_q_heads=None if heads is None else heads[0],
        rigrope_k_heads=None if heads is None else heads[1],
    )


def _band_admits(band_keys: str, same_view: bool, same_frame: bool) -> bool:
    """Whether the cross pass keys a band token, given whether it shares the query's view and frame."""
    if band_keys == "other_views":
        return not same_view
    return band_keys == "all_views" or same_view or same_frame


def _reference(
    packs: list[SequencePack],
    plan: dense.MultiviewMasklessPlan,
    *,
    radius: int,
    frame_zero: bool,
    band_keys: str = "all_views",
) -> torch.Tensor:  # [N_gen,H*D]
    """Every pass's keys concatenated under one softmax per (query, head), from the layout alone."""
    raw_q, raw_k, v = [get_full_only_seq(pack)[0] for pack in packs]  # [N_gen,H,D], [N_gen,H_kv,D]x2
    text_k, text_v = [get_causal_seq(pack)[0] for pack in packs[1:]]  # each [N_text,H_kv,D]
    assert plan.mrope_cos is not None and plan.mrope_sin is not None
    assert plan.rigrope_cos is not None and plan.rigrope_sin is not None
    mrope_q = apply_mrope_rotary(raw_q, plan.mrope_cos, plan.mrope_sin)  # [N_gen,H,D]
    mrope_k = apply_mrope_rotary(raw_k, plan.mrope_cos, plan.mrope_sin)  # [N_gen,H_kv,D]
    rig_q = apply_rig_rotary(raw_q, plan.rigrope_cos, plan.rigrope_sin)  # [N_gen,H,D]
    rig_k = apply_rig_rotary(raw_k, plan.rigrope_cos, plan.rigrope_sin)  # [N_gen,H_kv,D]
    heads, kv_heads = raw_q.shape[1], raw_k.shape[1]
    group = heads // kv_heads
    layout = [
        (sample, view, frame)
        for sample in range(_SAMPLES)
        for view in range(_VIEWS)
        for frame in range(_FRAMES)
        for _ in range(_SPATIAL)
    ]
    rows = []
    for query, (sample, view, frame) in enumerate(layout):
        band = {frame + offset for offset in range(-radius, radius + 1)} | ({0} if frame_zero else set())
        same_view = [key for key, (s, w, _f) in enumerate(layout) if s == sample and w == view]
        cross_view = [
            key
            for key, (s, w, f) in enumerate(layout)
            if s == sample and f in band and _band_admits(band_keys, w == view, f == frame)
        ]
        text = list(range(sample * _TEXT, (sample + 1) * _TEXT))
        head_rows = []
        for head in range(heads):
            kv = head // group
            use_rig = plan.rigrope_q_heads is None or bool(plan.rigrope_q_heads[head])
            if plan.rigrope_valid is not None:
                use_rig = use_rig and bool(plan.rigrope_valid[query])
            cross_q, cross_k = (rig_q, rig_k) if use_rig else (mrope_q, mrope_k)
            scores = torch.cat(
                [
                    mrope_k[same_view, kv] @ mrope_q[query, head],
                    cross_k[cross_view, kv] @ cross_q[query, head],
                    text_k[text, kv] @ mrope_q[query, head],
                ]
            ) / math.sqrt(_HEAD_DIM)  # [N_keys]
            values = torch.cat([v[same_view, kv], v[cross_view, kv], text_v[text, kv]])  # [N_keys,D]
            head_rows.append(scores.softmax(0) @ values)  # [D]
        rows.append(torch.cat(head_rows))  # [H*D]
    return torch.stack(rows)  # [N_gen,H*D]


@pytest.mark.parametrize(("radius", "frame_zero"), [(1, True), (0, True), (2, False)])
@pytest.mark.parametrize(("heads", "kv_heads"), [(2, 2), (4, 2)])
@pytest.mark.parametrize("missing_geometry", ["time_only", "mrope"])
@pytest.mark.parametrize("band_keys", ["all_views", "own_view", "other_views"])
def test_band_rotates_each_key_by_its_own_token(
    monkeypatch: pytest.MonkeyPatch,
    radius: int,
    frame_zero: bool,
    heads: int,
    kv_heads: int,
    missing_geometry: str,
    band_keys: str,
) -> None:
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    torch.manual_seed(11)
    lengths = _SAMPLES * (_TEXT + _TOKENS)
    tensors = [torch.randn(lengths, count, _HEAD_DIM) for count in (heads, kv_heads, kv_heads)]  # each [N,H,D]
    packs = _packs(tensors)
    plan = _plan(radius, frame_zero, packs, band_keys=band_keys)
    assert plan.cross_view_kv_gather is not None
    valid = None
    if missing_geometry == "mrope":
        valid = torch.zeros(plan.padded_gen_tokens, dtype=torch.bool)  # [N_gen]
        valid[_POSED] = True
    geometry = _geometry(
        plan,
        _time_only_coords(plan),
        torch.randn(plan.padded_gen_tokens, _HEAD_DIM // 2).repeat(1, 2),  # [N_gen,D]
        valid=valid,
        heads=dense.rigrope_local_head_masks(heads, kv_heads, kv_heads // 2),
    )

    actual = dense.multiview_maskless_gen_attention(*packs, plan=geometry)  # [N_gen,H*D]
    expected = _reference(packs, geometry, radius=radius, frame_zero=frame_zero, band_keys=band_keys)  # [N_real,H*D]
    torch.testing.assert_close(actual[: expected.shape[0]], expected, atol=2e-5, rtol=2e-5)


def test_own_view_band_keys_only_the_query_view_outside_its_instant() -> None:
    packs = _packs([torch.zeros(_SAMPLES * (_TEXT + _TOKENS), 1, _HEAD_DIM) for _ in range(3)])
    shared = _plan(1, True, packs)
    split = _plan(1, True, packs, band_keys="own_view")
    assert shared.cross_view_offsets is not None and split.cross_view_offsets is not None
    assert split.cross_view_gather is not None and split.cross_view_kv_gather is not None
    # One query run per (sample, instant, view) rather than per (sample, instant).
    assert torch.diff(shared.cross_view_offsets).tolist() == [_VIEWS * _SPATIAL] * (_SAMPLES * _FRAMES)
    assert torch.diff(split.cross_view_offsets).tolist() == [_SPATIAL] * (_SAMPLES * _FRAMES * _VIEWS)
    assert sorted(split.cross_view_gather.tolist()) == list(range(_SAMPLES * _TOKENS))
    # Sample 0, instant 2, view 0 (tokens 4..5): every view at instant 2, and only view 0 at
    # instants 0, 1 and 3 -- the band and frame 0.
    assert split.cross_view_kv_offsets is not None
    run = split.cross_view_gather.tolist().index(4) // _SPATIAL
    start, end = split.cross_view_kv_offsets[run : run + 2].tolist()
    view_1 = _FRAMES * _SPATIAL
    assert sorted(split.cross_view_kv_gather[start:end].tolist()) == [0, 1, 2, 3, 4, 5, 6, 7, view_1 + 4, view_1 + 5]


def test_band_keys_rejects_an_unknown_mode() -> None:
    packs = _packs([torch.zeros(_SAMPLES * (_TEXT + _TOKENS), 1, _HEAD_DIM) for _ in range(3)])
    with pytest.raises(ValueError, match="cross_view_band_keys"):
        _plan(1, True, packs, band_keys="bogus")


@pytest.mark.parametrize("cp_size", [2, 4])
def test_band_head_split_matches_context_parallel_head_shards(monkeypatch: pytest.MonkeyPatch, cp_size: int) -> None:
    """Each Ulysses rank's heads, with its local head masks, reproduce the full run's heads."""
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    torch.manual_seed(12)
    heads, kv_heads = 4, 2
    lengths = _SAMPLES * (_TEXT + _TOKENS)
    tensors = [torch.randn(lengths, count, _HEAD_DIM) for count in (heads, kv_heads, kv_heads)]  # each [N,H,D]
    plan = _plan(1, True, _packs(tensors))
    coords, base_angles = _time_only_coords(plan), torch.randn(plan.padded_gen_tokens, _HEAD_DIM // 2).repeat(1, 2)
    full = dense.multiview_maskless_gen_attention(
        *_packs(tensors),
        plan=_geometry(plan, coords, base_angles, valid=None, heads=dense.rigrope_local_head_masks(heads, kv_heads, 1)),
    )  # [N_gen,H*D]

    # ``context_parallel_attention`` repeats each KV head up to the CP size, then shards both
    # head axes contiguously.
    repeats = max(cp_size // kv_heads, 1)
    q_shards = tensors[0].chunk(cp_size, dim=1)  # each [N,H/cp,D]
    k_shards, v_shards = [tensor.repeat_interleave(repeats, dim=1).chunk(cp_size, dim=1) for tensor in tensors[1:]]
    shards = []
    for rank in range(cp_size):
        local = _geometry(
            plan,
            coords,
            base_angles,
            valid=None,
            heads=dense.rigrope_local_head_masks(heads, kv_heads, 1, cp_rank=rank, cp_size=cp_size),
        )
        shards.append(
            dense.multiview_maskless_gen_attention(
                *_packs([q_shards[rank], k_shards[rank], v_shards[rank]]), plan=local
            )
        )  # [N_gen,H/cp*D]
    torch.testing.assert_close(torch.cat(shards, dim=-1), full, atol=1e-6, rtol=1e-6)


def test_time_only_matches_no_rotation_without_the_band_and_differs_with_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    torch.manual_seed(13)
    lengths = _SAMPLES * (_TEXT + _TOKENS)
    packs = _packs([torch.randn(lengths, 2, _HEAD_DIM) for _ in range(3)])
    padded = _plan(0, False, packs).padded_gen_tokens
    time_only = _time_only_coords(_plan(0, False, packs))  # [N_gen,8]
    no_rotation = time_only.clone()
    no_rotation[_UNPOSED] = 0.0
    base_angles = torch.randn(padded, _HEAD_DIM // 2).repeat(1, 2)  # [N_gen,D]
    heads = dense.rigrope_local_head_masks(2, 2, 1)

    def run(radius: int, frame_zero: bool, coords: torch.Tensor) -> torch.Tensor:  # [N_gen,H*D]
        plan = _geometry(_plan(radius, frame_zero, packs), coords, base_angles, valid=None, heads=heads)
        return dense.multiview_maskless_gen_attention(*packs, plan=plan)

    # Within one instant every key shares the query's time, so the time rotation cancels.
    torch.testing.assert_close(run(0, False, time_only), run(0, False, no_rotation), atol=1e-5, rtol=1e-5)
    banded, content_only = run(1, True, time_only), run(1, True, no_rotation)
    assert not torch.allclose(banded[_UNPOSED, :_HEAD_DIM], content_only[_UNPOSED, :_HEAD_DIM], atol=1e-4)
    # The mRoPE head and the posed sample do not read the unposed sample's time channel.
    torch.testing.assert_close(banded[:, _HEAD_DIM:], content_only[:, _HEAD_DIM:], atol=0, rtol=0)
    torch.testing.assert_close(banded[_POSED], content_only[_POSED], atol=0, rtol=0)
