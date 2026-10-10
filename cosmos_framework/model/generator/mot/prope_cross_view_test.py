# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``prope_cross_view``: PRoPE on paired channels in the banded cross pass, against a per-token reference."""

import dataclasses
import math

import pytest
import torch

from cosmos_framework.model.generator.mot import multiview_maskless_attention as dense
from cosmos_framework.model.generator.mot.camera_relative_pose import apply_prope_pairs, prope_pair_channels
from cosmos_framework.model.generator.mot.camera_relative_pose_test import _reference_attention, _reference_merge
from cosmos_framework.model.generator.mot.rigrope import apply_mrope_rotary
from cosmos_framework.model.generator.mot.rigrope_band_test import (
    _FRAMES,
    _SAMPLES,
    _SPATIAL,
    _TEXT,
    _TOKENS,
    _VIEWS,
    _band_admits,
    _packs,
    _plan,
)
from cosmos_framework.model.generator.utils.camera_relative_pose import (
    invert_affine_transform,
    invert_rigid_transform,
)
from cosmos_framework.data.generator.sequence_packing.runtime import SequencePack, get_causal_seq, get_full_only_seq

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_HEAD_DIM = 32
_POSED = slice(0, _TOKENS)


def _rigid(count: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:  # [N,4,4]
    rotation, _ = torch.linalg.qr(torch.randn(count, 3, 3, dtype=dtype))  # [N,3,3]
    rotation = rotation * torch.linalg.det(rotation).sign()[:, None, None]  # [N,3,3], det +1
    transform = torch.eye(4, dtype=dtype).repeat(count, 1, 1)  # [N,4,4]
    transform[:, :3, :3] = rotation
    transform[:, :3, 3] = torch.randn(count, 3, dtype=dtype)
    return transform


def _pair_matrix(transform: torch.Tensor, head_dim: int) -> torch.Tensor:  # [4,4] -> [D,D]
    """``transform`` on every block of even split-half pairs ``(4b, 4b+D/2, 4b+2, 4b+2+D/2)``."""
    half = head_dim // 2
    matrix = torch.eye(head_dim, dtype=transform.dtype)  # [D,D]
    for start in range(0, half, 4):
        channels = torch.tensor([start, start + half, start + 2, start + 2 + half])  # [4]
        matrix[channels[:, None], channels[None, :]] = transform
    return matrix


def _split_half_angles(count: int, head_dim: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.randn(count, head_dim // 2, dtype=dtype).repeat(1, 2)  # [N,D]


def test_pair_channels_are_the_even_split_half_pairs() -> None:
    expected = (torch.arange(_HEAD_DIM) % (_HEAD_DIM // 2)) % 2 == 0  # [D]
    assert torch.equal(prope_pair_channels(_HEAD_DIM, torch.device("cpu")), expected)
    with pytest.raises(ValueError, match="divisible by 8"):
        prope_pair_channels(12, torch.device("cpu"))


def test_pair_transform_matches_its_matrix_and_leaves_the_other_channels() -> None:
    torch.manual_seed(1)
    features = torch.randn(5, 3, _HEAD_DIM, dtype=torch.float64)  # [N,H,D]
    poses = _rigid(5)  # [N,4,4]
    actual = apply_prope_pairs(features, poses)  # [N,H,D]
    expected = torch.stack([features[n] @ _pair_matrix(poses[n], _HEAD_DIM).T for n in range(5)])  # [N,H,D]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    others = ~prope_pair_channels(_HEAD_DIM, features.device)  # [D]
    assert torch.equal(actual[..., others], features[..., others])
    restored = apply_prope_pairs(actual, invert_rigid_transform(poses))  # [N,H,D]
    torch.testing.assert_close(restored, features, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("head_dim", [8, 16, 128])
def test_pair_transform_matches_legacy_layout_forward_and_backward(head_dim: int) -> None:
    """The direct lane layout is bitwise the old gather/scatter layout, including its gradients."""
    torch.manual_seed(2)
    features = torch.randn(5, 3, head_dim, dtype=torch.float64, requires_grad=True)  # [N,H,D]
    poses = torch.randn(5, 4, 4, dtype=torch.float64, requires_grad=True)  # [N,4,4]

    half = head_dim // 2
    starts = torch.arange(0, half, 4)
    selected = torch.stack((starts, starts + half, starts + 2, starts + 2 + half), dim=-1).reshape(-1)
    unselected_half = torch.arange(1, half, 2)
    unselected = torch.cat((unselected_half, unselected_half + half))
    order = torch.cat((selected, unselected))
    inverse = torch.argsort(order)
    ordered = features[..., order]
    vectors = ordered[..., :half].reshape(features.shape[0], features.shape[1], -1, 4)
    legacy = torch.einsum("nij,nhkj->nhki", poses, vectors).reshape(features.shape[0], features.shape[1], -1)
    legacy = torch.cat((legacy, ordered[..., half:]), dim=-1)[..., inverse]

    actual = apply_prope_pairs(features, poses)
    torch.testing.assert_close(actual, legacy, atol=0, rtol=0)
    weight = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * weight).sum(), (features, poses), retain_graph=True)
    legacy_grads = torch.autograd.grad((legacy * weight).sum(), (features, poses))
    for actual_grad, legacy_grad in zip(actual_grads, legacy_grads, strict=True):
        torch.testing.assert_close(actual_grad, legacy_grad, atol=1e-12, rtol=1e-12)


def test_pair_transform_keeps_its_dtype_and_computes_in_fp32() -> None:
    torch.manual_seed(2)
    features = torch.randn(4, 2, _HEAD_DIM).to(torch.bfloat16)  # [N,H,D]
    poses = _rigid(4, torch.float32)  # [N,4,4]
    actual = apply_prope_pairs(features, poses)  # [N,H,D]
    assert actual.dtype == torch.bfloat16
    expected = apply_prope_pairs(features.float(), poses).to(torch.bfloat16)  # [N,H,D]
    assert torch.equal(actual, expected)


def test_pair_transform_supports_fullgraph_torch_compile() -> None:
    torch.manual_seed(4)
    features = torch.randn(4, 2, _HEAD_DIM)  # [N,H,D]
    poses = _rigid(4, torch.float32)  # [N,4,4]
    compiled = torch.compile(apply_prope_pairs, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(features, poses), apply_prope_pairs(features, poses))


def test_paired_prope_and_mrope_are_jointly_relative() -> None:
    """Scores depend only on the relative pose and the position offset, on every channel.

    Moving both cameras by one rigid change of world frame, and shifting both positions by
    the same angles, leaves the score unchanged. Composing PRoPE after split-half mRoPE on the
    same channels, as the half-head variant does, would not be.
    """
    torch.manual_seed(3)
    pairs = prope_pair_channels(_HEAD_DIM, torch.device("cpu"))  # [D]
    q, k = torch.randn(2, 1, 1, _HEAD_DIM, dtype=torch.float64)  # each [1,1,D]
    poses = _rigid(2)  # [2,4,4]
    angles = _split_half_angles(2, _HEAD_DIM)  # [2,D]

    def score(world: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:  # [4,4], [D] -> []
        moved = poses @ world  # [2,4,4]
        shifted = angles + shift  # [2,D]
        rotated_q = apply_mrope_rotary(q, shifted[:1].cos(), shifted[:1].sin())  # [1,1,D]
        rotated_k = apply_mrope_rotary(k, shifted[1:].cos(), shifted[1:].sin())  # [1,1,D]
        query = apply_prope_pairs(torch.where(pairs, q, rotated_q), moved[:1].transpose(-1, -2))  # [1,1,D]
        key = apply_prope_pairs(torch.where(pairs, k, rotated_k), invert_rigid_transform(moved[1:]))  # [1,1,D]
        return (query * key).sum()

    reference = score(torch.eye(4, dtype=torch.float64), torch.zeros(_HEAD_DIM, dtype=torch.float64))
    for _ in range(3):
        world, shift = _rigid(1)[0], _split_half_angles(1, _HEAD_DIM)[0]  # [4,4], [D]
        torch.testing.assert_close(score(world, shift), reference, atol=1e-10, rtol=1e-10)


def _geometry(
    plan: dense.MultiviewMasklessPlan,
    poses: torch.Tensor,  # [N_gen,4,4]
    angles: torch.Tensor,  # [N_gen,D]
    heads: tuple[torch.Tensor, torch.Tensor] | None,  # [H_local], [H_kv_local]
) -> dense.MultiviewMasklessPlan:
    return dataclasses.replace(
        plan,
        mrope_cos=angles.cos(),
        mrope_sin=angles.sin(),
        prope_reference_to_camera=poses,
        prope_camera_to_reference=invert_affine_transform(poses),
        rigrope_q_heads=None if heads is None else heads[0],
        rigrope_k_heads=None if heads is None else heads[1],
    )


def _lifted_intrinsics(count: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:  # [N,4,4]
    """Random normalized pinhole K, lifted to 4x4 as PRoPE folds it into a pose."""
    lifted = torch.eye(4, dtype=dtype).repeat(count, 1, 1)  # [N,4,4]
    lifted[:, 0, 0] = 0.2 + torch.rand(count, dtype=dtype)
    lifted[:, 1, 1] = 0.2 + torch.rand(count, dtype=dtype)
    lifted[:, :2, 2] = torch.rand(count, 2, dtype=dtype) - 0.5
    return lifted


def _posed_transforms(padded: int, projective: bool = False) -> torch.Tensor:  # [N_gen,4,4]
    """One random pose per (view, frame) of the first sample; the identity everywhere else.

    ``projective`` left-multiplies each view's poses by its own random K, as ``prope_intrinsics`` does.
    """
    transforms = torch.eye(4, dtype=torch.float64).repeat(padded, 1, 1)  # [N_gen,4,4]
    posed = _rigid(_VIEWS * _FRAMES)  # [V*F,4,4]
    if projective:
        posed = _lifted_intrinsics(_VIEWS).repeat_interleave(_FRAMES, dim=0) @ posed  # [V*F,4,4]
    transforms[_POSED] = posed.repeat_interleave(_SPATIAL, dim=0)  # [V*F*S,4,4]
    return transforms


def _reference(
    packs: list[SequencePack],
    plan: dense.MultiviewMasklessPlan,
    *,
    radius: int,
    frame_zero: bool,
    band_keys: str = "all_views",
) -> torch.Tensor:  # [N_real,H*D]
    """Every pass's keys under one softmax per (query, head), with each PRoPE key and value
    carried into the query camera by the explicit relative transform."""
    raw_q, raw_k, v = [get_full_only_seq(pack)[0] for pack in packs]  # [N_gen,H,D], [N_gen,H_kv,D]x2
    text_k, text_v = [get_causal_seq(pack)[0] for pack in packs[1:]]  # each [N_text,H_kv,D]
    assert plan.mrope_cos is not None and plan.mrope_sin is not None
    assert plan.prope_reference_to_camera is not None
    poses = plan.prope_reference_to_camera  # [N_gen,4,4]
    head_dim = raw_q.shape[-1]
    pairs = (torch.arange(head_dim) % (head_dim // 2)) % 2 == 0  # [D]
    mrope_q = apply_mrope_rotary(raw_q, plan.mrope_cos, plan.mrope_sin)  # [N_gen,H,D]
    mrope_k = apply_mrope_rotary(raw_k, plan.mrope_cos, plan.mrope_sin)  # [N_gen,H_kv,D]
    mixed_q = torch.where(pairs, raw_q, mrope_q)  # [N_gen,H,D]
    mixed_k = torch.where(pairs, raw_k, mrope_k)  # [N_gen,H_kv,D]
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
            if plan.rigrope_q_heads is None or bool(plan.rigrope_q_heads[head]):
                cross_scores, cross_values = [], []
                for key in cross_view:
                    relative = _pair_matrix(poses[query] @ torch.linalg.inv(poses[key]), head_dim)  # [D,D]
                    cross_scores.append((relative @ mixed_k[key, kv]) @ mixed_q[query, head])
                    cross_values.append(relative @ v[key, kv])  # [D]
                cross = (torch.stack(cross_scores), torch.stack(cross_values))  # [N_cross], [N_cross,D]
            else:
                cross = (mrope_k[cross_view, kv] @ mrope_q[query, head], v[cross_view, kv])
            scores = torch.cat(
                [mrope_k[same_view, kv] @ mrope_q[query, head], cross[0], text_k[text, kv] @ mrope_q[query, head]]
            ) / math.sqrt(head_dim)  # [N_keys]
            values = torch.cat([v[same_view, kv], cross[1], text_v[text, kv]])  # [N_keys,D]
            head_rows.append(scores.softmax(0) @ values)  # [D]
        rows.append(torch.cat(head_rows))  # [H*D]
    return torch.stack(rows)  # [N_real,H*D]


@pytest.mark.parametrize(("radius", "frame_zero"), [(1, True), (0, False)])
@pytest.mark.parametrize(("heads", "kv_heads", "geometry_kv_heads"), [(2, 2, 1), (4, 2, 1), (2, 2, None)])
@pytest.mark.parametrize("band_keys", ["all_views", "own_view", "other_views"])
@pytest.mark.parametrize("projective", [False, True])
def test_cross_pass_matches_per_token_reference_forward_and_backward(
    monkeypatch: pytest.MonkeyPatch,
    radius: int,
    frame_zero: bool,
    heads: int,
    kv_heads: int,
    geometry_kv_heads: int | None,
    band_keys: str,
    projective: bool,
) -> None:
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    torch.manual_seed(21)
    lengths = _SAMPLES * (_TEXT + _TOKENS)
    tensors = [
        torch.randn(lengths, count, _HEAD_DIM, dtype=torch.float64, requires_grad=True)
        for count in (heads, kv_heads, kv_heads)
    ]  # each [N,H,D]
    packs = _packs(tensors)
    plan = _plan(radius, frame_zero, packs, band_keys=band_keys)
    head_masks = (
        None if geometry_kv_heads is None else dense.rigrope_local_head_masks(heads, kv_heads, geometry_kv_heads)
    )
    geometry = _geometry(
        plan,
        _posed_transforms(plan.padded_gen_tokens, projective),
        _split_half_angles(plan.padded_gen_tokens, _HEAD_DIM),
        head_masks,
    )

    actual = dense.multiview_maskless_gen_attention(*packs, plan=geometry)  # [N_gen,H*D]
    expected = _reference(packs, geometry, radius=radius, frame_zero=frame_zero, band_keys=band_keys)  # [N_real,H*D]
    actual = actual[: expected.shape[0]]
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    weight = torch.randn_like(actual)  # [N_real,H*D]
    actual_grads = torch.autograd.grad((actual * weight).sum(), tensors, retain_graph=True)
    expected_grads = torch.autograd.grad((expected * weight).sum(), tensors)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-9, rtol=1e-9)


def test_unposed_sample_and_mrope_heads_ignore_the_other_samples_poses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    torch.manual_seed(22)
    lengths = _SAMPLES * (_TEXT + _TOKENS)
    packs = _packs([torch.randn(lengths, 2, _HEAD_DIM, dtype=torch.float64) for _ in range(3)])
    plan = _plan(1, True, packs)
    angles = _split_half_angles(plan.padded_gen_tokens, _HEAD_DIM)  # [N_gen,D]
    heads = dense.rigrope_local_head_masks(2, 2, 1)
    posed = _posed_transforms(plan.padded_gen_tokens)  # [N_gen,4,4]
    reposed = _posed_transforms(plan.padded_gen_tokens)  # [N_gen,4,4]
    first = dense.multiview_maskless_gen_attention(*packs, plan=_geometry(plan, posed, angles, heads))
    second = dense.multiview_maskless_gen_attention(*packs, plan=_geometry(plan, reposed, angles, heads))
    unposed = slice(_TOKENS, 2 * _TOKENS)
    torch.testing.assert_close(first[unposed], second[unposed], atol=0, rtol=0)
    # Head 1 reads KV group 1, which keeps mRoPE on every sample.
    torch.testing.assert_close(first[:, _HEAD_DIM:], second[:, _HEAD_DIM:], atol=0, rtol=0)
    assert not torch.allclose(first[_POSED, :_HEAD_DIM], second[_POSED, :_HEAD_DIM], atol=1e-6)


@pytest.mark.parametrize("projective", [False, True])
def test_identical_world_change_leaves_the_output_unchanged(monkeypatch: pytest.MonkeyPatch, projective: bool) -> None:
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    torch.manual_seed(23)
    lengths = _SAMPLES * (_TEXT + _TOKENS)
    packs = _packs([torch.randn(lengths, 2, _HEAD_DIM, dtype=torch.float64) for _ in range(3)])
    plan = _plan(1, True, packs)
    angles = _split_half_angles(plan.padded_gen_tokens, _HEAD_DIM)  # [N_gen,D]
    poses = _posed_transforms(plan.padded_gen_tokens, projective)  # [N_gen,4,4]
    moved = poses.clone()
    moved[_POSED] = poses[_POSED] @ _rigid(1)[0]  # [N,4,4]
    expected = dense.multiview_maskless_gen_attention(*packs, plan=_geometry(plan, poses, angles, None))
    actual = dense.multiview_maskless_gen_attention(*packs, plan=_geometry(plan, moved, angles, None))
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_head_split_matches_context_parallel_head_shards(monkeypatch: pytest.MonkeyPatch, cp_size: int) -> None:
    """Each Ulysses rank's heads, with its local head masks, reproduce the full run's heads."""
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    torch.manual_seed(24)
    heads, kv_heads = 4, 2
    lengths = _SAMPLES * (_TEXT + _TOKENS)
    tensors = [torch.randn(lengths, count, _HEAD_DIM, dtype=torch.float64) for count in (heads, kv_heads, kv_heads)]
    plan = _plan(1, True, _packs(tensors))
    poses = _posed_transforms(plan.padded_gen_tokens)  # [N_gen,4,4]
    angles = _split_half_angles(plan.padded_gen_tokens, _HEAD_DIM)  # [N_gen,D]
    full = dense.multiview_maskless_gen_attention(
        *_packs(tensors), plan=_geometry(plan, poses, angles, dense.rigrope_local_head_masks(heads, kv_heads, 1))
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
            poses,
            angles,
            dense.rigrope_local_head_masks(heads, kv_heads, 1, cp_rank=rank, cp_size=cp_size),
        )
        shards.append(
            dense.multiview_maskless_gen_attention(
                *_packs([q_shards[rank], k_shards[rank], v_shards[rank]]), plan=local
            )
        )  # [N_gen,H/cp*D]
    torch.testing.assert_close(torch.cat(shards, dim=-1), full, atol=1e-12, rtol=1e-12)


def test_full_cross_pass_supports_fullgraph_torch_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    torch.manual_seed(19)
    lengths = _SAMPLES * (_TEXT + _TOKENS)
    packs = _packs([torch.randn(lengths, count, _HEAD_DIM) for count in (4, 2, 2)])
    plan = _plan(1, True, packs, band_keys="other_views")
    geometry = _geometry(
        plan,
        _posed_transforms(plan.padded_gen_tokens).float(),
        _split_half_angles(plan.padded_gen_tokens, _HEAD_DIM, torch.float32),
        dense.rigrope_local_head_masks(4, 2, 1),
    )
    compiled = torch.compile(dense.multiview_maskless_gen_attention, backend="eager", fullgraph=True)
    expected = dense.multiview_maskless_gen_attention(*packs, plan=geometry)
    torch.testing.assert_close(compiled(*packs, plan=geometry), expected, atol=1e-12, rtol=1e-12)


def test_cross_pass_requires_mrope_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    packs = _packs([torch.zeros(_SAMPLES * (_TEXT + _TOKENS), 1, _HEAD_DIM) for _ in range(3)])
    plan = _plan(1, True, packs)
    poses = torch.eye(4).repeat(plan.padded_gen_tokens, 1, 1)  # [N_gen,4,4]
    geometry = dataclasses.replace(plan, prope_reference_to_camera=poses, prope_camera_to_reference=poses)
    with pytest.raises(ValueError, match="must carry it"):
        dense.multiview_maskless_gen_attention(*packs, plan=geometry)
