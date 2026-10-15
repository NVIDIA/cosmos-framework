# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -----------------------------------------------------------------------------

"""Physical moment continuity and mRoPE parity for single-view samples in ragged packs."""

import dataclasses

import pytest
import torch

from cosmos_framework.configs.base.defaults.multiview_attention import MultiviewAttentionConfig
from cosmos_framework.model.generator.mot import multiview_maskless_attention as dense
from cosmos_framework.model.generator.mot.camera_relative_pose_test import _reference_attention, _reference_merge
from cosmos_framework.model.generator.mot.rigrope import RopeRigPE, apply_mrope_rotary, rig_features
from cosmos_framework.model.generator.mot.rigrope_attention_test import _check
from cosmos_framework.data.generator.sequence_packing.runtime import get_full_only_seq, sequence_pack_from_packed_sequence

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_metric_moment_is_continuous_through_the_camera_origin_axis() -> None:
    epsilon = 1e-6
    rays = torch.tensor([[[1.0, -epsilon, 0.0], [1.0, 0.0, 0.0], [1.0, epsilon, 0.0]]], requires_grad=True)  # [1,3,3]
    origin = torch.tensor([2.0, 0.0, 0.0])  # [3]
    times = torch.tensor([0.0, 10.0])  # [2]
    coords = rig_features(rays, origin, times, moment_scale_m=25.0, include_time=False)  # [2,1,3,8]
    doubled = rig_features(rays, 2 * origin, times, moment_scale_m=25.0, include_time=False)  # [2,1,3,8]
    torch.testing.assert_close(doubled[..., 3:6], 2 * coords[..., 3:6])
    assert torch.count_nonzero(coords[..., 6:]) == 0
    angles = RopeRigPE(128)(coords).reshape(2, 3, 128)  # [2,3,D]
    torch.testing.assert_close(angles[0], angles[1], atol=0, rtol=0)
    # The old unit-moment direction flips at this axis. Metric phases approach zero continuously.
    assert (angles[:, 2] - angles[:, 0]).abs().max() < 1e-4
    assert torch.count_nonzero(angles[..., 48:64]) == 0
    assert torch.count_nonzero(angles[..., 112:128]) == 0
    gradient = torch.autograd.grad(angles.square().sum(), rays)[0]  # [1,3,3]
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_metric_scale_is_rejected(scale: float) -> None:
    with pytest.raises(ValueError):
        MultiviewAttentionConfig(rigrope_moment_scale_m=scale)
    with pytest.raises(ValueError):
        rig_features(torch.ones(1, 1, 3), torch.ones(3), torch.zeros(1), moment_scale_m=scale)


@pytest.mark.parametrize("mixed", [False, True])
def test_single_view_with_control_retains_mrope(monkeypatch: pytest.MonkeyPatch, mixed: bool) -> None:
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    torch.manual_seed(97)
    views = [1, 2] if mixed else [1]
    lengths = [2 + 8 * view for view in views]
    gen_indices: list[int] = []
    text_indices: list[int] = []
    cursor = 0
    for length in lengths:
        text_indices.extend(range(cursor, cursor + 2))
        gen_indices.extend(range(cursor + 2, cursor + length))
        cursor += length
    tensors = [torch.randn(sum(lengths), 2, 128, requires_grad=True) for _ in range(3)]  # each [N,H,D]
    packs = [
        sequence_pack_from_packed_sequence(
            tensor,
            attn_modes=[mode for _ in views for mode in ("causal", "full")],
            split_lens=[length for view in views for length in (2, 8 * view)],
            sample_lens=lengths,
            packed_und_token_indexes=torch.tensor(text_indices),  # [N_text]
            packed_gen_token_indexes=torch.tensor(gen_indices),  # [N_gen]
        )
        for tensor in tensors
    ]
    plan = dense.build_multiview_maskless_plan(
        [view for view in views for _ in range(2)],
        [(2 * view, 1, 2) for view in views for _ in range(2)],
        device=torch.device("cpu"),
        items_per_sample=[2] * len(views),
        is_control=[True, False] * len(views),
        control_attends_sensor=True,
        padded_gen_tokens=get_full_only_seq(packs[0])[0].shape[0],
    )
    assert plan.cross_view_empty is (not mixed)
    angles = RopeRigPE(128)(torch.randn(plan.padded_gen_tokens, 8)).reshape(-1, 128)  # [N_gen,D]
    base_angles = torch.randn(plan.padded_gen_tokens, 64).repeat(1, 2)  # [N_gen,D]
    baseline_packs = [dict(pack) for pack in packs]
    for index in (0, 1):
        baseline_packs[index]["full_only_seq"] = apply_mrope_rotary(
            get_full_only_seq(packs[index])[0], base_angles.cos(), base_angles.sin()
        )  # [N_gen,H,D]
    geometry = dataclasses.replace(
        plan,
        mrope_cos=base_angles.cos(),  # [N_gen,D]
        mrope_sin=base_angles.sin(),  # [N_gen,D]
        rigrope_cos=angles.cos(),  # [N_gen,D]
        rigrope_sin=angles.sin(),  # [N_gen,D]
    )
    expected = dense.multiview_maskless_gen_attention(*baseline_packs, plan=plan)  # [N_gen,H*D]
    actual = dense.multiview_maskless_gen_attention(*packs, plan=geometry)  # [N_gen,H*D]
    torch.testing.assert_close(actual[:8], expected[:8], atol=0, rtol=0)
    weights = torch.randn_like(actual[:8])  # [8,H*D]
    actual_grads = torch.autograd.grad((actual[:8] * weights).sum(), tensors, retain_graph=True)  # tuple[[N,H,D]]
    expected_grads = torch.autograd.grad((expected[:8] * weights).sum(), tensors)  # tuple[[N,H,D]]
    for got, want in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-5)
    if mixed:
        assert not torch.allclose(actual[16:24], expected[16:24])


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_geometry_only_cross_view_against_independent_pairwise(monkeypatch: pytest.MonkeyPatch, kv_heads: int) -> None:
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    _check(
        torch.device("cpu"),
        torch.float32,
        kv_heads,
        atol=4e-6,
        rtol=4e-5,
        cross_view_only=True,
        separate_cross_view=True,
    )
