# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -----------------------------------------------------------------------------

"""Joint RGB/LiDAR/controls/captions against an independent pairwise softmax."""

import dataclasses
import math

import pytest
import torch

from cosmos_framework.configs.base.defaults.multiview_attention import MultiviewAttentionConfig
from cosmos_framework.model.generator.mot import multiview_maskless_attention as dense
from cosmos_framework.model.generator.mot.camera_relative_pose_test import _batch, _reference_attention, _reference_merge
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.rigrope import RopeRigPE
from cosmos_framework.data.generator.sequence_packing.runtime import SequencePack, get_causal_seq, get_full_only_seq


def _pairwise(
    packs: list[SequencePack],
    angles: torch.Tensor,
    *,
    cross_view_only: bool = False,
    mrope_angles: torch.Tensor | None = None,  # [N,D]
) -> torch.Tensor:  # angles: [N,D]; returns [N,H,D]
    q, k, v = [get_full_only_seq(pack)[0] for pack in packs]  # each [N,H,D]
    text_k, text_v = [get_causal_seq(pack)[0] for pack in packs[1:]]  # each [N_text,H,D]
    repeats = q.shape[1] // k.shape[1]
    k, v, text_k, text_v = [x.repeat_interleave(repeats, dim=1) for x in (k, v, text_k, text_v)]  # each [N,H,D]
    d = q.shape[-1] // 2
    # Independent complex-number rotation (the production operator uses real split halves).
    phase = torch.exp(1j * angles[:, None, :d])  # [N,1,D/2]
    qr = torch.complex(q[..., :d], q[..., d:]) * phase  # [N,H,D/2]
    kr = torch.complex(k[..., :d], k[..., d:]) * phase  # [N,H,D/2]
    if mrope_angles is not None:
        # Separate encodings: sensor cross-view scores above use raw Q/K and geometry only.
        # Same-view and caption queries use an independent complex mRoPE reference.
        mrope_phase = torch.exp(1j * mrope_angles[:, None, :d])  # [N,1,D/2]
        qm = torch.complex(q[..., :d], q[..., d:]) * mrope_phase  # [N,H,D/2]
        km = torch.complex(k[..., :d], k[..., d:]) * mrope_phase  # [N,H,D/2]
        q, k = torch.cat([qm.real, qm.imag], -1), torch.cat([km.real, km.imag], -1)  # each [N,H,D]
    layout = [
        (axis, view, frame, control)
        for axis, views, spatial in ((0, 3, 2), (1, 1, 1))
        for control in (True, False)
        for view in range(views)
        for frame in range(2)
        for _ in range(spatial)
    ]
    outputs: list[torch.Tensor] = []
    for qi, (axis, view, frame, control) in enumerate(layout):
        selected = [i for i, (a, v, _, _) in enumerate(layout) if a == axis and v == view]
        same_view_scores = (
            (q[qi] * k[selected]).sum(-1) if cross_view_only else (qr[qi].conj() * kr[selected]).real.sum(-1)
        )  # [N_same,H]
        cross = [i for i, (_, _, f, c) in enumerate(layout) if f == frame and not c] if not control else []
        cross_scores = (qr[qi].conj() * kr[cross]).real.sum(-1)  # [N_cross,H]
        sensor_scores = torch.cat((same_view_scores, cross_scores))  # [N_sensor,H]
        selected += cross
        first, last = ((0, 2), (2, 5), (5, 6))[view] if axis == 0 else (0, 6)
        caption_scores = (q[qi] * text_k[first:last]).sum(-1)  # [N_text,H]
        scores = torch.cat((sensor_scores, caption_scores)) / math.sqrt(q.shape[-1])  # [N_keys,H]
        values = torch.cat((v[selected], text_v[first:last]))  # [N_keys,H,D]
        outputs.append((scores.softmax(0)[..., None] * values).sum(0))  # [H,D]
    return torch.stack(outputs)  # [28,H,D]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_cross_view_attention_and_gradients(monkeypatch: pytest.MonkeyPatch, kv_heads: int) -> None:
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    _check(
        torch.device("cpu"),
        torch.float32,
        kv_heads,
        atol=3e-6,
        rtol=3e-5,
        cross_view_only=True,
        separate_cross_view=True,
    )


def _check(
    device: torch.device,
    dtype: torch.dtype,
    kv_heads: int,
    *,
    atol: float,
    rtol: float,
    cross_view_only: bool = False,
    separate_cross_view: bool = False,
) -> None:
    torch.manual_seed(43)
    tensors, packs, plan = _batch(device, dtype, 128, kv_heads)
    coords = torch.randn(plan.padded_gen_tokens, 8, device=device)  # [N,8]
    angles = RopeRigPE(128).to(device)(coords).reshape(-1, 128)  # [N,D]
    base_angles = torch.randn(plan.padded_gen_tokens, 64, device=device).repeat(1, 2)  # [N,D]
    plan = dataclasses.replace(
        plan,
        camera_relative_pose=None,
        rigrope_cos=angles.cos(),  # [N,D]
        rigrope_sin=angles.sin(),  # [N,D]
        mrope_cos=base_angles.cos().to(dtype) if separate_cross_view else None,  # [N,D] or None
        mrope_sin=base_angles.sin().to(dtype) if separate_cross_view else None,  # [N,D] or None
    )
    actual = dense.multiview_maskless_gen_attention(*packs, plan=plan)[:28].reshape(28, 2, 128)  # [N,H,D]
    reference_packs = []
    reference_tensors = [tensor.detach().float().requires_grad_(True) for tensor in tensors]  # each [N,H,D]
    # Retain the original layout while comparing the fused computation against full precision.
    for pack, tensor in zip(packs, reference_tensors, strict=True):
        copied = dict(pack)
        copied["full_only_seq"] = torch.cat((tensor[6:], tensor.new_zeros(1, *tensor.shape[1:])))  # [N_gen,H,D]
        copied["causal_seq"] = torch.cat((tensor[:6], tensor.new_zeros(1, *tensor.shape[1:])))  # [N_text,H,D]
        reference_packs.append(copied)
    expected = _pairwise(
        reference_packs,
        angles,
        cross_view_only=cross_view_only,
        mrope_angles=base_angles if separate_cross_view else None,
    )  # [28,H,D]
    torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=rtol)
    weights = torch.randn_like(actual)  # [28,H,D]
    actual_grads = torch.autograd.grad((actual * weights).sum(), tensors)  # tuple[[N,H,D]]
    expected_grads = torch.autograd.grad((expected * weights.float()).sum(), reference_tensors)  # tuple[[N,H,D]]
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad.float(), expected_grad, atol=atol * 4, rtol=rtol * 4)


@pytest.mark.L1
@pytest.mark.GPU
@pytest.mark.parametrize("cross_view_only,separate_cross_view", [(False, False), (True, False), (True, True)])
def test_fused_joint_attention_and_backward(cross_view_only: bool, separate_cross_view: bool) -> None:
    _check(
        torch.device("cuda"),
        torch.bfloat16,
        1,
        atol=0.03,
        rtol=0.04,
        cross_view_only=cross_view_only,
        separate_cross_view=separate_cross_view,
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "attention",
    [
        {"geometry_position_encoding": "rigrope_cross_view"},
        {"geometry_position_encoding": "prope_cross_view"},
        {"geometry_position_encoding": "prope"},
        {"maskless_cross_view_band_radius": 1},
        {"maskless_cross_view_include_frame_zero": True},
        {"maskless_cross_view_band_keys": "other_views"},
    ],
    ids=["rigrope", "prope_cross_view", "prope", "band_radius", "frame_zero", "band_keys"],
)
def test_multiview_action_conditioning_rejects_geometry_and_band_settings(attention: dict[str, object]) -> None:
    """Action conditioning builds its own plan without geometry or bands, so it must not drop them silently."""
    attention_config = MultiviewAttentionConfig(**attention)
    Cosmos3VFMNetworkConfig(multiview_attention_config=attention_config)
    with pytest.raises(ValueError, match="action conditioning supports neither"):
        Cosmos3VFMNetworkConfig(multiview_action_conditioning=True, multiview_attention_config=attention_config)
