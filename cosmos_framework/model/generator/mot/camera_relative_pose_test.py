# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -----------------------------------------------------------------------------

import dataclasses
import math
from typing import Any

import pytest
import torch

from cosmos_framework.model.generator.mot import multiview_maskless_attention as dense
from cosmos_framework.model.generator.mot.camera_relative_pose import (
    apply_camera_pose,
    build_camera_relative_pose_plan,
)
from cosmos_framework.model.generator.mot.merge_bridge import MergeAttentionsBridge
from cosmos_framework.model.generator.utils.camera_relative_pose import invert_rigid_transform
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    get_causal_seq,
    get_full_only_seq,
    sequence_pack_from_packed_sequence,
)


def _poses(device: torch.device, dtype: torch.dtype, projective: bool = False) -> torch.Tensor:  # [V,F,4,4]
    poses = torch.eye(4, device=device, dtype=dtype).expand(3, 2, 4, 4).clone()  # [V,F,4,4]
    poses[1, :, :3, :3] = poses.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # [F,3,3]
    poses[1, :, 0, 3] = 1.5  # [F]
    poses[2, :, 1, 3] = -0.75  # [F]
    if projective:
        lifted = torch.eye(4, device=device, dtype=dtype).repeat(3, 1, 1)  # [V,4,4]
        lifted[:, :3, :3] = poses.new_tensor(
            [
                [[0.6, 0.0, 0.05], [0.0, 0.8, -0.1], [0.0, 0.0, 1.0]],
                [[0.3, 0.0, -0.2], [0.0, 0.35, 0.1], [0.0, 0.0, 1.0]],
                [[1.2, 0.0, 0.0], [0.0, 1.1, 0.02], [0.0, 0.0, 1.0]],
            ]
        )  # [V,3,3], per-camera normalized K
        poses = lifted[:, None] @ poses  # [V,F,4,4]
    return poses


def _batch(
    device: torch.device, dtype: torch.dtype, head_dim: int, kv_heads: int = 2, projective: bool = False
) -> tuple[list[torch.Tensor], list[SequencePack], dense.MultiviewMasklessPlan]:
    # RGB control + target (3 views, 2 frames, 2 spatial tokens), then
    # LiDAR control + target (1 view, 2 frames, 1 spatial token), and 3 captions.
    tensors = [
        torch.randn(34, heads, head_dim, device=device, dtype=dtype, requires_grad=True)  # [N,H,D]
        for heads in (2, kv_heads, kv_heads)
    ]
    packs = [
        sequence_pack_from_packed_sequence(
            tensor,
            attn_modes=["causal", "full"],
            split_lens=[6, 28],
            sample_lens=[34],
            packed_und_token_indexes=torch.arange(6, device=device),  # [N_text]
            packed_gen_token_indexes=torch.arange(6, 34, device=device),  # [N_gen]
            text_caption_lens=[[2, 3, 1]],
        )
        for tensor in tensors
    ]
    plan = dense.build_multiview_maskless_plan(
        [3, 3, 1, 1],
        [(6, 1, 2), (6, 1, 2), (2, 1, 1), (2, 1, 1)],
        device=device,
        items_per_sample=[4],
        is_control=[True, False, True, False],
        control_attends_sensor=True,
        view_axis=[0, 0, 1, 1],
        seconds_per_frame=[0.133, 0.133, 0.1, 0.1],
        captions=[[(0, 2), (1, 3), (2, 1)]],
        padded_gen_tokens=get_full_only_seq(packs[0])[0].shape[0],
    )
    poses = _poses(device, dtype, projective)  # [V,F,4,4]
    pose_plan = build_camera_relative_pose_plan(plan, [poses, poses, None, None], projective=projective)
    return tensors, packs, dataclasses.replace(plan, camera_relative_pose=pose_plan)


def _reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(groups, dim=2)  # [1,Nk,H,D]
    v = v.repeat_interleave(groups, dim=2)  # [1,Nk,H,D]
    q_group = torch.searchsorted(
        kwargs["cumulative_seqlen_Q"][1:], torch.arange(q.shape[1], device=q.device), right=True
    )  # [Nq]
    k_group = torch.searchsorted(
        kwargs["cumulative_seqlen_KV"][1:], torch.arange(k.shape[1], device=k.device), right=True
    )  # [Nk]
    allowed = q_group[:, None] == k_group[None, :]  # [Nq,Nk]
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])  # [1,H,Nq,Nk]
    scores = scores.masked_fill(~allowed, torch.finfo(q.dtype).min)  # [1,H,Nq,Nk]
    probabilities = scores.softmax(-1) * allowed  # [1,H,Nq,Nk]
    output = torch.einsum("bhqk,bkhd->bqhd", probabilities, v)  # [1,Nq,H,D]
    return output, scores.logsumexp(-1).transpose(1, 2)  # [1,Nq,H,D], [1,Nq,H]


def _reference_merge(
    *, outputs: list[torch.Tensor], lse_tensors: list[torch.Tensor], **kwargs: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    normalizers = torch.stack(lse_tensors)  # [P,1,N,H]
    weights = normalizers.softmax(0).unsqueeze(-1)  # [P,1,N,H,1]
    return (torch.stack(outputs) * weights).sum(0), normalizers.logsumexp(0)  # [1,N,H,D], [1,N,H]


def _pairwise_reference(
    packs: list[SequencePack], use_pose: bool, projective: bool = False
) -> torch.Tensor:  # [28,H,D]
    q, k, v = [get_full_only_seq(pack)[0] for pack in packs]  # each [29,H,D]
    text_k, text_v = [get_causal_seq(pack)[0] for pack in packs[1:]]  # each [7,H,D]
    groups = q.shape[1] // k.shape[1]
    k, v, text_k, text_v = [item.repeat_interleave(groups, dim=1) for item in (k, v, text_k, text_v)]  # each [N,H,D]
    poses = _poses(q.device, q.dtype, projective)  # [V,F,4,4]
    # Independent explicit physical layout: (axis, camera, frame, is_control).
    layout = [
        (axis, view, frame, control)
        for axis, views, spatial in ((0, 3, 2), (1, 1, 1))
        for control in (True, False)
        for view in range(views)
        for frame in range(2)
        for _ in range(spatial)
    ]
    outputs = []
    for qi, (axis, view, frame, control) in enumerate(layout):
        keys, values = [], []
        # Same-view pass includes both control and target at all times.
        for ki, (ka, kv, _kf, _kc) in enumerate(layout):
            if axis == ka and view == kv:
                keys.append(k[ki])  # [H,D]
                values.append(v[ki])  # [H,D]
        if not control:
            for ki, (ka, kv, kf, kc) in enumerate(layout):
                if kc or kf != frame:
                    continue
                if use_pose and axis == 0 and ka == 0:
                    relative = poses[view, frame] @ torch.linalg.inv(poses[kv, kf])  # [4,4]
                    transform = torch.block_diag(
                        *([relative] * (q.shape[-1] // 8)),
                        torch.eye(q.shape[-1] // 2, device=q.device, dtype=q.dtype),  # [D/2,D/2]
                    )  # [D,D]
                    keys.append(k[ki] @ transform.T)  # [H,D]
                    values.append(v[ki] @ transform.T)  # [H,D]
                else:
                    keys.append(k[ki])  # [H,D]
                    values.append(v[ki])  # [H,D]
        caption_start, caption_end = ((0, 2), (2, 5), (5, 6))[view] if axis == 0 else (0, 6)
        keys.extend(text_k[caption_start:caption_end])  # each [H,D]
        values.extend(text_v[caption_start:caption_end])  # each [H,D]
        scores = (torch.stack(keys) * q[qi]).sum(-1) / math.sqrt(q.shape[-1])  # [Nk,H]
        outputs.append((scores.softmax(0).unsqueeze(-1) * torch.stack(values)).sum(0))  # [H,D]
    return torch.stack(outputs)  # [28,H,D]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("projective", [False, True])
def test_enabled_joint_attention_matches_pairwise_forward_and_backward(
    monkeypatch: pytest.MonkeyPatch, kv_heads: int, projective: bool
) -> None:
    torch.manual_seed(19)
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    tensors, packs, plan = _batch(torch.device("cpu"), torch.float64, 16, kv_heads, projective)
    actual = dense.multiview_maskless_gen_attention(*packs, plan=plan)[:28].reshape(28, 2, 16)  # [N,H,D]
    expected = _pairwise_reference(packs, use_pose=True, projective=projective)  # [N,H,D]
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    weight = torch.randn_like(actual)  # [N,H,D]
    actual_grads = torch.autograd.grad((actual * weight).sum(), tensors, retain_graph=True)  # tuple[[N,H,D]]
    expected_grads = torch.autograd.grad((expected * weight).sum(), tensors)  # tuple[[N,H,D]]
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-9, rtol=1e-9)
    baseline = dense.multiview_maskless_gen_attention(
        *packs, plan=dataclasses.replace(plan, camera_relative_pose=None)
    )[:28].reshape(28, 2, 16)  # [N,H,D]
    # Control and LiDAR query outputs are unchanged; camera sensor outputs change.
    torch.testing.assert_close(actual[:12], baseline[:12], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual[24:], baseline[24:], atol=1e-10, rtol=1e-10)
    assert not torch.allclose(actual[12:24], baseline[12:24])


@pytest.mark.L0
@pytest.mark.CPU
def test_projective_plan_takes_the_general_inverse() -> None:
    _, _, plan = _batch(torch.device("cpu"), torch.float64, 16, projective=True)
    pose = plan.camera_relative_pose
    assert pose is not None
    identity = torch.eye(4, dtype=torch.float64).expand_as(pose.reference_to_camera)  # [N_camera,4,4]
    torch.testing.assert_close(pose.camera_to_reference @ pose.reference_to_camera, identity, atol=1e-12, rtol=0)


@pytest.mark.L0
@pytest.mark.CPU
def test_partition_preserves_all_edges_and_excludes_control() -> None:
    _, _, plan = _batch(torch.device("cpu"), torch.float64, 16)
    pose = plan.camera_relative_pose
    assert pose is not None and pose.other is not None
    assert sorted(pose.cameras.query_indices.tolist()) == list(range(12, 24))
    assert sorted(pose.other.query_indices.tolist()) == list(range(12, 24)) + [26, 27]
    assert not set(pose.cameras.key_indices.tolist()) & {24, 25, 26, 27}
    assert set(pose.other.key_indices.tolist()) == set(range(12, 24)) | {26, 27}


@pytest.mark.L0
@pytest.mark.CPU
def test_mixed_rigs_remain_sample_local_and_keep_single_camera_lidar_edges() -> None:
    plan = dense.build_multiview_maskless_plan(
        [3, 3, 1, 1, 1, 1, 1, 1],
        [(6, 1, 2), (6, 1, 2), (2, 1, 1), (2, 1, 1)] + [(2, 1, 1)] * 4,
        device=torch.device("cpu"),
        items_per_sample=[4, 4],
        is_control=[True, False, True, False] * 2,
        control_attends_sensor=True,
        view_axis=[0, 0, 1, 1] * 2,
    )
    poses = _poses(torch.device("cpu"), torch.float64)  # [V,F,4,4]
    pose_plan = build_camera_relative_pose_plan(plan, [poses, poses, None, None] + [None] * 4)
    assert pose_plan is not None and pose_plan.other is not None
    assert max(pose_plan.cameras.query_indices.tolist()) < 28

    def edges(
        query: torch.Tensor, key: torch.Tensor, q_offsets: torch.Tensor, k_offsets: torch.Tensor
    ) -> set[tuple[int, int]]:
        return {
            (q, k)
            for qs, qe, ks, ke in zip(q_offsets[:-1], q_offsets[1:], k_offsets[:-1], k_offsets[1:], strict=True)
            for q in query[qs:qe].tolist()
            for k in key[ks:ke].tolist()
        }

    assert plan.cross_view_gather is not None and plan.cross_view_offsets is not None
    original = edges(plan.cross_view_gather, plan.cross_view_gather, plan.cross_view_offsets, plan.cross_view_offsets)
    partitions = [
        edges(part.query_indices, part.key_indices, part.query_offsets, part.key_offsets)
        for part in (pose_plan.cameras, pose_plan.other)
    ]
    assert partitions[0].isdisjoint(partitions[1])
    assert partitions[0] | partitions[1] == original
    assert (30, 34) in partitions[1]  # Single-camera target -> LiDAR target, same instant.
    assert all((query < 28) == (key < 28) for query, key in original)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("lidar", [False, True])
def test_single_camera_keeps_existing_plan(lidar: bool) -> None:
    count = 2 if lidar else 1
    plan = dense.build_multiview_maskless_plan(
        [1] * count,
        [(2, 1, 1)] * count,
        device=torch.device("cpu"),
        items_per_sample=[count],
        view_axis=list(range(count)),
    )
    assert build_camera_relative_pose_plan(plan, [None] * count) is None


@pytest.mark.L0
@pytest.mark.CPU
def test_pose_transform_compiles_and_keeps_untransformed_channels() -> None:
    features = torch.randn(6, 2, 16, requires_grad=True)  # [N,H,D]
    poses = _poses(torch.device("cpu"), torch.float32).reshape(6, 4, 4)  # [N,4,4]
    compiled = torch.compile(apply_camera_pose, backend="eager", fullgraph=True)
    actual = compiled(features, poses)  # [N,H,D]
    torch.testing.assert_close(actual, apply_camera_pose(features, poses), atol=0, rtol=0)
    torch.testing.assert_close(actual[..., 8:], features[..., 8:], atol=0, rtol=0)
    actual.sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


@pytest.mark.L0
@pytest.mark.CPU
def test_enabled_dense_path_compiles_as_one_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    _, packs, plan = _batch(torch.device("cpu"), torch.float64, 16)
    compiled = torch.compile(dense.multiview_maskless_gen_attention, backend="eager", fullgraph=True)
    expected = dense.multiview_maskless_gen_attention(*packs, plan=plan)  # [N,H*D]
    actual = compiled(*packs, plan=plan)  # [N,H*D]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.L0
@pytest.mark.CPU
def test_merge_bridge_uses_inverse_for_output_and_transpose_for_gradient() -> None:
    # Mimic merge_attentions' documented storage-patching contract without a GPU
    # kernel, and check both different operations for a non-orthogonal SE(3).
    inner = torch.randn(3, 2, 16, dtype=torch.float64, requires_grad=True)  # [N,H,D]
    poses = _poses(torch.device("cpu"), torch.float64)[:, 0]  # [N,4,4]
    lse = torch.zeros(3, 2, dtype=torch.float64)  # [N,H]

    def forward(out: torch.Tensor, norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_camera_pose(out, poses), norm.clone()  # [N,H,D], [N,H]

    def inverse(out: torch.Tensor, norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_camera_pose(out, invert_rigid_transform(poses)), norm  # [N,H,D], [N,H]

    def gradient(out: torch.Tensor, norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_camera_pose(out, poses.transpose(-1, -2)), norm  # [N,H,D], [N,H]

    outer, _ = MergeAttentionsBridge.apply(inner, lse, forward, inverse, gradient)  # [N,H,D], [N,H]
    merged = torch.randn_like(outer)  # [N,H,D]
    outer.data.copy_(merged)
    upstream = torch.randn_like(outer)  # [N,H,D]
    outer.backward(upstream)
    torch.testing.assert_close(inner, inverse(merged, lse)[0])
    torch.testing.assert_close(inner.grad, gradient(upstream, lse)[0])


@pytest.mark.L1
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires real GPU attention and merge kernels")
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_real_pose_attention_kernel_gradients(kv_heads: int) -> None:
    torch.manual_seed(19)
    tensors, packs, plan = _batch(torch.device("cuda"), torch.bfloat16, 64, kv_heads)
    actual = dense.multiview_maskless_gen_attention(*packs, plan=plan)[:28].reshape(28, 2, 64)  # [N,H,D]
    reference_tensors = [tensor.detach().double().requires_grad_() for tensor in tensors]  # list[[N,H,D]]
    reference_packs = [
        sequence_pack_from_packed_sequence(
            tensor,
            ["causal", "full"],
            [6, 28],
            [34],
            torch.arange(6, device="cuda"),
            torch.arange(6, 34, device="cuda"),
            text_caption_lens=[[2, 3, 1]],
        )
        for tensor in reference_tensors
    ]
    expected = _pairwise_reference(reference_packs, use_pose=True)  # [N,H,D]
    torch.testing.assert_close(actual.double(), expected, atol=0.03, rtol=0.04)
    weight = torch.randn_like(actual)  # [N,H,D]
    actual_grads = torch.autograd.grad((actual * weight).sum(), tensors)  # tuple[[N,H,D]]
    expected_grads = torch.autograd.grad((expected * weight.double()).sum(), reference_tensors)  # tuple[[N,H,D]]
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad.double(), expected_grad, atol=0.04, rtol=0.05)
