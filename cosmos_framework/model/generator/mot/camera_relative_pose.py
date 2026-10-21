# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -----------------------------------------------------------------------------

"""Relative camera encoding for the dense cross-view branch.

On half of each head, Q receives P^T, K/V receive P^-1 and the output
receives P, where P is reference-to-camera, or under ``prope_intrinsics`` the
projection lift(K) @ reference-to-camera. Thus camera i's query sees
P_i P_j^-1 for camera j. Existing mRoPE is retained before these transforms;
the other half of the head is untouched. This is a GTA-style composition with
Cosmos mRoPE, not the spatial-RoPE replacement in full PRoPE.

Only the camera-camera part of the cross-instant pass is transformed. Its
complement keeps the original Q/K/V, including every LiDAR query and key.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING

import torch

from cosmos_framework.model.generator.utils.camera_relative_pose import (
    invert_affine_transform,
    invert_rigid_transform,
)

if TYPE_CHECKING:
    from cosmos_framework.model.generator.mot.multiview_maskless_attention import MultiviewMasklessPlan


@dataclass(frozen=True)
class CameraPosePartition:
    query_indices: torch.Tensor  # [Nq]
    key_indices: torch.Tensor  # [Nk]
    query_offsets: torch.Tensor  # [G+1]
    key_offsets: torch.Tensor  # [G+1]
    max_query_len: int
    max_key_len: int


@dataclass(frozen=True)
class CameraRelativePosePass:
    cameras: CameraPosePartition
    query_reference_to_camera: torch.Tensor  # [Nq,4,4]
    query_camera_to_reference: torch.Tensor  # [Nq,4,4]
    key_camera_to_reference: torch.Tensor  # [Nk,4,4]


@dataclass(frozen=True)
class CameraRelativePosePlan:
    camera_passes: tuple[CameraRelativePosePass, ...]
    other_passes: tuple[CameraPosePartition, ...]


def _partition(
    queries: list[torch.Tensor], keys: list[torch.Tensor], device: torch.device
) -> CameraPosePartition | None:
    if not queries:
        return None
    q_lengths = [indices.numel() for indices in queries]
    k_lengths = [indices.numel() for indices in keys]
    return CameraPosePartition(
        query_indices=torch.cat(queries).to(device=device),  # [Nq]
        key_indices=torch.cat(keys).to(device=device),  # [Nk]
        query_offsets=torch.tensor([0, *accumulate(q_lengths)], dtype=torch.int32, device=device),  # [G+1]
        key_offsets=torch.tensor([0, *accumulate(k_lengths)], dtype=torch.int32, device=device),  # [G+1]
        max_query_len=max(q_lengths),
        max_key_len=max(k_lengths),
    )


def build_camera_relative_pose_plan(
    plan: MultiviewMasklessPlan, poses_per_item: list[torch.Tensor | None], *, projective: bool = False
) -> CameraRelativePosePlan | None:
    """Split each exact cross-view pass without adding, removing, or repeating edges.

    Called once outside compiled layers. Single-camera samples retain the ordinary
    partition even in a mixed batch. Control items are absent from cross-instant
    attention by the existing dense policy. ``projective`` poses carry intrinsics
    (``lift(K) @ T``) and take a general inverse instead of the rigid one.
    """
    if len(poses_per_item) != len(plan.num_views):
        raise ValueError("Camera poses must match the dense plan's sensor items")
    if plan.cross_view_empty:
        return None
    assert plan.cross_view_partitions, "Nonempty cross-view attention must have exact partitions"
    device = plan.cross_view_partitions[0].query_indices.device
    pose_ids = torch.full((plan.num_gen_tokens,), -1, dtype=torch.long, device="cpu")  # [N_gen]
    frame_poses: list[torch.Tensor] = []
    position = pose_offset = 0
    for views, shape, control, axis, poses in zip(
        plan.num_views, plan.token_shapes, plan.is_control, plan.view_axis, poses_per_item, strict=True
    ):
        latent_t, height, width = shape
        spatial = height * width
        item_len = latent_t * spatial
        if axis == 0 and views > 1:
            if poses is None or poses.shape != (views, latent_t // views, 4, 4):
                raise ValueError("Multiview camera items require one pose per camera and latent frame")
            if not control:
                flat_poses = poses.reshape(-1, 4, 4).to(device=device)  # [V*F,4,4]
                frame_poses.append(flat_poses)
                pose_ids[position : position + item_len] = (
                    torch.arange(latent_t, device="cpu") + pose_offset
                ).repeat_interleave(spatial)  # [V*F*S]
                pose_offset += latent_t
        position += item_len
    if not frame_poses:
        return None
    matrices = torch.cat(frame_poses)  # [N_camera_frame,4,4]
    inverse = (invert_affine_transform if projective else invert_rigid_transform)(matrices)  # [N_camera_frame,4,4]
    camera_passes: list[CameraRelativePosePass] = []
    other_passes: list[CameraPosePartition] = []
    for part in plan.cross_view_partitions:
        # Transfer existing group metadata once. Selecting variable-length groups on
        # the GPU here would synchronize once per instant; build their indices on CPU.
        query = part.query_indices.cpu()  # [Nq]
        key = part.key_indices.cpu()  # [Nk]
        q_bounds = part.query_offsets.cpu().tolist()
        k_bounds = part.key_offsets.cpu().tolist()
        camera_queries: list[torch.Tensor] = []
        camera_keys: list[torch.Tensor] = []
        other_queries: list[torch.Tensor] = []
        other_keys: list[torch.Tensor] = []
        for qs, qe, ks, ke in zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:], strict=True):
            queries = query[qs:qe]  # [Nq_group]
            keys = key[ks:ke]  # [Nk_group]
            q_mask = pose_ids[queries] >= 0  # [Nq_group]
            k_mask = pose_ids[keys] >= 0  # [Nk_group]
            q_camera, q_other = queries[q_mask], queries[~q_mask]  # [Nq_camera], [Nq_other]
            k_camera, k_other = keys[k_mask], keys[~k_mask]  # [Nk_camera], [Nk_other]
            if q_camera.numel():
                if k_camera.numel():
                    camera_queries.append(q_camera)
                    camera_keys.append(k_camera)
                if k_other.numel():
                    other_queries.append(q_camera)
                    other_keys.append(k_other)
            if q_other.numel():
                other_queries.append(q_other)
                other_keys.append(keys)
        cameras = _partition(camera_queries, camera_keys, device)
        if cameras is not None:
            query_pose_ids = pose_ids[torch.cat(camera_queries)].to(device=device)  # [Nq_camera]
            key_pose_ids = pose_ids[torch.cat(camera_keys)].to(device=device)  # [Nk_camera]
            camera_passes.append(
                CameraRelativePosePass(
                    cameras=cameras,
                    query_reference_to_camera=matrices[query_pose_ids],  # [Nq_camera,4,4]
                    query_camera_to_reference=inverse[query_pose_ids],  # [Nq_camera,4,4]
                    key_camera_to_reference=inverse[key_pose_ids],  # [Nk_camera,4,4]
                )
            )
        other = _partition(other_queries, other_keys, device)
        if other is not None:
            other_passes.append(other)
    if not camera_passes:
        return None
    return CameraRelativePosePlan(camera_passes=tuple(camera_passes), other_passes=tuple(other_passes))


def prope_pair_channels(head_dim: int, device: torch.device) -> torch.Tensor:  # [D] bool
    """The channels ``apply_prope_pairs`` transforms: both halves of every even split-half pair.

    Split-half rotary pairs channel ``i`` with ``i + D/2``. Taking whole pairs keeps the
    transform off the channels mRoPE rotates on the other pairs, so the two commute and each
    stays exactly relative; taking every other pair keeps every mRoPE axis and frequency band.
    """
    if head_dim % 8:
        raise ValueError("PRoPE pairs require head_dim divisible by 8")
    mask = torch.zeros(head_dim, dtype=torch.bool, device=device)  # [D]
    mask[_prope_block_channels(head_dim, device).reshape(-1)] = True
    return mask


def _prope_block_channels(head_dim: int, device: torch.device) -> torch.Tensor:  # [D/8,4]
    half = head_dim // 2
    starts = torch.arange(0, half, 4, device=device)  # [D/8], pairs 4b and 4b+2 per block
    return torch.stack((starts, starts + half, starts + 2, starts + 2 + half), dim=-1)  # [D/8,4]


def apply_prope_pairs(features: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:  # [N,H,D], [N,4,4] -> [N,H,D]
    """Transform the channels of ``prope_pair_channels`` in 4D blocks; leave the rest as given."""
    head_dim = features.shape[-1]
    if head_dim % 8:
        raise ValueError("PRoPE pairs require head_dim divisible by 8")
    half = head_dim // 2
    lower = features[..., :half].reshape(*features.shape[:-1], -1, 4)  # [N,H,D/8,4]
    upper = features[..., half:].reshape(*features.shape[:-1], -1, 4)  # [N,H,D/8,4]
    # A transformed block is split-half pairs (4b, 4b+D/2) and (4b+2, 4b+2+D/2).
    # Address those four strided lanes directly instead of building two index permutations and
    # gathering the complete head on every decoder layer.
    vectors = torch.stack((lower[..., 0], upper[..., 0], lower[..., 2], upper[..., 2]), dim=-1)
    dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=features.device.type, enabled=False):
        transformed = torch.einsum("nij,nhkj->nhki", matrices.to(dtype=dtype), vectors.to(dtype=dtype))  # [N,H,D/8,4]
    transformed = transformed.to(features.dtype)
    lower = torch.stack((transformed[..., 0], lower[..., 1], transformed[..., 2], lower[..., 3]), dim=-1)
    upper = torch.stack((transformed[..., 1], upper[..., 1], transformed[..., 3], upper[..., 3]), dim=-1)
    return torch.cat((lower.flatten(-2), upper.flatten(-2)), dim=-1)  # [N,H,D]


def apply_camera_pose(features: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:  # [N,H,D], [N,4,4] -> [N,H,D]
    """Transform half of a head in 4D blocks; no data-dependent host checks."""
    if features.shape[-1] % 8:
        raise ValueError("Camera relative pose encoding requires head_dim divisible by 8")
    half = features.shape[-1] // 2
    blocks = features[..., :half].reshape(features.shape[0], features.shape[1], -1, 4)  # [N,H,D/8,4]
    dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=features.device.type, enabled=False):
        transformed = torch.einsum("nij,nhkj->nhki", matrices.to(dtype=dtype), blocks.to(dtype=dtype))  # [N,H,D/8,4]
    transformed = transformed.reshape(features.shape[0], features.shape[1], half).to(features.dtype)  # [N,H,D/2]
    return torch.cat((transformed, features[..., half:]), dim=-1)  # [N,H,D]
