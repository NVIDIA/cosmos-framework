# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from cosmos_framework.data.generator.action.utils.action_spec import Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.generator.action.utils.pose_utils import (
    _normalize_rotation_matrices,
    _to_numpy_float32,
    build_abs_pose_from_components,
    compute_framewise_idle_frames,
    convert_rotation,
    pose_abs_to_abs9d,
    pose_abs_to_rel,
    pose_rel_to_abs,
)


@pytest.mark.L0
def test_compute_framewise_idle_frames_applies_fps_policy() -> None:
    action = torch.zeros((4, 10), dtype=torch.float32)
    action[:, 0] = 7.5e-4
    action[:, 3] = 1.0
    action[:, 7] = 1.0
    spec = build_action_spec(Pos(), Rot("rot6d"), Gripper())

    assert compute_framewise_idle_frames(action, spec, fps=10, pose_convention="backward_framewise") == 0
    assert compute_framewise_idle_frames(action, spec, fps=5, pose_convention="backward_framewise") == 4
    assert compute_framewise_idle_frames(action, spec, fps=5, pose_convention="backward_anchored") is None


def _make_example_poses_abs() -> np.ndarray:
    xyz = np.array(
        [
            [1.0, -0.5, 0.25],
            [1.5, 0.0, 0.75],
            [2.0, 0.5, 1.5],
            [2.5, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    euler = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, -0.2, 0.3],
            [0.2, 0.15, -0.1],
            [-0.25, 0.05, 0.4],
        ],
        dtype=np.float32,
    )
    return build_abs_pose_from_components(xyz, euler, "euler_xyz")


@pytest.mark.L0
def test_to_numpy_float32_raises_on_requires_grad_tensor() -> None:
    """Tensor inputs with gradients must be explicitly detached by callers."""
    x = torch.randn(2, 3, requires_grad=True)
    with pytest.raises(ValueError, match="non-differentiable"):
        _to_numpy_float32(x)


@pytest.mark.L0
def test_build_abs_pose_from_components_supports_quat_wxyz() -> None:
    """AV-style wxyz quaternions should produce the same matrices as xyzw."""
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    quat_xyzw = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ],
        dtype=np.float32,
    )
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]

    poses_xyzw = build_abs_pose_from_components(xyz, quat_xyzw, "quat_xyzw")
    poses_wxyz = build_abs_pose_from_components(xyz, quat_wxyz, "quat_wxyz")

    np.testing.assert_allclose(poses_xyzw, poses_wxyz, atol=1e-6)


@pytest.mark.L0
def test_build_abs_pose_from_components_matches_manual_euler_conversion() -> None:
    """Euler component helper should match the previous matrix-building pattern."""
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    euler = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, np.pi / 2],
            [0.0, np.pi / 4, np.pi / 2],
        ],
        dtype=np.float32,
    )

    poses_abs = build_abs_pose_from_components(xyz, euler, "euler_xyz")
    manual_poses_abs = np.tile(np.eye(4, dtype=np.float32), (xyz.shape[0], 1, 1))
    manual_poses_abs[:, :3, :3] = R.from_euler("xyz", euler).as_matrix()
    manual_poses_abs[:, :3, 3] = xyz

    actual = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_framewise")
    expected = pose_abs_to_rel(manual_poses_abs, rotation_format="rot6d", pose_convention="backward_framewise")

    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.L0
def test_build_abs_pose_from_components_applies_translation_scale() -> None:
    """Explicit translation scaling should be applied before building pose matrices."""
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    euler = np.zeros((3, 3), dtype=np.float32)

    poses_abs = build_abs_pose_from_components(xyz, euler, "euler_xyz", translation_scale=2.0)

    np.testing.assert_allclose(poses_abs[:, :3, 3], xyz / 2.0, atol=1e-6)


@pytest.mark.L0
def test_pose_abs_to_rel_rotation_formats_follow_centralized_conventions() -> None:
    """Relative-pose conversion should emit the canonical rot6d and euler_xyz blocks."""
    poses_abs = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    euler = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, np.pi / 4, np.pi / 2],
        ],
        dtype=np.float32,
    )
    matrices_np = R.from_euler("xyz", euler).as_matrix().astype(np.float32)
    poses_abs[1:, :3, :3] = matrices_np

    rel_6d = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_anchored")
    expected_rot6d = matrices_np[:, :, :2].transpose(0, 2, 1).reshape(2, 6)
    np.testing.assert_allclose(rel_6d[:, 3:], expected_rot6d, atol=1e-6)

    rel_3d = pose_abs_to_rel(poses_abs, rotation_format="euler_xyz", pose_convention="backward_anchored")
    expected_rot3d = R.from_matrix(matrices_np).as_euler("xyz", degrees=False)
    np.testing.assert_allclose(rel_3d[:, 3:], expected_rot3d, atol=1e-6)


@pytest.mark.L0
def test_convert_rotation_rot6d_to_matrix_uses_column_based_action_convention() -> None:
    """rot6d roundtrip should preserve matrices under the centralized column-based convention."""
    euler = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, np.pi / 4, np.pi / 2],
        ],
        dtype=np.float32,
    )
    matrices_np = R.from_euler("xyz", euler).as_matrix().astype(np.float32)
    rot6d = matrices_np[:, :, :2].transpose(0, 2, 1).reshape(2, 6)

    reconstructed = convert_rotation(rot6d, input_format="rot6d", output_format="matrix")

    np.testing.assert_allclose(reconstructed, matrices_np, atol=1e-6)


@pytest.mark.L0
def test_normalize_rotation_matrices_batched_matches_reference_loop() -> None:
    """Batched SVD normalization should match the previous per-matrix loop behavior."""
    rng = np.random.default_rng(42)
    matrices = rng.normal(size=(32, 3, 3)).astype(np.float32)

    # New batched implementation.
    actual = _normalize_rotation_matrices(matrices)

    # Reference: previous loop implementation.
    expected_list: list[np.ndarray] = []
    for rot_mat in matrices:
        U, _, Vt = np.linalg.svd(rot_mat)
        normalized = U @ Vt
        if np.linalg.det(normalized) < 0:
            U[:, -1] *= -1
            normalized = U @ Vt
        expected_list.append(normalized.astype(np.float32))
    expected = np.stack(expected_list, axis=0)

    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(actual), np.ones(actual.shape[0], dtype=np.float32), atol=1e-5)


@pytest.mark.L0
@pytest.mark.parametrize("rotation_format", ["rot9d", "rot6d", "quat_xyzw", "euler_xyz", "axisangle"])
@pytest.mark.parametrize(
    "pose_convention",
    ["backward_anchored", "backward_framewise", "backward_chunk_anchored_8f", "backward_chunk_anchored_16f"],
)
def test_pose_abs_to_rel_roundtrips_through_pose_rel_to_abs(
    rotation_format: str,
    pose_convention: str,
) -> None:
    """Relative pose encoding should invert back to the original absolute poses."""
    poses_abs = _make_example_poses_abs()

    poses_rel = pose_abs_to_rel(
        poses_abs,
        rotation_format=rotation_format,
        pose_convention=pose_convention,
    )
    reconstructed = pose_rel_to_abs(
        poses_rel,
        rotation_format=rotation_format,
        pose_convention=pose_convention,
        initial_pose=poses_abs[0],
    )

    np.testing.assert_allclose(reconstructed, poses_abs, atol=1e-5)


@pytest.mark.L0
def test_chunk_anchored_encoding_and_decoding() -> None:
    poses_abs = np.tile(np.eye(4, dtype=np.float32), (17, 1, 1))
    poses_abs[:, 0, 3] = np.arange(17, dtype=np.float32)
    chunk_rel = pose_abs_to_rel(
        poses_abs,
        rotation_format="rot6d",
        pose_convention="backward_chunk_anchored_8f",
    )

    np.testing.assert_allclose(chunk_rel[:, 0], [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8])
    reconstructed = pose_rel_to_abs(
        chunk_rel,
        rotation_format="rot6d",
        pose_convention="backward_chunk_anchored_8f",
    )
    np.testing.assert_allclose(reconstructed, poses_abs, atol=1e-6)


@pytest.mark.L0
def test_chunk_anchored_supports_partial_chunks_and_is_world_invariant() -> None:
    poses_abs = _make_example_poses_abs()
    world_transform = build_abs_pose_from_components(
        np.array([[4.0, -2.0, 1.0]], dtype=np.float32),
        np.array([[0.3, -0.2, 0.4]], dtype=np.float32),
        "euler_xyz",
    )[0]
    chunk_rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_chunk_anchored_8f")
    transformed_rel = pose_abs_to_rel(
        world_transform[None] @ poses_abs,
        rotation_format="rot6d",
        pose_convention="backward_chunk_anchored_8f",
    )
    np.testing.assert_allclose(transformed_rel, chunk_rel, atol=1e-5)
    reconstructed = pose_rel_to_abs(
        chunk_rel,
        rotation_format="rot6d",
        pose_convention="backward_chunk_anchored_8f",
        initial_pose=poses_abs[0],
    )
    np.testing.assert_allclose(reconstructed, poses_abs, atol=1e-5)


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pose9d_helpers_emit_float32_without_explicit_casts(dtype) -> None:
    """The 9D forms return (T-1, 9) float32, whatever dtype comes in.

    ``pose_abs_to_abs9d`` deliberately carries no dtype cast of its own beyond
    the position slice: ``convert_rotation`` normalizes through
    ``_to_numpy_float32``, and ``pose_abs_to_rel`` ends in
    ``.astype(np.float32)``. This pins that, so the casts cannot creep back in
    without a test failing.
    """
    poses_abs = _make_example_poses_abs().astype(dtype)  # (T, 4, 4)
    num_frames = len(poses_abs)

    rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_framewise")
    abs9 = pose_abs_to_abs9d(poses_abs)

    for name, out in (("rel", rel), ("abs", abs9)):
        assert out.shape == (num_frames - 1, 9), f"{name}: {out.shape}"
        assert out.dtype == np.float32, f"{name}: {out.dtype}"

    # The absolute form keeps the trajectory's own positions, offset by one
    # frame; the relative form must not.
    np.testing.assert_allclose(abs9[:, :3], poses_abs[1:, :3, 3], atol=1e-6)
    assert not np.allclose(rel[:, :3], abs9[:, :3])


@pytest.mark.L0
def test_backward_anchored_anchor_index_rows_are_relative_to_the_anchor_frame() -> None:
    """Every anchored row must be ``T_anchor^-1 @ T_k`` with the anchor frame skipped."""
    poses_abs = _make_example_poses_abs()  # (4, 4, 4)
    anchor = 2

    rel = pose_abs_to_rel(
        poses_abs,
        rotation_format="rot6d",
        pose_convention="backward_anchored",
        anchor_index=anchor,
    )

    assert rel.shape == (len(poses_abs) - 1, 9)
    inv_anchor = np.linalg.inv(poses_abs[anchor])
    frames = [f for f in range(len(poses_abs)) if f != anchor]
    for row, frame in zip(rel, frames):
        expected = inv_anchor @ poses_abs[frame]
        np.testing.assert_allclose(row[:3], expected[:3, 3], atol=1e-5)
        # Column-based rot6d: first two columns of R.
        np.testing.assert_allclose(row[3:6], expected[:3, 0], atol=1e-5)
        np.testing.assert_allclose(row[6:9], expected[:3, 1], atol=1e-5)


@pytest.mark.L0
def test_backward_anchored_anchor_index_zero_matches_the_legacy_behavior() -> None:
    """``anchor_index=0`` (the default) must reproduce the pre-anchor-index rows exactly."""
    poses_abs = _make_example_poses_abs()

    default_rows = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_anchored")
    explicit_rows = pose_abs_to_rel(
        poses_abs, rotation_format="rot6d", pose_convention="backward_anchored", anchor_index=0
    )

    np.testing.assert_allclose(default_rows, explicit_rows, atol=0)
    inv0 = np.linalg.inv(poses_abs[0])
    for i in range(len(poses_abs) - 1):
        np.testing.assert_allclose(default_rows[i, :3], (inv0 @ poses_abs[i + 1])[:3, 3], atol=1e-5)


@pytest.mark.L0
@pytest.mark.parametrize("anchor", [0, 1, 2, 3])
def test_backward_anchored_anchor_index_roundtrips(anchor: int) -> None:
    """Encoding at any anchor and decoding with the anchor pose must reproduce the trajectory."""
    poses_abs = _make_example_poses_abs()

    rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_anchored", anchor_index=anchor)
    reconstructed = pose_rel_to_abs(
        rel,
        rotation_format="rot6d",
        pose_convention="backward_anchored",
        initial_pose=poses_abs[anchor],
        anchor_index=anchor,
    )

    np.testing.assert_allclose(reconstructed, poses_abs, atol=1e-5)


@pytest.mark.L0
def test_backward_anchored_history_and_future_signs_oppose_for_monotonic_motion() -> None:
    """+z motion with the anchor mid-window: history rows -z, future rows +z."""
    poses = np.tile(np.eye(4, dtype=np.float64), (5, 1, 1))
    poses[:, 2, 3] = np.arange(5) * 0.1  # frames at z = 0.0 .. 0.4; anchor frame 2 at z = 0.2

    rel = pose_abs_to_rel(poses, rotation_format="rot6d", pose_convention="backward_anchored", anchor_index=2)

    np.testing.assert_allclose(rel[:2, 2], [-0.2, -0.1], atol=1e-6)  # history: where the EEF was
    np.testing.assert_allclose(rel[2:, 2], [0.1, 0.2], atol=1e-6)  # future: where it should go


@pytest.mark.L0
def test_anchor_index_is_rejected_for_backward_framewise() -> None:
    """Framewise deltas have no anchor; a stray anchor_index must fail loudly."""
    poses_abs = _make_example_poses_abs()
    with pytest.raises(ValueError, match="anchor_index"):
        pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_framewise", anchor_index=1)
    rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_framewise")
    with pytest.raises(ValueError, match="anchor_index"):
        pose_rel_to_abs(rel, rotation_format="rot6d", pose_convention="backward_framewise", anchor_index=1)


@pytest.mark.L0
def test_anchor_index_out_of_range_is_rejected() -> None:
    poses_abs = _make_example_poses_abs()
    with pytest.raises(ValueError, match="out of range"):
        pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_anchored", anchor_index=4)
    with pytest.raises(ValueError, match="out of range"):
        pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_anchored", anchor_index=-1)
