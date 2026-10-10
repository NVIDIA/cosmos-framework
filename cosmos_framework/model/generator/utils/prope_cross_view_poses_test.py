# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -----------------------------------------------------------------------------

from typing import Any

import pytest
import torch

from cosmos_framework.data.generator.multiview.camera_geometry import PER_FRAME_CAMERA_TO_RIG_KEY
from cosmos_framework.model.generator.utils.camera_relative_pose import (
    invert_affine_transform,
    invert_rigid_transform,
    prepare_camera_relative_poses,
    prepare_prope_cross_view_poses,
    prope_intrinsics,
)
from cosmos_framework.model.generator.utils.rigrope_geometry_test import (
    Tokenizer,
    _pose_world_record,
    _rotation,
    _transform_scene,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _prepare(
    records: list[dict[str, Any] | None], *, views: int = 2, allow_missing: bool = True, **kwargs: Any
) -> list[torch.Tensor | None]:  # each [V,F,4,4]
    return prepare_prope_cross_view_poses(
        records,
        pixel_shapes=[(3 * views, 32, 32)] * len(records),
        latent_frames=[3 * views] * len(records),
        num_views=[views] * len(records),
        items_per_sample=[1] * len(records),
        tokenizer=Tokenizer(),
        allow_missing=allow_missing,
        **kwargs,
    )


def _only(result: list[torch.Tensor | None]) -> torch.Tensor:  # [V,F,4,4]
    assert len(result) == 1 and result[0] is not None
    return result[0].double()


def _relative(poses: torch.Tensor) -> torch.Tensor:  # [V,F,4,4] -> [V,F,V,F,4,4]
    """Every token pair's ``P_q P_k^-1``, which is all the attention reads of the poses."""
    return poses[:, :, None, None] @ invert_rigid_transform(poses)[None, None]


def test_scene_frame_relative_poses_are_the_camera_to_camera_transforms() -> None:
    record = _pose_world_record()
    cameras = record["camera_to_world"]  # [V,F,4,4]
    poses = _only(_prepare([record], translation_scale_m=1.0))  # [V,F,4,4]
    expected = invert_rigid_transform(cameras)[:, :, None, None] @ cameras[None, None]  # [V,F,V,F,4,4]
    torch.testing.assert_close(_relative(poses), expected, atol=1e-6, rtol=0)
    # "exterior" sorts first and anchors the first latent frame.
    torch.testing.assert_close(poses[0, 0], torch.eye(4, dtype=torch.float64), atol=1e-6, rtol=0)


def test_rig_per_frame_anchors_each_frame_on_the_reference_camera() -> None:
    record = _pose_world_record()
    cameras = record["camera_to_world"]  # [V,F,4,4]
    poses = _only(_prepare([record], translation_scale_m=1.0, pose_world_frame="rig_per_frame"))  # [V,F,4,4]
    torch.testing.assert_close(poses[0], torch.eye(4, dtype=torch.float64).expand(3, 4, 4), atol=1e-6, rtol=0)
    torch.testing.assert_close(poses, invert_rigid_transform(cameras) @ cameras[:1], atol=1e-6, rtol=0)


@pytest.mark.parametrize("pose_world_frame", ["scene", "rig_per_frame"])
def test_rms_unit_makes_poses_invariant_to_the_scene_frame_and_scale(pose_world_frame: str) -> None:
    record = _pose_world_record(world_offset=4.0)
    moved = _transform_scene(record, _rotation(0.7, -0.4, 1.9), scale=3.0)
    options = dict(translation_normalization="per_sample_rms", pose_world_frame=pose_world_frame)
    torch.testing.assert_close(
        _only(_prepare([moved], **options)), _only(_prepare([record], **options)), atol=1e-5, rtol=0
    )
    # A fixed unit keeps the scene's metres.
    assert not torch.allclose(_only(_prepare([moved])), _only(_prepare([record])), atol=1e-3)


def test_rms_unit_is_the_rms_camera_distance_from_each_frames_centroid() -> None:
    record = _pose_world_record()
    cameras = record["camera_to_world"]  # [V,F,4,4]
    centers = cameras[..., :3, 3]  # [V,F,3]
    unit = float((centers - centers.mean(dim=0, keepdim=True)).square().sum(-1).mean().sqrt())
    expected = invert_rigid_transform(cameras) @ cameras[:1]  # [V,F,4,4]
    expected[..., :3, 3] /= unit
    poses = _only(
        _prepare([record], translation_normalization="per_sample_rms", pose_world_frame="rig_per_frame")
    )  # [V,F,4,4]
    torch.testing.assert_close(poses, expected, atol=1e-6, rtol=0)
    # The floor bounds the unit from below for a rig whose cameras nearly coincide.
    floored = _only(
        _prepare(
            [record],
            translation_normalization="per_sample_rms",
            translation_floor_m=10 * unit,
            pose_world_frame="rig_per_frame",
        )
    )  # [V,F,4,4]
    torch.testing.assert_close(floored[..., :3, 3], expected[..., :3, 3] / 10, atol=1e-6, rtol=0)


def test_reordering_the_cameras_reorders_the_poses_alone() -> None:
    record = _pose_world_record()
    flipped = dict(record)
    flipped["camera_keys"] = record["camera_keys"][::-1]
    flipped["calibration"] = record["calibration"][::-1]
    for key in ("camera_to_world", "pose_valid", "frame_times_ns", "time_valid", "image_from_calibration"):
        flipped[key] = record[key].flip(0)
    for frame in ("scene", "rig_per_frame"):
        torch.testing.assert_close(
            _only(_prepare([flipped], pose_world_frame=frame)),
            _only(_prepare([record], pose_world_frame=frame)).flip(0),
            atol=1e-6,
            rtol=0,
        )


def test_static_rig_reads_camera_to_rig_on_every_frame() -> None:
    record = _pose_world_record()
    rig = record["rig_geometry"]
    static = {**record, "rig_geometry": {"camera_to_rig": rig["camera_to_rig"]}}
    extrinsics = torch.tensor(
        [rig["camera_to_rig"][key] for key in record["camera_keys"]], dtype=torch.float64
    )  # [V,4,4]
    poses = _only(_prepare([static], translation_scale_m=1.0))  # [V,F,4,4]
    expected = (invert_rigid_transform(extrinsics) @ extrinsics[:1])[:, None].expand(-1, 3, -1, -1)  # [V,F,4,4]
    torch.testing.assert_close(poses, expected, atol=1e-6, rtol=0)


def test_missing_geometry_and_single_view_samples_take_no_poses() -> None:
    record = _pose_world_record()
    unposed = {**record, "pose_valid": torch.zeros((2, 3), dtype=torch.bool)}
    result = _prepare([record, None, unposed])
    assert result[0] is not None and result[1] is None and result[2] is None
    assert _prepare([None], views=1, allow_missing=False) == [None]
    with pytest.raises(ValueError, match="requires camera_geometry"):
        _prepare([None], allow_missing=False)


def test_pose_world_rig_with_invalid_poses_fails_closed_without_a_fallback() -> None:
    record = _pose_world_record()
    unposed = {**record, "pose_valid": torch.zeros((2, 3), dtype=torch.bool)}
    assert unposed["rig_geometry"][PER_FRAME_CAMERA_TO_RIG_KEY]
    with pytest.raises(ValueError, match="valid poses on every latent frame"):
        _prepare([unposed], allow_missing=False)


def _intrinsics_record() -> dict[str, Any]:
    """Two pinhole views with different K; the second is resized by half and cropped."""
    record = _pose_world_record()
    second = {**record["calibration"][1], "fx_px": 9.0, "fy_px": 11.0, "cx_px": 14.0, "cy_px": 17.5}
    affine = torch.eye(3).repeat(2, 1, 1)  # [V,3,3]
    affine[1] = torch.tensor([[0.5, 0.0, 4.25], [0.0, 0.5, -1.75], [0.0, 0.0, 1.0]])
    return {**record, "calibration": [record["calibration"][0], second], "image_from_calibration": affine}


def test_prope_intrinsics_project_camera_points_to_the_normalized_encoded_frame() -> None:
    record = _intrinsics_record()
    height, width = 24, 40
    points = torch.tensor([[0.3, -0.2, 2.0], [-1.0, 0.5, 4.0], [0.0, 0.0, 1.0]], dtype=torch.float64)  # [N,3]
    for view, calibration in enumerate(record["calibration"]):
        native = torch.tensor(
            [
                [calibration["fx_px"], 0.0, calibration["cx_px"]],
                [0.0, calibration["fy_px"], calibration["cy_px"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )  # [3,3]
        pixels = points @ native.T @ record["image_from_calibration"][view].double().T  # [N,3]
        # Integer-center pixels: the frame's outer edges, -0.5 and W - 0.5, map to -1/2 and 1/2.
        expected = (pixels[:, :2] / pixels[:, 2:] + 0.5) / pixels.new_tensor([width, height]) - 0.5  # [N,2]
        lifted = prope_intrinsics(record, view, (height, width))  # [4,4]
        projected = torch.cat((points, points.new_ones(3, 1)), dim=-1) @ lifted.T  # [N,4]
        torch.testing.assert_close(projected[:, :2] / projected[:, 2:3], expected, atol=1e-12, rtol=0)
        torch.testing.assert_close(projected[:, 3], points.new_ones(3), atol=0, rtol=0)


def test_prope_intrinsics_accept_small_but_invertible_normalized_projection() -> None:
    record = _intrinsics_record()
    # A tiny but invertible crop/resize can make det(normalize @ affine @ K) < 1e-12.
    # The old absolute determinant threshold rejected this valid projection.
    record["image_from_calibration"][1, :2, :2] *= 1e-4
    lifted = prope_intrinsics(record, 1, (480, 832))
    intrinsics = lifted[:3, :3]
    assert abs(float(torch.linalg.det(intrinsics))) < 1e-12
    assert int(torch.linalg.matrix_rank(intrinsics)) == 3
    assert torch.isfinite(lifted).all()


@pytest.mark.parametrize("failure", ["singular", "nonfinite"])
def test_prope_intrinsics_reject_invalid_projection(failure: str) -> None:
    record = _intrinsics_record()
    if failure == "singular":
        record["image_from_calibration"][1, 0] = 0
    else:
        record["image_from_calibration"][1, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite and invertible"):
        prope_intrinsics(record, 1, (480, 832))


def test_intrinsics_fold_each_views_projection_into_the_cross_view_poses() -> None:
    record = _intrinsics_record()
    extrinsic = _only(_prepare([record]))  # [V,F,4,4]
    projective = _only(_prepare([record], intrinsics=True))  # [V,F,4,4]
    lifted = torch.stack([prope_intrinsics(record, view, (32, 32)) for view in range(2)])  # [V,4,4]
    torch.testing.assert_close(projective, lifted[:, None] @ extrinsic, atol=1e-6, rtol=0)
    # The attention reads K_q T_q T_k^-1 K_k^-1 for each token pair.
    pairs = projective[:, :, None, None] @ invert_affine_transform(projective)[None, None]  # [V,F,V,F,4,4]
    expected = (
        lifted[:, None, None, None] @ _relative(extrinsic) @ torch.linalg.inv(lifted)[None, None, :, None]
    )  # [V,F,V,F,4,4]
    torch.testing.assert_close(pairs, expected, atol=1e-5, rtol=0)
    assert not torch.allclose(projective, extrinsic, atol=1e-3)


def test_per_camera_prope_folds_intrinsics_and_requires_calibration() -> None:
    record = _intrinsics_record()

    def prepare(geometry: dict[str, Any], intrinsics: bool) -> torch.Tensor:  # [V,F,4,4]
        result = prepare_camera_relative_poses(
            [geometry],
            pixel_shapes=[(6, 32, 32)],
            latent_frames=[6],
            num_views=[2],
            items_per_sample=[1],
            tokenizer=Tokenizer(),
            intrinsics=intrinsics,
        )
        return _only(result)

    lifted = torch.stack([prope_intrinsics(record, view, (32, 32)) for view in range(2)])  # [V,4,4]
    torch.testing.assert_close(
        prepare(record, intrinsics=True), lifted[:, None] @ prepare(record, intrinsics=False), atol=1e-6, rtol=0
    )
    uncalibrated = {**record, "calibration": [record["calibration"][0], None]}
    prepare(uncalibrated, intrinsics=False)
    with pytest.raises(ValueError, match="require each view's calibration"):
        prepare(uncalibrated, intrinsics=True)


def test_affine_inverse_matches_the_rigid_inverse_and_inverts_projections() -> None:
    torch.manual_seed(5)
    rotation, _ = torch.linalg.qr(torch.randn(4, 3, 3, dtype=torch.float64))  # [N,3,3]
    rigid = torch.eye(4, dtype=torch.float64).repeat(4, 1, 1)  # [N,4,4]
    rigid[:, :3, :3] = rotation * torch.linalg.det(rotation).sign()[:, None, None]
    rigid[:, :3, 3] = torch.randn(4, 3, dtype=torch.float64)
    torch.testing.assert_close(invert_affine_transform(rigid), invert_rigid_transform(rigid), atol=1e-12, rtol=0)
    lifted = prope_intrinsics(_intrinsics_record(), 1, (24, 40))  # [4,4]
    projective = (lifted @ rigid).float()  # [N,4,4]
    inverse = invert_affine_transform(projective)  # [N,4,4]
    assert inverse.dtype == torch.float32
    torch.testing.assert_close(inverse @ projective, torch.eye(4).expand(4, 4, 4), atol=1e-5, rtol=0)
