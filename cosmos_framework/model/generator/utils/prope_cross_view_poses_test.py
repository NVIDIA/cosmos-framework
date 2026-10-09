# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -----------------------------------------------------------------------------

from typing import Any

import pytest
import torch

from cosmos_framework.data.generator.multiview.camera_geometry import PER_FRAME_CAMERA_TO_RIG_KEY
from cosmos_framework.model.generator.utils.camera_relative_pose import (
    invert_rigid_transform,
    prepare_prope_cross_view_poses,
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
