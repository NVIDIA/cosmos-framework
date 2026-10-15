# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -----------------------------------------------------------------------------

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from cosmos_framework.data.generator.multiview.camera_geometry import PER_FRAME_CAMERA_TO_RIG_KEY
from cosmos_framework.model.generator.utils.rigrope_geometry import (
    camera_features,
    lidar_features,
    prepare_rigrope_features,
    rigrope_geometry_available,
)
from cosmos_framework.model.generator.tokenizers.lidar.geometry import pandar128_ray_directions
from cosmos_framework.model.generator.tokenizers.lidar.range_projection import V1P2_TRANSFER_RANGE_PROJECTION

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class Tokenizer:
    spatial_compression_factor: int = 16
    temporal_compression_factor: int = 1

    def get_latent_temporal_positions(self, **kwargs: Any) -> torch.Tensor:
        return torch.arange(kwargs["num_latent_frames"], dtype=torch.float32)  # [T]

    def get_pixel_num_frames(self, count: int, **kwargs: Any) -> int:
        return count


def _record(origin_y: float) -> dict[str, Any]:
    transform = torch.eye(4)  # [4,4]
    transform[1, 3] = origin_y  # []
    rig = {"camera_to_rig": {"front": transform.tolist()}, "lidar_to_rig": transform.tolist()}
    return {
        "camera_keys": ["front"],
        "rig_geometry": rig,
        "provenance": {"source_fps": [30.0]},
        "frame_indices": torch.tensor([[12, 13, 14]]),  # [1,F]
        "frame_times_ns": torch.tensor([[0, 33333333, 66666667]]),  # [1,F]
        "time_valid": torch.ones((1, 3), dtype=torch.bool),  # [1,F]
        "calibration": [
            {
                "stream_id": "front",
                "camera_model": "ftheta",
                "distortion_model": "ftheta",
                "distortion_coefficients": [16, 16, 32, 32, 0, 20, 0, 0, 0, 0, 0],
                "width_px": 32,
                "height_px": 32,
                "fx_px": 20.0,
                "fy_px": 20.0,
                "cx_px": 16.0,
                "cy_px": 16.0,
            }
        ],
        "image_from_calibration": torch.eye(3)[None],  # [1,3,3]
    }


def test_single_camera_uses_intrinsics_extrinsics_and_capture_time() -> None:
    record = _record(2.0)
    result = camera_features(
        record,
        record["rig_geometry"],
        pixel_shape=(3, 32, 32),
        latent_shape=(3, 2, 2),
        num_views=1,
        tokenizer=Tokenizer(),
    )  # [1,3,2,2,8]
    assert result.shape == (1, 3, 2, 2, 8)
    # VAE center (7.5,7.5) in integer-center pixels has radius sqrt(144.5) and forward f=20.
    angle = (144.5**0.5) / 20
    assert float(result[0, 0, 0, 0, 2]) == pytest.approx(math.cos(angle), abs=1e-6)
    assert float(result[0, 1, 0, 0, 7]) == pytest.approx(1 / 30)
    assert torch.count_nonzero(result[..., 6]) == 0
    assert not torch.equal(result[0, 0, 0, 0, :3], result[0, 0, 1, 1, :3])


def _pose_world_record(world_offset: float = 0.0) -> dict[str, Any]:
    """Two pinhole cameras in a scene frame; ``wrist`` translates and turns every frame."""
    poses = torch.eye(4, dtype=torch.float64).repeat(2, 3, 1, 1)  # [V,F,4,4]
    poses[0, :, 0, 3] = 1.0  # [F]
    for frame in range(3):
        angle = 0.2 * frame
        poses[1, frame, :3, :3] = torch.tensor(
            [[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)]]
        )  # [3,3]
        poses[1, frame, 1, 3] = 0.5 * frame  # []
    poses[..., :3, 3] += world_offset  # [V,F,3]
    pinhole = {
        "camera_model": "pinhole",
        "distortion_model": "none",
        "distortion_coefficients": [],
        "width_px": 32,
        "height_px": 32,
        "fx_px": 16.0,
        "fy_px": 16.0,
        "cx_px": 15.5,
        "cy_px": 15.5,
    }
    rig = {
        "camera_to_rig": {"exterior": poses[0, 0].tolist(), "wrist": poses[1, 0].tolist()},
        PER_FRAME_CAMERA_TO_RIG_KEY: True,
    }
    return {
        "camera_keys": ["exterior", "wrist"],
        "rig_geometry": rig,
        "provenance": {"source_fps": [30.0, 30.0]},
        "camera_to_world": poses,
        "pose_valid": torch.ones((2, 3), dtype=torch.bool),  # [V,F]
        "frame_times_ns": torch.tensor([[0, 33333333, 66666667]] * 2),  # [V,F]
        "time_valid": torch.ones((2, 3), dtype=torch.bool),  # [V,F]
        "calibration": [pinhole, pinhole],
        "image_from_calibration": torch.eye(3).repeat(2, 1, 1),  # [V,3,3]
    }


def _pose_world_features(record: dict[str, Any], **kwargs: Any) -> torch.Tensor:  # returns [V,T,H,W,8]
    return camera_features(
        record,
        record["rig_geometry"],
        pixel_shape=(6, 32, 32),
        latent_shape=(6, 2, 2),
        num_views=2,
        tokenizer=Tokenizer(),
        **kwargs,
    )


def _transform_scene(record: dict[str, Any], rotation: torch.Tensor, scale: float = 1.0) -> dict[str, Any]:
    """The same capture expressed in another scene frame: rotated by ``rotation``, positions scaled."""
    poses = record["camera_to_world"].clone()  # [V,F,4,4]
    poses[..., :3, :3] = rotation @ poses[..., :3, :3]  # [V,F,3,3]
    poses[..., :3, 3] = scale * poses[..., :3, 3] @ rotation.T  # [V,F,3]
    return {**record, "camera_to_world": poses}


def _rotation(yaw: float, pitch: float, roll: float) -> torch.Tensor:  # returns [3,3] float64
    cz, sz, cy, sy, cx, sx = (f(a) for a in (yaw, pitch, roll) for f in (math.cos, math.sin))
    rz = torch.tensor([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    ry = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=torch.float64)
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=torch.float64)
    return rz @ ry @ rx


def test_canonical_pose_world_rig_is_invariant_to_the_scene_rotation() -> None:
    record = _pose_world_record()
    turned = _transform_scene(record, _rotation(0.7, -0.4, 1.9))
    torch.testing.assert_close(
        _pose_world_features(record, canonical_frame=True),
        _pose_world_features(turned, canonical_frame=True),
        atol=1e-5,
        rtol=0,
    )
    # Without it the scene axes leak into every direction and moment.
    assert not torch.allclose(_pose_world_features(record), _pose_world_features(turned), atol=1e-3)


def test_canonical_pose_world_rig_puts_the_reference_camera_on_forward_left_up_axes() -> None:
    record = _transform_scene(_pose_world_record(), _rotation(0.3, 0.2, -0.5))
    features = _pose_world_features(record, canonical_frame=True, moment_scale_m=1.0)  # [2,3,2,2,8]
    # The top-left token's ray is half a focal length left (+y) and up (+z) of forward (+x).
    torch.testing.assert_close(
        features[0, 0, 0, 0, :3], torch.tensor([1.0, 0.5, 0.5]) / math.sqrt(1.5), atol=1e-6, rtol=0
    )
    # The wrist camera sits 1 m to the exterior camera's left (-x in OpenCV, +y here) at frame 0,
    # and the origin is their midpoint, so its moment is (0,0.5,0) x d.
    ray = features[1, 0, 0, 0, :3]  # [3]
    expected = torch.linalg.cross(torch.tensor([0.0, 0.5, 0.0]), ray)  # [3]
    torch.testing.assert_close(features[1, 0, 0, 0, 3:6], expected, atol=1e-6, rtol=0)


def test_canonical_static_rig_keeps_its_axes_and_centers_on_its_cameras() -> None:
    record = _pose_world_record()
    camera_to_rig = {}
    for view, key in enumerate(record["camera_keys"]):
        transform = record["camera_to_world"][view, 0].clone()  # [4,4]
        transform[2, 3] += 3.0  # the rig origin sits 3 m behind both cameras
        camera_to_rig[key] = transform.tolist()
    static = {**record, "rig_geometry": {"camera_to_rig": camera_to_rig}}
    plain = _pose_world_features(static, moment_scale_m=1.0)  # [2,3,2,2,8]
    canonical = _pose_world_features(static, moment_scale_m=1.0, canonical_frame=True)  # [2,3,2,2,8]
    torch.testing.assert_close(canonical[..., :3], plain[..., :3])
    # Taking the centroid (0.5,0,3) off each origin takes (0.5,0,3) x d off each moment.
    shift = torch.linalg.cross(torch.tensor([0.5, 0.0, 3.0]).expand_as(plain[..., :3]), plain[..., :3])
    torch.testing.assert_close(canonical[..., 3:6], plain[..., 3:6] - shift, atol=1e-5, rtol=0)


def test_per_sample_moment_unit_is_the_rms_camera_distance_and_scale_invariant() -> None:
    record = _pose_world_record()
    options: dict[str, Any] = dict(canonical_frame=True, moment_normalization="per_sample_rms")
    features = _pose_world_features(record, **options)  # [2,3,2,2,8]
    # Frame-0 origin (0.5,0,0); centers (0.5,0,0) and (-0.5,0.5t,0) for t=0,1,2, before the axis relabel.
    rms = math.sqrt((3 * 0.25 + sum(0.25 + (0.5 * t) ** 2 for t in range(3))) / 6)
    torch.testing.assert_close(
        features, _pose_world_features(record, canonical_frame=True, moment_scale_m=rms), atol=1e-6, rtol=0
    )
    # A capture ten times larger rotates by the same angles.
    enlarged = _transform_scene(record, torch.eye(3, dtype=torch.float64), scale=10.0)
    torch.testing.assert_close(_pose_world_features(enlarged, **options), features, atol=1e-5, rtol=0)


def test_per_sample_moment_unit_is_floored_for_nearly_coincident_cameras() -> None:
    record = _transform_scene(_pose_world_record(), torch.eye(3, dtype=torch.float64), scale=0.001)
    torch.testing.assert_close(
        _pose_world_features(record, moment_normalization="per_sample_rms", moment_scale_floor_m=0.05),
        _pose_world_features(record, moment_scale_m=0.05),
    )


_OPENCV_TO_FLU = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float64)  # [3,3]


def _move_rig(record: dict[str, Any], stride_m: float) -> dict[str, Any]:
    """Carry every camera on one body that drives ``stride_m`` and turns per frame."""
    poses = record["camera_to_world"].clone()  # [V,F,4,4]
    for frame in range(poses.shape[1]):
        body = torch.eye(4, dtype=torch.float64)  # [4,4]
        body[:3, :3] = _rotation(0.3 * frame, 0.1 * frame, -0.05 * frame)  # [3,3]
        body[:3, 3] = torch.tensor([stride_m * frame, 2.0 + 0.1 * stride_m * frame, 0.3 * frame])  # [3]
        poses[:, frame] = body @ poses[:, frame]  # [V,4,4]
    return {**record, "camera_to_world": poses}


def _rigid_record() -> dict[str, Any]:
    """``_pose_world_record`` with the wrist camera fixed to the exterior one at its frame-0 pose."""
    record = _pose_world_record()
    poses = record["camera_to_world"].clone()  # [V,F,4,4]
    poses[1] = poses[1, :1]  # [F,4,4]
    return {**record, "camera_to_world": poses}


def test_rig_per_frame_reads_a_moving_rigid_rig_as_its_static_extrinsics() -> None:
    moving = _move_rig(_rigid_record(), stride_m=15.0)
    # The static extrinsics that frame: the exterior camera relabelled forward-left-up, about the centroid.
    frame0 = _rigid_record()["camera_to_world"][:, 0]  # [V,4,4]
    to_rig = _OPENCV_TO_FLU @ frame0[0, :3, :3].T  # [3,3]
    camera_to_rig = {}
    for view, key in enumerate(moving["camera_keys"]):
        transform = torch.eye(4, dtype=torch.float64)  # [4,4]
        transform[:3, :3] = to_rig @ frame0[view, :3, :3]  # [3,3]
        transform[:3, 3] = to_rig @ (frame0[view, :3, 3] - frame0[:, :3, 3].mean(dim=0))  # [3]
        camera_to_rig[key] = transform.tolist()
    static = {**moving, "rig_geometry": {"camera_to_rig": camera_to_rig}}
    options: dict[str, Any] = dict(moment_scale_m=1.0)
    torch.testing.assert_close(
        _pose_world_features(moving, pose_world_frame="rig_per_frame", **options),
        _pose_world_features(static, canonical_frame=True, **options),
        atol=1e-5,
        rtol=0,
    )
    # Read in one frame, the drive and turn leak into every later frame's directions and moments.
    scene = _pose_world_features(moving, canonical_frame=True, **options)  # [2,3,2,2,8]
    assert not torch.allclose(scene[:, 1:], _pose_world_features(static, canonical_frame=True, **options)[:, 1:])


def test_rig_per_frame_drops_shared_motion_and_keeps_relative_motion() -> None:
    record = _pose_world_record()
    features = _pose_world_features(record, pose_world_frame="rig_per_frame")  # [2,3,2,2,8]
    torch.testing.assert_close(
        _pose_world_features(_move_rig(record, stride_m=15.0), pose_world_frame="rig_per_frame"),
        features,
        atol=1e-5,
        rtol=0,
    )
    # The wrist turns and slides against the exterior camera, so its descriptors still change.
    assert not torch.allclose(features[1, 0, ..., :3], features[1, 2, ..., :3], atol=1e-3)
    assert not torch.allclose(features[1, 0, ..., 3:6], features[1, 2, ..., 3:6], atol=1e-3)


def test_rig_per_frame_rms_unit_is_the_rig_spread_not_its_path() -> None:
    moving = _move_rig(_rigid_record(), stride_m=15.0)
    options: dict[str, Any] = dict(moment_normalization="per_sample_rms")
    # Both cameras sit 0.5 m from their midpoint at every frame.
    torch.testing.assert_close(
        _pose_world_features(moving, pose_world_frame="rig_per_frame", **options),
        _pose_world_features(moving, pose_world_frame="rig_per_frame", moment_scale_m=0.5),
        atol=1e-6,
        rtol=0,
    )
    assert not torch.allclose(
        _pose_world_features(moving, canonical_frame=True, **options),
        _pose_world_features(moving, canonical_frame=True, moment_scale_m=0.5),
        atol=1e-3,
    )


def test_rig_per_frame_leaves_static_rigs_alone() -> None:
    record = _record(2.0)
    kwargs: dict[str, Any] = dict(pixel_shape=(3, 32, 32), latent_shape=(3, 2, 2), num_views=1, tokenizer=Tokenizer())
    torch.testing.assert_close(
        camera_features(record, record["rig_geometry"], pose_world_frame="rig_per_frame", **kwargs),
        camera_features(record, record["rig_geometry"], **kwargs),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize(
    "options",
    [dict(canonical_frame=True), dict(moment_normalization="per_sample_rms"), dict(pose_world_frame="rig_per_frame")],
)
def test_canonical_frames_refuse_lidar_items(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="camera-only"):
        prepare_rigrope_features(
            {"camera_geometry": [_record(2.0)]},
            raw_vision=[torch.empty(1, 3, 3, 32, 32)],
            latent_vision=[torch.empty(1, 4, 3, 2, 2)],
            raw_lidar=[torch.empty(1, 3, 3, 128, 1808)],
            latent_lidar=[torch.empty(1, 4, 3, 8, 113)],
            num_views=[1],
            vision_counts=[1],
            lidar_counts=[1],
            camera_tokenizer=Tokenizer(),
            lidar_tokenizer=Tokenizer(),
            projection=V1P2_TRANSFER_RANGE_PROJECTION,
            **options,
        )


def test_pose_world_rig_follows_each_encoded_frame_pose() -> None:
    record = _pose_world_record()
    features = _pose_world_features(record)  # [2,3,2,2,8]
    # The token center (7.5, 7.5) sits half a focal length up-left of the principal point.
    ray = torch.tensor([-0.5, -0.5, 1.0]) / math.sqrt(1.5)  # [3]
    for frame in range(3):
        rotation = record["camera_to_world"][1, frame, :3, :3].float()  # [3,3]
        torch.testing.assert_close(features[1, frame, 0, 0, :3], rotation @ ray, atol=1e-6, rtol=0)
    # The fixed camera keeps its direction while the moving one turns.
    torch.testing.assert_close(features[0, 0, ..., :3], features[0, 2, ..., :3])
    assert not torch.allclose(features[1, 0, ..., :3], features[1, 2, ..., :3])
    assert not torch.allclose(features[1, 0, ..., 3:6], features[1, 2, ..., 3:6])


def test_pose_world_rig_is_invariant_to_the_world_origin() -> None:
    torch.testing.assert_close(
        _pose_world_features(_pose_world_record()), _pose_world_features(_pose_world_record(world_offset=5000.0))
    )


def test_pose_world_rig_requires_encoded_frame_poses() -> None:
    record = _pose_world_record()
    record["pose_valid"][1, 1] = False
    assert not rigrope_geometry_available(record, has_lidar=False)
    with pytest.raises(ValueError, match="pose-world"):
        _pose_world_features(record)


@pytest.mark.parametrize(
    "invalid", [{"fx_px": 0.0}, {"fy_px": -1.0}, {"cx_px": math.nan}], ids=["zero_fx", "negative_fy", "nan_cx"]
)
def test_required_geometry_rejects_invalid_calibration_before_unprojecting(invalid: dict[str, float]) -> None:
    """A fail-closed caller skips ``rigrope_geometry_available``, so the features check calibration."""
    record = _record(2.0)
    record["calibration"] = [{**record["calibration"][0], **invalid}]
    with pytest.raises(ValueError, match="front: invalid RigRoPE calibration"):
        prepare_rigrope_features(
            {"camera_geometry": [record]},
            raw_vision=[torch.empty(1, 3, 3, 32, 32)],
            latent_vision=[torch.empty(1, 4, 3, 2, 2)],
            raw_lidar=None,
            latent_lidar=None,
            num_views=[1],
            vision_counts=[1],
            lidar_counts=[0],
            camera_tokenizer=Tokenizer(),
            lidar_tokenizer=Tokenizer(),
            projection=V1P2_TRANSFER_RANGE_PROJECTION,
        )


def test_mixed_samples_and_control_items_do_not_share_geometry() -> None:
    records = [_record(2.0), _record(-2.0)]
    pixels = [torch.empty(1, 3, 3, 32, 32)] * 4  # each [1,C,F,H,W]
    latents = [torch.empty(1, 4, 3, 2, 2)] * 4  # each [1,C,T,H,W]
    cameras, lidars = prepare_rigrope_features(
        {"camera_geometry": records},
        raw_vision=pixels,
        latent_vision=latents,
        raw_lidar=None,
        latent_lidar=None,
        num_views=[1] * 4,
        vision_counts=[2, 2],
        lidar_counts=[0, 0],
        camera_tokenizer=Tokenizer(),
        lidar_tokenizer=Tokenizer(),
        projection=V1P2_TRANSFER_RANGE_PROJECTION,
    )
    assert lidars == []
    torch.testing.assert_close(cameras[0], cameras[1])
    torch.testing.assert_close(cameras[2], cameras[3])
    torch.testing.assert_close(cameras[0][..., :3], cameras[2][..., :3])
    torch.testing.assert_close(cameras[0][..., 3:6], -cameras[2][..., 3:6])


def test_lidar_angular_grid_wrap_and_nonidentity_rig_frame() -> None:
    record = _record(2.0)
    tokenizer = SimpleNamespace(spatial_compression=(16, 16), get_pixel_num_frames=lambda count: count)
    features = lidar_features(
        record["rig_geometry"],
        pixel_shape=(3, 128, 1808),
        latent_shape=(3, 8, 113),
        tokenizer=tokenizer,
        projection=V1P2_TRANSFER_RANGE_PROJECTION,
        times=torch.tensor([0.0, 0.1, 0.2]),
    )  # [1,T,H,W,8]
    rays = torch.from_numpy(pandar128_ray_directions(range_projection=V1P2_TRANSFER_RANGE_PROJECTION))  # [128,1800,3]
    torch.testing.assert_close(features[0, 0, 0, 0, :3], rays[8, 4], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(features[0, 0, 0, -1, :3], rays[8, 1796], atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(features[..., 6]) == 0
    assert torch.count_nonzero(features[..., 3:6]) > 0
    torch.testing.assert_close(features[0, :, 0, 0, 7], torch.tensor([0.0, 0.1, 0.2]))


@pytest.mark.parametrize("batched_ids", [False, True])
def test_packed_lidar_timestamps_preserve_ragged_samples_and_controls(batched_ids: bool) -> None:
    records = [_record(2.0), _record(-2.0)]
    ids = [torch.tensor([12, 15, 18]), torch.tensor([12, 15])]  # [F0], [F1]
    if batched_ids:
        # JointDataLoader._get_next_sample retains this axis for tensor-origin metadata.
        ids = [value[None] for value in ids]  # [1,F0], [1,F1]
    pixels = [torch.empty(1, 3, 3, 32, 32)] * 4  # each [1,C,F,H,W]
    latents = [torch.empty(1, 4, 3, 2, 2)] * 4  # each [1,C,T,H,W]
    raw_lidar = [torch.empty(1, 3, frames, 128, 1808) for frames in (3, 3, 2, 2)]  # [1,3,F,H,W]
    lidar_latents = [torch.empty(1, 4, frames, 8, 113) for frames in (3, 3, 2, 2)]  # [1,4,T,H_lat,W_lat]
    _, features = prepare_rigrope_features(
        {"camera_geometry": records, "lidar_frame_indices": ids},
        raw_vision=pixels,
        latent_vision=latents,
        raw_lidar=raw_lidar,
        latent_lidar=lidar_latents,
        num_views=[1] * 4,
        vision_counts=[2, 2],
        lidar_counts=[2, 2],
        camera_tokenizer=Tokenizer(),
        lidar_tokenizer=SimpleNamespace(spatial_compression=(16, 16), get_pixel_num_frames=lambda count: count),
        projection=V1P2_TRANSFER_RANGE_PROJECTION,
    )
    torch.testing.assert_close(features[0][0, :, 0, 0, 7], torch.tensor([0.0, 0.1, 0.2]))
    torch.testing.assert_close(features[2][0, :, 0, 0, 7], torch.tensor([0.0, 0.1]))
    torch.testing.assert_close(features[0], features[1])
    torch.testing.assert_close(features[2], features[3])
    torch.testing.assert_close(features[0][:, :2, ..., 3:6], -features[2][..., 3:6])
