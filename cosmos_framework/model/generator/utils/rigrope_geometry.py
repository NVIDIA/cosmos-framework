# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -----------------------------------------------------------------------------

"""Adapt native camera and Pandar128 calibration to VRAD's eight ray features.

Only calibration and capture coordinates enter this path. RGB values, LiDAR
ranges, and target depth are deliberately absent from the encoding. Ego-motion is
absent for static-extrinsic rigs; a pose-world rig (a scene-fixed frame shared by
every camera) instead carries each camera's per-frame pose, so moving cameras such
as wrist mounts keep their true rays. A pose-world rig can instead be read about
itself at every encoded frame (``pose_world_frame="rig_per_frame"``), which drops
the motion the cameras share and keeps only their motion against each other.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from torch.nn import functional as F

from cosmos_framework.data.generator.multiview.camera_geometry import (
    PER_FRAME_CAMERA_TO_RIG_KEY,
    _validate_calibration,
)
from cosmos_framework.data.generator.multiview.camera_geometry_contract import validate_camera_to_world_matrix
from cosmos_framework.model.generator.mot.rigrope import rig_features
from cosmos_framework.model.generator.utils.camera_rays import (
    SUPPORTED_RAY_CAMERA_MODELS,
    token_pixel_grid,
    unproject_calibration,
)
from cosmos_framework.utils.generator.image_resize import get_vision_data_resolution
from cosmos_framework.model.generator.tokenizers.lidar.geometry import pandar128_ray_directions
from cosmos_framework.model.generator.tokenizers.lidar.range_projection import LidarRangeProjectionConfig

MomentNormalization = Literal["fixed", "per_sample_rms"]
PoseWorldFrame = Literal["scene", "rig_per_frame"]

# Relabels an OpenCV camera vector (x right, y down, z forward) as forward-left-up.
_OPENCV_TO_FLU = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float64)  # [3,3]


def camera_features(
    record: Mapping[str, Any],
    rig: Mapping[str, Any],
    *,
    pixel_shape: tuple[int, int, int],
    latent_shape: tuple[int, int, int],
    num_views: int,
    tokenizer: Any,
    moment_scale_m: float | None = None,
    include_time: bool = True,
    canonical_frame: bool = False,
    moment_normalization: MomentNormalization = "fixed",
    moment_scale_floor_m: float = 0.05,
    pose_world_frame: PoseWorldFrame = "scene",
) -> torch.Tensor:  # returns [V,T,H,W,8]
    """Sample native rays at VAE centers in retained camera order, before DiT resizing.

    ``canonical_frame`` rotates a pose-world sample into its first view's first encoded camera,
    relabelled forward-left-up, and centers a static rig on its cameras; see
    ``MultiviewAttentionConfig.rigrope_canonical_frame``. ``moment_normalization="per_sample_rms"``
    replaces ``moment_scale_m`` with the sample's RMS camera distance from the moment origin,
    floored at ``moment_scale_floor_m``. ``pose_world_frame="rig_per_frame"`` instead expresses
    each encoded frame of a pose-world sample in that frame's first-view camera, relabelled
    forward-left-up, about that frame's camera centroid, which makes ``canonical_frame`` moot for
    it; see ``MultiviewAttentionConfig.rigrope_pose_world_frame``.
    """
    if moment_normalization not in ("fixed", "per_sample_rms"):
        raise ValueError(f"Unknown RigRoPE moment normalization {moment_normalization!r}")
    if pose_world_frame not in ("scene", "rig_per_frame"):
        raise ValueError(f"Unknown RigRoPE pose-world frame {pose_world_frame!r}")
    if not math.isfinite(moment_scale_floor_m) or moment_scale_floor_m <= 0:
        raise ValueError("RigRoPE moment scale floor must be finite and positive, in metres")
    total_frames, height, width = pixel_shape
    latent_total, latent_h, latent_w = latent_shape
    if total_frames % num_views or latent_total % num_views:
        raise ValueError("RigRoPE camera frames must be divisible by the view count")
    frames, latent_t = total_frames // num_views, latent_total // num_views
    keys = record["camera_keys"]
    if len(keys) != num_views or len(set(keys)) != num_views:
        raise ValueError("RigRoPE camera keys must match the physical views in packed order")
    stride = int(tokenizer.spatial_compression_factor)
    pixels = token_pixel_grid(latent_h, latent_w, stride)  # [H,W,3]
    resolution = get_vision_data_resolution((height, width))
    positions = tokenizer.get_latent_temporal_positions(
        num_pixel_frames=frames, resolution=resolution, num_latent_frames=latent_t
    )  # [T] or None
    if positions is None:
        indices = torch.tensor(
            [tokenizer.get_pixel_num_frames(i + 1, resolution=resolution) - 1 for i in range(latent_t)],
            dtype=torch.long,
        )  # [T]
    else:
        indices = (positions.detach().cpu() * tokenizer.temporal_compression_factor).round().long()  # [T]
    indices = indices.clamp(0, frames - 1)  # [T]
    timestamps = torch.as_tensor(record["frame_times_ns"], dtype=torch.float64).cpu()  # [V,F]
    time_valid = torch.as_tensor(record["time_valid"], dtype=torch.bool).cpu()  # [V,F]
    if timestamps.shape != (num_views, frames) or not time_valid.all():
        raise ValueError("RigRoPE requires valid timestamps on the selected camera window")
    origin_ns = float(timestamps.min())
    per_frame = bool(rig.get(PER_FRAME_CAMERA_TO_RIG_KEY, False))
    if per_frame:
        poses = torch.as_tensor(record["camera_to_world"], dtype=torch.float64).cpu()  # [V,F,4,4]
        pose_valid = torch.as_tensor(record["pose_valid"], dtype=torch.bool).cpu()  # [V,F]
        if poses.shape != (num_views, frames, 4, 4) or not pose_valid[:, indices].all():
            raise ValueError("A pose-world RigRoPE rig requires valid poses on every encoded camera frame")
        frame_poses = poses[:, indices]  # [V,T,4,4]
        rotations = frame_poses[..., :3, :3]  # [V,T,3,3]
        if pose_world_frame == "rig_per_frame":
            to_rig = _OPENCV_TO_FLU @ rotations[0].transpose(-1, -2)  # [T,3,3], scene -> frame-t reference FLU
            centers = frame_poses[..., :3, 3] - frame_poses[..., :3, 3].mean(dim=0)  # [V,T,3]
            rotations = to_rig @ rotations  # [V,T,3,3]
            centers = torch.einsum("tij,vtj->vti", to_rig, centers)  # [V,T,3]
        else:
            # Center on the sample's first encoded frame so moments do not depend on the world origin.
            centers = frame_poses[..., :3, 3] - frame_poses[:, 0, :3, 3].mean(dim=0)  # [V,T,3]
        if canonical_frame and pose_world_frame == "scene":
            to_canonical = _OPENCV_TO_FLU @ rotations[0, 0].T  # [3,3], scene -> reference camera FLU
            rotations = to_canonical @ rotations  # [V,T,3,3]
            centers = centers @ to_canonical.T  # [V,T,3]
    else:
        transforms = torch.tensor([rig["camera_to_rig"][key] for key in keys], dtype=torch.float64)  # [V,4,4]
        rotations = transforms[:, None, :3, :3]  # [V,1,3,3]
        centers = transforms[:, None, :3, 3]  # [V,1,3]
        if canonical_frame:
            centers = centers - centers.mean(dim=0, keepdim=True)  # [V,1,3]
    if moment_normalization == "per_sample_rms":
        moment_scale_m = max(float(centers.square().sum(dim=-1).mean().sqrt()), moment_scale_floor_m)
    rotations, centers = rotations.float(), centers.float()  # [V,T|1,3,3], [V,T|1,3]
    result: list[torch.Tensor] = []
    for view, key in enumerate(keys):
        calibration = record["calibration"][view]
        if calibration is None or calibration["camera_model"] not in SUPPORTED_RAY_CAMERA_MODELS:
            raise ValueError(f"{key}: RigRoPE requires native calibration with a supported camera model")
        # Fail-closed callers never pass a record through ``rigrope_geometry_available``, and an
        # invalid focal length would unproject to NaN rays rather than fail.
        try:
            _validate_calibration(calibration)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{key}: invalid RigRoPE calibration: {error}") from error
        affine = torch.as_tensor(record["image_from_calibration"][view], dtype=torch.float64).cpu()  # [3,3]
        native = pixels @ torch.linalg.inv(affine).T  # [H,W,3]
        rays = unproject_calibration(native[..., :2] / native[..., 2:], calibration).float()  # [H,W,3]
        times = ((timestamps[view, indices] - origin_ns) * 1e-9).float()  # [T]
        if per_frame:
            features = torch.cat(
                [
                    rig_features(
                        rays @ rotation.T,
                        center,
                        times[frame : frame + 1],
                        moment_scale_m=moment_scale_m,
                        include_time=include_time,
                    )  # [1,H,W,8]
                    for frame, (rotation, center) in enumerate(zip(rotations[view], centers[view], strict=True))
                ]
            )  # [T,H,W,8]
        else:
            features = rig_features(
                rays @ rotations[view, 0].T,
                centers[view, 0],
                times,
                moment_scale_m=moment_scale_m,
                include_time=include_time,
            )  # [T,H,W,8]
        result.append(features)
    return torch.stack(result)  # [V,T,H,W,8]


def lidar_features(
    rig: Mapping[str, Any],
    *,
    pixel_shape: tuple[int, int, int],
    latent_shape: tuple[int, int, int],
    tokenizer: Any,
    projection: LidarRangeProjectionConfig,
    times: torch.Tensor,  # [F], clip-relative seconds
    moment_scale_m: float | None = None,
    include_time: bool = True,
) -> torch.Tensor:  # returns [1,T,H,W,8]
    """Use calibrated angular rays including the tokenizer's actual circular padding."""
    frames, height, width = pixel_shape
    latent_t, latent_h, latent_w = latent_shape
    if projection.model_width_transform != "circular_pad" or (height, width) != (
        projection.semantic_height,
        projection.model_width,
    ):
        raise ValueError("RigRoPE LiDAR requires the configured native angular grid with circular padding")
    rays = torch.from_numpy(pandar128_ray_directions(range_projection=projection))  # [H,W_semantic,3]
    padding = projection.model_width - projection.semantic_width
    left = padding // 2
    columns = (torch.arange(width) - left) % projection.semantic_width  # [W_model]
    padded = rays[:, columns].permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W_model]
    spatial = tokenizer.spatial_compression if hasattr(tokenizer, "spatial_compression") else None
    spatial = spatial() if callable(spatial) else spatial
    stride_h, stride_w = spatial or (tokenizer.spatial_compression_factor,) * 2
    ys = (torch.arange(latent_h, dtype=torch.float32) + 0.5) * stride_h  # [H_lat]
    xs = (torch.arange(latent_w, dtype=torch.float32) + 0.5) * stride_w  # [W_lat]
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H_lat,W_lat], [H_lat,W_lat]
    grid = torch.stack((2 * xx / (width - 1) - 1, 2 * yy / (height - 1) - 1), dim=-1)  # [H_lat,W_lat,2]
    sampled = F.grid_sample(
        padded, grid[None], mode="bilinear", padding_mode="border", align_corners=True
    )  # [1,3,H_lat,W_lat]
    sampled = sampled[0].permute(1, 2, 0)  # [H_lat,W_lat,3]
    transform = torch.tensor(rig["lidar_to_rig"], dtype=torch.float32)  # [4,4]
    directions = sampled @ transform[:3, :3].T  # [H_lat,W_lat,3]
    indices = torch.tensor([tokenizer.get_pixel_num_frames(i + 1) - 1 for i in range(latent_t)])  # [T]
    if len(times) != frames or int(indices.max()) >= frames:
        raise ValueError(
            f"RigRoPE LiDAR timestamps must cover the encoded sweeps: times={tuple(times.shape)}, "
            f"pixel_frames={frames}, latent_frames={latent_t}, last_index={int(indices.max())}"
        )
    return rig_features(
        directions, transform[:3, 3], times[indices], moment_scale_m=moment_scale_m, include_time=include_time
    ).unsqueeze(0)  # [1,T,H_lat,W_lat,8]


def rigrope_geometry_available(record: Mapping[str, Any] | None, *, has_lidar: bool) -> bool:
    """Gate a whole sample on absent metadata; reject malformed supplied calibration first."""
    if record is None:
        return False
    if not isinstance(record, Mapping):
        raise ValueError("Supplied RigRoPE geometry must be a mapping")
    keys = record.get("camera_keys")
    calibration = record.get("calibration")
    available = keys is not None and calibration is not None
    if keys is not None and (not keys or len(keys) != len(set(keys))):
        raise ValueError("Supplied camera keys must identify distinct physical cameras")
    if calibration is not None:
        if keys is not None and len(calibration) != len(keys):
            raise ValueError("Supplied calibration must follow the selected camera order")
        for cal in calibration:
            if cal is None:
                available = False
                continue
            _validate_calibration(cal)
            if cal["camera_model"] not in SUPPORTED_RAY_CAMERA_MODELS:
                raise ValueError(f"RigRoPE cannot unproject camera model {cal['camera_model']!r}")
    provenance = record.get("provenance", {})
    rig = record.get("rig_geometry")
    if rig is None:
        cameras = provenance.get("cameras", [])
        rig = cameras[0].get("rig_geometry") if cameras else None
    if rig is None:
        available = False
    else:
        if not isinstance(rig, Mapping) or not isinstance(rig.get("camera_to_rig"), Mapping):
            raise ValueError("Malformed supplied sensor rig")
        for key, transform in rig["camera_to_rig"].items():
            validate_camera_to_world_matrix(tuple(tuple(row) for row in transform), label=key)
        if rig.get("lidar_to_rig") is not None:
            validate_camera_to_world_matrix(tuple(tuple(row) for row in rig["lidar_to_rig"]), label="LiDAR")
        if keys is not None and any(key not in rig["camera_to_rig"] for key in keys):
            available = False
        if has_lidar and rig.get("lidar_to_rig") is None:
            available = False
        if rig.get(PER_FRAME_CAMERA_TO_RIG_KEY, False):
            pose_valid = record.get("pose_valid")
            if pose_valid is None or record.get("camera_to_world") is None or not torch.as_tensor(pose_valid).all():
                available = False
    arrays: dict[str, torch.Tensor] = {}
    for key in ("image_from_calibration", "frame_times_ns", "time_valid", "frame_indices"):
        value = record.get(key)
        if value is None:
            if key != "frame_indices" or has_lidar:
                available = False
            continue
        tensor = torch.as_tensor(value)  # [V,3,3] or [V,F]
        expected_ndim = 3 if key == "image_from_calibration" else 2
        if tensor.ndim != expected_ndim or (keys is not None and tensor.shape[0] != len(keys)):
            raise ValueError(f"Malformed supplied {key} shape: {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite supplied {key}")
        arrays[key] = tensor
    affine = arrays.get("image_from_calibration")  # [V,3,3] or None
    if affine is not None:
        if affine.shape[1:] != (3, 3) or (torch.linalg.det(affine.double()).abs() < 1e-12).any():
            raise ValueError("Supplied image-from-calibration affine must be invertible")
    times = arrays.get("frame_times_ns")  # [V,F] or None
    if times is not None and (times.shape[1] == 0 or (times[:, 1:] < times[:, :-1]).any()):
        raise ValueError("Supplied camera timestamps must be nonempty and ordered")
    validity = arrays.get("time_valid")  # [V,F] or None
    if validity is not None:
        if times is not None and validity.shape != times.shape:
            raise ValueError("Supplied timestamp validity must match camera timestamps")
        if not validity.all():
            available = False
    fps = provenance.get("source_fps")
    if fps is not None:
        rates = torch.as_tensor(fps)  # [V]
        if rates.ndim != 1 or not torch.isfinite(rates).all() or (rates <= 0).any():
            raise ValueError("Supplied camera source FPS must be finite and positive")
        if keys is not None and rates.shape[0] != len(keys):
            raise ValueError("Supplied source FPS must follow camera order")
    elif has_lidar:
        available = False
    return available


def prepare_rigrope_features(
    data_batch: Mapping[str, Any],
    *,
    raw_vision: Sequence[torch.Tensor],  # each [1,C,V*F,H,W]
    latent_vision: Sequence[torch.Tensor],  # each [1,C,V*T,H_lat,W_lat]
    raw_lidar: Sequence[torch.Tensor] | None,  # each [1,C,F,H,W]
    latent_lidar: Sequence[torch.Tensor] | None,  # each [1,C,T,H_lat,W_lat]
    num_views: Sequence[int],
    vision_counts: Sequence[int],
    lidar_counts: Sequence[int],
    camera_tokenizer: Any,
    lidar_tokenizer: Any,
    projection: LidarRangeProjectionConfig | None,
    allow_missing: bool = False,
    moment_scale_m: float | None = None,
    include_time: bool = True,
    cross_view_only: bool = False,
    canonical_frame: bool = False,
    moment_normalization: MomentNormalization = "fixed",
    moment_scale_floor_m: float = 0.05,
    pose_world_frame: PoseWorldFrame = "scene",
) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
    """Prepare independent geometry for every logical sample and repeated control item."""
    if (canonical_frame or moment_normalization != "fixed" or pose_world_frame != "scene") and any(lidar_counts):
        # LiDAR descriptors would have to share each sample's camera frame and moment unit.
        raise ValueError("RigRoPE canonical frames and per-sample moment units are camera-only")
    records = data_batch.get("camera_geometry")
    if records is None and (allow_missing or cross_view_only):
        # Single-view camera samples need no geometry; cross-view samples still validate below.
        records = [None] * len(vision_counts)
    if records is None or len(records) != len(vision_counts) or len(lidar_counts) != len(vision_counts):
        raise ValueError("RigRoPE needs one camera_geometry record per logical sample")
    vision_result: list[torch.Tensor | None] = []
    lidar_result: list[torch.Tensor | None] = []
    vi = li = 0
    for sample, (record, vc, lc) in enumerate(zip(records, vision_counts, lidar_counts, strict=True)):
        # A camera-only single-view sample (including its control) has no cross-view pass.
        # It needs neither calibration nor descriptors, even inside a mixed packed batch.
        if cross_view_only and lc == 0 and all(views == 1 for views in num_views[vi : vi + vc]):
            vision_result.extend([None] * vc)
            vi += vc
            continue
        if allow_missing:
            available = rigrope_geometry_available(record, has_lidar=lc > 0)
            if lc and data_batch.get("lidar_frame_indices") is None:
                available = False
            if not available:
                vision_result.extend([None] * vc)
                lidar_result.extend([None] * lc)
                vi += vc
                li += lc
                continue
        if record is None:
            raise ValueError(f"Sample {sample}: missing RigRoPE camera geometry")
        rig = record.get("rig_geometry") or record["provenance"]["cameras"][0].get("rig_geometry")
        if rig is None:
            raise ValueError(f"Sample {sample}: missing calibrated physical sensor rig")
        for item in range(vi, vi + vc):
            features = camera_features(
                record,
                rig,
                pixel_shape=tuple(raw_vision[item].shape[-3:]),
                latent_shape=tuple(latent_vision[item].shape[-3:]),
                num_views=num_views[item],
                tokenizer=camera_tokenizer,
                moment_scale_m=moment_scale_m,
                include_time=include_time,
                canonical_frame=canonical_frame,
                moment_normalization=moment_normalization,
                moment_scale_floor_m=moment_scale_floor_m,
                pose_world_frame=pose_world_frame,
            )  # [V,T,H,W,8]
            vision_result.append(features)
        for item in range(li, li + lc):
            if raw_lidar is None or latent_lidar is None or projection is None or lidar_tokenizer is None:
                raise ValueError("RigRoPE LiDAR item counts disagree with the encoded stream")
            frame_ids = data_batch.get("lidar_frame_indices")
            if frame_ids is None:
                raise ValueError("RigRoPE needs selected LiDAR source-frame IDs to align capture time")
            ids = torch.as_tensor(frame_ids[sample], dtype=torch.float64).cpu()  # [1,F] or [F]
            # The joint packer preserves the singleton batch axis for tensor-origin metadata.
            if ids.ndim == 2 and ids.shape[0] == 1:
                ids = ids[0]  # [F]
            if ids.ndim != 1:
                raise ValueError(f"Sample {sample}: expected LiDAR source-frame IDs [F] or [1,F], got {ids.shape}")
            origin = int(torch.as_tensor(record["frame_indices"])[0, 0])
            source_fps = float(record["provenance"]["source_fps"][0])
            times = ((ids - origin) / source_fps).float()  # [F]
            features = lidar_features(
                rig,
                pixel_shape=tuple(raw_lidar[item].shape[-3:]),
                latent_shape=tuple(latent_lidar[item].shape[-3:]),
                tokenizer=lidar_tokenizer,
                projection=projection,
                times=times,
                moment_scale_m=moment_scale_m,
                include_time=include_time,
            )  # [1,T,H,W,8]
            lidar_result.append(features)
        vi += vc
        li += lc
    if vi != len(raw_vision) or li != len(raw_lidar or []):
        raise ValueError("RigRoPE item counts must cover all encoded sensor items")
    return vision_result, lidar_result
