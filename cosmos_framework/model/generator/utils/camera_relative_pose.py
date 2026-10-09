# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -----------------------------------------------------------------------------

"""Prepare camera extrinsics for attention, outside compiled model regions.

CameraGeometry poses are OpenCV camera-to-world matrices in meters. Attention
uses reference-to-camera matrices, centered on the same primary camera at each
instant. The dense camera branch only compares cameras at the same instant, so
this removes large world translations without changing its relative transforms.
Intrinsics and lens distortion are not used by this extrinsics-only ablation.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from cosmos_framework.utils.easy_io import easy_io
from cosmos_framework.data.generator.multiview.camera_geometry import (
    PER_FRAME_CAMERA_TO_RIG_KEY,
    _validate_calibration,
)
from cosmos_framework.model.generator.utils.rigrope_geometry import (
    MomentNormalization,
    PoseWorldFrame,
    rigrope_geometry_available,
)
from cosmos_framework.utils.generator.image_resize import get_vision_data_resolution


def invert_rigid_transform(matrix: torch.Tensor) -> torch.Tensor:  # [...,4,4] -> [...,4,4]
    rotation = matrix[..., :3, :3].transpose(-1, -2)  # [...,3,3]
    translation = -(rotation @ matrix[..., :3, 3:4])  # [...,3,1]
    upper = torch.cat((rotation, translation), dim=-1)  # [...,3,4]
    return torch.cat((upper, matrix[..., 3:4, :]), dim=-2)  # [...,4,4]


def validate_camera_poses(poses: torch.Tensor) -> None:  # [V,T,4,4]
    """Reject invalid SE(3) matrices before the compiled attention path."""
    if poses.ndim != 4 or poses.shape[-2:] != (4, 4) or min(poses.shape[:2]) < 1:
        raise ValueError(f"camera_to_world must have shape [V,T,4,4], got {tuple(poses.shape)}")
    if not torch.isfinite(poses).all():
        raise ValueError("camera_to_world must contain finite values")
    bottom = poses.new_tensor([0.0, 0.0, 0.0, 1.0]).expand_as(poses[..., 3, :])  # [V,T,4]
    rotation = poses[..., :3, :3]  # [V,T,3,3]
    identity = torch.eye(3, dtype=poses.dtype, device=poses.device).expand_as(rotation)  # [V,T,3,3]
    if not torch.allclose(poses[..., 3, :], bottom, atol=1e-6, rtol=0):
        raise ValueError("camera_to_world must have homogeneous bottom row [0,0,0,1]")
    if not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-4, rtol=0):
        raise ValueError("camera_to_world rotations must be orthonormal")
    if not torch.allclose(torch.linalg.det(rotation), poses.new_ones(poses.shape[:2]), atol=1e-4, rtol=0):
        raise ValueError("camera_to_world rotations must have determinant +1")


def _latent_frame_indices(
    tokenizer: Any, frames: int, latent_count: int, pixel_hw: tuple[int, int]
) -> torch.Tensor:  # [F]
    """The source frame each latent frame's pose is read at: its tokenizer patch's right edge."""
    resolution = get_vision_data_resolution(pixel_hw)
    positions = tokenizer.get_latent_temporal_positions(
        num_pixel_frames=frames, resolution=resolution, num_latent_frames=latent_count
    )  # [F] or None
    if positions is None:
        indices = torch.tensor(
            [tokenizer.get_pixel_num_frames(index + 1, resolution=resolution) - 1 for index in range(latent_count)],
            dtype=torch.long,
        )  # [F]
    else:
        indices = (positions.detach().cpu() * tokenizer.temporal_compression_factor).round().long()  # [F]
    # Noncausal tokenizer padding repeats the first/last real frame.
    return indices.clamp(0, frames - 1)  # [F]


def prepare_camera_relative_poses(
    geometry: Sequence[Mapping[str, Any] | None] | None,
    *,
    pixel_shapes: Sequence[tuple[int, int, int]],
    latent_frames: Sequence[int],
    num_views: Sequence[int],
    items_per_sample: Sequence[int],
    tokenizer: Any,
) -> list[torch.Tensor | None]:  # each tensor: [V,F,4,4]
    """Select poses at tokenizer patch right edges; duplicate them for control/target.

    ``pixel_shapes`` contains camera-major (V*T,H,W) shapes. ``geometry`` already
    describes the final sampled frames, so its time axis indexes those frames, not
    the original video's frame IDs. Missing poses are an error for multiview samples;
    single-camera samples take the exact existing attention path and need none.
    """
    if sum(items_per_sample) != len(num_views) or not (len(num_views) == len(pixel_shapes) == len(latent_frames)):
        raise ValueError("Camera pose metadata must match the flattened vision item layout")
    if geometry is not None and len(geometry) != len(items_per_sample):
        raise ValueError("camera_geometry must contain one record per logical sample")
    result: list[torch.Tensor | None] = []
    cursor = 0
    for sample, count in enumerate(items_per_sample):
        item_views = num_views[cursor : cursor + count]
        if not item_views or max(item_views) <= 1:
            result.extend([None] * count)
            cursor += count
            continue
        record = None if geometry is None else geometry[sample]
        if record is None:
            raise ValueError(
                f"Sample {sample}: use_camera_relative_pose_emb requires camera_geometry; "
                "enable load_camera_geometry in the loader or provide inference camera poses"
            )
        keys = list(record["camera_keys"])
        if len(keys) != len(set(keys)) or any(views != len(keys) for views in item_views):
            raise ValueError(f"Sample {sample}: camera_geometry camera order/count does not match the sampled rig")
        poses = torch.as_tensor(record["camera_to_world"], dtype=torch.float64).detach()  # [V,T,4,4]
        valid = torch.as_tensor(record["pose_valid"], device=poses.device, dtype=torch.bool)  # [V,T]
        if poses.ndim != 4 or poses.shape[0] != len(keys) or valid.shape != poses.shape[:2]:
            raise ValueError(f"Sample {sample}: camera pose shapes disagree with camera_keys")
        reference = _reference_view(keys)
        for item in range(cursor, cursor + count):
            views = num_views[item]
            total_frames, height, width = pixel_shapes[item]
            if total_frames % views or latent_frames[item] % views:
                raise ValueError("Camera pixel and latent frame counts must be divisible by the view count")
            frames, latent_count = total_frames // views, latent_frames[item] // views
            if poses.shape[1] != frames:
                raise ValueError(f"Sample {sample}: camera_geometry has {poses.shape[1]} frames, video has {frames}")
            indices = _latent_frame_indices(tokenizer, frames, latent_count, (height, width)).to(poses.device)  # [F]
            if indices.shape != (latent_count,) or not valid[:, indices].all():
                raise ValueError(f"Sample {sample}: camera poses are missing at selected latent frames")
            selected = poses[:, indices]  # [V,F,4,4]
            validate_camera_poses(selected)
            anchor = selected[reference : reference + 1]  # [1,F,4,4]
            relative = (invert_rigid_transform(selected) @ anchor).float()  # [V,F,4,4]
            result.append(relative)
        cursor += count
    return result


def _reference_view(keys: Sequence[str]) -> int:
    # Prefer front-wide, then front fisheye; other rigs choose a stable physical
    # identity so reordering the selected cameras does not change the reference.
    reference_key = next(
        (key for key in ("camera_front_wide_120fov", "camera_front_fisheye_200fov") if key in keys),
        min(keys),
    )
    return list(keys).index(reference_key)


def prepare_prope_cross_view_poses(
    geometry: Sequence[Mapping[str, Any] | None] | None,
    *,
    pixel_shapes: Sequence[tuple[int, int, int]],
    latent_frames: Sequence[int],
    num_views: Sequence[int],
    items_per_sample: Sequence[int],
    tokenizer: Any,
    allow_missing: bool,
    translation_normalization: MomentNormalization = "fixed",
    translation_scale_m: float = 25.0,
    translation_floor_m: float = 0.05,
    pose_world_frame: PoseWorldFrame = "scene",
) -> list[torch.Tensor | None]:  # each tensor: [V,F,4,4]
    """Reference-to-camera transforms for ``geometry_position_encoding="prope_cross_view"``.

    Extrinsics come from where RigRoPE reads them: a pose-world rig's per-frame
    ``camera_to_world`` at the latent frames, or a static rig's ``camera_to_rig``. With
    ``allow_missing``, a sample gets geometry exactly when RigRoPE would give it some, so the
    two modes train the same posed population; the rest are ``None`` and take the identity.
    ``pose_world_frame="rig_per_frame"`` anchors each frame on the reference camera at that
    frame, which drops the motion the cameras share across the band's instants, as RigRoPE's
    mode of that name does; ``"scene"`` anchors every frame on the reference camera at the
    first latent frame and so keeps it. Translations are divided by ``translation_scale_m``,
    or under ``"per_sample_rms"`` by the RMS camera distance from the per-frame (or
    first-frame) centroid, floored at ``translation_floor_m``.
    """
    if translation_normalization not in ("fixed", "per_sample_rms"):
        raise ValueError(f"Unknown PRoPE translation normalization {translation_normalization!r}")
    if pose_world_frame not in ("scene", "rig_per_frame"):
        raise ValueError(f"Unknown PRoPE pose-world frame {pose_world_frame!r}")
    for value, label in ((translation_scale_m, "scale"), (translation_floor_m, "floor")):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"PRoPE translation {label} must be finite and positive, in metres")
    if sum(items_per_sample) != len(num_views) or not (len(num_views) == len(pixel_shapes) == len(latent_frames)):
        raise ValueError("Camera pose metadata must match the flattened vision item layout")
    records: Sequence[Mapping[str, Any] | None] = (
        [None] * len(items_per_sample) if geometry is None and allow_missing else (geometry or [])
    )
    if len(records) != len(items_per_sample):
        raise ValueError("camera_geometry must contain one record per logical sample")
    result: list[torch.Tensor | None] = []
    cursor = 0
    for sample, (record, count) in enumerate(zip(records, items_per_sample, strict=True)):
        item_views = num_views[cursor : cursor + count]
        multiview = bool(item_views) and max(item_views) > 1
        if not multiview or (allow_missing and not rigrope_geometry_available(record, has_lidar=False)):
            result.extend([None] * count)
            cursor += count
            continue
        if record is None:
            raise ValueError(f"Sample {sample}: prope_cross_view requires camera_geometry for multiview samples")
        keys = list(record["camera_keys"])
        if len(keys) != len(set(keys)) or any(views != len(keys) for views in item_views):
            raise ValueError(f"Sample {sample}: camera_geometry camera order/count does not match the sampled rig")
        rig = record.get("rig_geometry") or record["provenance"]["cameras"][0].get("rig_geometry")
        if rig is None:
            raise ValueError(f"Sample {sample}: missing calibrated physical sensor rig")
        per_frame = bool(rig.get(PER_FRAME_CAMERA_TO_RIG_KEY, False))
        reference = _reference_view(keys)
        for item in range(cursor, cursor + count):
            views = num_views[item]
            total_frames, height, width = pixel_shapes[item]
            if total_frames % views or latent_frames[item] % views:
                raise ValueError("Camera pixel and latent frame counts must be divisible by the view count")
            frames, latent_count = total_frames // views, latent_frames[item] // views
            if per_frame:
                poses = torch.as_tensor(record["camera_to_world"], dtype=torch.float64).cpu()  # [V,T,4,4]
                valid = torch.as_tensor(record["pose_valid"], dtype=torch.bool).cpu()  # [V,T]
                indices = _latent_frame_indices(tokenizer, frames, latent_count, (height, width))  # [F]
                if poses.shape != (views, frames, 4, 4) or not valid[:, indices].all():
                    raise ValueError(f"Sample {sample}: a pose-world rig needs valid poses on every latent frame")
                selected = poses[:, indices]  # [V,F,4,4]
            else:
                static = torch.tensor([rig["camera_to_rig"][key] for key in keys], dtype=torch.float64)  # [V,4,4]
                selected = static[:, None].expand(-1, latent_count, -1, -1)  # [V,F,4,4]
            validate_camera_poses(selected)
            centers = selected[..., :3, 3]  # [V,F,3]
            if per_frame and pose_world_frame == "scene":
                anchor = selected[reference : reference + 1, :1]  # [1,1,4,4]
                spread = centers - centers[:, :1].mean(dim=0, keepdim=True)  # [V,F,3]
            else:
                anchor = selected[reference : reference + 1]  # [1,F,4,4]
                spread = centers - centers.mean(dim=0, keepdim=True)  # [V,F,3]
            unit = translation_scale_m
            if translation_normalization == "per_sample_rms":
                unit = max(float(spread.square().sum(dim=-1).mean().sqrt()), translation_floor_m)
            relative = invert_rigid_transform(selected) @ anchor  # [V,F,4,4]
            relative[..., :3, 3] /= unit
            result.append(relative.float())
        cursor += count
    return result


def load_camera_pose_geometry(
    camera_keys: Sequence[str],
    pose_paths: Sequence[str | None],
    *,
    num_frames: int,
    start_frame: int = 0,
    fps: float | None = None,
    image_hw: tuple[int, int] | None = None,
    require_calibration_mapping: bool = False,
    require_sensor_rig: bool = False,
) -> dict[str, Any]:
    """Read explicit per-camera pose JSONs for inference/validation.

    Each JSON contains ``camera_to_world`` as [T,4,4] (or one static [4,4])
    and optional ``pose_valid`` [T] (a single flag for a static pose). Frames must
    match the associated video's FPS and origin, unless ``source_frame_indices``
    maps the window onto the sidecar's own frames at its ``fps``; the returned
    ``provenance["source_fps"]`` is then that source clock. The selected camera
    order is preserved. No implied identity fallback.
    """
    if not camera_keys or len(camera_keys) != len(pose_paths) or len(set(camera_keys)) != len(camera_keys):
        raise ValueError("Pose paths must match unique camera keys in selected order")
    if num_frames < 1 or start_frame < 0:
        raise ValueError("Camera pose windows require num_frames > 0 and start_frame >= 0")
    all_poses: list[torch.Tensor] = []
    all_valid: list[torch.Tensor] = []
    calibrations: list[dict[str, Any]] = []
    affines: list[torch.Tensor] = []  # each [3,3]
    source_starts: list[int] = []
    sensor_rig: dict[str, Any] | None = None
    selected_source_indices: torch.Tensor | None = None
    selected_frame_times_ns: torch.Tensor | None = None
    # A source mapping's frame IDs count frames of the sidecar's own clock, not the output video's.
    mapped_source_fps: list[float] = []
    uses_source_mapping: bool | None = None
    for key, path in zip(camera_keys, pose_paths, strict=True):
        if path is None:
            raise ValueError(f"Camera {key!r} needs pose_path for use_camera_relative_pose_emb")
        payload = easy_io.load(path)
        poses = torch.as_tensor(payload["camera_to_world"], dtype=torch.float64)  # [T,4,4] or [4,4]
        if poses.ndim not in (2, 3) or poses.shape[-2:] != (4, 4) or poses.numel() == 0:
            raise ValueError(f"Camera {key!r}: camera_to_world must have shape [T,4,4] or [4,4]")
        if require_sensor_rig:
            current_rig = payload.get("rig_geometry")
            if current_rig is None or key not in current_rig.get("camera_to_rig", {}):
                # PAIBench direct pose sidecars are canonical camera-to-world streams.  For
                # non-MADS samples these may be moving cameras, so the shared world is the
                # only correct rig and each selected frame carries its own camera transform.
                current_rig = {
                    "camera_to_rig": {key: (poses if poses.ndim == 2 else poses[0]).tolist()},
                    "per_frame_camera_to_rig": True,
                }
            if sensor_rig is not None:
                if current_rig.get("per_frame_camera_to_rig") and sensor_rig.get("per_frame_camera_to_rig"):
                    sensor_rig["camera_to_rig"].update(current_rig["camera_to_rig"])
                elif current_rig != sensor_rig:
                    raise ValueError("Selected pose sidecars contain different physical rigs")
            else:
                sensor_rig = current_rig
        if payload.get("camera_key", key) != key:
            raise ValueError(f"Camera {key!r}: pose_path belongs to a different camera")
        source_frame_indices = payload.get("source_frame_indices")
        # frame_indices and frame_times_ns are shared across views, so one convention must hold for all.
        if uses_source_mapping is not None and uses_source_mapping != (source_frame_indices is not None):
            raise ValueError("Selected cameras must all provide source_frame_indices, or none of them")
        uses_source_mapping = source_frame_indices is not None
        if source_frame_indices is not None:
            if "fps" not in payload or not float(payload["fps"]) > 0:
                raise ValueError(f"Camera {key!r}: source_frame_indices require a positive source fps")
            source_indices = torch.as_tensor(source_frame_indices, dtype=torch.int64)
            if source_indices.ndim != 1 or len(source_indices) < start_frame + num_frames:
                raise ValueError(f"Camera {key!r}: source_frame_indices do not cover the requested window")
            source_indices = source_indices[start_frame : start_frame + num_frames]
            if selected_source_indices is not None and not torch.equal(selected_source_indices, source_indices):
                raise ValueError("Selected cameras have different source-frame mappings")
            selected_source_indices = source_indices
        elif fps is not None and "fps" in payload and not math.isclose(float(payload["fps"]), fps):
            raise ValueError(f"Camera {key!r}: pose FPS {payload['fps']} does not match video FPS {fps}")
        if poses.ndim == 2:
            # One static transform holds at every frame, so it needs no pose table covering the
            # window, however large its absolute source-frame IDs.
            static_valid = torch.as_tensor(payload.get("pose_valid", True), dtype=torch.bool)  # []
            if static_valid.ndim != 0:
                raise ValueError(f"Camera {key!r}: a static camera_to_world takes a single pose_valid flag")
            all_poses.append(poses.expand(num_frames, -1, -1))  # [F,4,4]
            all_valid.append(static_valid.expand(num_frames))  # [F]
        else:
            required_pose_frames = (
                int(source_indices.max()) + 1 if source_frame_indices is not None else start_frame + num_frames
            )
            valid = torch.as_tensor(payload.get("pose_valid", [True] * len(poses)), dtype=torch.bool)  # [T]
            if poses.shape[0] < required_pose_frames or valid.shape != poses.shape[:1]:
                raise ValueError(f"Camera {key!r}: pose sequence does not cover the requested video window")
            window = (
                source_indices if source_frame_indices is not None else slice(start_frame, start_frame + num_frames)
            )
            all_poses.append(poses[window])  # [F,4,4]
            all_valid.append(valid[window])  # [F]
        if source_frame_indices is not None:
            source_starts.append(int(source_indices[0]))
            source_fps = float(payload["fps"])
            if mapped_source_fps and not math.isclose(source_fps, mapped_source_fps[0]):
                raise ValueError("Selected cameras have different source FPS")
            mapped_source_fps.append(source_fps)
            frame_times = (source_indices.double() * (1e9 / source_fps)).round().long()
            if selected_frame_times_ns is not None and not torch.equal(selected_frame_times_ns, frame_times):
                raise ValueError("Selected cameras have different frame-time mappings")
            selected_frame_times_ns = frame_times
        else:
            source_starts.append(int(payload.get("source_frame_start", 0)) + start_frame)
        if image_hw is not None:
            # Caption and relative-ray geometry require native calibration and the
            # affine into staged media. Pose-only JSONs remain valid for extrinsics.
            if require_calibration_mapping and not payload.get("calibration_mapping_verified", False):
                raise ValueError(
                    f"Camera {key!r}: relative-ray inference requires a verified calibration-to-media mapping"
                )
            if not all(field in payload for field in ("calibration", "image_hw", "image_from_calibration")):
                raise ValueError(
                    f"Camera {key!r}: calibrated geometry requires calibration and image metadata in pose_path"
                )
            try:
                _validate_calibration(payload["calibration"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Camera {key!r}: invalid calibration in pose_path: {error}") from error
            calibrations.append(payload["calibration"])
            source_h, source_w = payload["image_hw"]
            height, width = image_hw
            if min(source_h, source_w, height, width) <= 0:
                raise ValueError("Camera image dimensions must be positive")
            ratio = max(width / source_w, height / source_h)
            resize_h, resize_w = math.ceil(ratio * source_h), math.ceil(ratio * source_w)
            top, left = round((resize_h - height) / 2), round((resize_w - width) / 2)
            sx, sy = resize_w / source_w, resize_h / source_h
            resize = torch.tensor(  # [3,3]
                [[sx, 0, (sx - 1) / 2 - left], [0, sy, (sy - 1) / 2 - top], [0, 0, 1]], dtype=torch.float64
            )
            affine = torch.as_tensor(payload["image_from_calibration"], dtype=torch.float64)  # [3,3]
            if affine.shape != (3, 3) or not torch.isfinite(affine).all():
                raise ValueError(f"Camera {key!r}: invalid image_from_calibration")
            affines.append(resize @ affine)  # [3,3]
    if len(set(source_starts)) != 1:
        raise ValueError("Selected camera pose sidecars have different source frame origins")
    geometry: dict[str, Any] = {
        "camera_keys": list(camera_keys),
        "camera_to_world": torch.stack(all_poses),  # [V,F,4,4]
        "pose_valid": torch.stack(all_valid),  # [V,F]
    }
    if not geometry["pose_valid"].all():
        raise ValueError("Selected camera pose window contains missing poses")
    validate_camera_poses(geometry["camera_to_world"])
    if image_hw is not None:
        geometry.update(
            calibration=calibrations,
            image_from_calibration=torch.stack(affines),  # [V,3,3]
            image_hw=torch.tensor([image_hw] * len(camera_keys)),  # [V,2]
            frame_indices=(
                selected_source_indices[None].repeat(len(camera_keys), 1)
                if selected_source_indices is not None
                else torch.tensor(source_starts)[:, None] + torch.arange(num_frames)[None, :]
            ),  # [V,F]
        )
    if require_sensor_rig:
        if fps is None or fps <= 0 or image_hw is None:
            raise ValueError("RigRoPE inference requires FPS and calibrated image dimensions")
        geometry.update(
            rig_geometry=sensor_rig,
            frame_times_ns=(
                selected_frame_times_ns[None].repeat(len(camera_keys), 1)
                if selected_frame_times_ns is not None
                else (torch.arange(num_frames, dtype=torch.float64) * (1e9 / fps))
                .round()
                .long()[None]
                .repeat(len(camera_keys), 1)
            ),  # [V,F]
            time_valid=torch.ones((len(camera_keys), num_frames), dtype=torch.bool),  # [V,F]
            provenance={"source_fps": mapped_source_fps or [fps] * len(camera_keys)},
        )
    return geometry
