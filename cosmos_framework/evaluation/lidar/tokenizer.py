# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Model-free reconstruction metrics on explicit metric LiDAR range views."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import KDTree

from cosmos_framework.model.generator.tokenizers.lidar.geometry import PANDAR128_ELEVATIONS_DEG

METRICS: tuple[str, ...] = ("range_mae_m", "range_median_ae_m", "depth_abs_rel", "chamfer_m", "validity_iou")


def load_projection(path: Path | None) -> dict[str, Any]:
    """Read sensor calibration; elevation files are relative to the JSON file."""
    configured = json.loads(path.read_text()) if path is not None else {}
    if not isinstance(configured, dict):
        raise ValueError("Projection JSON must be an object")
    if "elevation_file" in configured:
        if path is None or "elevation_degrees" in configured:
            raise ValueError("Supply exactly one of elevation_file and elevation_degrees")
        elevation_path = path.parent / configured["elevation_file"]
        configured["elevation_degrees"] = (
            np.load(elevation_path, allow_pickle=False).tolist()
            if elevation_path.suffix == ".npy"
            else json.loads(elevation_path.read_text())
        )
    if configured.get("sensor", "pandar128") != "pandar128" and "elevation_degrees" not in configured:
        raise ValueError("A different sensor requires explicit elevation_degrees or elevation_file")
    if configured.get("coordinate_system", "x_forward_y_left_z_up") != "x_forward_y_left_z_up":
        raise ValueError("Convert ranges/calibration to x-forward, y-left, z-up before scoring")
    return configured


def load_clip(path: Path, projection: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read unmasked metric range [T,H,W], boolean validity and integer sweep IDs."""
    with np.load(path, allow_pickle=False) as arrays:
        ranges = np.asarray(arrays["range_m"], dtype=np.float64)  # [T,H,W]
        valid = np.asarray(arrays["valid"])  # [T,H,W]
        ids = np.asarray(arrays["sweep_ids"])  # [T]
    if ranges.ndim != 3 or min(ranges.shape) < 1:
        raise ValueError(f"{path}: range_m must have nonempty shape [T,H,W]")
    if valid.dtype != np.bool_ or valid.shape != ranges.shape:
        raise ValueError(f"{path}: valid must be boolean with the same shape as range_m")
    if ids.shape != (len(ranges),) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"{path}: sweep_ids must be integer [T]")
    if not np.isfinite(ranges).all():
        raise ValueError(f"{path}: range_m must be finite, including unmasked predictions")
    semantic_width = int(projection.get("semantic_width", ranges.shape[-1]))
    model_width = int(projection.get("model_width", semantic_width))
    if ranges.shape[-1] != semantic_width:
        if (
            ranges.shape[-1] != model_width
            or model_width <= semantic_width
            or (model_width - semantic_width) % 2
            or projection.get("model_width_transform", "circular_pad") != "circular_pad"
        ):
            raise ValueError(f"{path}: width differs from declared semantic grid; no implicit resizing")
        left = (model_width - semantic_width) // 2
        ranges = ranges[..., left : left + semantic_width]  # [T,H,W_semantic]
        valid = valid[..., left : left + semantic_width]  # [T,H,W_semantic]
    return ranges, valid, ids


def range_to_points(ranges: np.ndarray, valid: np.ndarray, projection: dict[str, Any]) -> np.ndarray:
    """Unproject [H,W] to sensor XYZ [N,3] metres with calibrated row order."""
    elevations = np.asarray(projection.get("elevation_degrees", PANDAR128_ELEVATIONS_DEG), dtype=np.float64)  # [H]
    if elevations.shape != (ranges.shape[0],) or not np.isfinite(elevations).all():
        raise ValueError("Row count must match the finite elevation calibration; no generic vertical FOV fallback")
    elevation = np.deg2rad(elevations)[:, None]  # [H,1]
    azimuth = np.deg2rad(
        np.linspace(
            float(projection.get("azimuth_start_degrees", 180.0)),
            float(projection.get("azimuth_end_degrees", -180.0)),
            ranges.shape[1],
            endpoint=bool(projection.get("azimuth_endpoint", False)),
        )
    )[None, :]  # [1,W]
    directions = np.stack(
        (
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.broadcast_to(np.sin(elevation), ranges.shape),
        ),
        axis=-1,
    )  # [H,W,3]
    mask = valid & (ranges > 0)  # [H,W]
    return (directions[mask] * ranges[mask, None]).astype(np.float32)  # [N,3]


def _chamfer(first: np.ndarray, second: np.ndarray, max_points: int, seed: int) -> float | None:
    """Half-symmetric, non-squared mean nearest-neighbor Euclidean distance."""
    if not len(first) or not len(second):
        return None
    if len(first) > max_points:
        first = first[np.random.default_rng(seed).choice(len(first), max_points, replace=False)]  # [P,3]
    if len(second) > max_points:
        second = second[np.random.default_rng(seed + 1).choice(len(second), max_points, replace=False)]  # [P,3]
    forward = KDTree(second).query(first, k=1, eps=0.0, p=2, workers=1)[0]  # [N]
    reverse = KDTree(first).query(second, k=1, eps=0.0, p=2, workers=1)[0]  # [M]
    return float((forward.mean() + reverse.mean()) / 2)


def _mean(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return float(np.mean(available)) if available else None


def _check_reference(ranges: np.ndarray, valid: np.ndarray, projection: dict[str, Any]) -> None:
    minimum, maximum = float(projection.get("min_range_m", 5.0)), float(projection.get("max_range_m", 100.0))
    if not 0 < minimum < maximum:
        raise ValueError("Metric range must satisfy 0 < min_range_m < max_range_m")
    if np.any(valid & ((ranges < minimum) | (ranges > maximum))):
        raise ValueError("GT valid mask includes returns outside the declared inclusive metric band")


def evaluate_tokenizer(
    manifest: Path,
    aggregation: str = "macro",
    projection_path: Path | None = None,
    max_points: int = 50_000,
) -> dict[str, Any]:
    """Score JSON NPZ pairs, with frame→clip macro or pooled MAE/IoU micro."""
    if aggregation not in {"macro", "micro"} or max_points <= 0:
        raise ValueError("aggregation must be macro or micro and max_points must be positive")
    entries = json.loads(manifest.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must be a nonempty JSON list of {id,reference,prediction}")
    projection = load_projection(projection_path)
    seen: set[str] = set()
    clips: list[dict[str, Any]] = []
    total_error = 0.0
    total_valid = total_intersection = total_union = 0
    for entry in entries:
        clip_id = entry["id"]
        if not isinstance(clip_id, str) or not clip_id or clip_id in seen:
            raise ValueError("Manifest clip IDs must be distinct nonempty strings")
        seen.add(clip_id)
        gt, gt_valid, ids = load_clip(manifest.parent / entry["reference"], projection)  # [T,H,W], [T]
        pred, pred_valid, pred_ids = load_clip(manifest.parent / entry["prediction"], projection)  # [T,H,W], [T]
        if gt.shape != pred.shape or not np.array_equal(ids, pred_ids):
            raise ValueError(f"{clip_id}: shape or exact source sweep-ID alignment differs")
        _check_reference(gt, gt_valid, projection)
        cd_gt, cd_valid = gt, gt_valid  # [T,H,W]
        if "chamfer_reference" in entry:
            # A fixed physical GT canvas permits comparisons across tokenizer widths.
            cd_projection = {
                key: value for key, value in projection.items() if key not in {"semantic_width", "model_width"}
            }
            cd_gt, cd_valid, cd_ids = load_clip(
                manifest.parent / entry["chamfer_reference"], cd_projection
            )  # [T,H,W_cd], [T]
            if not np.array_equal(ids, cd_ids):
                raise ValueError(f"{clip_id}: Chamfer reference sweep IDs differ")
            _check_reference(cd_gt, cd_valid, projection)
        errors = np.abs(pred - gt)  # [T,H,W]
        valid_count = int(gt_valid.sum())
        intersection = int((gt_valid & pred_valid).sum())
        union = int((gt_valid | pred_valid).sum())
        error_sum = float(errors[gt_valid].sum())
        total_error += error_sum
        total_valid += valid_count
        total_intersection += intersection
        total_union += union
        frames: list[dict[str, float | None]] = []
        if aggregation == "macro":
            for index in range(len(gt)):
                valid = gt_valid[index]  # [H,W]
                if not valid.any():
                    raise ValueError(f"{clip_id}, frame {index}: no GT-valid rays; macro range errors are undefined")
                absolute = errors[index][valid]  # [N_valid]
                frame_union = int((valid | pred_valid[index]).sum())
                frames.append(
                    {
                        "range_mae_m": float(absolute.mean()),
                        "range_median_ae_m": float(np.median(absolute)),
                        "depth_abs_rel": float((absolute / gt[index][valid]).mean()),
                        "chamfer_m": _chamfer(
                            range_to_points(cd_gt[index], cd_valid[index], projection),
                            range_to_points(pred[index], pred_valid[index], projection),
                            max_points,
                            2 * index,
                        ),
                        "validity_iou": float((valid & pred_valid[index]).sum()) / frame_union if frame_union else 1.0,
                    }
                )
        row: dict[str, Any] = {
            "id": clip_id,
            "frames": len(gt),
            "gt_valid_rays": valid_count,
            "range_shape": list(gt.shape),
            "chamfer_reference_width": cd_gt.shape[-1],
        }
        row.update(
            {name: _mean([frame[name] for frame in frames]) for name in METRICS}
            if aggregation == "macro"
            else {
                "range_mae_m": error_sum / valid_count if valid_count else None,
                "validity_iou": intersection / union if union else 1.0,
            }
        )
        if aggregation == "macro":
            row["chamfer_frames"] = sum(frame["chamfer_m"] is not None for frame in frames)
            row["per_frame"] = frames
        clips.append(row)
    if not total_valid:
        raise ValueError("Corpus has no GT-valid rays")
    corpus = (
        {name: _mean([clip[name] for clip in clips]) for name in METRICS}
        if aggregation == "macro"
        else {
            "range_mae_m": total_error / total_valid,
            "validity_iou": total_intersection / total_union if total_union else 1.0,
        }
    )
    return {
        "aggregation": aggregation,
        "corpus": corpus,
        "clips": len(clips),
        "frames": sum(clip["frames"] for clip in clips),
        "per_clip": clips,
        "gt_valid_rays": total_valid,
        "validity_intersection": total_intersection,
        "validity_union": total_union,
        "chamfer_clips": sum(clip.get("chamfer_m") is not None for clip in clips) if aggregation == "macro" else None,
        "protocol": {
            "range_errors": "GT-valid rays; prediction range remains unmasked",
            "geometry": "sensor XYZ metres, x-forward/y-left/z-up; calibrated elevation rows",
            "macro": "equal frame mean within clip, then equal clip mean",
            "micro": "pooled absolute-error sum / GT-valid rays; pooled mask intersection / union",
            "chamfer": "half-symmetric non-squared Euclidean; null for either empty cloud",
            "chamfer_max_points_per_side": max_points,
            "chamfer_seeds": "2 * frame_index for GT, +1 for prediction; independent per-side sampling",
            "range_band_m": [projection.get("min_range_m", 5.0), projection.get("max_range_m", 100.0)],
            "projection": projection
            or {"sensor": "pandar128", "azimuth_start_degrees": 180.0, "azimuth_end_degrees": -180.0},
        },
    }
