# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tracked-cuboid vehicle temporal consistency and object support, without models."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from cosmos_framework.evaluation.lidar.generation import chamfer, corpus_temporal, load_clips, load_cloud, load_poses

VEHICLE_TYPES = frozenset(
    {
        "automobile",
        "bus",
        "car",
        "heavytruck",
        "largevehicle",
        "othervehicle",
        "train",
        "trainortramcar",
        "trailer",
        "trolleybus",
        "truck",
        "van",
        "vehicle",
    }
)


def _cuboid(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    pose = np.asarray(payload["object_to_world"], dtype=np.float64)  # [4,4]
    lwh = np.asarray(payload["lwh_m"], dtype=np.float64)  # [3]
    kind = payload.get("object_type", "Vehicle")
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("object_to_world must be finite [4,4]")
    if not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-6):
        raise ValueError("object_to_world must have a homogeneous final row")
    left, singular, right = np.linalg.svd(pose[:3, :3])  # [3,3],[3],[3,3]
    if np.linalg.det(pose[:3, :3]) <= 0 or np.any(np.abs(singular - 1) > 2e-2):
        raise ValueError("object_to_world rotation must be proper and near orthonormal")
    pose = pose.copy()  # [4,4]
    pose[:3, :3] = left @ right  # [3,3]
    if lwh.shape != (3,) or not np.isfinite(lwh).all() or np.any(lwh <= 0):
        raise ValueError("lwh_m must be three positive finite dimensions")
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("object_type must be a nonempty string")
    return pose, lwh, kind


def canonical_crop(world: np.ndarray, pose: np.ndarray, lwh: np.ndarray) -> np.ndarray:
    """Object-frame points within GT half extents plus 0.2 m per side."""
    points = (world - pose[:3, 3]) @ pose[:3, :3]  # [N,3]
    return points[(np.abs(points) <= 0.5 * lwh + 0.2).all(axis=1)]  # [M,3]


def _integer_bin(value: int, edges: tuple[int, ...]) -> str:
    for lower, upper in zip(edges, edges[1:]):
        if value < upper:
            return str(lower) if upper == lower + 1 else f"{lower}-{upper - 1}"
    return f"{edges[-1]}+"


def _range_bin(distance: float) -> str:
    lower = 0
    for upper in (20, 40, 60):
        if distance < upper:
            return f"{lower}-{upper}m"
        lower = upper
    return "60m+"


def _support_result(counts: list[int]) -> dict[str, int | float | None]:
    eligible, supported = counts
    recall = supported / eligible if eligible else None
    return {
        "eligible": eligible,
        "supported": supported,
        "recall": recall,
        "miss_rate": None if recall is None else 1 - recall,
    }


def evaluate_objects(manifest: Path) -> dict[str, object]:
    """Score external world-coordinate tracked cuboids and paired sensor XYZ.

    Uses GT >=30 returns, generated >=10 returns for support. Vehicle CTC uses
    adjacent eligible observations of the same vehicle track. Empty generated
    crops incur 2 m; all eligible pairs remain in the denominator. Corpus CTC
    is equal-clip macro over eligible clips; object support pools object counts.
    """
    populations: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    populations["overall"] = [0, 0]
    clips: list[dict[str, Any]] = []
    for entry in load_clips(manifest):
        generated = [load_cloud(manifest.parent / path) for path in entry["generated"]]
        reference = [load_cloud(manifest.parent / path) for path in entry["reference"]]
        poses = load_poses(manifest.parent / entry["lidar_to_world"], len(generated))  # [T,4,4]
        annotations = json.loads((manifest.parent / entry["annotations"]).read_text(encoding="utf-8"))
        if not isinstance(annotations, list) or len(annotations) != len(generated):
            raise ValueError("annotations must contain one track-id dictionary per ordered frame")
        if any(not isinstance(frame, dict) for frame in annotations):
            raise ValueError("each annotations frame must be a track-id dictionary")
        lengths = Counter(track for frame in annotations for track in frame)
        crops: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
        local_counts = [0, 0]
        for frame, (prediction, target, sensor_pose) in enumerate(zip(generated, reference, poses, strict=True)):
            predicted_world = prediction.astype(np.float64) @ sensor_pose[:3, :3].T + sensor_pose[:3, 3]  # [N,3]
            target_world = target.astype(np.float64) @ sensor_pose[:3, :3].T + sensor_pose[:3, 3]  # [M,3]
            vehicle_crops: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for track, payload in annotations[frame].items():
                object_pose, lwh, kind = _cuboid(payload)  # [4,4],[3],str
                target_crop = canonical_crop(target_world, object_pose, lwh)  # [R,3]
                if len(target_crop) < 30:
                    continue
                predicted_crop = canonical_crop(predicted_world, object_pose, lwh)  # [G,3]
                supported = int(len(predicted_crop) >= 10)
                local_counts[0] += 1
                local_counts[1] += supported
                groups = (
                    "overall",
                    f"class/{kind.strip().lower()}",
                    f"range/{_range_bin(float(np.linalg.norm(object_pose[:3, 3] - sensor_pose[:3, 3])))}",
                    f"gt_points/{_integer_bin(len(target_crop), (1, 5, 10, 30, 100, 300, 1000))}",
                    f"track_length/{_integer_bin(lengths[track], (1, 2, 5, 10, 20, 40, 80))}",
                    f"track/{entry['clip_id']}/{track}",
                )
                for group in groups:
                    populations[group][0] += 1
                    populations[group][1] += supported
                if re.sub(r"[^a-z0-9]", "", kind.strip().lower()) in VEHICLE_TYPES:
                    vehicle_crops[track] = predicted_crop, target_crop
            crops.append(vehicle_crops)
        generated_pairs: list[float] = []
        reference_pairs: list[float] = []
        nonempty_pairs = 0
        for previous, current in zip(crops, crops[1:]):
            for track in sorted(previous.keys() & current.keys()):
                pg, pr = previous[track]  # [Ng,3],[Nr,3]
                cg, cr = current[track]  # [Mg,3],[Mr,3]
                reference_pairs.append(chamfer(cr, pr))
                if len(pg) == 0 or len(cg) == 0:
                    generated_pairs.append(2.0)
                else:
                    generated_pairs.append(chamfer(cg, pg))
                    nonempty_pairs += 1
        temporal = None
        if reference_pairs:
            temporal = {
                "generated": math.fsum(generated_pairs) / len(generated_pairs),
                "reference": math.fsum(reference_pairs) / len(reference_pairs),
                "eligible_pair_count": len(reference_pairs),
                "generated_pair_count": nonempty_pairs,
            }
        clips.append(
            {"clip_id": entry["clip_id"], "object_support": _support_result(local_counts), "vehicle_ctc": temporal}
        )
    eligible = [clip["vehicle_ctc"] for clip in clips if clip["vehicle_ctc"] is not None]
    temporal_corpus: dict[str, float | int] | None = None
    if eligible:
        temporal_corpus = {
            **corpus_temporal(eligible),
            "eligible_clip_count": len(eligible),
            "scored_clip_count": len(eligible),
            "eligible_pair_count": sum(item["eligible_pair_count"] for item in eligible),
            "generated_pair_count": sum(item["generated_pair_count"] for item in eligible),
        }
    return {
        "object_support": _support_result(populations["overall"]),
        "object_support_groups": {
            group: _support_result(counts) for group, counts in sorted(populations.items()) if group != "overall"
        },
        "vehicle_ctc": temporal_corpus,
        "clip_count": len(clips),
        "clips": clips,
    }
