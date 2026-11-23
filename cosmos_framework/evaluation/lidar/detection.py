# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Score precomputed detections with center-distance matching and 101-point AP.

Inputs are post-NMS detections in the same calibrated sensor coordinates as
the targets. No detector or detector-specific box processing is imported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CLASSES = ("car", "pedestrian", "rider", "large_vehicle")
DISTANCE_THRESHOLDS_M = (1.0, 2.0, 3.0, 4.0)


def _validate_center(record: dict[str, Any]) -> None:
    center = np.asarray(record["center_xy_m"])  # [2]
    if not np.issubdtype(center.dtype, np.number) or np.issubdtype(center.dtype, np.complexfloating):
        raise ValueError("center_xy_m must contain real numeric coordinates")
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center_xy_m must contain two finite coordinates in meters")


def _average_precision(flags: np.ndarray, targets: int) -> float:
    """101 recall points, excluding recall <=0.1 and subtracting precision 0.1."""
    if targets == 0 or not flags.any():
        return 0.0
    true_count = np.cumsum(flags, dtype=np.float64)  # [N]
    recall = true_count / targets  # [N]
    precision = true_count / np.arange(1, len(flags) + 1)  # [N]
    curve = np.interp(np.linspace(0, 1, 101), recall, precision, right=0.0)  # [101]
    return float(np.maximum(curve[11:] - 0.1, 0).mean() / 0.9)


def _score_class(frames: list[dict[str, Any]], kind: str, distance: float) -> dict[str, object]:
    targets = {frame["frame_id"]: frame["targets"] for frame in frames}
    target_count = sum(item["class_name"] == kind for records in targets.values() for item in records)
    ranked = sorted(
        (
            (float(item["score"]), f"{frame['frame_id']}/{index:06d}", frame["frame_id"], item)
            for frame in frames
            for index, item in enumerate(frame["predictions"])
            if item["class_name"] == kind
        ),
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )
    assigned: set[tuple[str, int]] = set()
    matches: list[bool] = []
    for _, _, frame_id, prediction in ranked:
        px, py = prediction["center_xy_m"]
        nearest = min(
            (
                (float(np.hypot(px - target["center_xy_m"][0], py - target["center_xy_m"][1])), index)
                for index, target in enumerate(targets[frame_id])
                if target["class_name"] == kind and (frame_id, index) not in assigned
            ),
            default=(float("inf"), -1),
        )
        matched = nearest[0] < distance
        matches.append(matched)
        if matched:
            assigned.add((frame_id, nearest[1]))
    flags = np.asarray(matches, dtype=np.bool_)  # [N]
    selected = [matched for record, matched in zip(ranked, matches, strict=True) if record[0] >= 0.05]
    true_positive = sum(selected)
    count = len(selected)
    precision = true_positive / count if count else None
    recall = true_positive / target_count if target_count else None
    f1 = None
    if precision is not None and recall is not None and precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "average_precision": _average_precision(flags, target_count),
        "eligible_targets": target_count,
        "predictions": count,
        "true_positives": true_positive,
        "false_positives": count - true_positive,
        "false_negatives": target_count - true_positive,
        "precision": precision,
        "recall": recall,
        "miss_rate": None if recall is None else 1 - recall,
        "f1": f1,
    }


def evaluate_detections(manifest: Path) -> dict[str, Any]:
    """Score all same-class, same-frame predictions using strict XY distances.

    JSON has ``classes`` (optional, default four detector classes) and ``frames``.
    Each frame has ``frame_id``, ``targets``, and ``predictions``. Both record
    types contain ``class_name`` and ``center_xy_m``; predictions add ``score``.
    Extra fields are permitted. The caller supplies the desired class/range
    population and post-NMS predictions; this function performs no filtering.
    AP uses every supplied confidence; P/R/F1 use confidence >=0.05 at 3 meters.
    """
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    classes = payload.get("classes", list(DEFAULT_CLASSES))
    if not isinstance(classes, list) or not classes or any(not isinstance(kind, str) or not kind for kind in classes):
        raise ValueError("classes must be a nonempty list of class names")
    if len(set(classes)) != len(classes):
        raise ValueError("classes must be unique")
    frames = payload["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames must be a nonempty list")
    seen: set[str] = set()
    for frame in frames:
        frame_id = frame["frame_id"]
        if not isinstance(frame_id, str) or not frame_id or frame_id in seen:
            raise ValueError("frame_id must be a unique nonempty string")
        seen.add(frame_id)
        for record in frame["targets"] + frame["predictions"]:
            if record["class_name"] not in classes:
                raise ValueError("every class_name must be included in classes")
            _validate_center(record)
        for prediction in frame["predictions"]:
            score = prediction["score"]
            if not isinstance(score, (float, int)) or not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("prediction score must be finite and in [0,1]")
    detailed = {
        kind: {f"{distance:g}m": _score_class(frames, kind, distance) for distance in DISTANCE_THRESHOLDS_M}
        for kind in classes
    }
    operating = [scores["3m"] for scores in detailed.values()]
    tp = sum(item["true_positives"] for item in operating)
    fp = sum(item["false_positives"] for item in operating)
    fn = sum(item["false_negatives"] for item in operating)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    ap = [entry["average_precision"] for scores in detailed.values() for entry in scores.values()]
    present_ap = [
        entry["average_precision"]
        for scores in detailed.values()
        for entry in scores.values()
        if entry["eligible_targets"]
    ]
    return {
        "map": sum(ap) / len(ap),
        "map_present_classes": sum(present_ap) / len(present_ap) if present_ap else None,
        "precision_3m": precision,
        "recall_3m": recall,
        "miss_rate_3m": None if recall is None else 1 - recall,
        "f1_3m": 2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None,
        "true_positives_3m": tp,
        "false_positives_3m": fp,
        "false_negatives_3m": fn,
        "frame_count": len(frames),
        "classes": classes,
        "detailed_metrics": detailed,
    }
