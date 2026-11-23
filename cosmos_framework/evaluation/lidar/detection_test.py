# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Analytic checks of matching order, AP interpolation, and empty populations."""

import json
from pathlib import Path

import pytest

from cosmos_framework.evaluation.lidar.detection import evaluate_detections

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_strict_boundary_absent_class_and_operating_score(tmp_path: Path) -> None:
    data = {
        "classes": ["car", "pedestrian"],
        "frames": [
            {
                "frame_id": "a",
                "targets": [{"class_name": "car", "center_xy_m": [0, 0]}],
                "predictions": [
                    {"class_name": "car", "center_xy_m": [3, 0], "score": 0.9},
                    {"class_name": "car", "center_xy_m": [0, 0], "score": 0.01},
                ],
            }
        ],
    }
    path = tmp_path / "detections.json"
    path.write_text(json.dumps(data))
    result = evaluate_detections(path)
    assert result["true_positives_3m"] == 0
    assert result["false_positives_3m"] == 1
    assert result["false_negatives_3m"] == 1
    assert result["detailed_metrics"]["pedestrian"]["3m"]["average_precision"] == 0
    assert result["map_present_classes"] == pytest.approx(2 * result["map"])


def test_greedy_match_confidence_order_and_ap(tmp_path: Path) -> None:
    data = {
        "classes": ["car"],
        "frames": [
            {
                "frame_id": "target",
                "targets": [{"class_name": "car", "center_xy_m": [10, 0]}],
                "predictions": [
                    {"class_name": "car", "center_xy_m": [10, 0], "score": 0.8},
                    {"class_name": "car", "center_xy_m": [10, 0], "score": 0.9},
                ],
            },
            {
                "frame_id": "empty",
                "targets": [],
                "predictions": [{"class_name": "car", "center_xy_m": [10, 0], "score": 1.0}],
            },
        ],
    }
    path = tmp_path / "detections.json"
    path.write_text(json.dumps(data))
    result = evaluate_detections(path)
    assert result["true_positives_3m"] == 1
    assert result["false_positives_3m"] == 2
    assert result["recall_3m"] == 1
    assert result["precision_3m"] == pytest.approx(1 / 3)
    # Linear interpolation from recall=0/precision=0 to recall=1/precision=0.5;
    # the duplicate final recall=1 retains precision=1/3. After subtracting 0.1,
    # only recall points 0.21..0.99 and 1.0 contribute.
    expected = (sum(index / 200 - 0.1 for index in range(21, 100)) + 1 / 3 - 0.1) / 90 / 0.9
    assert result["map"] == pytest.approx(expected)


def test_no_targets_and_duplicate_frame_validation(tmp_path: Path) -> None:
    data = {"classes": ["car"], "frames": [{"frame_id": "a", "targets": [], "predictions": []}]}
    path = tmp_path / "detections.json"
    path.write_text(json.dumps(data))
    result = evaluate_detections(path)
    assert result["map"] == 0
    assert result["recall_3m"] is None
    assert result["precision_3m"] is None
    data["frames"] *= 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unique"):
        evaluate_detections(path)


def test_numeric_strings_rejected_as_detection_centers(tmp_path: Path) -> None:
    data = {
        "classes": ["car"],
        "frames": [
            {
                "frame_id": "a",
                "targets": [{"class_name": "car", "center_xy_m": [10, 0]}],
                "predictions": [{"class_name": "car", "center_xy_m": ["10", "0"], "score": 0.9}],
            }
        ],
    }
    path = tmp_path / "detections.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="real numeric"):
        evaluate_detections(path)
