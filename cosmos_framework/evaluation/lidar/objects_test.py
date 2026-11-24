# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Check canonical object coordinates, eligibility and empty-crop penalties."""

import json
from pathlib import Path

import numpy as np
import pytest

from cosmos_framework.evaluation.lidar.objects import evaluate_objects

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_object_support_and_vehicle_temporal_penalty(tmp_path: Path) -> None:
    generator = np.random.default_rng(4)
    center = np.array([10.0, 0, 0])  # [3]
    for frame in range(2):
        target = generator.uniform(-0.4, 0.4, (30, 3)) + center  # [30,3]
        np.save(tmp_path / f"r{frame}.npy", target)
        np.save(tmp_path / f"g{frame}.npy", target if frame == 0 else np.empty((0, 3)))
    poses = np.repeat(np.eye(4)[None], 2, axis=0)  # [2,4,4]
    np.save(tmp_path / "poses.npy", poses)
    object_pose = np.eye(4)  # [4,4]
    object_pose[:3, 3] = center  # [3]
    annotations = [
        {"vehicle_1": {"object_to_world": object_pose.tolist(), "lwh_m": [2, 2, 2], "object_type": "Car"}}
    ] * 2
    (tmp_path / "annotations.json").write_text(json.dumps(annotations))
    entry = {
        "clip_id": "clip",
        "generated": ["g0.npy", "g1.npy"],
        "reference": ["r0.npy", "r1.npy"],
        "lidar_to_world": "poses.npy",
        "annotations": "annotations.json",
    }
    manifest = tmp_path / "objects.json"
    manifest.write_text(json.dumps([entry]))
    result = evaluate_objects(manifest)
    assert result["object_support"] == {"eligible": 2, "supported": 1, "recall": 0.5, "miss_rate": 0.5}
    assert result["vehicle_ctc"]["generated"] == 2
    assert result["vehicle_ctc"]["reference"] > 0
    assert result["vehicle_ctc"]["eligible_pair_count"] == 1
    assert result["vehicle_ctc"]["generated_pair_count"] == 0
    assert result["object_support_groups"]["class/car"]["recall"] == 0.5
    # 29 returns make the first frame ineligible; it must not create a miss/pair.
    np.save(tmp_path / "r0.npy", np.load(tmp_path / "r0.npy")[:29])
    result = evaluate_objects(manifest)
    assert result["object_support"]["eligible"] == 1
    assert result["vehicle_ctc"] is None


def test_object_zero_reference_temporal_is_undefined(tmp_path: Path) -> None:
    points = np.zeros((30, 3)) + [10, 0, 0]  # [30,3]
    np.save(tmp_path / "points.npy", points)
    np.save(tmp_path / "poses.npy", np.repeat(np.eye(4)[None], 2, axis=0))
    pose = np.eye(4)  # [4,4]
    pose[0, 3] = 10
    annotation = {"v": {"object_to_world": pose.tolist(), "lwh_m": [2, 2, 2]}}
    (tmp_path / "annotations.json").write_text(json.dumps([annotation, annotation]))
    entry = {
        "clip_id": "a",
        "generated": ["points.npy"] * 2,
        "reference": ["points.npy"] * 2,
        "lidar_to_world": "poses.npy",
        "annotations": "annotations.json",
    }
    path = tmp_path / "objects.json"
    path.write_text(json.dumps([entry]))
    with pytest.raises(ValueError, match="positive"):
        evaluate_objects(path)
