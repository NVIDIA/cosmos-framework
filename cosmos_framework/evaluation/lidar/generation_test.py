# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Analytic metric checks independent of model weights or internal datasets."""

import json
from pathlib import Path

import numpy as np
import pytest

from cosmos_framework.evaluation.lidar.generation import (
    bev_counts,
    bev_mmd2,
    corpus_temporal,
    evaluate_clouds,
    frechet_distance,
    js_distance,
    polynomial_mmd2,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_bev_boundaries_and_distance() -> None:
    points = np.array([[5, 0, 0], [99, 0, 0], [100, 0, 0], [0, 0, 4], [-5, 0, 0]], dtype=np.float32)  # [5,3]
    assert bev_counts(points).sum() == 3
    first = np.array([1, 0])  # [2]
    second = np.array([0, 1])  # [2]
    assert js_distance(first, second) == pytest.approx(np.sqrt(np.log(2)))
    assert js_distance(first, first) == 0
    assert js_distance(first, np.zeros(2)) is None
    assert bev_mmd2(first[None], second[None]) == pytest.approx(2 - 2 * np.exp(-4))


def test_feature_analytic_and_low_rank() -> None:
    first = np.zeros((3, 2))  # [3,2]
    second = np.ones((3, 2))  # [3,2]
    assert frechet_distance(first, second) == pytest.approx(2)
    assert polynomial_mmd2(first, second, ["a", "b", "c"]) == pytest.approx(7)
    generator = np.random.default_rng(4)
    wide = generator.normal(size=(3, 10))  # [3,10]
    assert frechet_distance(wide, wide) == pytest.approx(0, abs=1e-12)
    assert frechet_distance(wide, wide + 1) == pytest.approx(10, abs=1e-12)
    varied = np.array([[0.0], [1.0], [2.0]])  # [3,1]
    # An unbiased estimate can be negative, even for identical sample sets.
    assert polynomial_mmd2(varied, varied, ["a", "b", "c"]) < 0


def test_complex_features_rejected_without_discarding_imaginary_values() -> None:
    generated = np.full((2, 2), 1 + 2j, dtype=np.complex128)  # [2,2]
    reference = np.ones((2, 2))  # [2,2]
    with pytest.raises(ValueError, match="real numeric"):
        frechet_distance(generated, reference)
    with pytest.raises(ValueError, match="real numeric"):
        polynomial_mmd2(generated, reference, ["a", "b"])


def test_ctc_corpus_ratio_is_ratio_of_macro_means() -> None:
    scores = [{"generated": 2.0, "reference": 1.0}, {"generated": 2.0, "reference": 3.0}]
    assert corpus_temporal(scores) == {"generated": 2.0, "reference": 2.0, "ratio": 1.0}
    with pytest.raises(ValueError, match="positive"):
        corpus_temporal([{"generated": 1.0, "reference": 0.0}])


def test_cloud_manifest_world_alignment_and_full_coverage(tmp_path: Path) -> None:
    for index, x in enumerate([10, 11]):
        np.save(tmp_path / f"g{index}.npy", np.array([[float(x), 0, 0]]))
        np.save(tmp_path / f"r{index}.npy", np.array([[float(x), 0, 0]]))
    poses = np.repeat(np.eye(4)[None], 2, axis=0)  # [2,4,4]
    poses[1, 0, 3] = -0.5
    np.save(tmp_path / "poses.npy", poses)
    entry = {
        "clip_id": "a",
        "generated": ["g0.npy", "g1.npy"],
        "reference": ["r0.npy", "r1.npy"],
        "lidar_to_world": "poses.npy",
    }
    path = tmp_path / "clips.json"
    path.write_text(json.dumps([entry]))
    result = evaluate_clouds(path)
    assert result["bev_jsd_global"] == 0
    assert result["bev_mmd2_local_sigma0p5_macro"] == 0
    assert result["ctc"] == {"generated": 0.5, "reference": 0.5, "ratio": 1.0}
    np.save(tmp_path / "g0.npy", np.empty((0, 3)))
    result = evaluate_clouds(path)
    assert result["ctc"] is None
    assert result["ctc_available_clip_count"] == 0
