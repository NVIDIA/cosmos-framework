# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Independent analytic checks for reconstruction scoring and input contracts."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cosmos_framework.evaluation.lidar.tokenizer import evaluate_tokenizer, load_projection, range_to_points

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _save(path: Path, ranges: list[list[float]], valid: list[list[bool]], ids: list[int] | None = None) -> None:
    values = np.asarray(ranges, dtype=np.float32)[:, None, :]  # [T,1,W]
    masks = np.asarray(valid, dtype=bool)[:, None, :]  # [T,1,W]
    np.savez(path, range_m=values, valid=masks, sweep_ids=np.asarray(ids or list(range(len(values))), dtype=np.int64))


def _inputs(root: Path, entries: list[dict[str, Any]], **projection_fields: Any) -> tuple[Path, Path]:
    manifest, projection = root / "pairs.json", root / "projection.json"
    manifest.write_text(json.dumps(entries))
    projection.write_text(json.dumps({"sensor": "custom", "elevation_degrees": [0.0], **projection_fields}))
    return manifest, projection


def test_macro_micro_mask_drop_and_chamfer_coverage(tmp_path: Path) -> None:
    _save(tmp_path / "a.gt.npz", [[10, 0, 0, 0]], [[True, False, False, False]])
    _save(tmp_path / "a.pred.npz", [[14, 0, 0, 0]], [[False, False, False, False]])
    _save(tmp_path / "b.gt.npz", [[10, 10, 10, 0]], [[True, True, True, False]])
    _save(tmp_path / "b.pred.npz", [[10, 10, 10, 0]], [[True, True, True, False]])
    manifest, projection = _inputs(
        tmp_path,
        [
            {"id": "a", "reference": "a.gt.npz", "prediction": "a.pred.npz"},
            {"id": "b", "reference": "b.gt.npz", "prediction": "b.pred.npz"},
        ],
    )
    macro = evaluate_tokenizer(manifest, projection_path=projection)
    assert macro["corpus"] == pytest.approx(
        {
            "range_mae_m": 2,
            "range_median_ae_m": 2,
            "depth_abs_rel": 0.2,
            "chamfer_m": 0,
            "validity_iou": 0.5,
        }
    )
    assert macro["per_clip"][0]["chamfer_m"] is None
    assert macro["per_clip"][0]["chamfer_frames"] == 0
    assert macro["chamfer_clips"] == 1
    # The dropped prediction still contributes 4 metres of range error.
    micro = evaluate_tokenizer(manifest, "micro", projection)
    assert micro["corpus"] == pytest.approx({"range_mae_m": 1, "validity_iou": 0.75})


def test_non_squared_chamfer_and_even_median(tmp_path: Path) -> None:
    _save(tmp_path / "gt.npz", [[10, 10, 0, 0]], [[True, True, False, False]])
    _save(tmp_path / "pred.npz", [[11, 13, 0, 0]], [[True, True, False, False]])
    manifest, projection = _inputs(tmp_path, [{"id": "clip", "reference": "gt.npz", "prediction": "pred.npz"}])
    scores = evaluate_tokenizer(manifest, projection_path=projection)["corpus"]
    assert scores == pytest.approx(
        {
            "range_mae_m": 2,
            "range_median_ae_m": 2,
            "depth_abs_rel": 0.2,
            "chamfer_m": 2,
            "validity_iou": 1,
        }
    )


def test_equal_clip_weight_after_frame_mean(tmp_path: Path) -> None:
    _save(tmp_path / "a.gt.npz", [[10], [10]], [[True], [True]])
    _save(tmp_path / "a.pred.npz", [[10], [14]], [[True], [True]])
    _save(tmp_path / "b.gt.npz", [[10]], [[True]])
    _save(tmp_path / "b.pred.npz", [[16]], [[True]])
    manifest, projection = _inputs(
        tmp_path,
        [
            {"id": "a", "reference": "a.gt.npz", "prediction": "a.pred.npz"},
            {"id": "b", "reference": "b.gt.npz", "prediction": "b.pred.npz"},
        ],
    )
    result = evaluate_tokenizer(manifest, projection_path=projection)
    assert result["corpus"]["range_mae_m"] == pytest.approx(4)
    assert result["corpus"]["chamfer_m"] == pytest.approx(4)


def test_explicit_symmetric_padding_is_not_a_physical_ray(tmp_path: Path) -> None:
    _save(tmp_path / "gt.npz", [[10, 10, 10, 10]], [[True] * 4])
    _save(tmp_path / "pred.npz", [[999, 11, 11, 11, 11, 999]], [[True] * 6])
    manifest, projection = _inputs(
        tmp_path,
        [
            {"id": "clip", "reference": "gt.npz", "prediction": "pred.npz"},
        ],
        semantic_width=4,
        model_width=6,
        model_width_transform="circular_pad",
    )
    result = evaluate_tokenizer(manifest, projection_path=projection)
    assert result["corpus"]["range_mae_m"] == pytest.approx(1)
    assert result["corpus"]["chamfer_m"] == pytest.approx(1)
    assert result["per_clip"][0]["range_shape"] == [1, 1, 4]


def test_elevation_file_and_fixed_chamfer_canvas(tmp_path: Path) -> None:
    np.save(tmp_path / "elevation.npy", np.array([0.0]))
    _save(tmp_path / "gt.npz", [[10, 0, 0, 0]], [[True, False, False, False]], [3])
    _save(tmp_path / "pred.npz", [[10, 0, 0, 0]], [[True, False, False, False]], [3])
    _save(tmp_path / "cd.npz", [[10, 0, 0, 0, 0, 0, 0, 0]], [[True] + [False] * 7], [3])
    manifest = tmp_path / "pairs.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "id": "clip",
                    "reference": "gt.npz",
                    "prediction": "pred.npz",
                    "chamfer_reference": "cd.npz",
                }
            ]
        )
    )
    projection = tmp_path / "projection.json"
    projection.write_text(
        json.dumps(
            {
                "sensor": "custom",
                "elevation_file": "elevation.npy",
                "semantic_width": 4,
                "azimuth_start_degrees": 90,
                "azimuth_end_degrees": -270,
            }
        )
    )
    result = evaluate_tokenizer(manifest, projection_path=projection)
    assert result["corpus"]["chamfer_m"] == pytest.approx(0)
    xyz = range_to_points(
        np.array([[10, 0, 0, 0]]), np.array([[True, False, False, False]]), load_projection(projection)
    )
    np.testing.assert_allclose(xyz, [[0, 10, 0]], atol=1e-6)


@pytest.mark.parametrize("failure", ["sweep_ids", "band", "nan"])
def test_reject_misaligned_or_invalid_ranges(tmp_path: Path, failure: str) -> None:
    gt = 4 if failure == "band" else 10
    prediction = float("nan") if failure == "nan" else 10
    _save(tmp_path / "gt.npz", [[gt]], [[True]], [3])
    _save(tmp_path / "pred.npz", [[prediction]], [[True]], [4 if failure == "sweep_ids" else 3])
    manifest, projection = _inputs(tmp_path, [{"id": "clip", "reference": "gt.npz", "prediction": "pred.npz"}])
    with pytest.raises(ValueError, match={"sweep_ids": "alignment", "band": "metric band", "nan": "finite"}[failure]):
        evaluate_tokenizer(manifest, projection_path=projection)


def test_micro_counts_false_positives_from_empty_gt_frames(tmp_path: Path) -> None:
    _save(tmp_path / "gt.npz", [[10], [0]], [[True], [False]])
    _save(tmp_path / "pred.npz", [[10], [10]], [[True], [True]])
    manifest, projection = _inputs(tmp_path, [{"id": "clip", "reference": "gt.npz", "prediction": "pred.npz"}])
    assert evaluate_tokenizer(manifest, "micro", projection)["corpus"] == {"range_mae_m": 0, "validity_iou": 0.5}
    with pytest.raises(ValueError, match="no GT-valid rays"):
        evaluate_tokenizer(manifest, projection_path=projection)


def test_builtin_pandar128_row_and_azimuth_convention() -> None:
    ranges = np.zeros((128, 4), dtype=np.float64)  # [128,4]
    ranges[0, 0] = 10
    valid = ranges > 0  # [128,4]
    xyz = range_to_points(ranges, valid, {})  # [1,3]
    elevation = np.deg2rad(14.436)
    np.testing.assert_allclose(xyz, [[-10 * np.cos(elevation), 0, 10 * np.sin(elevation)]], atol=1e-6)
