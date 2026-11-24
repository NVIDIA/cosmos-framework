# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run public scoring commands from outside the checkout without model weights."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_scoring_commands_outside_checkout(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    environment = {**os.environ, "PYTHONPATH": str(root)}
    # Block accidental imports from development-only packages in every subprocess.
    (tmp_path / "sitecustomize.py").write_text(
        "import sys\n"
        "class BlockInternalImports:\n"
        "    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:\n"
        "        if fullname.split('.')[0] in {'imaginaire', 'projects', 'cosmos_eval', 'cosmos_metrics'}:\n"
        "            raise ImportError('Unexpected dependency: ' + fullname)\n"
        "sys.meta_path.insert(0, BlockInternalImports())\n"
    )
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(root)))
    projection = {"elevation_degrees": [0], "semantic_width": 2, "model_width": 2}
    (tmp_path / "projection.json").write_text(json.dumps(projection))
    ranges = np.array([[[10.0, 12.0]]], dtype=np.float32)  # [T,H,W]
    np.savez(tmp_path / "range.npz", range_m=ranges, valid=ranges > 0, sweep_ids=np.array([7]))
    (tmp_path / "pairs.json").write_text(
        json.dumps([{"id": "clip", "reference": "range.npz", "prediction": "range.npz"}])
    )
    (tmp_path / "prepare.json").write_text(
        json.dumps([{"clip_id": "clip", "reference": "range.npz", "generated": "range.npz"}])
    )
    features = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)  # [N,D]
    np.save(tmp_path / "features.npy", features)
    (tmp_path / "keys.json").write_text(json.dumps(["clip/7", "clip/8"]))
    (tmp_path / "detections.json").write_text(
        json.dumps(
            {
                "classes": ["car"],
                "frames": [
                    {
                        "frame_id": "clip/7",
                        "targets": [{"class_name": "car", "center_xy_m": [10, 0]}],
                        "predictions": [{"class_name": "car", "center_xy_m": [10, 0], "score": 0.9}],
                    }
                ],
            }
        )
    )

    def run(arguments: list[str]) -> dict[str, object]:
        result = subprocess.run(
            [sys.executable, "-m", "cosmos_framework.scripts.evaluate_lidar", *arguments],
            cwd=tmp_path,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    assert run(["doctor"])["base_metrics_need_learned_model"] is False
    tokenizer = run(
        ["tokenizer", "--manifest", "pairs.json", "--projection", "projection.json", "--output", "tokenizer.json"]
    )
    assert tokenizer == json.loads((tmp_path / "tokenizer.json").read_text())
    run(["prepare", "--manifest", "prepare.json", "--projection", "projection.json", "--output-dir", "prepared"])
    clouds = run(["clouds", "--manifest", "prepared/clouds.json", "--output", "clouds.json"])
    assert clouds["bev_jsd_global"] == pytest.approx(0)
    feature_scores = run(
        [
            "features",
            "--generated",
            "features.npy",
            "--reference",
            "features.npy",
            "--sample-keys",
            "keys.json",
            "--output",
            "feature_scores.json",
        ]
    )
    assert feature_scores["fud"] == pytest.approx(0)
    detections = run(["detections", "--manifest", "detections.json", "--output", "detection_scores.json"])
    assert detections["f1_3m"] == pytest.approx(1)
