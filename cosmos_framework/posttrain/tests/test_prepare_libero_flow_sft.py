# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""LeRobot preparation preserves real action windows, RGB orientation and splits."""

import io
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cosmos_framework.posttrain.scripts.prepare_libero_flow_sft import prepare


def test_embedded_images_and_actions_stay_with_their_episode(tmp_path):
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "data/chunk-000").mkdir(parents=True)
    (source / "meta/info.json").write_text(
        json.dumps({"fps": 10, "features": {"observation.state": {"shape": [8]}, "action": {"shape": [7]}}})
    )
    pd.DataFrame({"task_index": [0]}, index=["move the cube"]).to_parquet(source / "meta/tasks.parquet")
    rows = []
    for episode in (0, 1):
        for frame in range(4):
            pixels = np.arange(36, dtype=np.uint8).reshape(4, 3, 3) + episode * 50 + frame
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            rows.append(
                {
                    "episode_index": episode,
                    "frame_index": frame,
                    "task_index": 0,
                    "observation.state": [float(frame)] * 8,
                    "action": [float(episode * 10 + frame)] * 7,
                    "observation.images.image": {"bytes": buffer.getvalue()},
                    "observation.images.image2": {"bytes": buffer.getvalue()},
                }
            )
    pd.DataFrame(rows).to_parquet(source / "data/chunk-000/file-000.parquet")
    out = tmp_path / "prepared"
    report = prepare(
        str(source),
        str(out),
        training_episodes=[0],
        validation_episodes=[1],
        stride=2,
        horizon=2,
        source_revision="fixture",
    )
    assert report["counts"] == {"train": 2, "validation": 2}
    train = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    val = [json.loads(line) for line in (out / "validation.jsonl").read_text().splitlines()]
    assert train[1]["actions"] == [[2.0] * 7, [3.0] * 7]
    assert val[0]["actions"] == [[10.0] * 7, [11.0] * 7]
    with Image.open(out / train[0]["media"]["image"]) as image:
        np.testing.assert_array_equal(np.asarray(image), np.arange(36, dtype=np.uint8).reshape(4, 3, 3))
    with pytest.raises(FileExistsError):
        prepare(str(source), str(out), training_episodes=[0], validation_episodes=[1], source_revision="fixture")
