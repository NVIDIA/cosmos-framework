# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tests for transfer SFT metadata filtering."""

import json

from cosmos_framework.data.generator.local_datasets.transfer_sft_dataset import _load_transfer_metadata


def test_load_transfer_metadata_can_disable_duration_and_window_prefilters(tmp_path):
    jsonl = tmp_path / "short_transfer_tasks.jsonl"
    record = {
        "uuid": "short-task",
        "duration": 120.0,
        "width": 256,
        "height": 256,
        "vision_path": "clips/short-task.mp4",
        "control_type": "edge",
        "t2w_windows": [
            {
                "start_frame": 0,
                "end_frame": 9,
                "temporal_interval": 1,
                "caption": "short task",
            }
        ],
    }
    jsonl.write_text(json.dumps(record) + "\n")

    assert _load_transfer_metadata(None, str(jsonl), min_frames=61) == []

    kept = _load_transfer_metadata(None, str(jsonl), min_frames=None, max_duration_s=None)
    assert len(kept) == 1
    assert kept[0]["uuid"] == "short-task"
    assert kept[0]["t2w_windows"][0]["caption"] == "short task"
