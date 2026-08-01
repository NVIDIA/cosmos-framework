# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json

import pytest

from cosmos_framework.configs.base.reasoner.experiment.wts_vlm import WTSLlavaDataset, wts_vlm_edge


def test_wts_dataset_resolves_video_paths_and_limit(tmp_path) -> None:
    media = tmp_path / "videos"
    media.mkdir()
    (media / "clip.mp4").write_bytes(b"video")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            [
                {
                    "video": "clip.mp4",
                    "conversations": [
                        {"from": "human", "value": "<video> what happens?"},
                        {"from": "gpt", "value": "A car stops."},
                    ],
                },
                {
                    "video": "clip.mp4",
                    "conversations": [
                        {"from": "human", "value": "<video> what happens next?"},
                        {"from": "gpt", "value": "Traffic moves."},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    dataset = WTSLlavaDataset(str(annotations), str(media), limit="1")
    assert len(dataset) == 1
    assert dataset[0]["video"] == str(media / "clip.mp4")


def test_wts_dataset_rejects_invalid_record(tmp_path) -> None:
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps([{"video": "clip.mp4"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="conversation"):
        WTSLlavaDataset(str(annotations), str(tmp_path))


def test_wts_edge_uses_native_edge_policy() -> None:
    assert wts_vlm_edge["defaults"][4] == {"override /vlm_policy": "cosmos3_edge_reasoner"}
    assert "lr_multipliers" not in wts_vlm_edge["optimizer"]
    assert wts_vlm_edge["model"]["config"]["policy"]["model_max_length"] == 16000
