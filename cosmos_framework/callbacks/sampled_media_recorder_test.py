# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.callbacks.sampled_media_recorder import SampledMediaRecorder

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_extract_records_from_consumed_image_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    callback = SampledMediaRecorder(enabled=True, output_uri="/tmp/samples.lance")
    callback.config = SimpleNamespace(job=SimpleNamespace(name="test_experiment"))
    batch = {
        "images": [object(), object()],
        "__key__": ["image-a", "image-b"],
        "__url__": ["s3r:profile//bucket/a:0-10", "s3r:profile//bucket/b:10-20"],
        "dataset_name": ["images", "images"],
        "source_dataset_name": ["source-a", "source-b"],
    }

    records = callback._extract_records(batch, iteration=7, rank=3)

    assert [record["sample_id"] for record in records] == ["image-a", "image-b"]
    assert [record["media_type"] for record in records] == ["image", "image"]
    assert [record["source_dataset_name"] for record in records] == ["source-a", "source-b"]
    assert all(record["run_id"] == "12345" for record in records)
    assert all(record["iteration"] == 7 for record in records)
    assert all(record["rank"] == 3 for record in records)


def test_media_type_accepts_batched_tensors() -> None:
    images = torch.empty(2, 3, 16, 16)  # [B,C,H,W]
    video = torch.empty(2, 3, 4, 16, 16)  # [B,C,T,H,W]

    assert SampledMediaRecorder._media_type({"images": images}) == "image"
    assert SampledMediaRecorder._media_type({"video": video}) == "video"
    assert SampledMediaRecorder._media_type({"images": images, "video": video}) == "image_video"
    assert SampledMediaRecorder._media_type({}) == "unknown"


def test_extract_records_preserves_repeated_sample_occurrences() -> None:
    callback = SampledMediaRecorder(enabled=True, output_uri="/tmp/samples.lance")
    callback.config = SimpleNamespace(job=SimpleNamespace(name="test_experiment"))
    batch = {
        "video": [object(), object()],
        "__key__": ["video-a", "video-a"],
        "__url__": ["s3://bucket/video-a.mp4", "s3://bucket/video-a.mp4"],
        "dataset_name": ["video_480", "video_480"],
        "source_dataset_name": ["source", "source"],
    }

    records = callback._extract_records(batch, iteration=9, rank=0)

    assert [record["sample_id"] for record in records] == ["video-a", "video-a"]
    assert [record["sample_index"] for record in records] == [0, 1]


def test_local_lance_append_without_cosmos_sila(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lance = pytest.importorskip("lance")
    monkeypatch.setitem(sys.modules, "cosmos_sila", None)
    output_uri = str(tmp_path / "samples.lance")
    callback = SampledMediaRecorder(
        enabled=True,
        output_uri=output_uri,
    )
    records = [
        {
            "recorded_at": "2026-07-03T00:00:00+00:00",
            "run_id": "run",
            "job_name": "job",
            "iteration": iteration,
            "batch_index": iteration,
            "sample_index": 0,
            "rank": 0,
            "media_type": "video",
            "dataset_name": "video_256",
            "source_dataset_name": "source",
            "sample_id": f"video-{iteration}",
            "media_url": f"s3://bucket/video-{iteration}.mp4",
        }
        for iteration in (1, 2)
    ]

    callback._write_lance_records(records[:1])
    callback._write_lance_records(records[1:])

    assert lance.dataset(output_uri).count_rows() == 2
