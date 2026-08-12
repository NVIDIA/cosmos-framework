# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.callbacks.sampled_media_recorder import SampledMediaRecorder

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _sample_record(iteration: int, caption: str | None = None) -> dict[str, object]:
    return {
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
        "caption": caption,
    }


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


def test_extract_records_prefers_action_fingerprint() -> None:
    callback = SampledMediaRecorder(enabled=True, output_uri="/tmp/samples.lance")
    callback.config = SimpleNamespace(job=SimpleNamespace(name="test_experiment"))
    batch = {
        "video": [object(), object()],
        "__key__": ["legacy-a", "legacy-b"],
        "action_sample_fingerprint": [
            "libero_10:v4:idx0:row5:start0:rank0:worker0:retry0:orig0",
            "robomind_franka:v4:idx2:row7:start2:rank1:worker3:retry0:orig2",
        ],
        "dataset_name": ["action_data", "action_data"],
        "source_dataset_name": ["libero_10", "robomind_franka"],
    }

    records = callback._extract_records(batch, iteration=12, rank=0)

    assert [record["sample_id"] for record in records] == [
        "libero_10:v4:idx0:row5:start0:rank0:worker0:retry0:orig0",
        "robomind_franka:v4:idx2:row7:start2:rank1:worker3:retry0:orig2",
    ]
    assert [record["source_dataset_name"] for record in records] == ["libero_10", "robomind_franka"]


def test_extract_records_accepts_action_fingerprint_without_legacy_key() -> None:
    callback = SampledMediaRecorder(enabled=True, output_uri="/tmp/samples.lance")
    callback.config = SimpleNamespace(job=SimpleNamespace(name="test_experiment"))
    batch = {
        "video": [object()],
        "action_sample_fingerprint": ["robomind_franka:v4:idx2:row7:start2:rank1:worker3:retry0:orig2"],
        "dataset_name": ["action_data"],
        "source_dataset_name": ["robomind_franka"],
    }

    records = callback._extract_records(batch, iteration=12, rank=1)

    assert [record["sample_id"] for record in records] == [
        "robomind_franka:v4:idx2:row7:start2:rank1:worker3:retry0:orig2"
    ]
    assert records[0]["rank"] == 1


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


@pytest.mark.parametrize(
    ("captions", "expected"),
    [
        (None, [None, None]),
        ([None, "recorded caption"], [None, "recorded caption"]),
    ],
)
def test_extract_records_preserves_missing_captions_as_null(
    captions: list[str | None] | None, expected: list[str | None]
) -> None:
    callback = SampledMediaRecorder(enabled=True, output_uri="/tmp/samples.lance", record_caption=True)
    batch = {
        "video": [object(), object()],
        "__key__": ["video-a", "video-b"],
        "ai_caption": captions,
    }

    records = callback._extract_records(batch, iteration=9, rank=0)

    assert [record["caption"] for record in records] == expected


def test_local_lance_append_without_cosmos_sila(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lance = pytest.importorskip("lance")
    monkeypatch.setitem(sys.modules, "cosmos_sila", None)
    output_uri = str(tmp_path / "samples.lance")
    callback = SampledMediaRecorder(
        enabled=True,
        output_uri=output_uri,
    )
    records = [_sample_record(iteration) for iteration in (1, 2)]

    callback._write_lance_records(records[:1])
    callback._write_lance_records(records[1:])

    assert lance.dataset(output_uri).count_rows() == 2


def test_local_lance_append_upgrades_legacy_table_metadata_only(tmp_path: Path) -> None:
    lance = pytest.importorskip("lance")
    pa = pytest.importorskip("pyarrow")
    output_uri = str(tmp_path / "samples.lance")
    callback = SampledMediaRecorder(enabled=True, output_uri=output_uri)
    legacy_schema = pa.schema([field for field in callback._table_schema() if field.name != "caption"])
    legacy_table = pa.Table.from_pylist([_sample_record(1)], schema=legacy_schema)
    lance.write_dataset(legacy_table, output_uri, mode="create")
    data_files_before = lance.dataset(output_uri).get_fragments()[0].data_files()

    callback._write_lance_records([_sample_record(2, caption="recorded caption")])

    upgraded = lance.dataset(output_uri)
    assert upgraded.get_fragments()[0].data_files() == data_files_before
    assert upgraded.to_table(columns=["caption"])["caption"].to_pylist() == [None, "recorded caption"]
