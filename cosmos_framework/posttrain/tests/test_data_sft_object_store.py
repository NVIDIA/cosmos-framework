# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""SFT S3/GCS JSONL loading, resource cleanup, and cluster-independent text identity."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from botocore.response import StreamingBody

from cosmos_framework.posttrain.config.sft import SFTDatasetConfig
from cosmos_framework.posttrain.data.datasets.sft import build_sft_dataset
from cosmos_framework.posttrain.roles.env.sft import build_sft_env
from cosmos_framework.posttrain.tests.helpers.object_store_fake import FakeS3Client
from cosmos_framework.posttrain.utils import object_store

pytestmark = [pytest.mark.L0, pytest.mark.CPU]
_URI = "s3://dataset-bucket/obj1k/train.jsonl"


def _install_client(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> tuple[list[str], list[io.BytesIO]]:
    client = FakeS3Client()
    client.plant("obj1k/train.jsonl", payload)
    credentials: list[str] = []
    streams: list[io.BytesIO] = []
    get_object = client.get_object

    def get(**kwargs: Any) -> dict[str, Any]:
        result = get_object(**kwargs)
        raw = result["Body"]
        streams.append(raw)
        # Exercise botocore's actual response wrapper rather than only BytesIO's TextIO compatibility.
        result["Body"] = StreamingBody(raw, len(payload))
        return result

    def build(path: str) -> FakeS3Client:
        credentials.append(path)
        return client

    monkeypatch.setattr(client, "get_object", get)
    monkeypatch.setattr(object_store, "build_s3_client", build)
    return credentials, streams


def _payload(answer: str = "pass") -> bytes:
    return (
        "\n".join(
            json.dumps({"messages": [{"role": "user", "content": str(i)}, {"role": "assistant", "content": answer}]})
            for i in range(4)
        )
        + "\n"
    ).encode()


def test_s3_text_matches_local_source_and_cursor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _payload()
    local = tmp_path / "train.jsonl"
    local.write_bytes(payload)
    credentials, streams = _install_client(monkeypatch, payload)
    config = SFTDatasetConfig(path=_URI, credentials="/mounted/gcs.json")
    remote_env = build_sft_env(config, 42, rank=0, world_size=2)
    local_env = build_sft_env(SFTDatasetConfig(path=str(local)), 42, rank=0, world_size=2)
    assert remote_env.source_fingerprint == local_env.source_fingerprint
    assert remote_env.get_samples(1) == local_env.get_samples(1)
    state = remote_env.state_dict()
    local_env.load_state_dict(state)
    assert remote_env.get_samples(3) == local_env.get_samples(3)
    assert credentials == ["/mounted/gcs.json"]
    assert len(streams) == 1 and streams[0].closed


def test_s3_changed_content_is_not_hidden_by_a_size_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(monkeypatch, _payload("pass"))
    config = SFTDatasetConfig(path=_URI)
    original = build_sft_env(config, 42)
    state = original.state_dict()
    _install_client(monkeypatch, _payload("stop"))  # same byte count, different content
    changed = build_sft_env(config, 42)
    assert original.source_fingerprint != changed.source_fingerprint
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        changed.load_state_dict(state)


def test_s3_parse_failure_closes_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _, streams = _install_client(monkeypatch, b"\nnot json\n")
    with pytest.raises(ValueError, match=r"s3://dataset-bucket/obj1k/train.jsonl:2: invalid JSON"):
        build_sft_dataset(SFTDatasetConfig(path=_URI))
    assert len(streams) == 1 and streams[0].closed


def test_s3_media_requires_explicit_local_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"local media; not decoded by this source")
    payload = json.dumps(
        {
            "messages": [
                {"role": "user", "content": [{"type": "image", "image": image.name}]},
                {"role": "assistant", "content": "a frame"},
            ]
        }
    ).encode()
    _install_client(monkeypatch, payload)
    with pytest.raises(ValueError, match="explicit local media_root"):
        build_sft_dataset(SFTDatasetConfig(path=_URI))
    dataset = build_sft_dataset(SFTDatasetConfig(path=_URI, media_root=str(tmp_path)))
    assert dataset[0]["messages"][0]["content"][0]["image"] == str(image)


@pytest.mark.parametrize("uri", ["s3://", "s3://bucket", "s3://bucket/", "s3://bucket/prefix/", "s3:///file.jsonl"])
def test_s3_config_requires_an_object_uri(uri: str) -> None:
    with pytest.raises(ValueError, match="object URI"):
        SFTDatasetConfig(path=uri).require_source()


def test_s3_config_requires_credentials() -> None:
    with pytest.raises(ValueError, match="credentials path"):
        SFTDatasetConfig(path=_URI, credentials="").require_source()
