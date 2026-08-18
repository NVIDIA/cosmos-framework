# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from cosmos_framework.utils.generator.reasoner import pretrained_models_downloader as downloader


def test_completion_marker_distinguishes_metadata_and_weights(tmp_path: Path) -> None:
    downloader._mark_cache_complete(str(tmp_path), include_model_weights=False)

    assert downloader._is_cache_complete(str(tmp_path), include_model_weights=False)
    assert not downloader._is_cache_complete(str(tmp_path), include_model_weights=True)

    downloader._mark_cache_complete(str(tmp_path), include_model_weights=True)

    assert downloader._is_cache_complete(str(tmp_path), include_model_weights=False)
    assert downloader._is_cache_complete(str(tmp_path), include_model_weights=True)


def test_tokenizer_json_cache_gets_marker_without_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_name = "test-model"
    s3_prefix = "hf_models"
    model_cache_dir = tmp_path / s3_prefix / model_name
    model_cache_dir.mkdir(parents=True)
    (model_cache_dir / "tokenizer.json").write_text("{}")
    manifest = [(f"{s3_prefix}/{model_name}/tokenizer.json", 2)]

    monkeypatch.setattr(downloader, "INTERNAL", True)
    monkeypatch.setattr(downloader, "s3_dir_exists", lambda *_args: True)
    monkeypatch.setattr(downloader, "_list_s3_prefix_objects", lambda *_args: manifest)

    def fail_download(*_args: object, **_kwargs: object) -> list[str]:
        raise AssertionError("An existing cache must not be downloaded again")

    monkeypatch.setattr(downloader, "parallel_download_s3_prefix_to_dir", fail_download)

    result = downloader.maybe_download_hf_model_from_s3(
        model_name,
        credentials="credentials.secret",
        bucket="bucket",
        cache_dir=str(tmp_path),
        s3_prefix=s3_prefix,
    )

    assert result == str(model_cache_dir)
    assert downloader._is_cache_complete(result, include_model_weights=False)


def test_concurrent_cache_miss_has_one_downloader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_name = "test-model"
    s3_prefix = "hf_models"
    download_count = 0
    count_lock = threading.Lock()
    manifest = [(f"{s3_prefix}/{model_name}/vocab.json", 2)]

    monkeypatch.setattr(downloader, "INTERNAL", True)
    monkeypatch.setattr(downloader, "s3_dir_exists", lambda *_args: True)
    monkeypatch.setattr(downloader, "_list_s3_prefix_objects", lambda *_args: manifest)
    monkeypatch.setattr(downloader, "_MARKER_POLL_SECONDS", 0.001)
    monkeypatch.setattr(downloader, "_MARKER_POLL_JITTER_SECONDS", 0.009)

    def download(
        _bucket: str,
        _prefix: str,
        dest_dir: str,
        _credentials: str,
        **_kwargs: object,
    ) -> list[str]:
        nonlocal download_count
        assert _kwargs["objects"] == manifest
        with count_lock:
            download_count += 1
        time.sleep(0.05)
        vocab_path = Path(dest_dir) / "vocab.json"
        vocab_path.write_text("{}")
        return [str(vocab_path)]

    monkeypatch.setattr(downloader, "parallel_download_s3_prefix_to_dir", download)

    def resolve_model() -> str:
        return downloader.maybe_download_hf_model_from_s3(
            model_name,
            credentials="credentials.secret",
            bucket="bucket",
            cache_dir=str(tmp_path),
            s3_prefix=s3_prefix,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _index: resolve_model(), range(8)))

    expected_cache_dir = str(tmp_path / s3_prefix / model_name)
    assert results == [expected_cache_dir] * 8
    assert download_count == 1
    assert downloader._is_cache_complete(expected_cache_dir, include_model_weights=False)
