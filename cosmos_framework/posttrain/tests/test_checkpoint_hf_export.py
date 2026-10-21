# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""HF export compatibility, completeness, and publication contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmos_framework.posttrain.checkpoint.conversion import hf
from cosmos_framework.posttrain.checkpoint.hf_export import PUBLICATION_COMMIT, export_files, publish_hf_export
from cosmos_framework.posttrain.tests.helpers.object_store_fake import FakeS3Client
from cosmos_framework.posttrain.utils import object_store

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _identity(hidden: int = 8) -> dict[str, object]:
    return {
        "model_type": "qwen3_vl_moe",
        "architectures": ["Qwen3VLMoeForConditionalGeneration"],
        "text_config": {"vocab_size": 100, "hidden_size": hidden, "num_hidden_layers": 2},
    }


def test_compatibility_overlay_validates_identity_and_preserves_original(tmp_path: Path) -> None:
    base, export = tmp_path / "base", tmp_path / "export"
    base.mkdir()
    export.mkdir()
    (base / "config.json").write_text(json.dumps({**_identity(), "transformers_version": "4.57"}))
    (export / "config.json").write_text(json.dumps({**_identity(), "transformers_version": "5.13"}))
    (base / "tokenizer.json").write_text("base")
    (export / "tokenizer.json").write_text("trainer")
    assert set(hf.overlay_compatibility_files(str(export), str(base))) == {"config.json", "tokenizer.json"}
    assert (export / "tokenizer.json.trainer.bak").read_text() == "trainer"
    (export / "config.json").write_text(json.dumps({**_identity(16), "transformers_version": "5.13"}))
    with pytest.raises(hf.ConversionError, match="identity mismatch"):
        hf.overlay_compatibility_files(str(export), str(base))


def test_hf_export_completeness_and_transactional_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    export = tmp_path / "hf"
    export.mkdir()
    shard = "model-00001-of-00001.safetensors"
    (export / "config.json").write_text("{}")
    (export / hf.INDEX_FILE).write_text(json.dumps({"weight_map": {"x": shard}}))
    assert not hf.hf_export_complete(str(export))
    (export / shard).write_bytes(b"weights")
    (export / "config.json.trainer.bak").write_text("local")
    assert hf.hf_export_complete(str(export))
    assert export_files(str(export)) == ["config.json", shard, hf.INDEX_FILE]

    client = FakeS3Client()
    monkeypatch.setattr(object_store, "build_s3_client", lambda credentials: client)
    monkeypatch.setattr(object_store, "_single_put_transfer_config", lambda: "single")
    result = publish_hf_export(
        str(export),
        "model",
        bucket="bucket",
        prefix="served/model",
        credentials="credentials.json",
        workers=2,
    )
    keys = [key for _bucket, key, _extra in client.uploads]
    assert result.written_objects == 4
    assert keys[-2:] == [f"served/model/{hf.INDEX_FILE}", f"served/model/{PUBLICATION_COMMIT}"]
    assert not any(key.endswith(".trainer.bak") for key in keys)
