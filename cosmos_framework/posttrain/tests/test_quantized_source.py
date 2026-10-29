# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Immutable-source contracts on tiny byte fixtures, without loading model weights."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cosmos_framework.posttrain.utils import adapter_source as source

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _snapshot(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    files = {
        "config.json": {
            "model_type": "qwen3_vl",
            "text_config": {"num_hidden_layers": 36, "hidden_size": 4096, "intermediate_size": 12288},
        },
        "tokenizer_config.json": {"chat_template": "fixture-template"},
        "tokenizer.json": {"fixture": True},
        "preprocessor_config.json": {"fixture": "image"},
        "video_preprocessor_config.json": {"fixture": "video"},
        "model.safetensors.index.json": {"weight_map": {"fixture.weight": "model-00001-of-00001.safetensors"}},
    }
    for name, value in files.items():
        (root / name).write_text(json.dumps(value))
    (root / "model-00001-of-00001.safetensors").write_bytes(b"source-hash-fixture-not-real-weights")
    identity = source._identity(root)
    (root / source.SOURCE_MANIFEST).write_text(json.dumps(identity))
    return root, identity


def test_snapshot_verification_binds_weights_processor_and_returns_independent_identity(tmp_path):
    root, expected = _snapshot(tmp_path)
    actual = source.verify_adapter_snapshot(root)
    assert actual == expected
    actual["files"].clear()
    assert source.verify_adapter_snapshot(root) == expected
    assert actual["weights_sha256"] != actual["processor_sha256"]


@pytest.mark.parametrize(
    "filename", ["model-00001-of-00001.safetensors", "tokenizer_config.json", "video_preprocessor_config.json"]
)
def test_source_mutation_invalidates_success_cache_even_with_restored_mtime(tmp_path, filename):
    root, _ = _snapshot(tmp_path)
    source.verify_adapter_snapshot(root)
    path = root / filename
    stamp = path.stat()
    content = path.read_bytes()
    if filename.endswith(".json"):
        payload = json.loads(content)
        payload["changed"] = True
        path.write_text(json.dumps(payload))
    else:
        path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match="contents differ"):
        source.verify_adapter_snapshot(root)


def test_missing_shard_and_inventory_addition_fail_explicitly(tmp_path):
    root, _ = _snapshot(tmp_path)
    (root / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory changed"):
        source.verify_adapter_snapshot(root)
    (root / "extra.json").unlink()
    (root / "model-00001-of-00001.safetensors").unlink()
    with pytest.raises(ValueError, match="inventory changed"):
        source.verify_adapter_snapshot(root)


@pytest.mark.parametrize(
    "change", [{"revision": "wrong"}, {"protocol": "unknown"}, {"files": {}}, {"repository": "other/model"}]
)
def test_incompatible_or_missing_identity_is_refused(tmp_path, change):
    root, identity = _snapshot(tmp_path)
    identity.update(change)
    (root / source.SOURCE_MANIFEST).write_text(json.dumps(identity))
    with pytest.raises(ValueError):
        source.verify_adapter_snapshot(root)


def test_native_model_and_processor_must_share_verified_source(tmp_path):
    root, identity = _snapshot(tmp_path)
    cfg = SimpleNamespace(
        adapter=object(),
        quantized_base=None,
        name_or_path="",
        checkpoint_path="",
        revision=source.SOURCE_REVISION,
        source_identity={"repository": source.SOURCE_REPOSITORY, "revision": source.SOURCE_REVISION},
    )
    spec = SimpleNamespace(model_id=str(root), model_config=cfg)
    assert source.prepare_sft_adapter_source(spec) == identity
    assert cfg.source_identity == identity
    cfg.name_or_path = str(root)
    assert source.validate_adapter_source(cfg) == identity
    cfg.checkpoint_path = str(tmp_path / "other")
    with pytest.raises(ValueError, match="sources must be identical"):
        source.validate_adapter_source(cfg)
    cfg.checkpoint_path = ""
    cfg.source_identity["processor_sha256"] = "wrong"
    with pytest.raises(ValueError, match="identity mismatch"):
        source.validate_adapter_source(cfg)


def test_unprepared_or_wrong_revision_never_falls_back_to_hub(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        source.verify_adapter_snapshot(tmp_path)
    with pytest.raises(ValueError, match="exact pinned"):
        source.validate_adapter_source(SimpleNamespace(revision="main", name_or_path="Qwen/Qwen3-VL-8B-Instruct"))


@pytest.mark.parametrize("value", [[], {}, 7, None, "../elsewhere.safetensors", "weights.bin"])
def test_malformed_shard_values_raise_intentional_validation_errors(tmp_path, value):
    root, _ = _snapshot(tmp_path)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"fixture.weight": value}}))
    with pytest.raises(ValueError, match="invalid shard path"):
        source._identity(root)


@pytest.mark.parametrize("filename", ["stale.json", "old.safetensors", "pytorch_model.bin", ".cache"])
def test_source_preparation_refuses_unmanifested_contents_before_download(tmp_path, monkeypatch, filename):
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **kwargs: pytest.fail("must refuse before download")),
    )
    root = tmp_path / "unprepared"
    root.mkdir()
    (root / filename).write_text("stale")
    with pytest.raises(ValueError, match="refusing unmanifested"):
        source.prepare_adapter_snapshot(str(root))
    assert (root / filename).read_text() == "stale"
    assert not (root / source.SOURCE_MANIFEST).exists()


def test_new_preparation_allows_only_real_cache_directory_and_pins_download(tmp_path, monkeypatch):
    fixture_parent = tmp_path / "fixture"
    fixture_parent.mkdir()
    fixture, expected = _snapshot(fixture_parent)
    root = tmp_path / "prepared"
    (root / ".cache").mkdir(parents=True)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        for path in fixture.iterdir():
            if path.name != source.SOURCE_MANIFEST:
                (Path(kwargs["local_dir"]) / path.name).write_bytes(path.read_bytes())

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    assert source.prepare_adapter_snapshot(str(root)) == expected
    assert calls[0]["repo_id"] == source.SOURCE_REPOSITORY
    assert calls[0]["revision"] == source.SOURCE_REVISION
    assert calls[0]["force_download"] is True
    assert source.prepare_adapter_snapshot(str(root)) == expected
    assert len(calls) == 1  # a completed source is verified without another download


def test_source_preparation_does_not_accept_cache_symlink(tmp_path):
    root, elsewhere = tmp_path / "root", tmp_path / "elsewhere"
    root.mkdir()
    elsewhere.mkdir()
    (root / ".cache").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="refusing unmanifested"):
        source.prepare_adapter_snapshot(str(root))


def test_cpu_preparation_modules_do_not_import_accelerator_or_hub_stacks():
    code = (
        "import sys; "
        "import cosmos_framework.posttrain.scripts.prepare_qlora_source; "
        "import cosmos_framework.posttrain.scripts.prepare_qlora_wts; "
        "assert not {'torch', 'transformers', 'bitsandbytes', 'huggingface_hub'}.intersection(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
