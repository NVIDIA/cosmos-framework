# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""`utils/artifact_identity.py`: the content identity a frozen KL reference directory contributes to the
objective exact-resume identity. Known answers: the same bytes at another location give the same identity;
other weights, other metadata, or any changed byte give another one; tokenizer files and the path never enter
it; an empty directory is refused."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cosmos_framework.posttrain.utils.artifact_identity import (
    ARTIFACT_IDENTITY_KIND,
    ARTIFACT_IDENTITY_VERSION,
    HASH_CHUNK_BYTES,
    model_artifact_identity,
)


def _weights(seed: int, size: int) -> bytes:
    import hashlib

    block = hashlib.sha256(f"weights-{seed}".encode()).digest()
    return (block * (size // len(block) + 1))[:size]


def _write_model_dir(root: Path, *, seed: int = 0, shards: int = 2, shard_bytes: int = 4096) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"architectures": ["Tiny"], "hidden_size": 8 + seed}))
    (root / "generation_config.json").write_text(json.dumps({"do_sample": True}))
    weight_map = {}
    for index in range(shards):
        name = f"model-{index + 1:05d}-of-{shards:05d}.safetensors"
        (root / name).write_bytes(_weights(seed * 100 + index, shard_bytes))
        weight_map[f"layer.{index}.weight"] = name
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (root / "tokenizer.json").write_text(json.dumps({"version": seed}))
    (root / ".cache").mkdir(exist_ok=True)
    (root / ".cache" / "model.safetensors").write_bytes(b"hidden directories are skipped")
    return root


@pytest.mark.L0
@pytest.mark.CPU
def test_identity_is_deterministic_and_location_independent(tmp_path: Path) -> None:
    base = _write_model_dir(tmp_path / "base", seed=1)
    first = model_artifact_identity(base)
    assert first == model_artifact_identity(str(base)), "two reads of one directory agree"
    assert first["version"] == ARTIFACT_IDENTITY_VERSION and first["kind"] == ARTIFACT_IDENTITY_KIND
    assert first["weight_files"] == 2 and first["weight_bytes"] == 2 * 4096
    assert isinstance(first["sha256"], str) and len(first["sha256"]) == 64
    copied = tmp_path / "elsewhere" / "copy-of-base"
    shutil.copytree(base, copied)
    assert model_artifact_identity(copied) == first, "the path is not part of the identity"
    # Tokenizer and processor files are not part of it either: they do not change reference logprobs.
    (copied / "tokenizer.json").write_text(json.dumps({"version": "other"}))
    assert model_artifact_identity(copied) == first


@pytest.mark.L0
@pytest.mark.CPU
def test_identity_changes_with_weights_metadata_or_layout(tmp_path: Path) -> None:
    base = model_artifact_identity(_write_model_dir(tmp_path / "base", seed=1))
    other_weights = model_artifact_identity(_write_model_dir(tmp_path / "step200", seed=2))
    assert other_weights["sha256"] != base["sha256"], "a checkpoint from another step is another anchor"
    same_path = _write_model_dir(tmp_path / "same", seed=1)
    before = model_artifact_identity(same_path)
    assert before == base
    _write_model_dir(same_path, seed=3)  # the same directory re-populated with other weights
    assert model_artifact_identity(same_path)["sha256"] != before["sha256"]
    config_only = _write_model_dir(tmp_path / "config_only", seed=1)
    (config_only / "config.json").write_text(json.dumps({"architectures": ["Tiny"], "hidden_size": 8, "tie": True}))
    assert model_artifact_identity(config_only)["sha256"] != base["sha256"], "metadata content is bound in full"
    resharded = _write_model_dir(tmp_path / "resharded", seed=1, shards=1, shard_bytes=8192)
    assert model_artifact_identity(resharded)["sha256"] != base["sha256"], "the artifact, not the function, is bound"


@pytest.mark.L0
@pytest.mark.CPU
def test_full_digest_covers_the_first_middle_and_last_bytes_of_a_large_shard(tmp_path: Path) -> None:
    """Every byte is bound, including bytes that a fixed-window fingerprint could skip."""
    size = HASH_CHUNK_BYTES * 2 + 12345
    root = _write_model_dir(tmp_path / "large", seed=1, shards=1, shard_bytes=size)
    shard = root / "model-00001-of-00001.safetensors"
    assert shard.stat().st_size == size
    base = model_artifact_identity(root)["sha256"]

    def flip(offset: int) -> None:
        with shard.open("r+b") as handle:
            handle.seek(offset)
            value = handle.read(1)
            assert len(value) == 1
            handle.seek(offset)
            handle.write(bytes([value[0] ^ 0xFF]))

    flip(0)
    first_byte = model_artifact_identity(root)["sha256"]
    assert first_byte != base
    flip(0)
    flip(size // 2)
    middle_byte = model_artifact_identity(root)["sha256"]
    assert middle_byte not in (base, first_byte)
    flip(size // 2)
    flip(size - 1)
    last_byte = model_artifact_identity(root)["sha256"]
    assert last_byte not in (base, first_byte, middle_byte)
    flip(size - 1)
    assert model_artifact_identity(root)["sha256"] == base, "restoring the bytes restores the identity"


@pytest.mark.L0
@pytest.mark.CPU
def test_identity_refuses_a_missing_or_weightless_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        model_artifact_identity(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no model weight file"):
        model_artifact_identity(empty)
    (empty / "config.json").write_text("{}")
    # Metadata without weights is a half-copied directory, not an anchor.
    with pytest.raises(ValueError, match="no model weight file"):
        model_artifact_identity(empty)
