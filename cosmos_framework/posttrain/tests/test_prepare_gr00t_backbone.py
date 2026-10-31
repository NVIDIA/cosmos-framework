# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fail closed on substituted/incomplete backbone metadata before publication."""

import hashlib
import json

import pytest

from cosmos_framework.posttrain.scripts import prepare_gr00t_backbone as preparation


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    blobs = {}
    for name in preparation.METADATA_BLOBS:
        content = json.dumps({"fixture": name}).encode()
        (root / name).write_bytes(content)
        blobs[name] = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
    monkeypatch.setattr(preparation, "METADATA_BLOBS", blobs)
    (root / "model.safetensors").write_bytes(b"must not be copied")
    return root


def test_identical_metadata_only_publication(source, tmp_path):
    destination = tmp_path / "prepared"
    record = preparation.stage(str(source), str(destination))
    assert set(p.name for p in destination.iterdir()) == set(preparation.METADATA_BLOBS) | {"metadata_provenance.json"}
    for name in record["files"]:
        assert (destination / name).read_bytes() == (source / name).read_bytes()
    assert json.loads((destination / "metadata_provenance.json").read_text()) == record
    with pytest.raises(FileExistsError):
        preparation.stage(str(source), str(destination))


@pytest.mark.parametrize("failure", ["substituted", "missing"])
def test_bad_metadata_publishes_nothing(source, tmp_path, failure):
    target = source / "vocab.json"  # late failure must not leave a partial snapshot
    if failure == "substituted":
        target.write_bytes(b"different tokenizer")
    else:
        target.unlink()
    destination = tmp_path / "prepared"
    with pytest.raises(ValueError, match="backbone metadata"):
        preparation.stage(str(source), str(destination))
    assert not destination.exists()
