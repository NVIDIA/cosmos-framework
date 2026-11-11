# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Historical flow source identities survive bounded-memory random access."""

import copy
import hashlib
import json
import pickle
import tracemalloc

import pytest

from cosmos_framework.posttrain.data.datasets.flow_sft import FlowSFTDataset
from cosmos_framework.posttrain.utils.identity import canonical_json_sha256


def rows_file(tmp_path, n=5):
    media = tmp_path / "frame.bin"
    media.write_bytes(b"frame")
    rows = [
        dict(
            id=f"row-{i}",
            episode_id=f"episode-{i}",
            observation={"state": [i, 0.25], "text": "\u00e9"},
            actions=[[0.1, 0.2]] * 64,
            media={"front": "frame.bin", "history": ["frame.bin"] * 4},
        )
        for i in range(n)
    ]
    path = tmp_path / "source.jsonl"
    path.write_text("\n" + "\n\n".join(json.dumps(row) for row in rows) + "\n")
    return path, rows


def test_exact_legacy_identity_random_access_and_pickle(tmp_path):
    path, rows = rows_file(tmp_path)
    historical = copy.deepcopy(rows)
    media_digest = hashlib.sha256(b"frame").hexdigest()
    for row in historical:
        row["media"] = {"front": media_digest, "history": [media_digest] * 4}
    expected = canonical_json_sha256({"format": "flow-sft-v1", "rows": historical}, context="legacy")
    data = FlowSFTDataset(str(path))
    assert data.cursor_fingerprint() == expected
    assert len(data) == len(rows)
    for i in [4, 0, 3, 1, 2]:
        row = data[i]
        assert row["actions"] == rows[i]["actions"]
        assert row["media"]["front"] == str(tmp_path / "frame.bin")
        row["observation"]["state"].append(9)
        assert data[i]["observation"] == rows[i]["observation"]
    assert data[-1] == data[4] and data.rows[1:3] == [data[1], data[2]]
    restored = pickle.loads(pickle.dumps(data))
    assert restored.cursor_fingerprint() == expected and restored[2] == data[2]
    with pytest.raises(IndexError):
        data[len(data)]
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    (relocated / path.name).write_bytes(path.read_bytes())
    (relocated / "frame.bin").write_bytes(b"frame")
    assert FlowSFTDataset(str(relocated / path.name)).cursor_fingerprint() == expected


def test_source_mutation_refused_after_indexing(tmp_path):
    path, _ = rows_file(tmp_path)
    data = FlowSFTDataset(str(path))
    path.write_bytes(path.read_bytes().replace(b"0.25", b"0.35"))
    with pytest.raises(ValueError, match="changed after identity"):
        data[0]


def test_media_aliases_remain_bound_to_the_validated_target(tmp_path):
    path, rows = rows_file(tmp_path, 2)
    (tmp_path / "other.bin").write_bytes(b"other frame")
    alias = tmp_path / "alias.bin"
    alias.symlink_to("frame.bin")
    rows[0]["media"] = {"front": "alias.bin"}
    rows[1]["media"] = {"front": "other.bin"}
    path.write_text("\n".join(json.dumps(row) for row in rows))
    data = FlowSFTDataset(str(path))
    identity = data.cursor_fingerprint()
    alias.unlink()
    alias.symlink_to("other.bin")
    assert data[0]["media"]["front"] == str(tmp_path / "frame.bin")
    assert data[1]["media"]["front"] == str(tmp_path / "other.bin")
    assert data.cursor_fingerprint() == identity


@pytest.mark.parametrize("separator", ["\n", "\r\n", "\r", "\v", "\f", "\x1c", "\x85", "\u2028", "\u2029"])
def test_legacy_line_separators_and_read_boundaries(tmp_path, separator):
    path, rows = rows_file(tmp_path, 3)
    # Cross the fixed binary read boundary, including a multibyte separator.
    rows[0]["observation"]["text"] = "a" * 65530
    expected = None
    for ending in ("\n", separator):
        path.write_bytes(
            (ending + "\u00a0" + ending * 2 + ending.join(json.dumps(row) for row in rows) + ending).encode()
        )
        data = FlowSFTDataset(str(path))
        assert [data[i]["id"] for i in range(3)] == [row["id"] for row in rows]
        if expected is None:
            expected = data.cursor_fingerprint()
        else:
            assert data.cursor_fingerprint() == expected


def test_large_action_payloads_are_not_retained_per_row(tmp_path):
    # 40MB of actions would occupy much more memory as decoded Python objects;
    # constructing an index must retain metadata, not all those action lists.
    (tmp_path / "frame.bin").write_bytes(b"frame")
    path = tmp_path / "source.jsonl"
    with path.open("w") as handle:
        for i in range(1000):
            handle.write(
                json.dumps(
                    dict(
                        id=str(i),
                        episode_id=str(i),
                        observation={},
                        actions=[[0.1] * 32] * 256,
                        media={"front": "frame.bin"},
                    )
                )
                + "\n"
            )
    tracemalloc.start()
    data = FlowSFTDataset(str(path))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # file_sha256 uses one fixed 8MiB read buffer even for the tiny shared
    # media file. Allow that buffer plus bounded row/index metadata.
    assert len(data) == 1000 and peak < 10 * 1024**2
    assert data[999]["actions"][0] == [0.1] * 32
