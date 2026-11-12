# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Lossless physical-data representation, identity, failure and fork checks."""

import copy
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pytest

from cosmos_framework.posttrain.data.datasets.flow_sft import FlowSFTDataset
from cosmos_framework.posttrain.data.flow_media import media_reference, open_flow_image, packed_index, read_packed_media
from cosmos_framework.posttrain.model.alpamayo.sft import physicalai_frames
from cosmos_framework.posttrain.scripts import pack_physicalai_flow_sft as pack
from cosmos_framework.posttrain.scripts import prepare_physicalai_flow_sft as prepare
from cosmos_framework.posttrain.scripts.prepare_physicalai_flow_sft import write_clip
from cosmos_framework.posttrain.tests.test_prepare_physicalai_flow_sft import source


def prepared(tmp_path):
    rows, clips, identities = [], [], []
    for name in ["first-clip", "second-clip"]:
        sample, calibration, identity = source()
        sample["clip_id"] = identity["clip_id"] = name
        directory = tmp_path / "clips" / name
        row, _ = write_clip(sample, calibration, directory, identity)
        rows.append(row)
        clips.append((directory, identity))
        identities.append(identity)
    return rows, clips, identities


def inventory(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()
    }


def dataset(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return FlowSFTDataset(str(path))


def _fork_read(queue, references):
    try:
        queue.put([hashlib.sha256(read_packed_media(r)).hexdigest() for r in references] * 3)
    except Exception as error:
        queue.put(repr(error))


def test_lossless_shard_preserves_complete_logical_dataset_and_reuse(tmp_path):
    rows, clips, identities = prepared(tmp_path)
    before = inventory(tmp_path / "clips")
    output = tmp_path / "packed"
    packed, marker = pack.write_shard(output, iter(clips), identities)
    a, b = dataset(tmp_path / "legacy.jsonl", rows), dataset(tmp_path / "packed.jsonl", packed)
    assert a.cursor_fingerprint() == b.cursor_fingerprint()
    assert a.ids == b.ids and a.episodes == b.episodes
    assert marker["clips"] == 2 and marker["images"] == 32
    assert {p.name for p in output.iterdir()} == {"media.bin", "index.json", "complete.json"}
    for i in [1, 0, 1]:
        assert a[i]["observation"] == b[i]["observation"] and a[i]["actions"] == b[i]["actions"]
        for original, stored in zip(physicalai_frames(a[i]), physicalai_frames(b[i]), strict=True):
            assert Path(original).read_bytes() == read_packed_media(stored)
            with open_flow_image(original) as left, open_flow_image(stored) as right:
                np.testing.assert_array_equal(np.asarray(left.convert("RGB")), np.asarray(right.convert("RGB")))
    committed = inventory(output)

    # A completed shard must not advance the source iterator or regenerate data.
    def forbidden():
        raise AssertionError("completed shard consumed its producer")
        yield

    assert pack.write_shard(output, forbidden(), identities) == (packed, marker)
    assert inventory(output) == committed and inventory(tmp_path / "clips") == before
    with pytest.raises(ValueError, match="another ordered"):
        pack.verify_shard(output, list(reversed(identities)))
    references = physicalai_frames(b[0])
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    processes = [context.Process(target=_fork_read, args=(queue, references)) for _ in range(2)]
    for process in processes:
        process.start()
    expected = [hashlib.sha256(read_packed_media(r)).hexdigest() for r in references] * 3
    for _ in processes:
        assert queue.get(timeout=20) == expected
    for process in processes:
        process.join(20)
        assert process.exitcode == 0


def test_interruption_keeps_partial_and_retry_uses_new_commit(tmp_path, monkeypatch):
    _, clips, identities = prepared(tmp_path)
    destination = tmp_path / "packed"
    real = pack._write_json

    def interrupt(path, value):
        if path.name == "complete.json":
            raise RuntimeError("interrupted before commit")
        return real(path, value)

    monkeypatch.setattr(pack, "_write_json", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        pack.write_shard(destination, iter(clips), identities)
    assert not destination.exists()
    (partial,) = tmp_path.glob(".packed.incomplete-*")
    before = inventory(partial)
    with pytest.raises(ValueError, match="incomplete"):
        packed_index(partial, hashlib.sha256((partial / "index.json").read_bytes()).hexdigest())
    monkeypatch.setattr(pack, "_write_json", real)
    pack.write_shard(destination, iter(clips), identities)
    assert inventory(partial) == before
    pack.verify_shard(destination, identities)


def test_relative_public_output_is_readable_from_nested_rows_file(tmp_path, monkeypatch):
    rows, clips, identities = prepared(tmp_path)
    monkeypatch.chdir(tmp_path)
    packed, _ = pack.write_shard("out/shard", iter(clips), identities)
    a = dataset(tmp_path / "legacy.jsonl", rows)
    b = dataset(tmp_path / "out/rows.jsonl", packed)
    assert a.cursor_fingerprint() == b.cursor_fingerprint()
    for original, stored in zip(physicalai_frames(a[0]), physicalai_frames(b[0]), strict=True):
        assert Path(original).read_bytes() == read_packed_media(stored)


def test_public_index_mutation_cannot_redirect_camera_frame_reads(tmp_path):
    _, clips, identities = prepared(tmp_path)
    root = tmp_path / "shard"
    rows, marker = pack.write_shard(root, iter(clips), identities)
    ref = physicalai_frames(rows[0])[0]
    expected = read_packed_media(ref)
    index, _ = packed_index(root, marker["index_sha256"])
    first, second = list(index["entries"])[:2]
    index["entries"][first] = index["entries"][second]
    index["clips"].clear()
    assert read_packed_media(ref) == expected
    original, _ = packed_index(root, marker["index_sha256"])
    assert original["entries"][first] != original["entries"][second]
    assert len(original["clips"]) == 2


def test_rows_publication_is_complete_atomic_and_reusable(tmp_path, monkeypatch):
    _, clips, identities = prepared(tmp_path)
    rows, _ = pack.write_shard(tmp_path / "shard", iter(clips), identities)
    path = tmp_path / "rows.jsonl"
    link = pack.os.link

    def interrupt(*args):
        raise RuntimeError("interrupted before row publication")

    monkeypatch.setattr(pack.os, "link", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        pack.publish_rows(path, rows)
    assert not path.exists()
    (partial,) = tmp_path.glob(".rows.jsonl.incomplete-*")
    saved = partial.read_bytes()
    monkeypatch.setattr(pack.os, "link", link)
    pack.publish_rows(path, rows)
    assert len(FlowSFTDataset(str(path))) == 2
    before = path.stat()
    pack.publish_rows(path, rows)
    assert path.stat() == before and partial.read_bytes() == saved
    with pytest.raises(ValueError, match="existing.*differ"):
        pack.publish_rows(path, rows[:1])


@pytest.mark.parametrize("mutation", ["payload", "index", "marker", "missing_marker", "wrong_member", "wrong_digest"])
def test_changed_or_incomplete_shard_rejected(tmp_path, mutation):
    _, clips, identities = prepared(tmp_path)
    destination = tmp_path / "packed"
    rows, marker = pack.write_shard(destination, iter(clips), identities)
    ref = physicalai_frames(rows[0])[0]
    read_packed_media(ref)  # populate index cache before mutating the source
    if mutation == "payload":
        with (destination / "media.bin").open("r+b") as f:
            f.seek(len(pack.MAGIC) + 7)
            value = f.read(1)
            f.seek(-1, 1)
            f.write(bytes([value[0] ^ 1]))
    elif mutation == "missing_marker":
        (destination / "complete.json").unlink()
    elif mutation in {"index", "marker"}:
        (destination / ("index.json" if mutation == "index" else "complete.json")).write_text("{}")
    else:
        ref = media_reference(
            destination,
            "0" * 64 if mutation == "wrong_digest" else marker["index_sha256"],
            "absent.png" if mutation == "wrong_member" else "first-clip/camera_cross_left_120fov-0.png",
        )
    with pytest.raises(ValueError):
        read_packed_media(ref)


def test_wrong_order_incomplete_producer_and_source_metadata_are_rejected(tmp_path):
    _, clips, identities = prepared(tmp_path)
    with pytest.raises(ValueError, match="frozen clip order"):
        pack.write_shard(tmp_path / "wrong-order", reversed(clips), identities)
    with pytest.raises(ValueError, match="ended before"):
        pack.write_shard(tmp_path / "missing", iter(clips[:1]), identities)
    different = copy.deepcopy(identities)
    different[0]["source_revision"] = "b" * 40
    with pytest.raises(ValueError, match="identity differs"):
        pack.write_shard(tmp_path / "wrong-source", [(clips[0][0], different[0]), clips[1]], different)


def test_future_preparation_uses_one_owned_scratch_clip_and_reuses_complete_shard(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    protected = scratch / "pre-existing-user-file"
    protected.write_bytes(b"must remain untouched")
    calls = []

    def identity(avdi, clip, **kwargs):
        result = source()[2]
        result["clip_id"] = clip
        return result

    def native(avdi, clip, out, **kwargs):
        assert len(list(scratch.glob("cosmos-physicalai-pack-*"))) == 1
        calls.append(clip)
        sample, calibration, contract = source()
        sample["clip_id"] = contract["clip_id"] = clip
        return write_clip(sample, calibration, Path(out) / "clips" / clip, contract)

    monkeypatch.setattr(prepare, "clip_identity", identity)
    monkeypatch.setattr(prepare, "prepare_clip", native)
    manifest = {"revision": "a" * 40, "files": {}}
    names = ["a", "b", "c"]
    prepare.prepare_packed_clips(None, names, tmp_path / "out", scratch, manifest, 2)
    assert calls == names and len(list((tmp_path / "out").iterdir())) == 2
    assert not list(scratch.glob("cosmos-physicalai-pack-*"))
    before = inventory(tmp_path / "out")
    prepare.prepare_packed_clips(None, names, tmp_path / "out", scratch, manifest, 2)
    assert calls == names and inventory(tmp_path / "out") == before
    assert protected.read_bytes() == b"must remain untouched"


def test_failed_future_preparation_preserves_only_its_new_scratch(tmp_path, monkeypatch):
    contract = source()[2]
    monkeypatch.setattr(prepare, "clip_identity", lambda *a, **k: contract)

    def failed(avdi, clip, out, **kwargs):
        (Path(out) / "failure-evidence").write_text("native decode failed")
        raise RuntimeError("native decode failed")

    monkeypatch.setattr(prepare, "prepare_clip", failed)
    with pytest.raises(RuntimeError, match="native decode failed"):
        prepare.prepare_packed_clips(
            None, ["first-clip"], tmp_path / "out", tmp_path / "scratch", {"revision": "a" * 40, "files": {}}
        )
    (evidence,) = (tmp_path / "scratch").glob("cosmos-physicalai-pack-*/failure-evidence")
    assert evidence.read_text() == "native decode failed"
    assert not list((tmp_path / "out").glob("shard-*/complete.json"))
