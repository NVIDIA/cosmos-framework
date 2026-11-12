# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Commit bounded-file-count PhysicalAI shards without changing PNG or row content.

Existing committed clips are read-only inputs. New staging may use one owned
scratch clip at a time; no migration or deletion of existing datasets is implied.
Completed shards are reusable; incomplete attempts remain for diagnosis and a
restart constructs a new shard rather than treating a partial file as complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

from cosmos_framework.posttrain.checkpoint.io import fsync_directory, rename_directory_noreplace
from cosmos_framework.posttrain.data.flow_media import (
    FORMAT,
    MAGIC,
    MAX_IMAGE_BYTES,
    media_reference,
    packed_index,
    read_packed_media,
)
from cosmos_framework.posttrain.model.alpamayo.sft import PHYSICALAI_CAMERAS, physicalai_frames
from cosmos_framework.posttrain.scripts.prepare_physicalai_flow_sft import verify_clip
from cosmos_framework.posttrain.utils.object_store import file_sha256


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        # Entry order is the payload order; it is validated on every index load.
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_shard(root, identities):
    root = Path(root)
    marker = json.loads((root / "complete.json").read_text())
    index, _ = packed_index(root, marker["index_sha256"])
    if [clip["preparation"]["identity"] for clip in index["clips"]] != identities:
        raise ValueError("PhysicalAI packed shard belongs to another ordered clip/source contract")
    expected_entries = {}
    for clip in index["clips"]:
        row, preparation = clip["row"], clip["preparation"]
        physicalai_frames(row)
        original_row_bytes = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
        if hashlib.sha256(original_row_bytes).hexdigest() != preparation["files"]["row.json"]:
            raise ValueError("PhysicalAI original row identity changed inside packed metadata")
        clip_id = preparation["identity"]["clip_id"]
        for camera in PHYSICALAI_CAMERAS:
            names = [f"{camera}-{i}.png" for i in range(4)]
            if row["media"][camera] != [f"clips/{clip_id}/{n}" for n in names]:
                raise ValueError("PhysicalAI packed source changed its native camera/frame paths")
            expected_entries.update({f"{clip_id}/{n}": preparation["files"][n] for n in names})
    if (
        {name: value["sha256"] for name, value in index["entries"].items()} != expected_entries
        or marker["clips"] != len(identities)
        or marker["images"] != len(expected_entries)
    ):
        raise ValueError("PhysicalAI packed PNG inventory differs from its committed native clips")
    if file_sha256(str(root / "media.bin")) != marker["payload_sha256"]:
        raise ValueError("PhysicalAI packed shard payload changed")
    for name in index["entries"]:
        read_packed_media(media_reference(root, marker["index_sha256"], name))
    return rows_for_shard(root, marker["index_sha256"], index), marker


def rows_for_shard(root, index_sha256, index=None):
    # References returned by the public CLI may be written beneath another
    # directory. An output-relative URI would be interpreted relative to that
    # JSONL a second time by FlowSFTDataset.
    root = Path(root).resolve()
    if index is None:
        index, _ = packed_index(root, index_sha256)
    result = []
    for clip in index["clips"]:
        row = json.loads(json.dumps(clip["row"]))
        clip_id = clip["preparation"]["identity"]["clip_id"]
        row["media"] = {
            camera: [media_reference(root, index_sha256, f"{clip_id}/{Path(p).name}") for p in paths]
            for camera, paths in row["media"].items()
        }
        result.append(row)
    return result


def write_shard(destination, clips, identities):
    """Pack an iterator of (committed_clip_directory, exact_identity), <=256 clips.

    The ordered identity list is known before any decoding/output inspection.
    The producer may release its own new scratch clip after the iterator advances.
    """
    root = Path(destination)
    if not 1 <= len(identities) <= 256 or len({x["clip_id"] for x in identities}) != len(identities):
        raise ValueError("a PhysicalAI shard requires 1..256 distinct, fixed ordered clip identities")
    if root.exists():
        return verify_shard(root, identities)
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = root.with_name(f".{root.name}.incomplete-{uuid.uuid4().hex}")
    temporary.mkdir()
    index = {"format": FORMAT, "clips": [], "entries": {}}
    payload_digest = hashlib.sha256(MAGIC)
    with (temporary / "media.bin").open("xb") as payload:
        payload.write(MAGIC)
        for position, (directory, identity) in enumerate(clips):
            if position >= len(identities) or identity != identities[position]:
                raise ValueError("packed PhysicalAI producer changed the frozen clip order/identity")
            directory = Path(directory)
            row, record = verify_clip(directory, identity)
            raw_row = (directory / "row.json").read_bytes()
            if hashlib.sha256(raw_row).hexdigest() != record["files"]["row.json"] or json.loads(raw_row) != row:
                raise ValueError("PhysicalAI row changed after source verification")
            for paths in row["media"].values():
                for path in paths:
                    filename = Path(path).name
                    member = f"{identity['clip_id']}/{filename}"
                    if member in index["entries"]:
                        raise ValueError("duplicate PhysicalAI packed image")
                    source = directory / filename
                    if not 0 < source.stat().st_size <= MAX_IMAGE_BYTES:
                        raise ValueError("PhysicalAI PNG exceeds bounded single-image buffer")
                    data = source.read_bytes()
                    digest = hashlib.sha256(data).hexdigest()
                    if digest != record["files"][filename]:
                        raise ValueError("PhysicalAI PNG changed after source verification")
                    index["entries"][member] = {"offset": payload.tell(), "bytes": len(data), "sha256": digest}
                    payload.write(data)
                    payload_digest.update(data)
            index["clips"].append({"row": row, "preparation": record})
        if len(index["clips"]) != len(identities):
            raise ValueError("PhysicalAI packed shard producer ended before all declared clips")
        payload.flush()
        os.fsync(payload.fileno())
    _write_json(temporary / "index.json", index)
    marker = {
        "format": FORMAT,
        "index_sha256": file_sha256(str(temporary / "index.json")),
        "payload_sha256": payload_digest.hexdigest(),
        "payload_bytes": (temporary / "media.bin").stat().st_size,
        "clips": len(identities),
        "images": len(index["entries"]),
    }
    _write_json(temporary / "complete.json", marker)
    verify_shard(temporary, identities)
    fsync_directory(str(temporary))
    rename_directory_noreplace(str(temporary), str(root))
    fsync_directory(str(root.parent))
    return rows_for_shard(root, marker["index_sha256"], index), marker


def publish_rows(path, rows):
    """Publish a complete bounded shard's rows; a partial JSONL is never visible."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # At most256 rows; image payloads remain streamed separately.
    encoded = "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows).encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError("existing PhysicalAI rows differ from the committed shard; use a fresh output path")
        return
    temporary = path.with_name(f".{path.name}.incomplete-{uuid.uuid4().hex}")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    # A hard link is an atomic no-replace publication for a same-directory
    # regular file. Interrupted temporary files are retained as evidence.
    os.link(temporary, path)
    fsync_directory(str(path.parent))
    temporary.unlink()
    fsync_directory(str(path.parent))


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", required=True, help="Ordered JSON list of directory/identity records, at most256")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", required=True, help="New JSONL path; existing files are never overwritten")
    args = parser.parse_args()
    records = json.loads(Path(args.clips).read_text())
    rows, _ = write_shard(
        args.out, ((x["directory"], x["identity"]) for x in records), [x["identity"] for x in records]
    )
    publish_rows(args.rows, rows)


if __name__ == "__main__":
    main()
