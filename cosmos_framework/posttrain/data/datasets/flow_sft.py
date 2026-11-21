# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prepared physical demonstrations with content identities and episode split guards.

Each row has id, episode_id, observation, actions and media. Media maps camera
names to a path or ordered paths; interpretation belongs to the model packer.
Paths are relative to the JSONL, all referenced bytes enter the cursor identity.
Only row offsets/digests and the media/episode census stay resident. The canonical
flow-sft-v1 identity is byte-identical to the original in-memory implementation.
"""

from __future__ import annotations

import hashlib
import json
import re
from array import array
from pathlib import Path

from cosmos_framework.posttrain.data.flow_media import media_sha256, resolve_media
from cosmos_framework.posttrain.utils.identity import canonical_json_bytes


def _binary_rows(handle):
    """Bounded reads with the legacy UTF-8 ``str.splitlines`` separators.

    Retain three bytes at the read boundary so CRLF and UTF-8 line separators
    cannot be split. Offsets and digests always describe the original bytes.
    """
    separator = re.compile(rb"\r\n|[\n\r\v\f\x1c-\x1e]|\xc2\x85|\xe2\x80[\xa8\xa9]")
    buffer, offset = b"", 0
    while True:
        chunk = handle.read(65536)
        buffer += chunk
        start = 0
        for match in separator.finditer(buffer):
            if chunk and match.end() > len(buffer) - 3:
                break
            end = match.end()
            yield offset + start, buffer[start:end]
            start = end
        offset += start
        buffer = buffer[start:]
        if not chunk:
            if buffer:
                yield offset, buffer
            return


class FlowSFTDataset:
    def __init__(self, path: str):
        self.source = Path(path).expanduser().resolve()
        self._source_stat = self._stat_identity()
        self._offsets = array("Q")
        self._lengths = array("Q")
        self._row_digests = bytearray()
        self.episodes = set()
        self.ids = set()
        self.media_hashes = {}
        self._media_resolutions = {}
        digest = hashlib.sha256(b'{"format":"flow-sft-v1","rows":[')
        with self.source.open("rb") as handle:
            for index, (offset, line) in enumerate(_binary_rows(handle)):
                text = line.decode("utf-8").splitlines()[0]
                if not text.strip():
                    continue
                row = json.loads(text)
                for key in ("id", "episode_id"):
                    if not isinstance(row.get(key), str) or not row[key]:
                        raise ValueError(f"flow row {index}: {key} must be a nonempty string")
                if row["id"] in self.ids:
                    raise ValueError(f"duplicate flow demonstration id: {row['id']}")
                if not isinstance(row.get("observation"), dict) or not isinstance(row.get("media"), dict):
                    raise ValueError(f"flow row {index} requires observation and media mappings")
                if not isinstance(row.get("actions"), list) or not row["actions"]:
                    raise ValueError(f"flow row {index} requires an action sequence")
                json.dumps(row, allow_nan=False)
                self.ids.add(row["id"])
                self.episodes.add(row["episode_id"])
                self._resolve_media(row, index)
                for paths in row["media"].values():
                    for p in paths if isinstance(paths, list) else [paths]:
                        if p not in self.media_hashes:
                            self.media_hashes[p] = media_sha256(p)
                row["media"] = {
                    k: [self.media_hashes[p] for p in v] if isinstance(v, list) else self.media_hashes[v]
                    for k, v in row["media"].items()
                }
                if self._offsets:
                    digest.update(b",")
                digest.update(canonical_json_bytes(row, context="flow SFT"))
                self._offsets.append(offset)
                self._lengths.append(len(line))
                self._row_digests.extend(hashlib.sha256(line).digest())
        if not self._offsets:
            raise ValueError("flow SFT dataset is empty")
        if self._stat_identity() != self._source_stat:
            raise ValueError("flow SFT source changed while constructing its content identity")
        digest.update(b"]}")
        self.fingerprint = digest.hexdigest()

    def _stat_identity(self):
        stat = self.source.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def _resolve_media(self, row, index):
        for camera, paths in row["media"].items():
            sequence = paths if isinstance(paths, list) else [paths]
            if not sequence or any(not isinstance(p, str) or not p for p in sequence):
                raise ValueError(f"flow row {index}: invalid camera paths for {camera}")
            resolved = []
            for path in sequence:
                if path not in self._media_resolutions:
                    self._media_resolutions[path] = resolve_media(path, self.source.parent)
                resolved.append(self._media_resolutions[path])
            row["media"][camera] = resolved if isinstance(paths, list) else resolved[0]

    @property
    def rows(self):
        """Keep the sequence view without materializing the complete corpus."""
        return self

    def __len__(self):
        return len(self._offsets)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._stat_identity() != self._source_stat:
            raise ValueError("flow SFT source changed after identity validation; reconstruct the dataset")
        with self.source.open("rb") as handle:
            handle.seek(self._offsets[index])
            line = handle.read(self._lengths[index])
        expected = self._row_digests[index * 32 : (index + 1) * 32]
        if hashlib.sha256(line).digest() != expected:
            raise ValueError(f"flow SFT row {index} changed after identity validation")
        row = json.loads(line.decode("utf-8").splitlines()[0])
        self._resolve_media(row, index)
        return row

    def cursor_fingerprint(self):
        return self.fingerprint


def flow_source_overlap(training, validation):
    a, b = training.dataset, validation.dataset
    shared = a.episodes & b.episodes
    if shared or a.ids & b.ids or set(a.media_hashes.values()) & set(b.media_hashes.values()):
        raise ValueError(
            f"flow SFT train/validation overlap: shared episodes={sorted(shared)[:5]}; use disjoint episodes and media"
        )
    return {
        "training_rows": len(a),
        "validation_rows": len(b),
        "training_items": len(a.episodes),
        "validation_items": len(b.episodes),
    }
