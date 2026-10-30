# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prepared physical demonstrations with content identities and episode split guards.

Each row has id, episode_id, observation, actions and media. Media maps camera
names to a path or ordered paths; interpretation belongs to the model packer.
Paths are relative to the JSONL, all referenced bytes enter the cursor identity.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from cosmos_framework.posttrain.utils.identity import canonical_json_sha256
from cosmos_framework.posttrain.utils.object_store import file_sha256


class FlowSFTDataset:
    def __init__(self, path: str):
        source = Path(path).expanduser().resolve()
        self.rows = []
        self.episodes = set()
        self.ids = set()
        self.media_hashes = {}
        for index, line in enumerate(source.read_text().splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
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
            for camera, paths in row["media"].items():
                sequence = paths if isinstance(paths, list) else [paths]
                if not sequence or any(not isinstance(p, str) or not p for p in sequence):
                    raise ValueError(f"flow row {index}: invalid camera paths for {camera}")
                resolved = [str((source.parent / p).resolve()) for p in sequence]
                for p in resolved:
                    if p not in self.media_hashes:
                        self.media_hashes[p] = file_sha256(p)
                row["media"][camera] = resolved if isinstance(paths, list) else resolved[0]
            self.rows.append(row)
        if not self.rows:
            raise ValueError("flow SFT dataset is empty")
        # Relocation does not change data identity; media digests replace paths.
        records = copy.deepcopy(self.rows)
        for row in records:
            row["media"] = {
                k: [self.media_hashes[p] for p in v] if isinstance(v, list) else self.media_hashes[v]
                for k, v in row["media"].items()
            }
        self.fingerprint = canonical_json_sha256({"format": "flow-sft-v1", "rows": records}, context="flow SFT")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return copy.deepcopy(self.rows[index])

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
