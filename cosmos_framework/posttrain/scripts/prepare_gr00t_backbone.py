# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Stage the exact GR00T-N1.7 backbone metadata from identical public files.

These nine Git blob IDs are from the immutable Cosmos-Reason2-2B Hub tree,
not inferred from another model's architecture. Qwen3-VL-2B-Instruct publishes
identical bytes at the revision below. GR00T's complete checkpoint supplies
all model weights; this tool never copies or downloads backbone weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

REFERENCE_REPO = "nvidia/Cosmos-Reason2-2B"
REFERENCE_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
PUBLIC_REPO = "Qwen/Qwen3-VL-2B-Instruct"
PUBLIC_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
# Verified against https://huggingface.co/api/models/nvidia/Cosmos-Reason2-2B/tree/
# 9ce19a195e423419c349abfc86fd07178b230561?recursive=true&expand=true
METADATA_BLOBS = {
    "chat_template.json": "e49e75bdda104c8de941eb20e0b3765184fb27bc",
    "config.json": "0cd8c646eb55794594b14fefcd1d3acea5cc000f",
    "generation_config.json": "7bb37933d2d86e4ceaabcc6540be989658f238e9",
    "merges.txt": "20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0",
    "preprocessor_config.json": "2ea84a437d448ff71b08df68fdd949d5cc4ebb64",
    "tokenizer.json": "c6cc1014128b19d1fc46b1d30a23e3b1d35db421",
    "tokenizer_config.json": "d3d3763207692c78780f4bf42d4dadf49a5c8012",
    "video_preprocessor_config.json": "3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe",
    "vocab.json": "4783fe10ac3adce15ac8f358ef5462739852c569",
}


def stage(source: str, out: str) -> dict:
    root, destination = Path(source), Path(out)
    if destination.exists():
        raise FileExistsError(f"backbone metadata destination already exists: {destination}")
    payloads, files = {}, {}
    for name, expected in METADATA_BLOBS.items():
        path = root / name
        if not path.is_file():
            raise ValueError(f"missing exact GR00T backbone metadata: {name}; stage {PUBLIC_REPO}@{PUBLIC_REVISION}")
        content = path.read_bytes()
        oid = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if oid != expected:
            raise ValueError(
                f"GR00T backbone metadata differs: {name}; expected Git blob {expected} "
                f"from {REFERENCE_REPO}@{REFERENCE_REVISION}, got {oid}; no files published"
            )
        payloads[name] = content
        files[name] = {"git_blob_oid": oid, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    record = {
        "format": "cosmos-gr00t-exact-backbone-metadata-v1",
        "reference": {"repository": REFERENCE_REPO, "revision": REFERENCE_REVISION},
        "public_identical_source": {"repository": PUBLIC_REPO, "revision": PUBLIC_REVISION},
        "files": files,
        "weights": "not included; load the complete GR00T-N1.7 checkpoint",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gr00t-backbone-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "metadata"
        staged.mkdir()
        for name, content in payloads.items():
            with (staged / name).open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        # Provenance is operational evidence. Adapter identity hashes the actual
        # processor files, so an authorized direct snapshot is interchangeable.
        (staged / "metadata_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
        staged.rename(destination)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="local pinned public Qwen3-VL-2B metadata snapshot")
    parser.add_argument("--out", required=True, help="new metadata-only directory")
    args = parser.parse_args()
    print(json.dumps(stage(args.source, args.out), indent=2))


if __name__ == "__main__":
    main()
