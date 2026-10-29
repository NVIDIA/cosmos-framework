# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Losslessly convert local WTS video-QA annotations to the existing SFT JSONL schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def convert(annotations: Path, media_root: Path, output: Path) -> dict:
    records = json.loads(annotations.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("WTS annotations must be a non-empty list")
    root = media_root.resolve(strict=True)
    rows, identifiers = [], set()
    for source in records:
        identifier = source["id"]
        if identifier in identifiers:
            raise ValueError(f"duplicate WTS id {identifier!r}")
        identifiers.add(identifier)
        video = (root / source["video"]).resolve(strict=True)
        if not video.is_relative_to(root) or not video.is_file():
            raise ValueError(f"WTS video escapes media root or is not a file: {video}")
        turns = source["conversations"]
        if len(turns) != 2 or [turn["from"] for turn in turns] != ["human", "gpt"]:
            raise ValueError(f"WTS row {identifier!r} must contain exactly human/gpt QA turns")
        question, answer = turns[0]["value"], turns[1]["value"].strip()
        if question.count("<video>") != 1 or answer not in ("A", "B", "C", "D"):
            raise ValueError(f"WTS row {identifier!r} has invalid video marker or multiple-choice answer")
        rows.append(
            {
                "id": identifier,
                "messages": [
                    {"role": "system", "content": "Answer the multiple-choice question with its letter."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": str(video)},
                            {"type": "text", "text": question.replace("<video>", "").strip()},
                        ],
                    },
                    {"role": "assistant", "content": answer},
                ],
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return {
        "rows": len(rows),
        "unique_ids": len(identifiers),
        "annotations_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
        "jsonl_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "output": str(output.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.annotations, args.media_root, args.out), indent=2))


if __name__ == "__main__":
    main()
