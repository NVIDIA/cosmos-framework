#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Import generated clips into the local video labeler without inferring labels."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def validate_config(config):
    assert config["schema_version"] == "1.0"
    for name in ("state", "phase"):
        track = config["tracks"][name]
        keys = [label["key"] for label in track["labels"]]
        hotkeys = [label["hotkey"].lower() for label in track["labels"]]
        assert len(keys) == len(set(keys)), (name, "duplicate label keys")
        assert len(hotkeys) == len(set(hotkeys)), (name, "duplicate hotkeys")
        assert track["fill_label"] in keys
    outcomes = [item["key"] for item in config["outcomes"]]
    assert len(outcomes) == len(set(outcomes))
    assert config["default_outcome"] in outcomes
    cameras = config["cameras"]
    assert len({camera["key"] for camera in cameras}) == len(cameras)
    assert sum(bool(camera.get("primary")) for camera in cameras) == 1
    assert any("/synthetic/" in camera["path_contains"] for camera in cameras)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", choices=("failure", "all"), required=True)
    args = parser.parse_args()
    batch, output = args.batch.resolve(), args.output.resolve()
    config = json.loads(args.config.read_text())
    validate_config(config)
    key = json.loads((batch / "review_key.json").read_text())
    selected = [
        (clip_id, record)
        for clip_id, record in sorted(key.items())
        if args.selection == "all" or record["condition"] == "failure_donor"
    ]
    if not selected:
        raise SystemExit("No clips matched the requested selection.")
    # The review IDs provide a stable shuffled order, with no class names in paths.
    episodes = []
    for index, (clip_id, record) in enumerate(selected, 1):
        source = (batch / record["video"]).resolve()
        source.relative_to(batch)
        if not source.is_file() or source.stat().st_size == 0:
            raise SystemExit(f"Missing/empty video: {source}")
        episodes.append(
            dict(
                episode_id=f"episode_{index:06d}",
                video=f"videos/synthetic/episode_{index:06d}.mp4",
                source_video=str(source),
                original_review_clip_id=clip_id,
                run_name=record["name"],
                condition=record["condition"],
                intended_failure=record["family"],
                seed=record["seed"],
                host_episode=record["host_episode"],
                source_sha256=digest(source),
            )
        )
    document = dict(
        schema_version=1,
        source_job=args.job_id,
        source_batch=str(batch),
        selection=args.selection,
        episodes=episodes,
        video_storage="Independent byte-identical copies; originals preserved.",
        labeling="Semantic labels require visual inspection; requested conditions are provenance only.",
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, expected in (("labeling_config.json", config), ("synthetic_manifest.json", document)):
        existing = output / name
        if existing.exists() and json.loads(existing.read_text()) != expected:
            raise SystemExit(f"Refusing to replace incompatible existing file: {existing}")
    for row in episodes:
        destination = output / row["video"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if digest(destination) != row["source_sha256"]:
                raise SystemExit(f"Existing video differs: {destination}")
        else:
            shutil.copy2(row["source_video"], destination)
    write_json(output / "labeling_config.json", config)
    write_json(output / "synthetic_manifest.json", document)
    print(f"Imported {len(episodes)} clips into {output}")
    print("Existing annotations are preserved. No semantic labels were assigned.")


if __name__ == "__main__":
    main()
