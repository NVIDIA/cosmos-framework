# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prepare a small, episode-disjoint YAM FD dataset; no GPU or video re-encoding."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from failuregen.yam_training.contract import (
    CAMERAS,
    CAPTION,
    CHUNK,
    CONTRACT,
    FPS,
    STRIDE,
    file_sha256,
    raw_actions,
    valid_starts,
)


def split_episodes(episodes: dict, seed: int) -> dict[str, list[int]]:
    """Stratify by recorded outcome, never by frame/window."""
    groups = {}
    for key, item in sorted(episodes.items()):
        groups.setdefault(item["outcome"], []).append(int(key.split("_")[-1]))
    result = {name: [] for name in ("train", "val", "test")}
    rng = random.Random(seed)
    for ids in groups.values():
        rng.shuffle(ids)
        held = max(1, round(len(ids) * 0.1)) if len(ids) >= 3 else 0
        result["val"].extend(ids[:held])
        result["test"].extend(ids[held : 2 * held])
        result["train"].extend(ids[2 * held :])
    return {name: sorted(ids) for name, ids in result.items()}


def prepare(
    dataset: Path,
    output: Path,
    *,
    seed: int = 42,
    annotations: str = "failuregen_annotations.json",
    splits_path: Path | None = None,
) -> dict:
    dataset = dataset.resolve()
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    if info["fps"] != 30:
        raise ValueError("This recipe requires 30 Hz source recordings")
    names = [
        f"observation.state.{arm}_yam_{joint}"
        for arm in ("left", "right")
        for joint in [*(f"joint{i}" for i in range(1, 7)), "gripper"]
    ]
    if info["features"]["observation.state"].get("names") != names:
        raise ValueError("Measured-state feature ordering differs from the YAM contract")
    for camera in CAMERAS:
        feature = info["features"][camera]
        if feature["shape"] != [480, 640, 3] or feature["video_info"]["video.fps"] != 30:
            raise ValueError(f"Unexpected shape/fps for {camera}")
    annotations_path = dataset / annotations
    selected = json.loads(annotations_path.read_text())["episodes"]
    splits = json.loads(splits_path.read_text()) if splits_path else split_episodes(selected, seed)
    ids = [int(key.split("_")[-1]) for key in selected]
    flat = [ep for values in splits.values() for ep in values]
    if set(splits) != {"train", "val", "test"} or sorted(flat) != sorted(ids) or len(set(flat)) != len(flat):
        raise ValueError("Splits must partition the selected episodes exactly once")
    if not all(splits.values()):
        raise ValueError("Train, validation, and test splits must be nonempty")
    output.mkdir(parents=True, exist_ok=False)
    (output / "states").mkdir()
    manifest = {
        "status": "in_progress",
        "contract": CONTRACT,
        "seed": seed,
        "splits": splits,
        "split_basis": "provided episode split" if splits_path else "outcome-stratified episode split",
        "collection_run_aware": False,
        "source_info_sha256": file_sha256(info_path),
        "annotations": annotations,
        "annotations_sha256": file_sha256(annotations_path),
        "window_stride": 1,
        "episodes": [],
        "timing_verified_visually": False,
    }
    train_actions = []
    for ep in sorted(ids):
        chunk = ep // info["chunks_size"]
        source = dataset / info["data_path"].format(episode_chunk=chunk, episode_index=ep)
        table = pq.read_table(source, columns=["observation.state", "timestamp", "frame_index"])
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        times = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64).reshape(-1)
        frames = np.asarray(table["frame_index"].to_pylist()).reshape(-1)
        if (
            not np.array_equal(frames, np.arange(len(states)))
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(f"Episode {ep}: non-contiguous frame indices or invalid timestamps")
        # The collector already pairs each row to a source-camera frame. Its
        # nearest-MCAP timestamps have small cumulative drift from encoded CFR
        # time. Preserve that pairing and disclose drift; it does not prove
        # synchronization of the other cameras, which still needs visual QA.
        grid_error = float(np.max(np.abs(times - np.arange(len(times)) / 30)))
        sampled = states[::STRIDE]
        actions = raw_actions(sampled)
        excluded = [segment for segment in selected[f"episode_{ep:06d}"]["segments"] if segment["label"] == "unknown"]
        starts = valid_starts(len(sampled), excluded)
        if len(sampled) < CHUNK + 1:
            raise ValueError(f"Episode {ep} is shorter than one training window")
        if ep in splits["train"]:
            transition_valid = np.array(
                [
                    not any(i / FPS < s["end"] and (i + 1) / FPS >= s["start"] for s in excluded)
                    for i in range(len(actions))
                ]
            )
            train_actions.append(actions[transition_valid])
        path = output / "states" / f"episode_{ep:06d}.npz"
        np.savez_compressed(
            path, states=sampled, source_indices=np.arange(0, len(states), STRIDE), source_timestamps=times[::STRIDE]
        )
        videos = {
            key: info["video_path"].format(episode_chunk=chunk, episode_index=ep, video_key=key) for key in CAMERAS
        }
        video_sizes = {key: (dataset / value).stat().st_size for key, value in videos.items()}
        split = next(name for name, values in splits.items() if ep in values)
        manifest["episodes"].append(
            {
                "episode": ep,
                "split": split,
                "states": str(path.relative_to(output)),
                "states_sha256": file_sha256(path),
                "source_sha256": file_sha256(source),
                "source_rows": len(states),
                "frames": len(sampled),
                "videos": videos,
                "timestamp_grid_max_error_seconds": grid_error,
                "video_sizes": video_sizes,
                "valid_starts": starts,
                "excluded_intervals": excluded,
                "outcome": selected[f"episode_{ep:06d}"]["outcome"],
                "caption": CAPTION,
            }
        )
    # Each unique TRAIN transition contributes once, irrespective of window overlap.
    raw = np.concatenate(train_actions)
    lo, hi = np.quantile(raw, [0.01, 0.99], axis=0)
    center, span = (lo + hi) / 2, np.maximum(hi - lo, 1e-3)
    stats = {
        "contract": CONTRACT,
        "train_episodes": splits["train"],
        "train_transitions": len(raw),
        "minimum_quantile_range": 1e-3,
        "q01": (center - span / 2).tolist(),
        "q99": (center + span / 2).tolist(),
    }
    stats_path = output / "action_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    manifest["stats_sha256"] = file_sha256(stats_path)
    manifest["window_counts"] = {
        name: sum(len(ep["valid_starts"]) for ep in manifest["episodes"] if ep["split"] == name) for name in splits
    }
    manifest["status"] = "prepared"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annotations", default="failuregen_annotations.json")
    parser.add_argument("--splits", type=Path, help="Optional JSON with train/val/test episode ID lists")
    args = parser.parse_args()
    result = prepare(args.dataset, args.output, seed=args.seed, annotations=args.annotations, splits_path=args.splits)
    print(
        json.dumps(
            {"episodes": {k: len(v) for k, v in result["splits"].items()}, "windows": result["window_counts"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
