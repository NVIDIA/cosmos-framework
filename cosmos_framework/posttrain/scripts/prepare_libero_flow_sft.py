# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prepare explicit LIBERO episode splits for native supervised action training.

Consumes staged LeRobot v3 parquet with embedded RGB images, eight state values
and seven physical actions. Preserves image orientation and action convention;
the selected model packer owns resizing, tokenization and normalization.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cosmos_framework.posttrain.utils.object_store import file_sha256


def prepare(
    source: str,
    out: str,
    *,
    training_episodes: list[int],
    validation_episodes: list[int],
    stride: int = 10,
    horizon: int = 10,
    source_revision: str,
):
    root, destination = Path(source), Path(out)
    if destination.exists():
        raise FileExistsError(f"prepared demonstration directory already exists: {destination}")
    if not training_episodes or not validation_episodes or set(training_episodes) & set(validation_episodes):
        raise ValueError("training/validation episode lists must be nonempty and disjoint")
    if stride < 1 or horizon < 1:
        raise ValueError("stride and horizon must be positive")
    info = json.loads((root / "meta/info.json").read_text())
    if info["features"]["observation.state"]["shape"] != [8] or info["features"]["action"]["shape"] != [7]:
        raise ValueError("LIBERO conversion requires state width 8 and action width 7")
    paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise ValueError("no staged LIBERO parquet data files")
    data = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    selected = set(training_episodes + validation_episodes)
    if not selected.issubset(set(data["episode_index"].unique())):
        raise ValueError(
            f"requested LIBERO episodes are not staged: {sorted(selected - set(data['episode_index'].unique()))}"
        )
    tasks = pd.read_parquet(root / "meta/tasks.parquet").reset_index()
    text_keys = [k for k in tasks.columns if k != "task_index" and isinstance(tasks[k].iloc[0], str)]
    if len(text_keys) != 1 or "task_index" not in tasks.columns:
        raise ValueError(f"unsupported LeRobot task table schema: {list(tasks.columns)}")
    prompts = dict(zip(tasks["task_index"], tasks[text_keys[0]], strict=True))
    destination.mkdir(parents=True)
    (destination / "media").mkdir()
    counts = {}
    for split, episodes in (("train", training_episodes), ("validation", validation_episodes)):
        rows = []
        for episode in episodes:
            frames = data[data["episode_index"] == episode].sort_values("frame_index").reset_index(drop=True)
            if len(frames) < horizon or np.any(np.diff(frames["frame_index"]) != 1):
                raise ValueError(f"episode {episode} is short or non-contiguous")
            for start in range(0, len(frames) - horizon + 1, stride):
                frame = frames.iloc[start]
                state = np.asarray(frame["observation.state"], dtype=np.float64)
                actions = np.stack(frames.iloc[start : start + horizon]["action"].to_list()).astype(np.float64)
                if (
                    state.shape != (8,)
                    or actions.shape != (horizon, 7)
                    or not np.isfinite(state).all()
                    or not np.isfinite(actions).all()
                ):
                    raise ValueError(f"invalid demonstration state/actions at episode={episode}, frame={start}")
                media = {}
                for column, camera in (
                    ("observation.images.image", "image"),
                    ("observation.images.image2", "wrist_image"),
                ):
                    payload = frame[column]
                    if not isinstance(payload, dict) or not payload.get("bytes"):
                        raise ValueError(
                            f"{column} must contain embedded encoded RGB bytes; external video conversion is separate"
                        )
                    name = f"media/episode{episode}-frame{int(frame['frame_index'])}-{camera}.png"
                    with Image.open(io.BytesIO(payload["bytes"])) as image:
                        image.convert("RGB").save(destination / name)
                    media[camera] = name
                rows.append(
                    {
                        "id": f"libero-{episode}-{int(frame['frame_index'])}",
                        "episode_id": f"libero-{source_revision}-episode{episode}",
                        "observation": {"state": state.tolist(), "prompt": prompts[int(frame["task_index"])]},
                        "actions": actions.tolist(),
                        "media": media,
                    }
                )
        (destination / (split + ".jsonl")).write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        counts[split] = len(rows)
    manifest = {
        "format": "cosmos-libero-flow-preparation-v1",
        "source_revision": source_revision,
        "source_files": {
            str(p.relative_to(root)): file_sha256(str(p))
            for p in [*paths, root / "meta/info.json", root / "meta/tasks.parquet"]
        },
        "training_episodes": training_episodes,
        "validation_episodes": validation_episodes,
        "stride": stride,
        "horizon": horizon,
        "source_fps": info["fps"],
        "transform": "embedded-RGB-unchanged-state-actions-unchanged",
        "counts": counts,
    }
    (destination / "preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "out", "source-revision", "training-episodes", "validation-episodes"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()
    result = prepare(
        args.source,
        args.out,
        source_revision=args.source_revision,
        training_episodes=[int(x) for x in args.training_episodes.split(",")],
        validation_episodes=[int(x) for x in args.validation_episodes.split(",")],
        stride=args.stride,
        horizon=args.horizon,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
