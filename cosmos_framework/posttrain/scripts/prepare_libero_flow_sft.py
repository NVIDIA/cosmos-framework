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
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from cosmos_framework.posttrain.utils.object_store import file_sha256


def iter_episodes(paths):
    """Bound image memory to one episode plus one 64-frame Arrow batch.

    LeRobot v3 shards can split an episode between files. Validate source order
    explicitly; repeated/non-monotonic episodes must be repaired upstream, never
    silently concatenated into a trajectory.
    """
    current, rows = None, []
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                episode = int(row["episode_index"])
                if current is not None and episode != current:
                    if episode <= current:
                        raise ValueError("LIBERO source episodes must be strictly ordered across parquet shards")
                    yield current, pd.DataFrame(rows)
                    rows = []
                current = episode
                rows.append(row)
    if current is not None:
        yield current, pd.DataFrame(rows)


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
    selected = set(training_episodes + validation_episodes)
    if len(selected) != len(training_episodes) + len(validation_episodes):
        raise ValueError("LIBERO episode lists must not contain duplicates")
    tasks = pd.read_parquet(root / "meta/tasks.parquet").reset_index()
    text_keys = [k for k in tasks.columns if k != "task_index" and isinstance(tasks[k].iloc[0], str)]
    if len(text_keys) != 1 or "task_index" not in tasks.columns:
        raise ValueError(f"unsupported LeRobot task table schema: {list(tasks.columns)}")
    prompts = dict(zip(tasks["task_index"], tasks[text_keys[0]], strict=True))
    destination.mkdir(parents=True)
    (destination / "media").mkdir()
    counts = {"train": 0, "validation": 0}
    training = set(training_episodes)
    seen, source_rows = set(), 0
    with ExitStack() as stack:
        streams = {split: stack.enter_context((destination / (split + ".jsonl")).open("w")) for split in counts}
        for episode, frames in iter_episodes(paths):
            source_rows += len(frames)
            if np.any(np.diff(frames["frame_index"]) != 1):
                raise ValueError(f"episode {episode} is non-contiguous")
            if episode not in selected:
                continue
            if len(frames) < horizon:
                raise ValueError(f"episode {episode} is shorter than the requested horizon")
            seen.add(episode)
            split = "train" if episode in training else "validation"
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
                row = {
                    "id": f"libero-{episode}-{int(frame['frame_index'])}",
                    "episode_id": f"libero-{source_revision}-episode{episode}",
                    "observation": {"state": state.tolist(), "prompt": prompts[int(frame["task_index"])]},
                    "actions": actions.tolist(),
                    "media": media,
                }
                streams[split].write(json.dumps(row, allow_nan=False) + "\n")
                counts[split] += 1
    if selected != seen:
        raise ValueError(f"requested LIBERO episodes are not staged: {sorted(selected - seen)}")
    if "total_frames" in info and source_rows != info["total_frames"]:
        raise ValueError(f"LIBERO staged frames {source_rows} differ from source manifest {info['total_frames']}")
    manifest = {
        "format": "cosmos-libero-flow-preparation-v2",
        "ordering": "source-episode-frame-order",
        "source_frames": source_rows,
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
