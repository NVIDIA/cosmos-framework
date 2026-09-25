# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compute Cosmos-compatible action normalization statistics from LeRobot data."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np


def _as_rows(value: object, *, source: str) -> np.ndarray:
    """Convert one LeRobot action value to a finite ``[N, D]`` float array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]

    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] == 0:
        raise ValueError(f"{source} must contain a non-empty vector or [N, D] array")
    if not np.isfinite(rows).all():
        raise ValueError(f"{source} contains NaN or infinite values")
    return rows


def compute_stats(
    values: Iterable[object],
    *,
    reservoir_size: int = 50_000,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Compute the six fields consumed by Cosmos action normalizers.

    Values may be individual ``[D]`` vectors or batches of ``[N, D]`` vectors.
    Quantiles are estimated from a deterministic uniform reservoir so the tool
    does not need to keep an entire dataset in memory.
    """
    if reservoir_size <= 0:
        raise ValueError("reservoir_size must be positive")

    rng = random.Random(seed)
    reservoir: list[np.ndarray] = []
    count = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None

    for value_index, value in enumerate(values):
        rows = _as_rows(value, source=f"action value {value_index}")
        if mean is None:
            mean = np.zeros(rows.shape[1], dtype=np.float64)
            m2 = np.zeros(rows.shape[1], dtype=np.float64)
            minimum = rows[0].copy()
            maximum = rows[0].copy()
        elif rows.shape[1] != mean.shape[0]:
            raise ValueError(
                f"action value {value_index} has dimension {rows.shape[1]}, expected {mean.shape[0]}"
            )

        assert m2 is not None and minimum is not None and maximum is not None
        minimum = np.minimum(minimum, rows.min(axis=0))
        maximum = np.maximum(maximum, rows.max(axis=0))
        for row in rows:
            count += 1
            delta = row - mean
            mean += delta / count
            m2 += delta * (row - mean)

            if len(reservoir) < reservoir_size:
                reservoir.append(row.copy())
            else:
                replacement = rng.randrange(count)
                if replacement < reservoir_size:
                    reservoir[replacement] = row.copy()

    if count == 0 or mean is None or m2 is None or minimum is None or maximum is None:
        raise ValueError("no action values were found")

    quantile_source = np.stack(reservoir)
    quantiles = np.quantile(quantile_source, [0.01, 0.99], axis=0)
    return {
        "q01": quantiles[0].tolist(),
        "q99": quantiles[1].tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(m2 / count).tolist(),
        "min": minimum.tolist(),
        "max": maximum.tolist(),
    }


def iter_dataset_actions(roots: Iterable[Path], action_key: str) -> Iterator[object]:
    """Yield action values from one or more local LeRobot dataset roots."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {root}")
        dataset = LeRobotDataset(
            repo_id="local",
            root=str(root),
            revision="local",
            delta_timestamps={action_key: [0.0]},
            download_videos=False,
        )
        for index in range(len(dataset)):
            sample = dataset[index]
            if action_key not in sample:
                raise KeyError(
                    f"action key {action_key!r} is missing from sample {index} in {root}"
                )
            yield sample[action_key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        nargs="+",
        required=True,
        help="One or more local LeRobot dataset roots.",
    )
    parser.add_argument(
        "--action-key", default="action", help="LeRobot feature key containing actions."
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--reservoir-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = compute_stats(
        iter_dataset_actions(args.dataset_root, args.action_key),
        reservoir_size=args.reservoir_size,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(stats['mean'])}-D action stats for {args.output}")


if __name__ == "__main__":
    main()
