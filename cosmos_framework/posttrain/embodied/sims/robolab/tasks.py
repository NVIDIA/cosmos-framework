# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Lightweight RoboLab task inventory; no Isaac imports."""

from __future__ import annotations


def build_work_units(
    *, tasks: list[str], metadata: str, num_envs: int, seeds: list[int], control_freq: int, action_horizon: int
) -> list:
    """Resolve official task budgets without importing a simulator or substituting estimates.

    Each task/seed pair becomes one work unit with ``num_envs`` parallel episodes.
    ``metadata`` supplies task durations in seconds; ``control_freq`` converts these
    to control-step budgets. ``action_horizon`` is the chunk prefix executed between
    policy queries, not the episode limit. ``tasks=["all"]`` selects the full inventory.
    """
    import json
    import math

    from cosmos_framework.posttrain.embodied.unit import EpisodeKey, WorkUnit
    from cosmos_framework.posttrain.schema.identity import digest

    with open(metadata, encoding="utf-8") as file:
        rows = json.load(file)
    inventory = {row["task_name"]: row for row in rows}
    if len(inventory) != len(rows):
        raise ValueError("RoboLab task metadata contains duplicate task names")
    selected = list(inventory) if tasks == ["all"] else tasks
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(inventory):
        raise ValueError("RoboLab tasks must be distinct names present in native metadata")
    if num_envs < 1 or control_freq != 15 or action_horizon < 1:
        raise ValueError("RoboLab requires positive lanes/horizon and supported control frequency")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("RoboLab seeds must be distinct nonnegative integers")
    units = []
    for task in selected:
        row = inventory[task]
        seconds = float(row["episode_s"])
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"RoboLab task {task!r} has an invalid native episode duration")
        for seed in seeds:
            units.append(
                WorkUnit.make(
                    suite="robolab",
                    scenario={
                        "task": task,
                        "max_steps": math.ceil(seconds * control_freq),
                        "num_envs": num_envs,
                        "native_batch_size": num_envs,
                        "seed": seed,
                        "action_horizon": action_horizon,
                        "native_task_sha": digest(row),
                    },
                    episodes=[EpisodeKey("robolab", task, lane, seed) for lane in range(num_envs)],
                    est_minutes=seconds * num_envs / 60,
                )
            )
    return units


def validate_task_metadata(*, task: str, metadata: str, expected_sha: str) -> None:
    """Refuse an image whose native task definition differs from the metadata used to prepare evaluation."""
    import json

    from cosmos_framework.posttrain.embodied.sim import SimProtocolError
    from cosmos_framework.posttrain.schema.identity import digest

    with open(metadata, encoding="utf-8") as file:
        rows = [row for row in json.load(file) if row["task_name"] == task]
    if len(rows) != 1 or digest(rows[0]) != expected_sha:
        raise SimProtocolError(
            f"RoboLab native metadata for {task!r} differs from the metadata used to prepare evaluation"
        )
