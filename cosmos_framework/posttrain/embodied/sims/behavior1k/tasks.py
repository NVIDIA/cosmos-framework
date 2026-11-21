# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Build evaluation work units from native task metadata."""

from __future__ import annotations

import json
import math
import os
import re

from cosmos_framework.posttrain.embodied.unit import EpisodeKey, WorkUnit


def load_mean_demo_steps(stats_path: str | None = None) -> dict[str, float]:
    """Mean human-demo length (steps) per task."""
    data_path = os.environ.get("OMNIGIBSON_DATA_PATH", "/data/og_data")
    path = stats_path or os.environ.get(
        "B1K_TASK_STATS_JSONL",
        os.path.join(data_path, "2026-challenge-task-instances", "metadata", "task.jsonl"),
    )
    out: dict[str, float] = {}
    try:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "task_name" in rec and "length" in rec:
                    out[str(rec["task_name"])] = float(rec["length"])
    except OSError as ex:
        raise ValueError(f"cannot resolve BEHAVIOR-1K step budgets from {path}: {ex}") from ex
    return out


def build_work_units(
    tasks_env: str,
    instances: int,
    num_envs: int,
    timeout_mult: float | None,
    max_steps: int | None,
    instance_ids: list[int] | None = None,
    *,
    seed: int = 0,
    mode: str = "public_test",
    action_horizon: int = 0,
    stats_path: str | None = None,
) -> list[WorkUnit]:
    """Freeze native task budgets and dispatch indices into standard work units."""
    if tasks_env.strip().lower() in ("", "all"):
        raise ValueError(
            "sim.tasks='all' is not supported for suite=behavior1k. The challenge table "
            "(B100_task_misc.csv) holds all 100 tasks and the sim package defines no "
            "canonical 20-task subset. List the tasks explicitly in the run YAML."
        )
    names = [task.strip() for task in tasks_env.split(",") if task.strip()]
    if any(re.search(r"^[._]|[/\\\x00]", name) for name in names):
        raise ValueError("task names must be safe path components")
    if len(names) != len(set(names)) or not names:
        raise ValueError("BEHAVIOR-1K requires distinct tasks")

    ids = list(range(instances)) if instance_ids is None else list(instance_ids)
    if not ids or len(ids) != len(set(ids)) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("BEHAVIOR-1K requires distinct nonnegative instance indices")
    if num_envs < 1 or action_horizon < 0:
        raise ValueError("num_envs must be positive and action_horizon nonnegative")
    lengths = load_mean_demo_steps(stats_path) if max_steps is None else {}
    mult = 1.5 if timeout_mult is None else float(timeout_mult)
    if not math.isfinite(mult) or mult <= 0:
        raise ValueError("timeout_mult must be finite and positive")
    units = []
    for name in names:
        if max_steps is None and (name not in lengths or not math.isfinite(lengths[name]) or lengths[name] <= 0):
            raise ValueError(f"no valid native demo length for {name}; provide the stats file or explicit max_steps")
        steps = int(lengths[name] * mult) if max_steps is None else max_steps
        if type(steps) is not int or steps < 1:
            raise ValueError("max_steps must be a positive integer")
        for offset in range(0, len(ids), num_envs):
            selected = ids[offset : offset + num_envs]
            units.append(
                WorkUnit.make(
                    suite="behavior1k",
                    scenario={
                        "task": name,
                        "num_envs": num_envs,
                        "max_steps": steps,
                        "mode": mode,
                        "seed": seed,
                        "action_horizon": action_horizon,
                    },
                    episodes=[EpisodeKey("behavior1k", name, instance, seed) for instance in selected],
                    est_minutes=steps * len(selected) * 0.06 / 60,
                )
            )
    return units
