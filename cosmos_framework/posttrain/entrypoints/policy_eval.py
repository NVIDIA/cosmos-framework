# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
r"""Run closed-loop work units locally and report per-episode results.

In a RoboLab simulator environment, with the DROID policy server running:

    python -m cosmos_framework.posttrain.entrypoints.policy_eval robolab \
        --task BananaInBowlTask --metadata /path/to/robolab/tasks/_metadata/task_metadata.json \
        --droid-server localhost --seed 0 --output runs/droid-smoke

"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path

import numpy as np

from cosmos_framework.posttrain.embodied.driver import run_unit
from cosmos_framework.posttrain.embodied.registry import build_sim
from cosmos_framework.posttrain.embodied.sinks.null import NullSink
from cosmos_framework.posttrain.embodied.unit import TERMINAL_DATA, Outcome


def _metrics(value):
    if isinstance(value, dict):
        return {key: _metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metrics(item) for item in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return _metrics(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate(units, sim, policy, embodiment, output):
    """Count only terminal episodes with the requested identity. A partial run is an error."""
    expected = {key for unit in units for key in unit.episodes}
    if not expected or len(expected) != sum(len(unit.episodes) for unit in units):
        raise ValueError("evaluation requires distinct, nonempty episodes")
    info = policy.describe()
    if info.embodiment != embodiment.name or info.action_dim != embodiment.action_dim:
        raise ValueError("policy and simulator must agree on embodiment and action width")
    completed = set()
    by_task = {}
    success = 0
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for unit in units:
            result = run_unit(unit, sim, policy, NullSink(), embodiment, force_render=False)
            for ep in result.episodes:
                if ep.key not in unit.episodes or ep.key in completed or ep.outcome not in TERMINAL_DATA:
                    raise RuntimeError(f"unexpected episode result: {ep.key.slug()} ({ep.outcome.value})")
                record = dict(
                    episode=ep.key.slug(),
                    task=ep.key.task,
                    seed=ep.key.seed,
                    instance=ep.key.instance,
                    steps=ep.steps,
                    outcome=ep.outcome.value,
                    metrics=_metrics(ep.metrics),
                )
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                completed.add(ep.key)
                success += ep.outcome == Outcome.SUCCESS
                task = by_task.setdefault(ep.key.task, {"completed": 0, "success": 0})
                task["completed"] += 1
                task["success"] += ep.outcome == Outcome.SUCCESS
            if result.error:
                raise RuntimeError(result.error)
    if expected != completed:
        raise RuntimeError(f"completed {len(completed)} of {len(expected)} requested episodes")
    summary = dict(
        planned=len(expected), completed=len(completed), success=success, success_rate=success / len(completed)
    )
    for row in by_task.values():
        row["success_rate"] = row["success"] / row["completed"]
    summary["tasks"] = by_task
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suite", choices=("robolab",))
    parser.add_argument("--task", nargs="+", required=True)
    parser.add_argument("--metadata", type=Path, required=True, help="RoboLab task_metadata.json")
    parser.add_argument("--output", type=Path, required=True, help="new directory for this run")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, help="maximum number of actions to execute per policy query")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--policy-factory", help="module:callable returning a configured Rollout")
    group.add_argument("--droid-server", help="host running the public Cosmos DROID WebSocket policy server")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.num_envs < 1 or args.seed < 0 or (args.action_horizon is not None and args.action_horizon < 1):
        parser.error("episode counts and action horizon must be positive, and seed must be nonnegative")
    from cosmos_framework.posttrain.embodied.sims.robolab.interface import get_embodiment_map
    from cosmos_framework.posttrain.embodied.sims.robolab.tasks import build_work_units

    units = build_work_units(
        tasks=args.task,
        metadata=str(args.metadata),
        num_envs=args.num_envs,
        seeds=[args.seed],
        control_freq=15,
        action_horizon=args.action_horizon or 32,
    )
    emb = get_embodiment_map("droid")
    sim_args = dict(n_lanes=args.num_envs, output_dir=str(args.output / "native"))
    if args.policy_factory:
        module, sep, name = args.policy_factory.partition(":")
        if not sep or not module or not name:
            parser.error("--policy-factory must be module:callable")
        policy = getattr(importlib.import_module(module), name)()
    else:
        from cosmos_framework.posttrain.embodied.rollouts.droid import DroidEndpoint

        policy = DroidEndpoint(args.droid_server, args.port)
    sim = build_sim(args.suite, embodiment=emb.name, **sim_args)
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        print(json.dumps(evaluate(units, sim, policy, emb, args.output)))
    finally:
        try:
            sim.close()
        finally:
            if callable(close := getattr(policy, "close", None)):
                close()


if __name__ == "__main__":
    main()
