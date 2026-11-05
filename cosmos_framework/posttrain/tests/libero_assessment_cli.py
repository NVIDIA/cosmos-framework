# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit-model LIBERO evaluation and independent paired result comparison."""

from __future__ import annotations

import csv
import json
import math
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from cosmos_framework.posttrain.checkpoint.io import atomic_write_json
from cosmos_framework.posttrain.control.termination import TerminationWatch
from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
    LIBERO_PROTOCOLS,
    LiberoCase,
    LiberoTrial,
    LiberoTrialManifest,
)
from cosmos_framework.posttrain.tests.helpers.libero_analysis import evaluation_identity
from cosmos_framework.posttrain.utils.entry import resolve_all, resolve_identical
from cosmos_framework.posttrain.utils.identity import json_identity_equal
from cosmos_framework.posttrain.workers.worker_group import CollectiveHandle, WorkerGroup, scatter


def _validate_episode(row: dict, version: int, expected: set[LiberoTrial]) -> LiberoTrial:
    trial = LiberoTrial(LiberoCase(row["task_id"], row["init_state_id"], row["suite"]), row["seed"])
    if row["trial_id"] != trial.id or trial not in expected:
        raise ValueError("evaluation contains an unexpected trial identity")
    if (
        type(row["publication_version"]) is not int
        or row["publication_version"] != version
        or type(row["success"]) is not bool
    ):
        raise ValueError(f"invalid evaluation model version/outcome: {trial.id}")
    protocol = LIBERO_PROTOCOLS[trial.case.suite]
    minimum = 1 if protocol.stop_on_success else protocol.execution_horizon
    if (
        type(row["steps"]) is not int
        or not minimum <= row["steps"] <= protocol.max_episode_steps
        or not row["success"]
        and row["steps"] != protocol.max_episode_steps
        or not protocol.stop_on_success
        and row["steps"] % protocol.execution_horizon
    ):
        raise ValueError(f"invalid evaluation episode length: {trial.id}")
    if len(row["control_rewards"]) != row["steps"] or not all(math.isfinite(v) for v in row["control_rewards"]):
        raise ValueError(f"evaluation raw rewards do not match executed steps: {trial.id}")
    if not math.isfinite(row["return"]) or not math.isclose(sum(row["control_rewards"]), row["return"], abs_tol=1e-8):
        raise ValueError(f"evaluation return differs from raw rewards: {trial.id}")
    return trial


def committed_trials(directory: Path, version: int, inventory: LiberoTrialManifest) -> list[dict]:
    """Read only atomic episode commits and check them against the explicit source inventory."""
    expected = set(inventory.trials)
    rows, seen = [], set()
    for path in sorted((directory / "episodes").rglob("*.json")):
        row = json.loads(path.read_text())
        trial = _validate_episode(row, version, expected)
        if trial in seen:
            raise ValueError(f"duplicate evaluation trial: {path}")
        parent = directory / "episodes" / f"seed-{trial.seed}"
        if path != parent / f"task-{trial.case.task_id:02d}-state-{trial.case.init_state_id:02d}.json":
            raise ValueError(f"evaluation record path differs from its trial identity: {path}")
        seen.add(trial)
        rows.append(row)
    return rows


def assess_pending_trials(
    collector: WorkerGroup,
    simulator: WorkerGroup,
    inventory: LiberoTrialManifest,
    *,
    version: int,
    replicas: int,
    directory: Path,
    watch: TerminationWatch,
    decisions: int,
) -> bool:
    """Dispatch at most one episode per replica and commit each completed result.

    Keep the original inventory's replica assignment when resuming. Drain an
    in-flight round on stop or worker failure; already completed trials remain
    durable and will not be run again by the next process.
    """
    done = {row["trial_id"] for row in committed_trials(directory, version, inventory)}
    for offset in range(0, len(inventory.trials), replicas):
        if watch.should_stop():
            return True
        batch = [None] * replicas
        for replica, trial in enumerate(inventory.trials[offset : offset + replicas]):
            if trial.id not in done:
                batch[replica] = trial
        if not any(batch):
            continue
        from dataclasses import asdict

        from cosmos_framework.posttrain.control.loop.decision_window import evaluate_decision_cases

        handle = evaluate_decision_cases(
            collector,
            simulator,
            version=version,
            cases=scatter([[asdict(trial.case)] if trial is not None else [] for trial in batch]),
            seed=scatter([trial.seed if trial is not None else 0 for trial in batch]),
            decisions=decisions,
        )
        outcomes = handle.as_completed() if isinstance(handle, CollectiveHandle) else enumerate(resolve_all(handle))
        failures = []
        for _, result in outcomes:
            if isinstance(result, Exception):
                failures.append(result)
                continue
            if not result["episodes"]:
                continue
            (episode,) = result["episodes"]
            trial = LiberoTrial(
                LiberoCase(episode["task_id"], episode["init_state_id"], inventory.suite), result["evaluation_seed"]
            )
            result = {**episode, "trial_id": trial.id, "seed": trial.seed, "publication_version": version}
            trial = _validate_episode(result, version, set(inventory.trials))
            if trial not in batch or result["trial_id"] != trial.id or trial.id in done:
                raise ValueError("assessment returned an unexpected or duplicate trial")
            parent = directory / "episodes" / f"seed-{trial.seed}"
            parent.mkdir(parents=True, exist_ok=True)
            path = parent / f"task-{trial.case.task_id:02d}-state-{trial.case.init_state_id:02d}.json"
            atomic_write_json(str(path), result)
            done.add(trial.id)
        if failures:
            raise failures[0]
    return False


def assessment_metrics(rows: list[dict], inventory: LiberoTrialManifest) -> dict[str, int | float]:
    expected = {trial.id for trial in inventory.trials}
    if {row["trial_id"] for row in rows} != expected or len(rows) != len(expected):
        raise ValueError(f"evaluation requires every one of its {len(expected)} unique trials")
    tasks = sorted({trial.case.task_id for trial in inventory.trials})
    metrics: dict[str, int | float] = {"val/episode_count": len(rows), "val/task_count": len(tasks)}
    rates = []
    for task in tasks:
        selected = [row for row in rows if row["task_id"] == task]
        successes = sum(row["success"] for row in selected)
        metrics[f"val/task_{task}/success_count"] = successes
        metrics[f"val/task_{task}/success_rate"] = successes / len(selected)
        rates.append(successes / len(selected))
    metrics["val/success_rate"] = sum(rates) / len(rates)
    metrics["val/raw_return_mean"] = sum(row["return"] for row in rows) / len(rows)
    metrics["val/episode_steps"] = sum(row["steps"] for row in rows) / len(rows)
    return metrics


def assessment_metric_frame(
    rows: list[dict], version: int, *, complete: bool, inventory: LiberoTrialManifest
) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {
        "val/assessment_completed_trials": len(rows),
        "rollout/publication_version": version,
    }
    if complete:
        metrics.update(assessment_metrics(rows, inventory))
    return metrics


def compare_assessments(base_path: Path, candidate_path: Path, output: Path) -> dict:
    """Compare explicit evaluation results; resample state clusters within each task."""
    import numpy as np

    results = []
    for path in (base_path, candidate_path):
        data = json.loads((path / "evaluation.json" if path.is_dir() else path).read_text())
        inventory = LiberoTrialManifest.from_payload(data["trial_manifest"])
        if data["evaluation"]["trials"] != inventory.digest:
            raise ValueError("evaluation result does not match its source trial manifest")
        for row in data["episodes"]:
            _validate_episode(row, data["publication_version"], set(inventory.trials))
        assessment_metrics(data["episodes"], inventory)
        results.append(data)
    base, candidate = results
    if not json_identity_equal(base["evaluation"], candidate["evaluation"], context="evaluation protocol"):
        raise ValueError("paired comparison requires identical evaluation protocols and trial inventories")
    before = {row["trial_id"]: row for row in base["episodes"]}
    clusters: dict[int, dict[int, list[int]]] = {}
    pairs, gained, lost = [], [], []
    for row in candidate["episodes"]:
        prior = before[row["trial_id"]]
        delta = int(row["success"]) - int(prior["success"])
        clusters.setdefault(row["task_id"], {}).setdefault(row["init_state_id"], []).append(delta)
        pairs.append(
            {
                "trial_id": row["trial_id"],
                "baseline_success": prior["success"],
                "candidate_success": row["success"],
                "delta": delta,
            }
        )
        if delta > 0:
            gained.append(row["trial_id"])
        elif delta < 0:
            lost.append(row["trial_id"])
    rng = np.random.default_rng(1234)
    estimates = np.zeros(10000)
    for task in sorted(clusters):
        values = [clusters[task][state] for state in sorted(clusters[task])]
        sums, counts = np.array([sum(v) for v in values]), np.array([len(v) for v in values])
        sampled = rng.integers(len(values), size=(len(estimates), len(values)))
        estimates += sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1) / len(clusters)
    baseline_metrics = assessment_metrics(base["episodes"], inventory)
    candidate_metrics = assessment_metrics(candidate["episodes"], inventory)
    report = {
        "baseline": {
            "path": str(base_path),
            "optimizer_step": base["optimizer_step"],
            "publication_version": base["publication_version"],
            "metrics": baseline_metrics,
        },
        "candidate": {
            "path": str(candidate_path),
            "optimizer_step": candidate["optimizer_step"],
            "publication_version": candidate["publication_version"],
            "metrics": candidate_metrics,
        },
        "evaluation": base["evaluation"],
        "trials_per_model": len(inventory.trials),
        "success_rate_delta": candidate_metrics["val/success_rate"] - baseline_metrics["val/success_rate"],
        "paired_ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "paired_gains": gained,
        "paired_losses": lost,
        "method": "10000 task-stratified initial-state cluster bootstraps; trial seeds remain paired; seed 1234",
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(str(output / "comparison.json"), report)
    with (output / "paired-trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    with (output / "per-task.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task_id", "trials", "baseline_successes", "candidate_successes"])
        for task in sorted(clusters):
            writer.writerow(
                [
                    task,
                    sum(len(v) for v in clusters[task].values()),
                    baseline_metrics[f"val/task_{task}/success_count"],
                    candidate_metrics[f"val/task_{task}/success_count"],
                ]
            )
    return report


def run_assessment(config: Any, *, manifest: str, output_dir: str) -> dict:
    import ray

    from cosmos_framework.posttrain.config.config_joint_ppo import resolve_joint_config

    config = resolve_joint_config(config)

    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.checkpoint.io import atomic_write_json
    from cosmos_framework.posttrain.config.config_joint_ppo import validate_joint_ppo_config
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync
    from cosmos_framework.posttrain.entrypoints.joint_ppo import assemble
    from cosmos_framework.posttrain.model.gr00t.build import libero10_model_identity
    from cosmos_framework.posttrain.utils.entry import dump_resolved
    from cosmos_framework.posttrain.utils.mesh import detect_mesh
    from cosmos_framework.posttrain.utils.report import (
        emit_metric_frame,
        finish_best_effort,
        make_reporters,
        start_best_effort,
    )

    nodes, gpus = detect_mesh()
    validate_joint_ppo_config(config, nodes * gpus)
    recipe = instantiate(config.rl)
    inventory = LiberoTrialManifest.read(manifest)
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    resume = None if recipe.resume in ("", "off") else recipe.resume
    with ExitStack() as cleanup:
        cleanup.callback(ray.shutdown)
        units = assemble(config, recipe, directory, resume, nodes * gpus, evaluation=True)
        collector, rollout, policy = units["collector"], units["rollout"], units["policy"]
        cleanup.callback(lambda: resolve_all(rollout.close()))
        cleanup.callback(lambda: resolve_all(collector.close()))
        cleanup.callback(lambda: resolve_all(units["sim"].close()))
        cleanup.callback(lambda: resolve_all(units["eval_sim"].close()))
        cleanup.callback(CheckpointLifecycle.finalize_units, {"policy": policy})
        reporters, failed_reporters = [], set()
        cleanup.callback(finish_best_effort, reporters, failed_reporters=failed_reporters)
        progress = resolve_identical(policy.progress(), operation="assessment model counters")
        step, version = progress["optimizer_step"], progress["publication_version"]
        weight_sync(rollout, policy, validation_step=version)
        before = resolve_all(policy.evaluation_state())
        identity = {
            "version": 2,
            "model_source": libero10_model_identity(config.model.checkpoint_path)
            if config.environment.config.suite == "libero_10"
            else {"checkpoint": config.model.checkpoint_path},
            "evaluation": evaluation_identity(config, inventory),
            "recipe": recipe.identity(),
            "optimizer_step": step,
            "publication_version": version,
            "model_shards": [row["parameters"] for row in before],
            "trials": inventory.digest,
        }
        identity_path = directory / "identity.json"
        if identity_path.exists():
            if not json_identity_equal(
                json.loads(identity_path.read_text()), identity, context="assessment resume identity"
            ):
                raise ValueError("assessment resume identity differs from its immutable model/protocol/trials")
        else:
            atomic_write_json(str(identity_path), identity)
        atomic_write_json(str(directory / "trials.json"), inventory.payload())
        rows = committed_trials(directory, version, inventory)
        if len(rows) == len(inventory.trials) and (directory / "evaluation.json").exists():
            return json.loads((directory / "summary.json").read_text())
        dump_resolved(config, str(directory))
        reporters.extend(
            make_reporters(
                config,
                str(directory),
            )
        )
        start_best_effort(reporters, failed_reporters=failed_reporters)
        started, stopped = time.monotonic(), False
        watch = TerminationWatch()
        stopped = assess_pending_trials(
            collector,
            units["eval_sim"],
            inventory,
            version=version,
            replicas=nodes * gpus,
            directory=directory,
            watch=watch,
            decisions=instantiate(config.environment).evaluation_decisions,
        )
        if resolve_all(policy.evaluation_state()) != before:
            raise RuntimeError("immutable-model assessment changed learner state")
        rows = committed_trials(directory, version, inventory)
        with (directory / "episodes.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        result = {
            "complete": len(rows) == len(inventory.trials),
            "completed_trials": len(rows),
            "optimizer_step": step,
            "publication_version": version,
            "seconds": time.monotonic() - started,
            "training_state_unchanged": True,
            "stopped": stopped,
        }
        result["metrics"] = assessment_metric_frame(rows, version, complete=result["complete"], inventory=inventory)
        atomic_write_json(str(directory / "summary.json"), result)
        if result["complete"]:
            atomic_write_json(
                str(directory / "evaluation.json"),
                {
                    **result,
                    "evaluation": identity["evaluation"],
                    "trial_manifest": inventory.payload(),
                    "episodes": rows,
                },
            )
            # One immutable model has one final metric frame. Partial jobs commit
            # trial progress only, so resumed online runs never reuse a report step.
            emit_metric_frame(reporters, result["metrics"], step, failed_reporters=failed_reporters)
        return result


def main() -> None:
    import argparse

    from cosmos_framework.utils.config import load_config
    from cosmos_framework.posttrain.config.config_joint_ppo import validate_joint_ppo_config
    from cosmos_framework.posttrain.utils.mesh import detect_mesh

    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--config")
    operation.add_argument("--compare", type=Path, nargs=2, metavar=("BASELINE", "CANDIDATE"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest")
    parser.add_argument("--validate-manifest", action="store_true")
    args, options = parser.parse_known_args()
    if args.compare:
        if args.output is None or options or args.validate_manifest:
            parser.error("comparison requires --output and accepts no training overrides")
        if not args.output.resolve().is_relative_to((Path(__file__).resolve().parents[1] / "outputs").resolve()):
            parser.error("comparison output must be under posttrain/outputs")
        result = compare_assessments(*args.compare, args.output)
    else:
        if args.output is None or not args.manifest:
            parser.error("evaluation requires --output and --manifest")
        config = load_config(args.config, ["--", *[item for item in options if item != "--"]])
        if args.validate_manifest:
            nodes, gpus = detect_mesh()
            validate_joint_ppo_config(config, nodes * gpus)
            print("GR00T evaluation configuration and explicit trial manifest are valid", flush=True)
            return
        result = run_assessment(config, manifest=args.manifest, output_dir=str(args.output))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
