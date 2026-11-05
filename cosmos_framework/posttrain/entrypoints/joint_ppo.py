# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synchronous joint actor/value PPO over the shared posttrain roles."""

from __future__ import annotations

import functools
import json
import os
import random
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from cosmos_framework.posttrain.config.config_joint_ppo import PosttrainJointPPOConfig, resolve_joint_config
from cosmos_framework.posttrain.config.rl import JointPPOConfig

ROLE_SPECS = (
    ("sim", "sim"),
    ("collector", "env"),
    ("rollout", "rollout"),
    ("rm", "reward"),
    ("replay", "buffer"),
    ("policy", "trainer"),
)


def validation_identity(config: PosttrainJointPPOConfig) -> dict:
    from dataclasses import asdict

    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.utils.identity import canonical_config_value

    config = resolve_joint_config(config)
    recipe = instantiate(config.rl)
    return {
        "enabled": True,
        "environment": instantiate(config.environment).identity(),
        "sampling": canonical_config_value(config.rollout),
        "seed": recipe.validation.seed,
        "schedule": canonical_config_value(asdict(recipe.validation)),
    }


def setup_policy(config: PosttrainJointPPOConfig, output: str, resume: str | None, recipe: JointPPOConfig) -> dict:
    import numpy as np
    import torch

    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer
    from cosmos_framework.posttrain.workers.ray import current_pdims

    config = resolve_joint_config(config)
    random.seed(recipe.seed)
    np.random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    model = instantiate(config.policy, output=output, parallel_dims=current_pdims())
    lifecycle = CheckpointLifecycle.from_config(config, recipe, run_out_dir=output, resume_checkpoint=resume)
    trainer = JointPPOTrainer(
        model,
        recipe,
        parallel_dims=current_pdims(),
        checkpoint_manager=lifecycle.manager,
        source_identity=instantiate(config.environment).identity(),
    )
    lifecycle.load_trainer(trainer)
    if recipe.validation.enabled:
        trainer.configure_validation(validation_identity(config))
    return {"policy": trainer}


def setup_experience(
    config: PosttrainJointPPOConfig,
    output: str,
    resume: str | None,
    recipe: JointPPOConfig,
    width: int,
    members: tuple[str, ...] = ("sim", "collector", "rollout", "rm", "replay"),
) -> dict:
    """Build independently owned bodies; ActorGroup membership controls their placement."""
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.config.build import REPLICA_INDEX_ENV
    from cosmos_framework.posttrain.control.buffer import ExperienceBuffer
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.roles.env.ppo_window import PPOWindowCollector, build_window_gateway
    from cosmos_framework.posttrain.roles.env.sim_role import require_sim_capability
    from cosmos_framework.posttrain.roles.reward.worker import RewardWorker

    config = resolve_joint_config(config)
    replica = int(os.environ[REPLICA_INDEX_ENV])
    environment = instantiate(config.environment)
    bodies = {}
    with ExitStack() as cleanup:
        if "rollout" in members:
            rollout = instantiate(config.rollout, output=output, replica=replica)
            cleanup.callback(rollout.close)
            bodies["rollout"] = rollout
        for name in ("sim", "eval_sim"):
            if name in members:
                gateway = build_window_gateway(
                    environment,
                    replica=replica,
                    n_lanes=recipe.num_envs // width,
                    seed=recipe.seed,
                    evaluation=name == "eval_sim",
                )
                cleanup.callback(gateway.close)
                require_sim_capability(gateway, "fixed_case_evaluation" if name == "eval_sim" else "decision_chunks")
                bodies[name] = gateway
        if "collector" in members:
            collector = PPOWindowCollector(recipe, replica, recipe.num_envs // width, environment=environment)
            cleanup.callback(collector.close)
            if resume:
                lifecycle = CheckpointLifecycle.from_config(
                    config, recipe, run_out_dir=output, resume_checkpoint=resume
                )
                state = lifecycle.manager.load_component(resume, "data")
                if state is None:
                    raise ValueError("joint PPO checkpoint is missing collection progress")
                collector.load_state_dict(state["collectors"][str(replica)])
            bodies["collector"] = collector
        if "rm" in members:
            bodies["rm"] = RewardWorker([instantiate(config.reward)], training_group_size=1)
        if "replay" in members:
            bodies["replay"] = ExperienceBuffer(exact_batch_size=recipe.decisions // width)
        cleanup.pop_all()
    return bodies


class CollectorCheckpointState:
    """Aggregate the small per-replica cursors for the shared checkpoint coordinator."""

    def __init__(self, collector: Any) -> None:
        self.collector = collector

    def state_dict(self) -> dict:
        from cosmos_framework.posttrain.utils.entry import resolve_all

        states = resolve_all(self.collector.state_dict())
        return {"collectors": {str(state["replica"]): state for state in states}}


def manifest(evaluation_enabled: bool = True):
    from cosmos_framework.posttrain.control.recipe import joint_ppo_manifest

    specs = list(ROLE_SPECS) + ([("eval_sim", "sim")] if evaluation_enabled else [])
    return joint_ppo_manifest(specs, weight_relationship="same_gpu")


def validate_config(config: PosttrainJointPPOConfig, ranks: int) -> None:
    """Typed-config, source-manifest and role-manifest checks shared by --validate-manifest and run."""
    from cosmos_framework.posttrain.config.config_joint_ppo import validate_joint_ppo_config
    from cosmos_framework.posttrain.control.bind import bind_check

    validate_joint_ppo_config(config, ranks)
    report = bind_check(manifest(bool(config.rl.validation.enabled)))
    if not report.ok:
        raise ValueError(f"joint PPO manifest failed: {report.format()}")


def assemble(
    config: PosttrainJointPPOConfig,
    recipe: JointPPOConfig,
    output: Path,
    resume: str | None,
    width: int,
    *,
    evaluation: bool | None = None,
) -> dict:
    """Bind model-independent role bodies; experience replicas preserve their local FIFO inventory."""
    from cosmos_framework.posttrain.control.placement import (
        ActorGroup,
        bind_device_groups,
        trainer_experience_device_groups,
        validate_peer_chain,
    )
    from cosmos_framework.posttrain.roles import roles

    evaluation = recipe.validation.enabled if evaluation is None else evaluation
    simulator = roles.env("sim", replicas=width, ranks=1, needs_bulk=True)
    evaluation_simulator = roles.env("eval_sim", replicas=width, ranks=1, needs_bulk=True) if evaluation else None
    collector = roles.env("collector", replicas=width, ranks=1, needs_bulk=True)
    rollout = roles.rollout("rollout", replicas=width, ranks=1, needs_bulk=True)
    reward = roles.reward("rm", replicas=width, ranks=1)
    replay = roles.buffer("replay", replicas=width, ranks=1)
    policy = roles.trainer("policy", replicas=1, ranks=width)
    experience_roles = (simulator, rollout, collector, reward, replay) + (
        (evaluation_simulator,) if evaluation_simulator is not None else ()
    )
    groups = trainer_experience_device_groups(
        ActorGroup(
            members=experience_roles,
            setup=functools.partial(
                setup_experience,
                config,
                str(output),
                resume,
                recipe,
                width,
                members=tuple(role.name for role in experience_roles),
            ),
            name="joint_ppo_experience",
        ),
        ActorGroup(
            members=(policy,),
            setup=functools.partial(setup_policy, config, str(output), resume, recipe),
            name="joint_ppo_policy",
        ),
        gpu_placement="colocated",
        name="joint_ppo",
    )
    bind_device_groups(
        groups, backend="ray", weight_sync=True, temp_dir=os.environ.get("RAY_TMPDIR", str(output / "ray"))
    )
    from cosmos_framework.posttrain.utils.entry import resolve_all, resolve_identical
    from cosmos_framework.posttrain.utils.identity import json_identity_equal

    with ExitStack() as cleanup:
        for role in (rollout, collector, simulator, evaluation_simulator):
            if role is not None:
                cleanup.callback(lambda role=role: resolve_all(role.close()))
        validate_peer_chain((simulator, collector, rollout))
        if evaluation_simulator is not None:
            validate_peer_chain((evaluation_simulator, collector, rollout))
        expected = resolve_identical(policy.ppo_identity(), operation="Joint PPO learner model identity")
        if any(
            not json_identity_equal(identity, expected, context="Joint PPO rollout model identity")
            for identity in resolve_all(rollout.ppo_identity())
        ):
            raise ValueError("Joint PPO rollout and learner model identities differ")
        resolve_all(collector.bind_rollout(rollout))
        cleanup.pop_all()
    return {role.name: role for role in (*experience_roles, policy)}


def run(config: Any, opts: list[str] | None = None) -> dict:
    import ray

    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.control.lifecycle import build_controller_observers
    from cosmos_framework.posttrain.control.loop.decision_window import collect_decision_window
    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync
    from cosmos_framework.posttrain.control.termination import TerminationWatch
    from cosmos_framework.posttrain.control.validation.boundary import ValidationBoundaryCoordinator
    from cosmos_framework.posttrain.control.validation.lifecycle import ValidationSchedule
    from cosmos_framework.posttrain.control.validation.rollout import WindowEvaluator
    from cosmos_framework.posttrain.control.validation.transaction import ValidationTransaction
    from cosmos_framework.posttrain.utils.entry import (
        dump_resolved,
        out_dir,
        pin_run_attempt,
        resolve_all,
        resolve_identical,
    )
    from cosmos_framework.posttrain.utils.log import info0
    from cosmos_framework.posttrain.utils.mesh import detect_mesh
    from cosmos_framework.posttrain.utils.report import (
        emit_metric_frame,
        finish_best_effort,
        make_reporters,
        start_best_effort,
    )

    config = resolve_joint_config(config)
    nodes, gpus = detect_mesh()
    width = nodes * gpus
    validate_config(config, width)
    attempt = pin_run_attempt(config, opts or [], out_dir(config))
    recipe = instantiate(config.rl)
    output = Path(attempt.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    dump_resolved(config, str(output))
    resume = attempt.resume_checkpoint
    lifecycle = CheckpointLifecycle.from_config(config, recipe, run_out_dir=str(output), resume_checkpoint=resume)
    with ExitStack() as cleanup:
        cleanup.callback(ray.shutdown)
        observers = build_controller_observers(
            config, recipe, controller="joint_ppo", world_size=width, topology="joint"
        )
        cleanup.callback(observers.on_app_end)
        observers.on_app_start()
        units = assemble(config, recipe, output, resume, width)
        collector, rollout, policy = units["collector"], units["rollout"], units["policy"]
        reward, replay = units["rm"], units["replay"]
        cleanup.callback(lambda: resolve_all(rollout.close()))
        cleanup.callback(lambda: resolve_all(collector.close()))
        cleanup.callback(lambda: resolve_all(units["sim"].close()))
        if "eval_sim" in units:
            cleanup.callback(lambda: resolve_all(units["eval_sim"].close()))
        reporters, failed_reporters = [], set()
        cleanup.callback(finish_best_effort, reporters, failed_reporters=failed_reporters)
        cleanup.callback(CheckpointLifecycle.finalize_units, {"policy": policy})
        progress = resolve_identical(policy.progress(), operation="Joint PPO restored counters")
        version, step = progress["publication_version"], progress["optimizer_step"]
        updates = recipe.decisions // recipe.global_minibatch
        if step != version * updates or step > recipe.max_steps:
            raise ValueError("restored optimizer/publication counters differ from the configured training budget")
        observers.on_train_start(train_iterations_start=version, train_samples_start=version * recipe.decisions)
        cleanup.callback(observers.on_train_end)
        checkpoints = lifecycle.coordinator(
            str(output),
            policy,
            data=CollectorCheckpointState(collector),
            initial_step=step,
            observer=observers,
        )
        reporters.extend(
            make_reporters(
                config,
                str(output),
                total_steps=recipe.max_steps,
                headline_keys=("train/joint_loss", "train/episode_success_rate", "val/success_rate"),
                run_metadata=observers.metadata or None,
            )
        )
        watch = TerminationWatch()
        transaction = ValidationTransaction(policy, checkpoints, reporters, failed_reporters)
        evaluator = WindowEvaluator(
            policy, collector, instantiate(config.environment), width, str(output), units.get("eval_sim")
        )
        validation = ValidationBoundaryCoordinator(
            ValidationSchedule.from_config(recipe.validation),
            recipe.max_steps,
            transaction,
            observers.evaluate(evaluator),
            watch.should_stop,
            publication_step=transaction.pending_publication_step,
            pending_reasons=transaction.pending_reasons,
        )
        window_count, stopped = 0, None
        start_best_effort(reporters, failed_reporters=failed_reporters)
        weight_sync(rollout, policy, validation_step=version)
        stopped = validation.resume_or_initial(step, resumed_from_checkpoint=bool(resume)).preempted
        while step < recipe.max_steps and stopped is None:
            started = time.monotonic()
            with observers.train_step():
                info0(f"Joint PPO collect version={version} optimizer_step={step}")
                trajectories = collect_decision_window(
                    collector, units["sim"], version=version, decisions=recipe.window_decisions
                )
                scored = reward.score_with_provider_report(trajectories)
                resolve_all(replay.put_samples(scored))
                batch = replay.get_exact_samples()
                receipt = resolve_identical(policy.prepare_ppo_targets(batch), operation="joint PPO targets")
                batch_id, success = receipt["batch_id"], False
                try:
                    with observers.train_batch():
                        result = resolve_identical(policy.train_batch(batch_id), operation="Joint PPO joint update")
                    if result["optimizer_step"] != step + updates or receipt["decisions"] != recipe.decisions:
                        raise RuntimeError("Joint PPO window counters differ from the fixed recipe")
                    step, version = result["optimizer_step"], version + 1
                    resolve_all(policy.set_publication_version(version))
                    weight_sync(rollout, policy, validation_step=version)
                    resolve_all(policy.clear_batch(batch_id))
                    success = True
                finally:
                    if not success:
                        resolve_all(policy.abort_batch(batch_id))
            seconds = time.monotonic() - started
            shards = resolve_all(collector.window_statistics())
            episodes = [episode for shard in shards for episode in shard["episodes"]]
            diagnostics = receipt["target_diagnostics"]
            frame = {
                f"train/{key}": value
                for key, value in result["metrics"].items()
                if key not in {"policy_lr", "policy_optimizer_step"}
            }
            frame.update(receipt["target_metrics"])
            frame.update(
                {
                    "rollout/publication_version": version,
                    "train/optimizer_step": step,
                    "train/episode_count": len(episodes),
                    "train/reward_sum": sum(row["reward"] for row in diagnostics),
                    "train/raw_reward_sum": sum(shard["reward"] for shard in shards),
                    "train/executed_control_steps": sum(row["executed_steps"] for row in diagnostics),
                    "train/global_grad_norm_preclip_mean": result["metrics"]["policy_grad_norm"],
                    "train/global_grad_norm_preclip_max": max(
                        row["policy_grad_norm"] for row in result["optimizer_metrics"]
                    ),
                    "train/actor_update_l2": result["parameter_update_l2"]["actor"],
                    "train/value_update_l2": result["parameter_update_l2"]["value"],
                    "optim/actor_learning_rate": result["learning_rates"]["actor"],
                    "optim/value_learning_rate": result["learning_rates"]["value"],
                    "rollout/behavior_verified": int(receipt["behavior_verified"]),
                    "timer/window_seconds": seconds,
                    "train/decisions_per_second": recipe.decisions / seconds,
                }
            )
            frame.update(
                {
                    f"rollout/{key}": receipt[key]
                    for key in ("max_logprob_delta", "initial_clip_fraction")
                    if receipt[key] is not None
                }
            )
            if episodes:
                frame.update(
                    {
                        "train/episode_success_rate": sum(e["success"] for e in episodes) / len(episodes),
                        "train/episode_return": sum(e["return"] for e in episodes) / len(episodes),
                    }
                )
            row = {
                **receipt,
                **result,
                "publication_version": version,
                "seconds": seconds,
                "episodes": episodes,
            }
            with (output / "windows.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            window_count += 1
            stopped = watch.should_stop()
            due = bool(recipe.save_every and step % recipe.save_every == 0) or step == recipe.max_steps
            if stopped and stopped != "signal SIGTERM":
                outcome = validation.finish_early(step, frame)
            else:
                outcome = validation.at_step(step, frame, scheduled_checkpoint_due=due)
            if outcome.report is None and outcome.preempted is None:
                if due or stopped:
                    checkpoints.save_full(step)
                emit_metric_frame(reporters, frame, step, failed_reporters=failed_reporters)
            stopped = stopped or outcome.preempted
    summary = {
        "windows": window_count,
        "restored_from": resume,
        "optimizer_step": step,
        "publication_version": version,
        "stop_reason": stopped,
        "output": str(output),
    }
    (output / f"summary-step-{step}.json").write_text(json.dumps(summary, indent=4) + "\n")
    return summary


def main() -> None:
    from cosmos_framework.utils.config import load_config
    from cosmos_framework.posttrain.utils.entry import cli
    from cosmos_framework.posttrain.utils.mesh import detect_mesh

    path, opts, validate = cli(__doc__, "cosmos_framework/posttrain/config/config_joint_ppo.py")
    config = load_config(path, ["--", *opts])
    if validate:
        nodes, gpus = detect_mesh()
        validate_config(config, nodes * gpus)
        print("Joint PPO configuration, sources and role manifest are valid", flush=True)
        return
    run(config, opts)


if __name__ == "__main__":
    main()
