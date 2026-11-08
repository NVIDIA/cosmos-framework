# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Offline Diffusion-DPO launch, role assembly, and training/validation loop.

Reusable source, checkpoint, and transaction mechanics live in their owning packages. This module
keeps the DPO-specific wiring and loop visible for local/torchrun and Ray launch modes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from cosmos_framework.posttrain.control.validation.lifecycle import ValidationReport
from cosmos_framework.posttrain.utils.entry import (
    cli,
    dump_resolved,
    render_manifest_validation,
    resolve,
)
from cosmos_framework.posttrain.utils.entry import out_dir as _out_dir
from cosmos_framework.posttrain.utils.entry import (
    resolve_identical as _resolve_identical,
)
from cosmos_framework.posttrain.utils.log import info0, logger

ROLE_SPECS = [("data", "env"), ("policy", "trainer")]  # role name → factory kind
# Only model-bearing roles span the rank mesh; SPMD helpers stay at one rank.
_MODEL_BEARING = {"trainer", "rollout"}


def _validate_dataset_factories(rl: Any) -> None:
    """Preflight configured train and held-out dataset factories before placement."""
    from cosmos_framework.posttrain.data.datasets.factory import validate_dataset_factory

    data_configs = [getattr(rl, "data", None)]
    validation = getattr(rl, "validation", None)
    if validation is not None and bool(getattr(validation, "enabled", False)):
        data_configs.append(getattr(validation, "data", None))
    for data_config in data_configs:
        factory_path = str(getattr(data_config, "factory", ""))
        if factory_path:
            validate_dataset_factory(factory_path)


def declare_wgs(ranks: int) -> dict:
    """Declare DPO role shapes; only model-bearing roles use the mesh width."""
    from cosmos_framework.posttrain.roles import roles as role_factory

    return {
        name: getattr(role_factory, kind)(name, ranks=ranks if kind in _MODEL_BEARING else 1)
        for name, kind in ROLE_SPECS
    }


def build_env(rl: Any) -> Any:
    """Build the rank-sharded training preference source."""
    from cosmos_framework.posttrain.roles.env import build_preference_env

    validation = getattr(rl, "validation", None)
    return build_preference_env(
        rl.data,
        int(getattr(rl, "seed", 0)),
        include_source_identity=validation is not None and bool(validation.enabled),
    )


def build_trainer(config: Any, rl: Any, world_size: int, resume_checkpoint: str | None = None) -> Any:
    """Assemble the DPO policy model, objective, optimizer, and trainer."""
    from cosmos_framework.utils import misc
    from cosmos_framework.posttrain.algorithm.dpo import make_objective
    from cosmos_framework.posttrain.callbacks.dpo import DPOMetricLogger, DPORewardTrendCallback
    from cosmos_framework.posttrain.config.build import build_backbone, build_optimizer
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.model.cosmos3 import Cosmos3FlowPolicy
    from cosmos_framework.posttrain.roles.env import build_preference_env
    from cosmos_framework.posttrain.roles.trainer.diffusion import CopyInventoryPolicy, DiffusionTrainer
    from cosmos_framework.posttrain.utils.dist import run_rank_converged

    configured_validation = getattr(rl, "validation", None)
    validation_config = configured_validation if configured_validation and configured_validation.enabled else None
    validation_source = None
    if validation_config is not None:
        validation_source = run_rank_converged(
            lambda: build_preference_env(
                validation_config.data,
                int(validation_config.seed),
                include_source_identity=True,
            ),
            "preference validation source setup",
        )

    checkpoint_lifecycle = CheckpointLifecycle.from_config(
        config,
        rl,
        resume_checkpoint=resume_checkpoint,
    )
    localized_init_from = checkpoint_lifecycle.localize_init_from(rl.init_from) if rl.init_from else None
    uses_frozen_reference = bool(rl.ref.build_ref) and int(rl.ref.update_every) <= 0

    # Build every rank from the same RNG stream so a random-init frozen reference has one identity.
    misc.set_random_seed(seed=int(rl.seed), by_rank=False)
    model = build_backbone(config, world_size)
    policy_model = Cosmos3FlowPolicy(model, build_ref=rl.ref.build_ref)
    opt = build_optimizer(model, rl.optimizer)
    objective = make_objective(rl.beta)  # <- objective is selected here (later: a small registry by name)
    trainer = DiffusionTrainer(
        policy_model,
        objective,
        opt,
        grad_accum=rl.grad_accum,
        update_ref_every_iter=rl.ref.update_every,
        callbacks=[DPOMetricLogger(log_every=1), DPORewardTrendCallback(window=max(1, rl.max_steps), warn=False)],
        grad_clip_norm=float(rl.optimizer.grad_clip_norm),
        force_finite_grads=bool(rl.optimizer.force_finite_grads),
        checkpoint_manager=checkpoint_lifecycle.manager,
        validation_source=validation_source,
        validation_batch_size=int(validation_config.batch_size) if validation_config else 1,
        validation_max_samples=validation_config.max_samples if validation_config else None,
        validation_seed=int(validation_config.seed) if validation_config else 0,
        validation_before_training=bool(validation_config.before_training) if validation_config else False,
        validation_every_steps=int(validation_config.every_steps) if validation_config else 0,
        validation_train_total_steps=int(rl.max_steps) if validation_config else 0,
        defer_frozen_reference_identity=localized_init_from is not None and uses_frozen_reference,
    )
    # init_from anchors both policy and frozen reference; resume then restores continuation state.
    # Both loads are required because frozen references are deliberately absent from checkpoints.
    ckpt = checkpoint_lifecycle.resume_checkpoint
    if localized_init_from is not None:
        trainer.load(
            localized_init_from,
            copy_mode=True,
            copy_inventory_policy=CopyInventoryPolicy(source_only_keys=tuple(rl.init_keys_to_skip_loading)),
        )
        info0(f"initialized net+ref from base {rl.init_from}")

    # Training uses rank-local randomness; exact resume replaces it with checkpointed RNG state.
    misc.set_random_seed(seed=int(rl.seed), by_rank=True)
    if ckpt:
        it = checkpoint_lifecycle.load_trainer(trainer, ckpt)
        info0(f"resumed from {ckpt} (iter {it})")
    return trainer


def make_setup(config_path: str, opts: list[str], roles: Sequence[str]) -> Callable[[], dict[str, Any]]:
    """Build the requested roles in-process or inside a Ray actor."""

    def _setup() -> dict:
        from cosmos_framework.utils.config import load_config
        from cosmos_framework.utils.lazy_config import instantiate
        from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
        from cosmos_framework.posttrain.utils.dist import (
            run_rank_converged,
            world_size_or_1,
        )

        world = world_size_or_1()
        config = load_config(config_path, ["--", *opts])
        rl = instantiate(config.rl)
        if bool(getattr(rl, "deterministic", False)):
            from cosmos_framework.posttrain.utils.determinism import enable_determinism

            enable_determinism()
        checkpoint_lifecycle = CheckpointLifecycle.from_config(config, rl)
        ckpt = checkpoint_lifecycle.resume_checkpoint
        out = {}
        if "data" in roles:
            logger.debug(f"setup: building env (jsonl={rl.data.jsonl_path})")
            env = run_rank_converged(lambda: build_env(rl), "preference training source setup")
            if ckpt:
                checkpoint_lifecycle.restore_cursor(env, str(rl.data.cursor_resume), ckpt)
            out["data"] = env
        if "policy" in roles:
            logger.debug(f"setup: building trainer (world={world})")
            out["policy"] = build_trainer(config, rl, world, ckpt)  # only this builds the (8B) model
        info0(f"setup done: roles={list(out)} world={world}")
        return out

    return _setup


# --------------------------------------------------------------------------- control loop
class _RejectedUpdateError(RuntimeError):
    """Carry a rejected trainer outcome to the control loop for reporting."""

    def __init__(self, metrics: dict[str, Any], microbatch: int) -> None:
        super().__init__(str(metrics.get("optim/skip_reason", "nonfinite")))
        self.metrics = dict(metrics)
        self.microbatch = microbatch


def train_step(data: Any, policy: Any, *, pairs_per_step: int, grad_accum: int) -> dict[str, Any]:
    """Run one optimizer step and merge its gradient-accumulation metric window."""
    if grad_accum < 1:
        raise ValueError(f"grad_accum must be >= 1, got {grad_accum}")
    from cosmos_framework.posttrain.utils.report import merge_training_metric_window

    reports: list[dict[str, Any]] = []
    for microbatch in range(grad_accum):
        pairs = data.get_samples(pairs_per_step)
        metrics = resolve(policy.train(pairs))
        if metrics.get("optim/update_outcome") == "skipped":
            raise _RejectedUpdateError(metrics, microbatch)
        reports.append(metrics)
    return reports[0] if len(reports) == 1 else merge_training_metric_window(reports)


def _log_validation_report(report: ValidationReport, model_step: int) -> None:
    """Log the DPO-specific held-out objective headline."""
    info0(
        f"validation step={model_step} pairs={report.sample_count} "
        f"loss={report.metrics['val/loss_avg']:.6f} reasons="
        f"{','.join(sorted(reason.value for reason in report.trigger_reasons))}"
    )


def loop(units: dict[str, Any], config: Any, out_dir: str) -> dict[str, Any]:
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.control.termination import TerminationWatch
    from cosmos_framework.posttrain.control.validation import ValidationSchedule
    from cosmos_framework.posttrain.control.validation.boundary import ValidationBoundaryCoordinator
    from cosmos_framework.posttrain.control.validation.transaction import (
        ValidationTransaction,
        evaluate_policy_validation,
    )
    from cosmos_framework.posttrain.utils.report import (
        emit_metric_frame,
        finish_best_effort,
        make_reporters,
        reportable_metrics,
        start_best_effort,
    )

    rl = instantiate(config.rl)
    data, policy = units["data"], units["policy"]
    # Resume-aware start: the trainer restored its iteration in setup (rl.resume); the
    # loop continues from there so step numbering and iter_* dirs stay continuous.
    start = int(_resolve_identical(policy.iteration(), operation="policy.iteration"))
    if start > int(rl.max_steps):
        raise RuntimeError(f"restored trainer iteration {start} exceeds configured total steps {rl.max_steps}")
    if start:
        info0(f"resuming loop at step {start}/{rl.max_steps}")
    resumed_probe = getattr(policy, "resumed_from_checkpoint", None)
    resumed_from_checkpoint = (
        bool(_resolve_identical(resumed_probe(), operation="policy.resumed_from_checkpoint"))
        if callable(resumed_probe)
        else start > 0
    )
    watch = TerminationWatch()  # Slurm preemption/timeout sentinel + direct signals
    # Reporting is a control-plane concern: the loop is the single writer (rank-0 /
    # driver), fed the all-reduced metrics — console + W&B behind one Reporter concept.
    reporters = make_reporters(config, out_dir, total_steps=int(rl.max_steps))  # DPO: fixed length
    failed_reporters: set[int] = set()
    start_best_effort(reporters, failed_reporters=failed_reporters)

    validation = getattr(rl, "validation", None)
    validation_schedule = (
        ValidationSchedule.from_config(validation)
        if validation is not None
        else ValidationSchedule(enabled=False, before_training=False, every_steps=0)
    )
    checkpoint_lifecycle = CheckpointLifecycle.from_config(config, rl, run_out_dir=out_dir)
    checkpoints = checkpoint_lifecycle.coordinator(out_dir, policy, data=data, initial_step=start)
    transaction = ValidationTransaction(policy, checkpoints, reporters, failed_reporters)
    evaluate = partial(evaluate_policy_validation, policy)
    validation_boundary = ValidationBoundaryCoordinator(
        schedule=validation_schedule,
        total_steps=int(rl.max_steps),
        transaction=transaction,
        evaluator=evaluate,
        should_stop=watch.should_stop,
        publication_step=transaction.pending_publication_step,
    )

    history: list[float] = []
    preempted: str | None = None
    try:
        import time

        validation_outcome = validation_boundary.resume_or_initial(
            start,
            resumed_from_checkpoint=resumed_from_checkpoint,
        )
        preempted = validation_outcome.preempted
        if validation_outcome.report is not None:
            _log_validation_report(validation_outcome.report, start)

        steps_to_run = range(start, rl.max_steps) if preempted is None else ()
        for step in steps_to_run:
            t0 = time.monotonic()
            # One loop step == one OPTIMIZER step (wallace-comparable iteration axis) via the shared
            # inner step the parity harness also drives, so the two can't diverge.
            try:
                metrics = train_step(data, policy, pairs_per_step=rl.pairs_per_step, grad_accum=rl.grad_accum)
            except _RejectedUpdateError as error:
                raise RuntimeError(
                    f"rejected non-finite update at optimizer step {step}, microbatch {error.microbatch}: "
                    f"{error}; attempt={error.metrics.get('optim/attempt_id')} "
                    f"total_skips={error.metrics.get('optim/nonfinite_skips_total')}"
                ) from error
            loss = float(metrics["train/loss_avg"])
            if not math.isfinite(loss):
                raise RuntimeError(
                    f"applied update returned non-finite reduced train/loss_avg at step {step + 1}: {loss}; "
                    "rank-consensus invariant violated"
                )
            history.append(loss)
            # Driver-perceived step wall time (data fetch + train + handle resolve) —
            # the timer/* signal the trainer's WandBCallback also exports.
            metrics = {**metrics, "timer/e2e_step_s": time.monotonic() - t0}
            model_step = step + 1
            restored_iteration = int(_resolve_identical(policy.iteration(), operation="policy.iteration"))
            if restored_iteration != model_step:
                raise RuntimeError(
                    f"trainer iteration {restored_iteration} does not match committed model step {model_step}"
                )
            frame = reportable_metrics(metrics)
            normal_save_due = bool(
                rl.save_every and ((step + 1) % rl.save_every == 0 or model_step == int(rl.max_steps))
            )
            validation_outcome = validation_boundary.at_step(
                model_step,
                frame,
                scheduled_checkpoint_due=normal_save_due,
            )
            if validation_outcome.report is not None:
                _log_validation_report(validation_outcome.report, model_step)
            elif validation_outcome.preempted is None:
                emit_metric_frame(reporters, frame, model_step, failed_reporters=failed_reporters)
            if validation_outcome.preempted is not None:
                preempted = validation_outcome.preempted
                break
            if normal_save_due:
                checkpoints.save_configured_if_needed(model_step)
            # Preemption/timeout: rank-consistent check at the step boundary (a mid-step
            # signal is caught here next step — the >=300s Slurm grace covers one step+save).
            preempted = watch.should_stop()
            if preempted:
                # rl.emergency_checkpoint_components: "full" upgrades to the exact-resume plan; "run" writes
                # the run's own cadence plan — see RunConfig for the trade.
                emergency_plan = str(rl.emergency_checkpoint_components)
                if checkpoints.save_emergency(model_step, components=emergency_plan):
                    logger.warning(f"termination trigger ({preempted}) — emergency checkpoint ({emergency_plan} plan)")
                else:
                    logger.warning(f"termination trigger ({preempted}) — already checkpointed, exiting")
                break
    finally:
        finish_best_effort(reporters, failed_reporters=failed_reporters)
    summary = {
        "steps": rl.max_steps,
        "start": start,
        "final_loss": history[-1] if history else None,
        **({"preempted": preempted} if preempted else {}),
    }
    info0(f"DONE {summary} resolved={out_dir}/config_resolved.yaml")
    return summary


# --------------------------------------------------------------------------- launch
def run(config: Any, config_path: str, opts: list[str], out_dir: str) -> None:
    """Bind local or Ray role groups and drive the DPO loop."""
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.control.bind import bind_check
    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.control.recipe import dpo_manifest

    # Pre-flight the SAME manifest `--validate-manifest` checks, before any cluster or model work: a wiring or
    # transport mistake fails here in milliseconds instead of after the placement group and the 8B build.
    report = bind_check(dpo_manifest(ROLE_SPECS))
    info0(report.format())
    report.raise_for_status()

    backend, grouping = str(config.grid.backend), str(config.grid.grouping)
    if backend not in ("local", "ray"):
        raise ValueError(f"grid.backend={backend!r}: expected 'local' or 'ray'")
    if grouping not in ("single", "per_role"):
        raise ValueError(f"grid.grouping={grouping!r}: expected 'single' or 'per_role'")
    if backend == "local" and grouping == "per_role":
        raise ValueError(
            "grid.grouping=per_role requires grid.backend=ray — the local backend hosts all "
            "roles in this one process, so there are no separate processes to split roles into."
        )
    rl = instantiate(config.rl)
    _validate_dataset_factories(rl)
    if bool(getattr(rl, "deterministic", False)):
        from cosmos_framework.posttrain.utils.determinism import enable_determinism

        enable_determinism()

    all_roles = [name for name, _ in ROLE_SPECS]
    groups = [all_roles] if grouping == "single" else [[r] for r in all_roles]

    from cosmos_framework.posttrain.utils.dist import init_local_process_group
    from cosmos_framework.posttrain.utils.entry import pin_run_attempt

    local_pg = backend == "local"
    if local_pg:
        # torchrun is the process launcher: join its NCCL world (= the FSDP mesh).
        init_local_process_group()
        ranks = 1  # per-WG mesh is 1×1; parallelism comes from torchrun's WORLD_SIZE
        opts_eff = list(opts)
    else:
        # ray sizes its own mesh: K worker processes per group, model FSDP-shards across them.
        ranks = int(config.grid.ranks)
        opts_eff = list(opts)
        if not any(o.startswith("model.config.parallelism.data_parallel_shard_degree=") for o in opts_eff):
            opts_eff = [*opts_eff, f"model.config.parallelism.data_parallel_shard_degree={ranks}"]

    attempt = pin_run_attempt(config, opts_eff, out_dir)
    out_dir, opts_eff = attempt.out_dir, list(attempt.opts)
    logger.info(f"run: backend={backend} grouping={grouping} groups={groups} out={out_dir}")

    dump_resolved(config, out_dir)
    wgs: dict = {}
    try:
        wgs = declare_wgs(ranks)
        actor_groups = tuple(
            ActorGroup(members=tuple(wgs[role] for role in group), setup=make_setup(config_path, opts_eff, group))
            for group in groups
        )
        if backend == "local":
            device_groups = [DeviceGroup(actor_groups=actor_groups, device="cpu")]
        elif grouping == "single":
            device_groups = [DeviceGroup(actor_groups=actor_groups, device="gpu")]
        else:
            device_groups = [DeviceGroup(actor_groups=(actor_group,), device="gpu") for actor_group in actor_groups]
        bind_device_groups(
            device_groups, backend=backend, gpu_node_domains=getattr(config.grid, "gpu_node_domains", None)
        )
        validation = getattr(rl, "validation", None)
        if validation is not None and bool(validation.enabled):
            from cosmos_framework.posttrain.roles.env.preference import validate_preference_sources

            validate_preference_sources(wgs["data"], wgs["policy"])
        loop(wgs, config, out_dir)
    finally:
        from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
        from cosmos_framework.posttrain.utils.dist import destroy_process_group_if_initialized

        CheckpointLifecycle.finalize_units(
            wgs,
            destroy_process_group_if_initialized if local_pg else None,
        )


def _validate_and_exit() -> None:
    """Run the GPU-free DPO manifest preflight."""
    from cosmos_framework.posttrain.control.recipe import dpo_manifest

    manifest = dpo_manifest(ROLE_SPECS)
    output, ok = render_manifest_validation(manifest)
    print(output)  # persisted to the run log
    raise SystemExit(0 if ok else 1)


def main() -> None:
    config_path, opts, validate = cli(
        "Cosmos3 offline Diffusion-DPO", default_config="cosmos_framework/posttrain/config/config_dpo.py"
    )
    if validate:
        _validate_and_exit()  # before load_config: the manifest is config-independent, so keep torch out

    from cosmos_framework.utils.config import load_config

    config = load_config(config_path, ["--", *opts])
    out_dir = _out_dir(config)
    run(config, config_path, opts, out_dir)


if __name__ == "__main__":
    main()
