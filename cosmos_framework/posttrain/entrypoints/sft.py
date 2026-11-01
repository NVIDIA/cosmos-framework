# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Offline multimodal conversation SFT using the shared LLM engine and lifecycle."""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict
from functools import partial
from typing import Any

from cosmos_framework.posttrain.config.sft import SFTConfig, SFTDatasetConfig
from cosmos_framework.posttrain.control.validation.lifecycle import ValidationReport, ValidationTrigger
from cosmos_framework.posttrain.utils.entry import (
    cli_flags,
    dump_resolved,
    out_dir,
    render_manifest_validation,
    resolve,
)
from cosmos_framework.posttrain.utils.log import info0

# Which native families seat SFT is declared by the family beside its builder (``model/sft_source.py``):
# today ``model/qwen3_vl.py`` registers Qwen3-VL / Qwen3-VL-MoE, built through the same
# policy adapter the online algorithms train. The HF backend keeps refusing MoE VLMs (``model/hf.py::_is_moe_vlm``), so
# a Qwen3-VL-MoE base reaches the SFT objective only through a registered native builder.


# SFT console headline columns. The trainer computes train/grad_norm, train/learning_rate and the
# DP-summed train/valid_token_count every step and the loop adds timer/e2e_step_s, but the default headline keys
# (utils/report.py::_CONSOLE_KEYS, the GRPO/DPO set) drop them, and with W&B off the step-size regime of a run was
# unobservable from its log. The console reporter prints |value| < 1e-3 in scientific notation, so lr 1e-5 is legible.
SFT_CONSOLE_KEYS: tuple[str, ...] = (
    "train/loss_avg",
    "train/lbl_loss",
    "train/grad_norm",
    "train/learning_rate",
    "train/valid_token_count",
    "timer/e2e_step_s",
    "val/loss_avg",
)

# The run directory's per-step record (utils/report.py::JsonlReporter): every public key of every committed frame,
# one JSON line per step; bounded, rank 0 only.
SFT_METRICS_FILE = "metrics.jsonl"


def role_specs(config: Any) -> list[tuple[str, str]]:
    roles = [("data", "env"), ("policy", "trainer")]
    if config.rl.validation.enabled:
        roles.append(("validation", "env"))
    return roles


def sft_policy_source(config: Any) -> tuple[str, bool]:
    """Resolve the one model/tokenizer artifact the SFT packer and the backbone share.

    ``model_backend='hf'``: ``model_hf.name_or_path`` (Qwen text models and dense Qwen3.5/3.8 through
    the HF backend). ``model_backend='i4'``: ``config.model`` must be a ``NativePolicyBuildSpec`` whose
    builder's model family has registered an SFT source (``model/sft_source.py``; the Qwen3-VL family
    checks ``model_type`` ``qwen3_vl`` / ``qwen3_vl_moe`` there); an undeclared builder is refused by
    name. No model or processor is constructed.

    Returns:
        ``(name_or_path, trust_remote_code)`` for ``build_sft_packer``.
    """
    from cosmos_framework.posttrain.config.build import select_policy_backend
    from cosmos_framework.posttrain.model.sft_source import resolve_native_sft_source

    selection = select_policy_backend(config)
    if selection.backend == "hf":
        if not selection.model_id:
            raise ValueError("SFT requires model_hf.name_or_path")
        return selection.model_id, selection.trust_remote_code
    spec = selection.native_spec
    if spec is None:
        raise ValueError(
            "SFT with model_backend='i4' requires config.model=NativePolicyBuildSpec for the Qwen3-VL native "
            "seat; legacy i4 model trees are not supported"
        )
    return resolve_native_sft_source(spec)


def enabled_sources(rl: SFTConfig) -> list[SFTDatasetConfig]:
    """The training source, then the held-out source when validation is enabled."""
    return [rl.data, *([rl.validation.data] if rl.validation.enabled else [])]


def validate_config_structure(config: Any) -> SFTConfig:
    """Configuration-only preflight: backend, grid, model source and source wiring; imports no factory and opens no
    data file (on the native seat the resolver reads the model's ``config.json``, the metadata read the packer
    builder performs as well).

    The row validator (``scripts/validate_sft_rows.py``) runs this alone and reports each selected source's
    availability as that split's load refusal instead of raising, so one unavailable split never hides the other.
    """
    from cosmos_framework.utils.lazy_config import instantiate

    if getattr(config.grid, "gpu_node_domains", None):
        raise ValueError("SFT local/torchrun placement does not support grid.gpu_node_domains")
    if config.grid.backend != "local" or config.grid.grouping != "single":
        raise ValueError("SFT currently supports local/torchrun with grid.grouping='single' only")
    sft_policy_source(config)
    rl = instantiate(config.rl)
    # ``SFTConfig.__post_init__`` judges ``mask_think_opener=True`` by the typed ``serve_contract`` field for every source
    # of the value (the recipe variant, a YAML, a ``--`` override).
    rl.__post_init__()
    if rl.lr_schedule.kind == "cosine" and (rl.lr_schedule.cycle_steps or rl.max_steps) <= rl.optimizer.warmup_steps:
        # Legal (a short canary of a long warmup) but worth one line: the run never leaves the warmup ramp.
        info0(
            f"SFT lr_schedule cosine: the cycle ({rl.lr_schedule.cycle_steps or rl.max_steps} steps) ends inside the "
            f"{rl.optimizer.warmup_steps}-step warmup; the learning rate stays on the ramp from lr*{rl.lr_schedule.f_start}"
        )
    for data in enabled_sources(rl):
        data.require_source()
    return rl


def require_source_available(data: SFTDatasetConfig) -> None:
    """A local JSONL must be readable and a factory importable; remote availability is checked when workers open it."""
    from cosmos_framework.posttrain.data.datasets.factory import validate_dataset_factory
    from cosmos_framework.posttrain.roles.env.dataset import require_readable_jsonl

    if data.path:
        if not data.path.startswith("s3://"):
            require_readable_jsonl(data.path)
    else:
        validate_dataset_factory(data.factory)


def validate_config(config: Any) -> SFTConfig:
    """Pure configuration/source preflight (structure, then every enabled source's availability); no model or processor."""
    from cosmos_framework.posttrain.config.build import refuse_rounding_or_unavailable_optimizer

    rl = validate_config_structure(config)
    refuse_rounding_or_unavailable_optimizer(config, rl.optimizer)
    for data in enabled_sources(rl):
        require_source_available(data)
    return rl


def build_trainer(config: Any, rl: SFTConfig, data: Any, validation: Any | None, checkpoint: Any) -> Any:
    from cosmos_framework.utils import misc
    from cosmos_framework.posttrain.algorithm.sft import make_objective
    from cosmos_framework.posttrain.config.build import (
        build_backbone,
        build_optimizer,
        build_sft_lr_scheduler,
        resolve_precision,
    )
    from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
    from cosmos_framework.posttrain.roles.trainer.validation.sft import evaluate_sft_batch
    from cosmos_framework.posttrain.roles.trainer.validation.state import ValidationState
    from cosmos_framework.posttrain.utils.dist import run_rank_converged, world_size_or_1

    name_or_path, trust_remote_code = sft_policy_source(config)
    packer = run_rank_converged(
        lambda: build_sft_packer(
            name_or_path,
            max_length=rl.max_length,
            processor_kwargs=rl.processor_kwargs,
            trust_remote_code=trust_remote_code,
            mask_think_opener=rl.mask_think_opener,
        ),
        "SFT tokenizer/processor setup",
    )
    objective = make_objective(
        packer,
        ce_exponent=rl.ce_exponent,
        moe_lbl_coeff=rl.moe_lbl_coeff,
        moe_lbl_method=rl.moe_lbl_method,
        serve_contract=rl.serve_contract,
    )
    objective.identity = {
        **objective.identity,
        "training_source": data.source_fingerprint,
        "micro_batch_size": rl.micro_batch_size,
        "grad_accum": rl.grad_accum,
        "dp_world": world_size_or_1(),
        "precision": asdict(resolve_precision(config)),
    }
    misc.set_random_seed(seed=rl.seed, by_rank=False)
    model = build_backbone(config)
    optimizer = build_optimizer(model.net, rl.optimizer)
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        lr = group["lr"]
        lr = float(lr.item()) if hasattr(lr, "item") else float(lr)
        groups.append(
            f"{group.get('name', f'group{index}')}: n={len(group['params'])} "
            f"wd={group.get('weight_decay', rl.optimizer.weight_decay):g} lr={lr:.3e}"
        )
    info0(
        f"SFT optimizer {type(optimizer).__name__}: {len(groups)} group(s) [{'; '.join(groups)}]; "
        f"decay_groups={rl.optimizer.decay_groups!r} lr_multipliers={rl.optimizer.lr_multipliers}"
    )
    trainer = LLMTrainer(
        model,
        objective,
        optimizer,
        grad_accum=rl.grad_accum,
        n_mini=1,
        max_grad_norm=rl.optimizer.grad_clip_norm,
        lr_scheduler=build_sft_lr_scheduler(optimizer, rl.optimizer, rl.lr_schedule, rl.max_steps),
        force_finite_grads=rl.optimizer.force_finite_grads,
        require_gradients=True,
        checkpoint_manager=checkpoint.manager,
        validation_state=ValidationState(),
        objective_evaluator=evaluate_sft_batch,
    )
    # Exact resume replaces this rank-local stream with the checkpoint's RNG state.
    misc.set_random_seed(seed=rl.seed, by_rank=True)
    if checkpoint.resume_checkpoint:
        restored = checkpoint.load_trainer_and_component(trainer, "data")
        if restored is None or restored[1] is None:
            raise ValueError("exact SFT resume requires a data cursor component")
        run_rank_converged(lambda: data.load_state_dict(restored[1]), "SFT cursor restore")
        provenance = trainer.loaded_checkpoint_provenance()
        if provenance is not None:
            data.set_loaded_checkpoint_provenance(provenance)
    validation_identity: dict[str, Any] = {"enabled": False}
    if validation is not None:
        validation_identity = {
            "enabled": True,
            "source_fingerprint": validation.source_fingerprint,
            "objective_identity": dict(objective.identity),
            "batch_size": rl.validation.batch_size,
            "max_samples": rl.validation.max_samples,
            "seed": rl.validation.seed,
            "schedule": {
                "before_training": rl.validation.before_training,
                "every_steps": rl.validation.every_steps,
                "train_total_steps": rl.max_steps,
            },
        }
    trainer.configure_validation(validation_identity)
    return trainer


def held_out_census(data: Any, validation: Any) -> dict[str, Any]:
    """The setup's held-out guard on every rank, with the census computed ONCE.

    ``source_overlap`` walks both corpora and hashes media bytes from shared storage, so running it on every training rank
    (what the setup did before this function: ``run_rank_converged`` over a per-rank census) multiplied the start-up I/O
    by the policy world size. Rank 0 computes the census and ``broadcast_from_rank0`` hands the same dict
    to every rank (a census failure on rank 0 is collective through its converged preparation); every rank then applies
    ``refuse_source_overlap`` on the identical census inside ``run_rank_converged``, so a refusal stays collective and
    the census is rank-identical by construction. The caller has already proven both source fingerprints rank-identical.
    """
    from cosmos_framework.posttrain.data.datasets.sft import refuse_source_overlap, source_overlap
    from cosmos_framework.posttrain.utils.dist import broadcast_from_rank0, run_rank_converged

    census = broadcast_from_rank0(
        lambda: source_overlap(data.dataset, validation.dataset), context="SFT held-out census"
    )
    run_rank_converged(lambda: refuse_source_overlap(census), "SFT held-out disjointness")
    return census


def make_setup(config_path: str, opts: list[str]) -> Any:
    def setup() -> dict[str, Any]:
        from cosmos_framework.utils.config import load_config
        from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
        from cosmos_framework.posttrain.roles.env.sft import build_sft_env
        from cosmos_framework.posttrain.utils.dist import rank_identical, rank_or_0, run_rank_converged, world_size_or_1

        config = load_config(config_path, ["--", *opts])
        rl = validate_config(config)
        if rl.deterministic:
            from cosmos_framework.posttrain.utils.determinism import enable_determinism

            enable_determinism()
        checkpoint = CheckpointLifecycle.from_config(config, rl)
        coordinates = {"rank": rank_or_0(), "world_size": world_size_or_1()}
        data = run_rank_converged(lambda: build_sft_env(rl.data, rl.seed, **coordinates), "SFT training source")
        rank_identical(data.source_fingerprint, "SFT training source identity")
        validation = None
        if rl.validation.enabled:
            validation = run_rank_converged(
                lambda: build_sft_env(rl.validation.data, rl.validation.seed, training=False, **coordinates),
                "SFT validation source",
            )
            rank_identical(validation.source_fingerprint, "SFT validation source identity")
            if data.source_fingerprint == validation.source_fingerprint:
                raise ValueError("SFT training and held-out sources must differ")
            census = held_out_census(data, validation)
            info0(
                f"SFT held-out source disjoint from training: {census['validation_rows']} validation rows "
                f"({census['validation_items']} with an item id) vs {census['training_rows']} training rows "
                f"({census['training_items']} items); shared rows 0, shared items 0"
            )
        policy = build_trainer(config, rl, data, validation, checkpoint)
        bodies = {"data": data, "policy": policy}
        if validation is not None:
            bodies["validation"] = validation
        return bodies

    return setup


def train_step(data: Any, policy: Any, rl: SFTConfig) -> dict[str, Any]:
    from cosmos_framework.posttrain.schema.metrics import SUMMED_TRAIN_COUNT_KEYS
    from cosmos_framework.posttrain.utils.report import merge_training_metric_window

    reports: list[dict[str, Any]] = []
    for micro in range(rl.grad_accum):
        metrics = resolve(policy.train(data.get_samples(rl.micro_batch_size)))
        expected = "applied" if micro == rl.grad_accum - 1 else "accumulating"
        if metrics.get("optim/update_outcome") != expected:
            raise RuntimeError(
                f"SFT rejected update at microbatch {micro}: {metrics.get('optim/skip_reason', metrics)}"
            )
        reports.append(metrics)
    # The clip record (``train/grad_clipped``, ``train/grad_clip_coef``) arrives in the applying microbatch's metrics
    # from ``LLMTrainer``, computed from the norm the clip measured; the loop adds the run's
    # ``train/grad_clip_fraction``.
    return merge_training_metric_window(
        reports,
        sum_keys=frozenset(SUMMED_TRAIN_COUNT_KEYS),
        weighted_mean_by={"train/loss_avg": "train/valid_token_count"},
    )


def evaluate(
    units: dict[str, Any], rl: SFTConfig, model_step: int, reasons: frozenset[ValidationTrigger]
) -> ValidationReport:
    """Restore held-out source and RNG state, including when decoding or evaluation fails."""
    from cosmos_framework.posttrain.roles.rollout.validation import distributed_validation_rng, seed_validation_rng
    from cosmos_framework.posttrain.utils.dist import run_rank_converged

    source = units["validation"]
    state = run_rank_converged(lambda: resolve(source.state_dict()), "SFT validation source snapshot")
    with distributed_validation_rng("SFT validation"):
        seed_validation_rng(rl.validation.seed)
        try:
            return _evaluate(units, rl, model_step, reasons)
        finally:
            run_rank_converged(lambda: resolve(source.load_state_dict(state)), "SFT validation source restoration")


def _evaluate(
    units: dict[str, Any], rl: SFTConfig, model_step: int, reasons: frozenset[ValidationTrigger]
) -> ValidationReport:
    """Finite held-out pass; counts and token numerators are reduced once across ranks."""
    from cosmos_framework.posttrain.utils.dist import rank_identical, run_rank_converged
    from cosmos_framework.posttrain.utils.rank_state import gather_rank_states

    started = time.monotonic()
    source, policy = units["validation"], units["policy"]
    count = resolve(source.validation_batch_count(rl.validation.batch_size, rl.validation.max_samples))
    rank_identical(count, "SFT validation batch count")
    local = {"numerator": 0.0, "tokens": 0.0, "sequences": 0.0}
    for index in range(count):
        batch = source.get_validation_batch(index, rl.validation.batch_size, rl.validation.max_samples)
        metrics = resolve(policy.evaluate_objective(batch))

        def collect() -> None:
            tokens = float(metrics["val/valid_token_count"])
            loss = float(metrics["val/loss_avg"])
            if not math.isfinite(loss) or not math.isfinite(tokens) or tokens < 0:
                raise ValueError("SFT held-out objective returned invalid loss/token count")
            local["numerator"] += loss * tokens
            local["tokens"] += tokens
            local["sequences"] += float(metrics["val/valid_sequence_count"])

        run_rank_converged(collect, "SFT held-out metric preparation")
    local["duration"] = time.monotonic() - started
    gathered = gather_rank_states(local, context="SFT held-out totals")
    tokens = sum(row["tokens"] for row in gathered.values())
    sequences = sum(row["sequences"] for row in gathered.values())
    if tokens <= 0 or sequences <= 0:
        raise ValueError("SFT held-out source contains no supervised tokens")
    loss = sum(row["numerator"] for row in gathered.values()) / tokens
    duration = max(row["duration"] for row in gathered.values())
    fingerprint = resolve(source.source_identity())
    return ValidationReport(
        model_step=model_step,
        sample_count=int(sequences),
        metrics={
            "val/loss_avg": loss,
            "val/valid_token_count": int(tokens),
            "val/valid_sequence_count": int(sequences),
            "timer/validation_s": duration,
        },
        source_fingerprint=fingerprint,
        trigger_reasons=reasons,
        duration_s=duration,
    )


def loop(units: dict[str, Any], config: Any, run_out_dir: str) -> dict[str, Any]:
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.control.termination import TerminationWatch
    from cosmos_framework.posttrain.control.validation import ValidationSchedule
    from cosmos_framework.posttrain.control.validation.boundary import ValidationBoundaryCoordinator
    from cosmos_framework.posttrain.control.validation.transaction import ValidationTransaction
    from cosmos_framework.posttrain.utils.report import (
        emit_metric_frame,
        finish_best_effort,
        make_reporters,
        prior_clip_counts,
        reportable_metrics,
        start_best_effort,
    )

    rl = instantiate(config.rl)
    data, policy = units["data"], units["policy"]
    start = int(resolve(policy.iteration()))
    if start > rl.max_steps:
        raise ValueError("restored SFT iteration exceeds configured max_steps")
    resumed = bool(resolve(policy.resumed_from_checkpoint()))
    checkpoint = CheckpointLifecycle.from_config(config, rl, run_out_dir=run_out_dir)
    checkpoints = checkpoint.coordinator(run_out_dir, policy, data=data, initial_step=start)
    reporters = make_reporters(
        config,
        run_out_dir,
        total_steps=rl.max_steps,
        headline_keys=SFT_CONSOLE_KEYS,
        jsonl_path=os.path.join(run_out_dir, SFT_METRICS_FILE),
        # A relaunch resumes at `start`, which can be older than the last committed row (a crash past the last save):
        # the file keeps the rows through `start` and drops the ones this attempt re-emits; a fresh attempt starts it
        # empty (an append-only file held step 3 twice after such a relaunch).
        jsonl_keep_through=start if resumed else -1,
    )
    failures: set[int] = set()
    start_best_effort(reporters, failed_reporters=failures)
    watch = TerminationWatch()
    transaction = ValidationTransaction(policy, checkpoints, reporters, failures)
    boundary = ValidationBoundaryCoordinator(
        schedule=ValidationSchedule.from_config(rl.validation),
        total_steps=rl.max_steps,
        transaction=transaction,
        evaluator=partial(evaluate, units, rl),
        should_stop=watch.should_stop,
        publication_step=transaction.pending_publication_step,
    )
    final_step, final_loss = start, None
    # The clip counters continue from the kept rows of the metrics file (rank 0, the rank that writes it), so the
    # fraction below is the share of the RUN's updates, not of this attempt's.
    clipped_steps, counted_steps = prior_clip_counts(reporters)
    try:
        initial = boundary.resume_or_initial(start, resumed_from_checkpoint=resumed)
        if initial.preempted:
            return {"final_step": start, "preempted": initial.preempted}
        for step in range(start, rl.max_steps):
            began = time.monotonic()
            metrics = train_step(data, policy, rl)
            final_step = step + 1
            if int(resolve(policy.iteration())) != final_step:
                raise RuntimeError("SFT loop and trainer optimizer clocks diverged")
            final_loss = float(metrics["train/loss_avg"])
            if not math.isfinite(final_loss):
                raise RuntimeError("SFT applied a non-finite update")
            metrics["timer/e2e_step_s"] = time.monotonic() - began
            if "train/grad_clipped" in metrics:
                # The share of this run's updates the clip has set the step size of, so far, across attempts.
                counted_steps += 1
                clipped_steps += int(metrics["train/grad_clipped"] > 0)
                metrics["train/grad_clip_fraction"] = clipped_steps / counted_steps
            frame = reportable_metrics(metrics)
            save_due = bool(rl.save_every and (final_step % rl.save_every == 0 or final_step == rl.max_steps))
            outcome = boundary.at_step(final_step, frame, scheduled_checkpoint_due=save_due)
            if outcome.report is None and outcome.preempted is None:
                emit_metric_frame(reporters, frame, final_step, failed_reporters=failures)
            if outcome.preempted:
                return {"final_step": final_step, "preempted": outcome.preempted}
            if save_due:
                checkpoints.save_configured_if_needed(final_step)
            stopped = watch.should_stop()
            if stopped:
                # rl.emergency_checkpoint_components: "full" upgrades to the exact-resume plan; "run" writes
                # the run's own cadence plan plus the model -- see RunConfig for the trade.
                emergency_plan = str(rl.emergency_checkpoint_components)
                if checkpoints.save_emergency(final_step, components=emergency_plan):
                    info0(f"termination trigger ({stopped}) — emergency checkpoint ({emergency_plan} plan)")
                else:
                    info0(f"termination trigger ({stopped}) — already checkpointed, exiting")
                return {"final_step": final_step, "preempted": stopped}
    finally:
        finish_best_effort(reporters, failed_reporters=failures)
    return {"start": start, "final_step": final_step, "final_loss": final_loss}


def run(config: Any, config_path: str, opts: list[str], run_out_dir: str) -> dict[str, Any]:
    from cosmos_framework.posttrain.control.bind import bind_check
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.control.recipe import sft_manifest
    from cosmos_framework.posttrain.roles import roles
    from cosmos_framework.posttrain.utils.dist import destroy_process_group_if_initialized, init_local_process_group
    from cosmos_framework.posttrain.utils.entry import pin_run_attempt

    validate_config(config)
    specs = role_specs(config)
    bind_check(sft_manifest(specs)).raise_for_status()
    units: dict[str, Any] = {}
    init_local_process_group()
    try:
        attempt = pin_run_attempt(config, opts, run_out_dir)
        dump_resolved(config, attempt.out_dir, provenance=True)
        units = {name: getattr(roles, kind)(name, ranks=1) for name, kind in specs}
        group = ActorGroup(members=tuple(units.values()), setup=make_setup(config_path, list(attempt.opts)))
        # torchrun owns the CUDA process group; local placement does not allocate another mesh.
        bind_device_groups([DeviceGroup(actor_groups=(group,), device="cpu")], backend="local")
        summary = loop(units, config, attempt.out_dir)
        info0(f"SFT DONE {summary}")
        return summary
    finally:
        CheckpointLifecycle.finalize_units(units, destroy_process_group_if_initialized)


def main() -> None:
    config_path, opts, flags = cli_flags(
        "Cosmos3 offline multimodal SFT",
        "cosmos_framework/posttrain/configs/sft_qwen3p8_27b_multimodal.py",
        extra_flags={
            "--validate-rows": (
                "CPU-only: run every training/held-out row through the real packer (length vs rl.max_length, "
                "loss tokens vs an independent assistant-token count, media tokens, role grammar) and exit 0/1 "
                "before any trainer is built; the per-row JSONL lands in the job output dir. Under torchrun the "
                "training row count is also checked against WORLD_SIZE (the DP world the setup refuses on), and EVERY "
                "rank repeats the validation and writes its own report, so run it on one rank; outside torchrun the "
                "train gate line says 'dp_world not checked' (pass --dp-world to scripts/validate_sft_rows.py for an "
                "explicit world)."
            )
        },
    )
    from cosmos_framework.utils.config import load_config

    config = load_config(config_path, ["--", *opts])
    if flags["validate_manifest"] or flags["validate_rows"]:
        ok = True
        if flags["validate_manifest"]:
            from cosmos_framework.posttrain.control.recipe import sft_manifest

            validate_config(config)
            rendered, ok = render_manifest_validation(sft_manifest(role_specs(config)))
            print(rendered)
        if ok and flags["validate_rows"]:
            from cosmos_framework.posttrain.scripts.validate_sft_rows import validate_from_config

            # The DP world the setup will see: torchrun exports WORLD_SIZE (the seat launches through
            # utils/launch_multinode_torchrun.sh); outside torchrun the divisibility is NOT checked and the train gate
            # line says so. A fresh timestamped report under the job output dir; nothing is overwritten.
            world = os.environ.get("WORLD_SIZE", "")
            dp_world = int(world) if world.isdigit() and int(world) > 0 else None
            ok = validate_from_config(config, report_dir=out_dir(config), dp_world=dp_world) == 0
        raise SystemExit(0 if ok else 1)
    run(config, config_path, opts, out_dir(config))


if __name__ == "__main__":
    main()
