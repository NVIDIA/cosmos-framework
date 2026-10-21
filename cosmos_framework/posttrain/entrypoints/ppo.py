# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synchronous PPO assembled from the shared posttrain roles and substrate.

PPO adds a frozen reference role and a trainable critic role. Data movement, replay,
placement, rollout execution, reward evaluation, training, and weight synchronization
reuse the same abstractions as GRPO and OPD. The entrypoint owns only role assembly and
the algorithm's imperative phase ordering.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from cosmos_framework.posttrain.utils.entry import cli, dump_resolved, resolve_all, resolve_identical
from cosmos_framework.posttrain.utils.entry import out_dir as _out_dir
from cosmos_framework.posttrain.utils.log import info0, logger

if TYPE_CHECKING:
    from cosmos_framework.posttrain.control.placement.gpu_pair import WeightRelationship

ROLE_SPECS: tuple[tuple[str, str], ...] = (
    ("data", "env"),
    ("rollout", "rollout"),
    ("rm", "reward"),
    ("replay", "buffer"),
    ("reference", "reference"),
    ("critic", "critic"),
    ("policy", "trainer"),
)
_MODEL_CONFIG_FIELDS: dict[str, str] = {
    "policy": "model_hf",
    "reference": "reference_model_hf",
    "critic": "critic_model_hf",
}


def _model_config_field(role: str) -> str:
    """Return the HF config field owned by one PPO model role."""
    try:
        return _MODEL_CONFIG_FIELDS[role]
    except KeyError as error:
        raise ValueError(f"unsupported PPO model role {role!r}") from error


def manifest(config: Any, *, weight_relationship: WeightRelationship) -> Any:
    """Return the inspectable PPO role graph consumed by the shared bind check."""
    from cosmos_framework.posttrain.algorithm.reward import RULE_REWARDS
    from cosmos_framework.posttrain.control.recipe import ppo_manifest

    data = config.rl.data
    built_in_arithmetic = not (data.name or data.path or data.factory)
    # A bound shared rule declares which observation fields it reads, so read the requirement off
    # the rule rather than off one hard-coded name: a rule needing "reference" must select the
    # ground-truth reward capability, or the GPU-less preflight passes a run that cannot score.
    terminal = str(config.rl.reward.terminal)
    rule = RULE_REWARDS.get(terminal)
    needs_reference = "reference" in rule.needs if rule is not None else terminal == "arithmetic_exact_match"
    return ppo_manifest(
        list(ROLE_SPECS),
        ground_truth=built_in_arithmetic or bool(data.response_column),
        reward_needs_ground_truth=needs_reference,
        weight_relationship=weight_relationship,
    )


def _role_hf_config(config: Any, role: str) -> Any:
    """Resolve one HF role config with rollout identity as the default."""
    field = _model_config_field(role)
    model_config = copy.deepcopy(getattr(config, field))
    if not model_config.name_or_path:
        model_config.name_or_path = config.rollout.name_or_path
    if model_config.revision is None:
        model_config.revision = config.rollout.revision
    if model_config.processor_revision is None:
        model_config.processor_revision = config.rollout.processor_revision
    return model_config


def _model_parallelism(config: Any, role: str, ranks: int) -> tuple[int, int]:
    """Resolve one model role's data-parallel factorization."""
    from cosmos_framework.posttrain.config.build import (
        normalize_model_backend,
        resolve_hf_model_parallelism,
        resolve_policy_parallelism,
    )

    if normalize_model_backend(str(config.model_backend)) == "i4":
        return resolve_policy_parallelism(config, ranks) if role == "policy" else (1, ranks)
    field = _model_config_field(role)
    return resolve_hf_model_parallelism(
        getattr(config, field),
        ranks,
        field_name=f"config.{field}.parallelism",
        ranks_field_name=f"grid.{role}_shape",
    )


def _model_source_identity(config: Any, role: str) -> dict[str, Any]:
    """Return the immutable artifact/build identity for one PPO model role."""
    from cosmos_framework.posttrain.config.build import native_policy_build_spec, normalize_model_backend

    backend = normalize_model_backend(str(config.model_backend))
    if backend == "hf":
        model_config = _role_hf_config(config, role)
        return {
            "backend": backend,
            "name_or_path": str(model_config.name_or_path),
            "revision": model_config.revision,
        }
    spec = native_policy_build_spec(config)
    if spec is None:
        raise ValueError("PPO native model source is not configured")
    builder_field = "value_builder" if role == "critic" else "builder"
    return {
        "backend": backend,
        "model_id": str(spec.model_id),
        "builder": str(getattr(spec, builder_field)),
    }


def _checkpoint_identity(config: Any, rl: Any, role: str) -> dict[str, Any]:
    """Bind immutable model and update-plan choices into shared trainer checkpoints."""
    identity: dict[str, Any] = {
        "role": role,
        "model": _model_source_identity(config, role),
        "update_plan": {
            "epochs": int(rl.ppo_epochs),
            "num_minibatches": int(rl.num_minibatches),
            "microbatch_size": int(rl.microbatch_size),
            "seed": int(rl.seed),
        },
    }
    if role == "policy":
        identity["reference"] = _model_source_identity(config, "reference")
        identity["rollout"] = {
            "name_or_path": str(config.rollout.name_or_path),
            "revision": config.rollout.revision,
            "processor_revision": config.rollout.processor_revision,
        }
        identity["data_packer_factory"] = str(getattr(config.data_packer, "factory", ""))
    return identity


def declare_wgs(config: Any) -> dict[str, Any]:
    """Declare the seven PPO roles through the shared role factory."""
    from cosmos_framework.posttrain.roles import roles

    grid = config.grid
    policy_replicas, policy_ranks = grid.policy_shape()
    rollout_replicas, rollout_ranks = grid.rollout_shape()
    groups: dict[str, Any] = {
        "data": roles.env("data", replicas=1, ranks=1),
        "rollout": roles.rollout(
            "rollout",
            replicas=rollout_replicas,
            ranks=rollout_ranks,
        ),
        "rm": roles.reward("rm", replicas=rollout_replicas, ranks=1, needs_bulk=True),
        "replay": roles.buffer("replay", replicas=1, ranks=1),
    }
    for role, factory, ranks in (
        ("reference", roles.reference, int(grid.reference_ranks)),
        ("critic", roles.critic, int(grid.critic_ranks)),
        ("policy", roles.trainer, policy_ranks),
    ):
        groups[role] = factory(
            role,
            replicas=policy_replicas if role == "policy" else 1,
            ranks=ranks,
            model_parallelism=_model_parallelism(config, role, ranks),
        )
    return groups


def _build_model_role(config: Any, role: str) -> tuple[Any, Any]:
    """Build one PPO model role on the placement-injected model mesh."""
    from cosmos_framework.posttrain.config.build import (
        build_native_model,
        build_native_policy,
        build_native_value_model,
        normalize_model_backend,
    )
    from cosmos_framework.posttrain.model.hf import build_hf_model
    from cosmos_framework.posttrain.model.value import DenseValueModel
    from cosmos_framework.posttrain.workers.ray import current_pdims

    if role not in ("policy", "reference", "critic"):
        raise ValueError(f"unsupported PPO model role {role!r}")
    pdims = current_pdims()
    if normalize_model_backend(str(config.model_backend)) == "hf":
        model = build_hf_model(_role_hf_config(config, role), pdims)
        return (DenseValueModel(model, pdims=pdims) if role == "critic" else model), pdims
    builder = {
        "policy": build_native_policy,
        "reference": build_native_model,
        "critic": build_native_value_model,
    }[role]
    model, _model_config = builder(config, pdims)
    return model, pdims


def build_policy(config: Any, rl: Any) -> Any:
    """Build the policy model and inject the PPO policy objective."""
    from cosmos_framework.posttrain.config.build import (
        build_lr_scheduler,
        build_optimizer,
    )
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.roles.trainer import PPOPolicyTrainer

    model, pdims = _build_model_role(config, "policy")
    optimizer = build_optimizer(model, rl.optimizer)
    scheduler = build_lr_scheduler(optimizer, rl.optimizer)
    checkpoint_manager = CheckpointLifecycle.from_config(config, rl).manager
    return PPOPolicyTrainer(
        model,
        optimizer,
        scheduler=scheduler,
        temperature=float(config.rollout.temperature),
        max_grad_norm=float(rl.optimizer.grad_clip_norm),
        force_finite_grads=bool(rl.optimizer.force_finite_grads),
        policy_clip=float(rl.policy_clip),
        entropy_coef=float(rl.entropy_coef),
        jsd_coef=float(rl.jsd_coef),
        jsd_beta=float(rl.jsd_beta),
        gamma=float(rl.gamma),
        gae_lambda=float(rl.gae_lambda),
        reference_kl_coef=float(rl.reference_kl_coef),
        checkpoint_identity=_checkpoint_identity(config, rl, "policy"),
        parallel_dims=pdims,
        checkpoint_manager=checkpoint_manager,
    )


def build_reference(config: Any, rl: Any) -> Any:
    """Build the frozen reference model."""
    from cosmos_framework.posttrain.roles.reference import PPOReferenceRole

    model, _pdims = _build_model_role(config, "reference")
    return PPOReferenceRole(
        model,
        top_k=int(rl.jsd_top_k),
        temperature=float(config.rollout.temperature),
    )


def build_critic(config: Any, rl: Any) -> Any:
    """Build the value model and inject the PPO value objective."""
    from cosmos_framework.posttrain.config.build import (
        build_lr_scheduler,
        build_optimizer,
    )
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.roles.critic import PPOCriticRole

    model, pdims = _build_model_role(config, "critic")
    optimizer = build_optimizer(model, rl.critic_optimizer)
    scheduler = build_lr_scheduler(optimizer, rl.critic_optimizer)
    checkpoint_manager = CheckpointLifecycle.from_config(config, rl).manager
    return PPOCriticRole(
        model,
        optimizer,
        scheduler=scheduler,
        max_grad_norm=float(rl.critic_optimizer.grad_clip_norm),
        force_finite_grads=bool(rl.critic_optimizer.force_finite_grads),
        value_clip=float(rl.value_clip),
        value_coef=float(rl.value_coef),
        checkpoint_identity=_checkpoint_identity(config, rl, "critic"),
        parallel_dims=pdims,
        checkpoint_manager=checkpoint_manager,
    )


def build_rollout(config: Any, rl: Any) -> Any:
    """Build the PPO vLLM rollout through the shared rollout construction seam."""
    import torch

    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.config.build import build_ppo_vllm_rollout, policy_data_packer_build_spec
    from cosmos_framework.posttrain.config.model import DataPackerConfig
    from cosmos_framework.posttrain.utils.dist import run_rank_converged

    rollout_config = instantiate(config.rollout)
    if rollout_config.tensor_parallel_size > 1:
        if not torch.distributed.is_initialized():
            raise RuntimeError("PPO rollout TP requires a per-replica process group")
        if torch.distributed.get_world_size() != rollout_config.tensor_parallel_size:
            raise RuntimeError("PPO rollout process-group world size must equal rollout.tensor_parallel_size")

    packer_spec = policy_data_packer_build_spec(
        config,
        instantiate(getattr(config, "data_packer", DataPackerConfig())),
    )
    packer = run_rank_converged(packer_spec.build, "PPO rollout DataPacker setup")
    return build_ppo_vllm_rollout(
        rollout_config,
        packer,
        behavior_top_k=int(rl.jsd_top_k),
    )


def build_data(rl: Any) -> Any:
    """Build the PPO prompt source through the shared DatasetEnv."""
    from cosmos_framework.posttrain.roles.env import (
        DatasetEnv,
        DeterministicArithmeticDataset,
        JsonlRows,
        project_prompt_items,
    )

    data_config = rl.data
    modality = str(rl.modality)
    modality_tags = ("text",) if modality == "text" else ("text", modality)
    if not (data_config.name or data_config.path or data_config.factory):
        dataset: Any = DeterministicArithmeticDataset(
            seed=int(rl.seed),
            length=max(1, int(rl.max_steps) * int(rl.prompts_per_iteration)),
            max_operand=int(rl.max_operand),
            modality=modality,
            image_size=int(rl.image_size),
            video_frames=int(rl.video_frames),
            video_fps=float(rl.video_fps),
            fingerprint_prefix="ppo-arithmetic",
        )
    elif data_config.factory:
        from cosmos_framework.posttrain.data.datasets.factory import build_custom_dataset

        dataset = build_custom_dataset(str(data_config.factory), dict(data_config.factory_kwargs))
    elif data_config.name:
        from datasets import load_dataset

        dataset_kwargs: dict[str, Any] = {"split": str(data_config.split)}
        if data_config.subset:
            dataset_kwargs["name"] = str(data_config.subset)
        if data_config.revision:
            dataset_kwargs["revision"] = str(data_config.revision)
        dataset = load_dataset(str(data_config.name), **dataset_kwargs)
    else:
        # A local path is read as written -- one JSON object per line or any whitespace-separated stream of
        # objects (pretty-printed or not), one JSON array, or the same gzip-compressed, gzip only (`JsonlRows`,
        # the GRPO loader's reader; the shapes the Hugging Face reader accepted) -- not
        # through ``datasets``: Arrow infers one struct type over every content block of a conversation
        # column and null-fills the keys a block lacks (a text block gains ``image: None``), which the Qwen
        # chat template (``'image' in content``) renders as an image placeholder with the text gone; a nested
        # reference column gains null keys the same way; and a column whose turns mix string and list content
        # is refused outright. `JsonlRows` carries the content fingerprint `project_prompt_items` needs for
        # exact resume.
        dataset = JsonlRows(str(data_config.path), fingerprint_prefix="ppo-jsonl")
    if data_config.name or data_config.path or data_config.factory:
        row_map = None
        if data_config.items_fn:
            from cosmos_framework.utils.serialization import load_callable

            row_map = load_callable(str(data_config.items_fn))
        dataset = project_prompt_items(
            dataset,
            source_name=str(data_config.name or data_config.path or data_config.factory),
            revision=str(data_config.revision),
            subset=str(data_config.subset),
            split=str(data_config.split),
            prompt_column=str(data_config.prompt_column),
            response_column=str(data_config.response_column),
            image_column=str(data_config.image_column),
            video_column=str(data_config.video_column),
            id_column=str(data_config.id_column),
            metadata_columns=tuple(data_config.metadata_columns),
            seed=int(rl.seed),
            modality=modality,
            video_fps=float(rl.video_fps),
            row_map=row_map,
            row_map_identity=str(data_config.items_fn),
            fingerprint_prefix="ppo-prompt",
        )
    return DatasetEnv(
        dataset,
        shuffle=bool(data_config.shuffle),
        seed=int(rl.seed),
        repeat=True,
        modality_tags=modality_tags,
    )


def build_reward(rl: Any) -> Any:
    """Compose the PPO reward adapter over the shared RewardWorker."""
    from cosmos_framework.posttrain.algorithm.ppo.reward import (
        PPO_REWARD_PROVIDER_NAMES,
        _enrich_ppo_reward_request,
        build_ppo_reward_catalog,
    )
    from cosmos_framework.posttrain.roles.reward import PPORewardRole, RewardWorker
    from cosmos_framework.posttrain.roles.reward.worker import _with_trusted_request_enricher

    catalog = build_ppo_reward_catalog(rl.reward)
    worker = RewardWorker(
        [catalog.build(name) for name in PPO_REWARD_PROVIDER_NAMES],
        training_group_size=1,
    )
    return PPORewardRole(_with_trusted_request_enricher(worker, _enrich_ppo_reward_request))


def make_setup(config_path: str, opts: list[str], role: str) -> Any:
    """Create one picklable setup closure for a declared PPO role."""
    if role not in {name for name, _kind in ROLE_SPECS}:
        raise ValueError(f"unknown PPO role {role!r}")

    def _setup() -> dict[str, Any]:
        from cosmos_framework.utils.config import load_config
        from cosmos_framework.utils.lazy_config import instantiate
        from cosmos_framework.posttrain.control import ExperienceBuffer
        from cosmos_framework.posttrain.utils.determinism import enable_determinism

        config = load_config(config_path, ["--", *opts])
        rl = instantiate(config.rl)
        if rl.deterministic:
            enable_determinism()
        builders = {
            "data": lambda: build_data(rl),
            "rollout": lambda: build_rollout(config, rl),
            "rm": lambda: build_reward(rl),
            "replay": lambda: ExperienceBuffer(exact_batch_size=int(rl.prompts_per_iteration)),
            "reference": lambda: build_reference(config, rl),
            "critic": lambda: build_critic(config, rl),
            "policy": lambda: build_policy(config, rl),
        }
        body = builders[role]()
        info0(f"PPO setup done: role={role}")
        return {role: body}

    return _setup


def _actor_groups(
    groups: Mapping[str, Any],
    config_path: str,
    opts: list[str],
) -> dict[str, Any]:
    """Create one process-isolated ActorGroup per role."""
    from cosmos_framework.posttrain.control.placement import ActorGroup

    return {
        role: ActorGroup((groups[role],), make_setup(config_path, opts, role), name=f"ppo_{role}")
        for role, _kind in ROLE_SPECS
    }


def _device_groups(actors: Mapping[str, Any], gpu_placement: str) -> list[Any]:
    """Lower the PPO roles through the shared DeviceGroup abstraction."""
    from cosmos_framework.posttrain.control.placement import DeviceGroup, trainer_experience_device_groups

    device_groups: list[Any] = [
        DeviceGroup(
            (actors["data"], actors["rm"], actors["replay"]),
            device="cpu",
            name="ppo_control",
        )
    ]
    device_groups.extend(
        trainer_experience_device_groups(
            actors["rollout"],
            actors["policy"],
            gpu_placement=gpu_placement,
            name="ppo",
        )
    )
    device_groups.extend(
        DeviceGroup((actors[role],), device="gpu", name=f"ppo_{role}") for role in ("reference", "critic")
    )
    return device_groups


def loop(units: Mapping[str, Any], config: Any, out_dir: str) -> dict[str, Any]:
    """Run the PPO role interaction sequence as plain imperative Python."""
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.algorithm.ppo.batch import make_ppo_update_plan
    from cosmos_framework.posttrain.algorithm.ppo.records import merge_ppo_step_metrics
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.control.loop.batching import resolve_prepared_batch
    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync
    from cosmos_framework.posttrain.control.termination import TerminationWatch
    from cosmos_framework.posttrain.utils.report import (
        emit_metric_frame,
        finish_best_effort,
        make_reporters,
        reportable_metrics,
        start_best_effort,
    )

    rl = instantiate(config.rl)
    data, rollout, reward, replay, reference, critic, policy = (units[name] for name, _kind in ROLE_SPECS)
    reporters = make_reporters(
        config,
        out_dir,
        total_steps=int(rl.max_steps),
        headline_keys=("train/policy_total_loss", "train/critic_loss", "train/reward_mean"),
    )
    failed_reporters: set[int] = set()
    start_best_effort(reporters, failed_reporters=failed_reporters)
    watch = TerminationWatch()
    # Two clocks, deliberately separate. `iteration` counts COLLECTED outer iterations and is the
    # loop's durable step (checkpoint boundary, metric step); `policy_version` counts PUBLISHED
    # actor updates and is the rollout's staleness contract. They coincide only when no critic-only
    # warmup is configured, which is why the loop cannot keep using one variable for both: during
    # warmup the actor must not publish, so a shared counter would either advertise weights that
    # never changed or silently reuse one checkpoint boundary for several iterations.
    warmup = int(rl.critic_warmup_iterations)
    # The configured warmup travels WITH the boundary. It cannot be recovered from the actor's
    # publication clock, which is pinned at 0 for the whole warmup phase and therefore reads exactly
    # the same under every warmup at least as long as the boundary's iteration count -- so a run
    # resumed with a lengthened warmup would keep training under a schedule it did not start with,
    # its first actor update silently sliding later, with nothing to report.
    loop_state = {"critic_warmup_iterations": warmup}
    iteration = 0
    policy_version = 0
    checkpoint_lifecycle = CheckpointLifecycle.from_config(config, rl, run_out_dir=out_dir)
    checkpoints = checkpoint_lifecycle.multi_role_coordinator(
        {"policy": policy, "critic": critic},
        data=data,
    )
    if checkpoints.restore(rl.resume) is not None:
        iteration = checkpoints.committed_step()
        policy_version = int(resolve_identical(policy.publication_version(), operation="PPO policy version restore"))
        committed_warmup = checkpoints.committed_loop_state().get("critic_warmup_iterations")
        if committed_warmup is None:
            # Boundaries committed before the marker carried loop state recorded no warmup. Absence
            # is UNKNOWN, not a mismatch: such a boundary keeps exactly the guard it shipped with --
            # the derived cross-check below and nothing more -- rather than failing closed and making
            # every existing run unresumable. What that costs is stated where it applies: the derived
            # check cannot tell two warmups apart once both exceed the boundary's iteration count.
            if warmup >= iteration:
                logger.warning(
                    "PPO boundary at iteration %d predates the recorded warmup length, so the "
                    "configured rl.critic_warmup_iterations=%d cannot be verified against it",
                    iteration,
                    warmup,
                )
        elif committed_warmup != warmup:
            raise RuntimeError(
                f"PPO resume disagrees with rl.critic_warmup_iterations: the boundary at iteration "
                f"{iteration} was committed with critic_warmup_iterations={committed_warmup!r}, but "
                f"this run is configured with {warmup}"
            )
        # Secondary and independent: the recorded warmup pins the SCHEDULE, this pins the actor's own
        # counter against it, so a restored actor whose publication clock does not match the boundary
        # it came from is still rejected.
        expected_version = max(0, iteration - warmup)
        if policy_version != expected_version:
            raise RuntimeError(
                f"PPO resume disagrees with rl.critic_warmup_iterations={warmup}: boundary iteration "
                f"{iteration} implies actor policy version {expected_version}, but the restored actor "
                f"reports {policy_version}"
            )
    final_metrics: dict[str, float] | None = None
    preempted: str | None = None
    try:
        weight_sync(rollout, policy, validation_step=policy_version)
        for outer_iteration in range(iteration, int(rl.max_steps)):
            prompts = data.get_samples(int(rl.prompts_per_iteration))
            trajectories = rollout.generate(prompts, policy_version=policy_version)
            scored = reward.score(trajectories)
            resolve_all(replay.put_samples(scored))
            batch = replay.get_exact_samples(int(rl.prompts_per_iteration))
            referenced = reference.score(batch)
            valued = critic.predict(referenced)
            targets = policy.prepare_batch(valued)
            prepared = resolve_prepared_batch(policy, targets, operation="PPO batch preparation")

            succeeded = False
            try:
                resolve_identical(
                    critic.install_targets(targets),
                    operation="PPO critic target installation",
                )
                on_policy = resolve_identical(
                    policy.score_cached_on_policy_metrics(prepared.batch_id),
                    operation="PPO on-policy validation",
                )
                # ONE plan for both roles, seeded by the OUTER iteration: the critic's schedule must
                # keep shuffling while the actor is frozen, and both roles pinned this plan's
                # epoch/minibatch shape into their checkpoint identity, so warmup may not vary it.
                plan = make_ppo_update_plan(rl, outer_iteration)
                policy_update = (
                    None
                    if outer_iteration < warmup
                    else resolve_identical(
                        policy.train_batch(prepared.batch_id, plan),
                        operation="PPO policy update",
                    )
                )
                critic_update = resolve_identical(
                    critic.train_batch(prepared.batch_id, plan),
                    operation="PPO critic update",
                )
                if policy_update is not None:
                    policy_version += 1
                    resolve_all(policy.set_policy_version(policy_version))
                    # Skipped during warmup: the actor is byte-unchanged, so the rollout already
                    # advertises exactly `policy_version` and `generate`'s staleness guard is an
                    # equality check that already holds. Re-pushing identical weights would only add
                    # a full model transfer, and a transport failure, to an iteration with no delta.
                    weight_sync(rollout, policy, validation_step=policy_version)
                iteration = outer_iteration + 1
                if int(rl.save_every) > 0 and (iteration % int(rl.save_every) == 0 or iteration == int(rl.max_steps)):
                    checkpoints.save_configured_if_needed(iteration, loop_state)
                final_metrics = reportable_metrics(
                    merge_ppo_step_metrics(prepared, on_policy, policy_update, critic_update)
                )
                emit_metric_frame(
                    reporters,
                    final_metrics,
                    iteration,
                    failed_reporters=failed_reporters,
                )
                succeeded = True
            finally:
                cleanup = "clear_batch" if succeeded else "abort_batch"
                resolve_all(getattr(policy, cleanup)(prepared.batch_id))
                resolve_all(getattr(critic, cleanup)(prepared.batch_id))

            preempted = watch.should_stop()
            if preempted:
                if checkpoints.save_configured_if_needed(iteration, loop_state):
                    logger.warning("termination trigger (%s) — emergency checkpoint", preempted)
                logger.warning(
                    "termination trigger (%s) after PPO iteration %d",
                    preempted,
                    outer_iteration + 1,
                )
                break
    finally:
        finish_best_effort(reporters, failed_reporters=failed_reporters)
        CheckpointLifecycle.finalize_units({"policy": policy, "critic": critic})

    summary: dict[str, Any] = {
        "outer_iteration": iteration,
        "policy_version": policy_version,
        "final_metrics": final_metrics,
    }
    if preempted is not None:
        summary["preempted"] = preempted
    info0(f"DONE PPO {summary} resolved={out_dir}/config_resolved.yaml")
    return summary


def run(config: Any, config_path: str, opts: list[str], out_dir: str) -> dict[str, Any]:
    """Validate, bind shared placement, and execute the PPO loop."""
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.config.config_ppo import validate_ppo_config
    from cosmos_framework.posttrain.control.bind import bind_check
    from cosmos_framework.posttrain.control.placement import (
        bind_device_groups,
        resolve_online_weight_placement,
        validate_online_weight_transport,
    )

    validate_ppo_config(config)
    rl = instantiate(config.rl)
    backend, gpu_placement, weight_relationship = resolve_online_weight_placement(
        config.grid.backend,
        config.grid.gpu_placement,
    )
    validate_online_weight_transport(weight_relationship, rl.weight_sync_transport)
    report = bind_check(manifest(config, weight_relationship=weight_relationship))
    info0(report.format())
    report.raise_for_status()
    dump_resolved(config, out_dir)

    groups = declare_wgs(config)
    actors = _actor_groups(groups, config_path, opts)
    device_groups = _device_groups(actors, gpu_placement)
    bind_device_groups(
        device_groups,
        backend=backend,
        weight_sync=True,
        weight_sync_transport=rl.weight_sync_transport,
        gpu_node_domains=getattr(config.grid, "gpu_node_domains", None),
    )
    return loop(groups, config, out_dir)


def main() -> None:
    from cosmos_framework.utils.config import load_config
    from cosmos_framework.posttrain.config.config_ppo import validate_ppo_config

    config_path, opts, validate = cli(
        "Cosmos3 synchronous PPO",
        default_config="cosmos_framework/posttrain/config/config_ppo.py",
    )
    config = load_config(config_path, ["--", *opts])
    if validate:
        validate_ppo_config(config)
        from cosmos_framework.posttrain.control.bind import bind_check
        from cosmos_framework.posttrain.control.placement import resolve_online_weight_placement

        _backend, _gpu_placement, weight_relationship = resolve_online_weight_placement(
            config.grid.backend,
            config.grid.gpu_placement,
        )
        report = bind_check(manifest(config, weight_relationship=weight_relationship))
        info0(report.format())
        report.raise_for_status()
        declare_wgs(config)
        info0("PPO config and role declarations are valid")
        return
    run(config, config_path, opts, _out_dir(config))


if __name__ == "__main__":
    main()
