# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Cosmos3 on-policy distillation with an independently ranked vLLM teacher.

**Deliberately thin: this entry reuses `entrypoints/grpo.py` rather than forking it.** OPD is GRPO
with a per-token teacher signal added to the learning objective, so the experience cycle — generate,
score, replay, train, weight-sync — is the same cycle. What differs is one extra role and one extra
objective, so that is all this module adds; `grpo.loop` takes an optional `"teacher"` unit and calls
it between the reward stage and the replay put. A copied loop would stop tracking the original the
first time someone fixes a dispatch bug in it.

    config.rl        OPDConfig      — GRPO's knobs + teacher signal (config/rl.py)
    config.rl.teacher TeacherConfig — the frozen teacher engine in its own WorkerGroup
    config.rollout   RolloutConfig  — the student's vLLM engine + sampling

Topology: GRPO's experience and policy groups plus an independent teacher WorkerGroup. The teacher
owns one process per TP lane and its own per-replica rendezvous, so rollout TP and teacher TP are
independent. The placement binder owns unequal-product routing and the disaggregated NIXL transport;
this entry only declares the three role groups and their configured shapes.

No weight-sync edge for the teacher: it is frozen. That is the whole reason it is cheaper to place
than the rollout despite being the same kind of engine.

Held-out validation remains GRPO's rollout/reward evaluation of the student. It deliberately skips
the teacher: OPD adds no teacher-signal validation objective or metric contract in this entry.

    python -m cosmos_framework.posttrain.entrypoints.opd -- \\
        rl.teacher.name_or_path=Qwen/Qwen2.5-0.5B-Instruct rl.max_steps=20

Alignment note: this entry's `distillation_mode=advantage`, `jsd_beta=1`, `kl_penalty_coef=1`, and
`top_k=32` settings match the cosmos-rl reference configuration. Its `loss_type=token-sum` has no equivalent here — use
`rl.loss_type=seq-mean-token-sum`, which reproduces that update up to the constant `mini_batch`
(see the design doc).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from cosmos_framework.posttrain.config.rl import OPDConfig
from cosmos_framework.posttrain.control import capabilities_builtin  # noqa: F401  (registers descriptors)
from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
from cosmos_framework.posttrain.control.loop.batching import prompts_per_data_cell, validate_batch_divisibility
from cosmos_framework.posttrain.control.manifest import Edge, Manifest, Plane, RoleDecl
from cosmos_framework.posttrain.control.placement import (
    resolve_online_weight_placement,
    validate_online_weight_transport,
)
from cosmos_framework.posttrain.control.placement.gpu_pair import WeightRelationship
from cosmos_framework.posttrain.entrypoints.grpo import (
    _build_reward_annotators,
    _data_pool_config,
    build_env,
    build_rollout,
    build_trainer,
    loop,
)
from cosmos_framework.posttrain.roles.env.prompt import prompt_source
from cosmos_framework.posttrain.utils.dist import world_size_or_1
from cosmos_framework.posttrain.utils.entry import cli, dump_resolved, render_manifest_validation
from cosmos_framework.posttrain.utils.entry import out_dir as _out_dir
from cosmos_framework.posttrain.utils.log import info0, logger

if TYPE_CHECKING:
    from cosmos_framework.posttrain.algorithm.reward import RewardProviderCatalog

ROLE_SPECS: list[tuple[str, str]] = [
    ("data", "env"),
    ("rollout", "rollout"),
    ("rm", "reward"),
    ("teacher", "teacher"),
    ("replay", "buffer"),
    ("policy", "trainer"),
]
OPD_EXPERIENCE: list[str] = ["data", "rollout", "rm", "replay"]
OPD_TEACHER: list[str] = ["teacher"]
_MODEL_BEARING: tuple[str, str] = ("rollout", "policy")


def _teacher_needs_bulk(input_mode: str) -> bool:
    """Return whether the selected teacher mode consumes media/tensor bulk."""
    if input_mode in ("shared_vocab_multimodal", "cross_token_multimodal"):
        return True
    if input_mode == "cross_token_text":
        return False
    raise ValueError(
        "teacher.input_mode must be shared_vocab_multimodal|cross_token_text|"
        f"cross_token_multimodal, got {input_mode!r}"
    )


def manifest(
    rl: OPDConfig,
    *,
    weight_relationship: WeightRelationship,
    catalog: RewardProviderCatalog | None = None,
) -> Manifest:
    """GRPO's recipe topology plus the teacher, for `bind_check`.

    Built here rather than extended from `recipe.grpo_manifest` because the edge set genuinely
    differs: the teacher sits BETWEEN reward and replay (`rm -> teacher -> replay` replaces
    `rm -> replay`). Its per-token signal remains part of the trajectory payload on the DATA plane;
    algorithmic credit assignment is distinct from transport flow-control credits.
    """
    from cosmos_framework.posttrain.algorithm.reward import resolve_reward_provider_catalog

    if catalog is None:
        catalog = resolve_reward_provider_catalog(str(getattr(rl, "reward_provider_catalog", "")))
    source, ground_truth = prompt_source(rl)
    env_key = f"grpo_env_{source}"
    if source != "arithmetic" and ground_truth:
        env_key += "_with_answers"
    impl_keys = {
        "env": env_key,
        "rollout": "grpo_rollout_with_answers" if ground_truth else "grpo_rollout",
        "reward": capabilities_builtin.grpo_reward_capability_key(catalog.spec(str(rl.reward)).needs),
        "teacher": "opd_teacher_vllm",
        "buffer": "opd_buffer",
        "trainer": "opd_trainer",
    }
    return Manifest(
        roles=tuple(RoleDecl(name=name, impl_key=impl_keys.get(kind, f"grpo_{kind}")) for name, kind in ROLE_SPECS),
        edges=(
            Edge("data", "rollout", Plane.DATA),
            Edge("rollout", "rm", Plane.DATA),
            # The teacher returns the same scored trajectories with TeacherSignal attached, so both
            # edges carry trajectory payloads on the data plane. Transport credits are flow control.
            Edge("rm", "teacher", Plane.DATA),
            Edge("teacher", "replay", Plane.DATA),
            Edge("replay", "policy", Plane.DATA),
            Edge(
                "policy",
                "rollout",
                Plane.WEIGHT,
                relationship=weight_relationship,
                transport_capability="tensor",
            ),
            # No policy->teacher weight edge: the teacher is frozen.
        ),
    )


def declare_wgs(
    ranks: int,
    replicas: int = 1,
    rm_needs_bulk: bool = False,
    *,
    shapes: dict[str, tuple[int, int]] | None = None,
    policy_parallelism: tuple[int, int] | None = None,
    teacher_ranks: int = 1,
    teacher_needs_bulk: bool = True,
) -> dict[str, Any]:
    """Declare GRPO roles plus one independently ranked teacher replica.

    ``teacher_ranks`` is its vLLM TP width; its replica count is fixed at one. Both multimodal modes
    materialize media and token bulk, while ``cross_token_text`` receives exact completion IDs/text
    as inline metadata and must leave media handles untouched.
    """
    from cosmos_framework.posttrain.roles import roles as role_factory

    if isinstance(teacher_ranks, bool) or not isinstance(teacher_ranks, int) or teacher_ranks < 1:
        raise ValueError(f"teacher_ranks must be a non-bool positive integer, got {teacher_ranks!r}")

    def _shape(name: str) -> tuple[int, int]:
        if name == "teacher":
            return 1, teacher_ranks
        if shapes:
            if name in shapes:
                return shapes[name]
            return shapes["rollout"][0], 1
        if name == "policy":
            return 1, ranks
        return replicas, ranks if name in _MODEL_BEARING else 1

    def _kwargs(name: str, kind: str) -> dict[str, Any]:
        role_replicas, role_ranks = _shape(name)
        kw: dict[str, Any] = {"replicas": role_replicas, "ranks": role_ranks}
        if (kind == "reward" and rm_needs_bulk) or (kind == "teacher" and teacher_needs_bulk):
            kw["needs_bulk"] = True
        elif kind == "teacher":
            kw["needs_bulk"] = False
        if kind == "trainer":
            kw["model_parallelism"] = policy_parallelism
        return kw

    return {name: getattr(role_factory, kind)(name, **_kwargs(name, kind)) for name, kind in ROLE_SPECS}


def build_teacher(config: Any) -> Any:
    """Build the teacher inside its own already-rendezvoused actor group.

    Shared-vocabulary mode uses the student packer to attach media without changing exact rollout
    IDs. Cross-token modes receive the student packer as their immutable completion codec;
    ``build_vllm_teacher`` additionally owns the teacher packer for multimodal DPCA.
    """
    import torch

    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.config.build import (
        build_vllm_teacher,
        policy_data_packer_build_spec,
    )
    from cosmos_framework.posttrain.config.model import DataPackerConfig
    from cosmos_framework.posttrain.utils.dist import run_rank_converged

    rl = instantiate(config.rl)
    data_packer_config = instantiate(getattr(config, "data_packer", DataPackerConfig()))

    def prepare() -> Any:
        packer_spec = policy_data_packer_build_spec(config, data_packer_config)
        teacher_tp = int(rl.teacher.tensor_parallel_size)
        if teacher_tp > 1:
            if not torch.distributed.is_initialized():
                raise RuntimeError("teacher per-replica rendezvous not run (need ranks=teacher TP>1 on GPU)")
            world_size = int(torch.distributed.get_world_size())
            if world_size != teacher_tp:
                raise RuntimeError(f"teacher TP world size {world_size} != configured TP={teacher_tp}")
        return packer_spec.build()

    student_packer = run_rank_converged(prepare, "OPD teacher pre-engine preparation")
    return build_vllm_teacher(rl.teacher, student_packer, data_packer_config)


def build_objective(rl: OPDConfig, model_cfg: Any, packer: Any) -> Any:
    """Bind the OPD objective, in the shape `build_trainer`'s seam expects.

    Same signature as `grpo.build_objective` so the two are interchangeable, and thin delegation to
    `config/build.py::build_opd_objective` for the same reason GRPO delegates: one config->objective
    mapping, so a new knob cannot reach one route and miss the other.

    The trainer-side packer is REQUIRED here where GRPO's is optional — both OPD objectives collate
    a padded batch to gather the teacher's support, so they cannot run off stored ids alone.

    Args:
        rl: Resolved OPD algorithm config.
        model_cfg: Policy identity supplied by the shared trainer assembly.
        packer: Policy-side completion codec used by the OPD objective.

    Returns:
        The trainer-facing OPD objective.
    """
    from cosmos_framework.posttrain.config.build import build_opd_objective, resolve_pdims

    return build_opd_objective(rl, packer, pdims=resolve_pdims())


def make_setup(config_path: str, opts: list[str], roles: list[str]) -> Callable[[], dict[str, Any]]:
    """GRPO's setup plus the teacher; only strings cross into a Ray actor."""

    def _setup() -> dict[str, Any]:
        from cosmos_framework.utils.config import load_config
        from cosmos_framework.utils.lazy_config import instantiate
        from cosmos_framework.posttrain.algorithm.reward import RewardCombiner, resolve_reward_provider_catalog
        from cosmos_framework.posttrain.config.build import effective_model_id
        from cosmos_framework.posttrain.control import ExperienceBuffer
        from cosmos_framework.posttrain.control.validation.rollout import validate_reward_algorithm_compatibility
        from cosmos_framework.posttrain.roles.reward import RewardWorker
        from cosmos_framework.posttrain.utils.determinism import enable_determinism
        from cosmos_framework.posttrain.utils.dist import run_rank_converged

        config = load_config(config_path, ["--", *opts])
        rl = instantiate(config.rl)
        if rl.deterministic:
            # Ray actors are separate processes; the driver's deterministic setup does not reach
            # the model-bearing bodies built here.
            enable_determinism()
        out: dict[str, Any] = {}
        if "data" in roles:
            out["data"] = run_rank_converged(
                lambda: build_env(rl, config),
                "OPD data and validation source setup",
            )
        if "rollout" in roles:
            out["rollout"] = build_rollout(config)
        if "rm" in roles:
            catalog = resolve_reward_provider_catalog(str(getattr(rl, "reward_provider_catalog", "")))
            validate_reward_algorithm_compatibility(rl, catalog)
            training_providers = [catalog.build(str(rl.reward))]
            validation = getattr(rl, "validation", None)
            validation_reward = (
                str(getattr(validation, "reward", "") or rl.reward)
                if validation is not None and bool(getattr(validation, "enabled", False))
                else str(rl.reward)
            )
            validation_providers = [catalog.build(validation_reward)] if validation_reward != str(rl.reward) else None
            out["rm"] = RewardWorker(
                training_providers,
                RewardCombiner(),
                annotators=_build_reward_annotators(config, rl, effective_model_id(config)),
                validation_providers=validation_providers,
                training_group_size=rl.group_size,
            )
        if "teacher" in roles:
            out["teacher"] = build_teacher(config)
        if "replay" in roles:
            out["replay"] = ExperienceBuffer()
        if "policy" in roles:
            # THE seam that makes this a distillation run: `build_trainer` is GRPO's and assembles
            # everything algorithm-independent (mesh, optimizer, checkpoint resume), while this
            # entry must explicitly supply the OPD objective. There is no fallback objective, so an
            # entry that omits its algorithm choice fails during assembly instead of silently
            # training plain GRPO while the teacher signal goes unread.
            out["policy"] = build_trainer(
                config,
                rl,
                build_objective_fn=build_objective,
            )
        info0(f"setup done: roles={list(out)}")
        return out

    return _setup


def run(config: Any, config_path: str, opts: list[str], out_dir: str) -> None:
    """Launch experience, ranked teacher, and policy as three ActorGroups.

    Kept as its own function rather than a flag on `grpo.run` because the two differ in the parts
    `run` exists to express — the manifest, the role set and the actor/device grouping — while the part they
    share, the loop, is shared by reuse.
    """
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.algorithm.reward import resolve_reward_provider_catalog
    from cosmos_framework.posttrain.config.build import (
        prefix_filter_enabled,
        resolve_policy_parallelism,
        select_policy_backend,
        validate_opd_runtime,
    )
    from cosmos_framework.posttrain.config.model import DataPackerConfig
    from cosmos_framework.posttrain.control.bind import bind_check
    from cosmos_framework.posttrain.control.lifecycle import ControllerLifecycleObservers, build_controller_observers
    from cosmos_framework.posttrain.control.placement import (
        ActorGroup,
        DeviceGroup,
        bind_device_groups,
        placement_gpu_count,
        trainer_experience_device_groups,
    )
    from cosmos_framework.posttrain.control.validation.rollout import validate_rollout_reward_sources
    from cosmos_framework.posttrain.data.packer import validate_data_packer_factories
    from cosmos_framework.posttrain.utils.determinism import enable_determinism

    app_start_time_ms = round(time.time() * 1000)  # the controller's app start, for the lifecycle observers
    from cosmos_framework.posttrain.utils.dist import (
        destroy_process_group_if_initialized,
        init_local_process_group,
    )
    from cosmos_framework.posttrain.utils.entry import pin_run_attempt

    rl = instantiate(config.rl)
    validate_data_packer_factories(instantiate(getattr(config, "data_packer", DataPackerConfig())))
    catalog = resolve_reward_provider_catalog(str(getattr(rl, "reward_provider_catalog", "")))
    rollout_config = instantiate(config.rollout)
    policy_selection = select_policy_backend(config)
    validate_opd_runtime(
        rl,
        rollout_config,
        policy_selection.backend,
        policy_selection.model_config,
        policy_model_id=policy_selection.model_id,
    )
    backend, gpu_placement, weight_relationship = resolve_online_weight_placement(
        config.grid.backend,
        getattr(config.grid, "gpu_placement", "colocated"),
    )
    weight_sync_transport = rl.weight_sync_transport
    validate_online_weight_transport(weight_relationship, weight_sync_transport)
    if rl.deterministic:
        # Run before local process-group setup touches CUDA. Ray actors repeat this inside
        # make_setup(), because they do not inherit the driver's process state.
        enable_determinism()
    report = bind_check(manifest(rl, weight_relationship=weight_relationship, catalog=catalog))
    info0(report.format())
    report.raise_for_status()
    validate_rollout_reward_sources(rl, catalog)
    reward_spec = catalog.spec(str(rl.reward))
    info0(
        f"reward provider name={rl.reward} revision={reward_spec.revision} "
        f"identity_hash={reward_spec.identity_hash} remote={reward_spec.is_remote}"
    )
    tp = int(rollout_config.tensor_parallel_size)
    teacher_tp = int(rl.teacher.tensor_parallel_size)
    configured_shapes = bool(getattr(config.grid, "shapes", None))
    if configured_shapes and backend != "ray":
        raise ValueError("grid.shapes requires grid.backend='ray'")
    if backend == "ray" and not configured_shapes and int(config.grid.num_replicas) != 1:
        raise ValueError(
            f"grid.num_replicas={config.grid.num_replicas} no longer implicitly replicates policy; "
            "set explicit grid.shapes.policy=[1,POLICY_RANKS] and "
            "grid.shapes.rollout=[ROLLOUT_REPLICAS,ROLLOUT_RANKS]"
        )
    if configured_shapes:
        rollout_ranks = int(config.grid.rollout_shape()[1])
        if rollout_ranks != tp:
            raise ValueError(
                f"grid.shapes rollout rank width {rollout_ranks} must equal rollout.tensor_parallel_size={tp}"
            )

    local_pg = backend == "local"
    if local_pg:
        if tp != 1 or teacher_tp != 1:
            raise ValueError(
                "grid.backend='local' supports only rollout TP=1 and teacher TP=1; "
                f"got rollout TP={tp}, teacher TP={teacher_tp}. Ranked vLLM groups require Ray."
            )
        init_local_process_group()
        ranks, opts_eff = 1, list(opts)
        if int(config.grid.num_replicas) > 1:
            raise ValueError(
                f"grid.num_replicas={config.grid.num_replicas} requires grid.backend='ray': the local "
                f"backend is a single torchrun world with no replica axis to place."
            )
        policy_shape = (1, world_size_or_1())
        rollout_shape = (1, 1)
        role_shapes = None
    else:
        # Placement owns the GPU request. The independently selected transport only serves the
        # resulting same-GPU or cross-device WEIGHT relationship.
        ranks, opts_eff = tp, list(opts)
        policy_shape = tuple(config.grid.policy_shape())
        rollout_shape = tuple(config.grid.rollout_shape())
        role_shapes = {"policy": policy_shape, "rollout": rollout_shape}
        if rollout_shape[1] != tp:
            raise ValueError(f"ROLLOUT_RANKS={rollout_shape[1]} must equal rollout.tensor_parallel_size={tp}")

    if policy_shape[0] != 1:
        raise ValueError(
            f"POLICY_REPLICAS={policy_shape[0]} is unsupported: policy placement R must be 1; "
            "configure HSDP with model dp_replicate and dp_shard inside POLICY_RANKS"
        )
    policy_parallelism = resolve_policy_parallelism(config, policy_shape[1])

    # Mesh-derived divisibility and prompt-conservation checks must fail before role placement.
    policy_dp_world = policy_parallelism[0] * policy_parallelism[1]
    validate_batch_divisibility(
        batch_size=int(rl.batch_per_replica),
        mini_batch_size=int(rl.mini_batch_per_replica) or None,
        micro_batch_size=int(rl.micro_batch_per_replica) or None,
        dp_world=policy_dp_world,
    )
    if not local_pg:
        prompts_per_data_cell(
            policy_batch_size=int(rl.batch_per_replica),
            rollouts_per_prompt=int(rl.group_size),
            data_replicas=rollout_shape[0],
        )

    replicas = int(config.grid.num_replicas)
    if replicas < 1:
        raise ValueError(f"grid.num_replicas={replicas} must be >= 1")
    rollout_replicas = int(rollout_shape[0])
    if rollout_replicas != 1:
        raise ValueError(
            f"ranked OPD currently supports one rollout/teacher replica, got rollout replicas={rollout_replicas}; "
            "teacher replicas and cross-replica teacher dispatch are out of scope"
        )
    if rl.weight_sync_r2r_broadcast:
        raise ValueError(
            "OPD currently supports exactly one rollout replica, so rl.weight_sync_r2r_broadcast "
            "has no replica fan-out to serve"
        )

    attempt = pin_run_attempt(config, opts_eff, out_dir)
    out_dir, opts_eff = attempt.out_dir, list(attempt.opts)

    teacher_input_mode = str(rl.teacher.input_mode)
    logger.info(
        f"run: backend={backend} policy={policy_shape} policy_mesh={policy_parallelism} "
        f"rollout={rollout_shape} teacher_TP={teacher_tp} replicas={replicas} "
        f"distillation={rl.distillation_mode} weight_relationship={weight_relationship} "
        f"weight_transport={weight_sync_transport or ('copy' if local_pg else 'ipc')} "
        f"teacher={rl.teacher.name_or_path} top_k={rl.teacher.top_k} out={out_dir}"
    )

    lifecycle = ControllerLifecycleObservers()
    wgs: dict[str, Any] = {}
    try:
        dump_resolved(config, out_dir)
        wgs = declare_wgs(
            ranks,
            replicas=rollout_replicas,
            rm_needs_bulk=prefix_filter_enabled(config.rl),
            shapes=role_shapes,
            policy_parallelism=policy_parallelism if not local_pg else None,
            teacher_ranks=teacher_tp,
            teacher_needs_bulk=_teacher_needs_bulk(teacher_input_mode),
        )
        actor_groups = (
            ActorGroup(
                members=tuple(wgs[name] for name in OPD_EXPERIENCE),
                setup=make_setup(config_path, opts_eff, OPD_EXPERIENCE),
                name="opd_experience",
            ),
            ActorGroup(
                members=(wgs["teacher"],),
                setup=make_setup(config_path, opts_eff, OPD_TEACHER),
                name="opd_teacher",
            ),
            ActorGroup(
                members=(wgs["policy"],),
                setup=make_setup(config_path, opts_eff, ["policy"]),
                name="opd_policy",
            ),
        )
        if local_pg:
            device_groups = [DeviceGroup(actor_groups=actor_groups, device="cpu", name="opd_local_cpu")]
        else:
            experience_actor, teacher_actor, policy_actor = actor_groups
            student_device_groups = trainer_experience_device_groups(
                experience_actor,
                policy_actor,
                gpu_placement=gpu_placement,
                name="opd_student",
            )
            experience_devices, *policy_devices = student_device_groups
            device_groups = [
                experience_devices,
                DeviceGroup(actor_groups=(teacher_actor,), device="gpu", name="opd_teacher_gpus"),
                *policy_devices,
            ]
        bind_device_groups(
            device_groups,
            backend=backend,
            threshold=0,
            weight_sync=True,
            weight_sync_transport=weight_sync_transport,
            weight_sync_bucket_bytes=rl.weight_sync_bucket_bytes,
            weight_sync_probe_bytes=rl.weight_sync_probe_bytes,
            weight_sync_min_gbps=rl.weight_sync_min_gbps,
            data_pool_config=_data_pool_config(rl) if backend == "ray" else None,
            gpu_node_domains=getattr(config.grid, "gpu_node_domains", None),
        )
        # The same lifecycle reporting as grpo.run (control/lifecycle.py): the world size is the GPU
        # count of the placement just bound, so the teacher's device group is part of it.
        lifecycle = build_controller_observers(
            config,
            rl,
            controller="opd",
            world_size=placement_gpu_count(device_groups) or world_size_or_1(),
            topology=f"POLICY_{policy_shape[0]}x{policy_shape[1]}_ROLLOUT_{rollout_shape[0]}x{rollout_shape[1]}"
            f"_TEACHER_{replicas}x{teacher_tp}_{gpu_placement}",
            policy_dp_world=int(policy_parallelism[0]) * int(policy_parallelism[1]),
            app_start_time_ms=app_start_time_ms,
        )
        lifecycle.on_app_start()
        loop(wgs, config, out_dir, observers=lifecycle)
    finally:
        # As in grpo.run: the observers' app end (their teardown) runs whether or not the checkpoint
        # drain raises, and the drain's failure still propagates.
        try:
            CheckpointLifecycle.finalize_units(
                wgs,
                destroy_process_group_if_initialized if local_pg else None,
            )
        finally:
            lifecycle.on_app_end()


def _validate_and_exit(
    rl: OPDConfig,
    rollout: Any,
    grid: Any,
    model_backend: str = "hf",
    model_config: Any | None = None,
    policy_model_id: str | None = None,
    data_packer_config: Any | None = None,
) -> None:
    """Run the GPU-free OPD manifest and rollout/reward source preflight."""
    try:
        from cosmos_framework.posttrain.algorithm.reward import resolve_reward_provider_catalog

        catalog = resolve_reward_provider_catalog(str(getattr(rl, "reward_provider_catalog", "")))
        _backend, _gpu_placement, weight_relationship = resolve_online_weight_placement(
            grid.backend,
            getattr(grid, "gpu_placement", "colocated"),
        )
        validate_online_weight_transport(weight_relationship, rl.weight_sync_transport)
        output, ok = render_manifest_validation(manifest(rl, weight_relationship=weight_relationship, catalog=catalog))
    except Exception as exc:  # noqa: BLE001 -- preflight always returns a report and exit code
        output, ok = f"bind_check: error: {exc}", False
    if ok:
        try:
            from cosmos_framework.posttrain.config.build import validate_opd_runtime
            from cosmos_framework.posttrain.config.model import DataPackerConfig
            from cosmos_framework.posttrain.control.validation.rollout import validate_rollout_reward_sources
            from cosmos_framework.posttrain.data.packer import validate_data_packer_factories

            resolved_data_packer = data_packer_config if data_packer_config is not None else DataPackerConfig()
            validate_data_packer_factories(resolved_data_packer)
            validate_opd_runtime(
                rl,
                rollout,
                model_backend,
                model_config,
                policy_model_id=policy_model_id,
            )
            validate_rollout_reward_sources(rl, catalog)
        except Exception as exc:  # noqa: BLE001 -- preflight always returns a report and exit code
            output = f"{output}\nsource_check: error: {exc}"
            ok = False
    print(output)  # persisted to the run log
    raise SystemExit(0 if ok else 1)


def main() -> None:
    from cosmos_framework.utils.config import load_config
    from cosmos_framework.utils.lazy_config import instantiate

    config_path, opts, validate = cli(
        "Cosmos3 on-policy distillation (vLLM rollout + vLLM teacher)",
        default_config="cosmos_framework/posttrain/config/config_opd.py",
    )
    config = load_config(config_path, ["--", *opts])
    if validate:
        from cosmos_framework.posttrain.config.build import select_policy_backend
        from cosmos_framework.posttrain.config.model import DataPackerConfig

        policy_selection = select_policy_backend(config)
        _validate_and_exit(
            instantiate(config.rl),
            instantiate(config.rollout),
            instantiate(config.grid),
            policy_selection.backend,
            policy_selection.model_config,
            policy_selection.model_id,
            instantiate(getattr(config, "data_packer", DataPackerConfig())),
        )
    run(config, config_path, opts, _out_dir(config))


if __name__ == "__main__":
    main()
