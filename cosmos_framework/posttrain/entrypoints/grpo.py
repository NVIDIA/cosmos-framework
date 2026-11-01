# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Online GRPO launch, role assembly, and training/validation loop.

Reusable prompt, rollout-semantics, checkpoint, and held-out evaluation mechanics live in their
owning packages. This module keeps the topology wiring and online loop visible.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

from cosmos_framework.posttrain.algorithm.annotators.advantage import INFORMATIVE_GROUP_GATE
from cosmos_framework.posttrain.config.build import resolve_pdims, resolve_policy_parallelism
from cosmos_framework.posttrain.config.rl import GRPOConfig
from cosmos_framework.posttrain.control.buffer import MetadataGate
from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
from cosmos_framework.posttrain.control.lifecycle import (
    ControllerLifecycleObserver,
    ControllerLifecycleObservers,
    build_controller_observers,
)
from cosmos_framework.posttrain.control.loop.batching import (
    informative_quota_per_cell,
    prompts_per_data_cell,
    validate_batch_divisibility,
)
from cosmos_framework.posttrain.control.loop.offload import offload_cycle_enabled, sleep_rollout
from cosmos_framework.posttrain.control.manifest import Manifest
from cosmos_framework.posttrain.control.placement import (
    resolve_gpu_pair_placement,
    resolve_online_weight_placement,
    validate_online_weight_transport,
)
from cosmos_framework.posttrain.control.placement.gpu_pair import WeightRelationship
from cosmos_framework.posttrain.control.validation.lifecycle import ValidationReport
from cosmos_framework.posttrain.roles.env.prompt import prompt_source
from cosmos_framework.posttrain.utils.batch_stats import RewardBatchStats, resolve_reward_report
from cosmos_framework.posttrain.utils.dist import world_size_or_1
from cosmos_framework.posttrain.utils.entry import (
    cli,
    dump_resolved,
    render_manifest_validation,
    resolve,
    resolve_all,
)
from cosmos_framework.posttrain.utils.entry import out_dir as _out_dir
from cosmos_framework.posttrain.utils.entry import (
    resolve_identical as _resolve_identical,
)
from cosmos_framework.posttrain.utils.log import info0, logger
from cosmos_framework.posttrain.utils.profile import ProfileRecord
from cosmos_framework.posttrain.utils.residency import memory_saver_env
from cosmos_framework.posttrain.workers.data_pool import has_pool_slices, release_pool_step_barrier
from cosmos_framework.posttrain.workers.worker_group import WorkerGroup, release_result_handles, scatter

if TYPE_CHECKING:
    from cosmos_framework.posttrain.algorithm.reward import RewardProviderCatalog

ROLE_SPECS = [("data", "env"), ("rollout", "rollout"), ("rm", "reward"), ("replay", "buffer"), ("policy", "trainer")]
_EXPERIENCE = ["data", "rollout", "rm", "replay"]  # the experience-side ActorGroup (vs the learner)
_MODEL_BEARING = ("rollout", "policy")  # FSDP/TP-shard across the mesh; every other role is an SPMD helper


def _sleep_rollout_before_trainer_finalization(rollout: Any) -> None:
    """Release rollout memory without replacing an active training/setup failure."""
    if isinstance(rollout, WorkerGroup) and rollout.body is None and rollout._backend is None:
        # Declaration exists, but setup/bind never produced a callable rollout body. There is no
        # device state to release; WorkerGroup.__getattr__ would otherwise synthesize a method and
        # fail only when dispatch reaches the missing body.
        return
    active_error = sys.exc_info()[1] is not None
    try:
        sleep_rollout(rollout)
    except BaseException as error:  # noqa: BLE001 -- cleanup cannot replace an active primary failure
        logger.error(f"rollout cleanup sleep failed: {error}")
        if not active_error:
            raise


def _require_native_policy_spec(selection: Any) -> Any:
    """Return the selected online native build spec without instantiating a legacy model tree."""
    spec = selection.native_spec
    if spec is None:
        raise TypeError(
            "online model_backend='i4' requires config.model=NativePolicyBuildSpec; "
            "legacy config.model trees are not online policy build specs"
        )
    return spec


def _validate_online_backend(config: Any, rl: Any) -> None:
    """Validate the generic ``i4 | hf`` online backend before placement or model loading."""
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.config.build import (
        select_policy_backend,
        validate_online_policy_backend,
    )
    from cosmos_framework.posttrain.config.model import DataPackerConfig
    from cosmos_framework.posttrain.config.rl import RolloutConfig
    from cosmos_framework.posttrain.data.packer import validate_data_packer_factories

    selection = select_policy_backend(config)
    native_spec = _require_native_policy_spec(selection) if selection.backend == "i4" else None
    # Model configs remain inert on the driver. Only the policy worker resolves a builder-owned
    # config, so a generic LazyCall cannot accidentally construct a large target during preflight.
    model_config = selection.model_config
    rollout_config = instantiate(getattr(config, "rollout", RolloutConfig()))
    data_packer_config = instantiate(getattr(config, "data_packer", DataPackerConfig()))
    validate_data_packer_factories(data_packer_config)
    validate_online_policy_backend(
        rl,
        rollout_config,
        selection.backend,
        model_config,
        policy_model_id=selection.model_id,
    )
    surrogate = str(rl.grpo_surrogate)
    requires_collated_forward = bool(getattr(native_spec, "requires_collated_policy_forward", False))
    if requires_collated_forward and surrogate != "clipped":
        raise ValueError(
            f"this native policy requires rl.grpo_surrogate='clipped', got {surrogate!r}; its build spec "
            "declares requires_collated_policy_forward=True"
        )


# The per-step console line: the headline trio, then the pre-clip gradient norm (`train/grad_norm` is
# the total norm `clip_grad_norm_` measured BEFORE clipping; W&B already receives it), then, when the
# KL anchor is on, the token-mean AND the token-max of the K3 penalty. The max is the column that
# identifies steps dominated by a small number of tokens.
_CONSOLE_HEADLINE_KEYS: tuple[str, ...] = (
    "train/loss_avg",
    "train/reward_mean",
    "train/grpo_informative_group_count",
    "train/grad_norm",
)
_CONSOLE_KL_KEYS: tuple[str, ...] = ("train/kl_mean", "train/kl_max")


def console_headline_keys(rl: Any) -> tuple[str, ...]:
    """Console columns for the per-step line; the KL columns appear only when `rl.kl_beta > 0`."""
    keys = _CONSOLE_HEADLINE_KEYS
    if float(getattr(rl, "kl_beta", 0.0)) > 0.0:
        keys = keys + _CONSOLE_KL_KEYS
    return keys


def _data_pool_config(rl: GRPOConfig) -> dict[str, int] | None:
    """Build the Ray host pool settings for an opted-in RDT/NIXL data plane."""
    pool = rl.data_pool
    if pool.data_plane == "bulk_store":
        return None
    if pool.data_plane != "ray_nixl_pool":
        raise ValueError(f"unsupported data_pool.data_plane {pool.data_plane!r}")
    return {
        "page_bytes": pool.page_bytes,
        "cuda_capacity_bytes": pool.cuda_capacity_bytes,
        "cpu_capacity_bytes": pool.cpu_capacity_bytes,
        "cuda_cache_bytes": pool.cuda_cache_bytes,
        "cpu_cache_bytes": pool.cpu_cache_bytes,
        "cpu_min_bytes": pool.cpu_min_bytes,
    }


def _short_cells(
    replay: Any,
    gate: MetadataGate,
    quotas: tuple[int, ...],
    *,
    scope: str = "cell",
    cell_share: int = 0,
) -> tuple[list[int], int]:
    """Replay cells that still need informative draws, and the informative trajectories the batch would hold.

    Batch assembly happens at each replay cell over its own data cell's draws: a cell contributes at
    most ``cell_share`` trajectories, gate-passing groups first (`ExperienceBuffer.select_prioritized`),
    so the assembled batch holds ``sum(min(count, cell_share))`` gate-passing trajectories. ``quotas``
    is the batching layer's exact split of the global knob
    (`control.loop.batching.informative_quota_per_cell`), one entry per replay cell in coordinate
    order, which ``resolve_all`` also uses; a cell-count mismatch is a topology fault, not a shortfall.

    ``cell`` scope keeps every cell to its own quota, so a cell short of its share cannot borrow from
    one that over-collected. ``pooled`` scope enforces ``sum(quotas)`` on the assembled batch: nothing
    is short once it is met; otherwise every cell whose share is not yet gate-passing may refill. An
    unmet pooled quota with every cell full asks for more than the batch holds, which raises rather
    than returning no short cells (that would skip both the refill and the budget check).
    """
    counts = [int(count) for count in resolve_all(replay.count(gate))]
    if len(counts) != len(quotas):
        raise RuntimeError(f"replay reported {len(counts)} cells but the batch plan spans {len(quotas)} data cells")
    assembled = sum(min(count, cell_share) for count in counts)
    if scope == "pooled":
        quota = sum(quotas)
        if assembled >= quota:
            return [], assembled
        short = [index for index, count in enumerate(counts) if count < cell_share]
        if not short:
            raise RuntimeError(
                f"pooled informative quota of {quota} trajectories exceeds the {cell_share * len(counts)} "
                "trajectories the assembled batch can hold"
            )
        return short, assembled
    return [index for index, (count, quota) in enumerate(zip(counts, quotas)) if count < quota], assembled


def _pool_owner_actors(units: Mapping[str, Any]) -> dict[str, Any]:
    """Read the stable bind-time producer registry."""
    owners: dict[str, Any] = {}
    for unit in units.values():
        registry = getattr(unit, "connection_registry", None)
        if callable(registry):
            owners.update(registry())
    return owners


def manifest(
    rl: GRPOConfig,
    catalog: RewardProviderCatalog | None = None,
    *,
    weight_relationship: WeightRelationship = "same_gpu",
) -> Manifest:
    """This config's recipe topology, for `bind_check`."""
    from cosmos_framework.posttrain.algorithm.reward import resolve_reward_provider_catalog
    from cosmos_framework.posttrain.control.recipe import grpo_manifest

    if catalog is None:
        catalog = resolve_reward_provider_catalog(str(getattr(rl, "reward_provider_catalog", "")))
    source, ground_truth = prompt_source(rl)
    return grpo_manifest(
        ROLE_SPECS,
        source=source,
        ground_truth=ground_truth,
        provider_spec=catalog.spec(str(rl.reward)),
        weight_relationship=weight_relationship,
    )


def declare_wgs(
    ranks: int,
    replicas: int = 1,
    rm_needs_bulk: bool = False,
    *,
    shapes: dict[str, tuple[int, int]] | None = None,
    policy_parallelism: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Declare uniform or role-specific worker-group shapes."""
    from cosmos_framework.posttrain.roles import roles as role_factory

    def _shape(name: str) -> tuple[int, int]:
        if shapes:
            if name in shapes:  # policy | rollout
                return shapes[name]
            return shapes["rollout"][0], 1
        if name == "policy":
            return 1, ranks
        return replicas, ranks if name in _MODEL_BEARING else 1

    def _kwargs(name: str, kind: str) -> dict[str, Any]:
        r, k = _shape(name)
        kw: dict[str, Any] = {"replicas": r, "ranks": k}
        if kind == "reward" and rm_needs_bulk:
            kw["needs_bulk"] = True
        if kind == "trainer":
            kw["model_parallelism"] = policy_parallelism
        return kw

    return {name: getattr(role_factory, kind)(name, **_kwargs(name, kind)) for name, kind in ROLE_SPECS}


# --------------------------------------------------------------------------- assembly
def build_env(rl: GRPOConfig, config: Any = None) -> Any:
    """Build the prompt source and optional finite held-out view."""
    from cosmos_framework.posttrain.roles.env.prompt import build_prompt_env

    return build_prompt_env(rl, config)


def _offload_cycle_enabled(config: Any, rl: Any) -> bool:
    """Whether this run hands each card between the rollout engine and the trainer every step.

    True only under ``grid.gpu_placement=colocated`` with ``rl.colocated_offload`` left on; the
    engine is built with sleep mode and the loop runs the hand-offs exactly when this is True.
    """
    gpu_placement = getattr(getattr(config, "grid", None), "gpu_placement", "colocated")
    return offload_cycle_enabled(
        gpu_placement=str(gpu_placement),
        colocated_offload=bool(getattr(rl, "colocated_offload", True)),
    )


def _park_rollout_for_resume(config: Any, rl: Any, rollout: Any) -> bool:
    """Sleep a just-built engine whose colocated trainer is about to restore a checkpoint.

    Bind sets up the experience roles before the policy, so an awake engine would still hold
    ``rollout.gpu_memory_utilization`` of each shared card while the trainer loads its optimizer
    state there. The parked engine's base weights are never sampled: the loop's first hand-off
    wakes it and publishes the restored policy. Returns whether the engine was parked.
    """
    checkpoint = CheckpointLifecycle.worker_resume_checkpoint(rl)
    if checkpoint is None or not _offload_cycle_enabled(config, rl):
        return False
    sleep_rollout(rollout)
    info0(f"rollout engine parked while the trainer restores {checkpoint}")
    return True


def build_rollout(config: Any) -> Any:
    """Build the vLLM engine inside its established tensor-parallel group."""
    import torch

    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.config.build import (
        build_vllm_rollout,
        policy_data_packer_build_spec,
        rollout_records_routing,
        rollout_topk_logprobs,
    )
    from cosmos_framework.posttrain.config.model import DataPackerConfig
    from cosmos_framework.posttrain.utils.dist import run_rank_converged

    def prepare() -> tuple[Any, Any, bool, bool, bool, int]:
        rollout_cfg = instantiate(config.rollout)
        rl = instantiate(config.rl)
        data_packer_config = instantiate(getattr(config, "data_packer", DataPackerConfig()))
        packer_spec = policy_data_packer_build_spec(config, data_packer_config)
        # Validate the external-launcher process group before vLLM enters a collective.
        if rollout_cfg.tensor_parallel_size > 1:
            if not torch.distributed.is_initialized():
                raise RuntimeError("per-replica rendezvous not run (need ranks=TP>1 on GPU)")
            ws = torch.distributed.get_world_size()
            if ws != rollout_cfg.tensor_parallel_size:
                raise RuntimeError(f"TP world size {ws} != TP={rollout_cfg.tensor_parallel_size}")
        # Rollout shaping uses the engine's tokenizer; the trainer scores the recorded token ids.
        return (
            rollout_cfg,
            packer_spec.build(),
            packer_spec.trust_remote_code,
            rollout_records_routing(rl),
            _offload_cycle_enabled(config, rl),
            rollout_topk_logprobs(rl),
        )

    build_profile = ProfileRecord("grpo_build_rollout", progress=True)
    rollout_cfg, packer, trust_remote_code, return_routed_experts, offload_cycle, topk_logprobs = run_rank_converged(
        prepare, "GRPO rollout pre-engine preparation"
    )
    build_profile.checkpoint("prepare")
    rollout = build_vllm_rollout(
        rollout_cfg,
        packer=packer,
        trust_remote_code=trust_remote_code,
        # The engine returns its per-token expert routing only when the trainer replays it.
        return_routed_experts=return_routed_experts,
        # Sleep mode is only needed when the loop will sleep the engine (colocated hand-off cycle on);
        # disaggregated placement keeps main's allocator (sleep mode on, never slept) so nothing changes there.
        enable_sleep_mode=offload_cycle or str(getattr(config.grid, "gpu_placement", "colocated")) != "colocated",
        # The engine returns and the body records the sampler's top-k head only under score centering.
        topk_logprobs=topk_logprobs,
    )
    build_profile.checkpoint("engine")
    build_profile.emit(gpu_memory_utilization=float(rollout_cfg.gpu_memory_utilization))
    return rollout


def build_objective(rl: Any, model_config: Any, packer: Any) -> Any:
    """Build the GRPO entry's configured objective.

    Args:
        rl: Resolved GRPO algorithm config.
        model_config: Policy identity supplied by the shared trainer assembly.
        packer: Policy-side completion codec supplied by the shared trainer assembly.

    Returns:
        The trainer-facing GRPO objective.
    """
    from cosmos_framework.posttrain.config.build import build_objective as build_grpo_objective

    if rl.loss_type == "seq-mean-token-sum-fixed":
        return build_grpo_objective(
            rl,
            model_config.name_or_path,
            packer=packer,
            max_response_length=model_config.max_response_length,
        )
    return build_grpo_objective(rl, model_config.name_or_path, packer=packer)


def _build_reward_annotators(config: Any, rl: Any, name_or_path: str) -> tuple[Any, ...]:
    """Build reward annotators, resolving a reward-role packer only when one is consumed."""
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.config.build import (
        build_reward_annotators,
        policy_data_packer_build_spec,
        prefix_filter_enabled,
    )
    from cosmos_framework.posttrain.config.model import DataPackerConfig

    packer = None
    if prefix_filter_enabled(rl):
        data_packer_config = instantiate(getattr(config, "data_packer", DataPackerConfig()))
        packer = policy_data_packer_build_spec(config, data_packer_config).build()
    return build_reward_annotators(rl, name_or_path, packer=packer)


# Fields through which a model-family config names the checkpoint its builder loads: the artifact
# identity (`name_or_path` — AutoConfig/tokenizer/processor and, when `checkpoint_path` is empty, the
# weights) and an optional storage mirror of the weights (`checkpoint_path`; `config/model.py`,
# `model/qwen3_vl.py::build_qwen3_vl_model` loads `checkpoint_path or name_or_path`).
_ARTIFACT_PATH_FIELDS = ("name_or_path", "checkpoint_path")


def _replaced(node: Any, **changes: Any) -> Any:
    """Copy of ``node`` with ``changes`` applied; a dataclass re-runs its ``__post_init__`` validators."""
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        return dataclasses.replace(node, **changes)
    clone = copy.copy(node)
    for name, value in changes.items():
        setattr(clone, name, value)
    return clone


def pointed_at(family_config: Any, model_dir: str) -> Any:
    """Copy of a model-family config whose NON-EMPTY artifact-path fields all name ``model_dir``.

    Empty fields keep their meaning (an empty ``checkpoint_path`` says "the weights live at
    ``name_or_path`` / the spec's ``model_id``"), so nothing is redirected that did not already point
    at the policy's artifact. The input is never mutated.
    """
    changes = {name: model_dir for name in _ARTIFACT_PATH_FIELDS if str(getattr(family_config, name, "") or "")}
    return _replaced(family_config, **changes) if changes else family_config


def reference_build_config(config: Any, model_dir: str) -> Any:
    """Shallow copy of the root ``config`` whose native policy spec loads ``model_dir`` (``rl.ref.model_dir``).

    Warm restart: the policy is started from its latest saved weights while the frozen KL reference
    must stay the ORIGINAL base. The reference is built by the SAME builder as the policy, so the one
    thing to change is where the weights come from — on a copy, never on the caller's ``config`` (the
    policy build that already consumed it, and everything else reading the root, keep seeing the
    policy's own directory). The redirected fields are exactly the ones the policy build reads
    (``config/build.py::native_policy_build_spec``):

    * ``model_backend='qwen3_vl'`` alias (the Ultra launchers): ``config.model_qwen3_vl`` — the spec's
      ``model_id`` is its ``name_or_path`` and the Qwen builder loads ``checkpoint_path or name_or_path``;
    * generic ``i4`` spec: ``config.model.model_id`` plus the populated path fields of the spec's
      family ``model_config`` (``name_or_path`` must equal ``model_id`` when set; ``checkpoint_path``
      may mirror the weights and would otherwise win over ``model_id``).

    The HF backend has no root-level redirect: ``build_trainer`` re-points the instantiated
    ``model_hf`` config directly with :func:`pointed_at`.
    """
    from cosmos_framework.posttrain.config.model import QWEN3_VL

    clone = copy.copy(config)
    if str(getattr(config, "model_backend", "")) == QWEN3_VL:
        clone.model_qwen3_vl = pointed_at(config.model_qwen3_vl, model_dir)
    else:
        spec = config.model
        clone.model = _replaced(spec, model_id=model_dir, model_config=pointed_at(spec.model_config, model_dir))
    return clone


def build_trainer(
    config: Any,
    rl: Any,
    *,
    build_objective_fn: Callable[[Any, Any, Any], Any],
) -> Any:
    """Assemble the FSDP online learner with an entry-owned objective builder.

    Backend selection is config-owned: HF uses the built-in adapter, while the native backend dispatches through
    ``NativePolicyBuildSpec.builder``. The entry chooses the algorithm without changing the shared
    model, optimizer, checkpoint, or trainer assembly.

    Args:
        config: Resolved online run config.
        rl: Resolved algorithm config.
        build_objective_fn: Objective builder selected by the calling entry.

    Returns:
        The configured online learner.
    """
    from types import SimpleNamespace

    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.config.build import (
        build_lr_scheduler,
        build_native_policy,
        build_optimizer,
        effective_model_id,
        forward_replay_source,
        normalize_model_backend,
        policy_data_packer_build_spec,
        resolve_precision,
    )
    from cosmos_framework.posttrain.config.model import DataPackerConfig
    from cosmos_framework.posttrain.model.hf import build_hf_model
    from cosmos_framework.posttrain.roles.trainer import LLMTrainer
    from cosmos_framework.posttrain.roles.trainer.llm import (
        KL_REFERENCE_DETERMINISM_REQUIREMENT,
        raise_if_forward_is_stochastic,
    )
    from cosmos_framework.posttrain.utils.artifact_identity import model_artifact_identity
    from cosmos_framework.posttrain.utils.dist import broadcast_from_rank0, run_rank_converged

    checkpoint_lifecycle = CheckpointLifecycle.from_config(config, rl)
    build_profile = ProfileRecord("grpo_build_trainer", progress=True)

    backend = normalize_model_backend(str(getattr(config, "model_backend", "i4")))
    ref_model_dir = getattr(getattr(rl, "ref", None), "model_dir", None)

    def require_ref_model_dir() -> None:
        """`rl.ref.model_dir` must exist HERE, where it is about to be loaded, on every rank, before
        any model is built — not at config construction. `RefConfig` validates the type and the
        `build_ref` pairing only, so the saved config reloads on a host without the checkpoint mount
        (tooling reading config_resolved.yaml, login-node validation, a resume after the base
        directory was pruned) and the refusal names the run that actually needs the directory."""
        if ref_model_dir is not None and not os.path.isdir(ref_model_dir):
            raise ValueError(
                f"rl.ref.model_dir={ref_model_dir!r} is not a directory; the frozen KL reference is loaded from "
                "it on every policy rank, so it must be a complete local model directory"
            )

    run_rank_converged(require_ref_model_dir, "rl.ref.model_dir preflight")
    builds_reference = float(getattr(rl, "kl_beta", 0.0)) > 0.0 and bool(
        getattr(getattr(rl, "ref", None), "build_ref", False)
    )
    # The anchor's WEIGHTS are part of the objective. Under `rl.ref.model_dir` they come from a
    # directory the config names, so the exact-resume identity binds that directory's CONTENT
    # (`utils/artifact_identity.py`: metadata and weight files by full-content SHA-256), never the
    # path string: an exact resume that names a different checkpoint, or the same
    # path re-populated with other weights, is refused by `LLMTrainer._validate_objective_identity`
    # instead of silently training toward another anchor. Read once, on rank 0, and broadcast — every
    # policy rank would otherwise hash the same shards from the shared filesystem. Absent when the
    # reference is the policy's own initial weights, which is what every earlier anchored checkpoint
    # trained under (the absence convention of `config/build.py::build_objective`).
    reference_artifact: dict[str, Any] | None = None
    if builds_reference and ref_model_dir is not None:
        reference_artifact = broadcast_from_rank0(
            lambda: model_artifact_identity(ref_model_dir),
            context="rl.ref.model_dir artifact identity",
        )
        build_profile.checkpoint("reference_artifact")
    if backend == "hf":
        model_config = instantiate(config.model_hf)
        # Trainer, rollout, and reward tokenization share one effective model identity.
        if not model_config.name_or_path:
            model_config.name_or_path = effective_model_id(config)
        policy_world = world_size_or_1()
        model_parallelism = resolve_policy_parallelism(config, policy_world)
        # `precision` splits STORAGE (rl.master_dtype, what the optimizer and master weights live
        # in) from COMPUTE (rl.param_dtype). Resolved ONCE from the INSTANTIATED rl + model_config
        # this function holds, not the raw config node, so the model's compute dtype and the
        # trainer's wire dtype cannot silently drift apart. Unset fields fall back to
        # `model_hf.dtype`, i.e. the pre-split behaviour. The reference GRPO recipe runs fp32 master
        # with bf16 compute, which is inexpressible without this split.
        precision = resolve_precision(
            SimpleNamespace(rl=rl, model_config=model_config, model_backend=backend),
        )
        # Ray injects the same model-owned factor before setup; local torchrun constructs it here.
        model = build_hf_model(
            model_config,
            resolve_pdims(parallelism=model_parallelism),
            precision=precision,
        )  # HFModel body + FSDP over the resolved mesh
    else:
        model, model_config = build_native_policy(config)
        # The native builder has already applied its model-config precision. Resolve the same values
        # here so the policy-to-rollout publisher declares a real transfer dtype instead of reaching
        # the constructor below with an unbound HF-only local.
        precision = resolve_precision(
            SimpleNamespace(rl=rl, model_config=model_config, model_backend=backend),
        )
    build_profile.checkpoint("model")
    opt = build_optimizer(model, rl.optimizer)
    build_profile.checkpoint("optimizer")
    # The learner constructs its policy packer eagerly because every online objective is assembled
    # through this shared seam, including objectives that consume rollout-recorded token ids.
    data_packer_config = instantiate(getattr(config, "data_packer", DataPackerConfig()))
    packer_spec = policy_data_packer_build_spec(config, data_packer_config)
    policy_packer = run_rank_converged(
        packer_spec.build,
        "GRPO policy DataPacker setup",
    )
    # Objective shaping depends on the explicit policy/tokenizer identity, not on a family-specific
    # builder config. A generic native config therefore need not mimic HF's ``name_or_path`` schema.
    objective_model_config = SimpleNamespace(
        name_or_path=effective_model_id(config),
        max_response_length=config.rollout.max_tokens,
    )
    objective = build_objective_fn(rl, objective_model_config, policy_packer)
    if reference_artifact is not None:
        # Before the trainer is constructed: `LLMTrainer.__init__` snapshots `objective.identity`
        # into the durable exact-resume identity it checkpoints and checks on resume.
        objective.identity["reference_artifact"] = reference_artifact
    # Networks besides the policy that the trainer must move off and back onto the card with it
    # (`LLMTrainer(auxiliary_models=...)`, `roles/trainer/llm.py::AuxiliaryModelResidency`).
    auxiliary_models: list[Any] = []
    if builds_reference:
        # Frozen KL reference (`algorithm/grpo.py::FrozenReference`): a SECOND instance from the SAME
        # builder, mesh, precision, and base checkpoint the policy just loaded, so at step 0 it IS
        # the policy (KL exactly 0) and it stays the run's initial weights: the resume below restores
        # the policy only, never this twin (the same un-checkpointed anchor DPO/NFT use). Under
        # `rl.ref.model_dir` the twin loads THAT directory instead: a warm restart from a model-only
        # save starts the policy from the save while the anchor stays the original base. The
        # objective gets the SCORER (it consumes reference logprobs and nothing else); the trainer
        # gets the MODEL for residency, so the twin leaves the card with the policy under colocation
        # without the engine reaching into the algorithm for a network.
        holder = getattr(objective, "reference_scorer", None)
        if holder is None or not callable(getattr(holder, "attach", None)):
            raise RuntimeError(
                f"rl.kl_beta={rl.kl_beta} with rl.ref.build_ref=true, but the {type(objective).__name__} "
                "objective carries no frozen-reference slot (`reference_scorer`); only the clipped GRPO "
                "objective consumes one"
            )
        # The twin scores under eval() while the policy trains under train(), at ANY n_mini, so a live
        # dropout would make z = logπ_ref − logπ_θ a dropout mask (non-zero at step 0 even at identical
        # weights) that the anchor pulls against every step. The screen the π_old ahead pass runs at
        # n_mini > 1 is applied to the policy here, before the twin's memory is spent. Fails on every
        # rank alike: the module structure is config-uniform.
        raise_if_forward_is_stochastic(getattr(model, "net", model), requirement=KL_REFERENCE_DETERMINISM_REQUIREMENT)
        if backend == "hf":
            reference_model = build_hf_model(
                model_config if ref_model_dir is None else pointed_at(model_config, ref_model_dir),
                resolve_pdims(parallelism=model_parallelism),
                precision=precision,
            )
        else:
            reference_model, _ = build_native_policy(
                config if ref_model_dir is None else reference_build_config(config, ref_model_dir)
            )
        holder.attach(reference_model)
        auxiliary_models.append(reference_model)
        build_profile.checkpoint("reference")
        if ref_model_dir is None:
            info0(f"built frozen KL reference (initial policy weights) for rl.kl_beta={rl.kl_beta}")
        else:
            info0(f"built frozen KL reference from {ref_model_dir} (rl.ref.model_dir)")
    trainer = LLMTrainer(
        model,
        objective,
        opt,
        # DERIVED, not configured: GRPO retires `rl.grad_accum` (GRPOConfig rejects a non-default
        # value). The trainer's accumulation count IS the micro count, so its existing
        # accumulate-then-step fires on the last micro call of each mini-batch (spec §5.2).
        grad_accum=rl.n_micro,
        # Bookkeeping only — the trainer never uses it, but the CHECKPOINT records the batching
        # shape it was written under, so a resume under a different n_mini fails loud instead of
        # dividing the optimizer-step clock into a silently wrong collected-batch step.
        n_mini=rl.n_mini,
        # The dtype weights are DECLARED as (and cast to) on the policy -> rollout hop. From the same
        # `resolve_precision` the model was built with, so the wire dtype and the storage dtype cannot
        # drift; without it the manifest would declare the trainer's STORAGE dtype and an fp32 policy
        # would ship fp32 buffers into a bf16 engine. `comm/weight_transfer.py` reads it off the trainer.
        transfer_dtype=precision.transfer,
        checkpoint_manager=checkpoint_lifecycle.manager,
        # grad hygiene from rl.optimizer (mirror dpo.py)
        lr_scheduler=build_lr_scheduler(opt, rl.optimizer),  # cosmos-rl optm_warmup_steps
        max_grad_norm=float(rl.optimizer.grad_clip_norm),
        force_finite_grads=bool(rl.optimizer.force_finite_grads),
        # Refuses a dense/HF policy here, before Ray or an engine exists, when the knob is not 'off'.
        forward_replay=forward_replay_source(rl, model),
        # The frozen KL reference, when built: the trainer owns its device residency, the objective
        # only its scores.
        auxiliary_models=auxiliary_models,
    )
    build_profile.checkpoint("trainer")
    # Checkpoints store optimizer steps; loop() converts that clock to collected-batch steps.
    ckpt = checkpoint_lifecycle.resume_checkpoint
    if ckpt:
        it = checkpoint_lifecycle.load_trainer(trainer, ckpt)
        info0(f"resumed from {ckpt} (iter {it})")
        build_profile.checkpoint("resume")
    build_profile.emit(backend=backend)
    return trainer


def make_setup(
    config_path: str,
    opts: list[str],
    roles: list[str],
    *,
    build_objective_fn: Callable[[Any, Any, Any], Any],
) -> Any:
    """Build only the roles assigned to one local process or Ray actor.

    Args:
        config_path: Config module loaded independently in the target process.
        opts: Resolved dotlist overrides shared by every role.
        roles: Role names assigned to this setup process.
        build_objective_fn: Entry-owned objective builder for the policy role.

    Returns:
        A zero-argument setup callable for the selected roles.
    """

    def _setup() -> dict:
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
            # THIS process's own call: under grid.backend=ray this closure runs inside a Ray actor —
            # a SEPARATE process from the driver, so the driver's enable_determinism() (in run()) does
            # not reach here. Must run before the model-bearing roles (rollout/policy) build below.
            enable_determinism()
        out = {}
        setup_profile = ProfileRecord("grpo_setup", progress=True)
        if "data" in roles:
            out["data"] = run_rank_converged(
                lambda: build_env(rl, config),
                "GRPO data and validation source setup",
            )
            setup_profile.checkpoint("data")
        if "rollout" in roles:
            out["rollout"] = build_rollout(config)
            setup_profile.checkpoint("rollout")
            if _park_rollout_for_resume(config, rl, out["rollout"]):
                setup_profile.checkpoint("rollout_parked_for_resume")
        if "rm" in roles:
            # Resolve the same persisted catalog path that the driver used for manifest/source bind.
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
        if "replay" in roles:
            out["replay"] = ExperienceBuffer()
        if "policy" in roles:
            # Only rollout/policy build a model; build_trainer dispatches the config-selected backend.
            out["policy"] = build_trainer(config, rl, build_objective_fn=build_objective_fn)
            setup_profile.checkpoint("policy")
        setup_profile.emit(roles="+".join(out))
        info0(f"setup done: roles={list(out)}")
        return out

    return _setup


def _log_validation_report(report: ValidationReport, model_step: int) -> None:
    """Log the GRPO-specific rollout/reward validation headline."""
    info0(
        f"validation step={model_step} samples={report.sample_count} "
        f"reward={report.metrics['val/reward_avg']:.6f} reasons="
        f"{','.join(sorted(reason.value for reason in report.trigger_reasons))}"
    )


def loop(
    units: dict[str, Any],
    config: Any,
    out_dir: str,
    *,
    observers: ControllerLifecycleObservers
    | ControllerLifecycleObserver
    | Iterable[ControllerLifecycleObserver]
    | None = None,
) -> dict[str, Any]:
    # Lifecycle observers (DESIGN §2.6; control/lifecycle.py): the loop emits generic events only —
    # train start/end, one train step per optimizer update, one train batch per forward/backward —
    # and the checkpoint coordinator emits the save events. No observers (the default) is a no-op.
    lifecycle = ControllerLifecycleObservers.coerce(observers)
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.control.loop.offload import (
        offload_trainer,
        onload_trainer,
        prepare_trainer_weight_sync,
        wake_rollout,
        wake_rollout_weights,
    )
    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync
    from cosmos_framework.posttrain.control.termination import TerminationWatch
    from cosmos_framework.posttrain.control.validation import ValidationSchedule
    from cosmos_framework.posttrain.control.validation.boundary import ValidationBoundaryCoordinator
    from cosmos_framework.posttrain.control.validation.rollout import prepare_rollout_validation
    from cosmos_framework.posttrain.control.validation.transaction import ValidationTransaction
    from cosmos_framework.posttrain.schema.metrics import SUMMED_TRAIN_COUNT_KEYS
    from cosmos_framework.posttrain.utils.report import (
        emit_metric_frame,
        finish_best_effort,
        log_sample_previews,
        make_reporters,
        merge_training_metric_window,
        reportable_metrics,
        start_best_effort,
    )

    rl = instantiate(config.rl)
    data_plane = getattr(getattr(rl, "data_pool", None), "data_plane", "bulk_store")
    from cosmos_framework.posttrain.config.rl import RolloutConfig

    rollout_config = instantiate(getattr(config, "rollout", RolloutConfig()))
    data, rollout, rm, replay, policy = (units[n] for n in ("data", "rollout", "rm", "replay", "policy"))
    teacher = units.get("teacher")
    pool_owner_actors = _pool_owner_actors(units) if data_plane == "ray_nixl_pool" else {}
    release_validation_payload: Callable[[Any], None] | None = None
    if data_plane == "ray_nixl_pool":
        # Future work: both this hook and the training-step barrier below release EVERY live
        # lease on EVERY producer, which is only correct because synchronous steps and validation
        # batches never overlap. A cleaner release would be payload-scoped: producers track which
        # leases a published payload references (or consumers acknowledge per lease) so the driver
        # releases exactly what a batch used. That also removes the duplicated resolve/release
        # sequence here and in the step loop, and is what async consumers would need.

        def release_validation_payload(trajectories: Any) -> None:
            """Return every producer lease a validation batch published, mirroring the step barrier."""
            completed = resolve_all(trajectories)
            release_result_handles(trajectories)
            release_pool_step_barrier(completed, pool_owner_actors)

    metric_weight_map_probe = getattr(policy, "training_metric_weight_map", None)
    raw_metric_weight_map = (
        _resolve_identical(metric_weight_map_probe(), operation="policy.training_metric_weight_map")
        if callable(metric_weight_map_probe)
        else {}
    )
    if not isinstance(raw_metric_weight_map, Mapping):
        raise TypeError(
            "policy.training_metric_weight_map must return a public metric-to-weight mapping, "
            f"got {raw_metric_weight_map!r}"
        )
    training_metric_weight_map = dict(raw_metric_weight_map)
    metric_max_keys_probe = getattr(policy, "training_metric_max_keys", None)
    raw_metric_max_keys = (
        _resolve_identical(metric_max_keys_probe(), operation="policy.training_metric_max_keys")
        if callable(metric_max_keys_probe)
        else frozenset()
    )
    if not isinstance(raw_metric_max_keys, (frozenset, set, list, tuple)) or any(
        not isinstance(key, str) for key in raw_metric_max_keys
    ):
        raise TypeError(
            "policy.training_metric_max_keys must return a collection of public metric keys, "
            f"got {raw_metric_max_keys!r}"
        )
    training_metric_max_keys = frozenset(raw_metric_max_keys)
    rollout_shape = getattr(config.grid, "rollout_shape", None)
    data_replicas = int(rollout_shape()[0]) if callable(rollout_shape) else 1
    prompt_draw_count = (
        prompts_per_data_cell(
            policy_batch_size=int(rl.batch_per_replica),
            rollouts_per_prompt=int(rl.group_size),
            data_replicas=data_replicas,
        )
        if callable(rollout_shape)
        else int(rl.prompts_per_data_replica)
    )
    # DAPO-style dynamic sampling (arXiv 2503.14476 §3.2; rl.min_informative_groups > 0 or
    # rl.min_informative_groups_frac set). The reward stage stamped every complete group's spread, so
    # the replay cells gate on that stamp and the drop decision is per group and
    # placement-independent. Quotas are per cell because assembly is per cell: each cell contributes
    # `per_cell_share` trajectories to the trained batch; the batching layer splits the GLOBAL group
    # count exactly across the data cells. The config resolves the literal count / fraction pair into
    # that ONE count (`GRPOConfig.effective_min_informative_groups`); the loop reads nothing else.
    min_informative_groups = int(rl.effective_min_informative_groups)
    dynamic_sampling = min_informative_groups > 0
    informative_gate = MetadataGate(*INFORMATIVE_GROUP_GATE)
    per_cell_share = prompt_draw_count * int(rl.group_size)
    informative_quotas = informative_quota_per_cell(
        min_informative_groups=min_informative_groups,
        rollouts_per_prompt=int(rl.group_size),
        data_replicas=data_replicas,
    )
    max_extra_rollouts = int(rl.dynamic_sampling_max_extra_rollouts)
    quota_scope = str(rl.dynamic_sampling_quota_scope)
    refill_dispatch = str(rl.dynamic_sampling_refill_dispatch) if dynamic_sampling else "per_round"
    rounds_per_dispatch = max_extra_rollouts if refill_dispatch == "whole_budget" else 1
    refill_request_width = rounds_per_dispatch * prompt_draw_count * int(rl.group_size)
    if dynamic_sampling:
        # The resolved quota is provenance: a literal count copied between batch layouts is a
        # different share of each, and only the log says which share this run trained under.
        info0(
            f"dynamic sampling ON: quota {min_informative_groups} informative groups of "
            f"{int(rl.batch_per_replica) // int(rl.group_size)} per step (rl.min_informative_groups="
            f"{rl.min_informative_groups}, rl.min_informative_groups_frac={rl.min_informative_groups_frac}); "
            f"per-cell quotas in trajectories {list(informative_quotas)}; "
            f"refill_mode={rl.dynamic_sampling_refill_mode} quota_scope={rl.dynamic_sampling_quota_scope}; "
            f"refill_dispatch={refill_dispatch} (budget {max_extra_rollouts} extra rounds, "
            f"{rounds_per_dispatch} per dispatch; per-engine refill request width {refill_request_width} "
            f"sequences = {rounds_per_dispatch} x prompts_per_data_cell {prompt_draw_count} x group_size "
            f"{int(rl.group_size)}, data cells 1:1 to engines)"
        )
    # Trainer iteration counts optimizer steps; the public loop counts collected batches.
    optimizer_step = int(_resolve_identical(policy.iteration(), operation="policy.iteration"))
    if optimizer_step % rl.n_mini:
        raise RuntimeError(
            f"restored optimizer step {optimizer_step} is not a multiple of rl.n_mini={rl.n_mini}: "
            "the checkpoint does not identify a complete collected-batch boundary"
        )
    start_batch = optimizer_step // rl.n_mini
    if start_batch > int(rl.max_steps):
        raise RuntimeError(f"restored batch step {start_batch} exceeds configured total steps {rl.max_steps}")
    if start_batch:
        info0(f"resuming loop at batch step {start_batch}/{rl.max_steps} (optimizer step {optimizer_step})")

    validation = getattr(rl, "validation", None)
    validation_schedule = (
        ValidationSchedule.from_config(validation)
        if validation is not None
        else ValidationSchedule(enabled=False, before_training=False, every_steps=0)
    )
    # The rollout boots from base weights, so resume must publish the restored policy before sampling.
    if validation_schedule.enabled and validation_schedule.every_steps % rl.n_mini:
        raise ValueError(
            f"rl.validation.every_steps={validation_schedule.every_steps} must be divisible by "
            f"rl.n_mini={rl.n_mini}; validation runs only at collected-batch boundaries"
        )
    total_model_steps = int(rl.max_steps) * int(rl.n_mini)
    # Rollout and trainer hand the card back and forth only when they share it and the run asks for
    # it: disjoint GPU pools have nothing to share, and rl.colocated_offload=False keeps both bodies
    # resident on a shared card (the engine was then built without sleep mode, so it is awake).
    offload_cycle = _offload_cycle_enabled(config, rl)
    if not offload_cycle and str(getattr(config.grid, "gpu_placement", "colocated")) == "colocated":
        info0("rl.colocated_offload=False: rollout engine stays resident; per-step sleep/offload cycle is off")
    # Both bodies built resident, except an engine parked for a colocated resume, whose weights the
    # wake below restores. Park the frozen reference and any eager optimizer state before
    # publication, retain policy shards as the IPC source, then yield the rest of the trainer.
    initial_profile = ProfileRecord("grpo_initial_sync")
    trainer_sync_parked_mib = prepare_trainer_weight_sync(policy) if offload_cycle else 0.0
    initial_profile.checkpoint("prepare_trainer_weight_sync")
    if offload_cycle:
        wake_rollout_weights(rollout)
    initial_profile.checkpoint("wake_rollout_weights")
    weight_sync(
        rollout,
        policy,
        validation_step=optimizer_step if validation_schedule.enabled else None,
        verify_replica_values=bool(getattr(rl, "weight_sync_r2r_broadcast", False)),
    )
    initial_profile.checkpoint("weight_sync")
    trainer_offloaded_mib = offload_trainer(policy) if offload_cycle else 0.0
    initial_profile.checkpoint("offload_trainer")
    if offload_cycle:
        wake_rollout(rollout)
    initial_profile.checkpoint("wake_rollout")
    initial_profile.emit(
        trainer_sync_parked_mib=trainer_sync_parked_mib,
        trainer_offloaded_mib=trainer_offloaded_mib,
    )
    reward_provider_identity = {
        "name": "disabled",
        "revision": "disabled",
        "catalog": "disabled",
        "identity_hash": "disabled",
    }
    if validation_schedule.enabled:
        from cosmos_framework.posttrain.algorithm.reward import resolve_reward_provider_catalog

        provider_catalog_path = str(getattr(rl, "reward_provider_catalog", ""))
        provider_catalog = resolve_reward_provider_catalog(provider_catalog_path)
        validation_reward_name = str(getattr(validation, "reward", "") or rl.reward)
        validation_provider_spec = provider_catalog.spec(validation_reward_name)
        reward_provider_identity = {
            "name": validation_reward_name,
            "revision": validation_provider_spec.revision,
            "catalog": provider_catalog_path or "builtin",
            "identity_hash": validation_provider_spec.identity_hash,
        }
    validation_plan = prepare_rollout_validation(
        data=data,
        rollout=rollout,
        reward=rm,
        validation=validation,
        rollout_config=rollout_config,
        training_reward=str(getattr(rl, "reward", "")),
        reward_provider_identity=reward_provider_identity,
        train_total_steps=total_model_steps,
        release_payload=release_validation_payload,
    )
    if validation_plan.sampling is not None:
        sampling = validation_plan.sampling
        info0(
            "validation sampling: "
            f"temperature={sampling['temperature']} top_p={sampling['top_p']} "
            f"top_k={sampling['top_k']} repetition_penalty={sampling['repetition_penalty']} "
            f"n_generation={sampling['n_generation']} max_tokens={sampling['max_tokens']} "
            f"seed={int(validation.seed)}"
        )
    configure_validation = getattr(policy, "configure_validation", None)
    if callable(configure_validation):
        resolve_all(configure_validation(validation_plan.checkpoint_identity))
    elif validation_schedule.enabled:
        raise RuntimeError("policy does not implement durable rollout/reward validation state")

    resumed_probe = getattr(policy, "resumed_from_checkpoint", None)
    resumed_from_checkpoint = (
        bool(_resolve_identical(resumed_probe(), operation="policy.resumed_from_checkpoint"))
        if callable(resumed_probe)
        else optimizer_step > 0
    )
    watch = TerminationWatch()  # Slurm preemption/timeout sentinel + direct signals
    # Reporting is a control-plane concern: the loop is the single writer (rank-0 / driver), fed the
    # all-reduced metrics — console + W&B behind one Reporter concept.
    reporters = make_reporters(
        config,
        out_dir,
        total_steps=total_model_steps,
        headline_keys=console_headline_keys(rl),
    )
    failed_reporters: set[int] = set()
    start_best_effort(reporters, failed_reporters=failed_reporters)

    validation_steps: list[int] = []

    checkpoint_lifecycle = CheckpointLifecycle.from_config(config, rl, run_out_dir=out_dir)

    def acquire_checkpoint_memory(_model_step: int) -> None:
        """Give a checkpoint the whole shared card before trainer state is restored."""
        sleep_rollout(rollout)

    def restore_rollout_after_checkpoint(model_step: int) -> None:
        """Return the shared card to rollout after a successful colocated save."""
        if int(rollout_config.sleep_level) == 2:
            # Level 2 discarded the engine weights. Retain policy shards as their source, republish,
            # then park the policy too. Level 1 kept a host backup and needs only the final wake.
            prepare_trainer_weight_sync(policy)
            wake_rollout_weights(rollout)
            weight_sync(
                rollout,
                policy,
                validation_step=model_step if validation_schedule.enabled else None,
            )
        offload_trainer(policy)
        wake_rollout(rollout)

    checkpoints = checkpoint_lifecycle.coordinator(
        out_dir,
        policy,
        initial_step=optimizer_step,
        observer=lifecycle,
        before_save=acquire_checkpoint_memory if offload_cycle else None,
        after_save=restore_rollout_after_checkpoint if offload_cycle else None,
    )
    transaction = ValidationTransaction(policy, checkpoints, reporters, failed_reporters)
    evaluator = validation_plan.runner
    validation_boundary = ValidationBoundaryCoordinator(
        schedule=validation_schedule,
        total_steps=total_model_steps,
        transaction=transaction,
        evaluator=evaluator.run if evaluator is not None else None,
        should_stop=watch.should_stop,
        publication_step=transaction.pending_publication_step,
        pending_reasons=transaction.pending_reasons,
    )

    history: list[float] = []
    preempted: str | None = None
    # π_old ahead-pass CAPABILITY, resolved across every policy cell once and then pinned.
    # A property of the objective, so it cannot legitimately change mid-run.
    ahead_capable: bool | None = None
    try:
        # The train interval opens inside the region whose `finally` closes it, so a raise anywhere
        # after this point pairs on_train_end with on_train_start (a raise in the setup above opens
        # nothing). A train iteration is one optimizer update; the trajectories consumed so far are the
        # collected batches trained (rl.batch_per_replica is the replica-global batch, whatever the
        # rollout data mesh).
        lifecycle.on_train_start(
            train_iterations_start=optimizer_step,
            train_samples_start=start_batch * int(rl.batch_per_replica),
        )
        validation_outcome = validation_boundary.resume_or_initial(
            optimizer_step,
            resumed_from_checkpoint=resumed_from_checkpoint,
        )
        preempted = validation_outcome.preempted
        if validation_outcome.report is not None:
            validation_steps.append(optimizer_step)
            _log_validation_report(validation_outcome.report, optimizer_step)

        steps_to_run = range(start_batch, rl.max_steps) if preempted is None else ()
        for batch_step in steps_to_run:
            t0 = time.monotonic()
            step_profile = ProfileRecord("grpo_step")
            pool_payload = False
            # Every controller-held payload result this step. The shared cleanup below releases them
            # on the proceed path and on every raise inside the step alike, so a failed step never
            # leaves draws, leftovers, or pooled pages live until outer teardown.
            step_handles: list[Any] = []
            completed_payload: list[Any] = []
            trainer_onloaded_mib = 0.0
            trainer_sync_parked_mib = 0.0
            try:
                # One rollout batch is consumed by n_mini optimizer steps of n_micro chunks each.
                prompts = data.get_samples(prompt_draw_count)
                trajs = rollout.generate(prompts, group=rl.group_size)  # vLLM n=group; seed is engine-level
                scored = rm.score_with_report(trajs)
                step_handles.extend((trajs, scored))
                reward_report = resolve_reward_report(rm.batch_stats(), scored)
                step_profile.checkpoint("experience")
                if reward_report.provider_provenance:
                    info0(
                        f"reward providers batch_step={batch_step} provenance={list(reward_report.provider_provenance)}"
                    )
                teacher_metrics: dict[str, float | int] = {}
                if teacher is not None:
                    # OPD adds credit after reward annotation and before replay; plain GRPO has no teacher.
                    scored = teacher.score(scored)
                    step_handles.append(scored)
                    raw_teacher_metrics = _resolve_identical(
                        teacher.batch_stats(),
                        operation="teacher.batch_stats",
                    )
                    if not isinstance(raw_teacher_metrics, Mapping):
                        raise TypeError(
                            f"teacher.batch_stats must return a tensor-free metric mapping, got {raw_teacher_metrics!r}"
                        )
                    teacher_metrics = {str(key): value for key, value in raw_teacher_metrics.items()}
                # Await every replica before reading replay so a failed write cannot under-fill a batch.
                resolve_all(replay.put_samples(scored))
                extra_rollouts = 0
                refill_dispatches = 0
                idle_refill_cells = 0
                dropped_trajectories = 0
                assembled_informative = 0
                if dynamic_sampling:
                    # Keep drawing whole prompt batches on participating cells until the quota holds at
                    # its configured scope or the extra-rollout budget is spent.
                    collected_reports = [reward_report]
                    short, assembled_informative = _short_cells(
                        replay, informative_gate, informative_quotas, scope=quota_scope, cell_share=per_cell_share
                    )
                    while short and extra_rollouts < max_extra_rollouts:
                        rounds = max_extra_rollouts - extra_rollouts if refill_dispatch == "whole_budget" else 1
                        extra_rollouts += rounds
                        refill_dispatches += 1
                        idle_cells: tuple[int, ...] = ()
                        if rl.dynamic_sampling_refill_mode == "short_cells":
                            idle_cells = tuple(cell for cell in range(data_replicas) if cell not in short)
                            prompts = data.get_samples(
                                scatter(
                                    [
                                        0 if cell in idle_cells else prompt_draw_count * rounds
                                        for cell in range(data_replicas)
                                    ]
                                )
                            )
                        else:
                            prompts = data.get_samples(prompt_draw_count * rounds)
                        trajs = rollout.generate(prompts, group=rl.group_size)
                        step_handles.append(trajs)
                        if idle_cells:
                            scored = rm.score_with_report(
                                trajs, idle=scatter([cell in idle_cells for cell in range(data_replicas)])
                            )
                        else:
                            scored = rm.score_with_report(trajs)
                        step_handles.append(scored)
                        collected_reports.append(resolve_reward_report(rm.batch_stats(), scored, idle_cells=idle_cells))
                        idle_refill_cells += len(idle_cells) * rounds
                        # No teacher pass: OPDConfig rejects rl.min_informative_groups, so a teacher
                        # never coexists with dynamic sampling.
                        resolve_all(replay.put_samples(scored))
                        short, assembled_informative = _short_cells(
                            replay, informative_gate, informative_quotas, scope=quota_scope, cell_share=per_cell_share
                        )
                    if short:
                        if quota_scope == "pooled":
                            shortfall = (
                                f"dynamic sampling batch_step={batch_step}: the assembled batch holds "
                                f"{assembled_informative // int(rl.group_size)} informative groups, below the pooled "
                                f"quota, after {extra_rollouts} extra rollouts"
                            )
                        else:
                            shortfall = (
                                f"dynamic sampling batch_step={batch_step}: replay cells {short} hold fewer "
                                f"informative trajectories than their quotas "
                                f"{[informative_quotas[cell] for cell in short]} after {extra_rollouts} extra rollouts"
                            )
                        shortfall += (
                            f" (refill_dispatch={refill_dispatch}, {refill_dispatches} refill dispatches; "
                            f"groups generated {sum(int(report.groups) for report in collected_reports)}, "
                            f"trained {int(rl.batch_per_replica) // int(rl.group_size)})"
                        )
                        if rl.dynamic_sampling_on_budget_exhausted == "raise":
                            raise RuntimeError(
                                f"{shortfall}; the reward regime cannot supply the quota of {min_informative_groups} "
                                f"informative groups (rl.min_informative_groups={rl.min_informative_groups}, "
                                f"rl.min_informative_groups_frac={rl.min_informative_groups_frac}) within "
                                f"rl.dynamic_sampling_max_extra_rollouts={max_extra_rollouts} (set "
                                "rl.dynamic_sampling_on_budget_exhausted=proceed to train on tied groups as "
                                "zero-advantage filler)"
                            )
                        info0(f"{shortfall}; proceeding with tied groups as zero-advantage filler")
                    # Diagnostics span EVERY draw this step, so the reward curve stays honest about the
                    # extra generation spent; they never feed the drop decision.
                    reward_report = RewardBatchStats.merge(collected_reports)
                    # Leftovers are dead on-policy data; the shared cleanup below drains them.
                    batch = replay.select_prioritized(per_cell_share, informative_gate)
                    dropped_trajectories = int(reward_report.trajectories) - int(rl.batch_per_replica)
                else:
                    batch = replay.get_samples(rl.batch_per_replica)
                step_handles.append(batch)
                # Previews after any top-up, so they sample the collection that formed the trained batch.
                log_sample_previews(reward_report.previews, batch_step)
                step_profile.checkpoint("reward_replay")
                # Every draw for this step is done: hand the card from the engine to the trainer.
                if offload_cycle:
                    sleep_rollout(rollout)
                step_profile.checkpoint("sleep_rollout")
                trainer_onloaded_mib = onload_trainer(policy) if offload_cycle else 0.0
                step_profile.checkpoint("onload_trainer")
                # BOTH halves of the ahead-pass gate: the config says the window needs a frozen π_old
                # (`needs_cached_old_logprobs` = n_mini > 1 and not use_rollout_logprobs_for_loss), and
                # the objective actually has somewhere to put it. `needs_cached_old_logprobs` is inherited
                # by OPDConfig, whose DIRECT objective forms no ratio and carries no cache — gating on
                # config alone aborts that run on its first chunk.
                #
                # `resolve_all`, NOT `resolve`: this is PER-CELL state, and `resolve` returns replica-0
                # rank-0 only. A skewed replica would otherwise let replica 0 disable the pass while
                # another trains with an inactive cache — silently reinstating the unclipped update this
                # whole change exists to remove — or the converse, crashing mid-record. Disagreement is a
                # fatal topology error. Resolved ONCE per run and pinned: a value that changes mid-run
                # means an actor was rebuilt differently, which must not be papered over.
                if ahead_capable is None:
                    caps = resolve_all(policy.needs_old_logprobs()) if rl.needs_cached_old_logprobs else [False]
                    if len(set(map(bool, caps))) > 1:
                        raise RuntimeError(
                            f"policy cells disagree on needs_old_logprobs(): {caps}. Every cell runs the "
                            "same objective, so this means the replicas were not built alike — refusing "
                            "rather than training some cells with a frozen π_old and others without."
                        )
                    ahead_capable = bool(caps[0])
                ahead = rl.needs_cached_old_logprobs and ahead_capable
                if ahead:
                    # π_old AHEAD PASS, before a single parameter moves. At n_mini > 1 this batch drives
                    # n_mini SEQUENTIAL optimizer steps; without a frozen reference the ratio is exp(0)
                    # on every one of them and the epsilon-clip never binds, which measurably destroys
                    # the policy (algorithm/grpo.py::OldLogprobCache). Skipped at n_mini == 1, where
                    # `cur.detach()` IS the collecting policy and this would be pure cost, and under
                    # `use_rollout_logprobs_for_loss`, where the objective reads rollout logprobs and
                    # never consults the cache — see GRPOConfig.needs_cached_old_logprobs.
                    # Same (mini, micro) product, same order as training — the cache is order-keyed.
                    for mini in range(rl.n_mini):
                        for micro in range(rl.n_micro):
                            # `resolve_all`, not `resolve`: a data-dependent failure or a failed
                            # determinism probe on ANY other cell would otherwise be dropped on the floor.
                            # That cell would then enter replay and train against an eval-mode π_old it
                            # never validated — silent surrogate corruption — or raise later and leave its
                            # peers blocked in collectives it never joins. Per-cell state, awaited per cell.
                            resolve_all(
                                policy.compute_old_logprobs(
                                    batch,
                                    mini=mini,
                                    micro=micro,
                                    n_mini=rl.n_mini,
                                    n_micro=rl.n_micro,
                                    ahead_pass=True,  # this call IS the second read
                                )
                            )
                    # Close and rewind the window ONCE, on EVERY cell, before the first train call.
                    # `resolve_all` because a cell that missed the ahead pass must not slip through unseen.
                    resolve_all(policy.begin_replay_window())
                step_profile.checkpoint("ahead")
                training_reports: list[dict[str, Any]] = []
                for mini in range(rl.n_mini):
                    # One lifecycle train step per optimizer update (the mini window); one train batch
                    # per forward/backward (each micro call) — imaginaire's train_iterations/train_batch.
                    # Each interval closes in a `finally` (the checkpoint coordinator's pattern), so a
                    # raise in the train call, its resolve or the skipped-update handling below never
                    # leaves a step or batch open; the happy-path order is unchanged (batch_end still
                    # precedes the skipped-update check), and a raise yields batch_end, step_end, then
                    # the outer `finally`'s train_end.
                    lifecycle.on_train_step_start()
                    try:
                        for micro in range(rl.n_micro):
                            # `ahead_pass` is NOT a selection input — it tells the routing layer the batch is
                            # materialized twice this loop step so it can size the bulk refcount. It must
                            # carry the SAME value as the ahead pass above, or the store pops early (KeyError)
                            # or never drains (leak). Passed ONLY when true, so the call shape is unchanged for
                            # everyone who does not run the pass; `grid.py` reads it with a `False` default.
                            lifecycle.on_train_batch_start()
                            try:
                                metrics = resolve(
                                    policy.train(
                                        batch,
                                        mini=mini,
                                        micro=micro,
                                        n_mini=rl.n_mini,
                                        n_micro=rl.n_micro,
                                        **({"ahead_pass": True} if ahead else {}),
                                    )
                                )
                            finally:
                                lifecycle.on_train_batch_end()
                            if metrics.get("optim/update_outcome") == "skipped":
                                failed_metrics = {
                                    **metrics,
                                    **reward_report.metrics(),
                                    **teacher_metrics,
                                    "timer/e2e_step_s": time.monotonic() - t0,
                                }
                                logger.error(
                                    "rejected GRPO optimizer attempt at batch_step=%d mini=%d micro=%d "
                                    "reason=%s attempt=%s total_skips=%s reward_mean=%s",
                                    batch_step,
                                    mini,
                                    micro,
                                    failed_metrics.get("optim/skip_reason", "nonfinite"),
                                    failed_metrics.get("optim/attempt_id"),
                                    failed_metrics.get("optim/nonfinite_skips_total"),
                                    failed_metrics.get("train/reward_mean"),
                                )
                                raise RuntimeError(
                                    f"rejected non-finite GRPO update at loop step {batch_step}, mini {mini}, "
                                    f"micro {micro}: {metrics.get('optim/skip_reason', 'nonfinite')}; "
                                    f"attempt={metrics.get('optim/attempt_id')} "
                                    f"total_skips={metrics.get('optim/nonfinite_skips_total')}"
                                )
                            training_reports.append(dict(metrics))
                    finally:
                        lifecycle.on_train_step_end()

                step_profile.checkpoint("train")

                if data_plane == "ray_nixl_pool":
                    # Resolve the trained payload so the barrier can clear its page references.
                    completed_payload = resolve_all(batch)
                    step_profile.checkpoint("pool_resolve")
                    pool_payload = has_pool_slices(completed_payload)
                else:
                    step_profile.set_phase("pool_resolve")
            finally:
                # Bounded per-step cleanup, shared by the proceed path and every raise inside the step
                # (budget exhaustion, a rejected update, a failed cell). `drain` is a lease-releasing
                # egress: the replay host releases every leftover BulkRef at its owning store without
                # fetching (filtered bulk never moves) and leftovers never survive into a later step;
                # the pooled-page barrier then returns every producer page the step published.
                if dynamic_sampling:
                    dropped = replay.drain()
                    resolve_all(dropped)
                    release_result_handles(dropped)
                release_result_handles(*step_handles)
                if data_plane == "ray_nixl_pool":
                    release_pool_step_barrier(completed_payload, pool_owner_actors)
                    step_profile.checkpoint("pool_release")
                else:
                    step_profile.set_phase("pool_release")

            metrics = merge_training_metric_window(
                training_reports,
                sum_keys=frozenset(SUMMED_TRAIN_COUNT_KEYS),
                # Maxima from their one declaration point, the objective adapter's `max_reduced` (probed above).
                max_keys=training_metric_max_keys,
                weighted_mean_by=training_metric_weight_map,
            )
            loss = float(metrics["train/loss_avg"])
            # LLMTrainer has already reached rank consensus and reports an applied
            # update here. This remains a control-plane invariant check before weights
            # are published to the rollout engine.
            if not (loss == loss and abs(loss) != float("inf")):
                raise RuntimeError(
                    f"applied update returned non-finite reduced train loss at batch step {batch_step}: {loss}; "
                    "rank-consensus invariant violated"
                )
            model_step = int(_resolve_identical(policy.iteration(), operation="policy.iteration"))
            expected_model_step = (batch_step + 1) * int(rl.n_mini)
            if model_step != expected_model_step:
                raise RuntimeError(
                    f"trainer iteration {model_step} does not match committed model step {expected_model_step}"
                )
            step_profile.checkpoint("metrics")
            normal_save_due = bool(
                rl.save_every and ((batch_step + 1) % rl.save_every == 0 or batch_step + 1 == int(rl.max_steps))
            )
            if offload_cycle and normal_save_due and not validation_boundary.runs_at(model_step):
                # Save while the update still holds the card and the engine is asleep. A save after
                # publication puts the engine back to sleep, which at sleep level 2 discards the weights
                # just published and costs a second publication. A validating boundary keeps its save
                # after publication, where it is merged with the validation checkpoint plan.
                checkpoints.save_configured_if_needed(model_step, window_held=True)
            step_profile.checkpoint("save")
            trainer_sync_parked_mib = prepare_trainer_weight_sync(policy) if offload_cycle else 0.0
            step_profile.checkpoint("prepare_trainer_weight_sync")
            # The policy shards are the only trainer tensors weight sync needs. Park optimizer,
            # gradients and auxiliary models before vLLM remaps its weights; this preserves direct
            # GPU IPC without recreating the post-update OOM peak.
            if offload_cycle:
                wake_rollout_weights(rollout)
            step_profile.checkpoint("wake_rollout_weights")
            weight_sync(
                rollout,
                policy,
                validation_step=model_step if validation_schedule.enabled else None,
            )
            step_profile.checkpoint("weight_sync")
            # Hand the card back to the engine for validation rollouts and the next step's draws.
            # A checkpoint save that still follows (a validating boundary, a disjoint placement, an
            # emergency save) onloads the trainer itself and is offloaded again below.
            trainer_offloaded_mib = offload_trainer(policy) if offload_cycle else 0.0
            step_profile.checkpoint("offload_trainer")
            if offload_cycle:
                wake_rollout(rollout)
            step_profile.checkpoint("wake_rollout")
            history.append(loss)
            metrics = {
                **metrics,
                **reward_report.metrics(),
                **teacher_metrics,
                "timer/e2e_step_s": time.monotonic() - t0,
            }
            if dynamic_sampling:
                metrics["train/grpo_dynamic_sampling_extra_rollouts"] = extra_rollouts
                metrics["train/grpo_dynamic_sampling_refill_dispatches"] = refill_dispatches
                metrics["train/grpo_dynamic_sampling_idle_cells"] = idle_refill_cells
                metrics["train/grpo_dynamic_sampling_avoided_prompts"] = idle_refill_cells * prompt_draw_count
                metrics["train/grpo_dynamic_sampling_batch_informative_groups"] = assembled_informative // int(
                    rl.group_size
                )
                metrics["train/grpo_dynamic_sampling_dropped_trajectories"] = dropped_trajectories
            frame = reportable_metrics(metrics)
            validation_outcome = validation_boundary.at_step(
                model_step,
                frame,
                scheduled_checkpoint_due=normal_save_due,
            )
            if validation_outcome.report is not None:
                validation_steps.append(model_step)
                _log_validation_report(validation_outcome.report, model_step)
            elif validation_outcome.preempted is None:
                emit_metric_frame(reporters, frame, model_step, failed_reporters=failed_reporters)
            if validation_outcome.preempted is not None:
                preempted = validation_outcome.preempted
                break
            if normal_save_due:
                # The coordinator emits the checkpoint lifecycle events, and only for an actual save;
                # a step already saved before publication is not written again.
                checkpoints.save_configured_if_needed(model_step)
                if batch_step + 1 == int(rl.max_steps):
                    # Publication or the save hook restored rollout for the generic checkpoint/validation
                    # contract. Reclaim the shared card before finalization onloads the trainer to drain each
                    # actor-local publisher. Teardown repeats this drain idempotently after the loop.
                    if offload_cycle:
                        sleep_rollout(rollout)
                    CheckpointLifecycle.finalize_units({"policy": policy})
            if offload_cycle:
                # Checkpoint and validation work above onloads the trainer; the next draw needs it gone.
                offload_trainer(policy)
            step_profile.checkpoint("report")
            step_profile.emit(
                step=batch_step + 1,
                data_plane=data_plane,
                has_pool_slices=pool_payload,
                trainer_onloaded_mib=trainer_onloaded_mib,
                trainer_sync_parked_mib=trainer_sync_parked_mib,
                trainer_offloaded_mib=trainer_offloaded_mib,
            )
            # Preemption/timeout: rank-consistent check at the step boundary (a mid-step signal is caught
            # here next step — the >=300s Slurm grace covers one step+save).
            preempted = watch.should_stop()
            if preempted:
                # rl.emergency_checkpoint_components: "full" upgrades to the exact-resume plan; "run"
                # writes the run's own cadence plan (the only save that fits the grace for a policy
                # whose full plan is TiB-scale) — see RunConfig for the trade.
                emergency_plan = str(rl.emergency_checkpoint_components)
                if checkpoints.save_emergency(model_step, components=emergency_plan):
                    logger.warning(f"termination trigger ({preempted}) — emergency checkpoint ({emergency_plan} plan)")
                else:
                    logger.warning(f"termination trigger ({preempted}) — already checkpointed, exiting")
                break
    finally:
        try:
            if offload_cycle:
                # Teardown finalization restores the trainer before draining checkpoint publishers.
                # This cleanup covers successful steps and every exceptional path inside the loop.
                _sleep_rollout_before_trainer_finalization(rollout)
        finally:
            lifecycle.on_train_end()
            if lifecycle is not observers:
                # The loop wrapped a raw observer or iterable, so it owns that list's worker thread; the
                # entry's own list is passed through above and is shut down by the entry's on_app_end.
                lifecycle.shutdown()
            finish_best_effort(reporters, failed_reporters=failed_reporters)
    summary = {
        "steps": rl.max_steps,
        "start": start_batch,
        "start_model_step": optimizer_step,
        "final_loss": history[-1] if history else None,
        "validation_steps": validation_steps,
        **({"preempted": preempted} if preempted else {}),
    }
    info0(f"DONE {summary} resolved={out_dir}/config_resolved.yaml")
    return summary


# --------------------------------------------------------------------------- launch
def run(
    config: Any,
    config_path: str,
    opts: list[str],
    out_dir: str,
    *,
    build_objective_fn: Callable[[Any, Any, Any], Any],
) -> None:
    """Launch the experience and learner ActorGroups, then drive the GRPO loop.

    Args:
        config: Resolved online run config.
        config_path: Config module reloaded inside each worker process.
        opts: Resolved dotlist overrides shared by every role.
        out_dir: Run-attempt output directory.
        build_objective_fn: Objective builder selected by the calling entry.
    """
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.algorithm.reward import resolve_reward_provider_catalog
    from cosmos_framework.posttrain.control.bind import bind_check
    from cosmos_framework.posttrain.control.placement import (
        ActorGroup,
        DeviceGroup,
        bind_device_groups,
        placement_gpu_count,
        trainer_experience_device_groups,
    )

    app_start_time_ms = round(time.time() * 1000)  # the controller's app start, for the lifecycle observers
    from cosmos_framework.posttrain.control.validation.rollout import validate_rollout_reward_sources
    from cosmos_framework.posttrain.utils.determinism import enable_determinism
    from cosmos_framework.posttrain.utils.dist import (
        destroy_process_group_if_initialized,
        init_local_process_group,
    )
    from cosmos_framework.posttrain.utils.entry import pin_run_attempt

    # Pre-flight: a source that cannot feed the configured reward fails here, not after the engine
    # and trainer have loaded.
    rl = instantiate(config.rl)
    catalog = resolve_reward_provider_catalog(str(getattr(rl, "reward_provider_catalog", "")))
    _validate_online_backend(config, rl)
    grid = getattr(config, "grid", None)
    backend_value = getattr(grid, "backend", "local")
    gpu_placement_value = getattr(grid, "gpu_placement", "colocated")
    pair_placement = resolve_gpu_pair_placement(gpu_placement_value, field_name="grid.gpu_placement")
    manifest_relationship: WeightRelationship = "host" if backend_value == "local" else pair_placement.relationship
    offload_cycle = _offload_cycle_enabled(config, rl)
    if rl.deterministic:
        # Before init_local_process_group() (local backend) touches CUDA/NCCL. The ray backend's
        # actors are SEPARATE processes and get their own call inside make_setup()._setup() below —
        # this driver-side call does not reach them.
        enable_determinism()
    report = bind_check(manifest(rl, catalog, weight_relationship=manifest_relationship))
    info0(report.format())
    report.raise_for_status()
    backend, gpu_placement, weight_relationship = resolve_online_weight_placement(
        backend_value,
        pair_placement.gpu_placement,
    )
    validate_online_weight_transport(weight_relationship, rl.weight_sync_transport)
    if weight_relationship != manifest_relationship:
        raise RuntimeError(
            f"GRPO manifest/runtime placement drifted: manifest={manifest_relationship!r}, "
            f"runtime={weight_relationship!r}"
        )
    validate_rollout_reward_sources(rl, catalog)
    reward_spec = catalog.spec(str(rl.reward))
    info0(
        f"reward provider name={rl.reward} revision={reward_spec.revision} "
        f"identity_hash={reward_spec.identity_hash} remote={reward_spec.is_remote}"
    )
    rollout_spec = instantiate(config.rollout)
    tp = int(rollout_spec.tensor_parallel_size)
    configured_shapes = bool(getattr(config.grid, "shapes", None))
    if configured_shapes and backend != "ray":
        raise ValueError("grid.shapes requires grid.backend='ray'")
    if backend == "ray" and not configured_shapes and int(config.grid.num_replicas) != 1:
        raise ValueError(
            f"grid.num_replicas={config.grid.num_replicas} no longer implicitly replicates policy; "
            "set explicit grid.shapes.policy=[1,POLICY_RANKS] and "
            "grid.shapes.rollout=[ROLLOUT_REPLICAS,ROLLOUT_RANKS]"
        )

    local_pg = backend == "local"
    if local_pg:
        # torchrun is the process launcher: join its NCCL world. vLLM TP>1 needs the ray per-replica
        # rendezvous, so local is the TP=1 / test path.
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
        # ray: the model-bearing WGs span the TP mesh (helpers stay ranks=1 — see declare_wgs).
        # GPU placement independently selects a shared IPC or disjoint NIXL weight relationship.
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

    # Mesh-derived checks cannot run in GRPOConfig.__post_init__.
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

    attempt = pin_run_attempt(config, opts_eff, out_dir)
    out_dir, opts_eff = attempt.out_dir, list(attempt.opts)
    from cosmos_framework.posttrain.config.build import prefix_filter_enabled

    rollout_replicas = int(config.grid.num_replicas)
    if rollout_replicas < 1:
        raise ValueError(f"grid.num_replicas={rollout_replicas} must be >= 1")
    logger.info(
        f"run: backend={backend} policy={policy_shape} policy_mesh={policy_parallelism} "
        f"rollout={rollout_shape} gpu_placement={gpu_placement} weight_relationship={weight_relationship} "
        f"weight_transport={rl.weight_sync_transport or ('copy' if local_pg else 'ipc')} "
        f"weight_sync_mode={'p2r+r2r' if rl.weight_sync_r2r_broadcast else 'direct_p2r'} out={out_dir}"
    )

    lifecycle = ControllerLifecycleObservers()
    wgs: dict = {}
    try:
        # provenance=True stamps which framework tree/commit/job produced THIS attempt: the run dir is
        # shared by every requeue and warm continuation, so the config alone cannot tell them apart.
        dump_resolved(config, out_dir, provenance=True)
        # setup() builds the vLLM engine + the FSDP trainer — inside try/finally so a build failure tears
        # the torchrun process group down instead of hanging the surviving ranks.
        wgs = declare_wgs(
            ranks,
            replicas=rollout_replicas,
            rm_needs_bulk=prefix_filter_enabled(config.rl),
            shapes=role_shapes,
            # Ray binder injects this into its K-rank actor world. Local torchrun has no actor-grid
            # K to validate; build_trainer resolves the same model config against the live world.
            policy_parallelism=policy_parallelism if not local_pg else None,
        )
        experience_actor = ActorGroup(
            members=tuple(wgs[name] for name in _EXPERIENCE),
            setup=make_setup(
                config_path,
                opts_eff,
                _EXPERIENCE,
                build_objective_fn=build_objective_fn,
            ),
            name="grpo_experience",
            # vLLM's custom all-reduce maps its TP peers' buffers over CUDA IPC, so the engine's processes
            # see the node's GPUs, their own first; the policy keeps Ray's single-GPU isolation.
            expose_peer_gpus=bool(getattr(rollout_spec, "custom_all_reduce", False)) and not local_pg,
        )
        # The policy owns its process under Ray, so it can pause every allocation it holds while the
        # engine generates; the memory-saver hook has to be in the environment before that process
        # starts. Without the package (or under the local backend) the trainer moves tensors instead.
        # A colocated run that opts out of the hand-off cycle needs no allocator interposition.
        # Disaggregated Ray runs keep main's MemorySaver preload for compatibility even though their
        # loop never calls offload/onload; rl.colocated_offload is inert for those disjoint pools.
        memory_saver_enabled = not local_pg and (gpu_placement != "colocated" or offload_cycle)
        policy_actor = ActorGroup(
            members=(wgs["policy"],),
            setup=make_setup(
                config_path,
                opts_eff,
                ["policy"],
                build_objective_fn=build_objective_fn,
            ),
            name="grpo_policy",
            env_vars=memory_saver_env() if memory_saver_enabled else None,
        )
        if local_pg:
            device_groups = [
                DeviceGroup(
                    actor_groups=(experience_actor, policy_actor),
                    device="cpu",
                    name="grpo_local_cpu",
                )
            ]
        else:
            device_groups = trainer_experience_device_groups(
                experience_actor,
                policy_actor,
                gpu_placement=gpu_placement,
                name="grpo",
            )
        bind_profile = ProfileRecord("grpo_bind")
        bind_device_groups(
            device_groups,
            backend=backend,
            threshold=0,
            weight_sync=True,  # trainer→rollout plane (REAL: separate models, streamed each step)
            weight_sync_r2r_broadcast=rl.weight_sync_r2r_broadcast,
            data_pool_config=_data_pool_config(rl) if backend == "ray" else None,
            weight_sync_transport=rl.weight_sync_transport,
            weight_sync_bucket_bytes=rl.weight_sync_bucket_bytes,
            weight_sync_probe_bytes=rl.weight_sync_probe_bytes,
            weight_sync_min_gbps=rl.weight_sync_min_gbps,
            gpu_node_domains=getattr(config.grid, "gpu_node_domains", None),
        )
        # Engine and trainer are both built and resident here.
        bind_profile.checkpoint("bind_and_setup")
        bind_profile.emit()
        # Lifecycle observers (control/lifecycle.py) read the run's resources from the placement just
        # bound: every GPU device group counts, not a guess from the role shapes.
        lifecycle = build_controller_observers(
            config,
            rl,
            controller="grpo",
            world_size=placement_gpu_count(device_groups) or world_size_or_1(),
            topology=f"POLICY_{policy_shape[0]}x{policy_shape[1]}_ROLLOUT_{rollout_shape[0]}x{rollout_shape[1]}"
            f"_{gpu_placement}",
            policy_dp_world=policy_dp_world,
            app_start_time_ms=app_start_time_ms,
        )
        lifecycle.on_app_start()
        loop(wgs, config, out_dir, observers=lifecycle)
    finally:
        # The observers' app end (their teardown: OneLogger finish/destroy) runs whether or not the
        # checkpoint drain raises; the drain's failure still propagates, it is the run's failure.
        try:
            try:
                if offload_cycle and "rollout" in wgs:
                    # Covers setup failures and loop failures before the loop's own cleanup region.
                    _sleep_rollout_before_trainer_finalization(wgs["rollout"])
            finally:
                CheckpointLifecycle.finalize_units(
                    wgs,
                    destroy_process_group_if_initialized if local_pg else None,
                )
        finally:
            lifecycle.on_app_end()


def _validate_and_exit(config: Any) -> None:
    """Run the GPU-free manifest and source preflight."""
    from cosmos_framework.utils.lazy_config import instantiate

    rl = instantiate(config.rl)
    try:
        from cosmos_framework.posttrain.algorithm.reward import resolve_reward_provider_catalog

        catalog = resolve_reward_provider_catalog(str(getattr(rl, "reward_provider_catalog", "")))
        _validate_online_backend(config, rl)
        grid = getattr(config, "grid", None)
        backend_value = getattr(grid, "backend", "local")
        gpu_placement_value = getattr(grid, "gpu_placement", "colocated")
        pair_placement = resolve_gpu_pair_placement(gpu_placement_value, field_name="grid.gpu_placement")
        manifest_relationship: WeightRelationship = "host" if backend_value == "local" else pair_placement.relationship
        output, ok = render_manifest_validation(manifest(rl, catalog, weight_relationship=manifest_relationship))
    except Exception as exc:  # noqa: BLE001 -- preflight always returns a report and exit code
        output, ok = f"bind_check: error: {exc}", False
    if ok:
        try:
            _backend, _gpu_placement, weight_relationship = resolve_online_weight_placement(
                backend_value,
                pair_placement.gpu_placement,
            )
            validate_online_weight_transport(weight_relationship, rl.weight_sync_transport)
            if weight_relationship != manifest_relationship:
                raise RuntimeError(
                    f"manifest/runtime placement drifted: manifest={manifest_relationship!r}, "
                    f"runtime={weight_relationship!r}"
                )
        except Exception as exc:  # noqa: BLE001 -- preflight always returns a report and exit code
            output = f"{output}\nplacement_check: error: {exc}"
            ok = False
    if ok:
        try:
            from cosmos_framework.posttrain.control.validation.rollout import validate_rollout_reward_sources

            validate_rollout_reward_sources(rl, catalog)
        except Exception as exc:  # noqa: BLE001 -- preflight always returns a report and exit code
            output = f"{output}\nsource_check: error: {exc}"
            ok = False
    print(output)  # persisted to the run log
    raise SystemExit(0 if ok else 1)


def main() -> None:
    from cosmos_framework.utils.config import load_config

    config_path, opts, validate = cli(
        "Cosmos3 online GRPO (vLLM rollout)", default_config="cosmos_framework/posttrain/config/config_grpo.py"
    )
    # Loaded even for manifest preflight, unlike dpo: GRPO's manifest is derived from the selected source and
    # reward. The check itself stays GPU- and cluster-free, which is what makes it a pre-flight.
    config = load_config(config_path, ["--", *opts])
    if validate:
        _validate_and_exit(config)
    run(
        config,
        config_path,
        opts,
        _out_dir(config),
        build_objective_fn=build_objective,
    )


if __name__ == "__main__":
    main()
