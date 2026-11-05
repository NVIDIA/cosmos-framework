# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""TEMPORARY vLLM weight-reload compatibility layer.

Delete this module once vLLM accepts canonical checkpoint-format updates directly. Until then it
owns the vLLM-version-specific mechanics needed by the rollout weight sink: live-model application,
receive-buffer ownership for deferred MoE reloads, the official routed-expert reload lifecycle, the
live routed-expert layout (``VllmMoeLayout``), and the packed-QKV live-weight witness. The rollout
body keeps only sync transaction orchestration.

This module is intentionally internal and has no registry or public backend-selection surface.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from cosmos_framework.posttrain.comm.weight_digest import tensor_digest_metadata
from cosmos_framework.posttrain.comm.weight_transfer import WeightWitnessSpec

# Module-level callables keep the vLLM compatibility boundary explicit and work with both the V1
# in-process client used by Posttrain and the V0 direct-model fallback below.
_FUSED_MOE_WIRE_SUFFIXES: tuple[str, ...] = (
    ".mlp.experts.down_proj",
    ".mlp.experts.gate_up_proj",
)


def _owned_checkpoint_weights(
    named: list[tuple[str, torch.Tensor]],  # tensors: [*weight]
) -> list[tuple[str, torch.Tensor]]:  # tensors: [*weight]
    """Own fused MoE inputs that vLLM's layerwise reload retains across stream buckets.

    NIXL carves every fetched bucket as views of one persistent receive buffer. The official vLLM
    reload lifecycle defers a routed-expert ``down_proj`` until the following ``gate_up_proj``
    completes that layer, so retaining the view would let the next RDT transfer overwrite it.
    Dense/QKV weights are consumed synchronously and keep their zero-copy path.
    """
    owned: list[tuple[str, torch.Tensor]] = []
    for name, tensor in named:
        retained = tensor.clone() if name.endswith(_FUSED_MOE_WIRE_SUFFIXES) else tensor  # [*weight]
        owned.append((name, retained))
    return owned


@dataclass(frozen=True)
class VllmMoeLayout:
    """Where the live vLLM engine owns routed-expert checkpoint parameters.

    A mapped destination addresses ``w13_weight``/``w2_weight`` on the module the checkpoint calls
    ``experts`` (the checkpoint-container form the trainer-side mapper emits). The live owner is that
    module itself while ``RoutedExperts`` is vLLM's alias of the ``FusedMoE`` class (vLLM 0.23), or
    the ``routed_experts`` child once ``FusedMoE`` builds a separate ``RoutedExperts`` module (vLLM
    0.24, PR #41184). ``owner_child`` is that attribute path, read from the live owner modules of
    the engine model rather than from a version string; ``owner_names`` are their qualified names.
    """

    owner_child: str
    owner_names: tuple[str, ...]

    def live_destination_names(self, destination_names: Iterable[str]) -> tuple[str, ...]:
        """Resolve checkpoint-container destinations onto the live owner modules."""
        owners = set(self.owner_names)
        resolved: list[str] = []
        for name in destination_names:
            container, _, parameter = name.rpartition(".")
            owner = f"{container}.{self.owner_child}" if self.owner_child else container
            resolved.append(f"{owner}.{parameter}" if owner in owners else name)
        return tuple(resolved)

    def addresses_routed_experts(self, live_names: Iterable[str]) -> bool:
        """Whether any live destination is a parameter of a routed-expert owner module."""
        owners = set(self.owner_names)
        return any(name.rpartition(".")[0] in owners for name in live_names)


def resolve_vllm_moe_layout(owner_names: Iterable[str], destination_names: Iterable[str]) -> VllmMoeLayout | None:
    """Decide the live routed-expert layout from the engine's owner modules.

    Every live owner must be addressed by exactly one mapped checkpoint container: the owner itself
    or its parent module. All owners must agree on that relation, so one engine has one layout.

    Args:
        owner_names: qualified names of the live ``RoutedExperts`` modules (empty for dense engines).
        destination_names: mapper destinations in checkpoint-container form.

    Returns:
        The resolved layout, or ``None`` when the engine has no routed-expert module.
    """
    owners = tuple(sorted(set(owner_names)))
    if not owners:
        return None
    containers = {name.rpartition(".")[0] for name in destination_names}
    children: set[str] = set()
    for owner in owners:
        if owner in containers:
            children.add("")
            continue
        parent, _, child = owner.rpartition(".")
        if parent in containers:
            children.add(child)
            continue
        raise ValueError(f"live routed-expert module {owner!r} is addressed by no mapped weight destination")
    if len(children) != 1:
        raise ValueError(f"live routed-expert modules disagree on their layout: child paths {sorted(children)!r}")
    return VllmMoeLayout(owner_child=children.pop(), owner_names=owners)


def _vllm_routed_experts_class() -> type:
    """Return the class of the live modules that own routed-expert checkpoint parameters.

    ``RoutedExperts`` is exported by the fused-MoE package in both supported layouts: as the alias of
    the ``FusedMoE`` class that owns ``w13_weight``/``w2_weight`` itself (vLLM 0.23), and as the child
    module ``FusedMoE`` builds to own them (vLLM 0.24). The layout is read from the live model
    (``resolve_vllm_moe_layout``), never from the import.
    """
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    return RoutedExperts


def _vllm_routed_expert_modules(model: Any) -> list[Any]:
    """Return only vLLM modules whose checkpoint weights are repacked after initial loading."""
    routed_experts = _vllm_routed_experts_class()
    return [module for module in model.modules() if isinstance(module, routed_experts)]


def _vllm_routed_expert_owner_names(model: Any) -> list[str]:
    """Return the qualified names of the live routed-expert owner modules (control metadata)."""
    routed_experts = _vllm_routed_experts_class()
    return [name for name, module in model.named_modules() if isinstance(module, routed_experts)]


def _vllm_begin_moe_reload(model: Any) -> int:
    """Restore canonical checkpoint-format parameters for live routed-expert modules."""
    from vllm.model_executor.model_loader.reload import initialize_layerwise_reload

    modules = _vllm_routed_expert_modules(model)
    for module in modules:
        initialize_layerwise_reload(module)
    return len(modules)


def _vllm_finish_moe_reload(model: Any, model_dtype: torch.dtype) -> int:
    """Repack reloaded routed experts into the existing backend kernel storage."""
    from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

    if not isinstance(model_dtype, torch.dtype):
        raise TypeError(f"vLLM MoE reload finalization requires a torch.dtype, got {model_dtype!r}")
    # The reload is deliberately scoped to RoutedExperts, whose module tree contains no Attention.
    # vLLM's finalizer accepts ModelConfig only to read `.dtype` in its separate Attention branch.
    finalize_config = SimpleNamespace(dtype=model_dtype)
    modules = _vllm_routed_expert_modules(model)
    for module in modules:
        finalize_layerwise_reload(module, finalize_config)
    return len(modules)


def _vllm_load_weights(
    model: Any,
    named: list[tuple[str, torch.Tensor]],  # tensors: [*weight]
) -> set[str] | None:
    """Load ``named`` full tensors into the in-worker vLLM ``model`` via its own ``load_weights``.

    Args:
        model: the vLLM model living in the EngineCore worker (this TP rank's shard target).
        named: list of ``(hf_name, full_tensor)`` pairs; vLLM slices each into this rank's TP shard.

    Returns:
        Set of ENGINE param names actually loaded (post fusion mapping), or ``None`` if the engine's
        ``load_weights`` doesn't report them — the coverage-check input.
    """
    # vLLM maps each (name, full tensor) → its param and slices per TP rank; returns the set of
    # ENGINE param names it actually loaded (post fusion mapping) — the coverage-check input.
    return model.load_weights(_owned_checkpoint_weights(named))


def _vllm_named_shapes(model: Any) -> list[tuple[str, tuple[int, ...], int]]:
    """Return the bounded live-parameter metadata needed for coverage and receive planning."""
    return [(name, tuple(parameter.shape), parameter.dim()) for name, parameter in model.named_parameters()]


def _verify_qkv_q_tp_shard(
    source_q: torch.Tensor,  # [Q,H]
    live_qkv: torch.Tensor,  # [Q_local+K_local+V_local,H]
    *,
    tp_rank: int,
    tp_world: int,
) -> torch.Tensor:  # returns [Q_local,H]
    """Match the source q-projection TP slice to the q segment of a live fused QKV tensor."""
    if source_q.ndim != 2 or live_qkv.ndim != 2:
        raise ValueError(
            f"packed-QKV witness requires rank-2 source/live weights, got {tuple(source_q.shape)} and "
            f"{tuple(live_qkv.shape)}"
        )
    if tp_world < 1 or not 0 <= tp_rank < tp_world:
        raise ValueError(f"invalid vLLM TP coordinates rank={tp_rank}, world={tp_world}")
    if source_q.shape[0] % tp_world:
        raise ValueError(f"q source rows {source_q.shape[0]} are not divisible by TP world {tp_world}")
    q_rows = source_q.shape[0] // tp_world
    if live_qkv.shape[0] < q_rows or tuple(live_qkv.shape[1:]) != tuple(source_q.shape[1:]):
        raise ValueError(
            f"live qkv shape {tuple(live_qkv.shape)} cannot contain q shard {(q_rows, *source_q.shape[1:])}"
        )
    if live_qkv.dtype != source_q.dtype:
        raise ValueError(f"q source dtype {source_q.dtype} != live qkv dtype {live_qkv.dtype}")
    if live_qkv.device != source_q.device:
        raise ValueError(f"q source device {source_q.device} != live qkv device {live_qkv.device}")
    expected_q = source_q.narrow(0, tp_rank * q_rows, q_rows).contiguous()  # [Q_local,H]
    actual_q = live_qkv.narrow(0, 0, q_rows).contiguous()  # [Q_local,H]
    expected_bytes = expected_q.view(torch.uint8).reshape(-1)  # [N_bytes]
    actual_bytes = actual_q.view(torch.uint8).reshape(-1)  # [N_bytes]
    if not torch.equal(actual_bytes, expected_bytes):
        raise ValueError(f"live qkv q segment does not exactly match q_proj TP shard rank={tp_rank}/{tp_world}")
    return expected_q


# Shared with the vLLM-Omni weight plane (`comm/weight_digest.py`); the underscore name stays for the
# witness tests that import it from here.
_tensor_digest_metadata = tensor_digest_metadata


def _vllm_load_weights_with_witness(
    model: Any,
    named: list[tuple[str, torch.Tensor]],  # tensors: [*weight]
    spec: WeightWitnessSpec,
) -> tuple[set[str] | None, dict[str, Any]]:
    """Load one bucket, then verify its q source against this lane's live packed QKV tensor."""
    owned_named = _owned_checkpoint_weights(named)
    loaded = model.load_weights(owned_named)
    source_by_name = dict(owned_named)
    source_q = source_by_name.get(spec.wire_name)  # [Q,H] or None
    if source_q is None:
        raise ValueError(f"mapped witness source {spec.wire_name!r} is absent from its selected bucket")
    if tuple(source_q.shape) != spec.shape or str(source_q.dtype) != spec.dtype:
        raise ValueError(
            f"mapped witness source metadata changed from {(spec.shape, spec.dtype)!r} to "
            f"{(tuple(source_q.shape), str(source_q.dtype))!r}"
        )
    if spec.layout != "packed_qkv" or spec.packed_part != "q":
        raise ValueError(f"vLLM inline witness supports packed_qkv/q, got {spec.layout!r}/{spec.packed_part!r}")
    get_parameter = getattr(model, "get_parameter", None)
    if not callable(get_parameter):
        raise TypeError(f"live vLLM model {type(model).__name__} exposes no get_parameter()")
    try:
        live_qkv = get_parameter(spec.destination_name)  # [Q_local+K_local+V_local,H]
    except (AttributeError, KeyError) as error:
        raise ValueError(f"live vLLM parameter {spec.destination_name!r} was not found") from error
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    expected_q = _verify_qkv_q_tp_shard(
        source_q,
        live_qkv,
        tp_rank=int(get_tensor_model_parallel_rank()),
        tp_world=int(get_tensor_model_parallel_world_size()),
    )  # [Q_local,H]
    return loaded, _tensor_digest_metadata(expected_q)


def _vllm_live_q_witness(model: Any, destination_name: str, q_rows: int) -> dict[str, Any]:
    """Re-read the final live q segment after every q/k/v contributor has loaded."""
    get_parameter = getattr(model, "get_parameter", None)
    if not callable(get_parameter):
        raise TypeError(f"live vLLM model {type(model).__name__} exposes no get_parameter()")
    try:
        live_qkv = get_parameter(destination_name)  # [Q_local+K_local+V_local,H]
    except (AttributeError, KeyError) as error:
        raise ValueError(f"live vLLM parameter {destination_name!r} was not found") from error
    if live_qkv.ndim != 2 or q_rows < 1 or live_qkv.shape[0] < q_rows:
        raise ValueError(f"live qkv shape {tuple(live_qkv.shape)} cannot expose final q rows {q_rows}")
    live_q = live_qkv.narrow(0, 0, q_rows)  # [Q_local,H]
    return _tensor_digest_metadata(live_q)


def _engine_model(engine: Any) -> Any:
    """Return the V0-only in-process nn.Module behind a vLLM frontend.

    The model lives at ``model_executor.driver_worker.model_runner.model``. It is absent under
    V1, which exposes ``apply_model`` instead.
    """
    return engine.llm_engine.model_executor.driver_worker.model_runner.model


def _apply_to_models(engine: Any, fn: Any) -> list[Any]:
    """Run ``fn`` on every live vLLM engine worker and retain every lane's result."""
    apply_model = getattr(engine, "apply_model", None)
    if apply_model is not None:  # V1: InprocClient dispatches directly to EngineCore
        return list(apply_model(fn))
    return [fn(_engine_model(engine))]  # V0: model in-process


def _apply_to_model(engine: Any, fn: Any) -> Any:
    """Run ``fn(model)`` where the first live vLLM model worker resides and return its result.

    Posttrain configures vLLM V1 with multiprocessing disabled, so ``LLM.apply_model`` reaches the
    live model through ``InprocClient`` without serializing the callable or tensors. vLLM V0 keeps
    its historical direct attribute path.
    """
    return _apply_to_models(engine, fn)[0]


def _vllm_broadcast_live_weights(model: Any, spec: Any, *, verify_values: bool = False) -> str:
    """Run replica broadcast against the underlying model in its owning vLLM process."""
    from cosmos_framework.posttrain.comm.weight_broadcast import broadcast_live_model

    return broadcast_live_model(model, spec, verify_values=verify_values)


def _vllm_initialize_weight_broadcast(model: Any, spec: Any) -> None:
    """Build the R2R side group inside the one vLLM process that owns this cell's model."""
    from cosmos_framework.posttrain.comm.weight_broadcast import initialize_replica_broadcast

    initialize_replica_broadcast(model, spec)
