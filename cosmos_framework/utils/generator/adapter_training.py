# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Installing adapters into a model builder's loaded base, single-process or distributed.

Training entry points call :func:`validate_adapter_dtypes` from their configuration. Model builders call
:func:`validate_adapter_build` before construction, :func:`check_full_base_fits` before materializing the
base, and :func:`prepare_adapter_model` after the base weights are loaded and before FSDP registers
parameters. :func:`adapter_optimizer_groups` derives the optimizer groups.
"""

from __future__ import annotations

import zlib
from typing import Any, NamedTuple

import torch
import torch.nn as nn

from cosmos_framework.utils.generator.adapter_config import AdapterIdentity, AdapterSetup, adapter_architecture
from cosmos_framework.utils.generator.adapters import (
    AdapterLinear,
    assert_trainability,
    canonical_name,
    install_adapters,
    installed_adapter_identity,
    installed_adapter_manifest,
    model_base_fingerprint,
    require_adapter_calls,
    trainable_spec,
)

# The full base is loaded on every rank before FSDP shards it; it must leave room for the rest of the step.
FULL_BASE_DEVICE_FRACTION = 0.9


class AdapterOptimizerGroup(NamedTuple):
    """One optimizer parameter group of the adapter parameters."""

    name: str
    parameter_names: tuple[str, ...]
    lr: float
    weight_decay: float


def validate_adapter_dtypes(
    model_type: str | None, *, storage_dtype: str, compute_dtype: str | None, reduce_dtype: str
) -> None:
    """Architecture and precision checks for adapter training, from a configuration.

    ``compute_dtype`` is the FSDP parameter dtype when it is configured separately from storage
    (``None`` otherwise).
    """
    adapter_architecture(model_type)
    if storage_dtype not in ("float32", "bfloat16"):
        raise ValueError(f"adapters require float32 or bfloat16 base storage, got {storage_dtype!r}")
    if compute_dtype is not None and compute_dtype != storage_dtype:
        raise ValueError(
            f"adapters keep the frozen base in one dtype: storage ({storage_dtype}) must equal compute "
            f"({compute_dtype}); adapter parameters are float32 regardless"
        )
    if reduce_dtype != "float32":
        raise ValueError(
            f"adapter gradients are reduced in float32; set the reduce dtype to float32, got {reduce_dtype!r}"
        )


def validate_adapter_build(
    model_config: Any, *, dtype: str, reduce_dtype: str, compute_dtype: str | None = None
) -> None:
    """The builder-side checks (builders are also called directly): quantization, architecture, dtypes."""
    if getattr(model_config, "quantization_config", None) is not None:
        raise ValueError("adapters require a dense float32/bfloat16 base; quantized bases are not supported")
    validate_adapter_dtypes(
        getattr(model_config, "model_type", None),
        storage_dtype=dtype,
        compute_dtype=compute_dtype,
        reduce_dtype=reduce_dtype,
    )


def full_base_bytes(net: nn.Module) -> int:
    """Bytes of every distinct parameter and buffer, counting a tied output embedding once.

    Before materialization a tied output embedding may still be a separate tensor; it is counted
    only once either way.
    """
    seen: set[int] = set()
    total = 0
    input_embeddings = getattr(net, "get_input_embeddings", lambda: None)()
    output_embeddings = getattr(net, "get_output_embeddings", lambda: None)()
    tied = bool(getattr(getattr(net, "config", None), "tie_word_embeddings", False))
    if (
        tied
        and input_embeddings is not None
        and output_embeddings is not None
        and output_embeddings.weight is not input_embeddings.weight
    ):
        seen.add(id(output_embeddings.weight))  # Tied later; it will share the input embedding.
    for tensor in [*net.parameters(), *net.buffers()]:
        if id(tensor) not in seen:
            seen.add(id(tensor))
            total += tensor.numel() * tensor.element_size()
    return total


def check_full_base_fits(num_bytes: int, device_memory: int | None = None) -> None:
    """Refuse a base whose unsharded copy would not fit on one device (every rank loads it before sharding).

    ``device_memory`` defaults to the current CUDA device's total memory; without CUDA nothing is checked.
    """
    if device_memory is None:
        if not torch.cuda.is_available():
            return
        device_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    if num_bytes > FULL_BASE_DEVICE_FRACTION * device_memory:
        raise ValueError(
            f"adapters load the full base on every rank before FSDP shards it: {num_bytes / 2**30:.1f} GiB does not fit "
            f"in {FULL_BASE_DEVICE_FRACTION:.0%} of the {device_memory / 2**30:.1f} GiB device; store the base in "
            "bfloat16 or use devices with more memory"
        )


def _distributed(group: Any) -> tuple[bool, int, int]:
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return False, 1, 0
    return True, dist.get_world_size(group), dist.get_rank(group)


def _agree(error: BaseException | None, values: list[int], device: torch.device, group: Any, what: str) -> list[int]:
    """All-reduce (MAX) a failure flag and ``values``; raise on every rank if any rank failed.

    Returns the maximum of each value over the ranks. Single-process callers just re-raise.
    """
    import torch.distributed as dist

    if not _distributed(group)[0]:
        if error is not None:
            raise error
        return values
    state = torch.tensor([0 if error is None else 1, *values], dtype=torch.int64, device=device)
    dist.all_reduce(state, op=dist.ReduceOp.MAX, group=group)
    if error is not None:
        raise error
    if state[0].item():
        raise RuntimeError(f"{what} failed on another rank")
    return [int(value) for value in state[1:].tolist()]


def _global_rank(group: Any, rank: int) -> int:
    import torch.distributed as dist

    return dist.get_global_rank(group, rank) if group is not None else rank


def _initialization_cost(module: AdapterLinear) -> int:
    out_features, in_features = module.weight.shape
    # PiSSA's SVD dominates; LoRA/DoRA initialization is linear in the weight size.
    return out_features * in_features * (min(out_features, in_features) if module.method == "pissa" else 1)


def _assign_owners(modules: list[tuple[str, AdapterLinear]], world: int) -> list[list[tuple[str, AdapterLinear]]]:
    """Largest-first onto the least-loaded rank; deterministic, so identical on every rank."""
    owned: list[list[tuple[str, AdapterLinear]]] = [[] for _ in range(world)]
    load = [0] * world
    for name, module in sorted(modules, key=lambda item: (-_initialization_cost(item[1]), item[0])):
        owner = min(range(world), key=lambda rank: (load[rank], rank))
        owned[owner].append((name, module))
        load[owner] += _initialization_cost(module)
    return [sorted(items, key=lambda item: item[0]) for items in owned]


@torch.no_grad()
def initialize_adapters(net: nn.Module, *, group: Any = None) -> None:
    """Initialize every installed adapter once and give all ranks of ``group`` the same values.

    Targets are spread over the ranks by initialization cost (PiSSA's SVD is the expensive part).
    Each target draws from its own generator, seeded by rank 0's process seed (``torch.initial_seed``)
    and the target name, so the result does not depend on the world size and the global RNG state is
    neither read beyond that seed nor advanced. A failure on any rank, or ranks disagreeing on the
    adapter tensors, raises on every rank. Each owner then broadcasts its targets in one flattened
    collective.
    """
    import torch.distributed as dist

    modules = sorted(
        ((canonical_name(name), m) for name, m in net.named_modules() if isinstance(m, AdapterLinear)),
        key=lambda item: item[0],
    )
    if not modules:
        return
    distributed, world, rank = _distributed(group)
    device = modules[0][1].lora_A.weight.device
    error, sizes, seed, plan = None, 0, 0, []
    try:
        sizes = sum(t.numel() for _, module in modules for t in module.adapter_tensors())
        if rank == 0:
            seeds = torch.Generator().manual_seed(torch.initial_seed())
            seed = int(torch.randint(0, 2**62, (1,), generator=seeds).item())
        plan = _assign_owners(modules, world)
    except BaseException as caught:  # noqa: BLE001 -- every rank must leave the collective together
        error = caught
    largest, negative_smallest, seed = _agree(error, [sizes, -sizes, seed], device, group, "adapter initialization")
    if largest != -negative_smallest:
        raise RuntimeError("adapter initialization: ranks installed different adapter tensors")
    error = None
    try:
        for name, module in plan[rank]:
            generator = torch.Generator(device=module.weight.device)
            generator.manual_seed((seed + zlib.crc32(name.encode())) % 2**63)
            module.initialize(generator)
    except BaseException as caught:  # noqa: BLE001
        error = caught
    _agree(error, [], device, group, "adapter initialization")
    if not distributed:
        return
    for owner, owned in enumerate(plan):
        if not owned:
            continue  # Identical on every rank: no collective for an owner without targets.
        tensors = [t for _, module in owned for t in module.adapter_tensors()]
        flat = torch.cat([t.reshape(-1) for t in tensors])
        dist.broadcast(flat, src=_global_rank(group, owner), group=group)
        if owner != rank:
            for tensor, value in zip(tensors, flat.split([t.numel() for t in tensors])):
                tensor.copy_(value.view_as(tensor))


def prepare_adapter_model(
    net: nn.Module, setup: AdapterSetup, *, base_dtype: str, architecture: str, group: Any = None
) -> AdapterIdentity:
    """Install adapters into a loaded, not yet sharded model and attach its adapter identity.

    Fresh adapters are initialized with :func:`initialize_adapters`; a checkpoint restore keeps
    placeholders; a warm start loads the portable adapter (which must match the base fingerprint).
    A failure in any of these steps on one rank raises on every rank. Returns the identity, also
    stored as ``net.adapter_identity``.
    """
    from cosmos_framework.utils.generator.adapter_artifacts import adapter_identity, load_adapter_artifact

    device = next(net.parameters()).device
    error = None
    try:
        manifest = install_adapters(net, setup.config, architecture=architecture, initialize=False)
        net.adapter_identity = adapter_identity(
            manifest, base_dtype=base_dtype, base_fingerprint=model_base_fingerprint(net, manifest)
        )
        if setup.warm_start:
            load_adapter_artifact(net, setup.warm_start)
        require_adapter_calls(net)
    except BaseException as caught:  # noqa: BLE001 -- every rank must leave the collective together
        error = caught
    _agree(error, [], device, group, "adapter installation")
    if not setup.warm_start and not setup.from_checkpoint:
        initialize_adapters(net, group=group)
    return installed_adapter_identity(net)


def adapter_optimizer_groups(net: nn.Module, lr: float, weight_decay: float) -> list[AdapterOptimizerGroup]:
    """One group per distinct ``(lr_multiplier, weight_decay)`` of the adapter settings.

    A component's ``weight_decay=None`` inherits ``weight_decay``; DoRA magnitudes are not decayed. Groups
    with ``lr_multiplier == 1`` come first, so the first group trains at ``lr`` whenever any target does.
    """
    manifest = installed_adapter_manifest(net)
    assert_trainability(net, manifest)
    spec = trainable_spec(manifest)
    groups: dict[tuple[float, float], list[str]] = {}
    for name, parameter in net.named_parameters():
        if parameter.requires_grad:
            multiplier, decay = spec[canonical_name(name)]
            groups.setdefault((multiplier, weight_decay if decay is None else decay), []).append(name)
    ordered = sorted(groups.items(), key=lambda item: item[0][0] != 1.0)  # stable: base-rate groups first
    return [
        AdapterOptimizerGroup(f"adapter_{index}", tuple(names), lr * multiplier, decay)
        for index, ((multiplier, decay), names) in enumerate(ordered)
    ]
