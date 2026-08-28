# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exact input and training-state parity probes."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
from collections.abc import Iterable

import numpy as np
import torch

_SCHEMA_VERSION = 3


def _to_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a detached, contiguous CPU snapshot of a tensor."""
    try:
        # Rank-filtered distributed probes must not launch a full-tensor
        # collective from only the selected rank. Compare the local FSDP shard.
        if hasattr(tensor, "to_local"):
            tensor = tensor.to_local()  # [*local_shape]
        elif hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()  # [*shape]
    except Exception:  # noqa: BLE001
        pass

    return tensor.detach().to("cpu").contiguous().clone()  # [*shape]


def _fingerprint_cpu(tensor: torch.Tensor) -> dict[str, object]:
    """Return an exact raw-byte hash plus useful numeric summaries."""
    raw_bytes = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    info: dict[str, object] = {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "nbytes": len(raw_bytes),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }
    if tensor.numel() == 0:
        return info

    if tensor.is_floating_point():
        tensor_float = tensor.float()  # [*shape]
        info["sum"] = tensor_float.sum().item()
        info["mean"] = tensor_float.mean().item()
        info["absmax"] = tensor_float.abs().max().item()
    else:
        info["sum"] = int(tensor.to(torch.int64).sum().item())
    return info


def _exact_tensor_record(tensor: torch.Tensor) -> dict[str, object]:
    distribution: dict[str, object] = {}
    device_mesh = getattr(tensor, "device_mesh", None)
    if device_mesh is not None:
        mesh_ranks = device_mesh.mesh.detach().cpu()  # [*mesh_shape]
        distribution = {
            "mesh_dim_names": tuple(device_mesh.mesh_dim_names or ()),
            "mesh_shape": tuple(mesh_ranks.shape),
            "mesh_ranks": mesh_ranks.tolist(),
            "mesh_coordinate": device_mesh.get_coordinate(),
            "placements": tuple(str(placement) for placement in getattr(tensor, "placements", ())),
        }
    tensor_cpu = _to_cpu_tensor(tensor)  # [*shape]
    raw_bytes = tensor_cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "shape": tuple(tensor_cpu.shape),
        "dtype": str(tensor_cpu.dtype),
        "numel": tensor_cpu.numel(),
        "nbytes": len(raw_bytes),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        **distribution,
    }


def _step_in_probe_range(step: int) -> bool:
    minimum = int(os.environ.get("COSMOS_PROBE_MIN_STEP", "0"))
    maximum = int(os.environ.get("COSMOS_PROBE_MAX_STEPS", "100"))
    return minimum <= step < maximum


def _probe_location(step: int, *, rank0_only: bool = False) -> tuple[str, int] | None:
    probe_dir = os.environ.get("COSMOS_PROBE_DIR")
    if not probe_dir or not _step_in_probe_range(step):
        return None
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank0_only and rank != 0:
        return None
    return probe_dir, rank


def _parity_probe_location(step: int) -> tuple[str, int] | None:
    return _probe_location(step, rank0_only=True)


def _deep_probe_enabled(step: int) -> bool:
    return _parity_probe_location(step) is not None


def _loss_probe_location(step: int) -> tuple[str, int] | None:
    return _probe_location(step)


def _small_tensor_record(tensor: torch.Tensor) -> dict[str, object]:
    record = _exact_tensor_record(tensor)
    tensor_cpu = _to_cpu_tensor(tensor)  # [*shape]
    raw_bytes = tensor_cpu.reshape(-1).view(torch.uint8).numpy().tobytes()  # [N_bytes]
    record["values"] = tensor_cpu.tolist()
    record["hex"] = raw_bytes.hex()
    return record


def maybe_dump_loss_reduction(
    *,
    step: int | None,
    tag: str | None,
    valid_counts: torch.Tensor,
    local_loss_sum: torch.Tensor,
    denominator_before: torch.Tensor,
    denominator_after: torch.Tensor,
    final_loss: torch.Tensor,
    exponent: float,
    loss_scaling_factor: float,
    world_size: int,
) -> None:
    """Save all local and globally reduced weighted-CE scalars on selected ranks."""
    if step is None or tag is None:
        return
    location = _loss_probe_location(step)
    if location is None:
        return
    probe_dir, rank = location
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "step": step,
        "rank": rank,
        "tag": tag,
        "valid_counts": _small_tensor_record(valid_counts),
        "local_loss_sum": _small_tensor_record(local_loss_sum),
        "denominator_before": _small_tensor_record(denominator_before),
        "denominator_after": _small_tensor_record(denominator_after),
        "final_loss": _small_tensor_record(final_loss),
        "exponent": exponent,
        "loss_scaling_factor": loss_scaling_factor,
        "world_size": world_size,
    }
    os.makedirs(probe_dir, exist_ok=True)
    path = os.path.join(probe_dir, f"step{step:05d}_rank{rank:03d}_loss_reduction.pt")
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _update_parity_record(step: int, tag: str, section: str, value: object) -> None:
    location = _parity_probe_location(step)
    if location is None:
        return
    probe_dir, rank = location
    os.makedirs(probe_dir, exist_ok=True)
    path = os.path.join(probe_dir, f"step{step:05d}_rank{rank:03d}_parity.pt")
    payload = (
        torch.load(path, map_location="cpu", weights_only=False)
        if os.path.exists(path)
        else {"schema_version": _SCHEMA_VERSION, "step": step, "rank": rank, "tag": tag, "sections": {}}
    )
    payload["sections"][section] = value
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _hash_named_tensors(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    include_requires_grad: bool = False,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, tensor in named_tensors:
        tensor_record = _exact_tensor_record(tensor)
        if include_requires_grad:
            tensor_record["requires_grad"] = tensor.requires_grad
        result[name] = tensor_record
    return result


def _rng_record() -> dict[str, object]:
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []  # [N_devices][N_state]
    return {
        "python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
        "torch_cpu": _exact_tensor_record(torch.get_rng_state()),
        "torch_cuda": [_exact_tensor_record(state) for state in cuda_states],
    }


def _config_record(model: torch.nn.Module) -> dict[str, object]:
    config = getattr(model, "config", None)
    config_value = config.to_dict() if hasattr(config, "to_dict") else repr(config)
    serialized = json.dumps(config_value, sort_keys=True, default=str).encode()
    return {
        "class": type(model).__qualname__,
        "config_sha256": hashlib.sha256(serialized).hexdigest(),
        "training": model.training,
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
    }


def maybe_dump_pre_forward(
    model: torch.nn.Module,
    kwargs: dict[str, object],
    step: int | None,
    tag: str | None,
) -> None:
    """Hash effective raw-model inputs, model state, configuration, and RNG."""
    if step is None or tag is None or not _deep_probe_enabled(step):
        return
    effective_inputs = {
        key: _exact_tensor_record(value) if isinstance(value, torch.Tensor) else repr(value)
        for key, value in kwargs.items()
        if value is not None
    }
    _update_parity_record(
        step,
        tag,
        "pre_forward",
        {
            "effective_inputs": effective_inputs,
            "parameters": _hash_named_tensors(model.named_parameters(), include_requires_grad=True),
            "buffers": _hash_named_tensors(model.named_buffers()),
            "rng": _rng_record(),
            "model": _config_record(model),
        },
    )


def maybe_dump_forward_result(
    logits: torch.Tensor,
    losses: dict[str, torch.Tensor],
    step: int,
    tag: str,
) -> None:
    """Hash logits and every scalar loss participating in backward."""
    if not _deep_probe_enabled(step):
        return
    _update_parity_record(
        step,
        tag,
        "forward_result",
        {
            "logits": _exact_tensor_record(logits),
            "losses": {name: _exact_tensor_record(loss) for name, loss in losses.items()},
        },
    )


def maybe_dump_gradients(model: torch.nn.Module, step: int, tag: str) -> None:
    """Hash every parameter gradient before clipping or optimizer mutation."""
    if not _deep_probe_enabled(step):
        return
    gradients = {
        name: None if parameter.grad is None else _exact_tensor_record(parameter.grad)
        for name, parameter in model.named_parameters()
    }
    _update_parity_record(step, tag, "gradients", gradients)


def maybe_dump_post_clip(
    model: torch.nn.Module,
    total_norm: torch.Tensor,
    max_norm: float,
    step: int,
    tag: str,
) -> None:
    """Hash the exact clipping scalar and every post-clipping gradient."""
    if not _deep_probe_enabled(step):
        return
    clip_coefficient = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)  # []
    norm_cpu = _to_cpu_tensor(total_norm)  # []
    gradients = {
        name: None if parameter.grad is None else _exact_tensor_record(parameter.grad)
        for name, parameter in model.named_parameters()
    }
    _update_parity_record(
        step,
        tag,
        "post_clip",
        {
            "total_norm": _exact_tensor_record(norm_cpu),
            "max_norm": max_norm,
            "clip_coefficient": _exact_tensor_record(clip_coefficient),
            "gradients": gradients,
        },
    )


def _iter_optimizers(optimizer: object) -> Iterable[object]:
    children = getattr(optimizer, "optimizers", None)
    if children is None:
        yield optimizer
        return
    for child in children:
        if isinstance(child, (list, tuple)):
            yield from child
        else:
            yield child


def _optimizer_record(model: torch.nn.Module, optimizer: object) -> dict[str, object]:
    parameter_name = {id(parameter): name for name, parameter in model.named_parameters()}
    state: dict[str, object] = {}
    for inner_optimizer in _iter_optimizers(optimizer):
        for parameter, parameter_state in getattr(inner_optimizer, "state", {}).items():
            name = parameter_name.get(id(parameter), f"<unmapped:{id(parameter)}>")
            state[name] = {
                key: _exact_tensor_record(value) if isinstance(value, torch.Tensor) else repr(value)
                for key, value in parameter_state.items()
            }
    return state


def maybe_dump_post_optimizer(
    model: torch.nn.Module,
    optimizer: object,
    step: int,
    tag: str,
) -> None:
    """Hash post-step parameters and Adam state independently of container layout."""
    if not _deep_probe_enabled(step):
        return
    _update_parity_record(
        step,
        tag,
        "post_optimizer",
        {
            "parameters": _hash_named_tensors(model.named_parameters(), include_requires_grad=True),
            "optimizer_state": _optimizer_record(model, optimizer),
        },
    )


def maybe_dump_model_inputs(
    kwargs: dict[str, object],
    step: int,
    tag: str,
    *,
    labels: torch.Tensor | None = None,
) -> None:
    """Dump forward kwargs for ``step`` for trainer label if the probe is enabled.

    Args:
        kwargs: the dict expanded into the model forward (before any wrapper
            filtering).
        step: current optimizer step / iteration.
        tag: trainer label, stored as metadata.
        labels: supervision tensor consumed by the external CE loss. It is not
            a model-forward kwarg, but is required for complete loss parity.
    """
    location = _probe_location(step)
    if location is None:
        return
    probe_dir, rank = location
    probe_values = dict(kwargs)
    auxiliary_keys: list[str] = []
    if labels is not None:
        probe_values["labels"] = labels
        auxiliary_keys.append("labels")

    record: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "step": int(step),
        "rank": int(rank),
        "tag": tag,
        "keys": sorted(probe_values),
        "forward_keys": sorted(kwargs),
        "auxiliary_keys": auxiliary_keys,
    }
    for key, value in probe_values.items():
        if isinstance(value, torch.Tensor):
            try:
                tensor_cpu = _to_cpu_tensor(value)
                record[key] = _fingerprint_cpu(tensor_cpu)
            except Exception as e:  # noqa: BLE001
                record[key] = {"error": repr(e)}
        else:
            record[key] = {"type": type(value).__name__, "value": repr(value)}

    ii = kwargs.get("input_ids")
    if isinstance(ii, torch.Tensor) and ii.ndim >= 2:
        record["seq_len"] = int(ii.shape[1])

    os.makedirs(probe_dir, exist_ok=True)
    path = os.path.join(probe_dir, f"step{step:05d}_rank{rank:03d}.pt")
    tmp = path + ".tmp"
    torch.save({"record": record, "tensors": {}}, tmp)
    os.replace(tmp, path)
