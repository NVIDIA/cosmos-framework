# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import hashlib
import json
import pickle
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed
from cosmos_framework.utils.callback import Callback

__all__: tuple[str, ...] = ("DMD2Metrics", "DMD2ParityLedger", "PARITY_KEYS")

PARITY_KEYS: tuple[str, ...] = (
    "vsd_loss",
    "total_generator_loss",
    "fake_score_loss",
    "total_critic_loss",
    "clip_grad_norm/net_selected_preclip",
    "clip_grad_norm/net_selected_clip_scale",
    "clip_grad_norm/net_selected_clip_norm",
    "clip_grad_norm/fake_score_selected_preclip",
    "clip_grad_norm/fake_score_selected_clip_scale",
    "clip_grad_norm/fake_score_selected_clip_norm",
)

_SAMPLE_KEY_FIELDS: tuple[str, ...] = ("__key__", "sample_id", "uuid")
_VOLATILE_INPUT_FIELDS: frozenset[str] = frozenset(
    {
        "_aug_step_times",
        "_aug_time",
        "_pre_aug_time",
        "_sample_time",
        "_worker_aug_step_times",
        "_worker_aug_time",
        "_worker_batch_time",
        "_worker_id",
        "_worker_io_time",
    }
)


def _local_rng_checksum() -> str:
    """Hash local RNG states without advancing any generator."""
    digest = hashlib.sha256()
    torch_state = torch.get_rng_state()  # [N_cpu_rng]
    digest.update(torch_state.numpy().tobytes())
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state()  # [N_cuda_rng]
        digest.update(cuda_state.cpu().numpy().tobytes())
    digest.update(pickle.dumps(np.random.get_state(), protocol=pickle.HIGHEST_PROTOCOL))
    digest.update(pickle.dumps(random.getstate(), protocol=pickle.HIGHEST_PROTOCOL))
    return digest.hexdigest()


def _gather_objects(payload: object) -> list[object]:
    if not dist.is_available() or not dist.is_initialized():
        return [payload]
    return distributed.all_gather_object(payload)


def _flatten_sample_keys(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        local_value = value.detach().cpu()  # [...]
        return [str(item) for item in local_value.reshape(-1).tolist()]
    if isinstance(value, bytes):
        return [value.decode(errors="replace")]
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_sample_keys(item))
        return flattened
    return [str(value)]


def _sample_keys(data_batch: dict[str, object]) -> list[str]:
    local_keys: list[str] = []
    for field in _SAMPLE_KEY_FIELDS:
        if field in data_batch:
            local_keys = _flatten_sample_keys(data_batch[field])
            break
    gathered_keys = _gather_objects(local_keys)
    return sorted(key for rank_keys in gathered_keys for key in _flatten_sample_keys(rank_keys))


def _rng_checksum() -> str:
    gathered_checksums = [str(checksum) for checksum in _gather_objects(_local_rng_checksum())]
    return hashlib.sha256("\n".join(gathered_checksums).encode()).hexdigest()


def _update_input_digest(digest: Any, value: object) -> None:
    if isinstance(value, torch.Tensor):
        to_local = getattr(value, "to_local", None)
        local_value = to_local() if callable(to_local) else value  # [...]
        cpu_value = local_value.detach().cpu().contiguous()  # [...]
        byte_view = cpu_value.view(torch.uint8)  # [N_bytes]
        digest.update(f"tensor:{cpu_value.dtype}:{tuple(cpu_value.shape)}:".encode())
        digest.update(byte_view.numpy().tobytes())
        return
    if isinstance(value, np.ndarray):
        contiguous_value = np.ascontiguousarray(value)
        digest.update(f"ndarray:{contiguous_value.dtype}:{contiguous_value.shape}:".encode())
        digest.update(contiguous_value.tobytes())
        return
    if isinstance(value, dict):
        digest.update(b"dict:")
        for key in sorted(value, key=lambda item: (type(item).__qualname__, str(item))):
            _update_input_digest(digest, key)
            _update_input_digest(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(f"{type(value).__qualname__}:{len(value)}:".encode())
        for item in value:
            _update_input_digest(digest, item)
        return
    if isinstance(value, bytes):
        digest.update(f"bytes:{len(value)}:".encode())
        digest.update(value)
        return
    if isinstance(value, str):
        encoded_value = value.encode()
        digest.update(f"str:{len(encoded_value)}:".encode())
        digest.update(encoded_value)
        return
    digest.update(
        pickle.dumps((type(value).__module__, type(value).__qualname__, value), protocol=pickle.HIGHEST_PROTOCOL)
    )


def _input_digest(data_batch: dict[str, object]) -> str:
    stable_batch = {key: value for key, value in data_batch.items() if key not in _VOLATILE_INPUT_FIELDS}
    local_hasher = hashlib.sha256()
    _update_input_digest(local_hasher, stable_batch)
    local_digest = local_hasher.hexdigest()
    gathered_digests = [str(digest) for digest in _gather_objects(local_digest)]
    return hashlib.sha256("\n".join(gathered_digests).encode()).hexdigest()


def _average_metric(value: object, device: torch.device) -> float:
    if isinstance(value, torch.Tensor):
        to_local = getattr(value, "to_local", None)
        local_value = to_local() if callable(to_local) else value  # [...]
        metric = local_value.detach().float().clone()  # [...]
    else:
        metric = torch.tensor(value, device=device, dtype=torch.float32)  # []
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(metric, op=dist.ReduceOp.AVG)
    return metric.mean().item()


def _phase(model: torch.nn.Module, iteration: int) -> str:
    get_phase = getattr(model, "get_phase", None)
    if callable(get_phase):
        return str(get_phase(iteration))
    get_optimizer_key = getattr(model, "get_optimizer_key", None)
    if callable(get_optimizer_key):
        return "student" if str(get_optimizer_key(iteration)) == "net" else "critic"
    raise AttributeError(f"{type(model).__name__} must define get_phase() or get_optimizer_key()")


class DMD2ParityLedger(Callback):
    """Append deterministic DMD2 parity evidence to rank-0 JSONL."""

    def __init__(self, output_path: str = "") -> None:
        super().__init__()
        self.output_path: str = output_path

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if not self.output_path:
            return

        grad_metrics = getattr(model, "_distillation_parity_grad_metrics", {})
        metric_values = {**output_batch, **grad_metrics}
        record: dict[str, object] = {
            "iteration": iteration,
            "phase": _phase(model, max(iteration - 1, 0)),
            "sample_keys": _sample_keys(data_batch),
            "input_digest": _input_digest(data_batch),
            "rng_checksum": _rng_checksum(),
        }
        for key in PARITY_KEYS:
            if key in metric_values:
                record[key] = _average_metric(metric_values[key], loss.device)

        if not distributed.is_rank0():
            return
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


class DMD2Metrics(DMD2ParityLedger):
    """Public OSS name for the DMD2 parity and diagnostic metrics callback."""
