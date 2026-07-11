# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Safe checkpoint loading helpers for tokenizer code."""

from __future__ import annotations

import io
import os
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf.nodes import AnyNode

from cosmos_framework.utils.easy_io import easy_io

TorchMapLocation = str | torch.device | Callable[[Any, str], Any] | dict[str, str] | None
DCP_MODEL_LOAD_INFO_KEY = "dcp_model_load_info"
_RQ_CODEBOOK_STAGE_KEY_PATTERN = re.compile(r"(?:^|\.)quantizer\.codebooks\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class DCPModelLoadInfo:
    """Model-key coverage discovered from one DCP component's metadata."""

    loaded_model_keys: frozenset[str]
    missing_model_keys: frozenset[str]
    unexpected_checkpoint_keys: frozenset[str]


def infer_rq_codebook_depth(state_keys: Iterable[str], *, source: str) -> int | None:
    """Infer a contiguous residual-quantizer depth from state-dict keys."""
    stage_indices = {
        int(match.group(1))
        for key in state_keys
        if (match := _RQ_CODEBOOK_STAGE_KEY_PATTERN.search(str(key))) is not None
    }
    if not stage_indices:
        return None

    inferred_depth = max(stage_indices) + 1
    expected_indices = set(range(inferred_depth))
    if stage_indices != expected_indices:
        raise ValueError(
            f"Residual-quantizer stages in {source} are not contiguous from zero: "
            f"found {sorted(stage_indices)}, expected {sorted(expected_indices)}."
        )
    return inferred_depth


def get_model_state_keys(model: Any) -> set[str]:
    """Return model-state names without constructing a mapping for torch modules."""
    if isinstance(model, nn.Module):
        parameter_keys = {name for name, _param in model.named_parameters(remove_duplicate=False)}
        buffer_keys = {name for name, _buffer in model.named_buffers(remove_duplicate=False)}
        return parameter_keys | buffer_keys

    state_dict = getattr(model, "state_dict", None)
    return set(state_dict()) if callable(state_dict) else set()


def validate_rq_checkpoint_depth(
    model: nn.Module,
    checkpoint_keys: Iterable[str],
    *,
    checkpoint_path: str,
    model_keys: Iterable[str] | None = None,
) -> None:
    """Reject RQ checkpoints whose persisted depth disagrees with the instantiated model."""
    resolved_model_keys = get_model_state_keys(model) if model_keys is None else model_keys
    model_depth = infer_rq_codebook_depth(resolved_model_keys, source="the instantiated model")
    checkpoint_depth = infer_rq_codebook_depth(
        checkpoint_keys,
        source=f"checkpoint {checkpoint_path}",
    )
    if model_depth is None or checkpoint_depth is None or model_depth == checkpoint_depth:
        return

    raise ValueError(
        f"Residual-quantizer depth mismatch for checkpoint {checkpoint_path}: "
        f"the instantiated model has {model_depth} stage(s), but the checkpoint contains {checkpoint_depth}. "
        "Historical UniAE configs may record quantizer_num_codebooks=1 even though their checkpoints contain "
        "four stages; migrate the config to the checkpoint depth before loading."
    )


def _build_s3_backend_args(checkpoint_path: str, s3_credential: str | None) -> dict[str, str] | None:
    """Build easy_io backend args for an S3 checkpoint when explicit credentials are provided."""
    if not checkpoint_path.startswith("s3://") or s3_credential is None:
        return None
    return {
        "backend": "s3",
        "s3_credential_path": s3_credential,
    }


def _safe_checkpoint_globals() -> list[Any]:
    """Return trusted non-tensor classes present in tokenizer training checkpoints."""
    return [
        Any,
        AnyNode,
        ContainerMetadata,
        DictConfig,
        ListConfig,
        Metadata,
        defaultdict,
        dict,
        int,
        list,
    ]


def load_torch_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: TorchMapLocation = "cpu",
    s3_credential: str | None = None,
    backend_args: dict[str, Any] | None = None,
    backend_key: str | None = None,
) -> Any:
    """Load a torch checkpoint through easy_io with safe tensor-only unpickling."""
    checkpoint_path_str = str(checkpoint_path)
    resolved_backend_args = (
        backend_args if backend_args is not None else _build_s3_backend_args(checkpoint_path_str, s3_credential)
    )
    with torch.serialization.safe_globals(_safe_checkpoint_globals()):
        return easy_io.load(
            checkpoint_path_str,
            backend_args=resolved_backend_args,
            backend_key=backend_key,
            map_location=map_location,
            weights_only=True,
        )


def load_torch_checkpoint_from_bytes(
    checkpoint_bytes: bytes,
    *,
    map_location: TorchMapLocation = "cpu",
) -> Any:
    """Load a torch checkpoint from bytes with safe tensor-only unpickling."""
    with io.BytesIO(checkpoint_bytes) as checkpoint_buffer:
        with torch.serialization.safe_globals(_safe_checkpoint_globals()):
            return torch.load(checkpoint_buffer, map_location=map_location, weights_only=True)


def load_torch_checkpoint_from_easy_io_backend(
    backend: Any,
    checkpoint_path: str,
    *,
    map_location: TorchMapLocation = "cpu",
) -> Any:
    """Load one checkpoint through an already-configured easy_io backend."""
    return load_torch_checkpoint_from_bytes(
        backend.get(filepath=checkpoint_path),
        map_location=map_location,
    )


def normalize_dcp_model_checkpoint_path(checkpoint_path: str | Path) -> str:
    """Map a trainer iteration directory or URI to its DCP model component."""
    checkpoint_path_str = str(checkpoint_path).rstrip("/")
    model_component_path = os.path.join(checkpoint_path_str, "model")
    if os.path.isdir(model_component_path):
        return model_component_path
    if checkpoint_path_str.startswith("s3://") and re.search(r"/iter_\d{9}$", checkpoint_path_str):
        return model_component_path
    return checkpoint_path_str


def _cpu_state_placeholder(value: Any) -> Any:
    """Create a CPU DCP target without duplicating source-device storage."""
    if isinstance(value, torch.Tensor):
        return torch.empty_like(value, device="cpu")  # [...]
    return value


def _add_dcp_ema_template_tensors(
    model: nn.Module,
    model_state: dict[str, Any],
    metadata_keys: set[str],
) -> None:
    """Request EMA buffers from DCP when checkpoint metadata says they exist."""
    from cosmos_framework.utils.ema import get_buffer_name

    for name, param in model.named_parameters():
        ema_key = f"ema.{get_buffer_name(name)}"
        if ema_key not in metadata_keys or ema_key in model_state:
            continue
        model_state[ema_key] = torch.empty_like(param.detach(), dtype=torch.float32, device="cpu")  # [...]


def load_dcp_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    include_ema: bool = False,
    allow_partial: bool = False,
    checkpoint_key_prefix: str = "",
    s3_credential: str | None = None,
) -> dict[str, Any]:
    """Load model state from a local or S3 torch.distributed.checkpoint component."""
    from torch.distributed.checkpoint import load
    from torch.distributed.checkpoint.filesystem import FileSystemReader

    checkpoint_path_str = normalize_dcp_model_checkpoint_path(checkpoint_path)

    def _build_storage_reader() -> Any:
        if checkpoint_path_str.startswith("s3://"):
            if not s3_credential:
                raise ValueError("s3_credential is required to load a remote DCP checkpoint.")
            from cosmos_framework.checkpoint.s3_filesystem import S3StorageReader

            return S3StorageReader(
                credential_path=s3_credential,
                path=checkpoint_path_str,
            )
        return FileSystemReader(checkpoint_path_str)

    storage_reader = _build_storage_reader()
    state_dict_metadata = storage_reader.read_metadata().state_dict_metadata
    metadata_keys = set(state_dict_metadata)
    if include_ema and checkpoint_key_prefix:
        raise ValueError("checkpoint_key_prefix cannot be combined with include_ema.")

    current_model_state = model.state_dict()
    checkpoint_key_by_model_key = {
        model_key: f"{checkpoint_key_prefix}{model_key}" for model_key in current_model_state
    }
    expected_checkpoint_keys = set(checkpoint_key_by_model_key.values())
    wrapped_matches = sum(f"model.{key}" in metadata_keys for key in expected_checkpoint_keys)
    raw_matches = sum(key in metadata_keys for key in expected_checkpoint_keys)
    if wrapped_matches == 0 and raw_matches == 0:
        raise ValueError(
            f"DCP checkpoint {checkpoint_path_str} does not contain tensors matching the requested model "
            f"(checkpoint sample: {sorted(metadata_keys)[:5]}; model sample: {sorted(expected_checkpoint_keys)[:5]})."
        )

    metadata_prefix = "model." if wrapped_matches >= raw_matches else ""
    component_metadata = {
        key.removeprefix(metadata_prefix): value
        for key, value in state_dict_metadata.items()
        if not metadata_prefix or key.startswith(metadata_prefix)
    }
    validate_rq_checkpoint_depth(
        model,
        component_metadata,
        checkpoint_path=checkpoint_path_str,
        model_keys=current_model_state,
    )
    checkpoint_model_state: dict[str, Any] = {}
    loaded_model_keys: set[str] = set()
    missing_model_keys: set[str] = set()
    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for model_key, current_value in current_model_state.items():
        checkpoint_key = checkpoint_key_by_model_key[model_key]
        checkpoint_metadata = component_metadata.get(checkpoint_key)
        if checkpoint_metadata is None:
            missing_model_keys.add(model_key)
            continue
        checkpoint_shape = getattr(checkpoint_metadata, "size", None)
        if isinstance(current_value, torch.Tensor) and checkpoint_shape is not None:
            current_shape = tuple(current_value.shape)
            saved_shape = tuple(checkpoint_shape)
            if current_shape != saved_shape:
                shape_mismatches.append((model_key, saved_shape, current_shape))
                continue
        checkpoint_model_state[checkpoint_key] = _cpu_state_placeholder(current_value)
        loaded_model_keys.add(model_key)

    if shape_mismatches:
        raise ValueError(
            f"DCP checkpoint {checkpoint_path_str} has {len(shape_mismatches)} model shape mismatches "
            f"(sample: {shape_mismatches[:5]})."
        )
    if missing_model_keys and not allow_partial:
        raise ValueError(
            f"DCP checkpoint {checkpoint_path_str} is missing {len(missing_model_keys)} model keys "
            f"(sample: {sorted(missing_model_keys)[:10]})."
        )

    if include_ema:
        _add_dcp_ema_template_tensors(
            model,
            checkpoint_model_state,
            set(component_metadata),
        )
    requested_checkpoint_keys = set(checkpoint_model_state)
    unexpected_checkpoint_keys = set(component_metadata) - requested_checkpoint_keys
    checkpoint_state = {"model": checkpoint_model_state} if metadata_prefix else checkpoint_model_state
    load(checkpoint_state, storage_reader=storage_reader, no_dist=True)
    loaded_model_state = {
        model_key: checkpoint_model_state[checkpoint_key_by_model_key[model_key]] for model_key in loaded_model_keys
    }
    loaded_model_state.update(
        {
            key: value
            for key, value in checkpoint_model_state.items()
            if key.startswith("ema.") and key not in loaded_model_state
        }
    )
    return {
        "model": loaded_model_state,
        DCP_MODEL_LOAD_INFO_KEY: DCPModelLoadInfo(
            loaded_model_keys=frozenset(loaded_model_keys),
            missing_model_keys=frozenset(missing_model_keys),
            unexpected_checkpoint_keys=frozenset(unexpected_checkpoint_keys),
        ),
    }
