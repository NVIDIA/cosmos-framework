# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Select portable tensors without consolidating the full training checkpoint."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch

from cosmos_framework.posttrain.checkpoint.io import load_component
from cosmos_framework.posttrain.checkpoint.manifest import dcp_tensor_inventory, is_canonical_relative_path


def _validate_dcp_storage(directory: Path, metadata) -> None:
    """Check the complete model inventory, including unselected payload extents.

    Partial DCP loading must not hide a truncated/missing frozen-weight shard.
    This checks structure without reading its tensor bytes. The enclosing stable
    provenance check still binds the entire immutable checkpoint inventory.
    """
    from torch.distributed.checkpoint.metadata import MetadataIndex, TensorStorageMetadata

    storage = metadata.storage_data
    if not isinstance(storage, dict):
        raise ValueError("incomplete DCP model: missing storage metadata")
    sizes = {}
    for index, item in storage.items():
        relative = item.relative_path
        if not is_canonical_relative_path(relative):
            raise ValueError(f"invalid DCP storage path for {index.fqn}: {relative!r}")
        path = directory / relative
        if path.resolve().is_relative_to(directory.resolve()) is False:
            raise ValueError(f"DCP storage escapes component directory: {relative!r}")
        if relative not in sizes:
            sizes[relative] = path.stat().st_size
        if (
            type(item.offset) is not int
            or type(item.length) is not int
            or item.offset < 0
            or item.length <= 0
            or item.offset + item.length > sizes[relative]
        ):
            raise ValueError(f"incomplete DCP storage extent for {index.fqn} in {relative}")
    for name, descriptor in metadata.state_dict_metadata.items():
        if not isinstance(descriptor, TensorStorageMetadata):
            if MetadataIndex(name) not in storage:
                raise ValueError(f"incomplete DCP model: missing storage for {name}")
            continue
        chunks = descriptor.chunks
        volume = 0
        for i, chunk in enumerate(chunks):
            if (
                len(chunk.offsets) != len(descriptor.size)
                or len(chunk.sizes) != len(descriptor.size)
                or any(o < 0 or s < 0 or o + s > n for o, s, n in zip(chunk.offsets, chunk.sizes, descriptor.size))
                or MetadataIndex(name, chunk.offsets) not in storage
            ):
                raise ValueError(f"incomplete/invalid DCP chunk for {name}")
            volume += math.prod(chunk.sizes)
            for previous in chunks[:i]:
                if all(
                    max(a, b) < min(a + sa, b + sb)
                    for a, b, sa, sb in zip(chunk.offsets, previous.offsets, chunk.sizes, previous.sizes)
                ):
                    raise ValueError(f"overlapping DCP chunks for {name}")
        if volume != math.prod(descriptor.size):
            raise ValueError(f"incomplete DCP tensor coverage for {name}")


def load_adapter_tensors(directory: str, spec: dict[str, tuple[int, ...]]) -> dict[str, torch.Tensor]:
    """Read exactly the portable inventory; never open the optimizer payload.

    Consolidated checkpoints use the existing mmap reader, then clone only the
    selected tensors so no full-model storage stays alive. DCP uses its normal
    load planner with a destination containing only the requested logical keys.
    Aliases are intentionally loaded independently: the artifact validator must
    detect disagreements rather than silently substituting their owner.
    """
    if os.path.isfile(os.path.join(directory, "model.pt")):
        model = load_component(directory, "model")["net"]
        result = {}
        for name, shape in spec.items():
            value = model.get(name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != torch.float32:
                raise ValueError(f"missing/invalid checkpoint adapter tensor {name}: expected float32 {shape}")
            result[name] = value.detach().clone(memory_format=torch.contiguous_format)
        return result

    import torch.distributed.checkpoint as dcp

    path = Path(directory) / "model"
    reader = dcp.FileSystemReader(path)
    metadata = reader.read_metadata()
    inventory = dcp_tensor_inventory(metadata, str(path / ".metadata"))
    _validate_dcp_storage(path, metadata)
    for name, shape in spec.items():
        descriptor = metadata.state_dict_metadata.get(name)
        if (
            inventory.get(name) != shape
            or getattr(getattr(descriptor, "properties", None), "dtype", None) != torch.float32
        ):
            raise ValueError(f"missing/invalid checkpoint adapter tensor {name}: expected float32 {shape}")
    result = {name: torch.empty(shape, dtype=torch.float32) for name, shape in spec.items()}
    # Do not enable allow_partial_load: every requested adapter/initial factor
    # must be present. Unrequested base keys are ignored by the standard planner.
    dcp.load(result, storage_reader=reader, no_dist=True)
    return result
