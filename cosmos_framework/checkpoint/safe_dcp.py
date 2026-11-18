# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""DCP storage reader and load planner that only reconstruct checkpoint data.

PyTorch's ``FileSystemReader`` unpickles the ``.metadata`` file and ``DefaultLoadPlanner`` restores
non-tensor values with ``torch.load(weights_only=False)``, so a crafted checkpoint can run code while it
is being resumed. The subclasses here restrict both steps to the types cosmos checkpoints contain; the
on-disk format is unchanged.
"""

import io
import pickle
from typing import Any

import numpy as np
import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint._traverse import set_element
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import Metadata, StorageMeta
from torch.distributed.checkpoint.planner import ReadItem

from cosmos_framework.utils import log
from cosmos_framework.utils.safe_unpickle import safe_loads

# Globals referenced by a DCP ``.metadata`` pickle: the metadata dataclasses, the filesystem storage
# info, the torch types that describe tensors, and the checkpoint id path (``pathlib._local`` on
# Python >= 3.13).
DCP_METADATA_PICKLE_GLOBALS = frozenset(
    {
        ("torch.distributed.checkpoint.metadata", "Metadata"),
        ("torch.distributed.checkpoint.metadata", "StorageMeta"),
        ("torch.distributed.checkpoint.metadata", "MetadataIndex"),
        ("torch.distributed.checkpoint.metadata", "TensorStorageMetadata"),
        ("torch.distributed.checkpoint.metadata", "BytesStorageMetadata"),
        ("torch.distributed.checkpoint.metadata", "ChunkStorageMetadata"),
        ("torch.distributed.checkpoint.metadata", "TensorProperties"),
        ("torch.distributed.checkpoint.metadata", "_MEM_FORMAT_ENCODING"),
        ("torch.distributed.checkpoint.filesystem", "_StorageInfo"),
        ("torch", "Size"),
        ("torch.serialization", "_get_layout"),
        ("pathlib", "PosixPath"),
        ("pathlib._local", "PosixPath"),
        *(
            ("torch", str(dtype).removeprefix("torch."))
            for dtype in vars(torch).values()
            if isinstance(dtype, torch.dtype)
        ),
    }
)

# Non-tensor values are written with ``torch.save``. Besides the builtins ``weights_only`` accepts, LR
# scheduler state carries numpy values (``np.cumsum`` cycle boundaries, ``np.float64`` multipliers).
_BYTES_SAFE_GLOBALS: list[Any] = [
    np._core.multiarray._reconstruct,
    np._core.multiarray.scalar,
    np.ndarray,
    np.dtype,
    *{type(np.dtype(code)) for code in "?bhilqBHILQefdg"},
]


class SafeFileSystemReader(FileSystemReader):
    """``FileSystemReader`` whose metadata unpickling resolves only DCP metadata types."""

    def read_metadata(self, *args: Any, **kwargs: Any) -> Metadata:
        path = self._get_metadata_path(kwargs.get("rank"))
        with self.fs.create_stream(path, "rb") as metadata_file:
            data = metadata_file.read()
        try:
            metadata = safe_loads(data, DCP_METADATA_PICKLE_GLOBALS)
            if type(metadata) is not Metadata:
                raise ValueError(f"expected Metadata, got {type(metadata).__name__}")
        except Exception as e:
            message = f"Refusing to load checkpoint metadata from '{path}': {e}"
            # dcp.load treats a failed read as missing metadata and only reports "metadata is None", so log why.
            log.error(message, rank0_only=False)
            raise ValueError(message) from e

        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        return metadata


class SafeLoadPlanner(DefaultLoadPlanner):
    """``DefaultLoadPlanner`` that restores non-tensor values with ``torch.load(weights_only=True)``."""

    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        fqn = read_item.dest_index.fqn
        try:
            with torch.serialization.safe_globals(_BYTES_SAFE_GLOBALS):
                obj = torch.load(value, weights_only=True)
        except pickle.UnpicklingError as e:
            raise ValueError(f"Refusing to load checkpoint value '{fqn}': {e}") from e
        if self.flatten_state_dict:
            set_element(self.original_state_dict, self.mappings[fqn], obj)
        else:
            self.state_dict[fqn] = obj
