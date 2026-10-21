# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CUDA-IPC tensor sharing (design spec §2.3 IpcMover).

torch.multiprocessing.reductions.reduce_tensor turns a CUDA tensor into a picklable (rebuild, args)
pair whose args carry a CUDA IPC handle -- the handle crosses the control plane, the GPU memory does
not. A consumer on the SAME physical GPU rebuilds a tensor pointing at that memory (zero-copy
transfer; the caller then D2D-loads to snapshot). GPU-only.

TODO(v1+, spec §3.5): versioned-buffer lifetime/lease (keep the exported tensor alive until the
consumer swaps) for async/off-policy.
"""

from __future__ import annotations

from typing import Any


def tensor_to_ipc(t: Any):
    """Return a picklable (rebuild_fn, args) handle for a CUDA tensor; args hold the IPC handle."""
    from torch.multiprocessing.reductions import reduce_tensor

    return reduce_tensor(t)


def tensor_from_ipc(handle) -> Any:
    """Rebuild a tensor from a handle produced by tensor_to_ipc. NOTE: the result ALIASES the
    producer's GPU memory (no copy) and is valid only while the producer keeps that storage alive;
    the caller must D2D-snapshot (`.clone()`) it to own an independent copy (`IpcMover.fetch` does so
    in the default snapshot mode; in `borrow=True` mode it hands back this alias, and the caller must
    copy out immediately)."""
    rebuild, args = handle
    return rebuild(*args)


def state_dict_to_ipc(state: dict) -> dict:
    return {k: tensor_to_ipc(v) for k, v in state.items()}


def state_dict_from_ipc(handles: dict) -> dict:
    return {k: tensor_from_ipc(h) for k, h in handles.items()}
