# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bucket ↔ flat-buffer layout for single-descriptor publishing (design §3.2).

`ray.put` cost is per-DESCRIPTOR, not per-byte (~0.31 ms/tensor measured), and a real
512 MiB transformer bucket holds 100-200 tensors. Publishing one flat tensor instead of one
per parameter collapses that to a single descriptor; the consumer carves parameters back out
using the layout, which rides the handle's metadata.

Byte-based offsets (not element counts) so a bucket may mix dtypes — `bucket_plan` already
computes `_dtype_bytes` PER PARAMETER (weight_transfer.py:97), i.e. the weight plane does not
assume a uniform bucket, and a bf16 model routinely carries fp32 buffers (rotary `inv_freq`,
some norms) in its `state_dict()`. Offsets are 16-byte aligned because `Tensor.view(dtype)`
requires the storage offset to be divisible by the target dtype's element size; 16 covers every
dtype torch has, so any entry can be viewed regardless of what precedes it.

Pure torch, no Ray: this is where flat-publish correctness lives, so it must be testable
without a cluster.
"""

from __future__ import annotations

from typing import Any

import torch

ALIGN = 16  # bytes; >= the largest dtype element size, so every .view(dtype) is legal


def _aligned(n: int) -> int:
    return ((n + ALIGN - 1) // ALIGN) * ALIGN


def flat_plan(state: dict) -> list[tuple[str, tuple, str, int, int]]:
    """Layout `state` inside one contiguous byte buffer.

    Args:
        state: `{param_name: tensor}` for one bucket.

    Returns:
        `[(name, shape, dtype_str, byte_offset, nbytes), ...]` in SORTED NAME order — the same
        ordering invariant `NcclMover.stage` and `bucket_plan` already carry
        (`_check_name_sorted`, weight_transfer.py:61), so a plan built here indexes the way the
        rest of the weight plane expects.
    """
    plan, off = [], 0
    for name in sorted(state):
        t = state[name]
        nbytes = t.element_size() * t.nelement()
        plan.append((name, tuple(t.shape), str(t.dtype), off, nbytes))
        off += _aligned(nbytes)
    return plan


def flat_nbytes(plan: list) -> int:
    """Total bytes the plan occupies, including alignment padding after the last entry.

    Args:
        plan: from `flat_plan`.

    Returns:
        Byte length the staging buffer must have to hold this bucket.
    """
    if not plan:
        return 0
    _name, _shape, _dtype, off, nbytes = plan[-1]
    return off + _aligned(nbytes)


def flat_nbytes_for_manifest(entries: list) -> int:
    """Bytes a bucket will occupy, computed from METADATA ONLY — no tensors required.

    Used at bind to size the staging buffer before any bucket has been materialized. Shares
    `_aligned` with `flat_plan`, so the sizing and the packing cannot drift: an over-tight
    size here would surface later as a `pack` "buffer too small" mid-sync.

    Derives each element size from the dtype itself rather than reusing
    `weight_transfer._dtype_bytes`, which is a 2/4/8 heuristic on the dtype string — and
    keeps `comm/transfer` independent of `comm/weight_transfer`.

    Args:
        entries: `[(name, shape, dtype_str), ...]` — one `bucket_plan` bucket.

    Returns:
        Byte length the staging buffer needs for this bucket, including alignment padding.
    """
    total = 0
    for _name, shape, dtype_str in entries:
        n = 1
        for d in shape:
            n *= d
        esize = torch.empty(0, dtype=getattr(torch, str(dtype_str).split(".")[-1])).element_size()
        total += _aligned(n * esize)
    return total


def pack(state: dict, plan: list, buf: Any) -> Any:
    """Copy `state` into `buf` per `plan`; return the used prefix as a uint8 view.

    Args:
        state: `{param_name: tensor}`; every tensor must be contiguous.
        plan: from `flat_plan(state)`.
        buf: a 1-D uint8 tensor at least `flat_nbytes(plan)` long.

    Returns:
        `buf[:flat_nbytes(plan)]` — the single tensor to publish.
    """
    need = flat_nbytes(plan)
    if buf.numel() < need:
        raise ValueError(f"staging buffer too small: {buf.numel()} bytes < {need} needed")
    for name, _shape, _dtype, off, nbytes in plan:
        t = state[name]
        if not t.is_contiguous():
            raise ValueError(
                f"{name!r} is not contiguous; pack would copy a strided view's raw bytes and "
                f"silently produce wrong weights. Call .contiguous() upstream."
            )
        buf[off : off + nbytes].copy_(t.reshape(-1).view(torch.uint8))
    return buf[:need]


def carve(flat: Any, plan: list) -> dict:
    """Split a flat uint8 buffer back into named tensors. Views only — no copy.

    Args:
        flat: the uint8 tensor produced by `pack` (or landed by the transport).
        plan: the layout that produced it.

    Returns:
        `{param_name: tensor}` viewing into `flat`.
    """
    out = {}
    for name, shape, dtype, off, nbytes in plan:
        dt = getattr(torch, dtype.split(".")[-1])
        out[name] = flat[off : off + nbytes].view(dt).reshape(shape)
    return out
