# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared validation for comparable weight names and same-layout tensor copies."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping

import torch


def check_weight_names(expected: Collection[str], actual: Collection[str], *, context: str) -> None:
    """Reject missing or unexpected destinations before loading a stream."""
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise ValueError(f"{context}: missing={missing[:10]}, unexpected={unexpected[:10]}")


def copy_named_tensors(
    weights: Iterable[tuple[str, torch.Tensor]], destinations: Mapping[str, torch.Tensor]
) -> set[str]:
    """Validate the entire bucket before copying into same-layout destinations.

    The caller owns name translation, synchronization and verification. Destination references must not survive
    a retained exception: they may be imported IPC views whose owner is shut down by the caller during recovery.
    """
    entries = []
    target = None
    try:
        entries = list(weights)
        for name, tensor in entries:
            target = destinations[name]
            if tuple(target.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"{name!r}: shape {tuple(tensor.shape)}, but the live parameter is {tuple(target.shape)}"
                )
            if target.dtype != tensor.dtype:
                raise ValueError(f"{name!r}: dtype {tensor.dtype}, but the live parameter is {target.dtype}")
        with torch.no_grad():
            for name, tensor in entries:
                destinations[name].copy_(tensor)
        return {name for name, _tensor in entries}
    finally:
        entries.clear()
        # The traceback also retains function arguments and caller-supplied mapping/item iterators.
        del target, destinations, weights
