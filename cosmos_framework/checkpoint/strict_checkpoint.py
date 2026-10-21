# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exact regular-network initialization for the sim-auto recipe."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict

from cosmos_framework.checkpoint.storage_reader import get_storage_reader
from cosmos_framework.utils import log


def validate_regular_checkpoint(state: dict[str, Any], metadata: Any) -> None:
    """Reject missing, unexpected, or differently shaped regular model tensors."""
    expected = {f"net.{key}": value for key, value in state.items() if not key.endswith("_extra_state")}
    actual = {
        key: value
        for key, value in metadata.state_dict_metadata.items()
        if key.startswith("net.") and not key.endswith("_extra_state")
    }
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    mismatched = [
        key
        for key in expected.keys() & actual.keys()
        if isinstance(expected[key], torch.Tensor)
        and (not hasattr(actual[key], "size") or tuple(expected[key].shape) != tuple(actual[key].size))
    ]
    if missing or unexpected or mismatched:
        raise ValueError(
            "Sim-auto requires an exact regular checkpoint: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}, shape_mismatch={sorted(mismatched)[:20]}"
        )


def load_regular_checkpoint(
    net: torch.nn.Module,
    ckpt_path: str,
    prefix: str = "net",
    credential_path: str | None = None,
) -> None:
    """Load exact regular weights, rejecting missing, unexpected, or mismatched tensors."""
    if prefix != "net":
        raise ValueError("Sim-auto initializes exclusively from regular net weights, not EMA.")
    reader = get_storage_reader(ckpt_path, credential_path)
    state = get_model_state_dict(net)
    metadata = reader.read_metadata()
    validate_regular_checkpoint(state, metadata)
    loading = {f"net.{key}": value for key, value in state.items() if not key.endswith("_extra_state")}
    dcp.load(loading, storage_reader=reader, planner=DefaultLoadPlanner(allow_partial_load=False))
    restored = {key: loading.get(f"net.{key}", value) for key, value in state.items()}
    result = set_model_state_dict(net, restored, options=StateDictOptions(strict=True))
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"Sim-auto regular restore was not exact: {result}")
    log.info(f"[SimAuto] Exact regular restore: {len(loading)} keys from {ckpt_path}")
