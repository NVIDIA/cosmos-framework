# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Load only the Reasoner subtree from a Cosmos training DCP checkpoint.

The training checkpoint stores the two model copies under distinct roots::

    net.language_model.*
    net_ema.language_model.*

This module maps exactly one of those roots onto an already-instantiated
Reasoner module.  It deliberately does not construct ``OmniMoTModel`` (and
therefore never constructs the Generator, VAE, or EMA model).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

ReasonerCheckpointSource = Literal["regular", "ema"]

_SOURCE_PREFIXES: dict[ReasonerCheckpointSource, str] = {
    "regular": "net.language_model.",
    "ema": "net_ema.language_model.",
}


@dataclass(frozen=True)
class ReasonerCheckpointLoadResult:
    """Summary of one successful Reasoner-only load."""

    checkpoint_path: Path
    source: ReasonerCheckpointSource
    checkpoint_prefix: str
    num_state_leaves: int


def _resolve_dcp_model_path(checkpoint_path: str | Path) -> Path:
    """Resolve either a DCP model-component directory or its iteration root."""
    raw_path = str(checkpoint_path)
    if "://" in raw_path:
        raise ValueError(f"Reasoner-only checkpoint loading currently supports local DCP paths only; got {raw_path!r}")

    path = Path(checkpoint_path)
    direct_metadata = path / ".metadata"
    nested_metadata = path / "model" / ".metadata"
    if direct_metadata.is_file():
        return path
    if nested_metadata.is_file():
        return path / "model"
    raise FileNotFoundError(f"Could not find DCP metadata. Expected either {direct_metadata} or {nested_metadata}.")


def _checkpoint_prefix(source: str) -> tuple[ReasonerCheckpointSource, str]:
    if source not in _SOURCE_PREFIXES:
        raise ValueError(f"source must be explicitly set to 'regular' or 'ema', got {source!r}")
    typed_source = cast(ReasonerCheckpointSource, source)
    return typed_source, _SOURCE_PREFIXES[typed_source]


def _is_generation_pathway_fqn(fqn: str) -> bool:
    """Whether an MoT language-model FQN belongs exclusively to the Generator tower."""
    return "moe_gen" in fqn


def _is_visual_pathway_fqn(fqn: str) -> bool:
    """Whether an MoT language-model FQN belongs to the optional visual tower."""
    return fqn == "visual" or fqn.startswith("visual.")


def _reasoner_target_state(reasoner: nn.Module) -> dict[str, Any]:
    state = dict(
        get_model_state_dict(
            reasoner,
            options=StateDictOptions(strict=True),
        )
    )
    if not state:
        raise ValueError("Reasoner module has an empty state dict")

    generation_leaves = sorted(name for name in state if _is_generation_pathway_fqn(name))
    if generation_leaves:
        raise ValueError(
            "Reasoner-only load target still contains Generator pathway state. Construct the language model "
            f"with include_gen_pathway=False; generator leaves={generation_leaves}"
        )

    unsupported = [name for name, value in state.items() if not torch.is_tensor(value)]
    if unsupported:
        raise TypeError(
            "Reasoner-only DCP loading currently supports tensor state leaves only; "
            f"non-tensor leaves={sorted(unsupported)}"
        )
    meta_leaves = [
        name for name, value in state.items() if isinstance(value, torch.Tensor) and value.device.type == "meta"
    ]
    if meta_leaves:
        raise ValueError(f"Reasoner must be materialized before loading; meta state leaves={sorted(meta_leaves)}")
    return state


def load_reasoner_only_dcp(
    reasoner: nn.Module,
    checkpoint_path: str | Path,
    *,
    source: ReasonerCheckpointSource,
) -> ReasonerCheckpointLoadResult:
    """Load one complete Reasoner subtree from a local training DCP checkpoint.

    Args:
        reasoner: The already-instantiated language-model/Reasoner module. It is
            the load target itself, not an Omni model or a ``net`` wrapper.
        checkpoint_path: Either the DCP ``model/`` component directory or its
            parent iteration directory.
        source: Explicitly select ``"regular"`` (``net.language_model.*``) or
            ``"ema"`` (``net_ema.language_model.*``). There is no implicit
            fallback between the two sources.

    Raises:
        ValueError: If the source is invalid, the target is unmaterialized, or
            the selected checkpoint subtree differs from the target state.
        FileNotFoundError: If no local DCP metadata is present.

    The key-set comparison happens before any tensor is read, so a missing or
    unexpected Reasoner leaf cannot produce a partially initialized model.
    Shape compatibility remains enforced by PyTorch's strict DCP planner.
    """
    typed_source, prefix = _checkpoint_prefix(source)
    model_path = _resolve_dcp_model_path(checkpoint_path)
    target_state = _reasoner_target_state(reasoner)

    reader = FileSystemReader(str(model_path))
    metadata = reader.read_metadata()
    checkpoint_keys = set(metadata.state_dict_metadata)
    selected_keys = {key for key in checkpoint_keys if key.startswith(prefix)}
    if not selected_keys:
        raise KeyError(
            f"DCP checkpoint {model_path} contains no Reasoner state under the explicitly selected prefix {prefix!r}"
        )

    expected_keys = {f"{prefix}{name}" for name in target_state}
    missing = sorted(expected_keys - selected_keys)
    # A full MoT language-model checkpoint contains both the frozen Reasoner
    # and ``*_moe_gen`` Generator tower. Shipped Nano checkpoints also contain
    # the optional visual encoder. Text-only cache extraction deliberately
    # constructs neither, so those source-only leaves are safe to omit. Visual
    # extras are allowed only when the target has no visual module at all; any
    # other extra leaf signals an architecture/config mismatch.
    allow_source_visual = not hasattr(reasoner, "visual")
    unexpected = sorted(
        key
        for key in selected_keys - expected_keys
        if not (
            _is_generation_pathway_fqn(key.removeprefix(prefix))
            or (allow_source_visual and _is_visual_pathway_fqn(key.removeprefix(prefix)))
        )
    )
    if missing or unexpected:
        raise ValueError(
            f"Reasoner checkpoint/target FQN mismatch under {prefix!r}: missing={missing}, unexpected={unexpected}"
        )

    prefixed_target_state = {f"{prefix}{name}": value for name, value in target_state.items()}
    dcp.load(
        state_dict=prefixed_target_state,
        storage_reader=reader,
        planner=dcp.DefaultLoadPlanner(allow_partial_load=False),
        # Every extraction worker owns a complete Reasoner replica. Independent
        # reads avoid nested collectives and collective-order deadlocks when one
        # worker encounters an I/O or materialization error.
        no_dist=True,
    )

    loaded_state = {name: prefixed_target_state[f"{prefix}{name}"] for name in target_state}
    incompatible = set_model_state_dict(
        reasoner,
        model_state_dict=loaded_state,
        options=StateDictOptions(strict=True),
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict Reasoner state installation unexpectedly reported incompatible keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    return ReasonerCheckpointLoadResult(
        checkpoint_path=model_path,
        source=typed_source,
        checkpoint_prefix=prefix,
        num_state_leaves=len(target_state),
    )


__all__ = [
    "ReasonerCheckpointLoadResult",
    "ReasonerCheckpointSource",
    "load_reasoner_only_dcp",
]
