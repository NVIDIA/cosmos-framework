# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Recorded forward routing on the policy path (the packer half of routing replay).

The record and both of its keys are the `schema/routing.py` contract: the rollout writes
``policy_info[ROLLOUT_ROUTED_EXPERTS_KEY]`` (``[T-1, L, K]`` integer expert ids, one row per input
position of prompt + completion). `attach_forward_routing` checks that record against the shaped
``input_ids`` and carries it on a `RoutedPolicyInput`; `add_forward_routing` pads a batch of them into
``FORWARD_REPLAY_BATCH_KEY`` ``[B, T_max, L, K]``, which the model adapter (`ForwardReplayPolicy`)
consumes to pin every forward of that batch. Alignment relies on ``input_ids`` being exactly the
engine's prompt + completion: the packers assert the stored prompt ids against their own prefix, so
a shifted prefix fails there and a wrong row count fails here, never silently.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol, runtime_checkable

import torch

from cosmos_framework.posttrain.schema.routing import FORWARD_REPLAY_BATCH_KEY, ROLLOUT_ROUTED_EXPERTS_KEY


@runtime_checkable
class RoutedPolicyInput(Protocol):
    """A packer's shaped sample that can carry recorded routing: a dataclass with these two fields
    (`RLPolicyInput`, `Qwen3VLPolicyInput`)."""

    input_ids: list[int]
    forward_routing: torch.Tensor | None  # [T-1,L,K] integer expert ids | None


def attach_forward_routing(policy_input: Any, policy_info: dict[str, Any]) -> Any:
    """Carry a transition's recorded routing on its shaped policy input, or return it unchanged.

    Args:
        policy_input: the packer's shaped sample.
        policy_info: the transition's ``policy_info``.

    Returns:
        ``policy_input`` with ``forward_routing`` set when the transition recorded routing.

    Raises:
        ValueError: the record is not an integer ``[T-1, L, K]`` tensor for this sample's ``T``.
        TypeError: the packer's policy input is not a `RoutedPolicyInput`.
    """
    routing = policy_info.get(ROLLOUT_ROUTED_EXPERTS_KEY)
    if routing is None:
        return policy_input
    if not isinstance(policy_input, RoutedPolicyInput):
        raise TypeError(f"{type(policy_input).__name__} cannot carry recorded routing (not a RoutedPolicyInput)")
    sequence_length = len(policy_input.input_ids)
    if not isinstance(routing, torch.Tensor) or routing.is_floating_point() or routing.dtype == torch.bool:
        raise ValueError(f"policy_info[{ROLLOUT_ROUTED_EXPERTS_KEY!r}] must be an integer tensor, got {routing!r}")
    if routing.ndim != 3 or routing.shape[0] != sequence_length - 1:
        raise ValueError(
            f"policy_info[{ROLLOUT_ROUTED_EXPERTS_KEY!r}] has shape {tuple(routing.shape)} for a "
            f"{sequence_length}-token prompt+completion; the engine routes every input position, so the "
            f"record must be [{sequence_length - 1}, layers, top_k]. The shaped input_ids are not the "
            "sequence the engine generated"
        )
    return dataclasses.replace(policy_input, forward_routing=routing)


def add_forward_routing(batch: dict[str, Any], processed_samples: list[Any], computed_max_len: int) -> None:
    """Collate the samples' recorded routing into ``batch[FORWARD_REPLAY_BATCH_KEY]``, ``[B, T_max, L, K]``.

    Rows without a record (each sequence's last token, and the padding) hold experts ``0..K-1``: a
    valid selection whose combine weight is zero on padded rows and whose output feeds no loss on the
    last token. Samples are truncated to ``computed_max_len`` exactly as their ``input_ids`` are.

    Args:
        batch: the collated policy batch, extended in place when any sample carries routing.
        processed_samples: the packer's shaped samples (``forward_routing`` set or None).
        computed_max_len: the batch's padded sequence length.

    Raises:
        ValueError: some but not all samples carry routing, or the records disagree on
            ``(layers, top_k)`` or dtype.
    """
    routings = [sample.forward_routing for sample in processed_samples]
    present = [routing for routing in routings if routing is not None]
    if not present:
        return
    if len(present) != len(routings):
        raise ValueError(
            f"{len(present)} of {len(routings)} samples carry recorded routing; a batch replays routing "
            "for every sequence or for none"
        )
    layers, top_k = present[0].shape[1:]
    dtype = present[0].dtype
    if any(tuple(routing.shape[1:]) != (layers, top_k) or routing.dtype != dtype for routing in present):
        raise ValueError(
            "recorded routing must agree on (layers, top_k) and dtype across the batch, got "
            f"{sorted({(tuple(r.shape[1:]), str(r.dtype)) for r in present})}"
        )
    collated = torch.arange(top_k, dtype=dtype).expand(len(present), computed_max_len, layers, top_k).clone()
    for index, routing in enumerate(present):  # routing: [T_i-1,L,K]
        rows = min(int(routing.shape[0]), computed_max_len)
        collated[index, :rows] = routing[:rows]
    batch[FORWARD_REPLAY_BATCH_KEY] = collated  # [B,T_max,L,K]
