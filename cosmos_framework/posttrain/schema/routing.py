# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The recorded-routing data contract (routing replay), owned here and nowhere else.

One record crosses three layers that must never import each other to agree on it: the rollout
engine PRODUCES it (`roles/rollout/vllm.py`), the data packer ALIGNS it to the shaped policy input
(`data/packer/forward_routing.py`), and the model capability CONSUMES it
(`model/protocol.py::ForwardReplayPolicy`, implemented by `model/qwen3_vl_moe.py`). Like the rest of
`schema/`, this module depends on the standard library only, so any of those layers -- and the
codec that carries the record as bulk -- can name the contract without pulling in a downstream
implementation.

Two keys, two layouts:

* ``ROLLOUT_ROUTED_EXPERTS_KEY`` in ``Transition.policy_info``: the engine's per-token top-k expert
  ids for every INPUT position of prompt + completion, an integer tensor ``[T-1, L, K]`` for a
  ``T``-token sequence (the last sampled token is never fed back, so it has no row) in the narrow
  dtype the engine sized by its expert count (uint8 up to 256 experts, else int16). ``L`` is the
  engine's routed-layer axis; what an expert id means and which blocks ``L`` indexes are model
  details the consuming adapter checks against itself.
* ``FORWARD_REPLAY_BATCH_KEY`` in the collated policy batch: the same ids token-aligned to
  ``input_ids``, ``[B, T, L, K]``, one row per padded position. The value's layout is model-owned;
  a `ForwardReplayPolicy` consumes it in ``prepare_policy_batch`` and it never reaches the network
  as an input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import torch

ROLLOUT_ROUTED_EXPERTS_KEY = "rollout_routed_experts"
"""``policy_info`` key of one sequence's recorded routing, a `RoutedExpertsRecord`."""

FORWARD_REPLAY_BATCH_KEY = "forward_replay_routing"
"""Policy-batch key of the collated, token-aligned routing, a `ForwardReplayBatch`."""

RoutedExpertsRecord: TypeAlias = "torch.Tensor"
"""Integer ``[T-1, L, K]`` expert ids, one row per input position of prompt + completion."""

ForwardReplayBatch: TypeAlias = "torch.Tensor"
"""Integer ``[B, T, L, K]`` expert ids, row ``t`` of sample ``b`` describing ``input_ids[b, t]``."""
