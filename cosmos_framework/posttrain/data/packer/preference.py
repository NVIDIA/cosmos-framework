# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared preference-pair shaping for offline objectives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from cosmos_framework.posttrain.schema import Preference, Trajectory


def _collate_mappings(samples: list[dict[str, Any]]) -> dict[str, list[Any]]:
    if not samples:
        raise ValueError("preference collation requires at least one sample")
    return {key: [sample[key] for sample in samples] for key in samples[0]}


def unpack_preferences(trajectories: Sequence[Trajectory]) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """Validate typed preference targets and collate their winner and loser sides."""
    pairs = [trajectory.transitions[0].targets for trajectory in trajectories]
    for pair in pairs:
        if not isinstance(pair, Preference):
            raise TypeError(
                f"Transition.targets must be a Preference, got {type(pair).__name__}; "
                "construct it with Preference(winner=..., loser=...)."
            )
    return _collate_mappings([pair.winner for pair in pairs]), _collate_mappings([pair.loser for pair in pairs])
