# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU contracts for shared rank-local checkpoint state primitives."""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.utils.rank_state import (
    RankStateWorldSizeMismatch,
    collect_rank_state,
    select_rank_state,
    state_for_current_rank,
)


@pytest.mark.L0
@pytest.mark.CPU
def test_optional_cursor_component_round_trip() -> None:
    assert collect_rank_state(None) is None
    component = collect_rank_state({"position": 3})
    assert component == {
        "version": 1,
        "world_size": 1,
        "states": {0: {"position": 3}},
    }
    assert state_for_current_rank(component) == {"position": 3}


@pytest.mark.L0
@pytest.mark.CPU
def test_rank_state_world_size_policy_is_explicit() -> None:
    states = {0: {"position": 1}, 1: {"position": 2}}
    assert select_rank_state(states, on_world_size_mismatch="skip") is None
    with pytest.raises(RankStateWorldSizeMismatch, match="world_size=2"):
        select_rank_state(states, on_world_size_mismatch="raise")


@pytest.mark.L0
@pytest.mark.CPU
def test_rank_state_rejects_malformed_inventory() -> None:
    assert select_rank_state(None, on_world_size_mismatch="skip") is None
    with pytest.raises(ValueError, match="indexed by integer rank"):
        select_rank_state({False: {"position": 0}}, on_world_size_mismatch="skip")
    with pytest.raises(ValueError, match="at least one rank"):
        select_rank_state({}, on_world_size_mismatch="skip")
    with pytest.raises(ValueError, match="contiguous"):
        select_rank_state(
            {1: {"position": 1}},
            on_world_size_mismatch="skip",
        )
    with pytest.raises(ValueError, match="world size"):
        state_for_current_rank(
            {
                "version": 1,
                "world_size": 2,
                "states": {0: {"position": 1}},
            }
        )
    with pytest.raises(ValueError, match="must be dictionaries"):
        state_for_current_rank(
            {
                "version": 1,
                "world_size": 1,
                "states": {0: None},
            },
            on_world_size_mismatch="skip",
        )
    with pytest.raises(ValueError, match="world size must be positive"):
        state_for_current_rank(
            {
                "version": 1,
                "world_size": 0,
                "states": {},
            }
        )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", 1.0, "version must be an integer"),
        ("version", True, "version must be an integer"),
        ("world_size", 1.0, "world size must be an integer"),
        ("world_size", "1", "world size must be an integer"),
    ],
)
def test_rank_state_rejects_coerced_integer_metadata(field: str, value: object, message: str) -> None:
    component: dict[str, object] = {
        "version": 1,
        "world_size": 1,
        "states": {0: {"position": 0}},
    }
    component[field] = value
    with pytest.raises(ValueError, match=message):
        state_for_current_rank(component)  # type: ignore[arg-type]
