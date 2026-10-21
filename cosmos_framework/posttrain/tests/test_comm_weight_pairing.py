# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Full-tensor policy→rollout flat-actor routing tests."""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.comm.weight_pairing import full_tensor_actor_targets

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _flatten_targets(routes: dict[tuple[int, int], tuple[tuple[int, int], ...]]) -> list[tuple[int, int]]:
    """Return every rollout cell in policy-flat, then target order."""
    return [target for targets in routes.values() for target in targets]


def test_policy64_to_rollout4_selects_four_actors_and_keeps_idle_routes() -> None:
    routes = full_tensor_actor_targets((1, 64), (1, 4))
    assert set(routes) == {(0, rank) for rank in range(64)}
    assert [routes[(0, rank)] for rank in range(4)] == [((0, rank),) for rank in range(4)]
    assert all(routes[(0, rank)] == () for rank in range(4, 64))
    assert _flatten_targets(routes) == [(0, rank) for rank in range(4)]


def test_policy4_to_rollout64_uses_balanced_direct_actor_fanout() -> None:
    routes = full_tensor_actor_targets((1, 4), (1, 64))
    assert routes[(0, 0)] == tuple((0, rank) for rank in range(0, 64, 4))
    assert routes[(0, 1)] == tuple((0, rank) for rank in range(1, 64, 4))
    assert routes[(0, 2)] == tuple((0, rank) for rank in range(2, 64, 4))
    assert routes[(0, 3)] == tuple((0, rank) for rank in range(3, 64, 4))
    assert sorted(_flatten_targets(routes)) == [(0, rank) for rank in range(64)]


def test_equal_product_shapes_preserve_flat_index_bijection() -> None:
    routes = full_tensor_actor_targets((1, 8), (4, 2))
    for flat_policy in range(8):
        policy_cell = (0, flat_policy)
        assert routes[policy_cell] == ((flat_policy // 2, flat_policy % 2),)


def test_relay_p2r_routes_only_to_replica_zero_shape() -> None:
    routes = full_tensor_actor_targets((1, 8), (1, 2))

    assert routes[(0, 0)] == ((0, 0),)
    assert routes[(0, 1)] == ((0, 1),)
    assert all(routes[(0, rank)] == () for rank in range(2, 8))


def test_unequal_actor_fanout_is_exact_once_and_balanced() -> None:
    routes = full_tensor_actor_targets((1, 3), (2, 4))
    assert routes == {
        (0, 0): ((0, 0), (0, 3), (1, 2)),
        (0, 1): ((0, 1), (1, 0), (1, 3)),
        (0, 2): ((0, 2), (1, 1)),
    }
    targets = _flatten_targets(routes)
    assert len(targets) == len(set(targets)) == 8
    assert set(targets) == {(replica, rank) for replica in range(2) for rank in range(4)}
    fanout = [len(targets_for_actor) for targets_for_actor in routes.values()]
    assert max(fanout) - min(fanout) <= 1


def test_pairing_rejects_empty_axes() -> None:
    with pytest.raises(ValueError, match="positive"):
        full_tensor_actor_targets((1, 0), (1, 4))
