# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Pure bind-time policy→rollout full-tensor actor pairing; no Ray and no torch.

Every currently supported policy actor materializes the same complete HF tensor bucket through
``DTensor.full_tensor()`` before weight movement. Routing therefore operates on the flat actor pools,
not on execution-replica or rank axes: every rollout actor is assigned to exactly one policy actor,
and policy actors with no destination remain explicit idle participants in the full-tensor collective.
"""

from __future__ import annotations

Cell = tuple[int, int]


def full_tensor_actor_targets(
    policy_shape: tuple[int, int],
    rollout_shape: tuple[int, int],
) -> dict[Cell, tuple[Cell, ...]]:
    """Build one balanced flat-actor route for every policy cell.

    Both ``(R, K)`` shapes are flattened in row-major order. Rollout actor ``q`` is assigned to
    policy actor ``q % P``, where ``P = R_policy * K_policy``. This preserves the same-index
    bijection when the products are equal, selects the first rollout-product policy actors when the
    policy pool is wider, and balances direct fan-out within one target when the rollout pool is
    wider. No route depends on either shape's execution-replica boundary.

    Every policy cell is present in the returned mapping. An empty target tuple is load-bearing: the
    idle policy actor must still materialize every bucket so the FSDP full-tensor collective stays
    aligned, but it must not fall through to the general ``dp_aligned`` peer route.

    This pairing relies on the current full-tensor contract. A future shard-aware weight mapper must
    replace it with layout-aware routing rather than assuming arbitrary policy actors are equivalent.

    Args:
        policy_shape: policy placement shape ``(R_policy, K_policy)``.
        rollout_shape: rollout placement shape ``(R_rollout, K_rollout)``.

    Returns:
        ``{policy_cell: (rollout_cell, ...)}``, covering every rollout cell exactly once and every
        policy cell exactly once as a key.
    """
    r_policy, k_policy = policy_shape
    r_rollout, k_rollout = rollout_shape
    if min(r_policy, k_policy, r_rollout, k_rollout) < 1:
        raise ValueError(
            "weight pairing shapes must have positive replica/rank axes, "
            f"got policy={policy_shape}, rollout={rollout_shape}"
        )

    policy_cells = [(replica, rank) for replica in range(r_policy) for rank in range(k_policy)]
    rollout_cells = [(replica, rank) for replica in range(r_rollout) for rank in range(k_rollout)]
    mutable: dict[Cell, list[Cell]] = {cell: [] for cell in policy_cells}
    for flat_rollout, rollout_cell in enumerate(rollout_cells):
        mutable[policy_cells[flat_rollout % len(policy_cells)]].append(rollout_cell)
    return {cell: tuple(targets) for cell, targets in mutable.items()}
