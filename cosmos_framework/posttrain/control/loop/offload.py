# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Serial GPU offload steps between a colocated rollout and trainer.

One step of an online loop alternates two phases that need the same card. These helpers move one
body at a time and block until every cell has finished moving, so the next role call submitted by
the loop observes the freed memory and a profiler checkpoint after each call measures that call
alone:

    generate -> sleep_rollout -> onload_trainer -> train
             -> prepare_trainer_weight_sync -> wake_rollout_weights -> weight_sync
             -> offload_trainer -> wake_rollout -> generate

Each helper is a no-op for bodies whose ``sleep``/``wake_up``/``offload``/``onload`` are the base
no-ops (HF and WFM rollouts share the trainer's model; CPU trainers own no device memory), so the
same loop runs unchanged on every backend.

Whether the loop runs the cycle at all is ``offload_cycle_enabled``: only a colocated card is
shared, and ``rl.colocated_offload=False`` opts a colocated run out of it (both bodies stay
resident, trading GPU memory for the hand-off time).
"""

from __future__ import annotations

from typing import Any

from cosmos_framework.posttrain.utils.entry import resolve_all


def _hand_off(body: Any, method: str, *args: Any) -> list[Any]:
    """Call one residency method and drain every cell; bodies without it (test doubles) are skipped."""
    fn = getattr(body, method, None)
    if not callable(fn):
        return []
    return resolve_all(fn(*args))


def _max_mib(values: list[Any]) -> float:
    """Largest per-cell MiB figure a hand-off reported; cells report None when they moved nothing."""
    numbers = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return max(numbers) if numbers else 0.0


def sleep_rollout(rollout: Any) -> None:
    """Rollout releases its weights and KV cache. Must precede ``onload_trainer`` on a shared card."""
    _hand_off(rollout, "sleep")


def onload_trainer(policy: Any) -> float:
    """Trainer state returns to the GPU for the update. Returns the MiB the largest cell moved."""
    return _max_mib(_hand_off(policy, "onload"))


def wake_rollout_weights(rollout: Any) -> None:
    """Restore only the engine weights; the KV cache stays released while the trainer is resident."""
    _hand_off(rollout, "wake_up", ("weights",))


def prepare_trainer_weight_sync(policy: Any) -> float:
    """Park update-only trainer state while retaining policy weights as the sync source.

    Returns the MiB moved by the largest policy cell. Bodies without a selective hand-off remain
    unchanged, which preserves shared-model and CPU trainer behavior.
    """
    return _max_mib(_hand_off(policy, "prepare_weight_sync"))


def offload_trainer(policy: Any) -> float:
    """Trainer state leaves the GPU; must precede ``wake_rollout`` on a shared card. Returns the MiB
    the largest cell now holds on the host."""
    return _max_mib(_hand_off(policy, "offload"))


def wake_rollout(rollout: Any) -> None:
    """Restore whatever the rollout still has released (the KV cache) before generation."""
    _hand_off(rollout, "wake_up")


def offload_cycle_enabled(*, gpu_placement: str, colocated_offload: bool) -> bool:
    """Whether the loop hands the card between rollout and trainer every step.

    True only on a shared card (``grid.gpu_placement=colocated``) with ``rl.colocated_offload`` left
    on. With the knob off the engine is built without sleep mode and is never put to sleep, so its
    awake guards never fire, and the trainer is never moved. Under ``disaggregated`` nothing is
    shared and the cycle is skipped either way, so the knob is inert there rather than an error.
    """
    return str(gpu_placement) == "colocated" and bool(colocated_offload)
