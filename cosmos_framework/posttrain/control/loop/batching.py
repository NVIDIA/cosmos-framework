# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Batch sizing invariants shared by online training loops."""

from __future__ import annotations

from typing import Any

from cosmos_framework.posttrain.utils.entry import resolve_all, resolve_identical


def resolve_prepared_batch(policy: Any, preparation: Any, *, operation: str) -> Any:
    """Resolve the preparation receipt, attributing a failure to its ACTUAL cause.

    Preparation runs on the policy cells asynchronously: the loop holds its result as an
    unresolved handle and asks for the small receipt next. When preparation raises, that
    exception stays unretrieved inside the handle and the receipt request finds an empty slot,
    so the only error the run reports is "no prepared batch receipt" — which describes the
    symptom on the controller and hides the defect on the cell. Draining the preparation
    handles on the failure path re-raises the real error instead.
    """
    try:
        return resolve_identical(policy.prepared_batch_receipt(), operation=operation)
    except Exception as receipt_error:
        resolve_all(preparation)
        raise receipt_error


def validate_batch_divisibility(
    *,
    batch_size: int,
    mini_batch_size: int | None,
    micro_batch_size: int | None,
    dp_world: int,
) -> None:
    """Require every policy batch axis to divide evenly across its complete DP world."""
    if dp_world < 1:
        raise ValueError(f"dp_world must be positive, got {dp_world}")
    resolved_mini_batch_size = mini_batch_size or batch_size
    resolved_micro_batch_size = micro_batch_size or resolved_mini_batch_size
    for name, value in (
        ("batch_size", batch_size),
        ("mini_batch_size", resolved_mini_batch_size),
        ("micro_batch_size", resolved_micro_batch_size),
    ):
        if value % dp_world:
            raise ValueError(
                f"{name}={value} must divide evenly by dp_world={dp_world} (the policy execution replica's "
                f"dp_replicate * dp_shard extent). Otherwise select_chunk gives ranks unequal shares."
            )


def prompts_per_data_cell(
    *,
    policy_batch_size: int,
    rollouts_per_prompt: int,
    data_replicas: int,
) -> int:
    """Return each data cell's prompt count while conserving the sole policy replica's batch."""
    if policy_batch_size < 1 or rollouts_per_prompt < 1 or data_replicas < 1:
        raise ValueError(
            "prompt sizing inputs must be positive, got "
            f"policy_batch_size={policy_batch_size}, rollouts_per_prompt={rollouts_per_prompt}, "
            f"data_replicas={data_replicas}"
        )
    if policy_batch_size % rollouts_per_prompt:
        raise ValueError(
            f"policy_batch_size={policy_batch_size} trajectories is not a multiple of "
            f"rollouts_per_prompt={rollouts_per_prompt}. Adjust the batch size or policy shape."
        )
    total_prompts = policy_batch_size // rollouts_per_prompt
    if total_prompts % data_replicas:
        raise ValueError(
            f"total prompts {total_prompts} is not a multiple of data_replicas={data_replicas}; "
            f"each data cell would draw a fractional share. Adjust the batch size, "
            f"rollouts_per_prompt, or worker shapes."
        )
    return total_prompts // data_replicas


def informative_quota_per_cell(
    *,
    min_informative_groups: int,
    rollouts_per_prompt: int,
    data_replicas: int,
) -> tuple[int, ...]:
    """Return each replay cell's informative-trajectory quota so the quotas sum to the global knob.

    ``min_informative_groups`` is a GLOBAL count over the trained batch. Batch assembly happens per
    replay cell, so the count is split exactly: the first ``min_informative_groups % data_replicas``
    cells owe one group more than the rest. Rounding every cell up would silently raise the global
    requirement to ``data_replicas * ceil(N / data_replicas)`` and make the knob depend on placement.
    Quotas are in trajectories (groups times ``rollouts_per_prompt``), the unit the buffer counts.
    """
    if min_informative_groups < 0 or rollouts_per_prompt < 1 or data_replicas < 1:
        raise ValueError(
            "informative quota inputs out of range, got "
            f"min_informative_groups={min_informative_groups}, rollouts_per_prompt={rollouts_per_prompt}, "
            f"data_replicas={data_replicas}"
        )
    base, remainder = divmod(min_informative_groups, data_replicas)
    return tuple((base + (1 if cell < remainder else 0)) * rollouts_per_prompt for cell in range(data_replicas))
