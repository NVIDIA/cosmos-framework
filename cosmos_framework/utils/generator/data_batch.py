# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Any

import torch


def slice_batch_item(value: Any, index: int) -> Any:
    """Extract one batch item while preserving the batch container shape."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value[index : index + 1]  # [1,...]
    if isinstance(value, (list, tuple)):
        return [value[index]]
    return value


def slice_batch_range(value: Any, limit: int) -> Any:
    """Extract the first ``limit`` batch items from tensor/list-like values."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value[:limit]  # [B,...]
    if isinstance(value, (list, tuple)):
        return list(value[:limit])
    return value


def first_sequence_plan(data_batch: dict[str, Any]) -> Any:
    sequence_plan = data_batch.get("sequence_plan")
    if isinstance(sequence_plan, (list, tuple)):
        return sequence_plan[0] if sequence_plan else None
    return sequence_plan


def sequence_plan_condition_frame_indexes_vision(sequence_plan: Any) -> list[int]:
    if sequence_plan is None:
        return []
    condition_indexes = getattr(sequence_plan, "condition_frame_indexes_vision", None)
    if condition_indexes is None and isinstance(sequence_plan, dict):
        condition_indexes = sequence_plan.get("condition_frame_indexes_vision")
    return [int(idx) for idx in condition_indexes or []]


def sequence_plan_has_vision_conditioning(sequence_plan: Any) -> bool:
    return bool(sequence_plan_condition_frame_indexes_vision(sequence_plan))


def sequence_plan_has_action(sequence_plan: Any) -> bool:
    if sequence_plan is None:
        return False
    has_action = getattr(sequence_plan, "has_action", None)
    if has_action is None and isinstance(sequence_plan, dict):
        has_action = sequence_plan.get("has_action")
    return bool(has_action)


def condition_frame_indexes_vision_from_batch(data_batch: dict[str, Any]) -> list[int]:
    return sequence_plan_condition_frame_indexes_vision(first_sequence_plan(data_batch))
