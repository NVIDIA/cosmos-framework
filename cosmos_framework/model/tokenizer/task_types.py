# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fail-closed parsing for tokenizer task-family metadata."""

from __future__ import annotations

import re
from enum import Enum
from functools import lru_cache


class TaskFamily(str, Enum):
    """Mutually exclusive tokenizer task families used for runtime routing."""

    IMAGE_RECONSTRUCTION = "image_rec"
    VIDEO_RECONSTRUCTION = "video_rec"
    IMAGE_TEXT = "image_text"
    IMAGE_CAPTION = "image_caption"
    IMAGE_VQA = "image_vqa"
    COCO_RETRIEVAL = "coco_retrieval"


_TASK_FAMILY_PATTERNS = {family: re.compile(rf"(?:^|[_-]){re.escape(family.value)}(?:$|[_-])") for family in TaskFamily}


@lru_cache(maxsize=256)
def get_task_family(task_type: str) -> TaskFamily | None:
    """Return one unambiguous known family, or ``None`` for an unrelated evaluation task."""
    if not isinstance(task_type, str) or not task_type or task_type != task_type.strip():
        raise ValueError(f"Tokenizer task type must be one non-empty normalized string, got {task_type!r}.")
    matches = [family for family, pattern in _TASK_FAMILY_PATTERNS.items() if pattern.search(task_type)]
    if len(matches) > 1:
        raise ValueError(
            f"Tokenizer task type {task_type!r} ambiguously matches multiple families: "
            f"{[family.value for family in matches]}."
        )
    return matches[0] if matches else None


def task_type_is(task_type: str, *families: TaskFamily) -> bool:
    """Return whether ``task_type`` belongs to one of the requested families."""
    return get_task_family(task_type) in families


def normalize_task_type(task_type: str | list[str] | tuple[str, ...] | None, *, default: str) -> str:
    """Normalize collated task metadata while rejecting mixed-family or mixed-name batches."""
    if isinstance(task_type, list | tuple):
        if not task_type:
            normalized = default
        else:
            if any(not isinstance(value, str) for value in task_type):
                raise ValueError(f"Tokenizer task type batch must contain only strings, got {task_type!r}.")
            unique_task_types = set(task_type)
            if len(unique_task_types) != 1:
                raise ValueError(f"Tokenizer batch mixes task types: {sorted(unique_task_types)}.")
            normalized = task_type[0]
    elif task_type is None:
        normalized = default
    elif isinstance(task_type, str):
        normalized = task_type
    else:
        raise ValueError(f"Tokenizer task type must be a string, sequence of strings, or None, got {task_type!r}.")
    get_task_family(normalized)
    return normalized


__all__ = ["TaskFamily", "get_task_family", "normalize_task_type", "task_type_is"]
