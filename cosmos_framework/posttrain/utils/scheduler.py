# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Runtime exact-resume identities for learning-rate schedulers."""

from __future__ import annotations

from typing import Any
from weakref import WeakKeyDictionary

from cosmos_framework.posttrain.utils.identity import normalize_json_dict

_SCHEDULER_RESUME_IDENTITIES: WeakKeyDictionary[Any, dict[str, Any]] = WeakKeyDictionary()


def bind_scheduler_resume_identity(scheduler: Any, identity: dict[str, Any]) -> None:
    """Associate immutable construction semantics with one live scheduler.

    The identity deliberately lives outside ``scheduler.__dict__``: PyTorch's generic scheduler
    ``state_dict()`` serializes arbitrary instance attributes, while the trainer persists this
    contract separately and validates it before loading mutable scheduler state.
    """
    _SCHEDULER_RESUME_IDENTITIES[scheduler] = normalize_json_dict(
        identity,
        context="configured scheduler exact-resume identity",
    )


def scheduler_resume_identity(scheduler: Any) -> dict[str, Any] | None:
    """Return a detached scheduler identity, or ``None`` for legacy/unpinned schedulers."""
    if scheduler is None:
        return None
    identity = _SCHEDULER_RESUME_IDENTITIES.get(scheduler)
    if identity is None:
        return None
    return normalize_json_dict(identity, context="configured scheduler exact-resume identity")
