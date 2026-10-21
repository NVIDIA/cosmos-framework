# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Placement (DESIGN §4) — logical/physical decoupling.

Final form: produce a mapping (WG, replica, rank) → (node, bundle, process-slot);
manage memory arbitration + heterogeneous resource profiles; colocation tiers +
narrow contract; place-fix hard constraints; the scheduler (§11.1) decides, this
layer executes.

v0: everything is local (one process), so the mapping is trivially "all local" and
colocation (incl. sim/inference) is the natural degenerate — no special mechanism.
"""

from __future__ import annotations

from typing import Any


class Placement:
    """v0: all-local. The interface is kept so a real scheduler/binder + Ray PG
    drop in later without touching role/runner code."""

    def __init__(self, mode: str = "local"):
        self.mode = mode  # TODO(§11.1): auto ↔ constrained ↔ full-pin spectrum

    def resolve(self, wg) -> dict[str, Any]:
        # Shape comes from the WG so the §4.1 scheduler is the future swap point —
        # replace resolve's internals there to honor heterogeneous profiles + memory
        # arbitration; callers need no change.
        # TODO(§4.1): real mapping + memory arbitration + heterogeneous profiles.
        # TODO(§4.3): honor place-fix constraints (pin_to/colocate_with/...) from wg.placement.
        return {"node": "local", "process": "main", "replicas": wg.replicas, "ranks": wg.ranks}
