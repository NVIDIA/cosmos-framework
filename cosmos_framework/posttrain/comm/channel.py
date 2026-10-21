# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Channel + transport (DESIGN §3.1-§3.3).

A channel connects two WG ports. Its semantics are `capacity / staleness / plane`;
the transport backend is resolved by placement. v0: everything is in one process,
so transport = `direct` (an in-process queue) and capacity is a plain bound.

TODO(v1+, §3): three planes (data/weight/control) with binder-enforced
stream/communicator isolation; pluggable transport (direct/IPC/UCX/NCCL) resolved
by placement; 5-layer stack (Codec / Transfer engine SQ-CQ / backends / registry);
metadata/bulk split + on-demand pull. NONE of that is needed for single-process v0.
"""

from __future__ import annotations

import collections
from enum import Enum
from typing import Any


class Plane(str, Enum):
    DATA = "data"
    WEIGHT = "weight"
    CONTROL = "control"


class Channel:
    """v0: an in-process FIFO. capacity=0 ⇒ synchronous (on-policy); capacity=k ⇒
    bounded buffer (the async/staleness knob, §2.2) — but v0 runner is sync, so
    capacity is effectively unbounded/ignored here.
    """

    def __init__(self, plane: Plane = Plane.DATA, capacity: int = 0, dispatch: Any = None):
        self.plane = plane
        self.capacity = capacity  # sync/async knob (§2.2); v0 sync
        self.dispatch = dispatch  # TODO(§2.4): DispatchPolicy; v0 = direct
        self._q: collections.deque = collections.deque()

    def put(self, item: Any) -> None:
        # DESIGN §3.5 contract: put transfers ownership. v0 in-process: just enqueue.
        self._q.append(item)

    def get(self) -> Any:
        return self._q.popleft()

    def __len__(self) -> int:
        return len(self._q)


# TODO(v1+): resolve_transport(payload, placement_relationship, plane, capability)
#   → direct | UCX | NCCL ... (DESIGN §3.3). v0 always `direct`.
