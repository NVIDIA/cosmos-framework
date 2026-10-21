# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""comm — DATA PLANE (DESIGN §3, §9).

v1+: NCCL/Gloo primitives → transport (Codec / SQ-CQ engine / backends / registry /
metadata-bulk) → channel semantics + weight_transfer. The controller never touches
bulk; bulk flows peer-to-peer between workers.

v0: single process ⇒ direct in-process channels; weight-transfer is a no-op.
"""

from cosmos_framework.posttrain.comm.bulk import BulkRef, BulkStore, store_for
from cosmos_framework.posttrain.comm.channel import Channel, Plane
from cosmos_framework.posttrain.comm.codec import materialize_obs, materialize_payload, pack_payload
from cosmos_framework.posttrain.comm.transfer.mover import MoveHandle, Mover
from cosmos_framework.posttrain.comm.transfer.movers import (
    CopyMover,
    IpcMover,
    LocalStoreMover,
    NcclMover,
    SharedMover,
    make_mover,
    resolve_mover,
)
from cosmos_framework.posttrain.comm.weight_transfer import (
    WeightMapper,
    WeightPayload,
    WeightSync,
    make_weight_backend,
)

__all__ = [
    "Channel",
    "Plane",
    # weight plane (§8.2 #4)
    "WeightSync",
    "WeightMapper",
    "WeightPayload",
    "make_weight_backend",
    # metadata/bulk + transfer primitive (§3.2-§3.4)
    "BulkRef",
    "BulkStore",
    "store_for",
    "materialize_obs",
    "pack_payload",
    "materialize_payload",
    "Mover",
    "MoveHandle",
    "SharedMover",
    "CopyMover",
    "IpcMover",
    "NcclMover",
    "LocalStoreMover",
    "make_mover",
    "resolve_mover",
]
