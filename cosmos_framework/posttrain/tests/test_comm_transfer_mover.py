# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""MoveHandle / BulkRef / Mover protocol basics (no torch, no ray)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from cosmos_framework.posttrain.comm.transfer.mover import (
    COLLECTIVE,
    STAGED,
    BulkRef,
    MoveHandle,
)


@pytest.mark.L0
@pytest.mark.CPU
def test_movehandle_fields_and_frozen():
    h = MoveHandle(kind="copy", payload={"a": 1}, meta={"store_id": "s0"})
    assert h.kind == "copy" and h.payload == {"a": 1} and h.meta == {"store_id": "s0"}
    with pytest.raises(FrozenInstanceError):  # frozen dataclass
        h.kind = "ipc"


@pytest.mark.L0
@pytest.mark.CPU
def test_bulkref_is_movehandle_subtype_with_accessors():
    ref = BulkRef.for_bulk(store_id="s0", key=7, nbytes=2048, dtype="float32")
    assert isinstance(ref, MoveHandle)
    assert ref.kind == "copy"
    assert ref.store_id == "s0" and ref.key == 7 and ref.nbytes == 2048 and ref.dtype == "float32"


@pytest.mark.L0
@pytest.mark.CPU
def test_lifetime_constants():
    assert STAGED == "staged" and COLLECTIVE == "collective"
