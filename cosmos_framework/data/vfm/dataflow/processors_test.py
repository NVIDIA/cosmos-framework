# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for IdentityProcessor."""

from __future__ import annotations

from cosmos_framework.data.vfm.dataflow.processors import IdentityProcessor


def test_identity_returns_item_unchanged():
    item = {"input_ids": [1, 2, 3], "label": 7}
    out = IdentityProcessor().process(item)
    assert out is item


def test_identity_passes_non_dict_through():
    obj = object()
    assert IdentityProcessor().process(obj) is obj
