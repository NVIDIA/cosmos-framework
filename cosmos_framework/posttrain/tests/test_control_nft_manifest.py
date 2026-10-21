# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GPU-less contract tests for the online-WFM manifest."""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.control.bind import bind_check
from cosmos_framework.posttrain.control.recipe import nft_manifest
from cosmos_framework.posttrain.entrypoints.nft import ROLE_SPECS


@pytest.mark.L0
@pytest.mark.CPU
def test_online_nft_manifest_is_fully_bound_and_colocated() -> None:
    manifest = nft_manifest(ROLE_SPECS)
    report = bind_check(manifest)
    assert report.ok, report.format()
    assert not report.obligations
    assert [(edge.src, edge.dst) for edge in manifest.edges] == [
        ("data", "rollout"),
        ("rollout", "rm"),
        ("rm", "replay"),
        ("replay", "policy"),
        ("policy", "rollout"),
    ]
    assert all(edge.relationship == "same_process" for edge in manifest.edges)
    assert manifest.edges[-1].transport_capability == "tensor"
