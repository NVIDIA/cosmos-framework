# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: the TRANSPORT rejection class. `transport_capability_ok` is the plane-aware predicate
lifted from `resolve_mover` (one source of truth); `bind_check` rejects weight-plane transport
mismatches AOT (category='transport'), data-plane edges are always movable, and an untagged edge
surfaces as an obligation rather than passing silently. resolve_mover's runtime parity is guarded by
tests/test_comm_transfer_movers.py."""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.comm.transfer.movers import transport_capability_ok
from cosmos_framework.posttrain.control.bind import bind_check
from cosmos_framework.posttrain.control.capability import CapabilityDescriptor, register_capability
from cosmos_framework.posttrain.control.manifest import Edge, Manifest, Plane, RoleDecl
from cosmos_framework.posttrain.control.recipe import dpo_manifest
from cosmos_framework.posttrain.entrypoints.dpo import ROLE_SPECS


@pytest.fixture(scope="module", autouse=True)
def _test_capabilities():
    """A test-only role supporting BOTH planes, so a weight-plane edge passes wiring and the transport
    check is exercised in isolation. Registered/restored per module so it does not leak into the
    process-global registry for the rest of the session."""
    from cosmos_framework.posttrain.control import capability

    saved = capability._REGISTRY.copy()
    register_capability("t_weight_role", CapabilityDescriptor(planes=frozenset({Plane.DATA, Plane.WEIGHT})))
    yield
    capability._REGISTRY.clear()
    capability._REGISTRY.update(saved)


def _weight_manifest(relationship: str | None, capability: str | None) -> Manifest:
    return Manifest(
        roles=(
            RoleDecl("policy", "t_weight_role"),
            RoleDecl("rollout", "t_weight_role"),
        ),
        edges=(
            Edge("policy", "rollout", plane=Plane.WEIGHT, relationship=relationship, transport_capability=capability),
        ),
    )


def _transport(report) -> list:
    return [e for e in report.errors if e.category == "transport"]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "relationship,plane,capability,expected_ok",
    [
        ("same_gpu", "data", "object", True),  # data plane always movable (mirrors the runtime early-return)
        ("same_process", "weight", "object", True),  # SharedMover, not tensors_only
        ("host", "weight", "object", True),  # CopyMover, not tensors_only
        ("same_gpu", "weight", "tensor", True),  # IpcMover with a tensor payload
        ("cross_device", "weight", "tensor", True),  # NcclMover with a tensor payload
        ("same_gpu", "weight", "object", False),  # IpcMover is tensors_only: object rejected
        ("cross_device", "weight", "object", False),
        ("cross_device", "weight", None, False),
        ("bogus", "weight", "tensor", False),  # unknown relationship
    ],
)
def test_transport_capability_ok_decision_table(relationship, plane, capability, expected_ok):
    """The predicate reproduces resolve_mover's per-plane decisions without constructing a mover."""
    ok, reason = transport_capability_ok(relationship, plane, capability)
    assert ok is expected_ok
    assert (reason == "") is expected_ok  # a reason is present iff the edge is rejected


@pytest.mark.L0
@pytest.mark.CPU
def test_rejects_transport_mismatch():
    report = bind_check(_weight_manifest("same_gpu", "object"))
    errs = _transport(report)
    assert len(errs) == 1
    # Golden text: the rejection message is the user-facing product — a reword is a deliberate test
    # change, never silent drift.
    assert errs[0].where == "policy->rollout"
    assert errs[0].message == "IpcMover is tensors_only; resolve with capability='tensor' (got 'object')"
    assert not [e for e in report.errors if e.category == "wiring"]  # t_weight_role supports WEIGHT


@pytest.mark.L0
@pytest.mark.CPU
def test_bind_rejects_unknown_relationship():
    errs = _transport(bind_check(_weight_manifest("bogus", "tensor")))
    assert len(errs) == 1 and "unknown relationship" in errs[0].message


@pytest.mark.L0
@pytest.mark.CPU
def test_valid_weight_edge_passes_transport():
    assert not _transport(bind_check(_weight_manifest("same_gpu", "tensor")))


@pytest.mark.L0
@pytest.mark.CPU
def test_untagged_edge_is_obligation_not_error():
    m = Manifest(
        roles=(
            RoleDecl("policy", "t_weight_role"),
            RoleDecl("rollout", "t_weight_role"),
        ),
        edges=(Edge("policy", "rollout", plane=Plane.WEIGHT),),  # relationship/transport_capability left None
    )
    report = bind_check(m)
    assert not _transport(report)
    assert any(o.category == "transport" for o in report.obligations)


@pytest.mark.L0
@pytest.mark.CPU
def test_tagged_dpo_manifest_transport_ok():
    report = bind_check(dpo_manifest(ROLE_SPECS))  # data-plane, same_process/object → always movable
    assert not _transport(report)
    assert report.ok
