# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0/CPU tests for bind_check: the real DPO manifest passes; wiring and field mismatches are
rejected by category; and the report renders obligations."""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.control.bind import CheckObligation, CheckReport, bind_check
from cosmos_framework.posttrain.control.capability import (
    SUPERVISION_KINDS,
    CapabilityDescriptor,
    FieldSpec,
    get_capability,
    register_capability,
)
from cosmos_framework.posttrain.control.manifest import Edge, Manifest, Plane, RoleDecl
from cosmos_framework.posttrain.control.recipe import dpo_manifest
from cosmos_framework.posttrain.entrypoints.dpo import ROLE_SPECS
from cosmos_framework.posttrain.roles.env.preference import _pair_trajectory
from cosmos_framework.posttrain.schema import Preference


@pytest.fixture(scope="module", autouse=True)
def _test_capabilities():
    """Register this module's test-only impls, then restore the registry. Registering at import instead
    would leave them in the process-global registry for the rest of the session, making its contents
    depend on which test modules pytest collected."""
    from cosmos_framework.posttrain.control import capability

    saved = capability._REGISTRY.copy()
    register_capability(
        "t_kind_mismatch_consumer",
        CapabilityDescriptor(consumes=(FieldSpec("targets", "reward"),), planes=frozenset({Plane.DATA})),
    )
    register_capability(
        "t_missing_field_consumer",
        CapabilityDescriptor(consumes=(FieldSpec("reward", "reward"),), planes=frozenset({Plane.DATA})),
    )
    yield
    capability._REGISTRY.clear()
    capability._REGISTRY.update(saved)


@pytest.mark.L0
@pytest.mark.CPU
def test_real_dpo_manifest_passes():
    report = bind_check(dpo_manifest(ROLE_SPECS))
    assert report.ok, report.format()
    assert report.errors == ()


@pytest.mark.L0
@pytest.mark.CPU
def test_role_specs_drift_is_visible_to_the_check():
    """The manifest is derived from the entry's ROLE_SPECS, so a role added there without a registered
    capability is a wiring error — the check cannot keep reporting ok for a topology that changed."""
    drifted = [*ROLE_SPECS, ("rollout", "rollout")]
    errors = bind_check(dpo_manifest(drifted)).errors
    assert any(e.category == "wiring" and "dpo_rollout" in e.message for e in errors)


@pytest.mark.L0
@pytest.mark.CPU
def test_dangling_edge_is_wiring_error():
    m = Manifest(roles=(RoleDecl("data", "dpo_env"),), edges=(Edge("data", "ghost"),))
    errors = bind_check(m).errors
    assert len(errors) == 1 and errors[0].category == "wiring"
    # Golden text: the rejection message is the user-facing product — a reword is a deliberate test
    # change, never silent drift.
    assert errors[0].where == "data->ghost"
    assert errors[0].message == "edge dst 'ghost' names no role"


@pytest.mark.L0
@pytest.mark.CPU
def test_unknown_impl_key_is_wiring_error():
    m = Manifest(roles=(RoleDecl("data", "nope"),), edges=())
    assert any(e.category == "wiring" and "nope" in e.message for e in bind_check(m).errors)


@pytest.mark.L0
@pytest.mark.CPU
def test_unsupported_plane_is_wiring_error():
    # dpo_env / dpo_trainer support DATA only; a WEIGHT edge between them is unsupported.
    m = Manifest(
        roles=(
            RoleDecl("data", "dpo_env"),
            RoleDecl("policy", "dpo_trainer"),
        ),
        edges=(Edge("data", "policy", plane=Plane.WEIGHT),),
    )
    assert any(e.category == "wiring" and "plane" in e.message for e in bind_check(m).errors)


@pytest.mark.L0
@pytest.mark.CPU
def test_field_kind_mismatch_is_one_field_error():
    m = Manifest(
        roles=(
            RoleDecl("data", "dpo_env"),
            RoleDecl("c", "t_kind_mismatch_consumer"),
        ),
        edges=(Edge("data", "c", plane=Plane.DATA),),
    )
    field_errors = [e for e in bind_check(m).errors if e.category == "field"]
    assert len(field_errors) == 1  # 'targets' present but kind preference != reward


@pytest.mark.L0
@pytest.mark.CPU
def test_missing_required_field_is_field_error():
    m = Manifest(
        roles=(
            RoleDecl("data", "dpo_env"),
            RoleDecl("c", "t_missing_field_consumer"),
        ),
        edges=(Edge("data", "c", plane=Plane.DATA),),
    )
    errors = bind_check(m).errors
    assert len(errors) == 1 and errors[0].category == "field"
    # Golden text: the rejection message is the user-facing product — a reword is a deliberate test
    # change, never silent drift.
    assert errors[0].where == "data->c"
    assert errors[0].message == "consumer needs 'reward' (reward); producer emits ['targets']"


@pytest.mark.L0
@pytest.mark.CPU
def test_report_format_obligations_and_raise():
    ok_with_obligation = CheckReport(
        ok=True,
        errors=[],
        obligations=(CheckObligation("placement", "policy", "verify GPU budget on the live cluster"),),
    )
    assert "runtime obligations" in ok_with_obligation.format()
    ok_with_obligation.raise_for_status()  # obligations never fail
    with pytest.raises(ValueError):
        bind_check(Manifest(roles=(RoleDecl("data", "nope"),), edges=())).raise_for_status()


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_env_kind_grounded_in_real_supervision_type():
    # The 'dpo_env' produced kind must equal the kind of a real _pair_trajectory preference transition.
    transition = _pair_trajectory(0, "p", winner={"v": 1}, loser={"v": 2}).transitions[0]
    pref = transition.targets  # parse-at-production: the producer emits a typed Preference
    assert isinstance(pref, Preference)
    assert SUPERVISION_KINDS[type(pref)] == get_capability("dpo_env").produces[0].kind
