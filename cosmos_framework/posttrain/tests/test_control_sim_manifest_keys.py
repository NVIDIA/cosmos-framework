# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""One ``impl_key`` must mean the same thing in all three registries.

An ``impl_key`` names a role in three places that never check each other: the
``CapabilityDescriptor`` (``control/capabilities_builtin.py``), the ``RoleDecl``
(``control/recipe.py::sim_manifest``) and the body builder (``roles/core/registry.py``). When
those are three string literals, a near-miss spelling is caught by a campaign rather than by
``bind_check`` -- the manifest validates a contract for a role whose body the framework cannot
build, and both halves look right in isolation.

This is also the file the docstring gate once caught `control/recipe.py` claiming existed. It
asserted ``tests/test_control_recipe_sim.py`` "pins" the ``bind_check`` vacuity; that file was
never in the tree and nothing exercised ``sim_manifest`` at all. So the paragraph below is what
that claim should have been standing on.
"""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.control import capabilities_builtin as caps
from cosmos_framework.posttrain.control.bind import bind_check
from cosmos_framework.posttrain.control.capability import get_capability
from cosmos_framework.posttrain.control.recipe import sim_manifest
from cosmos_framework.posttrain.roles.core import registry
from cosmos_framework.posttrain.roles.env import sim_gateway, sim_role

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_the_manifest_names_no_key_that_is_not_the_shared_constant():
    """A third literal is exactly what this pair of constants exists to prevent."""
    declared = {r.impl_key for r in sim_manifest().roles}
    assert declared == {caps.SIM_IMPL_KEY, caps.POLICY_IMPL_KEY}


@pytest.mark.parametrize("key", ["SIM_IMPL_KEY", "POLICY_IMPL_KEY"])
def test_every_manifest_key_resolves_a_capability(key):
    assert get_capability(getattr(caps, key)) is not None


def test_the_sim_key_now_resolves_a_body_builder_too():
    """The registry entry the review asked for. Before it, the env table was empty."""
    assert registry._TABLES["env"][caps.SIM_IMPL_KEY] is sim_gateway.build_sim_gateway_body


def test_building_through_the_registry_reaches_the_real_constructor():
    """The production call SHAPE, not a stub: `build_env` must return a live body.

    A test that only checked the table entry would pass against a builder whose signature no
    caller could satisfy -- registered, resolvable, and unusable.

    ``sim_python``/``sim_gpu`` are mandatory and that is the constructor REFUSING to guess: a
    guessed interpreter yields a simulator that runs in the wrong venv, and a guessed device
    one that runs on another cell's card. Both produce a run rather than a failure. Passing
    them here is the production shape, not test scaffolding.
    """
    body = registry.build_env(
        caps.SIM_IMPL_KEY,
        {"sim": "behavior1k", "sink": "null", "render_every": 3, "sim_python": "/usr/bin/python3", "sim_gpu": "0"},
    )
    assert isinstance(body, sim_gateway.SimGatewayBody)
    assert body._child_cfg["sink"] == "null"
    assert body._child_cfg["render_every"] == 3


def test_the_builder_does_not_soften_the_constructor_refusal():
    """Going through the registry must not become a way to skip a refusal."""
    from cosmos_framework.posttrain.embodied.wire import WireError

    with pytest.raises(WireError, match="sim_python"):
        registry.build_env(caps.SIM_IMPL_KEY, {"sim": "behavior1k", "sink": "null"})


def test_a_mistyped_key_names_the_ones_that_exist():
    with pytest.raises(KeyError, match=caps.SIM_IMPL_KEY):
        registry.build_env("behavior1k-sim", {})


def test_the_manifest_still_binds():
    report = bind_check(sim_manifest())
    assert not report.errors, report.errors


# ── module ownership: the placement policy is not part of the contract ──────


def test_the_same_card_refusal_lives_with_the_gateway_not_with_the_contract():
    """S-04 is a property of the body that can violate it, not of every `SimRole`.

    An in-process `CollectorBody` holds one card by construction and cannot cross; only the
    gateway, whose simulator is a separate process, can. Beside the contract this read as an
    obligation on everything implementing `SimRole`.
    """
    assert hasattr(sim_gateway, "refuse_cross_card_gateway")
    assert hasattr(sim_gateway, "CardPinningError")
    assert not hasattr(sim_role, "refuse_cross_card_gateway")
    assert not hasattr(sim_role, "CardPinningError")


def test_the_contract_module_still_exports_the_contract():
    """The move must not have taken anything a caller needs with it."""
    for name in ("SimRole", "capabilities_as_wire", "capabilities_from", "validate_sim_role"):
        assert hasattr(sim_role, name), name
