# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Behavioural tests for the three contract gates, each pinned by the payload that fooled it.

``assert_wire_safe``, ``capabilities_from`` and ``validate_sim_role`` are the checks that stand
between this framework and a simulator written by someone else. Until this file they had no
direct tests: every one of the defects below was present while the suite was green, because the
tests that existed exercised the happy path and the gates only speak on the unhappy one.

So each case here is a payload that PASSED, with what it would have caused. A gate whose
negative cases are untested is a gate whose failure mode is that it silently stops gating.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict

import numpy as np
import pytest

from cosmos_framework.posttrain.embodied.sim import SimCapabilities, SimProtocolError, assert_wire_safe
from cosmos_framework.posttrain.embodied.unit import EpisodeKey, WorkUnit
from cosmos_framework.posttrain.roles.env.sim_role import (
    capabilities_as_wire,
    capabilities_from,
    validate_sim_role,
)


def _payload(**over):
    base = dict(
        protocol_version=1,
        backend="example",
        backend_version="v0",
        embodiment="example",
        max_lanes=1,
        camera_shapes={"head": [64, 64]},
        proprio_width=7,
        extras=["depth"],
        supports_render_skip=False,
    )
    base.update(over)
    return base


class _Role:
    """The minimum ``validate_sim_role`` calls: ``probe()`` returning a capability payload."""

    def __init__(self, caps: SimCapabilities) -> None:
        self._caps = caps

    def probe(self) -> dict:
        return {"capabilities": capabilities_as_wire(self._caps), "host": "h", "gpu": None}


# ── assert_wire_safe ────────────────────────────────────────────────────────────


@pytest.mark.L0
@pytest.mark.CPU
def test_a_structured_dtype_holding_objects_is_rejected():
    """``dtype == object`` is False for ``[("payload", object)]`` -- the box got through.

    The array is still a box holding arbitrary Python: a torch tensor or an engine handle in
    that field crosses the seam and is unpickled on the other side, pulling the engine with it.
    That is precisely what the object-dtype rule exists to stop, and the equality test admitted
    it while the docstring claimed otherwise.
    """
    arr = np.empty(1, dtype=[("payload", object)])
    arr[0]["payload"] = lambda: None
    with pytest.raises(SimProtocolError, match="holds Python"):
        assert_wire_safe({"x": arr}, "obs")


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "value",
    [defaultdict(list, {"a": 1}), OrderedDict(a=1)],
    ids=["defaultdict", "OrderedDict"],
)
def test_a_container_subclass_is_rejected_even_when_every_item_is_safe(value):
    """The walk visits ITEMS; a subclass carries state it never visits.

    ``defaultdict.default_factory`` is a callable and any subclass may hold instance
    attributes. cloudpickle ships that hidden state regardless, so "every item is wire-safe"
    stopped implying "this object is wire-safe".
    """
    with pytest.raises(SimProtocolError, match="SUBCLASS"):
        assert_wire_safe(value, "obs")


@pytest.mark.L0
@pytest.mark.CPU
def test_the_plain_shapes_still_pass():
    """THE POSITIVE CONTROL. A tightening that rejects everything is not a tightening."""
    assert_wire_safe({"cam": np.zeros((2, 2), np.float32), "n": 1, "s": "t", "l": [1, 2], "t": (None, True)}, "obs")


# ── capabilities_from ───────────────────────────────────────────────────────────


@pytest.mark.L0
@pytest.mark.CPU
def test_a_wire_roundtrip_is_the_identity():
    """The positive control, and the one that must survive every check added above."""
    caps = SimCapabilities(1, "example", "v0", "example", 1, {"head": (64, 64)}, 7, ("depth",), False)
    assert capabilities_from(capabilities_as_wire(caps)) == caps


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "over, why",
    [
        ({"supports_render_skip": "false"}, "bool('false') is True -- render skip turns ON"),
        ({"protocol_version": 1.9}, "int(1.9) is 1 -- a version skew reported as agreement"),
        ({"max_lanes": True}, "bool subclasses int, so True read as 1 lane"),
        ({"extras": "depth"}, "tuple('depth') is five one-character extras"),
        ({"camera_shapes": {"head": 5}}, "shape[0] on an int raised TypeError, not SimProtocolError"),
        ({"camera_shapes": {"head": [64]}}, "shape[1] on a 1-sequence raised IndexError"),
        ({"camera_shapes": {"head": [0, 64]}}, "a zero-height camera is not a resolution"),
        ({"max_lanes": -1}, "a negative ceiling passes every >= comparison below it"),
        ({"backend": 5}, "str(5) is '5' -- a backend name invented by the codec"),
    ],
)
def test_a_malformed_capability_payload_is_refused_as_a_protocol_error(over, why):
    """Each case COERCED to something plausible before, or escaped as the wrong exception.

    This payload crosses a process boundary and, since the container split, may come from
    another team's simulator built against the published contract. It is the one place where a
    foreign implementation's mistake can still become a legible error instead of a run that
    produces well-formed data meaning something else.
    """
    with pytest.raises(SimProtocolError):
        capabilities_from(_payload(**over))


@pytest.mark.L0
@pytest.mark.CPU
def test_a_missing_field_is_still_named():
    with pytest.raises(SimProtocolError, match="missing"):
        p = _payload()
        del p["embodiment"]
        capabilities_from(p)


# ── validate_sim_role ───────────────────────────────────────────────────────────


@pytest.mark.L0
@pytest.mark.CPU
def test_a_camera_the_run_requires_and_the_backend_lacks_is_a_mismatch():
    """Absence used to pass: only a non-``None`` shape was compared.

    A run naming ``head`` at 720 is stating a requirement. A backend with no head camera binds,
    boots, and produces episodes missing the stream the policy reads -- discovered minutes in,
    against a unit already claimed off the shared lease plane.
    """
    role = _Role(SimCapabilities(1, "b", "v", "e", 1, {}, 7, (), False))
    with pytest.raises(SimProtocolError, match="declares no such camera"):
        validate_sim_role(role, expect=SimCapabilities(1, "b", "v", "e", 1, {"head": (720, 720)}, 7, (), False))


@pytest.mark.L0
@pytest.mark.CPU
def test_a_run_that_names_no_camera_still_binds():
    """THE CASE THE OLD RULE WAS REALLY ABOUT, and the reason absence was permitted at all.

    The default configuration requests no resolution, so a backend declaring none -- or extra
    cameras nobody asked for -- must remain bindable. Tightening absence into a mismatch would
    be worthless if it made the default unbindable.
    """
    role = _Role(SimCapabilities(1, "b", "v", "e", 1, {"wrist": (480, 480)}, 7, (), False))
    assert validate_sim_role(role, expect=SimCapabilities(1, "b", "v", "e", 1, {}, 7, (), False)).max_lanes == 1


# ── WorkUnit.make ───────────────────────────────────────────────────────────────


@pytest.mark.L0
@pytest.mark.CPU
def test_a_generator_of_episodes_survives_identity():
    """The digest walked the iterable, then ``tuple()`` got an exhausted one and stored ``()``.

    The unit's id was a digest of episodes the unit did not contain, and every coverage check
    reads a unit with no episodes as complete.
    """
    unit = WorkUnit.make(
        suite="s",
        scenario={"task": "t"},
        episodes=(EpisodeKey(suite="s", task="t", instance=i, seed=i) for i in range(3)),
        est_minutes=1.0,
    )
    assert len(unit.episodes) == 3


@pytest.mark.L0
@pytest.mark.CPU
def test_mutating_the_callers_scenario_cannot_reach_the_unit():
    """``WorkUnit`` is frozen; the mapping handed to it was not.

    A caller mutating their dict afterwards changed ``unit.scenario`` while ``unit_id`` -- a
    digest taken before the mutation -- stayed the same, so a persisted lease named different
    work than it described. NESTED, because a shallow copy passes the top-level version of
    this test while leaving the hole open.
    """
    scenario = {"task": "t", "nested": {"depth": 1}}
    unit = WorkUnit.make(suite="s", scenario=scenario, episodes=(), est_minutes=1.0)
    scenario["task"] = "other"
    scenario["nested"]["depth"] = 99
    assert unit.scenario["task"] == "t"
    assert unit.scenario["nested"]["depth"] == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_a_unit_refuses_episodes_from_another_suite():
    """``slug()`` omits the suite, so these two keys collide inside one unit.

    Landing either marks both covered and ``missing()`` returns empty. Refused here rather than
    by widening the slug: that string is a path under the sink's root and a component of
    ``unit_id``, which names a lease file on NFS -- widening it re-identifies every episode in
    every existing corpus and resume stops recognising landed work.
    """
    with pytest.raises(ValueError, match="Build one unit per suite"):
        WorkUnit.make(
            suite="behavior1k",
            scenario={"task": "t"},
            episodes=(EpisodeKey("behavior1k", "t", 0, 0), EpisodeKey("robolab", "t", 0, 0)),
            est_minutes=1.0,
        )
