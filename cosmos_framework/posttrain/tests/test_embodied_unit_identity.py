# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""``WorkUnit.unit_id`` is a PERSISTED identity, so its encoding is pinned here.

A unit id names a lease file on shared storage. Re-identifying units mid-campaign would make
every claimed unit look unclaimed and every finished one unfinished -- so the failure mode of
a "tidy up the canonical encoding" edit is not a wrong number, it is a campaign that
re-dispatches work it already has on disk.

These tests exist because the encoding MOVED: ``WorkUnit.make`` used to inline
``json.dumps(..., sort_keys=True)`` + ``sha1[:16]``, and now calls ``schema.identity.digest``.
The move is only safe if it is byte-identical, and "byte-identical" is a claim that needs a
test rather than a reading. The literals below were produced by the pre-move implementation.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest

from cosmos_framework.posttrain.embodied.unit import EpisodeKey, WorkUnit
from cosmos_framework.posttrain.schema import identity

#: MODULE-LEVEL, not one decorator per test: an unmarked test in this repo SKIPS silently, so a
#: per-test marker is one merge conflict away from disarming the check it guards.
pytestmark = [pytest.mark.L0, pytest.mark.CPU]

#: Produced by the INLINE implementation this module's subject replaced. Not recomputed from
#: the code under test -- a pin recomputed from its subject pins nothing.
_PINNED_UNIT_ID = "862892ac19171f9d"
_PINNED_BLOB = '{"a": [1, 2], "b": 1}'
_PINNED_DIGEST = "764fdd79efeda11b"


def _reference_unit_id(suite: str, scenario: dict, episodes) -> str:
    """The pre-move implementation, kept verbatim as the differential's other side."""
    blob = json.dumps(
        {"suite": suite, "scenario": dict(sorted(scenario.items())), "eps": [e.slug() for e in episodes]},
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:16]  # noqa: S324 -- identity, not a credential


def _unit(**over):
    scenario = {
        "task": "turning_on_radio",
        "instances": [0, 1],
        "num_envs": 1,
        "max_steps": 2150,
        "mode": "public_test",
    }
    scenario.update(over.pop("scenario", {}))
    episodes = over.pop(
        "episodes",
        tuple(EpisodeKey(suite="behavior1k", task="turning_on_radio", instance=i, seed=i) for i in range(2)),
    )
    return WorkUnit.make(suite="behavior1k", scenario=scenario, episodes=episodes, est_minutes=10.0)


def test_a_known_unit_id_did_not_move():
    assert _unit().unit_id == _PINNED_UNIT_ID


def test_the_canonical_settings_did_not_move():
    """Separators, key order, hash and truncation, pinned as one observable value."""
    assert identity.canonical_blob({"b": 1, "a": [1, 2]}) == _PINNED_BLOB
    assert identity.digest({"b": 1, "a": [1, 2]}) == _PINNED_DIGEST
    assert identity.HASH == "sha1"
    assert identity.ID_LENGTH == 16


@pytest.mark.parametrize(
    "scenario",
    [
        {},
        {"nested": {"b": [1, {"c": True}], "a": None}},
        {"unicode": "radius ✓ ±½ µm — Ω", "empty_map": {}, "empty_list": []},
        {"floats": [0.1, -2.5, 1e300], "bools": [True, False], "none": None},
        {"tuple_becomes_list": (1, 2, ("x",))},
        {"z": 1, "a": 2, "m": 3},  # insertion order must not reach the digest
    ],
    ids=["plain", "nested", "unicode-and-empties", "scalars", "tuple", "key-order"],
)
def test_the_extracted_encoder_agrees_with_the_inlined_one(scenario):
    unit = _unit(scenario=scenario)
    merged = {
        "task": "turning_on_radio",
        "instances": [0, 1],
        "num_envs": 1,
        "max_steps": 2150,
        "mode": "public_test",
        **scenario,
    }
    episodes = tuple(EpisodeKey(suite="behavior1k", task="turning_on_radio", instance=i, seed=i) for i in range(2))
    assert unit.unit_id == _reference_unit_id("behavior1k", merged, episodes)


def test_dict_ordering_cannot_change_the_id():
    forward = _unit(scenario={"a": 1, "b": 2})
    reverse_scenario = {"mode": "public_test", "max_steps": 2150, "num_envs": 1, "instances": [0, 1]}
    reverse_scenario.update({"b": 2, "a": 1, "task": "turning_on_radio"})
    assert forward.unit_id == _unit(scenario=reverse_scenario).unit_id


@pytest.mark.parametrize(
    "value,expected",
    [
        ({"x": float("nan")}, ValueError),
        ({"x": float("inf")}, ValueError),
        ({1: "int key"}, TypeError),
        ({"x": {object()}}, TypeError),
        ({"x": object()}, TypeError),
    ],
    ids=["nan", "inf", "int-key", "set", "object"],
)
def test_the_two_silent_identity_hazards_are_refused(value, expected):
    """Both cases could otherwise produce an id nobody can trust; neither was valid before."""
    with pytest.raises(expected):
        identity.digest(value)


def test_the_refusal_names_where_the_bad_value_is():
    with pytest.raises(ValueError, match=r"\$\.outer\[1\]\.inner"):
        identity.digest({"outer": [{}, {"inner": float("nan")}]})


def test_the_identity_module_needs_nothing_but_the_standard_library():
    """The reason it is a separate module from ``utils/identity.py``, asserted not asserted-in-prose.

    ``utils.identity`` implements the same rule but imports torch, attrs and omegaconf, none of
    which exist in the BEHAVIOR-1K container. If this module ever grows one of those imports the
    simulator surface stops loading -- in the sim venv, not here -- so the check has to be an
    import-closure measurement rather than a source grep.
    """
    code = (
        "import sys, json;"
        "before = set(sys.modules);"
        "import cosmos_framework.posttrain.schema.identity;"
        "roots = {m.split('.')[0] for m in set(sys.modules) - before};"
        # `sys.stdlib_module_names` rather than a hand-written allow list: a list would need
        # editing every time CPython moves a helper, and the edit that keeps the test green is
        # indistinguishable from the edit that lets a real dependency through.
        "print(json.dumps(sorted(roots - sys.stdlib_module_names)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    third_party = json.loads(out.stdout.strip().splitlines()[-1])
    assert third_party == ["projects"], f"schema.identity pulled in {third_party}"


def test_that_dependency_check_can_actually_fail():
    """The positive control. Same measurement against a module that DOES import torch."""
    code = (
        "import sys, json;"
        "before = set(sys.modules);"
        "import cosmos_framework.posttrain.utils.identity;"
        "roots = {m.split('.')[0] for m in set(sys.modules) - before};"
        "print(json.dumps(sorted(roots - sys.stdlib_module_names)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    third_party = json.loads(out.stdout.strip().splitlines()[-1])
    assert "torch" in third_party, f"utils.identity was expected to be heavy; got {third_party}"
