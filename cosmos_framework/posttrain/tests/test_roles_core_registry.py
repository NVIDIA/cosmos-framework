# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: the role-body builder registry (roles/core/registry.py).

Bodies are constructed by heterogeneous builders — `build_vllm_rollout(cfg, packer=...)`,
`LLMTrainer(model, opt, objective, ...)` — so the registry stores BUILDERS keyed by impl_key, not
classes behind one uniform constructor. Registering classes would force every existing body to grow
a `from_config`, which is churn with no payoff.

Duplicate registration raises, matching `control/capability.py::register_capability`, so the key
namespace stays shared with the capability descriptors `bind_check` reads. `allow_override=True` is
the explicit escape hatch an out-of-tree body (AlpaGym) needs to replace a built-in under one key.
"""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.roles.core import registry


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Give each test a clean registry; the tables are module-level dicts.

    It CLEARS as well as restores, and the difference is not cosmetic. Saving-and-restoring
    alone hands each test whatever the ambient process already registered, so
    ``registered("env") == ["apple", "zebra"]`` below held only while that table happened to be
    empty -- i.e. this file was order-dependent on which other module had been imported first,
    and the first real registrant would turn it red for a reason that has nothing to do with
    the registry. `roles/env/sim_gateway.py` registering under `SIM_IMPL_KEY` is that
    registrant. Clearing makes the exact assertion mean what it says.
    """
    saved = {kind: dict(table) for kind, table in registry._TABLES.items()}
    for table in registry._TABLES.values():
        table.clear()
    yield
    for kind, table in saved.items():
        registry._TABLES[kind].clear()
        registry._TABLES[kind].update(table)


@pytest.mark.L0
@pytest.mark.CPU
def test_registered_builder_is_called_with_the_config():
    seen = {}

    @registry.register_rollout("toy")
    def build(config):
        seen["config"] = config
        return "body"

    assert registry.build_rollout("toy", {"a": 1}) == "body"
    assert seen["config"] == {"a": 1}


@pytest.mark.L0
@pytest.mark.CPU
def test_decorator_returns_the_builder_unchanged():
    """The decorator must not wrap: a builder stays directly callable and importable by name."""

    def build(config):
        return 1

    decorated = registry.register_rollout("plain")(build)
    assert decorated is build


@pytest.mark.L0
@pytest.mark.CPU
def test_duplicate_registration_raises():
    @registry.register_rollout("dup")
    def first(config):
        return 1

    with pytest.raises(ValueError, match="already registered"):

        @registry.register_rollout("dup")
        def second(config):
            return 2


@pytest.mark.L0
@pytest.mark.CPU
def test_allow_override_replaces_the_builder():
    @registry.register_rollout("over")
    def first(config):
        return 1

    @registry.register_rollout("over", allow_override=True)
    def second(config):
        return 2

    assert registry.build_rollout("over", None) == 2


@pytest.mark.L0
@pytest.mark.CPU
def test_unknown_key_raises_and_names_the_registered_ones():
    """The error has to name what IS registered: a typo'd impl_key is the common case, and a bare
    KeyError leaves the caller guessing which spelling the entry expected."""

    @registry.register_trainer("known")
    def build(config):
        return 1

    with pytest.raises(KeyError, match="known"):
        registry.build_trainer("missing", None)


@pytest.mark.L0
@pytest.mark.CPU
def test_kinds_have_separate_namespaces():
    @registry.register_rollout("same")
    def build_r(config):
        return "rollout"

    @registry.register_trainer("same")
    def build_t(config):
        return "trainer"

    @registry.register_env("same")
    def build_e(config):
        return "env"

    assert registry.build_rollout("same", None) == "rollout"
    assert registry.build_trainer("same", None) == "trainer"
    assert registry.build_env("same", None) == "env"


@pytest.mark.L0
@pytest.mark.CPU
def test_registered_lists_keys_sorted():
    @registry.register_env("zebra")
    def build_z(config):
        return 1

    @registry.register_env("apple")
    def build_a(config):
        return 2

    assert registry.registered("env") == ["apple", "zebra"]
