# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Role-body builder registry: `impl_key -> builder(config) -> body`.

Bodies are constructed by heterogeneous builders — `build_vllm_rollout(cfg, packer=...)`,
`LLMTrainer(model, opt, objective, ...)` — so what is registered is the BUILDER, not the class. A
class registry constructed through one uniform signature would force every existing body to grow a
`from_config`, which is churn with no payoff; entrypoints already own `build_rollout(config)` /
`build_trainer(config, rl)` helpers of exactly this shape.

The key namespace is shared with `control/capability.py`: a body registered here under `impl_key`
should register its `CapabilityDescriptor` under the SAME key, so `bind_check` works without a
second naming scheme. The one body registered so far, `"alpagym_rollout"`, declares none, because
its entry runs no `bind_check` by decision rather than by omission (see the note above
`ROLE_SPECS` in `entrypoints/alpagym_clrl.py`). So the shared namespace is a convention this
registry keeps open, not one it enforces — the first body that wants AOT validation is what makes
it load-bearing.

Duplicate registration raises, matching `register_capability`. `allow_override=True` is the explicit
escape hatch for replacing a built-in under one key — what an out-of-tree body (AlpaGym) needs.

SCOPE, WIDER THAN TODAY'S USE, DELIBERATELY. `rollout` has one registrant (AlpaGym's) and one call
site; `env` and `trainer` have none yet. That is below the bar this repo usually sets for an
abstraction, and DESIGN §11.3 still lists component registration as deferred — so read the three
kinds as the declared seam for out-of-tree bodies, kept knowingly rather than left by oversight.
Collapsing it to `rollout` alone would make the first env or trainer plugged in from outside the
tree re-open the same design question.

NOT this: selecting objectives or callbacks by name from config. Those are separate surfaces with
their own open TODOs (`algorithm/__init__.py`, `callbacks/dpo.py`); this registry does
not address them.
"""

from __future__ import annotations

from typing import Any, Callable

Builder = Callable[[Any], Any]

_TABLES: dict[str, dict[str, Builder]] = {"env": {}, "rollout": {}, "trainer": {}}


def _register(kind: str, key: str, allow_override: bool) -> Callable[[Builder], Builder]:
    """Return a decorator registering a builder for `key` under `kind`.

    Args:
        kind: role kind — one of `_TABLES`' keys.
        key: the impl_key this builder answers to.
        allow_override: replace an existing registration instead of raising.

    Returns:
        A decorator that records the builder and returns it UNCHANGED, so the function stays
        directly callable and importable by name.
    """
    table = _TABLES[kind]

    def decorate(builder: Builder) -> Builder:
        existing = table.get(key)
        if existing is not None and not allow_override:
            raise ValueError(
                f"{kind} builder for {key!r} already registered ({existing.__name__}); "
                f"pass allow_override=True to replace it"
            )
        table[key] = builder
        return builder

    return decorate


def register_env(key: str, *, allow_override: bool = False) -> Callable[[Builder], Builder]:
    """Register an env-body builder under `key`."""
    return _register("env", key, allow_override)


def register_rollout(key: str, *, allow_override: bool = False) -> Callable[[Builder], Builder]:
    """Register a rollout-body builder under `key`."""
    return _register("rollout", key, allow_override)


def register_trainer(key: str, *, allow_override: bool = False) -> Callable[[Builder], Builder]:
    """Register a trainer-body builder under `key`."""
    return _register("trainer", key, allow_override)


def _build(kind: str, key: str, config: Any) -> Any:
    """Construct the body registered for `key`.

    Raises:
        KeyError: `key` is not registered. The message names every registered key: a mistyped
            impl_key is the common failure, and a bare KeyError leaves the caller guessing which
            spelling the entry expected.
    """
    table = _TABLES[kind]
    if key not in table:
        raise KeyError(f"unknown {kind} impl_key {key!r}; registered: {sorted(table)}")
    return table[key](config)


def build_env(key: str, config: Any) -> Any:
    """Construct the env body registered under `key`."""
    return _build("env", key, config)


def build_rollout(key: str, config: Any) -> Any:
    """Construct the rollout body registered under `key`."""
    return _build("rollout", key, config)


def build_trainer(key: str, config: Any) -> Any:
    """Construct the trainer body registered under `key`."""
    return _build("trainer", key, config)


def registered(kind: str) -> list[str]:
    """Sorted impl_keys registered for `kind`."""
    return sorted(_TABLES[kind])
