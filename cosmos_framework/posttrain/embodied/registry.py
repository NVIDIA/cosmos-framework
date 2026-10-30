# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Name -> factory. A dict, not a framework.

It closes the gap ``docs/design/DESIGN.md`` names explicitly ("Deferred -- component registration:
there is no by-name registry yet"), and it exists for one mechanical reason: the lazy
``__import__`` is what keeps the policy venv from ever importing ``omnigibson``, and the
Isaac venv from ever importing the native policy stack. A registry that imported its modules
eagerly would defeat the entire import discipline the core is built around.
"""

from __future__ import annotations

from typing import Any, Callable

_SIMS: dict[str, Callable[..., Any]] = {}
_ROLLOUTS: dict[str, Callable[..., Any]] = {}
_SINKS: dict[str, Callable[..., Any]] = {}
_SOURCES: dict[str, Callable[..., Any]] = {}


def _register(table: dict, kind: str, name: str, factory: Callable[..., Any]) -> Callable[..., Any]:
    prior = table.get(name)
    if prior is not None and prior is not factory:
        # Silent replacement would make "which sim am I running" answerable only at
        # runtime, and a typo'd name in a second module would shadow the real one.
        raise ValueError(f"{kind} {name!r} already registered as {prior!r}")
    table[name] = factory
    return factory


def register_sim(name: str, factory: Callable[..., Any]) -> Callable[..., Any]:
    return _register(_SIMS, "sim", name, factory)


def register_rollout(name: str, factory: Callable[..., Any]) -> Callable[..., Any]:
    """Register something that satisfies :class:`rollout.Rollout`.

    ROLLOUT, not "policy", and the subpackage is ``rollouts/`` for the same reason. Every
    other plugin directory here is the plural of the module holding its protocol --
    ``sim.py``/``sims``, ``sink.py``/``sinks``, ``source.py``/``sources`` -- and this one is
    ``rollout.py``. It was ``policies/``, which was the only name in the set that did not
    match its own contract: the protocol is ``Rollout``, ``describe()`` returns
    ``RolloutInfo``, and ``driver.run_unit`` takes the parameter as ``rollout``.

    Distinct from ``roles/rollout/``, which is the FRAMEWORK's generation role. That one hosts
    a model and answers ``act``; this one is whatever the simulator calls to get actions, and
    under the gateway it is a proxy for the other.
    """
    return _register(_ROLLOUTS, "rollout", name, factory)


def register_sink(name: str, factory: Callable[..., Any]) -> Callable[..., Any]:
    return _register(_SINKS, "sink", name, factory)


def register_source(name: str, factory: Callable[..., Any]) -> Callable[..., Any]:
    return _register(_SOURCES, "source", name, factory)


def _build(table: dict, kind: str, subpkg: str, name: str, kw: dict) -> Any:
    if name not in table:
        try:
            __import__(f"{__package__}.{subpkg}.{name}")
        except ImportError as exc:
            raise ImportError(
                f"no {kind} named {name!r}: importing {__package__}.{subpkg}.{name} failed "
                f"({exc}). Registered: {sorted(table) or '(none)'}."
            ) from exc
    if name not in table:
        raise KeyError(
            f"module {__package__}.{subpkg}.{name} imported but did not register a {kind} "
            f"under {name!r}. Registered: {sorted(table) or '(none)'}."
        )
    return table[name](**kw)


def build_sim(name: str, **kw: Any) -> Any:
    return _build(_SIMS, "sim", "sims", name, kw)


def build_rollout(name: str, **kw: Any) -> Any:
    return _build(_ROLLOUTS, "rollout", "rollouts", name, kw)


def build_sink(name: str, **kw: Any) -> Any:
    return _build(_SINKS, "sink", "sinks", name, kw)


def build_source(name: str, **kw: Any) -> Any:
    return _build(_SOURCES, "source", "sources", name, kw)


def registered() -> dict[str, list[str]]:
    """What is registered *right now* -- i.e. what has already been imported.

    Deliberately not a discovery mechanism. Listing the modules on disk would import them,
    which is the one thing this file exists to avoid.
    """
    return {
        "sims": sorted(_SIMS),
        "rollouts": sorted(_ROLLOUTS),
        "sinks": sorted(_SINKS),
        "sources": sorted(_SOURCES),
    }
