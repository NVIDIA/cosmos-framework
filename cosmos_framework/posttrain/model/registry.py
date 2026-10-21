# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Model registry (DESIGN §2.7 — explicit construction, no lazy-instantiate).

Maps a config `name` to a builder that returns a `FlowMatchingPolicy`. Config
selects a model by name (`model.name = "cosmos3" | "hf_sdxl"`); the builder
constructs it explicitly from the typed config — there is no `_target_` / Hydra
`instantiate` indirection, matching the framework's "control is explicit" stance.

Builders register on import of their module (e.g. `model.cosmos3`), so importing this
package wires the available backends. Keeping the registry tiny and string-keyed lets a
new backend drop in without touching callers. (The toy model tests use is a direct-import
fixture, not a registered backend — see `tests/helpers/toy_model.py`.)
"""

from __future__ import annotations

from typing import Any, Callable

from cosmos_framework.posttrain.model.protocol import FlowMatchingPolicy

_BUILDERS: dict[str, Callable[[Any], FlowMatchingPolicy]] = {}


def register_model(name: str) -> Callable[[Callable[[Any], FlowMatchingPolicy]], Callable[[Any], FlowMatchingPolicy]]:
    """Decorator: register a `build(cfg) -> FlowMatchingPolicy` under `name`."""

    def deco(fn: Callable[[Any], FlowMatchingPolicy]) -> Callable[[Any], FlowMatchingPolicy]:
        if name in _BUILDERS:
            raise ValueError(f"model {name!r} already registered")
        _BUILDERS[name] = fn
        return fn

    return deco


def build_model(cfg: Any) -> FlowMatchingPolicy:
    """Construct the model named `cfg.name` from the typed config (explicit)."""
    name = getattr(cfg, "name", None)
    if name not in _BUILDERS:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(_BUILDERS)}")
    return _BUILDERS[name](cfg)
