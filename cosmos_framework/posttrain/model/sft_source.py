# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Which native policy builders seat offline SFT, declared by the model family that owns the builder.

Previously ``entrypoints/sft.py`` carried a builder-string and model-type allowlist
(``NATIVE_SFT_BUILDER``, ``NATIVE_SFT_MODEL_TYPES``), so another native SFT-capable family would have edited the
entrypoint. The declaration now lives beside the builder: a model module registers a resolver for its builder's dotted
path with :func:`register_native_sft_source`, and the entrypoint imports the builder's module (the registration runs at
import) and asks :func:`resolve_native_sft_source`. A builder that registers nothing is refused by name. This module is
dependency-light on purpose (no torch, no transformers): the preflight imports it without pulling a model family.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cosmos_framework.posttrain.config.model import NativePolicyBuildSpec

SFTSourceResolver = Callable[["NativePolicyBuildSpec"], tuple[str, bool]]
"""``spec -> (name_or_path, trust_remote_code)``: the one model/tokenizer artifact the SFT packer and the backbone share,
after the family's own checks (model type, sidecars); no model or processor is constructed."""

_RESOLVERS: dict[str, SFTSourceResolver] = {}


def register_native_sft_source(builder: str) -> Callable[[SFTSourceResolver], SFTSourceResolver]:
    """Declare that ``builder`` (an absolute dotted callable) seats SFT; the decorated function resolves its source."""
    if not isinstance(builder, str) or "." not in builder:
        raise ValueError(f"register_native_sft_source needs an absolute dotted builder, got {builder!r}")

    def deco(fn: SFTSourceResolver) -> SFTSourceResolver:
        registered = _RESOLVERS.get(builder)
        if registered is not None and registered is not fn:
            raise ValueError(f"an SFT source is already registered for the native builder {builder!r}")
        _RESOLVERS[builder] = fn
        return fn

    return deco


def registered_native_sft_builders() -> tuple[str, ...]:
    """The builders whose modules have registered an SFT source so far (sorted)."""
    return tuple(sorted(_RESOLVERS))


def resolve_native_sft_source(spec: NativePolicyBuildSpec) -> tuple[str, bool]:
    """Import the builder's module so its registration runs, then resolve; refuse an undeclared builder by name."""
    module_name = spec.builder.rsplit(".", 1)[0]
    try:
        importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(
            f"SFT: the native builder {spec.builder!r} declares no SFT source (its module {module_name!r} could not "
            f"be imported: {error}); registered builders: {list(registered_native_sft_builders())}"
        ) from error
    resolver = _RESOLVERS.get(spec.builder)
    if resolver is None:
        raise ValueError(
            f"SFT: the native builder {spec.builder!r} declares no SFT source; register one beside the builder with "
            f"model/sft_source.py::register_native_sft_source; registered builders: {list(registered_native_sft_builders())}"
        )
    name_or_path, trust_remote_code = resolver(spec)
    if not isinstance(name_or_path, str) or not name_or_path:
        raise ValueError(f"SFT: the source resolver of {spec.builder!r} returned no model/tokenizer artifact")
    return name_or_path, bool(trust_remote_code)
