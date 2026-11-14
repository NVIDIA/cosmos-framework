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
_FLOW_PACKERS: dict[str, Callable] = {}
_FLOW_GRADIENT_WINDOWS: dict[str, Callable] = {}


def register_flow_sft_gradient_window(builder: str):
    """Register the family's optional post-optimizer gradient-policy admission.

    The factory receives the constructed native policy, final optimizer and SFT
    configuration. It installs the shared owned-shard policy; it does not own a
    training loop or change the family's native objective/forward.
    """

    def register(factory):
        if builder in _FLOW_GRADIENT_WINDOWS and _FLOW_GRADIENT_WINDOWS[builder] is not factory:
            raise ValueError(f"a flow SFT gradient-window factory is already registered for {builder!r}")
        _FLOW_GRADIENT_WINDOWS[builder] = factory
        return factory

    return register


def flow_sft_gradient_window_factory(spec: NativePolicyBuildSpec | None):
    """Resolve a declared native factory without importing another family.

    Configuration preflight has already resolved the model source. Import only
    its dependency-light builder module, leaving heavyweight model admission to
    the factory after materialization and final optimizer construction.
    """
    if spec is not None:
        try:
            importlib.import_module(spec.builder.rsplit(".", 1)[0])
        except ImportError as error:
            raise ValueError(
                f"native flow SFT builder {spec.builder!r} has no available gradient-window factory"
            ) from error
        if spec.builder in _FLOW_PACKERS and spec.builder in _FLOW_GRADIENT_WINDOWS:
            return _FLOW_GRADIENT_WINDOWS[spec.builder]
    raise ValueError("owned FP64 gradients require a native flow SFT builder with a registered gradient-window factory")


def register_flow_sft_packer(builder: str):
    """Bind a native flow model's collator beside its existing SFT source.

    The factory receives the native build spec and typed SFT config. Its result
    is callable over trajectories and exposes validation_semantics_identity().
    It owns preprocessing, never an optimizer or a training loop.
    """

    def register(factory):
        if builder in _FLOW_PACKERS and _FLOW_PACKERS[builder] is not factory:
            raise ValueError(f"a flow SFT packer is already registered for {builder!r}")
        _FLOW_PACKERS[builder] = factory
        return factory

    return register


def native_sft_task(spec: NativePolicyBuildSpec) -> str:
    resolve_native_sft_source(spec)
    return "flow_bc" if spec.builder in _FLOW_PACKERS else "conversation"


def build_flow_sft_packer(spec: NativePolicyBuildSpec, config):
    if native_sft_task(spec) != "flow_bc":
        raise ValueError(f"native builder {spec.builder!r} has no supervised-flow SFT packer")
    return _FLOW_PACKERS[spec.builder](spec, config)


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
