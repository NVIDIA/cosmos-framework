# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Implementation-owned capability descriptors + the GPU-less registry `bind_check` reads.

A `CapabilityDescriptor` declares what an implementation produces/consumes and on which planes; it
is registered under the impl_key, so a manifest can select an impl but never fabricate a capability.
Torch/ray-free: field kinds are string tags, and the preference kind derives from the real `Preference`
type so descriptors cannot drift from the type system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cosmos_framework.posttrain.schema import Preference, SFTTarget, TeacherSignal

if TYPE_CHECKING:
    from cosmos_framework.posttrain.control.manifest import Plane


@dataclass(frozen=True)
class FieldSpec:
    """A required/produced field: a name plus a string kind tag (e.g. 'preference').
    Deliberately a string, not an enum — the Record combine-semantics FieldKind is a separate layer."""

    name: str
    kind: str
    required: bool = True
    # Appended to the error when this field is missing: the check sees roles, not the config that
    # selected them, so naming the key to change is the descriptor's job.
    remedy: str = ""


# Descriptor field kinds derive from the real supervision types — one source of truth, no free-authored tags.
SUPERVISION_KINDS: dict[type, str] = {
    Preference: "preference",
    TeacherSignal: "teacher_signal",
    SFTTarget: "sft_conversation",
}


@dataclass(frozen=True)
class CapabilityDescriptor:
    produces: tuple[FieldSpec, ...] = ()
    consumes: tuple[FieldSpec, ...] = ()
    planes: frozenset[Plane] = frozenset()


_REGISTRY: dict[str, CapabilityDescriptor] = {}


def register_capability(impl_key: str, descriptor: CapabilityDescriptor) -> None:
    """Register a `CapabilityDescriptor` under `impl_key`."""
    if impl_key in _REGISTRY:
        raise ValueError(f"capability {impl_key!r} already registered")
    _REGISTRY[impl_key] = descriptor


def get_capability(impl_key: str) -> CapabilityDescriptor:
    """Resolve the descriptor for `impl_key`; unknown key raises `KeyError`."""
    if impl_key not in _REGISTRY:
        raise KeyError(f"unknown impl_key {impl_key!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[impl_key]
