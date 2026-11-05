# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Dependency-light checkpoint component selection shared by config and runtime."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, cast

CheckpointComponent = Literal["model", "optimizer", "training_state"]
FULL_CHECKPOINT_COMPONENTS: tuple[CheckpointComponent, ...] = (
    "model",
    "optimizer",
    "training_state",
)
_CHECKPOINT_COMPONENT_SET = frozenset(FULL_CHECKPOINT_COMPONENTS)

EmergencyCheckpointPlan = Literal["full", "run"]
EMERGENCY_CHECKPOINT_PLANS: tuple[EmergencyCheckpointPlan, ...] = ("full", "run")
"""Which component plan a termination-triggered (preemption / timeout) emergency save writes.

``"full"``: the full plan, whatever the run's ``checkpoint_components`` selects — an exact-resume
checkpoint. ``"run"``: the run's own configured plan, i.e. the cadence save the loop would have written
at its next ``save_every`` boundary, in the same location with the same commit marker. See
``RunConfig.emergency_checkpoint_components`` for when a run chooses the latter and what it gives up.
"""


@dataclass(frozen=True)
class CheckpointComponentPlan:
    """Validated logical checkpoint content and its exact-resume capability."""

    selected: frozenset[CheckpointComponent]

    def __post_init__(self) -> None:
        if not isinstance(self.selected, frozenset):
            raise TypeError(f"checkpoint component plan must be a frozenset, got {type(self.selected).__name__}")
        if not self.selected:
            raise ValueError("checkpoint components must select at least one component")
        if any(not isinstance(name, str) for name in self.selected):
            raise TypeError(f"checkpoint component names must be strings, got {self.selected!r}")
        unsupported = sorted(set(self.selected) - _CHECKPOINT_COMPONENT_SET)
        if unsupported:
            raise ValueError(
                f"unsupported checkpoint components {unsupported}; expected a subset of "
                f"{list(FULL_CHECKPOINT_COMPONENTS)}"
            )

    @classmethod
    def from_value(
        cls,
        components: Iterable[str] | CheckpointComponentPlan | None,
    ) -> CheckpointComponentPlan:
        """Normalize a user/runtime component selection in canonical logical names."""
        if isinstance(components, cls):
            return components
        if components is None:
            return cls(frozenset(FULL_CHECKPOINT_COMPONENTS))
        if isinstance(components, str):
            raise TypeError("checkpoint components must be an iterable of component names, not a string")
        names = tuple(components)
        if any(name in names[:index] for index, name in enumerate(names)):
            raise ValueError(f"checkpoint components must not contain duplicates, got {names!r}")
        try:
            selected = frozenset(names)
        except TypeError as error:
            raise TypeError(f"checkpoint component names must be strings, got {names!r}") from error
        return cls(cast(frozenset[CheckpointComponent], selected))

    @property
    def names(self) -> tuple[CheckpointComponent, ...]:
        """Return selected names in stable config and manifest order."""
        return tuple(name for name in FULL_CHECKPOINT_COMPONENTS if name in self.selected)

    @property
    def is_full(self) -> bool:
        """Whether this plan contains every state family required for exact resume."""
        return self.selected == _CHECKPOINT_COMPONENT_SET

    def includes(self, component: CheckpointComponent) -> bool:
        """Whether one logical state family is selected."""
        return component in self.selected
