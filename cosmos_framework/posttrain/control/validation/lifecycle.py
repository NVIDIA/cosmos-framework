# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared train-time model-validation lifecycle.

This module is deliberately control-plane-only and dependency-light. It decides *when* a pinned
model version must be validated and persists whether that phase was interrupted; algorithm
entrypoints still decide *how* to evaluate (objective or rollout/reward).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from cosmos_framework.posttrain.config.rl import ValidationScheduleConfig
from cosmos_framework.posttrain.schema.checkpoint import CheckpointComponentPlan
from cosmos_framework.posttrain.schema.metrics import validate_public_metric_keys


class ValidationTrigger(str, Enum):
    """Why one logical validation run is due."""

    BEFORE_TRAINING = "before_training"
    PERIODIC = "periodic"
    FINAL = "final"
    PENDING = "pending"


@dataclass(frozen=True)
class ValidationSchedule:
    """Pure trigger policy over public committed optimizer steps."""

    enabled: bool
    before_training: bool
    every_steps: int
    save_checkpoint: bool = False
    checkpoint_components: CheckpointComponentPlan = field(
        default_factory=lambda: CheckpointComponentPlan.from_value(None)
    )

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(f"validation enabled must be a bool, got {self.enabled!r}")
        if not isinstance(self.before_training, bool):
            raise TypeError(f"validation before_training must be a bool, got {self.before_training!r}")
        if isinstance(self.every_steps, bool) or not isinstance(self.every_steps, int):
            raise TypeError(f"validation every_steps must be an integer, got {self.every_steps!r}")
        if self.every_steps < 0:
            raise ValueError(f"validation every_steps must be >= 0, got {self.every_steps}")
        if not isinstance(self.save_checkpoint, bool):
            raise TypeError(f"validation save_checkpoint must be a bool, got {self.save_checkpoint!r}")
        if not isinstance(self.checkpoint_components, CheckpointComponentPlan):
            raise TypeError(
                "validation checkpoint_components must be a CheckpointComponentPlan, got "
                f"{type(self.checkpoint_components).__name__}"
            )

    @classmethod
    def from_config(cls, config: ValidationScheduleConfig) -> ValidationSchedule:
        """Build the immutable trigger policy from the typed config."""
        return cls(
            enabled=bool(config.enabled),
            before_training=bool(config.before_training),
            every_steps=int(config.every_steps),
            save_checkpoint=bool(getattr(config, "save_checkpoint", False)),
            checkpoint_components=CheckpointComponentPlan.from_value(getattr(config, "checkpoint_components", None)),
        )

    def reasons(
        self,
        model_step: int,
        total_steps: int,
        *,
        fresh_start: bool = False,
        after_update: bool = False,
        pending_step: int | None = None,
    ) -> frozenset[ValidationTrigger]:
        """Return the deduplicated reasons to validate ``model_step`` now.

        A restored checkpoint is neither a fresh start nor a newly committed update, so it yields no
        trigger unless it explicitly carries ``pending_step``. This is the distinction between exact
        resume and loading weights into a new run.
        """
        if isinstance(model_step, bool) or not isinstance(model_step, int) or model_step < 0:
            raise ValueError(f"model_step must be >= 0, got {model_step}")
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 0:
            raise ValueError(f"total_steps must be >= 0, got {total_steps}")
        if model_step > total_steps:
            raise ValueError(f"model_step {model_step} exceeds total_steps {total_steps}")
        if not self.enabled:
            return frozenset()
        if fresh_start and after_update:
            raise ValueError("fresh_start and after_update are mutually exclusive validation phases")
        if pending_step is not None:
            if pending_step != model_step:
                raise ValueError(
                    f"pending validation step {pending_step} does not match restored model step {model_step}"
                )
            return frozenset({ValidationTrigger.PENDING})
        if fresh_start:
            if model_step != 0:
                raise ValueError(f"fresh-start validation requires model_step=0, got {model_step}")
            reasons: set[ValidationTrigger] = set()
            if self.before_training:
                reasons.add(ValidationTrigger.BEFORE_TRAINING)
            if total_steps == 0:
                reasons.add(ValidationTrigger.FINAL)
            return frozenset(reasons)
        if not after_update:
            return frozenset()
        if model_step == 0:
            raise ValueError("after-update validation requires model_step > 0")

        reasons: set[ValidationTrigger] = set()
        if self.every_steps > 0 and model_step % self.every_steps == 0:
            reasons.add(ValidationTrigger.PERIODIC)
        if model_step == total_steps:
            reasons.add(ValidationTrigger.FINAL)
        return frozenset(reasons)


@dataclass
class ValidationProgress:
    """Checkpointable identity of an interrupted validation phase.

    The finite validation cursor itself is intentionally not checkpointed. Resuming a pending phase
    restarts that pass deterministically. Completion is made durable before the control plane makes
    one best-effort external reporting attempt.
    """

    pending_step: int | None = None
    pending_reasons: frozenset[ValidationTrigger] = field(default_factory=frozenset)
    pending_metrics: dict[str, int | float | bool | str] = field(default_factory=dict)

    @staticmethod
    def _validated_metrics(
        metrics: dict[str, int | float | bool | str],
    ) -> dict[str, int | float | bool | str]:
        validate_public_metric_keys(metrics)
        validated: dict[str, int | float | bool | str] = {}
        for key, value in metrics.items():
            if not isinstance(value, (int, float, bool, str)):
                raise ValueError(f"pending metric {key!r} is not a primitive scalar: {value!r}")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"pending metric {key!r} must be finite, got {value!r}")
            validated[key] = value
        return validated

    def begin(
        self,
        model_step: int,
        reasons: frozenset[ValidationTrigger],
        metrics: dict[str, int | float | bool | str] | None = None,
    ) -> None:
        """Mark one validation phase pending before its first data read."""
        if isinstance(model_step, bool) or not isinstance(model_step, int) or model_step < 0:
            raise ValueError(f"model_step must be >= 0, got {model_step}")
        if not reasons:
            raise ValueError("a validation phase requires at least one trigger reason")
        if not isinstance(reasons, frozenset) or any(not isinstance(reason, ValidationTrigger) for reason in reasons):
            raise TypeError(f"validation reasons must be a frozenset of ValidationTrigger values, got {reasons!r}")
        if self.pending_step is not None:
            raise RuntimeError(f"validation step {self.pending_step} is already pending")
        validated_metrics = self._validated_metrics(dict(metrics or {}))
        self.pending_step = model_step
        self.pending_reasons = reasons
        self.pending_metrics = validated_metrics

    def complete(self, model_step: int) -> None:
        """Clear the pending marker after successful reduction, before external reporting."""
        if self.pending_step != model_step:
            raise RuntimeError(f"cannot complete validation step {model_step}; pending step is {self.pending_step}")
        self.pending_step = None
        self.pending_reasons = frozenset()
        self.pending_metrics = {}

    def state_dict(self) -> dict[str, Any]:
        """Return the dependency-free checkpoint representation."""
        return {
            "pending_step": self.pending_step,
            "pending_reasons": sorted(reason.value for reason in self.pending_reasons),
            "pending_metrics": dict(self.pending_metrics),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore and validate a checkpointed pending phase."""
        unknown = set(state) - {"pending_step", "pending_reasons", "pending_metrics"}
        if unknown:
            raise ValueError(f"unknown validation progress fields: {sorted(unknown)}")
        pending_step = state.get("pending_step")
        if pending_step is not None and (
            isinstance(pending_step, bool) or not isinstance(pending_step, int) or pending_step < 0
        ):
            raise ValueError(f"invalid pending validation step: {pending_step!r}")
        raw_reasons = state.get("pending_reasons", [])
        if not isinstance(raw_reasons, list) or any(not isinstance(reason, str) for reason in raw_reasons):
            raise ValueError("pending_reasons must be a list of strings")
        reasons = frozenset(ValidationTrigger(reason) for reason in raw_reasons)
        raw_metrics = state.get("pending_metrics", {})
        if not isinstance(raw_metrics, dict):
            raise ValueError("pending_metrics must be a dictionary")
        metrics = self._validated_metrics(raw_metrics)
        if (pending_step is None) != (not reasons):
            raise ValueError("pending_step and pending_reasons must either both be present or both be empty")
        if pending_step is None and metrics:
            raise ValueError("pending_metrics require a pending validation step")
        self.pending_step = pending_step
        self.pending_reasons = reasons
        self.pending_metrics = metrics


@dataclass(frozen=True)
class ValidationReport:
    """Small immutable control-plane result produced by either evaluator family."""

    model_step: int
    sample_count: int
    metrics: dict[str, int | float]
    source_fingerprint: str
    trigger_reasons: frozenset[ValidationTrigger]
    duration_s: float

    def __post_init__(self) -> None:
        if isinstance(self.model_step, bool) or not isinstance(self.model_step, int) or self.model_step < 0:
            raise ValueError(f"model_step must be >= 0, got {self.model_step}")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count < 1:
            raise ValueError(f"sample_count must be >= 1, got {self.sample_count}")
        if not self.source_fingerprint:
            raise ValueError("source_fingerprint must be non-empty")
        if not self.trigger_reasons:
            raise ValueError("trigger_reasons must be non-empty")
        if not isinstance(self.trigger_reasons, frozenset) or any(
            not isinstance(reason, ValidationTrigger) for reason in self.trigger_reasons
        ):
            raise TypeError(
                f"trigger_reasons must be a frozenset of ValidationTrigger values, got {self.trigger_reasons!r}"
            )
        if not self.metrics:
            raise ValueError("validation metrics must be non-empty")
        validate_public_metric_keys(self.metrics)
        for key, value in self.metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"validation metric {key!r} must be a non-bool number, got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"validation metric {key!r} must be finite, got {value!r}")
        if isinstance(self.duration_s, bool) or not isinstance(self.duration_s, (int, float)):
            raise TypeError(f"duration_s must be a real number, got {self.duration_s!r}")
        if not math.isfinite(float(self.duration_s)) or self.duration_s < 0.0:
            raise ValueError(f"duration_s must be >= 0, got {self.duration_s}")
