# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Checkpointable validation identity and pending state shared by trainers."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

from cosmos_framework.posttrain.control.validation.lifecycle import ValidationProgress, ValidationTrigger
from cosmos_framework.posttrain.utils.identity import json_identity_equal


def _copy_identity(identity: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(identity)


def _exact_identity_equal(first: Any, second: Any) -> bool:
    return json_identity_equal(first, second, context="exact-resume identity")


class ValidationState:
    """Bind one validation identity to its durable pending phase."""

    _normalizer: Callable[[dict[str, Any]], dict[str, Any]]
    _identity_equal: Callable[[Any, Any], bool]
    identity: dict[str, Any]
    progress: ValidationProgress
    configured: bool
    restored_state: dict[str, Any] | None

    def __init__(
        self,
        identity: dict[str, Any] | None = None,
        *,
        normalizer: Callable[[dict[str, Any]], dict[str, Any]] = _copy_identity,
        identity_equal: Callable[[Any, Any], bool] = _exact_identity_equal,
    ) -> None:
        self._normalizer = normalizer
        self._identity_equal = identity_equal
        self.identity = self._normalizer(identity) if identity is not None else {"enabled": False}
        self.progress = ValidationProgress()
        self.configured = identity is not None
        self.restored_state = None

    @property
    def enabled(self) -> bool:
        return self.configured and bool(self.identity["enabled"])

    def defer_restore(self, state: Any) -> None:
        if state is not None and not isinstance(state, dict):
            raise RuntimeError("trainer checkpoint validation state must be a dictionary")
        self.restored_state = copy.deepcopy(state)

    def configure(
        self,
        identity: dict[str, Any],
        *,
        resumed_from_checkpoint: bool,
        iteration: int,
        extra_fields_by_version: Mapping[int, frozenset[str]] | None = None,
    ) -> dict[str, Any] | None:
        """Bind a deferred live identity and restore its compatible pending phase."""
        normalized = self._normalizer(identity)
        if self.configured:
            if not self._identity_equal(normalized, self.identity):
                raise RuntimeError("validation identity cannot change after configuration")
            return None
        restored = self.restored_state
        if restored is None and normalized["enabled"] and resumed_from_checkpoint:
            raise RuntimeError(
                "checkpoint predates the validation identity journal; exact resume cannot enable "
                "validation on an existing run (start a new run from model weights instead)"
            )
        restored_progress = ValidationProgress()
        restored_extra: dict[str, Any] = {}
        if restored is not None:
            fields_by_version = extra_fields_by_version or {1: frozenset()}
            version = restored.get("version")
            extra_fields = fields_by_version.get(version)
            if extra_fields is None or set(restored) != {"version", "identity", "progress", *extra_fields}:
                raise RuntimeError("trainer checkpoint validation state has an unsupported schema")
            restored_identity = self._normalizer(restored["identity"])
            if not self._identity_equal(restored_identity, normalized):
                differing = sorted(
                    key
                    for key in set(normalized) | set(restored_identity)
                    if not self._identity_equal(normalized.get(key), restored_identity.get(key))
                )
                raise RuntimeError(f"trainer checkpoint validation identity differs for {differing}")
            progress = restored["progress"]
            if not isinstance(progress, dict):
                raise RuntimeError("trainer checkpoint validation progress must be a dictionary")
            restored_progress.load_state_dict(progress)
            if not normalized["enabled"] and restored_progress.pending_step is not None:
                raise RuntimeError("disabled validation identity cannot carry a pending validation phase")
            restored_extra = {field: copy.deepcopy(restored[field]) for field in extra_fields}
        if restored_progress.pending_step is not None and restored_progress.pending_step != iteration:
            raise RuntimeError(
                f"pending validation step {restored_progress.pending_step} does not match "
                f"restored trainer iteration {iteration}"
            )
        self.identity = normalized
        self.progress = restored_progress
        self.configured = True
        self.restored_state = None
        return restored_extra

    def require_enabled(self, message: str) -> None:
        if not self.enabled:
            raise RuntimeError(message)

    def begin(
        self,
        model_step: int,
        trigger_reasons: frozenset[ValidationTrigger],
        *,
        metrics: dict[str, int | float | bool | str] | None = None,
    ) -> None:
        if self.progress.pending_step is None:
            self.progress.begin(model_step, trigger_reasons, metrics=metrics)
            return
        if self.progress.pending_step != model_step:
            raise RuntimeError(
                f"validation step {self.progress.pending_step} is already pending; cannot begin step {model_step}"
            )
        if metrics is not None and self.progress.pending_metrics != metrics:
            raise RuntimeError(f"validation step {model_step} is already pending with a different metric frame")

    def state_dict(
        self,
        *,
        version: int,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = {
            "version": version,
            "identity": copy.deepcopy(self.identity),
            "progress": self.progress.state_dict(),
        }
        state.update(copy.deepcopy(dict(extra or {})))
        return state

    def load_state_dict(
        self,
        state: dict[str, Any],
        *,
        extra_fields_by_version: Mapping[int, frozenset[str]],
    ) -> dict[str, Any]:
        version = state.get("version")
        extra_fields = extra_fields_by_version.get(version)
        if extra_fields is None or set(state) != {"version", "identity", "progress", *extra_fields}:
            raise RuntimeError("trainer checkpoint validation state has an unsupported schema")
        restored_identity = state.get("identity")
        if not self._identity_equal(restored_identity, self.identity):
            raise RuntimeError(
                f"trainer checkpoint validation identity={restored_identity!r}, current run expects {self.identity!r}"
            )
        progress_state = state.get("progress")
        if not isinstance(progress_state, dict):
            raise RuntimeError("trainer checkpoint validation progress must be a dictionary")
        self.progress.load_state_dict(progress_state)
        if self.progress.pending_step is not None and not self.enabled:
            raise RuntimeError("disabled validation identity cannot have pending evaluation")
        return {field: copy.deepcopy(state[field]) for field in extra_fields}
