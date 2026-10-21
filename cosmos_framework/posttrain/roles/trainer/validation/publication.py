# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Durable checkpoint publication after held-out validation."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from cosmos_framework.posttrain.roles.trainer.validation.state import ValidationState
from cosmos_framework.posttrain.utils.dist import rank_identical, run_rank_converged

if TYPE_CHECKING:
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager


class PublicationJournal:
    """Durably publish validation completion without repeating evaluation."""

    state: ValidationState
    pending_step: int | None
    record: dict[str, Any] | None

    def __init__(self, state: ValidationState) -> None:
        self.state = state
        self.pending_step: int | None = None
        self.record: dict[str, Any] | None = None

    def assert_can_begin(self) -> None:
        if self.pending_step is not None:
            raise RuntimeError(f"validation step {self.pending_step} is awaiting checkpoint publication")

    def complete(self, model_step: int) -> None:
        if self.pending_step is not None:
            raise RuntimeError(f"validation step {self.pending_step} is already awaiting publication")
        self.state.progress.complete(model_step)
        self.pending_step = model_step
        self.record = None

    def checkpoint_record(self) -> dict[str, Any] | None:
        return copy.deepcopy(self.record)

    @staticmethod
    def validate_record(record: dict[str, Any], iteration: int) -> dict[str, Any]:
        from cosmos_framework.posttrain.checkpoint.manifest import validate_publication_record

        return copy.deepcopy(validate_publication_record(record, iteration))

    def restore(self, record: Any, *, iteration: int) -> None:
        if record is None:
            self.pending_step = None
            self.record = None
            return
        if not isinstance(record, dict):
            raise RuntimeError("validation publication record must be a dictionary")
        validated = self.validate_record(record, iteration)
        if self.state.progress.pending_step is not None:
            raise RuntimeError("validation evaluation and checkpoint publication cannot both be pending")
        if not self.state.enabled:
            raise RuntimeError("disabled validation identity cannot await checkpoint publication")
        self.pending_step = validated["step"]
        self.record = validated

    def validate_checkpoint_state(self) -> None:
        if self.state.progress.pending_step is not None and self.pending_step is not None:
            raise RuntimeError("validation evaluation and checkpoint publication cannot both be pending")
        if not self.state.enabled and self.state.progress.pending_step is not None:
            raise RuntimeError("disabled validation cannot have a pending evaluation")
        if not self.state.enabled and self.pending_step is not None:
            raise RuntimeError("disabled validation cannot await checkpoint publication")
        if (self.pending_step is None) != (self.record is None):
            raise RuntimeError("validation publication step and record must either both be present or both be empty")

    def _build_record(
        self,
        path: str,
        model_step: int,
        *,
        manager: "CheckpointManager",
        publication_identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        storage_identity = manager.publication_identity(path)
        destination = storage_identity.get("destination")
        checkpoint_generation = storage_identity.get("checkpoint_generation")
        if not isinstance(destination, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in destination.items()
        ):
            raise RuntimeError("checkpoint publication destination must contain string fields")
        if not isinstance(checkpoint_generation, str) or not checkpoint_generation:
            raise RuntimeError("checkpoint publication generation must be a non-empty string")
        payload: dict[str, Any] = {
            "protocol": "posttrain.validation-publication.v1",
            "step": model_step,
            "archive_required": bool(manager.upload),
            "destination": destination,
            "checkpoint_generation": checkpoint_generation,
            "validation_identity": self.state.identity,
        }
        overlap = sorted(set(payload) & set(publication_identity))
        if overlap:
            raise ValueError(f"validation publication identity collides with reserved fields: {overlap}")
        payload.update(copy.deepcopy(dict(publication_identity)))
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        publication_id = hashlib.sha256(encoded).hexdigest()
        return {
            "step": model_step,
            "id": publication_id,
            "archive_required": bool(manager.upload),
            "destination": destination,
            "manifest": f".validation-publication-v1-{publication_id}.json",
        }

    def publish(
        self,
        path: str,
        *,
        manager: "CheckpointManager",
        publication_identity: Mapping[str, Any],
        trainer_state: Callable[[], dict[str, Any]],
        on_checkpoint: Callable[[], None],
    ) -> None:
        """Commit completed validation through the selected primary store and optional archive."""
        if not self.state.enabled:
            raise RuntimeError("cannot persist validation completion when model validation is disabled")
        if self.state.progress.pending_step is not None:
            raise RuntimeError(
                f"cannot persist validation completion while step {self.state.progress.pending_step} is pending"
            )
        publication_step = self.pending_step
        if publication_step is None:
            raise RuntimeError("cannot persist validation completion when no publication is pending")

        expected_record = run_rank_converged(
            lambda: self._build_record(
                path,
                publication_step,
                manager=manager,
                publication_identity=publication_identity,
            ),
            "validation publication record preparation",
        )
        expected_record = rank_identical(expected_record, "validation publication record")
        self.record = copy.deepcopy(rank_identical(self.record, "validation publication durable record"))
        previous_record = copy.deepcopy(self.record)
        record_needs_persist = self.record is None
        if self.record is None:
            self.record = expected_record
        elif (
            self.record["archive_required"] != expected_record["archive_required"]
            or self.record["destination"] != expected_record["destination"]
        ):
            rebound_record = run_rank_converged(
                lambda: manager.rebind_copied_publication_record(path, self.record or {}, expected_record),
                "copied validation publication record rebind",
            )
            rebound_record = rank_identical(rebound_record, "copied validation publication record")
            if rebound_record is None:
                raise RuntimeError(
                    "validation publication destination changed across resume: "
                    f"checkpoint={self.record!r}, current={expected_record!r}"
                )
            self.record = rebound_record
            record_needs_persist = True
        publication_record = copy.deepcopy(self.record)
        if record_needs_persist:
            try:
                pending_state = run_rank_converged(trainer_state, "validation publication-journal serialization")
                manager.replace_component(path, "trainer", pending_state)
            except BaseException:  # noqa: BLE001 -- a retry must not assume an uncommitted record is durable
                self.record = previous_record
                raise

        saved_step = self.pending_step
        saved_record = self.record

        def completed_trainer_state() -> dict[str, Any]:
            self.pending_step = None
            self.record = None
            try:
                return trainer_state()
            finally:
                self.pending_step = saved_step
                self.record = saved_record

        completed_state = run_rank_converged(
            completed_trainer_state,
            "validation completed-journal serialization",
        )
        manager.publish_validation(
            path,
            completed_trainer_state=completed_state,
            publication_record=publication_record,
        )

        self.pending_step = None
        self.record = None
        run_rank_converged(on_checkpoint, "validation completion checkpoint callback")
