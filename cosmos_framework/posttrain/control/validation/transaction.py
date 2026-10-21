# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Durable validation transaction shared by training loops."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cosmos_framework.posttrain.control.checkpointing import CheckpointCoordinator
from cosmos_framework.posttrain.control.validation.lifecycle import ValidationReport, ValidationTrigger
from cosmos_framework.posttrain.schema.checkpoint import CheckpointComponentPlan
from cosmos_framework.posttrain.schema.metrics import MetricScalar
from cosmos_framework.posttrain.utils.entry import resolve_all, resolve_identical
from cosmos_framework.posttrain.utils.report import Reporter, emit_metric_frame

ValidationEvaluator = Callable[[int, frozenset[ValidationTrigger]], ValidationReport]


def evaluate_policy_validation(
    policy: Any,
    model_step: int,
    reasons: frozenset[ValidationTrigger],
) -> ValidationReport:
    """Resolve a trainer-owned objective validation across every policy cell."""
    report = resolve_identical(
        policy.validate(model_step, reasons),
        operation="policy.validate",
    )
    if not isinstance(report, ValidationReport):
        raise TypeError(f"policy validation returned {type(report).__name__}, expected ValidationReport")
    return report


@dataclass
class ValidationTransaction:
    """Journal, checkpoint, evaluate, commit, then report one validation phase."""

    policy: Any
    checkpoints: CheckpointCoordinator
    reporters: list[Reporter]
    failed_reporters: set[int]

    def pending_step(self) -> int | None:
        """Return the rank-consistent pending model step."""
        return resolve_identical(
            self.policy.pending_validation_step(),
            operation="policy.pending_validation_step",
        )

    def pending_publication_step(self) -> int | None:
        """Return an optional post-evaluation publication retry marker."""
        capability = getattr(self.policy, "pending_validation_publication_step", None)
        if not callable(capability):
            return None
        return resolve_identical(
            capability(),
            operation="policy.pending_validation_publication_step",
        )

    def pending_metrics(self) -> dict[str, int | float | bool | str]:
        """Return the train metric frame stored with a pending validation."""
        return dict(
            resolve_identical(
                self.policy.pending_validation_metrics(),
                operation="policy.pending_validation_metrics",
            )
        )

    def pending_reasons(self) -> frozenset[ValidationTrigger]:
        """Return stored reasons when the policy exposes them."""
        capability = getattr(self.policy, "pending_validation_reasons", None)
        if not callable(capability):
            return frozenset()
        return frozenset(
            resolve_identical(
                capability(),
                operation="policy.pending_validation_reasons",
            )
        )

    def mark_pending(
        self,
        model_step: int,
        reasons: frozenset[ValidationTrigger],
        metrics: dict[str, int | float | bool | str],
    ) -> None:
        """Mark a phase pending on every policy cell."""
        resolve_all(self.policy.mark_validation_pending(model_step, reasons, metrics))

    def defer(
        self,
        model_step: int,
        reasons: frozenset[ValidationTrigger],
        metrics: dict[str, int | float | bool | str],
    ) -> None:
        """Persist a pending phase without reading held-out data."""
        self.mark_pending(model_step, reasons, metrics)
        self.checkpoints.save_full(model_step, publish=False)

    @staticmethod
    def _evaluate(
        model_step: int,
        reasons: frozenset[ValidationTrigger],
        evaluate: ValidationEvaluator,
    ) -> ValidationReport:
        if not reasons:
            raise ValueError("validation transaction requires at least one trigger reason")
        report = evaluate(model_step, reasons)
        if not isinstance(report, ValidationReport):
            raise TypeError(f"validation evaluator returned {type(report).__name__}, expected ValidationReport")
        return report

    def _report(
        self,
        model_step: int,
        report: ValidationReport,
        train_metrics: dict[str, int | float | bool | str] | None,
    ) -> None:
        report_frame: dict[str, MetricScalar] = {**dict(train_metrics or {}), **report.metrics}
        emit_metric_frame(
            self.reporters,
            report_frame,
            model_step,
            failed_reporters=self.failed_reporters,
        )

    def _complete_durable(
        self,
        model_step: int,
        reasons: frozenset[ValidationTrigger],
        evaluate: ValidationEvaluator,
        train_metrics: dict[str, int | float | bool | str] | None,
    ) -> ValidationReport:
        report = self._evaluate(model_step, reasons, evaluate)
        resolve_all(self.policy.complete_validation(model_step))
        self.checkpoints.persist_validation_completion(model_step)
        self._report(model_step, report, train_metrics)
        return report

    def run(
        self,
        model_step: int,
        reasons: frozenset[ValidationTrigger],
        evaluate: ValidationEvaluator,
        train_metrics: dict[str, int | float | bool | str] | None = None,
    ) -> ValidationReport:
        """Begin and execute one retry-safe validation transaction."""
        if not reasons:
            raise ValueError("validation transaction requires at least one trigger reason")
        frame = dict(train_metrics or {})
        self.mark_pending(model_step, reasons, frame)
        self.checkpoints.save_full(model_step, publish=False)
        return self._complete_durable(model_step, reasons, evaluate, frame)

    def resume(
        self,
        model_step: int,
        reasons: frozenset[ValidationTrigger],
        evaluate: ValidationEvaluator,
        train_metrics: dict[str, int | float | bool | str] | None = None,
    ) -> ValidationReport:
        """Finish an already-journaled validation without rewriting its checkpoint first."""
        return self._complete_durable(model_step, reasons, evaluate, train_metrics)

    def run_ephemeral(
        self,
        model_step: int,
        reasons: frozenset[ValidationTrigger],
        evaluate: ValidationEvaluator,
        train_metrics: dict[str, int | float | bool | str] | None = None,
    ) -> ValidationReport:
        """Run one in-memory validation phase without checkpoint or publication state."""
        frame = dict(train_metrics or {})
        self.mark_pending(model_step, reasons, frame)
        report = self._evaluate(model_step, reasons, evaluate)
        resolve_all(self.policy.complete_ephemeral_validation(model_step))
        self._report(model_step, report, frame)
        return report

    def run_partial_checkpoint(
        self,
        model_step: int,
        reasons: frozenset[ValidationTrigger],
        evaluate: ValidationEvaluator,
        checkpoint_components: CheckpointComponentPlan,
        train_metrics: dict[str, int | float | bool | str] | None = None,
    ) -> ValidationReport:
        """Validate in memory, then commit one explicitly non-resumable checkpoint artifact."""
        if checkpoint_components.is_full:
            raise ValueError("run_partial_checkpoint requires a partial checkpoint component plan")
        frame = dict(train_metrics or {})
        self.mark_pending(model_step, reasons, frame)
        report = self._evaluate(model_step, reasons, evaluate)
        resolve_all(self.policy.complete_ephemeral_validation(model_step))
        self.checkpoints.save(model_step, components=checkpoint_components)
        self._report(model_step, report, frame)
        return report
