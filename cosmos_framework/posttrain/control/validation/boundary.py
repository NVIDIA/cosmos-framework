# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared validation-boundary policy for training loops."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from cosmos_framework.posttrain.control.validation.lifecycle import (
    ValidationReport,
    ValidationSchedule,
    ValidationTrigger,
)
from cosmos_framework.posttrain.control.validation.transaction import ValidationEvaluator, ValidationTransaction
from cosmos_framework.posttrain.schema.checkpoint import CheckpointComponentPlan
from cosmos_framework.posttrain.utils.log import info0, logger


@dataclass(frozen=True)
class ValidationBoundaryOutcome:
    """Control-plane effects of one validation boundary."""

    report: ValidationReport | None = None
    preempted: str | None = None


@dataclass
class ValidationBoundaryCoordinator:
    """Select a validation transaction at resume, step zero, or an update boundary."""

    schedule: ValidationSchedule
    total_steps: int
    transaction: ValidationTransaction
    evaluator: ValidationEvaluator | None
    should_stop: Callable[[], str | None]
    publication_step: Callable[[], int | None] | None = None
    pending_reasons: Callable[[], frozenset[ValidationTrigger]] | None = None

    def __post_init__(self) -> None:
        if self.schedule.enabled and self.evaluator is None:
            raise AssertionError("enabled validation has no evaluator")

    def _evaluator(self) -> ValidationEvaluator:
        evaluator = self.evaluator
        if evaluator is None:
            raise AssertionError("enabled validation has no evaluator")
        return evaluator

    def _run(
        self,
        model_step: int,
        reasons: frozenset[ValidationTrigger],
        metrics: dict[str, int | float | bool | str] | None = None,
        scheduled_checkpoint_due: bool = False,
    ) -> ValidationBoundaryOutcome:
        checkpoint_plan = self._checkpoint_plan(scheduled_checkpoint_due)
        if checkpoint_plan is None:
            report = self.transaction.run_ephemeral(model_step, reasons, self._evaluator(), train_metrics=metrics)
            return ValidationBoundaryOutcome(report=report)
        if checkpoint_plan.is_full:
            report = self.transaction.run(model_step, reasons, self._evaluator(), train_metrics=metrics)
            return ValidationBoundaryOutcome(report=report)
        if scheduled_checkpoint_due:
            self.transaction.checkpoints.save(model_step, components=checkpoint_plan)
            report = self.transaction.run_ephemeral(model_step, reasons, self._evaluator(), train_metrics=metrics)
            return ValidationBoundaryOutcome(report=report)
        report = self.transaction.run_partial_checkpoint(
            model_step,
            reasons,
            self._evaluator(),
            checkpoint_plan,
            train_metrics=metrics,
        )
        return ValidationBoundaryOutcome(report=report)

    def _checkpoint_plan(
        self,
        scheduled_checkpoint_due: bool,
    ) -> CheckpointComponentPlan | None:
        """Merge normal and validation requests so one model step is written at most once."""
        validation_plan = self.schedule.checkpoint_components if self.schedule.save_checkpoint else None
        return self.transaction.checkpoints.resolve_component_plan(
            include_configured=scheduled_checkpoint_due,
            additional=validation_plan,
        )

    def _resume(
        self,
        model_step: int,
        reasons: frozenset[ValidationTrigger],
        metrics: dict[str, int | float | bool | str],
    ) -> ValidationBoundaryOutcome:
        report = self.transaction.resume(model_step, reasons, self._evaluator(), train_metrics=metrics)
        return ValidationBoundaryOutcome(report=report)

    def _run_ephemeral(
        self,
        model_step: int,
        reasons: frozenset[ValidationTrigger],
    ) -> ValidationBoundaryOutcome:
        report = self.transaction.run_ephemeral(model_step, reasons, self._evaluator())
        return ValidationBoundaryOutcome(report=report)

    def resume_or_initial(self, model_step: int, *, resumed_from_checkpoint: bool) -> ValidationBoundaryOutcome:
        """Retry durable state or run fresh step-zero validation."""
        if not self.schedule.enabled:
            return ValidationBoundaryOutcome()

        # WorkerGroup proxies synthesize arbitrary methods, so algorithms explicitly opt in to
        # optional policy capabilities rather than probing them with ``getattr``.
        publication_step = self.publication_step() if self.publication_step is not None else None
        pending_step = self.transaction.pending_step()
        if publication_step is not None and pending_step is not None:
            raise RuntimeError(
                f"validation step {pending_step} and publication step {publication_step} cannot both be pending"
            )
        checkpoints = self.transaction.checkpoints
        if publication_step is not None:
            if publication_step != model_step:
                raise RuntimeError(
                    f"pending validation publication step {publication_step} does not match restored model step {model_step}"
                )
            checkpoints.copy_external_pending(publication_step)
            preempted = self.should_stop()
            if preempted:
                logger.warning(
                    f"termination trigger ({preempted}) — validation checkpoint publication for step "
                    f"{publication_step} remains pending"
                )
                info0(f"retaining existing publication-pending checkpoint {checkpoints.path(publication_step)}")
            else:
                checkpoints.persist_validation_completion(publication_step)
            return ValidationBoundaryOutcome(preempted=preempted)

        if pending_step is not None:
            if pending_step != model_step:
                raise RuntimeError(
                    f"pending validation step {pending_step} does not match restored model step {model_step}"
                )
            # Preserve the established stop-then-copy order for validation-pending resumes.
            preempted = self.should_stop()
            checkpoints.copy_external_pending(pending_step)
            if preempted:
                logger.warning(f"termination trigger ({preempted}) — validation step {pending_step} remains pending")
                info0(f"retaining existing pending checkpoint {checkpoints.path(pending_step)}")
                return ValidationBoundaryOutcome(preempted=preempted)
            reasons = self.schedule.reasons(pending_step, self.total_steps, pending_step=pending_step)
            if self.pending_reasons is not None:
                reasons |= self.pending_reasons()
            return self._resume(pending_step, reasons, self.transaction.pending_metrics())

        if model_step != 0:
            return ValidationBoundaryOutcome()
        reasons = self.schedule.reasons(0, self.total_steps, fresh_start=not resumed_from_checkpoint)
        if not reasons:
            return ValidationBoundaryOutcome()
        ephemeral = reasons == frozenset({ValidationTrigger.BEFORE_TRAINING})
        preempted = self.should_stop()
        if preempted:
            if ephemeral:
                logger.warning(
                    f"termination trigger ({preempted}) — skipping ephemeral before-training validation; "
                    "a fresh start will retry it"
                )
                return ValidationBoundaryOutcome(preempted=preempted)
            self.transaction.defer(0, reasons, {})
            logger.warning(f"termination trigger ({preempted}) — emergency checkpoint before validation")
            return ValidationBoundaryOutcome(preempted=preempted)
        if ephemeral:
            return self._run_ephemeral(0, reasons)
        return self._run(0, reasons)

    def runs_at(self, model_step: int) -> bool:
        """Whether ``at_step`` validates the committed update that produced ``model_step``."""
        return bool(self.schedule.reasons(model_step, self.total_steps, after_update=True))

    def at_step(
        self,
        model_step: int,
        metrics: dict[str, int | float | bool | str],
        *,
        scheduled_checkpoint_due: bool = False,
    ) -> ValidationBoundaryOutcome:
        """Run or defer validation after one committed optimizer update."""
        reasons = self.schedule.reasons(model_step, self.total_steps, after_update=True)
        if not reasons:
            return ValidationBoundaryOutcome()
        preempted = self.should_stop()
        if preempted:
            self.transaction.defer(model_step, reasons, metrics)
            logger.warning(f"termination trigger ({preempted}) — emergency checkpoint with validation pending")
            return ValidationBoundaryOutcome(preempted=preempted)
        return self._run(
            model_step,
            reasons,
            metrics,
            scheduled_checkpoint_due=scheduled_checkpoint_due,
        )

    def finish_early(self, model_step: int, metrics: dict[str, int | float | bool | str]) -> ValidationBoundaryOutcome:
        """Finish a complete update boundary when the launcher reserves time for final evaluation.

        The caller must distinguish an advance wall-time notice from immediate termination.
        A full checkpoint is durable before the evaluator starts; interruption resumes its journal.
        """
        if not self.schedule.enabled:
            return ValidationBoundaryOutcome()
        return self._run(model_step, frozenset({ValidationTrigger.FINAL}), metrics, scheduled_checkpoint_due=True)
