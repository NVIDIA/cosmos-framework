# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cosmos_framework.posttrain.config.rl import ValidationScheduleConfig
from cosmos_framework.posttrain.control.validation import (
    ValidationProgress,
    ValidationReport,
    ValidationSchedule,
    ValidationTrigger,
)
from cosmos_framework.posttrain.control.validation.boundary import ValidationBoundaryCoordinator
from cosmos_framework.posttrain.control.validation.transaction import ValidationTransaction
from cosmos_framework.posttrain.schema.checkpoint import CheckpointComponentPlan


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_schedule_trigger_contract() -> None:
    schedule = ValidationSchedule.from_config(
        ValidationScheduleConfig(enabled=True, before_training=True, every_steps=2)
    )
    observed: list[tuple[int, frozenset[ValidationTrigger]]] = []
    initial = schedule.reasons(0, 4, fresh_start=True)
    if initial:
        observed.append((0, initial))
    for step in range(1, 5):
        reasons = schedule.reasons(step, 4, after_update=True)
        if reasons:
            observed.append((step, reasons))
    assert [step for step, _ in observed] == [0, 2, 4]
    assert observed[-1][1] == frozenset({ValidationTrigger.PERIODIC, ValidationTrigger.FINAL})

    final_only = ValidationSchedule.from_config(ValidationScheduleConfig(enabled=True, every_steps=0))
    assert final_only.reasons(2, 3, after_update=True) == frozenset()
    assert final_only.reasons(3, 3, after_update=True) == frozenset({ValidationTrigger.FINAL})
    assert final_only.reasons(0, 0, fresh_start=True) == frozenset({ValidationTrigger.FINAL})

    assert schedule.reasons(2, 4) == frozenset()
    assert schedule.reasons(2, 4, pending_step=2) == frozenset({ValidationTrigger.PENDING})

    before_training = ValidationSchedule.from_config(ValidationScheduleConfig(enabled=True, before_training=True))
    assert before_training.reasons(0, 3) == frozenset()
    assert before_training.reasons(0, 3, fresh_start=True) == frozenset({ValidationTrigger.BEFORE_TRAINING})
    with pytest.raises(ValueError, match="model_step=0"):
        before_training.reasons(1, 3, fresh_start=True)


@pytest.mark.L0
@pytest.mark.CPU
def test_fresh_before_training_boundary_is_ephemeral() -> None:
    """A pure initial pass evaluates and reports without touching the durable checkpoint surface."""
    from cosmos_framework.posttrain.control.validation.boundary import ValidationBoundaryCoordinator
    from cosmos_framework.posttrain.control.validation.transaction import ValidationTransaction

    lifecycle_events: list[str] = []
    policy = SimpleNamespace(
        pending_validation_step=lambda: None,
        pending_validation_metrics=lambda: {},
        mark_validation_pending=lambda *_args: lifecycle_events.append("mark"),
        complete_validation=lambda _step: lifecycle_events.append("complete"),
        complete_ephemeral_validation=lambda _step: lifecycle_events.append("complete_ephemeral"),
    )
    checkpoints = SimpleNamespace(
        save=lambda *_args, **_kwargs: lifecycle_events.append("save"),
        persist_validation_completion=lambda _step: lifecycle_events.append("persist"),
    )
    transaction = ValidationTransaction(
        policy=policy,
        checkpoints=checkpoints,  # type: ignore[arg-type]
        reporters=[],
        failed_reporters=set(),
    )

    def evaluate(model_step: int, reasons: frozenset[ValidationTrigger]) -> ValidationReport:
        lifecycle_events.append("evaluate")
        return ValidationReport(
            model_step=model_step,
            sample_count=1,
            metrics={"val/loss_avg": 0.25},
            source_fingerprint="held-out-v1",
            trigger_reasons=reasons,
            duration_s=0.01,
        )

    boundary = ValidationBoundaryCoordinator(
        schedule=ValidationSchedule(enabled=True, before_training=True, every_steps=0),
        total_steps=1,
        transaction=transaction,
        evaluator=evaluate,
        should_stop=lambda: None,
    )

    outcome = boundary.resume_or_initial(0, resumed_from_checkpoint=False)

    assert outcome.report is not None
    assert outcome.report.trigger_reasons == frozenset({ValidationTrigger.BEFORE_TRAINING})
    assert outcome.preempted is None
    assert lifecycle_events == ["mark", "evaluate", "complete_ephemeral"]


def _partial_validation_boundary(events: list[str], *, fail: bool = False) -> ValidationBoundaryCoordinator:
    """Build the common model-only validation boundary used by ordering tests."""
    plan = CheckpointComponentPlan.from_value(("model",))
    policy = SimpleNamespace(
        pending_validation_step=lambda: None,
        pending_validation_metrics=lambda: {},
        mark_validation_pending=lambda *_args: events.append("mark"),
        complete_ephemeral_validation=lambda _step: events.append("complete"),
    )

    def save(_step: int, *, components: object, publish: bool = True) -> None:
        assert publish
        assert CheckpointComponentPlan.from_value(components) == plan  # type: ignore[arg-type]
        events.append("save")

    def resolve_component_plan(*, include_configured: bool, additional: object) -> CheckpointComponentPlan:
        assert CheckpointComponentPlan.from_value(additional) == plan  # type: ignore[arg-type]
        events.append(f"plan:{include_configured}")
        return plan

    def evaluate(model_step: int, reasons: frozenset[ValidationTrigger]) -> ValidationReport:
        events.append("evaluate")
        if fail:
            raise RuntimeError("held-out evaluator failed")
        return ValidationReport(
            model_step=model_step,
            sample_count=1,
            metrics={"val/loss_avg": 0.25},
            source_fingerprint="held-out-v1",
            trigger_reasons=reasons,
            duration_s=0.01,
        )

    checkpoints = SimpleNamespace(save=save, resolve_component_plan=resolve_component_plan)
    return ValidationBoundaryCoordinator(
        schedule=ValidationSchedule(
            enabled=True,
            before_training=False,
            every_steps=1,
            save_checkpoint=True,
            checkpoint_components=plan,
        ),
        total_steps=2,
        transaction=ValidationTransaction(  # type: ignore[arg-type]
            policy=policy,
            checkpoints=checkpoints,
            reporters=[],
            failed_reporters=set(),
        ),
        evaluator=evaluate,
        should_stop=lambda: None,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_only_partial_artifact_is_saved_after_ephemeral_completion() -> None:
    events: list[str] = []
    boundary = _partial_validation_boundary(events)
    boundary.at_step(1, {"train/loss_avg": 0.5})

    assert events == ["plan:False", "mark", "evaluate", "complete", "save"]


@pytest.mark.L0
@pytest.mark.CPU
def test_regular_partial_checkpoint_is_committed_before_overlapping_validation_failure() -> None:
    events: list[str] = []
    boundary = _partial_validation_boundary(events, fail=True)

    with pytest.raises(RuntimeError, match="held-out evaluator failed"):
        boundary.at_step(1, {"train/loss_avg": 0.5}, scheduled_checkpoint_due=True)

    assert events == ["plan:True", "save", "mark", "evaluate"]


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_progress_round_trip_and_exact_completion() -> None:
    progress = ValidationProgress()
    reasons = frozenset({ValidationTrigger.PERIODIC, ValidationTrigger.FINAL})
    progress.begin(4, reasons, metrics={"train/loss_avg": 0.25})
    restored = ValidationProgress()
    restored.load_state_dict(progress.state_dict())
    assert restored.pending_step == 4 and restored.pending_reasons == reasons
    assert restored.pending_metrics == {"train/loss_avg": 0.25}
    with pytest.raises(RuntimeError, match="pending step is 4"):
        restored.complete(3)
    restored.complete(4)
    assert restored.state_dict() == {
        "pending_step": None,
        "pending_reasons": [],
        "pending_metrics": {},
    }


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "state",
    [
        {"pending_step": True, "pending_reasons": ["periodic"], "pending_metrics": {}},
        {
            "pending_step": 1,
            "pending_reasons": ["periodic"],
            "pending_metrics": {"train/loss_avg": float("nan")},
        },
    ],
)
def test_validation_progress_rejects_ambiguous_or_nonfinite_checkpoint_state(state: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ValidationProgress().load_state_dict(state)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"enabled": 1, "before_training": False, "every_steps": 0}, TypeError),
        ({"enabled": True, "before_training": 0, "every_steps": 0}, TypeError),
        ({"enabled": True, "before_training": False, "every_steps": True}, TypeError),
        ({"enabled": True, "before_training": False, "every_steps": -1}, ValueError),
    ],
)
def test_validation_schedule_rejects_ambiguous_direct_construction(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        ValidationSchedule(**kwargs)  # type: ignore[arg-type]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "metrics",
    [
        {"grpo_loss": 1.0},
        {"train/loss_avg": float("inf")},
        {"train/loss_avg": object()},
    ],
)
def test_validation_progress_rejects_invalid_pending_metric_frames(metrics: dict[str, object]) -> None:
    progress = ValidationProgress()
    with pytest.raises((TypeError, ValueError)):
        progress.begin(
            1,
            frozenset({ValidationTrigger.PERIODIC}),
            metrics=metrics,  # type: ignore[arg-type]
        )
    assert progress.state_dict() == {
        "pending_step": None,
        "pending_reasons": [],
        "pending_metrics": {},
    }


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "metrics",
    [
        {},
        {"dpo_loss": 1.0},
        {"val/loss_avg": float("nan")},
        {"val/complete": True},
    ],
)
def test_validation_report_rejects_invalid_public_metrics(metrics: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ValidationReport(
            model_step=1,
            sample_count=1,
            metrics=metrics,  # type: ignore[arg-type]
            source_fingerprint="source-v1",
            trigger_reasons=frozenset({ValidationTrigger.FINAL}),
            duration_s=0.1,
        )
