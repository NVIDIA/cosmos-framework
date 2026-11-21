# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Deterministic attempt boundaries using real CPU updates and checkpoint I/O."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from cosmos_framework.posttrain.checkpoint import CheckpointManager
from cosmos_framework.posttrain.config.build import build_sft_lr_scheduler
from cosmos_framework.posttrain.config.rl import OptimizerConfig
from cosmos_framework.posttrain.config.sft import SFTConfig, SFTLRScheduleConfig, SFTValidationConfig
from cosmos_framework.posttrain.entrypoints import sft
from cosmos_framework.posttrain.roles import roles
from cosmos_framework.posttrain.roles.env.sft import build_sft_env
from cosmos_framework.posttrain.tests.test_entrypoints_sft import _quiet_loop, _runtime
from cosmos_framework.posttrain.utils.report import Reporter

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2", 301])
def test_pause_limit_is_an_absolute_typed_optimizer_boundary(limit: Any) -> None:
    with pytest.raises(ValueError, match="pause_after_step"):
        SFTConfig(max_steps=300, pause_after_step=limit)


def _configured_runtime(path: Path, *, validation_every: int = 0, **options: Any):
    config, units, trainer = _runtime(path, keep_optim_last=0, **options)
    trainer.checkpoint_manager = CheckpointManager(
        checkpoint_format=config.rl.checkpoint_format,
        save_async=config.rl.save_async,
        keep_optim_last=0,
        checkpoint_components=config.rl.checkpoint_components,
    )
    trainer.lr_scheduler = build_sft_lr_scheduler(
        trainer.opt, config.rl.optimizer, config.rl.lr_schedule, config.rl.max_steps
    )
    if validation_every:
        config.rl.validation = SFTValidationConfig(
            enabled=True,
            before_training=True,
            every_steps=validation_every,
            data=config.rl.data,
            batch_size=2,
        )
        units["validation"] = roles.env("validation", body=build_sft_env(config.rl.data, 7, training=False))
        # Reconfigure the pristine fixture before any update; its publication
        # journal retains this same ValidationState object.
        trainer.validation_state.configured = False
        trainer.configure_validation(
            {"enabled": True, "schedule": {"every_steps": validation_every, "total_steps": config.rl.max_steps}}
        )
    return config, units, trainer


def _restore(config: Any, units: dict[str, Any], trainer: Any, checkpoint: str) -> None:
    validation_identity = copy.deepcopy(trainer.validation_state.identity)
    trainer.load(checkpoint)
    trainer.configure_validation(validation_identity)
    units["data"].load_state_dict(trainer.checkpoint_manager.load_component(checkpoint, "data")).result()
    config.rl.resume = checkpoint


def _exact(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _exact(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for value, wanted in zip(actual, expected, strict=True):
            _exact(value, wanted)
    else:
        assert actual == expected


def test_mature_pause_and_resume_preserve_schedule_validation_and_full_state(tmp_path, monkeypatch) -> None:
    """Run the real 300-step horizon, pausing at 250 without shortening it."""
    _quiet_loop(monkeypatch)
    frames: list[tuple[int, dict[str, Any]]] = []

    class Capture(Reporter):
        def _emit(self, metrics: dict[str, Any], step: int) -> None:
            frames.append((step, dict(metrics)))

    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.report.make_reporters",
        lambda *a, **kw: [Capture()],
    )
    options = dict(
        max_steps=300,
        save_every=50,
        checkpoint_format="dcp",
        lr_schedule=SFTLRScheduleConfig(kind="cosine", cycle_steps=300),
        optimizer=OptimizerConfig(lr=0.01, warmup_steps=30),
        validation_every=50,
    )
    config, units, full = _configured_runtime(tmp_path / "source", **options)
    complete = sft.loop(units, config, str(tmp_path / "complete"))
    assert complete.keys() == {"start", "final_step", "final_loss"}
    expected_frames = copy.deepcopy(frames)
    expected = copy.deepcopy((full.model.state_dict(), full.opt.state_dict(), full._trainer_state()))
    expected_data = units["data"].state_dict().result()

    frames.clear()
    config, units, partial = _configured_runtime(tmp_path / "source", pause_after_step=250, **options)
    save_steps = []
    save = partial.save

    def tracked_save(path, **kwargs):
        save_steps.append(Path(path).name)
        return save(path, **kwargs)

    monkeypatch.setattr(partial, "save", tracked_save)
    paused = sft.loop(units, config, str(tmp_path / "paused"))
    assert paused["paused"] is True and paused["final_step"] == 250 and paused["start"] == 0
    assert config.rl.max_steps == config.rl.lr_schedule.cycle_steps == 300
    assert save_steps.count("iter_000250") == 1, "the ordinary full validation boundary must not be saved twice"
    assert partial.pending_validation_step() is None
    checkpoint = paused["checkpoint"]
    assert partial.checkpoint_manager.resolve_resume(checkpoint, str(tmp_path / "paused/checkpoints")) == checkpoint
    assert [step for step, frame in frames if "val/loss_avg" in frame] == [0, 50, 100, 150, 200, 250]

    config, units, resumed = _configured_runtime(tmp_path / "source", **options)
    _restore(config, units, resumed, checkpoint)
    summary = sft.loop(units, config, str(tmp_path / "resumed"))
    assert summary["start"] == 250 and summary["final_step"] == 300 and "paused" not in summary
    assert [step for step, _ in frames] == [step for step, _ in expected_frames]
    for (_, actual), (_, wanted) in zip(frames, expected_frames, strict=True):
        # Wall clocks are attempt-local; numerical training/validation metrics are unchanged.
        _exact(
            {k: v for k, v in actual.items() if not k.startswith("timer/")},
            {k: v for k, v in wanted.items() if not k.startswith("timer/")},
        )
    _exact((resumed.model.state_dict(), resumed.opt.state_dict(), resumed._trainer_state()), expected)
    _exact(units["data"].state_dict().result(), expected_data)


@pytest.mark.parametrize(
    "checkpoint_format,async_save", [("consolidated", False), ("consolidated", True), ("dcp", False)]
)
@pytest.mark.parametrize("cadence", ["disabled", "model_only"])
def test_pause_forces_full_committed_state_even_without_full_save_cadence(
    tmp_path, monkeypatch, checkpoint_format, async_save, cadence
) -> None:
    _quiet_loop(monkeypatch)
    options = dict(
        save_every=0 if cadence == "disabled" else 1,
        checkpoint_format=checkpoint_format,
        save_async=async_save,
        pause_after_step=1,
    )
    if cadence == "model_only":
        options["checkpoint_components"] = ["model"]
    config, units, trainer = _configured_runtime(tmp_path / "source", **options)
    result = sft.loop(units, config, str(tmp_path / "run"))
    assert result["paused"] is True and trainer.iteration() == 1
    checkpoint = result["checkpoint"]
    assert Path(checkpoint).name == "iter_000001"
    manager = CheckpointManager(checkpoint_format=checkpoint_format)
    assert manager.resolve_resume(checkpoint, str(tmp_path / "run/checkpoints")) == checkpoint
    assert manager.load_component(checkpoint, "data") is not None
    fresh_config, fresh_units, restored = _configured_runtime(tmp_path / "source", **options)
    _restore(fresh_config, fresh_units, restored, checkpoint)
    _exact(restored.model.state_dict(), trainer.model.state_dict())
    _exact(restored.opt.state_dict(), trainer.opt.state_dict())
    _exact(restored._trainer_state(), trainer._trainer_state())
    assert restored._update_state_dict()["accum"] == 0


@pytest.mark.parametrize("limit,validation_steps", [(3, [0, 2]), (4, [0, 2, 4])])
def test_pause_keeps_ordinary_validation_cadence(tmp_path, monkeypatch, limit, validation_steps) -> None:
    _quiet_loop(monkeypatch)
    config, units, trainer = _configured_runtime(
        tmp_path / "source", pause_after_step=limit, save_every=0, validation_every=2
    )
    observed = []
    evaluate = sft.evaluate

    def tracked_evaluate(units, rl, step, reasons):
        observed.append(step)
        return evaluate(units, rl, step, reasons)

    monkeypatch.setattr(sft, "evaluate", tracked_evaluate)
    result = sft.loop(units, config, str(tmp_path / "run"))
    assert observed == validation_steps
    assert result["paused"] is True and result["final_step"] == limit
    assert trainer.pending_validation_step() is None
    restored_config, restored_units, restored = _configured_runtime(
        tmp_path / "source", save_every=0, validation_every=2
    )
    _restore(restored_config, restored_units, restored, result["checkpoint"])
    assert restored.pending_validation_step() is None


def test_validation_failure_cannot_be_reported_as_a_pause(tmp_path, monkeypatch) -> None:
    _quiet_loop(monkeypatch)
    config, units, trainer = _configured_runtime(tmp_path / "source", pause_after_step=2, validation_every=2)
    evaluate = sft.evaluate

    def fail_at_boundary(units, rl, step, reasons):
        if step == 2:
            raise RuntimeError("validation failure")
        return evaluate(units, rl, step, reasons)

    monkeypatch.setattr(sft, "evaluate", fail_at_boundary)
    with pytest.raises(RuntimeError, match="validation failure"):
        sft.loop(units, config, str(tmp_path / "run"))
    assert trainer.pending_validation_step() == 2


@pytest.mark.parametrize("limit", [1, 2])
def test_restored_pause_at_or_before_start_fails_before_training_or_validation(tmp_path, monkeypatch, limit) -> None:
    _quiet_loop(monkeypatch)
    config, units, _ = _configured_runtime(tmp_path / "source", pause_after_step=2)
    paused = sft.loop(units, config, str(tmp_path / "first"))
    config, units, restored = _configured_runtime(tmp_path / "source", pause_after_step=limit)
    _restore(config, units, restored, paused["checkpoint"])
    before = units["data"].state_dict().result()
    monkeypatch.setattr(sft, "train_step", lambda *a: pytest.fail("stale pause must not train"))
    monkeypatch.setattr(sft, "evaluate", lambda *a: pytest.fail("stale pause must not validate"))
    with pytest.raises(ValueError, match="must exceed the restored"):
        sft.loop(units, config, str(tmp_path / "second"))
    _exact(units["data"].state_dict().result(), before)


@pytest.mark.parametrize("failure", ["save", "finalize_checkpoints"])
def test_failed_checkpoint_cannot_report_paused(tmp_path, monkeypatch, failure) -> None:
    _quiet_loop(monkeypatch)
    config, units, trainer = _configured_runtime(tmp_path / "source", save_every=0, pause_after_step=1)

    def fail(*a, **kw):
        raise RuntimeError("checkpoint failure")

    monkeypatch.setattr(trainer, failure, fail)
    with pytest.raises(RuntimeError, match="checkpoint failure"):
        sft.loop(units, config, str(tmp_path / "run"))


def test_termination_takes_precedence_over_requested_pause(tmp_path, monkeypatch) -> None:
    _quiet_loop(monkeypatch)
    config, units, _ = _configured_runtime(tmp_path / "source", pause_after_step=1)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.control.termination.TerminationWatch",
        lambda: SimpleNamespace(should_stop=lambda: "signal"),
    )
    result = sft.loop(units, config, str(tmp_path / "run"))
    assert result == {"final_step": 1, "preempted": "signal"}
