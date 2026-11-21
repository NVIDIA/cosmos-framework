# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real CPU SFT/DCP transactions with deterministic scoring, not WTS/CUDA qualification."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest

from cosmos_framework.posttrain.config.rl import OptimizerConfig
from cosmos_framework.posttrain.config.sft import SFTLRScheduleConfig, SFTWTSConfig
from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
from cosmos_framework.posttrain.entrypoints import sft
from cosmos_framework.posttrain.roles.trainer.validation.development import DevelopmentEvaluation
from cosmos_framework.posttrain.tests.test_entrypoints_sft import _quiet_loop
from cosmos_framework.posttrain.tests.test_entrypoints_sft_pause import _configured_runtime, _exact

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _ScoringJournal(DevelopmentEvaluation):
    """Keep real journal validation/publication; substitute only model scoring/codec."""

    def __init__(self, settings, output, *, pass_at):
        self.settings, self.output, self.horizon = settings, output, 4
        self.history, self.artifacts = [], {}
        self.pipeline = None
        self.pass_at = pass_at
        self.evaluated = []
        self.verified = []

    def evaluate(self, model, optimizer, step):
        if step not in self.settings.steps:
            return {}
        assert step == self.settings.steps[len(self.history)]
        self.evaluated.append(step)
        correct = 4 if self.pass_at is not None and step >= self.pass_at else 3
        ce = 0.1 if correct == 4 else 0.2
        metrics = {
            "rows": 4,
            "restricted_choice_accuracy": correct / 4,
            "token_weighted_ce": ce,
            "native_supervised_tokens": 8,
        }
        self.history.append(
            {
                "step": step,
                "rows": 4,
                "correct": correct,
                "token_weighted_ce": ce,
                "target_met": correct >= self.settings.min_correct and ce <= self.settings.max_token_ce,
                "metrics": metrics,
                "scores_sha256": str(step) * 64,
                "selected_content_sha256": "a" * 64,
                "processed_inputs_sha256": "b" * 64,
            }
        )
        directory = self.output / "wts" / f"step_{step:06d}" / "stub"
        directory.mkdir(parents=True)
        self.artifacts[step] = {
            "directory": str(directory.relative_to(self.output)),
            "checkpoint": str(self.output / "checkpoints" / f"iter_{step:06d}"),
        }
        return {"val/wts_correct": correct, "val/wts_rows": 4, "val/wts_token_weighted_ce": ce}

    def verify_checkpoint(self, checkpoint, manager, model, optimizer):
        # The fixture is a real dense CPU trainer, not the native adapter codec.
        assert optimizer is self.optimizer
        assert manager.resolve_resume(checkpoint, str(self.output / "checkpoints")) == checkpoint
        state = manager.load_component(checkpoint, "trainer")["validation"]
        assert state["version"] == 3
        assert state["wts"] == self.state_dict()
        assert state["progress"]["pending_step"] is None
        assert state["publication"] is None
        self.verified.append(checkpoint)


def _development_runtime(source: Path, output: Path, *, pass_at=None, stop_on_target=True, pause=None):
    config, units, trainer = _configured_runtime(
        source,
        validation_every=2,
        max_steps=4,
        pause_after_step=pause,
        save_every=2,
        checkpoint_format="dcp",
        deterministic=True,
        lr_schedule=SFTLRScheduleConfig(kind="cosine", cycle_steps=4),
        optimizer=OptimizerConfig(lr=0.01, warmup_steps=1),
    )
    config.rl.validation.wts = SFTWTSConfig(
        enabled=True,
        data=replace(config.rl.data, shuffle=False),
        source_sha256="a" * 64,
        manifest_path=str(source / "manifest.json"),
        manifest_sha256="b" * 64,
        expected_rows=4,
        steps=[2, 4],
        min_correct=4,
        max_token_ce=0.15,
        stop_on_target=stop_on_target,
    )
    evaluator = _ScoringJournal(config.rl.validation.wts, output, pass_at=pass_at)
    evaluator.optimizer = trainer.opt
    trainer.development_evaluator = evaluator
    trainer.validation_state.configured = False
    trainer.configure_validation(
        {
            "enabled": True,
            "schedule": {"every_steps": 2, "total_steps": 4},
            "wts": {"steps": [2, 4], "stop_on_target": stop_on_target, "min_correct": 4, "max_token_ce": 0.15},
        }
    )
    return config, units, trainer, evaluator


def _restore_development(config, units, trainer, checkpoint, output):
    # Match the production builder: load one stable trainer/cursor generation,
    # bind provenance on both roles, then configure the deferred journal.
    identity = copy.deepcopy(trainer.validation_state.identity)
    trainer.validation_state.configured = False
    config.rl.resume = checkpoint
    lifecycle = CheckpointLifecycle.from_config(config, config.rl, run_out_dir=str(output))
    restored = lifecycle.load_trainer_and_component(trainer, "data")
    assert restored is not None and restored[1] is not None
    units["data"].load_state_dict(restored[1]).result()
    provenance = trainer.loaded_checkpoint_provenance()
    assert provenance is not None
    units["data"].set_loaded_checkpoint_provenance(provenance).result()
    trainer.configure_validation(identity)


def test_first_pass_stops_after_committed_checkpoint_without_later_update(tmp_path, monkeypatch):
    _quiet_loop(monkeypatch)
    output = tmp_path / "run"
    config, units, trainer, evaluator = _development_runtime(tmp_path / "source", output, pass_at=2)
    started = time.monotonic()
    result = sft.loop(units, config, str(output))
    assert result["final_step"] == trainer.iteration() == 2
    assert result["target_reached"] is True
    assert evaluator.evaluated == [2] and len(evaluator.verified) == 1
    assert not (output / "checkpoints/iter_000004").exists()
    recorded = json.loads((output / "wts/target-result.json").read_text())
    assert recorded["complete"] is True and recorded["first_passing_step"] == 2
    assert started < recorded["observations"][0]["available_monotonic_seconds"] <= time.monotonic()


def test_nonstopping_evaluation_reaches_declared_horizon_after_early_pass(tmp_path, monkeypatch):
    _quiet_loop(monkeypatch)
    output = tmp_path / "run"
    config, units, trainer, evaluator = _development_runtime(
        tmp_path / "source", output, pass_at=2, stop_on_target=False
    )
    result = sft.loop(units, config, str(output))
    assert result["final_step"] == trainer.iteration() == 4
    assert evaluator.evaluated == [2, 4]
    recorded = json.loads((output / "wts/target-result.json").read_text())
    assert recorded["complete"] is True and recorded["first_passing_step"] == 2


def test_paused_wts_journal_restores_exact_updates_schedule_rng_and_cursor(tmp_path, monkeypatch):
    _quiet_loop(monkeypatch)
    source = tmp_path / "source"
    config, units, full, full_eval = _development_runtime(source, tmp_path / "complete")
    complete = sft.loop(units, config, str(tmp_path / "complete"))
    assert complete["final_step"] == 4 and complete["target_reached"] is False
    expected = copy.deepcopy((full.model.state_dict(), full.opt.state_dict(), full._trainer_state()))
    expected_data = units["data"].state_dict().result()

    config, units, partial, partial_eval = _development_runtime(source, tmp_path / "paused", pause=2)
    paused = sft.loop(units, config, str(tmp_path / "paused"))
    assert paused["paused"] is True and partial_eval.evaluated == [2]
    assert not json.loads((tmp_path / "paused/wts/target-result.json").read_text())["complete"]
    config, units, resumed, resumed_eval = _development_runtime(source, tmp_path / "resumed")
    _restore_development(config, units, resumed, paused["checkpoint"], tmp_path / "resumed")
    assert resumed_eval.state_dict() == partial_eval.state_dict()
    result = sft.loop(units, config, str(tmp_path / "resumed"))
    assert result["start"] == 2 and result["final_step"] == 4 and result["target_reached"] is False
    assert resumed_eval.evaluated == [4]
    _exact((resumed.model.state_dict(), resumed.opt.state_dict(), resumed._trainer_state()), expected)
    _exact(units["data"].state_dict().result(), expected_data)
    assert resumed_eval.state_dict() == full_eval.state_dict()
    recorded = json.loads((tmp_path / "resumed/wts/target-result.json").read_text())
    assert recorded["horizon_censored"] is True
    assert recorded["observations"][0]["available_monotonic_seconds"] is None
    assert recorded["observations"][0]["available_on_this_attempt"] is False


@pytest.mark.parametrize("failure", ["scoring", "publication", "readback"])
def test_failed_validation_or_readback_cannot_publish_a_passing_target(tmp_path, monkeypatch, failure):
    _quiet_loop(monkeypatch)
    output = tmp_path / "run"
    config, units, trainer, evaluator = _development_runtime(tmp_path / "source", output, pass_at=2)

    def fail(*args, **kwargs):
        raise RuntimeError(f"{failure} failed")

    if failure == "scoring":
        evaluate = evaluator.evaluate

        def fail_scoring(model, optimizer, step):
            return fail() if step == 2 else evaluate(model, optimizer, step)

        monkeypatch.setattr(evaluator, "evaluate", fail_scoring)
    elif failure == "publication":
        monkeypatch.setattr(trainer.checkpoint_manager, "publish_validation", fail)
    else:
        monkeypatch.setattr(evaluator, "verify_checkpoint", fail)
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        sft.loop(units, config, str(output))
    assert trainer.iteration() == 2
    assert not (output / "wts/target-result.json").exists()
    assert not (output / "checkpoints/iter_000004").exists()
    saved = trainer.checkpoint_manager.load_component(str(output / "checkpoints/iter_000002"), "trainer")
    assert saved["validation"]["progress"]["pending_step"] == (2 if failure == "scoring" else None)
    assert bool(saved["validation"]["publication"]) is (failure == "publication")


@pytest.mark.parametrize("failure", ["scoring", "publication"])
def test_resume_retries_only_the_pending_validation_phase(tmp_path, monkeypatch, failure):
    _quiet_loop(monkeypatch)
    source, original = tmp_path / "source", tmp_path / "original"
    config, units, trainer, evaluator = _development_runtime(source, original, pass_at=2)

    def fail(*args, **kwargs):
        raise RuntimeError("injected interruption")

    if failure == "scoring":
        evaluate = evaluator.evaluate

        def fail_scoring(model, optimizer, step):
            return fail() if step == 2 else evaluate(model, optimizer, step)

        monkeypatch.setattr(evaluator, "evaluate", fail_scoring)
    else:
        monkeypatch.setattr(trainer.checkpoint_manager, "publish_validation", fail)
    with pytest.raises(RuntimeError, match="injected interruption"):
        sft.loop(units, config, str(original))
    checkpoint = str(original / "checkpoints/iter_000002")
    output = tmp_path / "resumed"
    config, units, resumed, resumed_eval = _development_runtime(source, output, pass_at=2)
    _restore_development(config, units, resumed, checkpoint, output)
    before = copy.deepcopy((resumed.model.state_dict(), resumed.opt.state_dict()))
    result = sft.loop(units, config, str(output))
    assert result["start"] == result["final_step"] == 2 and result["target_reached"] is True
    assert resumed_eval.evaluated == ([2] if failure == "scoring" else [])
    assert [row["step"] for row in resumed_eval.history] == [2]
    _exact((resumed.model.state_dict(), resumed.opt.state_dict()), before)
    assert resumed.pending_validation_step() is None
    assert resumed.pending_validation_publication_step() is None
    recorded = json.loads((output / "wts/target-result.json").read_text())
    assert recorded["observations"][0]["available_on_this_attempt"] is (failure == "scoring")
