# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU runner contracts; real CUDA parity and continuous timing remain separate."""

import copy
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.native_qlora_qualification.test_validation_target import run, setup_target, write_scores


@pytest.mark.parametrize("mode", [None, "restart", "native"])
def test_actual_parser_preserves_restart_default(monkeypatch, mode):
    command = ["run.py", "experiment"]
    for name in (
        "spec",
        "output",
        "build",
        "model",
        "data-root",
        "score-data",
        "split-manifest",
        "time-binary",
        "kernel-gate",
    ):
        command += ["--" + name, "/unused/" + name]
    if mode is not None:
        command += ["--evaluation-mode", mode]
    received = []
    monkeypatch.setattr(sys, "argv", command)
    monkeypatch.setattr(run, "execute", received.append)
    run.main()
    assert len(received) == 1 and received[0].evaluation_mode == (mode or "restart")


@pytest.mark.parametrize(
    "changes",
    [
        {"validation_target": None},
        {"action": "evaluate"},
        {"action": "frozen"},
        {"split": "final"},
        {"evaluation_mode": "unknown"},
    ],
)
def test_native_mode_refuses_ambiguous_or_unsealed_work(changes):
    args = SimpleNamespace(
        action="train-eval", split="development", validation_target="target.json", evaluation_mode="native"
    )
    vars(args).update(changes)
    with pytest.raises(ValueError):
        run.evaluation_mode(args)


def test_missing_mode_is_historical_restart():
    assert run.evaluation_mode(SimpleNamespace()) == "restart"


def emit_native_result(output, target, *, pass_at=200, damage=None):
    native = output / "native/own-run"
    wts = native / "wts"
    wts.mkdir(parents=True)
    records = []
    steps = (
        target["evaluation_steps"]
        if pass_at is None
        else target["evaluation_steps"][: target["evaluation_steps"].index(pass_at) + 1]
    )
    if damage == "continued_after_pass":
        steps = target["evaluation_steps"]
    for index, step in enumerate(steps):
        checkpoint = native / "checkpoints" / f"iter_{step:06d}"
        checkpoint.mkdir(parents=True)
        directory = wts / f"step_{step:06d}" / "attempt"
        directory.parent.mkdir()
        write_scores(directory, checkpoint, target, ce=0.2 if step == pass_at else 0.4)
        records.append(
            {
                "step": step,
                "directory": str(directory.relative_to(native)),
                "checkpoint": str(checkpoint.resolve()),
                "target_met": step == pass_at,
                "scores_sha256": run.file_sha256(directory / "scores.jsonl"),
                "available_monotonic_seconds": 20.0 + index * 20,
            }
        )
    stop = steps[-1]
    updates = [{"step": step, "train/loss_avg": 0.2} for step in range(1, stop + 1)]
    summary = {"start": 0, "final_step": stop, "final_loss": 0.2, "target_reached": pass_at is not None}
    result = {
        "protocol": "cosmos.native-sft-wts-target.v1",
        "complete": True,
        "stop_on_target": True,
        "resumed_from_checkpoint": False,
        "horizon": 300,
        "final_step": stop,
        "checkpoint": records[-1]["checkpoint"],
        "observations": records,
        "first_passing_step": pass_at,
        "horizon_censored": pass_at is None,
    }
    if damage == "extra_update":
        updates.append({"step": stop + 1, "train/loss_avg": 0.2})
    elif damage == "future_checkpoint":
        (native / "checkpoints/iter_000301").mkdir()
    elif damage == "skipped_grid":
        records.pop(0)
    elif damage == "foreign_directory":
        records[0]["directory"] = "/foreign/scores"
    elif damage == "escaping_directory":
        records[0]["directory"] = "../foreign/scores"
    elif damage == "duplicate_directory":
        records[-1]["directory"] = records[0]["directory"]
    elif damage == "foreign_checkpoint":
        records[0]["checkpoint"] = "/foreign/iter_000100"
    elif damage == "false_pass":
        records[0]["target_met"] = True
    elif damage == "changed_scores":
        records[0]["scores_sha256"] = "0" * 64
    elif damage == "incomplete":
        result["complete"] = False
    elif damage == "nan_clock":
        records[0]["available_monotonic_seconds"] = float("nan")
    elif damage == "earlier_attempt_clock":
        records[0]["available_monotonic_seconds"] = 1
    elif damage == "future_clock":
        records[0]["available_monotonic_seconds"] = 999
    elif damage == "false_censor":
        result["horizon_censored"] = True
    elif damage == "wrong_horizon":
        result["horizon"] = 200
    elif damage == "screen_mode":
        result["stop_on_target"] = False
    elif damage == "resumed":
        result["resumed_from_checkpoint"] = True
    elif damage == "preempted":
        summary["preempted"] = "signal"
    elif damage == "wrong_start":
        summary["start"] = 100
    (native / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in updates))
    log = output / "stages/training_wts/output.log"
    log.parent.mkdir(parents=True)
    log.write_text("SFT DONE " + repr(summary) + "\n")
    (wts / "target-result.json").write_text(json.dumps(result))


def invoke(tmp_path, monkeypatch, *, pass_at=200, damage=None, experiment_changes=None, evaluation_steps=None):
    args, target, experiment = setup_target(tmp_path)
    experiment = dataclasses.replace(experiment, **(experiment_changes or {}))
    if evaluation_steps is not None:
        target["evaluation_steps"] = evaluation_steps
        Path(args.validation_target).write_text(json.dumps(target))
    args.evaluation_mode = "native"
    clock, commands = [10.0], []
    monkeypatch.setattr(run.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        run,
        "final_checkpoint",
        lambda output, exp, receipt, *, step: output / "native/own-run/checkpoints" / f"iter_{step:06d}",
    )

    class Stages:
        def run(self, name, command):
            commands.append((name, command))
            emit_native_result(tmp_path, target, pass_at=pass_at, damage=damage)
            clock[0] = max(80.0, 20.0 * len(target["evaluation_steps"]) + 10.0)
            if damage == "target_changed":
                changed = copy.deepcopy(target)
                changed["maximum_token_ce"] = 10
                Path(args.validation_target).write_text(json.dumps(changed))

    result = run.train_native_to_target(args, experiment, tmp_path, {}, Stages(), target, 0.0)
    return args, experiment, commands, result


@pytest.mark.parametrize("arm", ["nf4", "bf16"])
@pytest.mark.parametrize("pass_at", [100, None])
def test_smoothing_native_ttq_preserves_fixed_horizon_grid_and_target(tmp_path, monkeypatch, arm, pass_at):
    grid = [50, 100, 150, 200, 250, 300]
    args, experiment, commands, (_, _, result) = invoke(
        tmp_path,
        monkeypatch,
        pass_at=pass_at,
        experiment_changes={"arm": arm, "wts_answer_smoothing": 0.02},
        evaluation_steps=grid,
    )
    assert len(commands) == 1 and commands[0][0] == "training_wts"
    command = commands[0][1]
    assert command[command.index("--config") + 1] == experiment.recipe
    overrides = command[command.index("--") + 1 :]
    values = {key: json.loads(value) for item in overrides for key, value in [item.split("=", 1)] if key != "rl.resume"}
    assert values["rl.wts_answer_smoothing"] == 0.02
    assert values["rl.max_steps"] == values["rl.lr_schedule.cycle_steps"] == 300
    assert values["rl.optimizer.warmup_steps"] == 30
    assert values["rl.validation.wts.steps"] == grid
    assert values["rl.validation.wts.stop_on_target"] is True
    assert values["rl.validation.wts.min_correct"] == 6
    assert values["rl.validation.wts.max_token_ce"] == 0.3
    assert "rl.resume=off" in overrides
    assert not any("pause_after_step" in key for key in values)
    baseline = dataclasses.replace(experiment, wts_answer_smoothing=0)
    target = json.loads(Path(args.validation_target).read_text())
    assert run.native_evaluation_overrides(args, experiment, target) == run.native_evaluation_overrides(
        args, baseline, target
    )
    assert result["last_validated_step"] == (pass_at or 300)
    assert result["censored_at_training_horizon"] is (pass_at is None)


@pytest.mark.parametrize("arm", ["nf4", "bf16"])
def test_smoothing_experiment_preserves_native_hard_gold_evaluation(tmp_path, arm):
    import torch

    from cosmos_framework.posttrain.tests.test_sft_wts_smoothing import (
        VariablePacker,
        _policy,
        _row,
        _trajectory,
        evaluate_row,
        evaluate_sft_batch,
        make_objective,
    )

    _, _, baseline = setup_target(tmp_path)
    experiment = dataclasses.replace(baseline, arm=arm, wts_answer_smoothing=0.02)
    encoded = next(
        value.split("=", 1)[1] for value in experiment.overrides() if value.startswith("rl.wts_answer_smoothing=")
    )
    packer, policy = VariablePacker(), _policy()
    hard = make_objective(packer)
    candidate = make_objective(packer, wts_answer_smoothing=json.loads(encoded))
    row = _row()
    hard_metrics = evaluate_sft_batch(policy, hard, [_trajectory(row)])
    candidate_metrics = evaluate_sft_batch(policy, candidate, [_trajectory(row)])
    assert hard_metrics.keys() == candidate_metrics.keys() and "sft_objective" not in candidate_metrics
    assert all(
        torch.equal(torch.as_tensor(hard_metrics[key]), torch.as_tensor(candidate_metrics[key])) for key in hard_metrics
    )
    assert evaluate_row(policy, packer, candidate, row) == evaluate_row(policy, packer, hard, row)


@pytest.mark.parametrize("pass_at", [100, 200, None])
def test_one_native_process_preserves_horizon_cache_and_publication_time(tmp_path, monkeypatch, pass_at):
    args, experiment, commands, (checkpoint, directory, result) = invoke(tmp_path, monkeypatch, pass_at=pass_at)
    assert len(commands) == 1 and commands[0][0] == "training_wts"
    command = commands[0][1]
    assert "rl.max_steps=300" in command and "rl.lr_schedule.cycle_steps=300" in command
    assert "rl.optimizer.warmup_steps=30" in command and "rl.resume=off" in command
    assert not any(x.startswith("rl.pause_after_step=") for x in command)
    assert "rl.keep_optim_last=0" in command and "rl.keep_model_last=0" in command
    values = {
        k.removeprefix("rl.validation.wts."): json.loads(v)
        for x in command
        if x.startswith("rl.validation.wts.")
        for k, v in [x.split("=", 1)]
    }
    assert values == {
        "enabled": True,
        "stop_on_target": True,
        "data.path": str(Path(args.score_data).resolve()),
        "source_sha256": run.file_sha256(args.score_data),
        "manifest_path": str(Path(args.split_manifest).resolve()),
        "manifest_sha256": run.file_sha256(args.split_manifest),
        "expected_rows": 6,
        "steps": [100, 200, 300],
        "cache_bytes": experiment.cache_bytes,
        "min_correct": 6,
        "max_token_ce": 0.3,
    }
    stop = pass_at or 300
    assert checkpoint.name == f"iter_{stop:06d}" and directory.name == "attempt"
    assert result["first_validated_checkpoint_seconds"] == (20.0 + (stop // 100 - 1) * 20 if pass_at else None)
    assert result["last_validated_step"] == stop and result["censored_at_training_horizon"] == (pass_at is None)
    assert result["evaluation_mode"] == "native" and result["quality_qualified"] is False
    assert result["observations"][-1]["elapsed_from_runner_start_seconds"] < 80
    assert result["native_process_complete_from_runner_start_seconds"] == 80
    assert result["native_process_elapsed_seconds"] == 70
    assert (
        result["publication_to_child_return_seconds"]
        == 80 - result["observations"][-1]["elapsed_from_runner_start_seconds"]
    )


@pytest.mark.parametrize(
    "damage",
    [
        "extra_update",
        "future_checkpoint",
        "skipped_grid",
        "foreign_directory",
        "escaping_directory",
        "duplicate_directory",
        "foreign_checkpoint",
        "false_pass",
        "changed_scores",
        "incomplete",
        "nan_clock",
        "earlier_attempt_clock",
        "future_clock",
        "false_censor",
        "wrong_horizon",
        "screen_mode",
        "resumed",
        "preempted",
        "wrong_start",
        "continued_after_pass",
        "target_changed",
    ],
)
def test_native_result_refuses_incomplete_foreign_or_overrun_evidence(tmp_path, monkeypatch, damage):
    with pytest.raises(ValueError):
        invoke(tmp_path, monkeypatch, damage=damage)
    assert not (tmp_path / "target-result.json").exists()
