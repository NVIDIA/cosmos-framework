# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU workflow contracts; these do not establish GPU cadence or time-to-quality."""

import copy
import dataclasses
import importlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
run = importlib.import_module("run")
from protocol import Experiment, file_sha256

from tools.native_qlora_qualification.common import digest_json
from tools.native_qlora_qualification.test_quality import fixtures


def setup_target(tmp_path):
    data, manifest = tmp_path / "development.jsonl", tmp_path / "manifest.json"
    examples = fixtures()[0]["trained_nf4"]
    data.write_text("".join(json.dumps({"id": row["id"]}) + "\n" for row in examples))
    manifest.write_text(
        json.dumps(
            {
                "protocol": "cosmos.wts-group-split.v1",
                "role": "development",
                "data_sha256": file_sha256(data),
                "group_count": 3,
                "rows": [
                    {
                        "id": row["id"],
                        "raw_row_sha256": digest_json({"id": row["id"]}),
                        "media_sha256": row["media_sha256"],
                        "group_sha256": str(i // 2),
                    }
                    for i, row in enumerate(examples)
                ],
            }
        )
    )
    args = SimpleNamespace(
        action="train-eval",
        split="development",
        score_data=str(data),
        split_manifest=str(manifest),
        selection_receipt=None,
        validation_target=str(tmp_path / "target.json"),
    )
    target = {
        "protocol": "cosmos.qlora-validation-target.v1",
        "evaluation_steps": [100, 200, 300],
        "expected_rows": 6,
        "minimum_correct": 6,
        "maximum_token_ce": 0.3,
        "score_data_sha256": file_sha256(data),
        "split_manifest_sha256": file_sha256(manifest),
    }
    Path(args.validation_target).write_text(json.dumps(target))
    return args, target, Experiment("target_nf4", "nf4", rank=32, alpha=64, cache_bytes=3 * 1024**3)


@pytest.mark.parametrize("arm", ["nf4", "bf16"])
@pytest.mark.parametrize("cache_bytes", [0, 3 * 1024**3])
def test_cadence_scorer_receives_its_separate_cache_flag(tmp_path, monkeypatch, arm, cache_bytes):
    args, _, experiment = setup_target(tmp_path)
    experiment = dataclasses.replace(experiment, arm=arm, cache_bytes=cache_bytes)
    command = run.score_command(args, experiment, tmp_path / "scores", tmp_path / "iter_000100")
    score = importlib.import_module("tools.native_qlora_qualification.score")
    parsed = []
    monkeypatch.setattr(score, "run", lambda value: parsed.append(value) or {"parser_only": True})
    script_index = next(i for i, value in enumerate(command) if value.endswith("score.py"))
    monkeypatch.setattr(sys, "argv", command[script_index:])
    score.main()
    assert len(parsed) == 1
    assert parsed[0].preprocessing_cache_bytes == cache_bytes
    assert parsed[0].all_rows and parsed[0].split == "development"
    assert parsed[0].config == experiment.recipe
    assert f"rl.preprocessing_cache_bytes={cache_bytes}" in parsed[0].override


@pytest.mark.parametrize("attribute", ["absent", None])
def test_non_cadence_score_command_preserves_historical_argv(tmp_path, attribute):
    args, _, experiment = setup_target(tmp_path)
    cadence = run.score_command(args, experiment, tmp_path / "scores", tmp_path / "iter_000100")
    if attribute == "absent":
        del args.validation_target
    else:
        args.validation_target = None
    historical = run.score_command(args, experiment, tmp_path / "scores", tmp_path / "iter_000100")
    assert cadence == [*historical, "--preprocessing-cache-bytes", str(experiment.cache_bytes)]
    assert "--preprocessing-cache-bytes" not in historical


def test_negative_cadence_capacity_cannot_reach_model_work(tmp_path):
    _, _, experiment = setup_target(tmp_path)
    with pytest.raises(ValueError):
        dataclasses.replace(experiment, cache_bytes=-1)
    score = importlib.import_module("tools.native_qlora_qualification.score")
    with pytest.raises(ValueError, match="nonnegative integer"):
        score.run(SimpleNamespace(preprocessing_cache_bytes=-1))


@pytest.mark.parametrize(
    "changes",
    [
        {"evaluation_steps": []},
        {"evaluation_steps": [100, 100, 300]},
        {"evaluation_steps": [200, 100, 300]},
        {"evaluation_steps": [100, 200]},
        {"evaluation_steps": [True, 300]},
        {"evaluation_steps": [0, 300]},
        {"expected_rows": True},
        {"minimum_correct": 7},
        {"minimum_correct": 1.0},
        {"maximum_token_ce": float("nan")},
        {"maximum_token_ce": -1},
        {"score_data_sha256": "foreign"},
        {"split_manifest_sha256": "foreign"},
        {"unknown": 1},
    ],
)
def test_target_refuses_invalid_grid_metrics_or_input_identity(tmp_path, changes):
    args, target, experiment = setup_target(tmp_path)
    target.update(changes)
    Path(args.validation_target).write_text(json.dumps(target))
    with pytest.raises(ValueError):
        run.read_validation_target(args.validation_target, args, experiment)


@pytest.mark.parametrize(
    "action,split",
    [("frozen", "development"), ("evaluate", "development"), ("train-eval", "final"), ("train-eval", "regression")],
)
def test_target_cannot_select_using_final_or_regression_outcomes(tmp_path, action, split):
    args, _, experiment = setup_target(tmp_path)
    args.action, args.split = action, split
    with pytest.raises(ValueError, match="fresh development"):
        run.read_validation_target(args.validation_target, args, experiment)


def write_scores(directory, checkpoint, target, *, ce=0.2):
    directory.mkdir()
    rows = fixtures()[0]["trained_nf4"]
    for row in rows:
        row["native_sft_ce"] = ce
        row["raw_row_sha256"] = digest_json({"id": row["id"]})
    (directory / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / "selected_rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    selection = {
        "source_sha256": target["score_data_sha256"],
        "split_role": "development",
        "split_manifest_sha256": target["split_manifest_sha256"],
        "selected_rows_file_sha256": file_sha256(directory / "selected_rows.jsonl"),
    }
    (directory / "selection.json").write_text(json.dumps(selection))
    summary = {
        "complete": True,
        "checkpoint": str(checkpoint.resolve()),
        "selection_sha256": digest_json(selection),
        "scores_sha256": file_sha256(directory / "scores.jsonl"),
        "metrics": {
            "rows": len(rows),
            "restricted_choice_accuracy": 1.0,
            "token_weighted_ce": math.fsum(row["native_sft_ce"] * row["native_supervised_tokens"] for row in rows)
            / sum(row["native_supervised_tokens"] for row in rows),
        },
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    return rows, summary


def write_boundary(output, checkpoint, *, start, stop, horizon):
    checkpoint.mkdir(parents=True, exist_ok=True)
    rows = [{"step": step, "train/loss_avg": 0.2} for step in range(1, stop + 1)]
    (checkpoint.parent.parent / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {"start": start, "final_step": stop, "final_loss": 0.2}
    if stop < horizon:
        summary.update(paused=True, checkpoint=str(checkpoint))
    log = output / "stages" / f"training_{stop:06d}" / "output.log"
    log.parent.mkdir(parents=True)
    log.write_text(f"[rank0] SFT DONE {summary}\n")
    return summary, log


@pytest.mark.parametrize(
    "damage",
    [
        "extra_update",
        "missing_update",
        "duplicate_update",
        "future_checkpoint",
        "missing_pause",
        "wrong_start",
        "preempted",
        "foreign_checkpoint",
        "two_summaries",
    ],
)
def test_boundary_does_not_accept_saved_checkpoint_from_an_overrun(tmp_path, damage):
    checkpoint = tmp_path / "native/checkpoints/iter_000002"
    summary, log = write_boundary(tmp_path, checkpoint, start=0, stop=2, horizon=4)
    metrics = checkpoint.parent.parent / "metrics.jsonl"
    if damage in {"extra_update", "missing_update", "duplicate_update"}:
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
        if damage == "extra_update":
            rows.append({"step": 3, "train/loss_avg": 0.2})
        elif damage == "missing_update":
            rows.pop(0)
        else:
            rows.append(rows[-1])
        metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif damage == "future_checkpoint":
        (checkpoint.parent / "iter_000004").mkdir()
    elif damage == "missing_pause":
        summary.pop("paused")
    elif damage == "wrong_start":
        summary["start"] = 1
    elif damage == "preempted":
        summary["preempted"] = "signal"
    elif damage == "foreign_checkpoint":
        summary["checkpoint"] = "/elsewhere/iter_000002"
    log.write_text(f"SFT DONE {summary}\n" * (2 if damage == "two_summaries" else 1))
    with pytest.raises(ValueError):
        run.verify_segment_boundary(tmp_path, checkpoint, start=0, stop=2, horizon=4)
    assert not (tmp_path / "boundary-000002.json").exists()


@pytest.mark.parametrize("damage", ["foreign_checkpoint", "missing_rows", "changed_bytes", "summary", "inputs"])
def test_crossing_requires_owned_complete_scores_and_unchanged_inputs(tmp_path, damage):
    args, target, _ = setup_target(tmp_path)
    checkpoint, directory = tmp_path / "iter_000100", tmp_path / "scores"
    rows, summary = write_scores(directory, checkpoint, target)
    baseline = run.target_observation(directory, checkpoint, target, None, manifest_path=Path(args.split_manifest))
    expected = baseline["processed_inputs_sha256"]
    if damage == "foreign_checkpoint":
        summary["checkpoint"] = "/foreign/iter_000100"
    elif damage == "missing_rows":
        rows.pop()
    elif damage == "changed_bytes":
        (directory / "scores.jsonl").write_text((directory / "scores.jsonl").read_text() + "\n")
    elif damage == "summary":
        summary["metrics"]["token_weighted_ce"] = 0.0
    else:
        expected = "foreign processed inputs"
    if damage == "missing_rows":
        (directory / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        summary["scores_sha256"] = file_sha256(directory / "scores.jsonl")
    (directory / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        run.target_observation(directory, checkpoint, target, expected, manifest_path=Path(args.split_manifest))


@pytest.mark.parametrize("pass_at", [100, 200, None])
def test_native_segments_preserve_horizon_and_include_all_prior_scoring_cost(tmp_path, monkeypatch, pass_at):
    args, target, experiment = setup_target(tmp_path)
    time_state, commands = [10.0], []
    monkeypatch.setattr(run.time, "monotonic", lambda: time_state[0])
    monkeypatch.setattr(
        run,
        "final_checkpoint",
        lambda output, exp, receipt, *, step: output / "native" / "checkpoints" / f"iter_{step:06d}",
    )

    class Stages:
        def run(self, name, command):
            commands.append((name, command))
            step = int(name.rsplit("_", 1)[-1])
            if name.startswith("training"):
                time_state[0] += 100.0
                write_boundary(
                    tmp_path,
                    tmp_path / "native/checkpoints" / f"iter_{step:06d}",
                    start=step - 100,
                    stop=step,
                    horizon=300,
                )
                assert "rl.max_steps=300" in command and "rl.lr_schedule.cycle_steps=300" in command
                assert "rl.optimizer.warmup_steps=30" in command
                assert (f"rl.pause_after_step={step}" in command) == (step < 300)
                if step == 100:
                    assert 'rl.resume="off"' in command
                else:
                    assert any(c.startswith('rl.resume="') and f"iter_{step - 100:06d}" in c for c in command)
            else:
                time_state[0] += 40.0
                checkpoint = Path(command[command.index("--checkpoint") + 1])
                directory = Path(command[command.index("--output") + 1])
                write_scores(directory, checkpoint, target, ce=0.2 if step == pass_at else 0.4)

    checkpoint, directory, result = run.train_to_target(args, experiment, tmp_path, {}, Stages(), target, 0.0)
    stop = pass_at or 300
    assert checkpoint.name == f"iter_{stop:06d}" and directory.name == f"scores_{stop:06d}"
    assert [name for name, _ in commands] == [
        name for s in range(100, stop + 1, 100) for name in (f"training_{s:06d}", f"scoring_{s:06d}")
    ]
    assert result["observations"][-1]["elapsed_from_runner_start_seconds"] == 10.0 + 140.0 * (stop // 100)
    assert result["target_met"] == (pass_at is not None)
    assert result["censored_at_training_horizon"] == (pass_at is None)
    assert result["first_validated_checkpoint_seconds"] == (None if pass_at is None else time_state[0])
    assert result["quality_qualified"] is False


def test_target_mutation_between_segments_refused_before_next_native_launch(tmp_path, monkeypatch):
    args, target, experiment = setup_target(tmp_path)
    changed = copy.deepcopy(target)
    changed["maximum_token_ce"] = 99
    Path(args.validation_target).write_text(json.dumps(changed))

    class Never:
        def run(self, *args):
            pytest.fail("launched after target mutation")

    with pytest.raises(ValueError, match="changed after"):
        run.train_to_target(args, experiment, tmp_path, {}, Never(), target, 0.0)


@pytest.mark.parametrize("phase", ["training", "scoring"])
def test_mutation_during_segment_refuses_first_crossing(tmp_path, monkeypatch, phase):
    args, target, experiment = setup_target(tmp_path)
    monkeypatch.setattr(run, "final_checkpoint", lambda *args, step: tmp_path / f"iter_{step:06d}")
    monkeypatch.setattr(run, "verify_segment_boundary", lambda *args, **kwargs: {})

    class ChangesInput:
        def run(self, name, command):
            if name.startswith(phase):
                Path(args.score_data).write_text("changed during work")
            elif name.startswith("scoring"):
                pytest.fail("launched scoring after native changed the input")

    with pytest.raises(ValueError, match="sealed score_data"):
        run.train_to_target(args, experiment, tmp_path, {}, ChangesInput(), target, 0.0)
    assert not (tmp_path / "target-result.json").exists()


def test_complete_self_consistent_foreign_first_selection_is_rejected(tmp_path):
    args, target, _ = setup_target(tmp_path)
    checkpoint, directory = tmp_path / "iter_000100", tmp_path / "scores"
    _, summary = write_scores(directory, checkpoint, target)
    selection = json.loads((directory / "selection.json").read_text())
    selection["source_sha256"] = "another source"
    (directory / "selection.json").write_text(json.dumps(selection))
    summary["selection_sha256"] = digest_json(selection)
    (directory / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="actual scoring selection"):
        run.target_observation(directory, checkpoint, target, None, manifest_path=Path(args.split_manifest))
