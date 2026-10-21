# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit trial inventories, durable evaluation progress and independent comparisons."""

import json
from pathlib import Path

import pytest

from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
    LIBERO_PROTOCOLS,
    LiberoCase,
    LiberoTrial,
    LiberoTrialManifest,
)
from cosmos_framework.posttrain.tests.libero_assessment_cli import (
    assessment_metric_frame,
    assessment_metrics,
    committed_trials,
    compare_assessments,
)

pytestmark = [pytest.mark.L1, pytest.mark.CPU]


def inventory() -> LiberoTrialManifest:
    return LiberoTrialManifest(
        "libero_10",
        tuple(
            LiberoTrial(LiberoCase(task, state, "libero_10"), seed)
            for task in (1, 7)
            for state in range(12)
            for seed in (0, 3, 11)
        ),
    )


def record(trial: LiberoTrial, version: int, success: bool) -> dict:
    protocol = LIBERO_PROTOCOLS[trial.case.suite]
    steps = protocol.execution_horizon if success else protocol.max_episode_steps
    return {
        "trial_id": trial.id,
        "suite": trial.case.suite,
        "task_id": trial.case.task_id,
        "init_state_id": trial.case.init_state_id,
        "seed": trial.seed,
        "publication_version": version,
        "success": success,
        "steps": steps,
        "return": float(success),
        "control_rewards": [0.0] * (steps - 1) + [float(success)],
    }


def commit(directory: Path, trial: LiberoTrial, row: dict) -> Path:
    parent = directory / "episodes" / f"seed-{trial.seed}"
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f"task-{trial.case.task_id:02d}-state-{trial.case.init_state_id:02d}.json"
    path.write_text(json.dumps(row))
    return path


def test_unexpected_committed_json_cannot_hide_outside_template_glob(tmp_path: Path) -> None:
    manifest = inventory()
    path = tmp_path / "episodes" / "unexpected.json"
    path.parent.mkdir()
    path.write_text(json.dumps(record(manifest.trials[0], 3, True)))
    with pytest.raises(ValueError, match="record path"):
        committed_trials(tmp_path, 3, manifest)


def test_trial_commit_rejects_corruption_and_ignores_pending(tmp_path: Path) -> None:
    from cosmos_framework.posttrain.utils.report import validate_metric_frame

    manifest = inventory()
    trial = manifest.trials[0]
    row = record(trial, 3, True)
    validate_metric_frame(assessment_metric_frame([row], 3, complete=False, inventory=manifest))
    path = commit(tmp_path, trial, row)
    path.with_suffix(".pending-123").write_text("interrupted write")
    assert committed_trials(tmp_path, 3, manifest) == [row]
    with pytest.raises(ValueError, match="unique trials"):
        assessment_metrics([row], manifest)
    for key, bad, match in (
        ("publication_version", 4, "version"),
        ("publication_version", False, "version"),
        ("steps", 721, "length"),
        ("success", False, "length"),
        ("control_rewards", [0.0], "raw rewards"),
        ("return", float("nan"), "return"),
        ("trial_id", "wrong", "trial"),
    ):
        path.write_text(json.dumps({**row, key: bad}))
        with pytest.raises(ValueError, match=match):
            committed_trials(tmp_path, 3, manifest)
    path.write_text(json.dumps(row))
    path.with_name("task-duplicate-state-duplicate.json").write_text(json.dumps(row))
    with pytest.raises(ValueError, match="duplicate"):
        committed_trials(tmp_path, 3, manifest)


def test_comparison_uses_explicit_steps_inventory_and_state_clusters(tmp_path: Path) -> None:
    manifest = inventory()
    paths = []
    for name, step, version in (("base", 12, 3), ("candidate", 28, 7)):
        path = tmp_path / f"{name}.json"
        episodes = [
            record(trial, version, name == "candidate" and trial.case.init_state_id < 6) for trial in manifest.trials
        ]
        path.write_text(
            json.dumps(
                {
                    "optimizer_step": step,
                    "publication_version": version,
                    "trial_manifest": manifest.payload(),
                    "evaluation": {"trials": manifest.digest, "suite": manifest.suite, "mode": "native_ode"},
                    "episodes": episodes,
                }
            )
        )
        paths.append(path)
    report = compare_assessments(*paths, tmp_path / "report")
    assert report["trials_per_model"] == len(manifest.trials)
    assert report["baseline"]["optimizer_step"] == 12 and report["candidate"]["optimizer_step"] == 28
    assert report["success_rate_delta"] == 0.5 and len(report["paired_gains"]) == len(manifest.trials) // 2
    low, high = report["paired_ci95"]
    assert low < 0.5 < high and high - low > 0.25  # correlated seeds stay in the same state cluster
    data = json.loads(paths[1].read_text())
    data["evaluation"]["mode"] = "ode"
    paths[1].write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identical evaluation protocols"):
        compare_assessments(*paths, tmp_path / "bad-report")


def test_explicit_subset_weights_tasks_equally_with_unequal_trial_counts() -> None:
    manifest = LiberoTrialManifest(
        "libero_10",
        (
            LiberoTrial(LiberoCase(1, 0, "libero_10"), 0),
            *(LiberoTrial(LiberoCase(7, state, "libero_10"), 11) for state in range(3)),
        ),
    )
    rows = [record(trial, 3, trial.case.task_id == 1) for trial in manifest.trials]
    metrics = assessment_metrics(rows, manifest)
    assert metrics["val/success_rate"] == 0.5
    assert metrics["val/success_rate"] != sum(row["success"] for row in rows) / len(rows)
    assert metrics["val/episode_count"] == 4 and metrics["val/task_count"] == 2


def test_control_commits_each_trial_and_resumes_only_unfinished_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import signal
    from dataclasses import replace

    from cosmos_framework.posttrain.config.rl import JointPPOConfig
    from cosmos_framework.posttrain.control.termination import TerminationWatch
    from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer
    from cosmos_framework.posttrain.tests import libero_assessment_cli as libero_assessment
    from cosmos_framework.posttrain.tests.helpers.ppo_continuous import (
        EvaluationIntegratorGateway,
        GaussianPolicy,
        make_collector,
    )

    recipe = JointPPOConfig()
    model = GaussianPolicy()
    model.config = replace(model.config, execution_horizon=8)
    learner = JointPPOTrainer(model, recipe)
    gateway = EvaluationIntegratorGateway()
    collector = make_collector(model, gateway, recipe, 0, 1)
    source = LiberoTrialManifest("libero_10", inventory().trials[:3])
    before = learner.evaluation_state(), collector.state_dict()
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR1, signal.SIGTERM)}
    sentinel = tmp_path / "stop"
    watch = TerminationWatch(str(sentinel))
    original_write = libero_assessment.atomic_write_json

    def interrupt_after_commit(path, value):
        original_write(path, value)
        sentinel.touch()

    try:
        with monkeypatch.context() as patch:
            patch.setattr(libero_assessment, "atomic_write_json", interrupt_after_commit)
            assert libero_assessment.assess_pending_trials(
                collector.collector,
                collector.simulator,
                source,
                version=0,
                replicas=1,
                directory=tmp_path,
                watch=watch,
                decisions=collector.environment.evaluation_decisions,
            )
        assert len(committed_trials(tmp_path, 0, source)) == len(gateway.evaluated) == 1
        resumed_watch = TerminationWatch(str(tmp_path / "next-job-stop"))
        assert not libero_assessment.assess_pending_trials(
            collector.collector,
            collector.simulator,
            source,
            version=0,
            replicas=1,
            directory=tmp_path,
            watch=resumed_watch,
            decisions=collector.environment.evaluation_decisions,
        )
        assert len(committed_trials(tmp_path, 0, source)) == len(gateway.evaluated) == len(source.trials)
        assert (learner.evaluation_state(), collector.state_dict()) == before
    finally:
        collector.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
