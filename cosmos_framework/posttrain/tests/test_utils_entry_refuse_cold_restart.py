# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""No silent cold restart over committed model-only artifacts.

A requeued ``rl.resume=auto`` run whose saves were ``rl.checkpoint_components=[model]`` used to find no
checkpoint, restart at step 0 inside the same attempt directory and die at its first save on the committed
artifact it collided with. These tests pin the outcomes of `pin_run_attempt` over fake attempt directories:
refusal over a committed artifact, normal resume when a full checkpoint at or after the newest artifact exists
beside it, refusal of a full checkpoint behind the newest artifact (resuming it would collide one save later),
an unaffected fresh run, no refusal over an interrupted (manifest-less) artifact directory, and a legacy resume
path without an ``iter_*`` segment whose iteration is taken from its validated trainer state: at or beyond the
newest artifact it resumes, behind it or with no derivable iteration it is refused, and without committed
artifacts it resumes exactly as before.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.posttrain.checkpoint import committed_artifact_iterations
from cosmos_framework.posttrain.checkpoint.artifact import (
    CHECKPOINT_ARTIFACT_MANIFEST,
    CHECKPOINT_ARTIFACTS_DIR,
    build_artifact_manifest,
    read_local_artifact_manifest,
)
from cosmos_framework.posttrain.checkpoint.io import LATEST_FILE, atomic_write_json, is_complete_iter_dir
from cosmos_framework.posttrain.utils.entry import RunAttempt, pin_run_attempt

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

RUN_TS = "20260916091100"


def _commit_artifact(attempt: Path, iteration: int) -> Path:
    """A committed model-only artifact exactly as the trainer lays it out: payload, then the manifest last."""
    artifact = attempt / CHECKPOINT_ARTIFACTS_DIR / f"iter_{iteration:06d}"
    artifact.mkdir(parents=True)
    payload = artifact / "model.pt"
    payload.write_bytes(b"weights")
    descriptor = {"kind": "file", "path": "model.pt", "size": payload.stat().st_size}
    manifest = build_artifact_manifest(str(artifact), "consolidated", ("model",), {"model": descriptor})
    atomic_write_json(str(artifact / CHECKPOINT_ARTIFACT_MANIFEST), manifest)
    assert read_local_artifact_manifest(str(artifact))["iteration"] == iteration
    return artifact


def _interrupted_artifact(attempt: Path, iteration: int) -> Path:
    """An artifact directory whose save never reached the commit-last manifest."""
    artifact = attempt / CHECKPOINT_ARTIFACTS_DIR / f"iter_{iteration:06d}"
    artifact.mkdir(parents=True)
    (artifact / "model.pt").write_bytes(b"partial")
    assert not (artifact / CHECKPOINT_ARTIFACT_MANIFEST).exists()
    return artifact


def _full_checkpoint(attempt: Path, iteration: int) -> Path:
    """A minimally resumable full checkpoint under ``checkpoints/`` with the latest pointer advanced."""
    checkpoint = attempt / "checkpoints" / f"iter_{iteration:06d}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.pt").write_bytes(b"weights")
    (checkpoint / "trainer.pt").write_bytes(b"clock")
    (attempt / "checkpoints" / LATEST_FILE).write_text(checkpoint.name + "\n")
    assert is_complete_iter_dir(str(checkpoint))
    return checkpoint


def _legacy_directory(attempt: Path, trainer_state: dict | None) -> Path:
    """A resumable checkpoint directory without an ``iter_*`` name; its clock lives only in ``trainer.pt``."""
    checkpoint = attempt / "checkpoints" / "legacy"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.pt").write_bytes(b"weights")
    if trainer_state is not None:
        torch.save(trainer_state, checkpoint / "trainer.pt")
    return checkpoint


def _legacy_file(attempt: Path, clocks: dict) -> Path:
    """A legacy single-file checkpoint: model, optimizer and the clocks at the top level of one saved mapping."""
    checkpoint = attempt / "checkpoints" / "legacy.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {"w": torch.zeros(1)}, "opt": {}, **clocks}, checkpoint)
    return checkpoint


def _model_only_config(resume: str) -> SimpleNamespace:
    return SimpleNamespace(
        job=SimpleNamespace(project="project", group="group", name="name"),
        rl=SimpleNamespace(run_ts="", resume=resume, checkpoint_components=["model"]),
    )


def _pin(config: SimpleNamespace, name_dir: Path) -> RunAttempt:
    """`pin_run_attempt` through the real checkpoint lifecycle; a model-only plan announces itself once."""
    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        return pin_run_attempt(config, [], str(name_dir))


def test_committed_artifact_iterations_counts_manifests_only(tmp_path: Path) -> None:
    attempt = tmp_path / RUN_TS
    assert committed_artifact_iterations(str(attempt)) == [], "no checkpoint_artifacts/ directory"
    _commit_artifact(attempt, 200)
    _commit_artifact(attempt, 100)
    _interrupted_artifact(attempt, 300)
    (attempt / CHECKPOINT_ARTIFACTS_DIR / "notes.txt").write_text("not an iteration\n")
    assert committed_artifact_iterations(str(attempt)) == ["iter_000100", "iter_000200"]


def test_resume_auto_refuses_cold_restart_over_committed_model_only_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    name_dir = tmp_path / "project" / "group" / "name"
    attempt = name_dir / RUN_TS
    _commit_artifact(attempt, 100)
    _commit_artifact(attempt, 200)
    assert not (attempt / "checkpoints").exists(), "a model-only plan never creates the resumable root"

    config = _model_only_config("auto")
    with pytest.raises(RuntimeError) as info:
        _pin(config, name_dir)

    message = str(info.value)
    assert config.rl.run_ts == RUN_TS, "the refusal names the attempt directory resume='auto' adopted"
    assert repr(str(attempt)) in message and "2 committed" in message and "newest iter_000200" in message
    assert "refusing to restart from step 0" in message
    for remedy in ("new job.name", f"rl.init_from={attempt / CHECKPOINT_ARTIFACTS_DIR / 'iter_000200'}"):
        assert remedy.lower() in message.lower(), remedy
    assert f"slice-run --run-dir {attempt}" in message and "rl.resume=off" in message
    assert config.rl.resume == "auto", "the config is left untouched; nothing is pinned for the workers"


def test_resume_auto_prefers_a_full_checkpoint_beside_committed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    name_dir = tmp_path / "project" / "group" / "name"
    attempt = name_dir / RUN_TS
    _commit_artifact(attempt, 100)
    checkpoint = _full_checkpoint(attempt, 150)

    config = _model_only_config("auto")
    pinned = _pin(config, name_dir)

    assert pinned.out_dir == str(attempt)
    assert pinned.resume_checkpoint == str(checkpoint)
    assert config.rl.resume == str(checkpoint)
    assert pinned.opts == (f"rl.run_ts={RUN_TS}", f"rl.resume={checkpoint}")


def test_resume_auto_refuses_a_full_checkpoint_older_than_the_newest_committed_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    name_dir = tmp_path / "project" / "group" / "name"
    attempt = name_dir / RUN_TS
    checkpoint = _full_checkpoint(attempt, 150)
    _commit_artifact(attempt, 200)

    config = _model_only_config("auto")
    with pytest.raises(RuntimeError) as info:
        _pin(config, name_dir)

    message = str(info.value)
    assert repr(str(checkpoint)) in message and "iter_000150" in message and "newest iter_000200" in message
    assert "iteration 150" in message and "iteration 200" in message
    assert "refusing to resume behind it" in message
    for remedy in ("new job.name", f"rl.init_from={attempt / CHECKPOINT_ARTIFACTS_DIR / 'iter_000200'}"):
        assert remedy.lower() in message.lower(), remedy
    assert f"slice-run --run-dir {attempt}" in message and "rl.resume=off" in message
    assert config.rl.resume == "auto", "the config is left untouched; nothing is pinned for the workers"


def test_resume_auto_allows_a_full_checkpoint_at_or_after_the_newest_committed_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    name_dir = tmp_path / "project" / "group" / "name"
    attempt = name_dir / RUN_TS
    _commit_artifact(attempt, 100)
    _commit_artifact(attempt, 200)
    checkpoint = _full_checkpoint(attempt, 200)

    config = _model_only_config("auto")
    pinned = _pin(config, name_dir)

    assert pinned.out_dir == str(attempt)
    assert pinned.resume_checkpoint == str(checkpoint)
    assert config.rl.resume == str(checkpoint)
    assert pinned.opts == (f"rl.run_ts={RUN_TS}", f"rl.resume={checkpoint}")


@pytest.mark.parametrize(
    ("layout", "reason"),
    [
        pytest.param(("file", {}), "records neither optimizer_step nor iter", id="file-without-clocks"),
        pytest.param(("directory", None), "holds no trainer.pt", id="directory-without-trainer-state"),
        pytest.param(
            ("directory", {"iter": 200, "optimizer_step": 199}), "optimizer_step 199 but iter 200", id="clocks-disagree"
        ),
        pytest.param(("directory", {"iter": "200"}), "invalid iter: '200'", id="clock-not-an-int"),
        pytest.param(("file", {"optimizer_step": -1}), "invalid optimizer_step: -1", id="clock-negative"),
    ],
)
def test_explicit_legacy_resume_without_a_derivable_iteration_is_refused_over_committed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: tuple, reason: str
) -> None:
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    name_dir = tmp_path / "project" / "group" / "name"
    attempt = name_dir / RUN_TS
    _commit_artifact(attempt, 200)
    kind, state = layout
    legacy = _legacy_file(attempt, state) if kind == "file" else _legacy_directory(attempt, state)

    config = _model_only_config(str(legacy))
    with pytest.raises(RuntimeError) as info:
        _pin(config, name_dir)

    message = str(info.value)
    assert repr(str(legacy)) in message and "no iter_* segment" in message and "newest iter_000200" in message
    assert "could not be established" in message and reason in message, message
    assert "refusing to resume a checkpoint that cannot be proven to be at or beyond iter_000200" in message
    for remedy in ("new job.name", f"rl.init_from={attempt / CHECKPOINT_ARTIFACTS_DIR / 'iter_000200'}"):
        assert remedy.lower() in message.lower(), remedy
    assert f"slice-run --run-dir {attempt}" in message and "rl.resume=off" in message
    assert config.rl.resume == str(legacy), "the config is left untouched; nothing is pinned for the workers"


@pytest.mark.parametrize(
    ("kind", "iteration"),
    [
        pytest.param("directory", 200, id="directory-at-the-artifact"),
        pytest.param("file", 250, id="file-beyond-the-artifact"),
    ],
)
def test_explicit_legacy_resume_derives_its_iteration_from_trainer_state_and_resumes_at_or_beyond_the_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, iteration: int
) -> None:
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    name_dir = tmp_path / "project" / "group" / "name"
    attempt = name_dir / RUN_TS
    _commit_artifact(attempt, 100)
    _commit_artifact(attempt, 200)
    clocks = {"iter": iteration, "optimizer_step": iteration}
    legacy = _legacy_file(attempt, clocks) if kind == "file" else _legacy_directory(attempt, clocks)

    config = _model_only_config(str(legacy))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pinned = pin_run_attempt(config, [], str(name_dir))

    messages = [str(warning.message) for warning in caught]
    assert any("partial checkpoint artifact" in text for text in messages), messages
    assert not any("no iter_* segment" in text for text in messages), (
        "the comparison ran on the trainer state; nothing was skipped",
        messages,
    )
    assert pinned.out_dir == str(attempt)
    assert pinned.resume_checkpoint == str(legacy) and config.rl.resume == str(legacy)
    assert pinned.opts == (f"rl.run_ts={RUN_TS}", f"rl.resume={legacy}")


def test_explicit_legacy_resume_behind_the_newest_committed_artifact_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    name_dir = tmp_path / "project" / "group" / "name"
    attempt = name_dir / RUN_TS
    legacy = _legacy_directory(attempt, {"iter": 150, "optimizer_step": 150})
    _commit_artifact(attempt, 200)

    config = _model_only_config(str(legacy))
    with pytest.raises(RuntimeError) as info:
        _pin(config, name_dir)

    message = str(info.value)
    assert repr(str(legacy)) in message and "iteration 150, from its trainer state" in message
    assert "newest iter_000200" in message and "iteration 200" in message
    assert "refusing to resume behind it" in message
    for remedy in ("new job.name", f"rl.init_from={attempt / CHECKPOINT_ARTIFACTS_DIR / 'iter_000200'}"):
        assert remedy.lower() in message.lower(), remedy
    assert f"slice-run --run-dir {attempt}" in message and "rl.resume=off" in message
    assert config.rl.resume == str(legacy), "the config is left untouched; nothing is pinned for the workers"


def test_explicit_legacy_resume_without_committed_artifacts_resumes_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    name_dir = tmp_path / "project" / "group" / "name"
    attempt = name_dir / RUN_TS
    legacy = _legacy_file(attempt, {})
    assert committed_artifact_iterations(str(attempt)) == []

    config = _model_only_config(str(legacy))
    pinned = _pin(config, name_dir)

    assert pinned.out_dir == str(attempt)
    assert pinned.resume_checkpoint == str(legacy) and config.rl.resume == str(legacy)
    assert pinned.opts == (f"rl.run_ts={RUN_TS}", f"rl.resume={legacy}")


def test_resume_auto_fresh_run_without_artifacts_is_unaffected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    name_dir = tmp_path / "project" / "group" / "name"

    config = _model_only_config("auto")
    pinned = _pin(config, name_dir)

    minted = os.path.basename(pinned.out_dir)
    assert re.fullmatch(r"\d{14}", minted) and os.path.dirname(pinned.out_dir) == str(name_dir)
    assert config.rl.resume == "off" and pinned.resume_checkpoint is None
    assert pinned.opts == (f"rl.run_ts={minted}", "rl.resume=off")

    # the same command re-executed over an attempt that saved nothing yet adopts it and starts fresh, as before
    (name_dir / RUN_TS).mkdir(parents=True, exist_ok=True)
    config = _model_only_config("auto")
    pinned = _pin(config, name_dir)
    assert pinned.out_dir == str(name_dir / RUN_TS)
    assert pinned.opts == (f"rl.run_ts={RUN_TS}", "rl.resume=off")


def test_resume_auto_ignores_an_interrupted_artifact_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    name_dir = tmp_path / "project" / "group" / "name"
    attempt = name_dir / RUN_TS
    _interrupted_artifact(attempt, 100)
    assert committed_artifact_iterations(str(attempt)) == []

    config = _model_only_config("auto")
    pinned = _pin(config, name_dir)

    assert pinned.out_dir == str(attempt)
    assert config.rl.resume == "off" and pinned.resume_checkpoint is None
    assert pinned.opts == (f"rl.run_ts={RUN_TS}", "rl.resume=off")


def test_explicit_resume_off_mints_a_fresh_attempt_beside_committed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    name_dir = tmp_path / "project" / "group" / "name"
    _commit_artifact(name_dir / RUN_TS, 100)

    config = _model_only_config("off")
    pinned = _pin(config, name_dir)

    minted = os.path.basename(pinned.out_dir)
    assert re.fullmatch(r"\d{14}", minted) and minted != RUN_TS, "resume=off never adopts; it mints a new timestamp"
    assert os.path.dirname(pinned.out_dir) == str(name_dir)
    assert config.rl.resume == "off" and pinned.resume_checkpoint is None
    assert pinned.opts == (f"rl.run_ts={minted}", "rl.resume=off")
