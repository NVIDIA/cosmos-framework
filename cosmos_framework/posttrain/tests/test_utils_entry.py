# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0 CPU tests for entrypoint run-attempt and checkpoint-lifecycle resolution."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cosmos_framework.posttrain.utils.entry import pin_run_attempt


def _pin_run_attempt_asymmetry_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    divergence: str,
) -> None:
    import torch.distributed as dist

    from cosmos_framework.posttrain.control import checkpointing

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=10),
    )
    original_lifecycle = checkpointing.CheckpointLifecycle
    original_isdir = os.path.isdir
    original_realpath = os.path.realpath
    resume_checkpoint = "/external/run/iter_000007"

    class _Lifecycle:
        @classmethod
        def from_config(cls, _config: Any, _rl: Any, **_kwargs: Any) -> _Lifecycle:
            return cls()

        def resolve_resume(self, _requested: str) -> str:
            return resume_checkpoint

    try:
        setattr(checkpointing, "CheckpointLifecycle", _Lifecycle)
        if divergence == "directory":
            os.path.isdir = lambda _path: rank == 0
            expected = "external checkpoint directory selection differs across ranks"
        elif divergence == "successor":
            os.path.isdir = lambda _path: True
            os.path.realpath = lambda path: "same" if rank == 0 else original_realpath(path)
            expected = "external checkpoint successor selection differs across ranks"
        else:
            raise AssertionError(f"unknown divergence mode {divergence!r}")
        config = SimpleNamespace(
            job=SimpleNamespace(project="project", group="group", name="name"),
            rl=SimpleNamespace(run_ts="20260901120000", resume="auto"),
        )
        error_text: str | None = None
        try:
            pin_run_attempt(config, [], "/runs/project/group/name")
        except RuntimeError as error:
            error_text = str(error)
        errors: list[str | None] = [None] * world_size
        dist.all_gather_object(errors, error_text)
        assert all(error is not None and expected in error for error in errors)
    finally:
        setattr(checkpointing, "CheckpointLifecycle", original_lifecycle)
        os.path.isdir = original_isdir
        os.path.realpath = original_realpath
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "resume,resolved",
    [
        ("auto", "s3://direct-checkpoints/run/iter_000007"),
        ("off", None),
        ("s3://direct-checkpoints/other/iter_000004", "s3://direct-checkpoints/other/iter_000004"),
    ],
)
def test_pin_run_attempt_resolves_resume_through_checkpoint_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resume: str,
    resolved: str | None,
) -> None:
    from cosmos_framework.posttrain.control import checkpointing

    calls: list[tuple[str, str]] = []
    lifecycle_kwargs: list[dict[str, object]] = []

    class _Lifecycle:
        def resolve_resume(self, requested: str) -> str | None:
            calls.append((requested, "lifecycle"))
            return resolved

    lifecycle = _Lifecycle()
    monkeypatch.setattr(
        checkpointing.CheckpointLifecycle,
        "from_config",
        classmethod(
            lambda _cls, _config, _rl, **kwargs: (
                lifecycle_kwargs.append(dict(kwargs)),
                lifecycle,
            )[1]
        ),
    )
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    run_ts = "20260901120000"
    config = SimpleNamespace(
        job=SimpleNamespace(project="project", group="group", name="name"),
        rl=SimpleNamespace(
            run_ts=run_ts,
            resume=resume,
        ),
    )
    name_dir = tmp_path / "project" / "group" / "name"

    attempt = pin_run_attempt(config, [], str(name_dir))

    expected_out = name_dir / run_ts
    assert calls == [(resume, "lifecycle")]
    assert lifecycle_kwargs == [{"run_out_dir": str(expected_out)}]
    assert attempt.out_dir == str(expected_out)
    assert attempt.resume_checkpoint == resolved
    assert config.rl.resume == (resolved or "off")
    assert attempt.opts == (
        f"rl.run_ts={run_ts}",
        f"rl.resume={resolved or 'off'}",
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("divergence", ["directory", "successor"])
def test_pin_run_attempt_converges_rank_asymmetric_local_path_selection(
    tmp_path: Path,
    divergence: str,
) -> None:
    import torch.multiprocessing as mp

    mp.spawn(
        _pin_run_attempt_asymmetry_worker,
        args=(2, str(tmp_path / f"pin-run-attempt-{divergence}"), divergence),
        nprocs=2,
        join=True,
    )
