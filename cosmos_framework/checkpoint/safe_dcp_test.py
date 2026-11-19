# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import builtins
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException

from cosmos_framework.checkpoint.safe_dcp import SafeFileSystemReader, SafeLoadPlanner
from cosmos_framework.utils.functional.lr_scheduler import LambdaWarmUpCosineScheduler, WSDScheduler

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


class _CallOnLoad:
    def __init__(self, func: Any, *args: Any) -> None:
        self.func = func
        self.args = args

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return self.func, self.args


def _marker_payload(marker: Path) -> _CallOnLoad:
    return _CallOnLoad(builtins.exec, f"open({str(marker)!r}, 'w').write('loaded')")


def _scheduler(schedule: Any) -> torch.optim.lr_scheduler.LambdaLR:
    optimizer = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=1.0)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _rejection(excinfo: pytest.ExceptionInfo[Exception]) -> str:
    """Message of the rejection, unwrapping the ``CheckpointException`` that ``dcp.load`` reports it in."""
    error = excinfo.value
    if isinstance(error, CheckpointException):
        error = next(iter(error.failures.values()))[0]
    assert isinstance(error, ValueError)
    return str(error)


def _load(path: Path, template: dict[str, Any]) -> dict[str, Any]:
    dcp.load(template, storage_reader=SafeFileSystemReader(path), planner=SafeLoadPlanner(), no_dist=True)
    return template


@pytest.mark.L0
@pytest.mark.CPU
def test_round_trips_tensors_and_plain_values(tmp_path: Path) -> None:
    state = {
        "weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),  # [2,3]
        "iteration": 7,
        "grad_scaler": {"scale": 65536.0, "_growth_tracker": 3},
        "meta": {"name": "run", "flags": [True, None], "betas": (0.9, 0.99)},
    }
    dcp.save(state, checkpoint_id=tmp_path, no_dist=True)

    template = {
        "weight": torch.zeros(2, 3, dtype=torch.bfloat16),  # [2,3]
        "iteration": 0,
        "grad_scaler": {"scale": 0.0, "_growth_tracker": 0},
        "meta": {"name": None, "flags": None, "betas": None},
    }
    loaded = _load(tmp_path, template)

    torch.testing.assert_close(loaded["weight"], state["weight"], rtol=0, atol=0)
    assert loaded["iteration"] == 7
    assert loaded["grad_scaler"] == state["grad_scaler"]
    assert loaded["meta"] == state["meta"]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("make_schedule", "steps"),
    [
        pytest.param(
            lambda: LambdaWarmUpCosineScheduler(
                warm_up_steps=[2, 2], f_min=[0.1, 0.1], f_max=[1.0, 1.0], f_start=[0.01, 0.01], cycle_lengths=[5, 5]
            ),
            7,
            id="warmup-cosine",
        ),
        pytest.param(
            lambda: WSDScheduler(warm_up_steps=2, total_steps=20, decay_steps=5, f_start=0.01, f_max=1.0, f_min=0.1),
            18,  # into the decay phase, where the multiplier is an np.float64
            id="wsd",
        ),
    ],
)
def test_round_trips_scheduler_state_with_numpy_values(make_schedule: Any, steps: int, tmp_path: Path) -> None:
    scheduler = _scheduler(make_schedule())
    for _ in range(steps):
        scheduler.optimizer.step()
        scheduler.step()
    state = scheduler.state_dict()
    assert any(isinstance(value, np.ndarray | np.generic) for value in state["lr_lambdas"][0].values())
    dcp.save(state, checkpoint_id=tmp_path, no_dist=True)

    loaded = _load(tmp_path, _scheduler(make_schedule()).state_dict())

    assert repr(loaded) == repr(state)


@pytest.mark.L0
@pytest.mark.CPU
def test_value_payload_is_rejected_without_running(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    dcp.save({"iteration": _marker_payload(marker)}, checkpoint_id=tmp_path / "ckpt", no_dist=True)

    with pytest.raises((ValueError, CheckpointException)) as excinfo:
        _load(tmp_path / "ckpt", {"iteration": 0})

    assert "Refusing to load checkpoint value 'iteration'" in _rejection(excinfo)

    assert not marker.exists()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("hide_in_metadata", [False, True])
def test_metadata_payload_is_rejected_without_running(hide_in_metadata: bool, tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    dcp.save({"weight": torch.ones(2)}, checkpoint_id=tmp_path / "ckpt", no_dist=True)
    metadata_path = tmp_path / "ckpt" / ".metadata"
    if hide_in_metadata:
        metadata = pickle.loads(metadata_path.read_bytes())
        metadata.planner_data = {"weight": _marker_payload(marker)}
        metadata_path.write_bytes(pickle.dumps(metadata))
    else:
        metadata_path.write_bytes(pickle.dumps(_marker_payload(marker)))

    with pytest.raises(ValueError, match="Refusing to load checkpoint metadata .*forbidden global builtins.exec"):
        SafeFileSystemReader(tmp_path / "ckpt").read_metadata()
    # dcp.load reports a rejected metadata read as missing metadata; it must still fail.
    with pytest.raises((AssertionError, CheckpointException)):
        _load(tmp_path / "ckpt", {"weight": torch.zeros(2)})

    assert not marker.exists()


@pytest.mark.L0
@pytest.mark.CPU
def test_metadata_must_be_metadata_object(tmp_path: Path) -> None:
    dcp.save({"weight": torch.ones(2)}, checkpoint_id=tmp_path, no_dist=True)
    (tmp_path / ".metadata").write_bytes(pickle.dumps({"state_dict_metadata": {}}))

    with pytest.raises(ValueError, match="expected Metadata, got dict"):
        SafeFileSystemReader(tmp_path).read_metadata()
