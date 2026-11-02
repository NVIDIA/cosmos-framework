# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only observer contracts; no native scheduler or remote filesystem calls."""

import errno
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import slurm_controller_watch as observer  # noqa: E402


@pytest.fixture
def registry_data(tmp_path):
    return {
        "uid": os.getuid(),
        "deadline": 2000.0,
        "jobs": {"123": {"attempt_base": str(tmp_path / "attempts"), "max_allocations": 3}},
    }


def save_registry(root, value):
    (root / "registry.json").write_text(json.dumps(value))


def test_registry_round_trip(tmp_path, registry_data):
    save_registry(tmp_path, registry_data)
    assert observer.registry(tmp_path) == registry_data


@pytest.mark.parametrize(
    "changes",
    [
        {"uid": -1},
        {"deadline": float("inf")},
        {"deadline": float("nan")},
        {"jobs": []},
    ],
)
def test_registry_rejects_invalid_identity(tmp_path, registry_data, changes):
    save_registry(tmp_path, registry_data | changes)
    with pytest.raises(ValueError):
        observer.registry(tmp_path)


@pytest.mark.parametrize("job_id", ["0", "0123", "-1", "123;true", ""])
def test_registry_rejects_invalid_native_id(tmp_path, registry_data, job_id):
    registry_data["jobs"] = {job_id: registry_data["jobs"]["123"]}
    save_registry(tmp_path, registry_data)
    with pytest.raises(ValueError, match="identity or allocation cap"):
        observer.registry(tmp_path)


@pytest.mark.parametrize(
    "changes",
    [{"attempt_base": "relative"}, {"max_allocations": 0}, {"max_allocations": True}, {"max_allocations": 1.5}],
)
def test_registry_rejects_invalid_job_contract(tmp_path, registry_data, changes):
    registry_data["jobs"]["123"].update(changes)
    save_registry(tmp_path, registry_data)
    with pytest.raises(ValueError, match="identity or allocation cap"):
        observer.registry(tmp_path)


@pytest.fixture
def live_observer(tmp_path, registry_data, monkeypatch):
    monkeypatch.setattr(observer.time, "time", lambda: 1000.0)
    save_registry(tmp_path, registry_data)
    state = {"state": "WATCHING", "uid": os.getuid(), "deadline": 2000.0, "updated_at": 999.0}
    observer.atomic(tmp_path / "watch.json", state)
    with (tmp_path / "watch.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield state


def test_ready_requires_live_lock(tmp_path, registry_data, monkeypatch):
    monkeypatch.setattr(observer.time, "time", lambda: 1000.0)
    save_registry(tmp_path, registry_data)
    observer.atomic(
        tmp_path / "watch.json",
        {"state": "WATCHING", "uid": os.getuid(), "deadline": 2000.0, "updated_at": 999.0},
    )
    with pytest.raises(ValueError, match="not running"):
        observer.ready(tmp_path)


def test_ready_accepts_live_owner(tmp_path, live_observer):
    assert observer.ready(tmp_path) == live_observer


@pytest.mark.parametrize(
    "changes",
    [
        {"state": "STOPPED"},
        {"uid": -1},
        {"deadline": 2001.0},
        {"updated_at": 910.0},
        {"updated_at": 1001.0},
    ],
)
def test_ready_rejects_stale_or_conflicting_receipt(tmp_path, live_observer, changes):
    observer.atomic(tmp_path / "watch.json", live_observer | changes)
    with pytest.raises(ValueError, match="not ready"):
        observer.ready(tmp_path)


def test_register_is_idempotent_and_cannot_replace_identity(tmp_path, live_observer):
    base = tmp_path / "another-attempt"
    observer.register(tmp_path, "124", base, 2)
    saved = (tmp_path / "registry.json").read_bytes()
    observer.register(tmp_path, "124", base, 2)
    assert (tmp_path / "registry.json").read_bytes() == saved
    for new_base, cap in [(base, 3), (tmp_path / "other", 2)]:
        with pytest.raises(ValueError, match="cannot be replaced"):
            observer.register(tmp_path, "124", new_base, cap)
        assert (tmp_path / "registry.json").read_bytes() == saved


@pytest.fixture
def native(monkeypatch):
    # Any attempted submit/requeue/cancel would fail the exact-argv assertion.
    fields = {"JobId": "123", "UserId": f"user({os.getuid()})", "Restarts": "0", "JobState": "RUNNING"}

    def query(argv, **kwargs):
        assert argv == ["scontrol", "show", "job", "-o", "123"]
        assert kwargs == {"capture_output": True, "text": True, "timeout": 30, "check": True}
        return subprocess.CompletedProcess(argv, 0, " ".join(f"{k}={v}" for k, v in fields.items()))

    monkeypatch.setattr(observer.subprocess, "run", query)
    probe = Mock(return_value={"phase": "NOT_ENTERED"})
    monkeypatch.setattr(observer, "observe", probe)
    return fields, probe


@pytest.mark.parametrize("generation", [0, 1, 2])
def test_tick_observes_each_allowed_generation(registry_data, native, generation):
    fields, probe = native
    fields["Restarts"] = str(generation)
    entry = registry_data["jobs"]["123"]
    assert observer.tick("123", entry) == {"state": "OBSERVING", "generation": generation, "phase": "NOT_ENTERED"}
    probe.assert_called_once_with(Path(entry["attempt_base"]) / f"123-g{generation}", "123")


@pytest.mark.parametrize("state", ["PENDING", "COMPLETED", "FAILED", "CANCELLED"])
def test_tick_never_probes_nonrunning_jobs(registry_data, native, state):
    fields, probe = native
    fields["JobState"] = state
    assert observer.tick("123", registry_data["jobs"]["123"]) == {
        "state": "OBSERVING",
        "native_state": state,
        "generation": 0,
    }
    probe.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"JobId": "124"},
        {"UserId": "other(-1)"},
        {"Restarts": "-1"},
        {"Restarts": "3"},
        {"Restarts": "999999999999999999999999999"},
        {"Restarts": "invalid"},
    ],
)
def test_tick_blocks_identity_or_generation_mismatch(registry_data, native, changes):
    fields, probe = native
    fields.update(changes)
    assert observer.tick("123", registry_data["jobs"]["123"]) == {"state": "BLOCKED", "error_type": "ValueError"}
    probe.assert_not_called()


def test_tick_blocks_missing_native_restart_field(registry_data, native):
    fields, probe = native
    del fields["Restarts"]
    assert observer.tick("123", registry_data["jobs"]["123"]) == {"state": "BLOCKED", "error_type": "KeyError"}
    probe.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        OSError(errno.EIO, "private detail"),
        subprocess.TimeoutExpired("private command", 30, stderr="private detail"),
        subprocess.CalledProcessError(1, "private command", stderr="private detail"),
    ],
)
def test_tick_waits_on_scheduler_errors_without_recording_details(registry_data, native, monkeypatch, failure):
    _, probe = native
    monkeypatch.setattr(observer.subprocess, "run", Mock(side_effect=failure))
    assert observer.tick("123", registry_data["jobs"]["123"]) == {
        "state": "WAITING",
        "error_type": type(failure).__name__,
        "errno": getattr(failure, "errno", None),
    }
    probe.assert_not_called()


def test_tick_waits_on_shared_storage_error(registry_data, native):
    _, probe = native
    probe.side_effect = OSError(errno.EIO, "private path")
    assert observer.tick("123", registry_data["jobs"]["123"]) == {
        "state": "WAITING",
        "error_type": "OSError",
        "errno": errno.EIO,
    }
