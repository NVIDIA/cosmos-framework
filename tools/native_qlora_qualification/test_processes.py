# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real Linux process cleanup across elastic-like detached rank sessions."""

from __future__ import annotations

import errno
import json
import os
import platform
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from tools.native_qlora_qualification.processes import OwnedProcessTree, process_identity


def wait_for(predicate, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise TimeoutError("test process condition did not arrive")


@pytest.mark.skipif(sys.platform != "linux" or platform.machine() != "x86_64", reason="Linux x86_64 pidfd ABI required")
@pytest.mark.parametrize("backend", ["automatic", "syscall"])
@pytest.mark.parametrize("rank_exits", [True, False], ids=["interrupted-rank", "timeout-live-rank"])
def test_detached_rank_exit_does_not_leave_prefetch_worker_or_kill_unrelated_process(
    tmp_path, monkeypatch, rank_exits, backend
):
    from tools.native_qlora_qualification import processes

    if backend == "syscall":
        monkeypatch.setattr(processes, "_has_native_pidfds", lambda: False)
    capability = OwnedProcessTree.check_support()
    if backend == "syscall":
        assert capability["pidfd_backend"] == "linux_x86_64_pidfd_syscalls"
    assert capability["self_pidfd_open"] and capability["self_signal_zero"]
    ready, release = tmp_path / "ready.json", tmp_path / "release"
    rank_script = """
import json,os,subprocess,sys,time
from pathlib import Path
worker=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])
Path(sys.argv[1]).write_text(json.dumps({'rank':os.getpid(),'worker':worker.pid}))
while not Path(sys.argv[2]).exists():time.sleep(.01)
os._exit(86)
"""
    launcher_script = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',sys.argv[1],*sys.argv[2:]],start_new_session=True); p.wait()"
    launcher = subprocess.Popen(
        [sys.executable, "-c", launcher_script, rank_script, str(ready), str(release)], start_new_session=True
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    tracker = OwnedProcessTree(launcher.pid, interval=0.01)
    try:
        wait_for(ready.exists)
        record = json.loads(ready.read_text())
        wait_for(lambda: record["worker"] in tracker.known)
        snapshot = {"rank": process_identity(record["rank"]), "descendants": [process_identity(record["worker"])]}
        assert snapshot["rank"]["pgid"] != launcher.pid
        if rank_exits:
            release.touch()
            launcher.wait(timeout=8)
        tracker.validate_interruption(snapshot)
        forged = {"rank": snapshot["rank"], "descendants": [process_identity(unrelated.pid)]}
        with pytest.raises(ValueError, match="ownership was not observed"):
            tracker.validate_interruption(forged)
        result = tracker.cleanup()
        launcher.wait(timeout=8)
        assert result["observed_processes"] == 3
        assert result["pidfd_backend"] == capability["pidfd_backend"]
        wait_for(lambda: (identity := process_identity(record["worker"])) is None or identity["state"] == "Z")
        assert unrelated.poll() is None
    finally:
        try:
            tracker.cleanup()
        finally:
            if launcher.poll() is None:
                launcher.kill()
                launcher.wait()
            unrelated.kill()
            unrelated.wait()


def test_pid_birth_change_refuses_signal(monkeypatch):
    from tools.native_qlora_qualification import processes

    tracker = object.__new__(OwnedProcessTree)
    expected = {"pid": 100, "start_ticks": 10, "uid": 0, "cgroup": "fixture"}
    tracker._pidfd = SimpleNamespace(open=lambda _: 123, send_signal=lambda *a: pytest.fail("signaled reused PID"))
    monkeypatch.setattr(processes.os, "close", lambda _: None)
    monkeypatch.setattr(processes, "process_identity", lambda _: {**expected, "start_ticks": 11, "state": "S"})
    with pytest.raises(RuntimeError, match="reused"):
        tracker._signal(expected, processes.signal.SIGKILL)


@pytest.mark.parametrize("change", [None, "unknown", "start_ticks", "uid", "cgroup"])
def test_stale_children_snapshot_accepts_only_previously_verified_reparent(monkeypatch, change):
    from tools.native_qlora_qualification import processes

    parent = {"pid": 100, "ppid": 50, "start_ticks": 10, "uid": 7, "cgroup": "owned"}
    child = {"pid": 200, "ppid": 100, "start_ticks": 20, "uid": 7, "cgroup": "owned"}
    actual = {100: parent.copy(), 200: child.copy()}
    tracker = object.__new__(OwnedProcessTree)
    tracker.root = parent
    tracker.known = {100: parent.copy()}
    if change != "unknown":
        tracker.known[200] = child.copy()
    tracker._lock = threading.RLock()
    monkeypatch.setattr(processes, "process_identity", lambda pid: actual[pid].copy())

    def snapshot_then_reparent(pid):
        if pid != 100:
            return set()
        # The kernel returned this child while its parent was live, but the
        # parent's exit reparents it before refresh reads the child's identity.
        actual[200]["ppid"] = 1
        if change in {"start_ticks", "uid"}:
            actual[200][change] += 1
        elif change == "cgroup":
            actual[200]["cgroup"] = "unrelated"
        return {200}

    monkeypatch.setattr(processes, "children", snapshot_then_reparent)
    if change is None:
        tracker.refresh()
        assert tracker.known[200] == {**child, "ppid": 1}
    else:
        with pytest.raises(RuntimeError, match="ownership changed|identity was reused or moved"):
            tracker.refresh()
        assert tracker.known.get(200) == (None if change == "unknown" else child)


@pytest.mark.parametrize("operation", ["open", "send_signal"])
@pytest.mark.parametrize("error_number", [errno.ENOSYS, errno.EPERM])
def test_unavailable_or_denied_pidfd_fails_before_launch(monkeypatch, operation, error_number):
    from tools.native_qlora_qualification import processes

    monkeypatch.setattr(processes, "_has_native_pidfds", lambda: False)
    closed = []
    monkeypatch.setattr(processes.os, "close", closed.append)
    monkeypatch.setattr(processes._PidfdBackend, "open", lambda self, pid: 123)

    def fail(*args):
        raise OSError(error_number, os.strerror(error_number))

    monkeypatch.setattr(processes._PidfdBackend, operation, fail)
    with pytest.raises(RuntimeError, match="pidfd capability probe failed") as captured:
        OwnedProcessTree.check_support()
    assert captured.value.__cause__.errno == error_number
    assert closed == ([123] if operation == "send_signal" else [])


def test_partial_python_api_pair_uses_only_verified_syscalls(monkeypatch):
    from tools.native_qlora_qualification import processes

    monkeypatch.setattr(processes.os, "pidfd_open", lambda _: pytest.fail("mixed native API"), raising=False)
    monkeypatch.setattr(processes.signal, "pidfd_send_signal", None, raising=False)
    assert OwnedProcessTree.check_support()["pidfd_backend"] == "linux_x86_64_pidfd_syscalls"


def test_unknown_syscall_architecture_fails_closed(monkeypatch):
    from tools.native_qlora_qualification import processes

    monkeypatch.setattr(processes, "_has_native_pidfds", lambda: False)
    monkeypatch.setattr(processes.platform, "machine", lambda: "unsupported")
    with pytest.raises(RuntimeError, match="explicit Linux x86_64"):
        OwnedProcessTree.check_support()


@pytest.mark.parametrize("short_type", ["c_void_p", "c_long"])
def test_non_lp64_python_fails_closed_for_syscall_backend(monkeypatch, short_type):
    from tools.native_qlora_qualification import processes

    monkeypatch.setattr(processes, "_has_native_pidfds", lambda: False)
    original = processes.ctypes.sizeof
    shortened = getattr(processes.ctypes, short_type)
    monkeypatch.setattr(processes.ctypes, "sizeof", lambda value: 4 if value is shortened else original(value))
    with pytest.raises(RuntimeError, match="Linux x86_64 LP64"):
        OwnedProcessTree.check_support()


def test_native_python_api_pair_does_not_use_the_syscall_abi_guard(monkeypatch):
    from tools.native_qlora_qualification import processes

    monkeypatch.setattr(processes, "_has_native_pidfds", lambda: True)
    monkeypatch.setattr(processes.ctypes, "sizeof", lambda value: pytest.fail("native API inspected syscall ABI"))
    assert processes._PidfdBackend().backend == "python_pidfd"
