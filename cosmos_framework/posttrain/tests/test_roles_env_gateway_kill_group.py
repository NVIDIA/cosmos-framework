# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Teardown must reap the child's process GROUP, including when the leader is already dead.

The defect this covers was a guard that read the wrong process. `_kill_child` asked whether the
LEADER was alive (`self._proc.poll() is None`) and returned when it was not -- but a dead leader
is the ordinary case on every path that tears down: the planned restart (`collector.py`) and the
orphan watchdog (`wire.py`) both leave via `os._exit`, and a native `ERROR_DEVICE_LOST` or a
segfault takes the faulting process alone. In each of those the helpers Isaac forked into the
session are still running and still holding the card, and the guard skipped the `killpg` that
would have reaped them. The next `_spawn` then booted a fresh Isaac beside them -- a GPU busy for
no visible reason.

No simulator is involved. The "child" is a Python process started the way `_spawn` starts
sim_server (`start_new_session=True`), which forks one helper into its own session -- standing in
for Kit's workers -- and then leaves the way the planned restart does. The REAL `_kill_child` is
then called and the question is asked of the group.

The parametrisation is the point. `leader-alive` is the control: it passed both before and after
the fix, which is what proves `killpg` was never the broken part -- the guard in front of it was.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from cosmos_framework.posttrain.embodied.wire import group_alive
from cosmos_framework.posttrain.roles.env.sim_gateway import SimGatewayBody

#: Forks a helper into its OWN session (as Isaac does), reports its pid, then either stays up
#: (the control) or leaves via `os._exit` (what the planned restart and the watchdog do).
_HELPER = (
    "import os, subprocess, sys, time\n"
    "h = subprocess.Popen(['sleep', '300'])\n"
    "print(h.pid, flush=True)\n"
    "if sys.argv[1] == 'exit9':\n"
    "    os._exit(9)\n"
    "time.sleep(300)\n"
)


def _bare_gateway(proc: subprocess.Popen) -> SimGatewayBody:
    """A body carrying only the three fields `_kill_child` reads. No socket, no child config.

    Built with `object.__new__` deliberately: `__init__` would start a real handshake, and the
    method under test must be reachable without one -- a child that dies before handshaking is
    one of the cases that has to be reaped.
    """
    g = object.__new__(SimGatewayBody)
    g._attach = False
    g._proc = proc
    # What a successful `hello` would have recorded. Under `start_new_session=True` the leader's
    # pid IS the group id, so this is the same number the fallback would compute.
    g._child_pgid = proc.pid
    return g


def _spawn_leader(mode: str) -> tuple[subprocess.Popen, int]:
    proc = subprocess.Popen(
        [sys.executable, "-c", _HELPER, mode],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    helper_pid = int(proc.stdout.readline().strip())
    # Without this the test would pass vacuously against any implementation: a helper that is
    # NOT in the child's group is not the thing the group teardown exists to reap.
    assert os.getpgid(helper_pid) == proc.pid, "helper must share the leader's session"
    return proc, helper_pid


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "mode",
    ["alive", "exit9"],
    ids=["leader-alive (control)", "leader-exited-9 (the finding)"],
)
def test_kill_child_tears_down_the_whole_group(mode: str) -> None:
    proc, helper_pid = _spawn_leader(mode)
    pgid = proc.pid
    try:
        if mode == "exit9":
            assert proc.wait(timeout=10) == 9, "leader should have left with the planned-restart code"
            assert proc.poll() is not None, "precondition: the leader is gone"
        assert group_alive(pgid), "precondition: the helper keeps the group alive before teardown"

        _bare_gateway(proc)._kill_child()

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and group_alive(pgid):
            time.sleep(0.1)
        assert not group_alive(pgid), (
            f"process group {pgid} is still alive after _kill_child(); helper pid {helper_pid} "
            f"survived. A helper left behind by a leader that already exited is exactly what "
            f"holds the card into the next spawn."
        )
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
