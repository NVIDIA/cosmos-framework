# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The two-process example in ``docs/howto/sim-integration.md`` §4.2, run for real.

A guide that tells a simulation team to run two commands is a claim about those two commands.
Every other gate here proves one half: ``test_embodied_example`` runs the loop in ONE process,
and the wire tests exercise framing with both ends in the same interpreter. Neither would notice
that the documented pair no longer rendezvous -- a renamed flag, a changed default, an argument
the framework half stopped passing through -- and the reader who finds out is the one we asked
to try it.

So this starts ``sim_server --listen`` as a REAL subprocess and drives it through
``sim_gateway.main``, which is the same code path the guide prints. It is CPU-only because the
``example`` backend is: no engine, no card, no Ray on either side.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
#: The backend the CHILD must import to register `example`. Passed as `--preload`, which is the
#: property worth exercising: it must reach the simulator's interpreter and not this one.
_EXAMPLE = "cosmos_framework.posttrain.embodied.sims.example"


@pytest.fixture
def sim_socket(tmp_path):
    """A short socket path, deliberately not ``tmp_path``.

    ``AF_UNIX`` caps ``sun_path`` at ~107 bytes and pytest's ``tmp_path`` is long enough on CI
    to blow it -- the failure is a bare ``OSError`` naming neither the limit nor the path, which
    is the exact trap §4.2 warns the reader about. Using ``/dev/shm`` here keeps the test
    measuring the rendezvous rather than the length of its own fixture directory.
    """
    base = "/dev/shm" if os.path.isdir("/dev/shm") else str(tmp_path)
    path = os.path.join(base, f"gwtest-{uuid.uuid4().hex[:8]}.sock")
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def listening_simulator(sim_socket):
    """``sim_server --listen``, up and bound, or the test fails rather than hangs."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "cosmos_framework.posttrain.roles.env.sim_server",
            "--socket",
            sim_socket,
            "--listen",
        ],
        cwd=_REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        # POLL FOR THE BOUND SOCKET, not a sleep: the gateway's own `connect_with_retry` would
        # paper over a simulator that never came up by retrying until the start timeout, and the
        # failure would then read as a transport problem rather than "the child died on import".
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if os.path.exists(sim_socket):
                break
            if proc.poll() is not None:
                pytest.fail(f"sim_server exited {proc.returncode} before binding:\n{proc.stdout.read()}")
            time.sleep(0.05)
        else:
            pytest.fail("sim_server never bound the socket")
        yield sim_socket, proc
    finally:
        proc.kill()
        proc.wait(timeout=30)


@pytest.mark.L0
@pytest.mark.CPU
def test_the_documented_two_process_pair_actually_rendezvous(listening_simulator, capsys):
    """§4.2's two commands, in order, end to end: a unit crosses the split and comes back."""
    from cosmos_framework.posttrain.roles.env import sim_gateway

    sock, _proc = listening_simulator
    rc = sim_gateway.main(
        [
            "--socket",
            sock,
            "--sim",
            "example",
            "--embodiment",
            "example",
            "--action-dim",
            "7",
            "--preload",
            _EXAMPLE,
            "--episodes",
            "2",
            "--max-steps",
            "8",
            # BOUNDED, and not for speed. The gateway's default start timeout is minutes long
            # -- right for a production Isaac boot, wrong here: a pair that has stopped
            # rendezvousing would make this test HANG and be killed by the job timeout, which
            # reads as flaky infrastructure rather than as the guide being wrong. A short
            # bound turns that failure into a red test with the connect error in it.
            "--start-timeout-s",
            "30",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # The EPISODES, not just the return code: `main` returns 0 for a unit that ran no episodes
    # at all, so asserting on rc alone would pass against a rendezvous that connected and did
    # nothing. Outcomes are the example backend's fixed script -- instance 0 succeeds at step 4.
    assert "demo/i0  success   steps=4" in out
    assert "demo/i1  failure   steps=8" in out
    # The ARM RECEIPT for the crossing verifier. A run where the payload never crossed shared
    # memory prints `crossings_verified=0` and otherwise looks identical, so the count is the
    # only thing separating "the wire worked" from "nothing went over it".
    assert "crossings_verified=12" in out


@pytest.mark.L0
@pytest.mark.CPU
def test_the_framework_half_never_imports_the_backend(listening_simulator):
    """``--preload`` is passed THROUGH. The framework interpreter must not import the backend.

    This is the property the whole split is for, and it is invisible to the test above: that
    one passes just as well if this process imported `example` itself and the child inherited
    nothing.

    Asked in a FRESH SUBPROCESS rather than by reading this process's ``sys.modules``. The
    ambient version passed alone and failed in a full run, because `test_embodied_example.py`
    imports the same backend and pytest shares one interpreter -- so the assertion was really
    about which tests ran first. A subprocess has only what the gateway itself imported, which
    is the question.
    """
    sock, _proc = listening_simulator
    argv = [
        "--socket",
        sock,
        "--sim",
        "example",
        "--embodiment",
        "example",
        "--action-dim",
        "7",
        "--preload",
        _EXAMPLE,
        "--episodes",
        "1",
        "--start-timeout-s",
        "30",
    ]
    probe = (
        "import sys\n"
        "from cosmos_framework.posttrain.roles.env import sim_gateway\n"
        f"rc = sim_gateway.main({argv!r})\n"
        "print('RC', rc)\n"
        f"print('IMPORTED', {_EXAMPLE!r} in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], cwd=_REPO, capture_output=True, text=True, timeout=180)
    assert "RC 0" in out.stdout, f"the gateway half failed:\n{out.stdout}\n{out.stderr}"
    assert "IMPORTED False" in out.stdout, (
        "the framework half imported the simulator backend. `preload` names modules for the "
        "CHILD's interpreter; importing them here re-couples the two venvs the split separates."
        f"\n{out.stdout}"
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_an_unregistered_embodiment_is_refused_here_rather_than_guessed(listening_simulator, capsys):
    """No ``--action-dim`` and no local map is a REFUSAL, and §4.2 documents it as one.

    The framework side needs one number -- the action width -- and cannot get it by importing
    the backend without undoing the split. A default would be a guess that produces a running
    rollout of the wrong shape, which the simulator would accept and the corpus would record.
    """
    from cosmos_framework.posttrain.roles.env import sim_gateway

    sock, _proc = listening_simulator
    rc = sim_gateway.main(
        [
            "--socket",
            sock,
            "--sim",
            "example",
            # A name NOTHING registers, rather than "example". `test_embodied_example.py`
            # imports the example backend, which registers `example` process-wide and pytest
            # shares one interpreter -- so naming it here made the refusal depend on which
            # file ran first. It passed alone and vanished in a full run.
            "--embodiment",
            "no_such_embodiment_this_test_owns",
            "--episodes",
            "1",
            "--start-timeout-s",
            "30",
        ]
    )
    assert rc == 2
    assert "pass --action-dim" in capsys.readouterr().out
