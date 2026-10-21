# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU tests for the preemption/timeout termination watch (control.termination).

Covers the two detection paths — the Slurm sentinel file (the mechanism that works
through pyxis) and a directly delivered signal — plus the quiet default, and the SIGTERM
contract the driver relies on when a scheduler preempts without a user signal: the watch's
handler is flag-only (the process survives SIGTERM and reports it at the next step
boundary) and it replaces an earlier handler that would have exited the process (what
`ray.init` installs on the driver). TerminationWatch installs process-global
SIGUSR1/SIGTERM handlers, so every test restores the previous handlers (a leaked handler
would outlive the test and swallow real signals for the rest of the pytest session)."""

from __future__ import annotations

import os
import signal
import sys

import pytest

from cosmos_framework.posttrain.control.termination import TerminationWatch


@pytest.fixture()
def restore_signal_handlers():
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR1, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


@pytest.mark.L0
@pytest.mark.CPU
def test_no_trigger_is_quiet(tmp_path, restore_signal_handlers):
    watch = TerminationWatch(sentinel=str(tmp_path / "SIGUSR1_RECEIVED"))
    assert watch.should_stop() is None


@pytest.mark.L0
@pytest.mark.CPU
def test_sentinel_file_triggers(tmp_path, restore_signal_handlers):
    sentinel = tmp_path / "SIGUSR1_RECEIVED"
    watch = TerminationWatch(sentinel=str(sentinel))
    assert watch.should_stop() is None
    sentinel.touch()  # what the Slurm batch-shell trap does on SIGUSR1
    reason = watch.should_stop()
    assert reason and "sentinel" in reason


@pytest.mark.L0
@pytest.mark.CPU
def test_direct_signal_triggers(tmp_path, restore_signal_handlers):
    watch = TerminationWatch(sentinel=str(tmp_path / "nope"))
    os.kill(os.getpid(), signal.SIGUSR1)  # torchrun-outside-container path
    reason = watch.should_stop()
    assert reason and "SIGUSR1" in reason


@pytest.mark.L0
@pytest.mark.CPU
def test_sentinel_from_slurm_log_dir_env(tmp_path, monkeypatch, restore_signal_handlers):
    monkeypatch.setenv("SLURM_LOG_DIR", str(tmp_path))
    watch = TerminationWatch()  # default path comes from the env
    (tmp_path / "SIGUSR1_RECEIVED").touch()
    assert watch.should_stop()


@pytest.mark.L0
@pytest.mark.CPU
def test_direct_sigterm_flags_without_terminating_the_process(tmp_path, restore_signal_handlers):
    """Preemption without `preempt_send_user_signal` delivers SIGTERM first (no SIGUSR1). The watch's
    handler is flag-only: the process is still here after the signal, and the next step-boundary
    check names it — the driver's chance to run ONE emergency save inside the grace."""
    watch = TerminationWatch(sentinel=str(tmp_path / "nope"))
    assert watch.should_stop() is None
    os.kill(os.getpid(), signal.SIGTERM)
    reason = watch.should_stop()  # we are alive to ask
    assert reason == "signal SIGTERM"
    assert watch.should_stop() == "signal SIGTERM"  # the flag is sticky, not consumed


@pytest.mark.L0
@pytest.mark.CPU
def test_watch_replaces_an_earlier_exiting_sigterm_handler(tmp_path, restore_signal_handlers):
    """`ray.init` installs `sys.exit(signum)` on SIGTERM in the driver (ray/_private/worker.py, the
    connect path) before the training loop starts; the loop constructs the watch AFTER setup, so its
    flag-only handler must win — a SIGTERM must not raise SystemExit out of the loop."""

    def ray_style_exit(signum, frame):  # the handler ray.init installs on the driver
        sys.exit(signum)

    signal.signal(signal.SIGTERM, ray_style_exit)
    watch = TerminationWatch(sentinel=str(tmp_path / "nope"))
    assert signal.getsignal(signal.SIGTERM) is not ray_style_exit
    os.kill(os.getpid(), signal.SIGTERM)  # would raise SystemExit(15) under the earlier handler
    assert watch.should_stop() == "signal SIGTERM"


@pytest.mark.L0
@pytest.mark.CPU
def test_sentinel_names_the_reason_over_a_direct_signal(tmp_path, restore_signal_handlers):
    """When both the batch shell's TERM trap (sentinel) and the direct signal arrive, the reason names
    the sentinel: it is the path a container run can rely on, and the log line should say so."""
    sentinel = tmp_path / "SIGUSR1_RECEIVED"
    watch = TerminationWatch(sentinel=str(sentinel))
    os.kill(os.getpid(), signal.SIGTERM)
    sentinel.touch()  # `trap 'touch "$SLURM_LOG_DIR/SIGUSR1_RECEIVED"' USR1 TERM` in the batch script
    reason = watch.should_stop()
    assert reason == f"sentinel {sentinel}"
