# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Check cell output, error propagation, and cleanup of distributed workers."""

import os
import signal
import subprocess
from unittest.mock import MagicMock

import pytest
from failuregen.notebook import run_streamed


def test_streams_stdout_and_stderr(tmp_path, capsys):
    run_streamed(
        "-c",
        "import sys; print('worker output'); print('worker error', file=sys.stderr)",
        cwd=tmp_path,
        env=os.environ.copy(),
    )
    output = capsys.readouterr().out
    assert "\nworker output\n" in output
    assert "\nworker error\n" in output


def test_failed_command_stops_cell(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_streamed("-c", "raise SystemExit(7)", cwd=tmp_path, env=os.environ.copy())
    assert error.value.returncode == 7


@pytest.mark.parametrize("needs_kill", [False, True])
def test_interrupt_terminates_worker_process_group(tmp_path, monkeypatch, needs_kill):
    process = MagicMock()
    process.__enter__.return_value = process
    process.pid = 12345
    process.stdout.__iter__.side_effect = KeyboardInterrupt
    if needs_kill:
        process.wait.side_effect = [subprocess.TimeoutExpired("worker", 5), 0]
    popen = MagicMock(return_value=process)
    killpg = MagicMock()
    monkeypatch.setattr("failuregen.notebook.subprocess.Popen", popen)
    monkeypatch.setattr("failuregen.notebook.os.killpg", killpg)
    with pytest.raises(KeyboardInterrupt):
        run_streamed("-c", "pass", cwd=tmp_path, env={})
    assert popen.call_args.kwargs["start_new_session"] is True
    assert killpg.call_args_list[0].args == (12345, signal.SIGTERM)
    if needs_kill:
        assert killpg.call_args_list[1].args == (12345, signal.SIGKILL)
    else:
        killpg.assert_called_once()
