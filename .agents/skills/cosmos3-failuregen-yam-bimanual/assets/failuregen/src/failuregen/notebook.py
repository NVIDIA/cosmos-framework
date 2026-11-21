# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Stream command output into notebook cells and stop workers on interruption."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path


def run_streamed(*args, cwd: Path, env: dict[str, str]) -> None:
    command = [sys.executable, *map(str, args)]
    print(shlex.join(command), flush=True)
    with subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    ) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            returncode = process.wait()
        except BaseException:
            # torchrun starts child ranks; terminate the whole process group so
            # interrupting a notebook cell does not leave workers using the GPUs.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
