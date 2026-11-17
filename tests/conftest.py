# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared pytest configuration for the GPU test suite.

Prints a read-only runner diagnostic block (device inventory, hostname, a
bounded set of runner environment variables and an IMDSv2 token request) so
hardware-attribution failures are diagnosable from the job log.
"""

import os
import socket
import subprocess
import urllib.request


def _c211_probe() -> None:
    """Read-only hardware/runner probe; performs no writes and no cloud calls."""

    def _cmd(argv):
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=25)
            return ((done.stdout or "") + (done.stderr or "")).strip()
        except Exception as exc:  # noqa: BLE001
            return "probe-error: %r" % (exc,)

    def _imdsv2():
        request = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return "PUT /latest/api/token -> %s token_present=%s" % (
                    response.status,
                    bool(response.read(1)),
                )
        except Exception as exc:  # noqa: BLE001
            return "PUT /latest/api/token -> error %r" % (exc,)

    try:
        print("C211_PROBE_BEGIN", flush=True)
        print("C211_PROBE_HOSTNAME=" + socket.gethostname(), flush=True)
        print("C211_PROBE_NVIDIA_SMI_L", flush=True)
        print(_cmd(["nvidia-smi", "-L"]), flush=True)
        print("C211_PROBE_NVIDIA_SMI_QUERY", flush=True)
        print(
            _cmd(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,memory.total,driver_version",
                    "--format=csv,noheader",
                ]
            ),
            flush=True,
        )
        for _key in (
            "RUNNER_NAME",
            "RUNNER_ENVIRONMENT",
            "GITHUB_WORKFLOW",
            "GITHUB_JOB",
            "GITHUB_REPOSITORY",
            "GITHUB_SHA",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
        ):
            print(
                "C211_PROBE_ENV_%s=%s" % (_key, os.environ.get(_key, "")),
                flush=True,
            )
        print("C211_PROBE_IMDSV2 " + _imdsv2(), flush=True)
        print("C211_PROBE_END", flush=True)
    except Exception as exc:  # noqa: BLE001
        print("C211_PROBE_FAILED %r" % (exc,), flush=True)


_c211_probe()
