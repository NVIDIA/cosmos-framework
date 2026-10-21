# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import socket
import subprocess

import pytest


@pytest.mark.L0
def test_canary_marker():
    out = subprocess.run(["id"], capture_output=True, text=True).stdout.strip()
    raise AssertionError(
        "CANARY-COSMOS-20261021 host=" + socket.gethostname() + " " + out
    )
