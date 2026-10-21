# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

from pathlib import Path


def get_action_robot_asset_path(filename: str) -> Path:
    """Return the path to a committed action robot asset."""

    return Path(__file__).resolve().parent / filename
