# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""LIBERO registration without importing the simulator runtime on protocol reads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cosmos_framework.posttrain.embodied.registry import register_sim

if TYPE_CHECKING:
    from cosmos_framework.posttrain.embodied.sims.libero.runtime import LiberoSimulator


def build_simulator(**kwargs: Any) -> LiberoSimulator:
    from cosmos_framework.posttrain.embodied.sims.libero.runtime import LiberoSimulator

    return LiberoSimulator(**kwargs)


register_sim("libero", build_simulator)
