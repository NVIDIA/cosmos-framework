# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboLab registration without importing Isaac or native policy clients."""

from typing import Any

from cosmos_framework.posttrain.embodied.registry import register_sim


def build_simulator(**kwargs: Any) -> Any:
    from cosmos_framework.posttrain.embodied.sims.robolab.runtime import RoboLabSimulator

    return RoboLabSimulator(**kwargs)


register_sim("robolab", build_simulator)
