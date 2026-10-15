# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""YAM joint/EEF adaptation for Cosmos3-Nano; GPU imports are opt-in."""


def register_domain(action_space="joint"):
    from cosmos_framework.data.generator.action.utils.domain_utils import (
        EMBODIMENT_TO_DOMAIN_ID,
        EMBODIMENT_TO_RAW_ACTION_DIM,
    )

    from .contract import DOMAIN, DOMAIN_ID

    if action_space == "eef":
        if EMBODIMENT_TO_DOMAIN_ID.get("xdof_yam") != 16 or EMBODIMENT_TO_RAW_ACTION_DIM.get("xdof_yam") != 20:
            raise ValueError("Expected existing xdof_yam domain 16 with width 20")
        return
    if action_space != "joint":
        raise ValueError(f"Unknown action space: {action_space}")

    occupied = [name for name, value in EMBODIMENT_TO_DOMAIN_ID.items() if value == DOMAIN_ID and name != DOMAIN]
    if occupied:
        raise ValueError(f"YAM domain {DOMAIN_ID} is occupied by {occupied}; use the documented framework revision")
    if DOMAIN in EMBODIMENT_TO_DOMAIN_ID and EMBODIMENT_TO_DOMAIN_ID[DOMAIN] != DOMAIN_ID:
        raise ValueError("Existing YAM registration has a different domain ID")
    if DOMAIN in EMBODIMENT_TO_RAW_ACTION_DIM and EMBODIMENT_TO_RAW_ACTION_DIM[DOMAIN] != 14:
        raise ValueError("Existing YAM registration has a different action width")
    EMBODIMENT_TO_DOMAIN_ID[DOMAIN] = DOMAIN_ID
    EMBODIMENT_TO_RAW_ACTION_DIM[DOMAIN] = 14
