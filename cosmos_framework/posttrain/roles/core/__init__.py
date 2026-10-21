# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""roles.core — shared role base + the user-facing factory (DESIGN §2.7)."""

from cosmos_framework.posttrain.roles.core.factory import (
    buffer,
    critic,
    env,
    reference,
    reward,
    rollout,
    teacher,
    trainer,
)

__all__ = ["env", "rollout", "trainer", "reference", "critic", "reward", "teacher", "buffer"]
