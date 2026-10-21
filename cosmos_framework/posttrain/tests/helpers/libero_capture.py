# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Retain the existing native reset observation for isolated parity inspection."""

from cosmos_framework.posttrain.embodied.sims.libero.vector import LiberoLane


class RecordingNativeLane(LiberoLane):
    def reset(self):
        observation = super().reset()
        self.last_reset_observation = observation
        return observation
