# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generate candidate robot-failure videos with Cosmos neural simulation.

FailureGen selects compatible donor and host recordings and transfers measured
motion. Cosmos predicts the visual continuation from the host observation,
transferred actions, and neutral task text. Review determines the observed event.
"""

from .config import ArmSpec, CameraProfile, DatasetProfile, Profile, RobotProfile
from .splice import splice

__version__ = "0.1.0"
__all__ = ["Profile", "RobotProfile", "DatasetProfile", "CameraProfile", "ArmSpec", "splice"]
