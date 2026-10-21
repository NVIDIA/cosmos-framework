# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Read-only RNG observation inside an isolated RLinf reference worker."""

import numpy as np
from libero.libero.envs import OffScreenRenderEnv


class ObservedReferenceEnv(OffScreenRenderEnv):
    @property
    def native_numpy_rng(self):
        return np.random.get_state()
