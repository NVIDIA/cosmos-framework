# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared evaluation isolation at the process that executes a decision policy."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cosmos_framework.posttrain.roles.rollout.validation import isolated_validation_rng


class DecisionRolloutMixin(ABC):
    """Isolate evaluation without wrapping or replacing a body's weight receiver contract."""

    def evaluation_actions(self, observations: list[dict], *, seed: int, weight_version: int) -> list:
        with isolated_validation_rng():
            return self._evaluation_actions(observations, seed=seed, weight_version=weight_version)

    @abstractmethod
    def _evaluation_actions(self, observations: list[dict], *, seed: int, weight_version: int) -> list:
        """Model-owned evaluation under the public RNG-isolated entry."""
