# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Explicit local Transfer-Auto curriculum registrations (not causal Sim-Auto)."""

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.simulation.common.transfer_auto_recipes import make_transfer_auto_experiment

for _phase in (1, 2, 3, 4):
    ConfigStore.instance().store(
        group="experiment",
        package="_global_",
        name=f"transfer_auto_phase{_phase}",
        node=make_transfer_auto_experiment(_phase),
    )
