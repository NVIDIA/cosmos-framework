# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Register the whole-window Sim-Transfer teacher without importing the upstream package."""

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.simulation.common.transfer_stage0_profiles import (
    STAGE0_EXPERIMENT_NAME,
    make_transfer_bidirectional_experiment,
)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name=STAGE0_EXPERIMENT_NAME,
    node=make_transfer_bidirectional_experiment(),
)
