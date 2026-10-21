# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Register the production Transfer SF-DMD profile without research catalogs."""

from hydra.core.config_store import ConfigStore

import cosmos_framework.configs.simulation.experiment.transfer.stage2_tfdcm  # noqa: F401
from cosmos_framework.configs.simulation.common.transfer_stage3_profiles import (
    EXPERIMENT_NAME,
    make_transfer_self_forcing_dmd_experiment,
)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name=EXPERIMENT_NAME,
    node=make_transfer_self_forcing_dmd_experiment(),
)
