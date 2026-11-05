# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Focused registration for production Transfer stage2_tfdcm."""

from hydra.core.config_store import ConfigStore

import cosmos_framework.configs.simulation.experiment.transfer.stage1_tf  # noqa: F401
from cosmos_framework.configs.simulation.common.transfer_profiles import (
    STAGE2_EXPERIMENT_NAME,
    make_transfer_teacher_forcing_dcm_experiment,
)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name=STAGE2_EXPERIMENT_NAME,
    node=make_transfer_teacher_forcing_dcm_experiment(),
)
