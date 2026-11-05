# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Focused registration for production Transfer stage1_tf."""

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.simulation.common.transfer_profiles import (
    STAGE1_EXPERIMENT_NAME,
    make_transfer_teacher_forcing_experiment,
)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name=STAGE1_EXPERIMENT_NAME,
    node=make_transfer_teacher_forcing_experiment(),
)
