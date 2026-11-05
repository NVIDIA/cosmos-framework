# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Joint Sim-Auto TF-dCM with own Stage-1 regular student/teacher weights."""

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.simulation.common.auto_profiles import make_auto_experiment

EXPERIMENT_NAME = "sim_auto_joint_stage2_tfdcm_all"
ConfigStore.instance().store(group="experiment", package="_global_", name=EXPERIMENT_NAME, node=make_auto_experiment(2))
