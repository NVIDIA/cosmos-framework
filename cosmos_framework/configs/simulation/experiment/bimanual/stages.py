# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Register the four Bimanual stages without historical checkpoint inputs."""

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.simulation.common.bimanual_profiles import (
    EXPERIMENT_NAMES,
    LOCAL_EXPERIMENT_NAMES,
    make_bimanual_experiment,
    make_local_bimanual_experiment,
)
from cosmos_framework.utils.lazy_config import LazyDict

for _stage, _name in enumerate(EXPERIMENT_NAMES):
    ConfigStore.instance().store(
        group="experiment",
        package="_global_",
        name=_name,
        node=LazyDict(make_bimanual_experiment(_stage), flags={"allow_objects": True}),
    )
    ConfigStore.instance().store(
        group="experiment",
        package="_global_",
        name=LOCAL_EXPERIMENT_NAMES[_stage],
        node=LazyDict(make_local_bimanual_experiment(_stage), flags={"allow_objects": True}),
    )
