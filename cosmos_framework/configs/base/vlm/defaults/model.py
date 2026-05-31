# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.configs.base.vlm.defaults.policy_config import VLMModelConfig
from cosmos_framework.model.vfm.vlm_model import VLMModel

VLM_FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(VLMModel)(
        config=VLMModelConfig(
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=8,
            ),
        ),
        checkpoint="${checkpoint}",
        _recursive_=False,
    ),
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="vlm_fsdp", node=VLM_FSDP_CONFIG)
