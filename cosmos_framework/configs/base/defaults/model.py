# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig
from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.model.vfm.omni_mot_model import OmniMoTModel

MOT_DDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(OmniMoTModel)(
        config=OmniMoTModelConfig(),
        _recursive_=False,
    ),
)


MOT_FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(OmniMoTModel)(
        config=OmniMoTModelConfig(
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=8,
            ),
        ),
        _recursive_=False,
    ),
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="mot_ddp", node=MOT_DDP_CONFIG)
    cs.store(group="model", package="_global_", name="mot_fsdp", node=MOT_FSDP_CONFIG)
