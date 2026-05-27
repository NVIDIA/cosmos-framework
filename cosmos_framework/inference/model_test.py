# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import attrs
import hydra

from cosmos_framework.inference.args import DEFAULT_CHECKPOINT
from cosmos_framework.inference.common.config import structure_config
from cosmos_framework.inference.model import Cosmos3OmniConfig
from cosmos_framework.configs.base.defaults.model_config import ParallelismConfig


def test_config():
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        context_parallel_shard_degree=2,
        cfg_parallel_shard_degree=2,
        use_torch_compile=True,
        use_cuda_graphs=True,
    )
    parallelism_kwargs = attrs.asdict(parallelism)
    checkpoint_path = DEFAULT_CHECKPOINT.download()
    config = Cosmos3OmniConfig.from_pretrained(checkpoint_path, parallelism=parallelism_kwargs)
    assert (
        hydra.utils.instantiate(structure_config(config.model["config"]["parallelism"], ParallelismConfig))
        == parallelism
    )
