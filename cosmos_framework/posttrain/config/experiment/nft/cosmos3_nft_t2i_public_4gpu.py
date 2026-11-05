# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Four-GPU public T2I DiffusionNFT recipe with a generic remote reward contract."""

from __future__ import annotations

from copy import deepcopy

from hydra.core.config_store import ConfigStore

from cosmos_framework.posttrain.config.experiment.nft import NFT_T2I_PUBLIC_TOY_4GPU_CONTRACT
from cosmos_framework.posttrain.config.experiment.nft.cosmos3_nft_t2i_public import cosmos3_nft_t2i_public

_PROFILE = NFT_T2I_PUBLIC_TOY_4GPU_CONTRACT

cosmos3_nft_t2i_public_4gpu = deepcopy(cosmos3_nft_t2i_public)
cosmos3_nft_t2i_public_4gpu["grid"]["ranks"] = _PROFILE.group_size
cosmos3_nft_t2i_public_4gpu["model"]["config"]["parallelism"]["data_parallel_shard_degree"] = (
    _PROFILE.data_parallel_shard_degree
)
cosmos3_nft_t2i_public_4gpu["dataloader_train"]["expected_rows"] = _PROFILE.expected_rows
cosmos3_nft_t2i_public_4gpu["dataloader_train"]["num_replicas"] = _PROFILE.num_replicas
cosmos3_nft_t2i_public_4gpu["rl"].profile = _PROFILE.profile
cosmos3_nft_t2i_public_4gpu["rl"].group_size = _PROFILE.group_size
cosmos3_nft_t2i_public_4gpu["rl"].grad_accum = _PROFILE.grad_accum
cosmos3_nft_t2i_public_4gpu["rl"].sample_steps = _PROFILE.sample_steps
cosmos3_nft_t2i_public_4gpu["rl"].reward.provider_name = _PROFILE.provider_name
cosmos3_nft_t2i_public_4gpu["rl"].reward.score_key = _PROFILE.score_key
cosmos3_nft_t2i_public_4gpu["job"]["group"] = "cosmos3_nft_t2i_public"
cosmos3_nft_t2i_public_4gpu["job"]["name"] = "cosmos3_nft_t2i_public_4gpu"

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="cosmos3_nft_t2i_public_4gpu",
    node=cosmos3_nft_t2i_public_4gpu,
)
