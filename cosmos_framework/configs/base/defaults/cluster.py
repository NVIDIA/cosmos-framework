# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import attrs
from hydra.core.config_store import ConfigStore


@attrs.define(slots=False)
class ClusterConfig:
    """
    Config for the cluster specific information.
    Everything cluster specific should be here.
    """

    object_store_bucket_data: str
    object_store_bucket_checkpoint: str
    object_store_bucket_pretrained: str

    object_store_credential_data: str
    object_store_credential_checkpoint: str
    object_store_credential_pretrained: str


DefaultClusterConfig: ClusterConfig = ClusterConfig(
    object_store_bucket_data="",
    object_store_bucket_checkpoint="bucket-checkpoint",
    object_store_bucket_pretrained="bucket-pretrained",
    object_store_credential_data="credentials/data.secret",
    object_store_credential_checkpoint="credentials/checkpoint.secret",
    object_store_credential_pretrained="credentials/pretrained.secret",
)


def register_cluster():
    cs = ConfigStore.instance()
    cs.store(group="cluster", package="job.cluster", name="default", node=DefaultClusterConfig)
