# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Canonical Hydra-group registry for checkpoint SKUs and checkpointer types."""

from typing import Dict

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils import config
from cosmos_framework.checkpoint.dummy import Checkpointer as DummyCheckpointer
from cosmos_framework.utils.config import CheckpointConfig
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.checkpoint.dcp import DistributedCheckpointer

local_object_store = config.ObjectStoreConfig(
    enabled=False,
)

pdx_object_store = config.ObjectStoreConfig(
    enabled=True,
    credentials="credentials/pdx_vfm_checkpoint.secret",
    bucket="checkpoints",
)

s3_object_store = config.ObjectStoreConfig(
    enabled=True,
    credentials="credentials/s3_checkpoint.secret",
    bucket="bucket4",
)

s3_eu_object_store = config.ObjectStoreConfig(
    enabled=True,
    credentials="credentials/s3_training_eu.secret",
    bucket="checkpoints-eu-west-3",
)

s3_east2_object_store = config.ObjectStoreConfig(
    enabled=True,
    credentials="credentials/s3_east2_checkpoint.secret",
    bucket="bucket",
)

gcp_object_store = config.ObjectStoreConfig(
    enabled=True,
    credentials="credentials/gcp_checkpoint.secret",
    bucket="bucket1",
)

neb_eu_object_store = config.ObjectStoreConfig(
    enabled=True,
    credentials="credentials/neb_eu.secret",
    bucket="nv-01-10206-checkpoint-experiments",
)

CHECKPOINT_LOCAL = CheckpointConfig(
    save_to_object_store=local_object_store,
    load_from_object_store=local_object_store,
    save_iter=5000,
    broadcast_via_filesystem=True,
    dcp_async_mode_enabled=True,
    dcp_load_dedup=True,
)

CHECKPOINT_PDX = CheckpointConfig(
    save_to_object_store=pdx_object_store,
    load_from_object_store=pdx_object_store,
    save_iter=5000,
    broadcast_via_filesystem=True,
    dcp_async_mode_enabled=True,
    dcp_load_dedup=True,
)

CHECKPOINT_S3 = CheckpointConfig(
    save_to_object_store=s3_object_store,
    load_from_object_store=s3_object_store,
    save_iter=5000,
    broadcast_via_filesystem=True,
    dcp_async_mode_enabled=True,
    dcp_load_dedup=True,
)

CHECKPOINT_S3_EU = CheckpointConfig(
    save_to_object_store=s3_eu_object_store,
    load_from_object_store=s3_eu_object_store,
    save_iter=5000,
    broadcast_via_filesystem=True,
    dcp_async_mode_enabled=True,
    dcp_load_dedup=True,
)

CHECKPOINT_GCP = CheckpointConfig(
    save_to_object_store=gcp_object_store,
    save_iter=1000,
    load_from_object_store=gcp_object_store,
    load_path="",
    load_training_state=False,
    strict_resume=True,
    enable_gcs_patch_in_boto3=True,
    dcp_async_mode_enabled=True,
)

CHECKPOINT_S3_EAST2 = CheckpointConfig(
    save_to_object_store=s3_east2_object_store,
    save_iter=1000,
    load_from_object_store=s3_east2_object_store,
    load_path="",
    load_training_state=False,
    strict_resume=True,
    enable_gcs_patch_in_boto3=False,
    dcp_async_mode_enabled=True,
    dcp_load_dedup=True,
)

CHECKPOINT_NEB_EU = CheckpointConfig(
    save_to_object_store=neb_eu_object_store,
    load_from_object_store=neb_eu_object_store,
    save_iter=2000,
    broadcast_via_filesystem=True,
    dcp_async_mode_enabled=True,
    dcp_load_dedup=True,
)


def register_checkpoint():
    cs = ConfigStore.instance()
    cs.store(group="checkpoint", package="checkpoint", name="local", node=CHECKPOINT_LOCAL)
    cs.store(group="checkpoint", package="checkpoint", name="pdx", node=CHECKPOINT_PDX)
    cs.store(group="checkpoint", package="checkpoint", name="s3", node=CHECKPOINT_S3)
    cs.store(group="checkpoint", package="checkpoint", name="s3_eu", node=CHECKPOINT_S3_EU)
    cs.store(group="checkpoint", package="checkpoint", name="gcp", node=CHECKPOINT_GCP)
    cs.store(group="checkpoint", package="checkpoint", name="neb_eu", node=CHECKPOINT_NEB_EU)
    cs.store(group="checkpoint", package="checkpoint", name="s3_east2", node=CHECKPOINT_S3_EAST2)


DUMMY_CHECKPOINTER: Dict[str, str] = L(DummyCheckpointer)()
DISTRIBUTED_CHECKPOINTER: Dict[str, str] = L(DistributedCheckpointer)()


def register_ckpt_type():
    cs = ConfigStore.instance()
    cs.store(group="ckpt_type", package="checkpoint.type", name="dummy", node=DUMMY_CHECKPOINTER)
    cs.store(group="ckpt_type", package="checkpoint.type", name="dcp", node=DISTRIBUTED_CHECKPOINTER)
