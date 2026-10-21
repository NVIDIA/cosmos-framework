# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Dict

from hydra.core.config_store import ConfigStore

from cosmos_framework.checkpoint.dummy import Checkpointer as DummyCheckpointer
from cosmos_framework.utils.lazy_config import LazyCall as L

# DCP checkpointer optimized for Cosmos3
from cosmos_framework.checkpoint.dcp import DistributedCheckpointer as DistributedCheckpointerCosmos3
from cosmos_framework.checkpoint.dcp_distill import (
    DistributedCheckpointer as DistillCheckpointer,  # noqa: F401 - release-only checkpointer binding
)


DUMMY_CHECKPOINTER: Dict[str, str] = L(DummyCheckpointer)()
DISTRIBUTED_CHECKPOINTER_COSMOS3: Dict[str, str] = L(DistributedCheckpointerCosmos3)()
DISTRIBUTED_CHECKPOINTER_DISTILL_COSMOS3: Dict[str, str] = L(DistillCheckpointer)()


def register_distill_ckpt_type() -> None:
    cs = ConfigStore.instance()
    cs.store(group="ckpt_type", package="checkpoint.type", name="dummy", node=DUMMY_CHECKPOINTER)
    cs.store(group="ckpt_type", package="checkpoint.type", name="dcp_cosmos3", node=DISTRIBUTED_CHECKPOINTER_COSMOS3)
    cs.store(
        group="ckpt_type",
        package="checkpoint.type",
        name="dcp_distill_cosmos3",
        node=DISTRIBUTED_CHECKPOINTER_DISTILL_COSMOS3,
    )
