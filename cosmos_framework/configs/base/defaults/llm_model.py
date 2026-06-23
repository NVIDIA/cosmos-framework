# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Register LLM and DiT model configs alongside the existing ``mot_fsdp``."""

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.model.vfm.llm.dit.image_dit_model import DiTPretrainModel, DiTPretrainModelConfig
from cosmos_framework.model.vfm.llm.llm_pretrain_model import LLMPretrainModel, LLMPretrainModelConfig

# ── FSDP config (production, multi-GPU) ──────────────────────────────────────

LLM_FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(LLMPretrainModel)(
        config=LLMPretrainModelConfig(),
        _recursive_=False,
    ),
)

# ── DDP config (debug, single-node) ─────────────────────────────────────────

LLM_DDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(LLMPretrainModel)(
        config=LLMPretrainModelConfig(),
        _recursive_=False,
    ),
)

# ── Image DiT configs ───────────────────────────────────────────────────────

DIT_FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(DiTPretrainModel)(
        config=DiTPretrainModelConfig(),
        _recursive_=False,
    ),
)

DIT_DDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(DiTPretrainModel)(
        config=DiTPretrainModelConfig(),
        _recursive_=False,
    ),
)


def register_llm_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="llm_fsdp", node=LLM_FSDP_CONFIG)
    cs.store(group="model", package="_global_", name="llm_ddp", node=LLM_DDP_CONFIG)
    cs.store(group="model", package="_global_", name="dit_fsdp", node=DIT_FSDP_CONFIG)
    cs.store(group="model", package="_global_", name="dit_ddp", node=DIT_DDP_CONFIG)
