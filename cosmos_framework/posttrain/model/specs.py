# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Declarative plans that make the parallelize/weight-load primitives model-agnostic.

A model body DECLARES a ShardPlan (what to FSDP-wrap) and a LoadPlan (how to import a
pretrained checkpoint); the primitives (model/parallelize.py, model/weight_loader.py)
CONSUME them and stay free of any model-specific knowledge. Named LoadPlan (not
*WeightMap*) to avoid confusion with the weight-sync WeightMapper (a different, runtime
trainer->rollout remap)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class ShardPlan:
    """What `parallelize()` needs. `block_types`: module types that are the repeated
    shardable blocks (leaf-first fully_shard); the root wrap (embeddings/norm/head) is
    whatever remains. `param_dtype`/`reduce_dtype` -> MixedPrecisionPolicy; `param_dtype`
    is the compute dtype the FSDP parameter cast delivers, or None when the wrap delivers
    none (`model/precision.py::fsdp_param_dtype`, autocast mode)."""

    block_types: tuple[type, ...]
    param_dtype: torch.dtype | None
    reduce_dtype: torch.dtype | None


@dataclass(frozen=True)
class LoadPlan:
    """What `load_weights()` needs. `rename`: ckpt-key -> model-key (None = identity).
    `skip_patterns`: regex on the resolved model key -> skip-copy AND tolerate-absent."""

    rename: Callable[[str], str] | None = None
    skip_patterns: tuple[str, ...] = ()
