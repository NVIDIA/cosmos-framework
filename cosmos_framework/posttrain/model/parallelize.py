# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""General FSDP2 wrap driven by a ShardPlan over an injected ParallelDims.

Model-agnostic: it shards the module types named in the plan, then the root; it never
constructs a mesh (the model config plus actor/torchrun runtime do that). No-op on single device so
the same call preserves today's single-GPU path. Does NOT materialize meta tensors — the caller
(HFModel.setup) does that between parallelize() and load. See design spec section 4.2."""

from __future__ import annotations

import torch.nn as nn

from cosmos_framework.posttrain.model.parallel_dims import ParallelDims
from cosmos_framework.posttrain.model.specs import ShardPlan


def parallelize(model: nn.Module, plan: ShardPlan, pdims: ParallelDims | None) -> None:
    # Wrap when EITHER data-parallel axis is live. Gating on `dp_shard_enabled` alone skipped the
    # REPLICATE-ONLY topology (dp_replicate=D>1, dp_shard=1) — and skipping `fully_shard` there means
    # no FSDP wrapper and no gradient all-reduce across the model's replicate axis. The model-owned
    # mesh would be built correctly and then discarded here. dp_shard=1 is the degenerate HSDP case
    # (replication without sharding); FSDP2 handles it, and the reduce over the replicate axis is
    # what makes it worth wrapping at all.
    if pdims is None or not (pdims.dp_shard_enabled or pdims.dp_replicate > 1):
        # NO WRAP => NO MixedPrecisionPolicy => `plan.param_dtype` IS NOT DELIVERED. The compute
        # dtype is a property of the FSDP cast; without a wrapper the forward simply runs in the
        # parameter's STORAGE dtype. That is invisible when the two agree (the pre-split world, and
        # every recipe that leaves `rl.master_dtype` unset), and silently wrong when they do not:
        # `master_dtype="float32", param_dtype="bfloat16"` on one GPU stores fp32 and computes in
        # fp32, ignoring the requested bf16 — measured, and against a reference that DOES wrap
        # (cosmos-rl wraps unconditionally) that is an unaccounted precision difference.
        #
        # Refused rather than emulated: this is the `mixed_precision_mode="fsdp"` contract. Silently
        # substituting `torch.autocast` would apply a PER-OP dtype policy (layernorm/softmax stay fp32)
        # where FSDP casts the PARAMETER, so the same config would compute differently at dp_shard=1
        # than at dp_shard=2. The explicit alternative is `mixed_precision_mode="autocast"`
        # (`model/precision.py`): the plan then carries `param_dtype=None` -- the wrap delivers no
        # compute dtype, so there is nothing here that could go undelivered.
        if plan.param_dtype is not None:
            mismatched = next(
                (p for p in model.parameters() if p.is_floating_point() and p.dtype != plan.param_dtype),
                None,
            )
            if mismatched is not None:
                raise ValueError(
                    f"rl.param_dtype={plan.param_dtype} cannot be honoured without an FSDP wrap: this "
                    f"mesh has dp_shard=1 and dp_replicate=1, so no MixedPrecisionPolicy is attached and "
                    f"the forward would run in the storage dtype ({mismatched.dtype}) instead. Either "
                    f"set rl.master_dtype == rl.param_dtype (storage == compute, nothing to cast), or "
                    f"run on a mesh with dp_shard>1 / dp_replicate>1, or set rl.mixed_precision_mode='autocast'."
                )
        return  # single device only -> nothing to shard and nothing to reduce
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mesh = pdims.dp_mesh if pdims.dp_replicate > 1 else pdims.dp_shard_mesh
    # `param_dtype=None` (autocast mode) is FSDP2's documented "no cast": no parameter cast, and
    # `cast_forward_inputs` is disarmed with it, so the wrap is dtype-transparent and only `reduce_dtype`
    # takes effect. Carrying the storage dtype instead would keep the forward-input cast live.
    kwargs = {
        "mesh": mesh,
        "mp_policy": MixedPrecisionPolicy(param_dtype=plan.param_dtype, reduce_dtype=plan.reduce_dtype),
    }
    # `mod is not model` excludes the root: model.modules() always yields `model` first, so if
    # type(model) matches plan.block_types (single-block / custom models, TinyDecoder-style test
    # helpers) it would be sharded here AND again by the explicit root fully_shard below — and
    # fully_shard on an already-sharded FSDP2 module is undefined + corrupts param state. The root
    # is always sharded exactly once, by the explicit call.
    blocks = [mod for mod in model.modules() if mod is not model and isinstance(mod, plan.block_types)]
    for block in reversed(blocks):  # leaf-first
        fully_shard(block, **kwargs)
    fully_shard(model, **kwargs)  # root: embeddings / norm / head / projector (sharded once)
