# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Flow-matching behaviour cloning as an `Objective`: the model's own supervised flow-matching loss, driven by
`LLMTrainer`. Model-agnostic: `collate` turns the batch into model inputs (a packer or a recipe partial), and the
model satisfies `SupervisedFlowMatchingPolicy` (`model/protocol.py`): `flow_matching_loss(inputs)` returns the
masked-mean loss and the supervision mask.

Normalization is data-parallel global. The model's loss is a rank-local masked mean, and FSDP averages rank
gradients with equal weight, which equals the global masked mean only when every rank supervises the same number
of elements. So each rank rescales its loss to `masked_sum / dp_mean_count` (`Reduction.TOKEN_SUM`, with
`OBJECTIVE_WEIGHT_SUM` = the DP-mean masked count): the mean over ranks is then the global masked mean, and on
equal shards the scale is exactly 1.0, so the single-GPU parity gate stays bitwise. Preconditions: every rank has
at least one sample (Isaac's collator needs one, and dispatch hands ranks equal shares), and the count average is
exact for a power-of-two world size, the only topologies the gate has measured.

Not RL. Two structural constraints: `n_mini == 1` (the forward is stochastic, so the pi_old ahead pass must not run)
and `grad_accum == 1` (a reference trainer divides the loss by 1 too, keeping a per-step parity comparison exact).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from cosmos_framework.posttrain.algorithm.primitives.objective import (
    OBJECTIVE_WEIGHT_SUM,
    MetricAdapter,
    Objective,
    Reduction,
)
from cosmos_framework.posttrain.model.protocol import SupervisedFlowMatchingPolicy
from cosmos_framework.posttrain.schema.trajectory import Trajectory
from cosmos_framework.posttrain.utils.dist import rank_mean_count

if TYPE_CHECKING:
    import torch

_METRIC_ADAPTER = MetricAdapter(
    names={"flow_bc_loss": "loss_avg", "flow_bc_masked_elements": "flow_bc_masked_elements"},
    internal_prefix="flow_bc_",
    phases=frozenset({"train", "val"}),
    # The trainer-facing divisor; publicly it would only duplicate `flow_bc_masked_elements`.
    ignored=frozenset({OBJECTIVE_WEIGHT_SUM}),
)


def make_objective(collate: Callable[[list[Trajectory]], dict], *, training_prefix: int, action_dim: int) -> Objective:
    """`collate` rebuilds the model inputs, including the supervision mask over `[:training_prefix, :action_dim]`;
    the two ints are recorded in the identity so a resume cannot silently change the loss."""

    def fn(model: SupervisedFlowMatchingPolicy, batch: list[Trajectory]) -> tuple[torch.Tensor, dict[str, Any]]:
        out = model.flow_matching_loss(collate(batch))
        masked = int(out.action_mask.float().sum().item())  # the mask arrives in bf16 under the parameter-cast mode
        denom = rank_mean_count(masked, out.loss.device)  # DP-mean supervised count; == masked outside a process group
        loss = out.loss * (masked / denom)  # x/x is exactly 1.0 in IEEE 754: equal shards stay bitwise identical
        metrics = {"flow_bc_loss": float(loss.detach()), "flow_bc_masked_elements": masked, OBJECTIVE_WEIGHT_SUM: denom}
        return loss, metrics

    return Objective(
        fn=fn,
        reduction=Reduction.TOKEN_SUM,
        metric_adapter=_METRIC_ADAPTER,
        identity={"algorithm": "flow_bc", "training_prefix": int(training_prefix), "action_dim": int(action_dim)},
    )
