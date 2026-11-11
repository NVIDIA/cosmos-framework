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

Not RL. `n_mini == 1`: the forward is stochastic, so the pi_old ahead pass must not run.
TOKEN_SUM accumulation reconstructs each microbatch's masked sum and divides once
by the complete update's supervised count. Single-microbatch arithmetic is unchanged.
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
from cosmos_framework.posttrain.utils.dist import rank_mean_count, rank_sum_count, run_rank_converged

if TYPE_CHECKING:
    import torch

_METRIC_ADAPTER = MetricAdapter(
    names={"flow_bc_loss": "loss_avg", "flow_bc_masked_elements": "flow_bc_masked_elements"},
    internal_prefix="flow_bc_",
    phases=frozenset({"train", "val"}),
    # The trainer-facing divisor; publicly it would only duplicate `flow_bc_masked_elements`.
    ignored=frozenset({OBJECTIVE_WEIGHT_SUM}),
)


def make_objective(
    collate: Callable[[list[Trajectory]], dict],
    *,
    training_prefix: int,
    action_dim: int,
    sft_metrics: bool = False,
    allow_empty_training_padding: bool = False,
) -> Objective:
    """`collate` rebuilds the model inputs, including the supervision mask over `[:training_prefix, :action_dim]`;
    the two ints are recorded in the identity so a resume cannot silently change the loss."""

    if type(allow_empty_training_padding) is not bool:
        raise TypeError("allow_empty_training_padding must be bool")

    def fn(model: SupervisedFlowMatchingPolicy, batch: list[Trajectory]) -> tuple[torch.Tensor, dict[str, Any]]:
        # Decode/processor failures are local. Converge them before any peer
        # enters the policy's FSDP collectives; never wrap the collective-bearing
        # model forward itself in a recoverable preparation transaction.
        inputs = run_rank_converged(lambda: collate(batch), "flow_bc batch preparation")
        out = model.flow_matching_loss(inputs)
        masked = int(out.action_mask.float().sum().item())  # the mask arrives in bf16 under the parameter-cast mode
        denom = rank_mean_count(masked, out.loss.device)  # DP-mean supervised count; == masked outside a process group
        if denom <= 0:
            nonpadding = sum(
                not (
                    row.transitions[0].algo_extra.get("training_padding") is True
                    and row.transitions[0].algo_extra.get("validation_weight") == 0.0
                )
                for row in batch
            )
            if not allow_empty_training_padding or rank_sum_count(nonpadding, out.loss.device):
                raise ValueError("flow_bc batch contains no supervised action elements on any rank")
        # A padding-only chunk retains its model graph with a zero gradient. The
        # trainer rejects a wholly empty update, before AdamW or scheduler steps.
        loss = out.loss * (masked / (denom if denom > 0 else 1.0))
        metrics = {"flow_bc_loss": float(loss.detach()), "flow_bc_masked_elements": masked, OBJECTIVE_WEIGHT_SUM: denom}
        if sft_metrics:
            metrics["flow_bc_valid_action_count"] = masked
            metrics["flow_bc_valid_sequence_count"] = int(
                out.action_mask.reshape(out.action_mask.shape[0], -1).any(dim=1).sum().item()
            )
            if not model.net.training:
                # The SFT validation transaction combines local masked sums,
                # unlike training's average of DP-normalized rank losses.
                metrics["flow_bc_loss"] = float(out.loss.detach())
        return loss, metrics

    return Objective(
        fn=fn,
        reduction=Reduction.TOKEN_SUM,
        metric_adapter=MetricAdapter(
            names={
                **_METRIC_ADAPTER.names,
                "flow_bc_valid_action_count": "valid_action_count",
                "flow_bc_valid_sequence_count": "valid_sequence_count",
            },
            internal_prefix="flow_bc_",
            phases=frozenset({"train", "val"}),
            ignored=frozenset({OBJECTIVE_WEIGHT_SUM}),
        )
        if sft_metrics
        else _METRIC_ADAPTER,
        identity={
            "algorithm": "flow_bc",
            "training_prefix": int(training_prefix),
            "action_dim": int(action_dim),
            **({"finite_training_padding": "zero_weight_v1"} if allow_empty_training_padding else {}),
        },
    )
