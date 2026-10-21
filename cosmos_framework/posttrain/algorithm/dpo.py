# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Model-agnostic Flow-DPO objective.

Flow-DPO computes a policy/reference denoise-error delta for each side of a
preference pair. Winner and loser share one noise trajectory, while held-out
validation gives their stochastic encoders separate deterministic streams.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F

from cosmos_framework.posttrain.algorithm.primitives.objective import (
    OBJECTIVE_WEIGHT_SUM,
    MetricAdapter,
    Objective,
    PerSampleReducer,
    Reduction,
    ValidationBatchMetadata,
    read_validation_batch_metadata,
)
from cosmos_framework.posttrain.data.packer.preference import unpack_preferences
from cosmos_framework.posttrain.model.flow import converged_denoise_error
from cosmos_framework.posttrain.model.protocol import (
    FlowCondition,
    FlowConditionMask,
    FlowMatchingPolicy,
    PreferenceFlowPolicy,
)
from cosmos_framework.posttrain.schema import Trajectory
from cosmos_framework.posttrain.utils.dist import run_rank_converged
from cosmos_framework.posttrain.utils.seed import derive_u64_seeds

_DPO_METRIC_NAMES: dict[str, str] = {
    "dpo_loss": "loss_avg",
    "dpo_reward_margin": "dpo_reward_margin",
    "dpo_reward_winner": "dpo_reward_winner",
    "dpo_reward_loser": "dpo_reward_loser",
    "dpo_accuracy": "dpo_accuracy",
    "dpo_err_pol_w": "dpo_err_pol_w",
    "dpo_err_ref_w": "dpo_err_ref_w",
    "dpo_err_pol_l": "dpo_err_pol_l",
    "dpo_err_ref_l": "dpo_err_ref_l",
    "dpo_raw_model_loss": "dpo_raw_model_loss",
    "dpo_raw_ref_loss": "dpo_raw_ref_loss",
}
_DPO_METRIC_ADAPTER = MetricAdapter(
    names=_DPO_METRIC_NAMES,
    internal_prefix="dpo_",
    phases=frozenset({"train", "val"}),
    ignored=frozenset({OBJECTIVE_WEIGHT_SUM}),
)
_WINNER_ENCODE_SEED_DOMAIN = b"posttrain.dpo.encode.v1\0winner\0"
_LOSER_ENCODE_SEED_DOMAIN = b"posttrain.dpo.encode.v1\0loser\0"


def dpo_objective(
    model: FlowMatchingPolicy,
    winner_batch: dict[str, Any],
    loser_batch: dict[str, Any],
    beta: float,
    sample_weights: list[float] | None = None,
    sample_seeds: list[int] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Encode a preference batch, run four denoisers, and apply Flow-DPO."""

    def prepare_pair() -> tuple[
        torch.Tensor,
        FlowCondition,
        torch.Tensor,
        FlowCondition,
        FlowConditionMask,
        torch.Tensor,
        torch.Tensor,
    ]:
        if not isinstance(model, PreferenceFlowPolicy):
            raise TypeError(
                "[DPO] preference objectives require the model adapter to implement "
                "PreferenceFlowPolicy.assert_same_condition_context()"
            )
        if sample_seeds is None:
            # Preserve the ordinary training RNG path and adapter call signature.
            x0_w, cond_w = model.encode_and_condition(winner_batch)  # x0_w: [B,...]
            x0_l, cond_l = model.encode_and_condition(loser_batch)  # x0_l: [B,...]
        else:
            x0_w, cond_w = model.encode_and_condition(
                winner_batch,
                sample_encode_seeds=derive_u64_seeds(sample_seeds, domain=_WINNER_ENCODE_SEED_DOMAIN),
            )  # x0_w: [B,...]
            x0_l, cond_l = model.encode_and_condition(
                loser_batch,
                sample_encode_seeds=derive_u64_seeds(sample_seeds, domain=_LOSER_ENCODE_SEED_DOMAIN),
            )  # x0_l: [B,...]
        model.assert_same_condition_context(cond_w, cond_l)
        if tuple(x0_w.shape) != tuple(x0_l.shape):
            raise ValueError(
                f"[DPO] x0 shape mismatch: winner={tuple(x0_w.shape)} loser={tuple(x0_l.shape)}; "
                "both sides must share one noise trajectory"
            )

        condition_mask = model.condition_mask(cond_w)  # [B,...] or None
        if sample_seeds is None:
            sigma, epsilon = model.draw_sigma_epsilon(x0_w.shape, cond_w)  # sigma: [B,1], epsilon: [B,...]
        else:
            sigma, epsilon = model.draw_sigma_epsilon(
                x0_w.shape,
                cond_w,
                sample_seeds=sample_seeds,
            )  # sigma: [B,1], epsilon: [B,...]
        return x0_w, cond_w, x0_l, cond_l, condition_mask, sigma, epsilon

    x0_w, cond_w, x0_l, cond_l, condition_mask, sigma, epsilon = run_rank_converged(
        prepare_pair,
        "DPO objective preparation",
    )

    # Collective order is load-bearing: policy/reference x winner/loser.
    requests: tuple[tuple[str, torch.nn.Module, torch.Tensor, FlowCondition], ...] = (
        ("policy-winner", model.net, x0_w, cond_w),
        ("reference-winner", cast(torch.nn.Module, model.net_ref), x0_w, cond_w),
        ("policy-loser", model.net, x0_l, cond_l),
        ("reference-loser", cast(torch.nn.Module, model.net_ref), x0_l, cond_l),
    )
    errors = [
        converged_denoise_error(
            model,
            net=net,
            x0=x0,
            cond=cond,
            sigma=sigma,
            epsilon=epsilon,
            condition_mask=condition_mask,
            context=f"DPO {label}",
        )
        for label, net, x0, cond in requests
    ]  # each [B]
    err_pol_w, err_ref_w, err_pol_l, err_ref_l = errors  # each [B]

    return run_rank_converged(
        lambda: _flow_dpo_loss(
            err_pol_w,
            err_pol_l,
            err_ref_w,
            err_ref_l,
            beta,
            sample_weights=sample_weights,
        ),
        "DPO objective finalization",
    )


def _flow_dpo_loss(
    err_pol_w: torch.Tensor,
    err_pol_l: torch.Tensor,
    err_ref_w: torch.Tensor,
    err_ref_l: torch.Tensor,
    beta: float,
    sample_weights: list[float] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:  # errors: [B], returns loss: []
    """Compute the closed-form Flow-DPO loss from four per-sample errors."""
    reward_winner = -0.5 * beta * (err_pol_w - err_ref_w)  # [B]
    reward_loser = -0.5 * beta * (err_pol_l - err_ref_l)  # [B]
    margin = reward_winner - reward_loser  # [B]
    per_sample_loss = -F.logsigmoid(margin)  # [B]
    reducer = PerSampleReducer.from_weights(per_sample_loss, sample_weights, context="DPO")
    loss = reducer.mean(per_sample_loss)  # []

    # Ties count as chance. Non-finite values propagate to the trainer's rank-consistent guard.
    per_sample_accuracy = (margin > 0).float() + 0.5 * (margin == 0).float()  # [B]
    metrics: dict[str, Any] = {
        "dpo_loss": loss.detach(),  # []
        "dpo_reward_margin": reducer.mean(margin).detach(),  # []
        "dpo_reward_winner": reducer.mean(reward_winner).detach(),  # []
        "dpo_reward_loser": reducer.mean(reward_loser).detach(),  # []
        "dpo_accuracy": reducer.mean(per_sample_accuracy).detach(),  # []
        "dpo_err_pol_w": reducer.mean(err_pol_w).detach(),  # []
        "dpo_err_ref_w": reducer.mean(err_ref_w).detach(),  # []
        "dpo_err_pol_l": reducer.mean(err_pol_l).detach(),  # []
        "dpo_err_ref_l": reducer.mean(err_ref_l).detach(),  # []
        "dpo_raw_model_loss": (0.5 * (reducer.mean(err_pol_w) + reducer.mean(err_pol_l))).detach(),  # []
        "dpo_raw_ref_loss": (0.5 * (reducer.mean(err_ref_w) + reducer.mean(err_ref_l))).detach(),  # []
    }
    if reducer.weight_sum is not None:
        metrics[OBJECTIVE_WEIGHT_SUM] = reducer.weight_sum.detach()  # []
    return loss, metrics


def make_objective(beta: float) -> Objective:
    """Bind the DPO coefficient to the generic trainer objective contract."""

    def objective(
        model: FlowMatchingPolicy,
        trajectories: list[Trajectory],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        def prepare_batch() -> tuple[dict[str, Any], dict[str, Any], ValidationBatchMetadata]:
            winner_batch, loser_batch = unpack_preferences(trajectories)
            metadata = read_validation_batch_metadata(trajectories)
            return winner_batch, loser_batch, metadata

        winner_batch, loser_batch, metadata = run_rank_converged(
            prepare_batch,
            "DPO objective batch preparation",
        )
        return dpo_objective(
            model,
            winner_batch,
            loser_batch,
            beta,
            sample_weights=metadata.sample_weights,
            sample_seeds=metadata.sample_seeds,
        )

    return Objective(
        objective,
        reduction=Reduction.MEAN,
        metric_adapter=_DPO_METRIC_ADAPTER,
        identity={"algorithm": "dpo", "beta": float(beta)},
    )
