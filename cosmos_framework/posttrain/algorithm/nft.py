# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""DiffusionNFT objective for online flow-matching policies.

The generated final latent is treated as ``x0``. At one sampled forward-process
point, policy and moving-old velocities define positive and negative clean-sample
estimates that are mixed by the sample's group-relative advantage. NFT retains
only the final latent—no UniPC history or denoising-step log-probability is part
of this objective.

The pure loss contract separates a periodically refreshed old-policy prediction,
which constructs the NFT direction, from an independent frozen-reference prediction,
which anchors the optional KL term. The online-flow capability returns both roles in
one typed result, while a missing frozen-reference prediction remains valid only when
KL is disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Literal

import torch

from cosmos_framework.posttrain.algorithm.annotators.advantage import require_advantages
from cosmos_framework.posttrain.algorithm.primitives.objective import MetricAdapter, Objective, Reduction
from cosmos_framework.posttrain.model.protocol import FlowCondition, FlowConditionMask, OnlineFlowPolicy
from cosmos_framework.posttrain.schema import WFM_CONDITIONING_KEY, Trajectory, WFMConditioningInput
from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed, run_rank_converged
from cosmos_framework.posttrain.utils.wfm import repeat_conditioning_batch

MODEL_BATCH_KEY = WFM_CONDITIONING_KEY


@dataclass(frozen=True)
class NFTObjectiveConfig:
    """Validated, algorithm-only DiffusionNFT knobs."""

    beta: float = 1.0
    kl_beta: float = 1.0
    policy_loss_scale: float = 5.0
    advantage_high: float = 5.0
    advantage_low: float = -5.0
    r_map: Literal["linear", "sigmoid"] = "linear"
    r_tau: float = 0.5
    uniform_timestep: bool = False

    def __post_init__(self) -> None:
        if not isfinite(self.beta) or self.beta <= 0:
            raise ValueError(f"nft beta must be finite and > 0, got {self.beta}")
        if not isfinite(self.kl_beta) or self.kl_beta < 0:
            raise ValueError(f"nft kl_beta must be finite and >= 0, got {self.kl_beta}")
        if not isfinite(self.policy_loss_scale) or self.policy_loss_scale <= 0:
            raise ValueError(f"nft policy_loss_scale must be finite and > 0, got {self.policy_loss_scale}")
        if not (isfinite(self.advantage_low) and isfinite(self.advantage_high)):
            raise ValueError("nft advantage bounds must be finite")
        if self.advantage_high <= self.advantage_low:
            raise ValueError(
                f"nft advantage_high must exceed advantage_low, got {self.advantage_high} <= {self.advantage_low}"
            )
        if self.r_map not in ("linear", "sigmoid"):
            raise ValueError(f"nft r_map must be 'linear' or 'sigmoid', got {self.r_map!r}")
        if self.r_map == "sigmoid" and (not isfinite(self.r_tau) or self.r_tau <= 0):
            raise ValueError(f"nft r_tau must be finite and > 0 for sigmoid mapping, got {self.r_tau}")


def optimality_weight(advantages: torch.Tensor, config: NFTObjectiveConfig) -> torch.Tensor:
    """Map group-relative advantages to NFT optimality weights ``r in [0,1]``."""
    if advantages.ndim != 1:
        raise ValueError(f"advantages must have shape [B], got {tuple(advantages.shape)}")
    if config.r_map == "sigmoid":
        return torch.sigmoid(advantages / config.r_tau)  # [B]
    clipped = advantages.clamp(config.advantage_low, config.advantage_high)  # [B]
    r = (clipped - config.advantage_low) / (config.advantage_high - config.advantage_low)  # [B]
    return r.clamp(0.0, 1.0)  # [B]


def optimality_weight_std(
    mean: float | torch.Tensor,
    second_moment: float | torch.Tensor,
) -> float:
    """Derive population std after data-parallel averaging of ``E[r]`` and ``E[r^2]``."""

    def scalar(value: float | torch.Tensor, name: str) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{name} must be scalar, got shape {tuple(value.shape)}")
            result = float(value.detach().cpu().item())  # []
        else:
            result = float(value)
        if not isfinite(result):
            raise ValueError(f"{name} must be finite, got {result}")
        return result

    mean_value = scalar(mean, "optimality-weight mean")
    second_moment_value = scalar(second_moment, "optimality-weight second moment")
    tolerance = 1e-6
    if not -tolerance <= mean_value <= 1.0 + tolerance:
        raise ValueError(f"optimality-weight mean must be in [0,1], got {mean_value}")
    if not -tolerance <= second_moment_value <= 1.0 + tolerance:
        raise ValueError(f"optimality-weight second moment must be in [0,1], got {second_moment_value}")
    variance = second_moment_value - mean_value**2
    if variance < -tolerance:
        raise ValueError(f"optimality-weight moments are inconsistent: E[r^2] - E[r]^2 = {variance}")
    return sqrt(max(variance, 0.0))


def nft_loss_terms(
    *,
    x0: torch.Tensor,
    x_t: torch.Tensor,
    t_effective: torch.Tensor,
    v_policy: torch.Tensor,
    v_old: torch.Tensor,
    v_ref: torch.Tensor | None,
    advantages: torch.Tensor,
    config: NFTObjectiveConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate NFT with moving-old direction and an optional frozen-reference KL anchor."""
    if x0.ndim < 2:
        raise ValueError(f"x0 must have a batch axis and at least one feature axis, got {tuple(x0.shape)}")
    expected = tuple(x0.shape)
    for name, tensor in (("x_t", x_t), ("v_policy", v_policy), ("v_old", v_old)):
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} != x0 shape {expected}")
    if v_ref is not None and tuple(v_ref.shape) != expected:
        raise ValueError(f"v_ref shape {tuple(v_ref.shape)} != x0 shape {expected}")
    if config.kl_beta > 0 and v_ref is None:
        raise ValueError("NFT with kl_beta>0 requires a frozen reference prediction")
    batch = int(x0.shape[0])
    if tuple(advantages.shape) != (batch,):
        raise ValueError(f"advantages must have shape ({batch},), got {tuple(advantages.shape)}")
    try:
        torch.broadcast_shapes(tuple(t_effective.shape), expected)
    except RuntimeError as error:
        raise ValueError(f"t_effective shape {tuple(t_effective.shape)} is not broadcastable to {expected}") from error

    old = v_old.detach()  # [B,...]
    r = optimality_weight(advantages, config).to(device=x0.device, dtype=x0.dtype)  # [B]
    reduce_dims = tuple(range(1, x0.ndim))
    v_positive = config.beta * v_policy + (1.0 - config.beta) * old  # [B,...]
    v_negative = (1.0 + config.beta) * old - config.beta * v_policy  # [B,...]
    x0_positive = x_t - t_effective * v_positive  # [B,...]
    x0_negative = x_t - t_effective * v_negative  # [B,...]
    positive_residual = x0_positive - x0  # [B,...]
    negative_residual = x0_negative - x0  # [B,...]
    with torch.no_grad():
        positive_weight = positive_residual.abs().mean(dim=reduce_dims, keepdim=True).clamp_min(1e-5)  # [B,1,...]
        negative_weight = negative_residual.abs().mean(dim=reduce_dims, keepdim=True).clamp_min(1e-5)  # [B,1,...]
    positive_error = (positive_residual.square() / positive_weight).mean(dim=reduce_dims)  # [B]
    negative_error = (negative_residual.square() / negative_weight).mean(dim=reduce_dims)  # [B]
    policy_per_sample = (r * positive_error + (1.0 - r) * negative_error) / config.beta  # [B]
    policy_loss = policy_per_sample.mean() * config.policy_loss_scale  # []
    if v_ref is None or config.kl_beta == 0:
        kl_loss = policy_loss.new_zeros(())  # []
    else:
        kl_loss = (v_policy - v_ref.detach()).square().mean()  # []
    loss = policy_loss + config.kl_beta * kl_loss  # []
    # Preserve the quantized r values used by the loss, but accumulate their
    # cancellation-sensitive telemetry moments in fp32.
    r_metrics = r.detach().float()  # [B]
    # Online NFT v1 has one sample per DP rank and the trainer averages scalar metrics across ranks.
    # A rank-local r min/max would therefore reduce to the global MEAN while retaining a min/max
    # label. Report only the merge-safe mean here; global reward-shape facts come from the estimator.
    metrics = {
        "nft_loss": loss.detach(),
        "nft_policy_loss": policy_loss.detach(),
        "nft_kl_loss": kl_loss.detach(),
        "nft_advantage_mean": advantages.mean().detach(),
        "nft_r_mean": r_metrics.mean(),  # []
        # With one sample per rank, averaging rank-local r.std() would always
        # report zero. E[r] and E[r^2] are merge-safe; the entry derives the
        # population standard deviation after the trainer's DP reduction.
        "nft_r_second_moment": r_metrics.square().mean(),  # []
        "nft_positive_error": positive_error.mean().detach(),
        "nft_negative_error": negative_error.mean().detach(),
    }
    return loss, metrics


def _latent_batch(latents: list[torch.Tensor]) -> torch.Tensor:
    if not latents:
        raise ValueError("NFT objective got no generated latents")
    if any(latent.ndim < 2 or latent.shape[0] != 1 for latent in latents):
        raise ValueError(
            "each NFT trajectory action must keep a leading singleton sample axis; "
            f"got {[tuple(latent.shape) for latent in latents]}"
        )
    shapes = {tuple(latent.shape[1:]) for latent in latents}
    if len(shapes) != 1:
        raise ValueError(f"NFT currently requires uniform latent shapes, got {sorted(shapes)}")
    return torch.cat(latents, dim=0)  # [B,...]


def unpack_nft_batch(trajectories: list[Trajectory]) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Validate and unpack generated-sample trajectories for the objective."""
    require_advantages(trajectories)
    conditioning_inputs: list[WFMConditioningInput] = []
    latents: list[torch.Tensor] = []
    advantages: list[float] = []
    for index, trajectory in enumerate(trajectories):
        if len(trajectory.transitions) != 1:
            raise ValueError(f"NFT trajectory {index} must contain exactly one terminal transition")
        transition = trajectory.transitions[0]
        conditioning = transition.obs.representations.get(MODEL_BATCH_KEY)
        if not isinstance(conditioning, WFMConditioningInput):
            raise TypeError(
                f"NFT trajectory {index} needs obs.representations[{MODEL_BATCH_KEY!r}] as WFMConditioningInput"
            )
        if not isinstance(transition.action, torch.Tensor):
            raise TypeError(f"NFT trajectory {index} action must be the generated latent tensor")
        conditioning_inputs.append(conditioning)
        latents.append(transition.action)
        advantages.append(float(transition.advantage))
    x0 = _latent_batch(latents)  # [B,...]
    advantage_tensor = torch.tensor(advantages, device=x0.device, dtype=torch.float32)  # [B]
    source_ids = {conditioning.id for conditioning in conditioning_inputs}
    if len(source_ids) != 1:
        raise ValueError(f"NFT objective batch must contain one conditioning group, got {sorted(source_ids)}")
    template_identities = {conditioning.template_identity for conditioning in conditioning_inputs}
    if len(template_identities) != 1:
        raise ValueError(
            "NFT objective batch must contain one conditioning template, "
            f"got {sorted(template_identities)} for source {next(iter(source_ids))!r}"
        )
    source_batch = repeat_conditioning_batch(conditioning_inputs[0], len(conditioning_inputs))
    return source_batch, x0, advantage_tensor


def nft_objective(
    model: OnlineFlowPolicy,
    trajectories: list[Trajectory],
    config: NFTObjectiveConfig,
    *,
    sigma: torch.Tensor | None = None,
    epsilon: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Prepare rank-local inputs safely, run policy/old/reference forwards, and evaluate NFT.

    This narrow pre-forward convergence boundary belongs here because trajectory validation,
    conditioning encode/masking, and stochastic-input selection happen after the trainer's generic
    device-transfer guard but before the policy/old/reference FSDP forwards. A rank-local preparation error
    must reach every peer before any surviving rank enters those model collectives. The trainer
    continues to own generic loss, backward, and optimizer convergence.
    """
    reference_error = (
        ValueError("online NFT with kl_beta>0 requires an independent frozen reference network")
        if config.kl_beta > 0 and model.net_ref is None
        else None
    )
    raise_if_any_rank_failed(reference_error, "NFT frozen-reference requirement")
    prepared: (
        tuple[
            torch.Tensor,
            torch.Tensor,
            FlowCondition,
            FlowConditionMask,
            torch.Tensor,
            torch.Tensor,
        ]
        | None
    ) = None
    preparation_error: Exception | None = None
    try:
        if (sigma is None) != (epsilon is None):
            raise ValueError("NFT sigma and epsilon overrides must be supplied together or both omitted")
        source_batch, x0, advantages = unpack_nft_batch(trajectories)
        _unused_source_x0, cond = model.encode_and_condition(source_batch)
        condition_mask = model.condition_mask(cond)
        drawn_sigma: torch.Tensor | None = None
        drawn_epsilon: torch.Tensor | None = None
        if sigma is None or epsilon is None:
            drawn_sigma, drawn_epsilon = model.draw_nft_sigma_epsilon(x0.shape, cond)
        sigma_value = (
            drawn_sigma
            if sigma is None
            else sigma.to(device=x0.device, dtype=drawn_sigma.dtype if drawn_sigma is not None else x0.dtype)
        )
        epsilon_value = (
            drawn_epsilon
            if epsilon is None
            else epsilon.to(device=x0.device, dtype=drawn_epsilon.dtype if drawn_epsilon is not None else x0.dtype)
        )
        if sigma_value is None or epsilon_value is None:
            raise RuntimeError("NFT stochastic-input selection produced a missing sigma or epsilon")
        if config.uniform_timestep and sigma is None:
            # Pinned reference draw order deliberately keeps the ordinary epsilon+sigma draw above,
            # then replaces only the effective timestep with this uniform draw.
            sigma_value = torch.rand(
                sigma_value.shape,
                device=sigma_value.device,
                dtype=sigma_value.dtype,
            ).clamp(1e-3, 1.0 - 1e-3)  # adapter-defined sigma shape
        prepared = (x0, advantages, cond, condition_mask, sigma_value, epsilon_value)
    except Exception as error:  # noqa: BLE001 — converge before policy/reference FSDP forwards
        preparation_error = error
    raise_if_any_rank_failed(preparation_error, "NFT objective preparation")
    if prepared is None:
        raise RuntimeError("NFT objective preparation converged without prepared inputs")
    x0, advantages, cond, condition_mask, sigma_value, epsilon_value = prepared
    predictions = model.nft_velocity_predictions(
        x0=x0,
        cond=cond,
        sigma=sigma_value,
        epsilon=epsilon_value,
        condition_mask=condition_mask,
    )

    def finalize() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, metrics = nft_loss_terms(
            x0=x0,
            x_t=predictions.x_t,
            t_effective=predictions.t_effective,
            v_policy=predictions.v_policy,
            v_old=predictions.v_old,
            v_ref=predictions.v_ref,
            advantages=advantages,
            config=config,
        )
        reward_values = [trajectory.final_reward for trajectory in trajectories]
        if all(value is not None for value in reward_values):
            rewards = torch.tensor(
                [float(value) for value in reward_values],
                device=loss.device,
                dtype=torch.float32,
            )  # [B]
            metrics["nft_reward_mean"] = rewards.mean()  # []
        return loss, metrics

    return run_rank_converged(finalize, "NFT objective finalization")


def make_objective(config: NFTObjectiveConfig) -> Objective:
    """Bind NFT to the generic diffusion-trainer objective contract."""

    def objective(
        model: OnlineFlowPolicy, trajectories: list[Trajectory]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return nft_objective(model, trajectories, config)

    metric_adapter = MetricAdapter(
        internal_prefix="nft_",
        names={
            "nft_loss": "loss_avg",
            "nft_policy_loss": "policy_loss",
            "nft_kl_loss": "kl_loss",
            "nft_advantage_mean": "advantage_mean",
            "nft_r_mean": "optimality_mean",
            "nft_r_second_moment": "optimality_second_moment",
            "nft_positive_error": "positive_error",
            "nft_negative_error": "negative_error",
            "nft_reward_mean": "reward_mean",
        },
        phases=frozenset({"train"}),
    )
    return Objective(
        objective,
        reduction=Reduction.MEAN,
        metric_adapter=metric_adapter,
        identity={
            "algorithm": "diffusion_nft",
            "beta": config.beta,
            "kl_beta": config.kl_beta,
            "policy_loss_scale": config.policy_loss_scale,
            "advantage_high": config.advantage_high,
            "advantage_low": config.advantage_low,
            "r_map": config.r_map,
            "r_tau": config.r_tau,
            "uniform_timestep": config.uniform_timestep,
        },
    )
