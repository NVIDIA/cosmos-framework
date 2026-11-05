# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Trainer-facing PPO policy and value objectives over prepared Trajectories."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch

from cosmos_framework.posttrain.algorithm.ppo.jsd import topk_residual_jsd
from cosmos_framework.posttrain.algorithm.ppo.losses import (
    categorical_entropy,
    masked_mean,
    ppo_policy_loss,
    ppo_value_loss,
)
from cosmos_framework.posttrain.algorithm.ppo.views import (
    PPOTrainingEpisode,
    processor_inputs_for_role,
    trajectory_to_training_episode,
)
from cosmos_framework.posttrain.algorithm.primitives.objective import (
    OBJECTIVE_WEIGHT_SUM,
    MetricAdapter,
    Objective,
    Reduction,
)
from cosmos_framework.posttrain.model.value import (
    action_state_values,
    extract_causal_logits,
    forward_causal_model,
    model_device,
    response_action_logits,
    sampled_action_logprobs,
)
from cosmos_framework.posttrain.schema import Trajectory
from cosmos_framework.posttrain.utils.dist import rank_identical, rank_mean_count, run_rank_converged

_POLICY_METRIC_ADAPTER = MetricAdapter(
    internal_prefix="ppo_",
    names={
        "ppo_policy_total_loss": "policy_total_loss",
        "ppo_policy_loss": "policy_loss",
        "ppo_clip_fraction": "clip_fraction",
        "ppo_ratio_mean": "ratio_mean",
        "ppo_entropy": "entropy",
        "ppo_jsd_loss": "jsd_loss",
        "ppo_valid_tokens": "valid_token_count",
        OBJECTIVE_WEIGHT_SUM: "weight_sum",
    },
    phases=frozenset({"train"}),
    weighted_by={
        "ppo_policy_total_loss": "ppo_valid_tokens",
        "ppo_policy_loss": "ppo_valid_tokens",
        "ppo_clip_fraction": "ppo_valid_tokens",
        "ppo_ratio_mean": "ppo_valid_tokens",
        "ppo_entropy": "ppo_valid_tokens",
        "ppo_jsd_loss": "ppo_valid_tokens",
    },
)
_VALUE_METRIC_ADAPTER = MetricAdapter(
    internal_prefix="ppo_",
    names={
        "ppo_value_loss": "value_loss",
        "ppo_critic_loss": "critic_loss",
        "ppo_value_clip_fraction": "value_clip_fraction",
        "ppo_valid_tokens": "valid_token_count",
        OBJECTIVE_WEIGHT_SUM: "weight_sum",
    },
    phases=frozenset({"train"}),
    weighted_by={
        "ppo_value_loss": "ppo_valid_tokens",
        "ppo_critic_loss": "ppo_valid_tokens",
        "ppo_value_clip_fraction": "ppo_valid_tokens",
    },
)


@dataclass(frozen=True)
class _PreparedEpisode:
    item: PPOTrainingEpisode
    inputs: dict[str, Any]


@dataclass(frozen=True)
class _PolicyTerms:
    current_logprobs: torch.Tensor  # [T]
    old_logprobs: torch.Tensor  # [T]
    advantages: torch.Tensor  # [T]
    action_mask: torch.Tensor  # [T]
    entropy_logits: torch.Tensor | None  # [T,V] or None
    jsd_values: torch.Tensor | None  # [T] or None


@dataclass(frozen=True)
class _ValueTerms:
    current_values: torch.Tensor  # [T]
    old_values: torch.Tensor  # [T]
    returns: torch.Tensor  # [T]
    action_mask: torch.Tensor  # [T]


def _require_finite_coefficient(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    coefficient = float(value)
    if not math.isfinite(coefficient) or (coefficient <= 0.0 if positive else coefficient < 0.0):
        relation = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {relation}, got {value!r}")
    return coefficient


def _require_finite_outputs(loss: torch.Tensor, metrics: Mapping[str, Any]) -> None:
    """Reject a finite-input coefficient combination that overflowed before backward."""
    values = {"objective_loss": loss, **metrics}
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise TypeError(f"PPO objective metric {name!r} must be scalar, got shape {tuple(value.shape)!r}")
            value = value.detach().cpu().item()
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"PPO objective metric {name!r} is non-finite: {value!r}")


def _prepare_episodes(
    model: Any, trajectories: list[Trajectory], role: str
) -> tuple[list[_PreparedEpisode], int, float, torch.device]:
    """Validate already-prepared annotations and materialize model inputs before forward."""

    def prepare() -> tuple[list[_PreparedEpisode], int, torch.device]:
        device = model_device(model)
        if not trajectories:
            raise ValueError(f"PPO {role} objective received an empty trajectory batch")
        prepared: list[_PreparedEpisode] = []
        valid_tokens = 0
        for trajectory in trajectories:
            if not isinstance(trajectory, Trajectory):
                raise TypeError(f"PPO {role} objective requires Trajectory inputs")
            targets = trajectory.transitions[0].targets if len(trajectory.transitions) == 1 else None
            if not isinstance(targets, Mapping) or not isinstance(targets.get("ppo"), Mapping):
                raise ValueError("PPO objective trajectory is missing policy-computed training targets")
            item = trajectory_to_training_episode(trajectory)
            raw_inputs = processor_inputs_for_role(trajectory, role)
            inputs = {
                name: value.to(device=device) if isinstance(value, torch.Tensor) else value
                for name, value in raw_inputs.items()
            }
            prepared.append(_PreparedEpisode(item=item, inputs=inputs))
            valid_tokens += sum(item.scored.episode.action_mask)
        if valid_tokens < 1:
            raise ValueError(f"PPO {role} objective rank-local batch contains no valid action tokens")
        return prepared, valid_tokens, device

    prepared, valid_tokens, device = run_rank_converged(prepare, f"PPO {role} objective preparation")
    rank_identical(len(prepared), f"PPO {role} objective trajectory count")
    mean_valid_tokens = rank_mean_count(valid_tokens, device)
    if not math.isfinite(mean_valid_tokens) or mean_valid_tokens <= 0.0:
        raise ValueError("PPO objective data-parallel mean valid-token count must be finite and positive")
    return prepared, valid_tokens, mean_valid_tokens, device


def _policy_terms(
    prepared: _PreparedEpisode,
    output: Any,
    *,
    entropy_coef: float,
    jsd_coef: float,
    jsd_beta: float,
    temperature: float,
    device: torch.device,
) -> _PolicyTerms:
    item = prepared.item
    episode = item.scored.episode
    logits = extract_causal_logits(output)  # [1,P+T,V]
    action_logits = response_action_logits(  # [T,V]
        logits,
        len(episode.prompt_token_ids),
        len(episode.action_token_ids),
    )[0]
    action_ids = torch.tensor(episode.action_token_ids, dtype=torch.long, device=device)  # [T]
    current_logprobs = sampled_action_logprobs(
        action_logits,
        action_ids,
        temperature=temperature,
    )  # [T]
    old_logprobs = torch.tensor(episode.old_logprobs, dtype=torch.float32, device=device)  # [T]
    advantages = torch.tensor(item.advantages, dtype=torch.float32, device=device)  # [T]
    action_mask = torch.tensor(episode.action_mask, dtype=torch.bool, device=device)  # [T]
    entropy_logits = action_logits.float() / temperature if entropy_coef > 0.0 else None  # [T,V] or None
    jsd_values: torch.Tensor | None = None
    if jsd_coef > 0.0:
        if not episode.behavior_topk_token_ids:
            raise ValueError(f"JSD requires behavior-policy top-k token IDs for episode {episode.episode_id!r}")
        reference = item.scored.reference
        token_values: list[torch.Tensor] = []
        for token_index, (support_ids_row, support_probabilities_row, residual_probability) in enumerate(
            zip(
                reference.support_token_ids,
                reference.support_probabilities,
                reference.residual_probabilities,
                strict=True,
            )
        ):
            support_ids = torch.tensor(support_ids_row, dtype=torch.long, device=device)  # [K_union]
            support_probabilities = torch.tensor(  # [K_union]
                support_probabilities_row,
                dtype=torch.float64,
                device=device,
            )
            residual = torch.tensor(residual_probability, dtype=torch.float64, device=device)  # []
            token_values.append(
                topk_residual_jsd(
                    action_logits[token_index],
                    support_ids,
                    support_probabilities,
                    residual,
                    beta=jsd_beta,
                    temperature=temperature,
                )
            )
        jsd_values = torch.stack(token_values)  # [T]
    return _PolicyTerms(
        current_logprobs=current_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        action_mask=action_mask,
        entropy_logits=entropy_logits,
        jsd_values=jsd_values,
    )


def _policy_loss(
    terms: list[_PolicyTerms],
    *,
    policy_clip: float,
    entropy_coef: float,
    jsd_coef: float,
    valid_tokens: int,
    mean_valid_tokens: float,
) -> tuple[torch.Tensor, dict[str, Any]]:  # returns loss: []
    current = torch.cat([term.current_logprobs for term in terms])  # [N]
    old = torch.cat([term.old_logprobs for term in terms]).detach()  # [N]
    advantages = torch.cat([term.advantages for term in terms]).detach()  # [N]
    action_mask = torch.cat([term.action_mask for term in terms])  # [N]
    surrogate, policy_metrics = ppo_policy_loss(
        current,
        old,
        advantages,
        action_mask,
        policy_clip=policy_clip,
    )
    entropy = current.new_zeros(())  # []
    if entropy_coef > 0.0:
        entropy_inputs = [term.entropy_logits for term in terms]
        if any(value is None for value in entropy_inputs):
            raise RuntimeError("PPO policy entropy terms are incomplete")
        entropy = categorical_entropy(  # []
            torch.cat([value for value in entropy_inputs if value is not None]),  # [N,V]
            action_mask,
        )
    jsd_loss = current.new_zeros(())  # []
    if jsd_coef > 0.0:
        jsd_inputs = [term.jsd_values for term in terms]
        if any(value is None for value in jsd_inputs):
            raise RuntimeError("PPO policy JSD terms are incomplete")
        jsd_loss = masked_mean(  # []
            torch.cat([value for value in jsd_inputs if value is not None]),  # [N]
            action_mask,
        )
    total = surrogate - entropy_coef * entropy + jsd_coef * jsd_loss  # []
    loss = total * (float(valid_tokens) / mean_valid_tokens)  # []
    metrics: dict[str, Any] = {
        "ppo_policy_total_loss": total.detach(),  # []
        "ppo_policy_loss": policy_metrics["policy_loss"],  # []
        "ppo_clip_fraction": policy_metrics["clip_fraction"],  # []
        "ppo_ratio_mean": policy_metrics["ratio_mean"],  # []
        "ppo_entropy": entropy.detach(),  # []
        "ppo_jsd_loss": jsd_loss.detach(),  # []
        "ppo_valid_tokens": valid_tokens,
        OBJECTIVE_WEIGHT_SUM: mean_valid_tokens,
    }
    _require_finite_outputs(loss, metrics)
    return loss, metrics


def make_ppo_policy_objective(
    *,
    policy_clip: float = 0.2,
    entropy_coef: float = 0.0,
    jsd_coef: float = 0.0,
    jsd_beta: float = 0.5,
    temperature: float = 1.0,
) -> Objective:
    """Bind PPO policy coefficients to the shared trainer Objective contract."""
    policy_clip = _require_finite_coefficient("policy_clip", policy_clip, positive=True)
    if not math.isfinite(policy_clip) or not 0.0 < policy_clip < 1.0:
        raise ValueError(f"policy_clip must be finite and in (0, 1), got {policy_clip!r}")
    entropy_coef = _require_finite_coefficient("entropy_coef", entropy_coef)
    jsd_coef = _require_finite_coefficient("jsd_coef", jsd_coef)
    jsd_beta = _require_finite_coefficient("jsd_beta", jsd_beta)
    if jsd_beta > 1.0:
        raise ValueError(f"jsd_beta must be finite and in [0, 1], got {jsd_beta!r}")
    temperature = _require_finite_coefficient("temperature", temperature, positive=True)

    def objective(model: Any, trajectories: list[Trajectory]) -> tuple[torch.Tensor, dict[str, Any]]:
        prepared, valid_tokens, mean_valid_tokens, device = _prepare_episodes(model, trajectories, "policy")
        terms: list[_PolicyTerms] = []
        for episode_index, item in enumerate(prepared):
            output = forward_causal_model(model, item.inputs)
            term = run_rank_converged(
                lambda item=item, output=output: _policy_terms(
                    item,
                    output,
                    entropy_coef=entropy_coef,
                    jsd_coef=jsd_coef,
                    jsd_beta=jsd_beta,
                    temperature=temperature,
                    device=device,
                ),
                f"PPO policy objective post-forward episode {episode_index}",
            )
            terms.append(term)
        return run_rank_converged(
            lambda: _policy_loss(
                terms,
                policy_clip=policy_clip,
                entropy_coef=entropy_coef,
                jsd_coef=jsd_coef,
                valid_tokens=valid_tokens,
                mean_valid_tokens=mean_valid_tokens,
            ),
            "PPO policy objective finalization",
        )

    return Objective(
        objective,
        reduction=Reduction.TOKEN_SUM,
        metric_adapter=_POLICY_METRIC_ADAPTER,
        identity={
            "algorithm": "ppo_policy",
            "policy_clip": policy_clip,
            "entropy_coef": entropy_coef,
            "jsd_coef": jsd_coef,
            "jsd_beta": jsd_beta,
            "temperature": temperature,
        },
    )


def _value_terms(prepared: _PreparedEpisode, output: Any, *, device: torch.device) -> _ValueTerms:
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"PPO value model must return a tensor, got {type(output).__name__}")
    item = prepared.item
    episode = item.scored.episode
    aligned = action_state_values(  # [T+1]
        output,
        len(episode.prompt_token_ids),
        len(episode.action_token_ids),
    )[0]
    current_values = aligned[:-1]  # [T]
    old_values = torch.tensor(item.scored.critic.values[:-1], dtype=torch.float32, device=device)  # [T]
    returns = torch.tensor(item.returns, dtype=torch.float32, device=device)  # [T]
    action_mask = torch.tensor(episode.action_mask, dtype=torch.bool, device=device)  # [T]
    return _ValueTerms(
        current_values=current_values,
        old_values=old_values,
        returns=returns,
        action_mask=action_mask,
    )


def _value_loss(
    terms: list[_ValueTerms],
    *,
    value_clip: float,
    value_coef: float,
    valid_tokens: int,
    mean_valid_tokens: float,
) -> tuple[torch.Tensor, dict[str, Any]]:  # returns loss: []
    current = torch.cat([term.current_values for term in terms])  # [N]
    old = torch.cat([term.old_values for term in terms]).detach()  # [N]
    returns = torch.cat([term.returns for term in terms]).detach()  # [N]
    action_mask = torch.cat([term.action_mask for term in terms])  # [N]
    value_loss, value_metrics = ppo_value_loss(
        current,
        old,
        returns,
        action_mask,
        value_clip=value_clip,
    )
    critic_loss = value_coef * value_loss  # []
    loss = critic_loss * (float(valid_tokens) / mean_valid_tokens)  # []
    metrics: dict[str, Any] = {
        "ppo_value_loss": value_metrics["value_loss"],  # []
        "ppo_critic_loss": critic_loss.detach(),  # []
        "ppo_value_clip_fraction": value_metrics["value_clip_fraction"],  # []
        "ppo_valid_tokens": valid_tokens,
        OBJECTIVE_WEIGHT_SUM: mean_valid_tokens,
    }
    _require_finite_outputs(loss, metrics)
    return loss, metrics


def make_ppo_value_objective(*, value_clip: float = 0.2, value_coef: float = 1.0) -> Objective:
    """Bind PPO critic coefficients to the shared trainer Objective contract."""
    value_clip = _require_finite_coefficient("value_clip", value_clip, positive=True)
    value_coef = _require_finite_coefficient("value_coef", value_coef)

    def objective(model: Any, trajectories: list[Trajectory]) -> tuple[torch.Tensor, dict[str, Any]]:
        prepared, valid_tokens, mean_valid_tokens, device = _prepare_episodes(model, trajectories, "critic")
        terms: list[_ValueTerms] = []
        for episode_index, item in enumerate(prepared):
            output = model(**item.inputs)
            term = run_rank_converged(
                lambda item=item, output=output: _value_terms(item, output, device=device),
                f"PPO value objective post-forward episode {episode_index}",
            )
            terms.append(term)
        return run_rank_converged(
            lambda: _value_loss(
                terms,
                value_clip=value_clip,
                value_coef=value_coef,
                valid_tokens=valid_tokens,
                mean_valid_tokens=mean_valid_tokens,
            ),
            "PPO value objective finalization",
        )

    return Objective(
        objective,
        reduction=Reduction.TOKEN_SUM,
        metric_adapter=_VALUE_METRIC_ADAPTER,
        identity={"algorithm": "ppo_value", "value_clip": value_clip, "value_coef": value_coef},
    )


def _public_weighted_names(adapter: MetricAdapter, phase: str = "train") -> frozenset[str]:
    """Public keys of the metrics this adapter declares as token-weighted."""
    return frozenset(f"{phase}/{adapter.names[internal]}" for internal in adapter.weighted_by)


def _public_name(adapter: MetricAdapter, internal: str, phase: str = "train") -> str:
    return f"{phase}/{adapter.names[internal]}"


POLICY_WINDOW_WEIGHTED = _public_weighted_names(_POLICY_METRIC_ADAPTER)
VALUE_WINDOW_WEIGHTED = _public_weighted_names(_VALUE_METRIC_ADAPTER)
WINDOW_WEIGHT_KEY = _public_name(_POLICY_METRIC_ADAPTER, "ppo_valid_tokens")
WINDOW_SUMMED = frozenset({_public_name(_POLICY_METRIC_ADAPTER, OBJECTIVE_WEIGHT_SUM)})


def combine_window_metrics(
    chunks: Sequence[Mapping[str, Any]],
    *,
    weighted: frozenset[str],
    weight_key: str = WINDOW_WEIGHT_KEY,
    summed: frozenset[str] = WINDOW_SUMMED,
) -> dict[str, float]:
    """Recombine one optimizer step's per-micro-batch metrics into whole-window values.

    The shared engine reduces and returns metrics for EVERY chunk, including the accumulating ones,
    and each returned dict carries the data-parallel-global weight under ``weight_key``. Because
    each chunk value is already the DP-global token-weighted mean for that chunk, the window mean is
    recovered exactly by re-weighting:

        window(m) = Σ_c m_c · n_c / Σ_c n_c

    No collective is needed — the per-chunk reduction already crossed the ranks. A metric listed in
    ``summed`` (the objective's own divisor) accumulates instead, matching the engine's window
    divisor ``Σ_c DPmean(n_c)``.

    Args:
        chunks: One reduced public-metric mapping per micro-batch, in call order.
        weighted: Public metric keys to combine as a weighted mean.
        weight_key: Public key holding each chunk's DP-global weight.
        summed: Public metric keys that accumulate rather than average.

    Returns:
        One mapping over the window, carrying the weighted metrics, the summed keys, and the total
        weight under ``weight_key``.

    Raises:
        ValueError: no chunks, a missing/negative weight, or a non-finite contribution.
    """
    if not chunks:
        raise ValueError("PPO window metrics require at least one micro-batch")
    numerators: dict[str, float] = dict.fromkeys(weighted, 0.0)
    totals: dict[str, float] = dict.fromkeys(summed, 0.0)
    total_weight = 0.0
    for index, chunk in enumerate(chunks):
        if weight_key not in chunk:
            raise ValueError(f"PPO window metrics micro-batch {index} is missing weight {weight_key!r}")
        weight = _finite_metric(chunk[weight_key], f"{weight_key} (micro-batch {index})")
        if weight < 0.0:
            raise ValueError(f"PPO window metric weight must be non-negative, got {weight!r}")
        total_weight += weight
        for key in weighted:
            if key not in chunk:
                raise ValueError(f"PPO window metrics micro-batch {index} is missing {key!r}")
            numerators[key] += _finite_metric(chunk[key], f"{key} (micro-batch {index})") * weight
        for key in summed:
            if key in chunk:
                totals[key] += _finite_metric(chunk[key], f"{key} (micro-batch {index})")
    if total_weight <= 0.0:
        raise ValueError("PPO window metrics require a positive total weight")
    combined = {key: numerator / total_weight for key, numerator in numerators.items()}
    combined.update(totals)
    combined[weight_key] = total_weight
    return combined


def _finite_metric(value: Any, label: str) -> float:
    """Coerce one already-reduced public metric to a finite float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"PPO window metric {label} must be a real scalar, got {value!r}")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"PPO window metric {label} is non-finite")
    return scalar


__all__ = [
    "POLICY_WINDOW_WEIGHTED",
    "VALUE_WINDOW_WEIGHTED",
    "WINDOW_WEIGHT_KEY",
    "combine_window_metrics",
    "make_ppo_policy_objective",
    "make_ppo_value_objective",
]
