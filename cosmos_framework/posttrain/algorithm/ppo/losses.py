# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Pure numerical losses shared by PPO policy and critic updates."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from cosmos_framework.posttrain.algorithm.primitives import policy_gradient
from cosmos_framework.posttrain.algorithm.primitives.numerics import validate_finite


def _broadcast_mask(mask: torch.Tensor, shape: torch.Size) -> torch.Tensor:  # mask: [...], returns [...]
    if mask.dtype is not torch.bool:
        raise TypeError(f"mask must be boolean, got dtype={mask.dtype}")
    try:
        return torch.broadcast_to(mask, shape)  # [...]
    except RuntimeError as error:
        raise ValueError(f"mask shape {tuple(mask.shape)} is not broadcast-compatible with {tuple(shape)}") from error


def _require_same_shape(named_tensors: dict[str, torch.Tensor]) -> torch.Size:
    shapes = {name: tuple(tensor.shape) for name, tensor in named_tensors.items()}
    expected = next(iter(named_tensors.values())).shape
    if any(tensor.shape != expected for tensor in named_tensors.values()):
        raise ValueError(f"tensor shapes must match, got {shapes}")
    return expected


def _stable_math_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """Use float32 for low-precision inputs and preserve explicit float64."""
    return torch.float64 if any(tensor.dtype == torch.float64 for tensor in tensors) else torch.float32


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # values: [...], mask: [...], returns []
    """Return the mean of ``values`` over positions selected by a boolean mask."""
    if values.numel() == 0:
        raise ValueError("masked_mean does not accept an empty values tensor")
    expanded_mask = _broadcast_mask(mask, values.shape)  # [...]
    selected = values.masked_select(expanded_mask)  # [N]
    if selected.numel() == 0:
        raise ValueError("masked_mean mask selected zero values")
    validate_finite(selected, "values")
    return selected.mean()  # []


def ppo_policy_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    policy_clip: float = 0.2,
    *,
    dual_clip: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:  # tensors: [...], returns loss: []
    """Compute PPO's clipped policy surrogate with fixed rollout statistics."""
    if not math.isfinite(policy_clip) or not 0.0 < policy_clip < 1.0:
        raise ValueError(f"policy_clip must be finite and in (0, 1), got {policy_clip}")
    if dual_clip is not None and (not math.isfinite(dual_clip) or dual_clip <= 1.0):
        raise ValueError(f"dual_clip must be finite and > 1, got {dual_clip}")
    shape = _require_same_shape(
        {
            "current_logprobs": current_logprobs,
            "old_logprobs": old_logprobs,
            "advantages": advantages,
        }
    )
    if current_logprobs.numel() == 0:
        raise ValueError("ppo_policy_loss does not accept empty tensors")
    validate_finite(current_logprobs, "current_logprobs")
    validate_finite(old_logprobs, "old_logprobs")
    validate_finite(advantages, "advantages")
    expanded_mask = _broadcast_mask(mask, shape)  # [...]
    work_dtype = _stable_math_dtype(current_logprobs, old_logprobs, advantages)
    current_work = current_logprobs.to(dtype=work_dtype)  # [...]
    old_fixed = old_logprobs.detach().to(dtype=work_dtype)  # [...]
    advantages_fixed = advantages.detach().to(dtype=work_dtype)  # [...]
    log_ratio = current_work - old_fixed  # [...]
    lower_ratio = 1.0 - policy_clip
    upper_ratio = 1.0 + policy_clip
    log_lower_ratio = math.log(lower_ratio)
    log_upper_ratio = math.log(upper_ratio)
    positive_advantage = advantages_fixed >= 0.0  # [...]
    clipped_branch = (positive_advantage & (log_ratio > log_upper_ratio)) | (
        ~positive_advantage & (log_ratio < log_lower_ratio)
    )  # [...]
    # bfloat16 spans nearly the full float32 range, so its finite advantages need wider ratio products.
    surrogate_dtype = (
        torch.float64
        if any(tensor.dtype == torch.bfloat16 for tensor in (current_logprobs, old_logprobs, advantages))
        else work_dtype
    )
    log_ratio_for_surrogate = log_ratio.to(dtype=surrogate_dtype)  # [...]
    # torch.where evaluates both arguments. Replacing clipped-branch log ratios
    # before exponentiation avoids an irrelevant overflow without changing the
    # standard PPO objective: those branches use the constant clip boundary.
    inactive_for_objective = clipped_branch | ~expanded_mask
    active_log_ratio = torch.where(
        inactive_for_objective,
        torch.zeros_like(log_ratio_for_surrogate),
        log_ratio_for_surrogate,
    )
    active_ratio = active_log_ratio.exp()  # [...]
    if not bool(torch.isfinite(active_ratio).all().item()):
        raise ValueError("active PPO probability ratio is not representable")
    current_for_surrogate = current_work.to(dtype=surrogate_dtype)  # [...]
    old_for_surrogate = old_fixed.to(dtype=surrogate_dtype)  # [...]
    advantages_for_surrogate = advantages_fixed.to(dtype=surrogate_dtype)  # [...]
    clipped_log_ratio = torch.where(
        positive_advantage,
        torch.full_like(advantages_for_surrogate, log_upper_ratio),
        torch.full_like(advantages_for_surrogate, log_lower_ratio),
    )  # [...]
    clipped_current = old_for_surrogate + clipped_log_ratio  # [...]
    masked_current = torch.where(expanded_mask, current_for_surrogate, old_for_surrogate)  # [...]
    safe_current = torch.where(clipped_branch, clipped_current, masked_current)  # [...]
    per_token_loss = policy_gradient.clipped_policy_gradient_loss(  # [...]
        safe_current,
        old_for_surrogate,
        advantages_for_surrogate,
        policy_clip,
        policy_clip,
        lower_bound_ratio=None,
        log_ratio_limit=None,
    )
    if dual_clip is not None:
        per_token_loss = torch.minimum(per_token_loss, dual_clip * advantages_for_surrogate.abs())
    if not bool(torch.isfinite(per_token_loss).all().item()):
        raise ValueError("PPO policy surrogate is not representable")
    loss = masked_mean(per_token_loss, expanded_mask)  # []
    clipped = (log_ratio < log_lower_ratio) | (log_ratio > log_upper_ratio)  # [...]
    clip_fraction = masked_mean(clipped.to(dtype=work_dtype), expanded_mask)  # []
    # ratio_mean is a diagnostic only. Saturate its exponent so an irrelevant
    # clipped branch cannot make an otherwise finite standard PPO update fail.
    metric_log_limit = math.log(torch.finfo(work_dtype).max) - 2.0
    metric_ratio = log_ratio.clamp(min=-metric_log_limit, max=metric_log_limit).exp()  # [...]
    ratio_mean = masked_mean(metric_ratio, expanded_mask)  # []
    metrics = {
        "policy_loss": loss.detach(),  # []
        "clip_fraction": clip_fraction.detach(),  # []
        "ratio_mean": ratio_mean.detach(),  # []
    }
    return loss, metrics


def ppo_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    value_clip: float = 0.2,
    *,
    huber_delta: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:  # tensors: [...], returns loss: []
    """Compute the clipped PPO critic loss against fixed old values and returns."""
    if not math.isfinite(value_clip) or value_clip <= 0.0:
        raise ValueError(f"value_clip must be finite and > 0, got {value_clip}")
    if huber_delta is not None and (not math.isfinite(huber_delta) or huber_delta <= 0.0):
        raise ValueError(f"huber_delta must be finite and > 0, got {huber_delta}")
    shape = _require_same_shape({"values": values, "old_values": old_values, "returns": returns})
    if values.numel() == 0:
        raise ValueError("ppo_value_loss does not accept empty tensors")
    validate_finite(values, "values")
    validate_finite(old_values, "old_values")
    validate_finite(returns, "returns")
    expanded_mask = _broadcast_mask(mask, shape)  # [...]
    # A finite float32/bf16 value can be ~3e38, so even a perfectly valid critic
    # error can overflow fp32 when squared. Value tensors are token-sized (not
    # [B,T,V] logits), making fp64 here cheap relative to the model forward and
    # preserving the mathematical loss for practical finite critic inputs.
    work_dtype = torch.float64 if huber_delta is None else _stable_math_dtype(values, old_values, returns)
    values_work = values.to(dtype=work_dtype)  # [...]
    old_fixed = old_values.detach().to(dtype=work_dtype)  # [...]
    returns_fixed = returns.detach().to(dtype=work_dtype)  # [...]
    value_delta = values_work - old_fixed  # [...]
    clipped_values = old_fixed + value_delta.clamp(min=-value_clip, max=value_clip)  # [...]
    if huber_delta is None:
        raw_error = 0.5 * (values_work - returns_fixed).square()  # [...]
        clipped_error = 0.5 * (clipped_values - returns_fixed).square()  # [...]
    else:
        raw_error = torch.nn.functional.huber_loss(values_work, returns_fixed, reduction="none", delta=huber_delta)
        clipped_error = torch.nn.functional.huber_loss(
            clipped_values, returns_fixed, reduction="none", delta=huber_delta
        )
    pessimistic_error = torch.maximum(raw_error, clipped_error)  # [...]
    loss = masked_mean(pessimistic_error, expanded_mask)  # []
    clipped = value_delta.abs() > value_clip  # [...]
    metric_dtype = _stable_math_dtype(values, old_values, returns)
    clip_fraction = masked_mean(clipped.to(dtype=metric_dtype), expanded_mask)  # []
    metrics = {
        "value_loss": loss.detach(),  # []
        "value_clip_fraction": clip_fraction.detach(),  # []
    }
    return loss, metrics


def normalize_advantages(
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    reduce_stats: Callable[[torch.Tensor], torch.Tensor] | None = None,
    eps: float = 1e-8,
    correction: int = 0,
    eps_inside_sqrt: bool = True,
    statistics_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:  # advantages: [...], mask: [...], reduction stats: [3], returns [...]
    """Normalize masked advantages using optional two-pass global statistics.

    ``reduce_stats`` is the distributed sum seam. It is called twice with
    ``statistics_dtype`` (default float64) ``[value, count]`` tensors: first for the raw sum and then for the
    centered square sum. The second pass avoids subtracting two nearly equal
    large moments, which is unstable even in float64 for sufficiently offset
    advantages. Process-group policy remains with the caller.
    """
    if advantages.numel() == 0:
        raise ValueError("normalize_advantages does not accept an empty advantages tensor")
    if not advantages.is_floating_point():
        raise TypeError(f"advantages must be floating point, got dtype={advantages.dtype}")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and > 0, got {eps}")
    if type(correction) is not int or correction < 0:
        raise ValueError("advantage correction must be a non-negative integer")
    validate_finite(advantages, "advantages")
    expanded_mask = _broadcast_mask(mask, advantages.shape)  # [...]
    if statistics_dtype not in (torch.float32, torch.float64):
        raise ValueError("advantage statistics_dtype must be float32 or float64")
    work = advantages.to(dtype=statistics_dtype)  # [...]
    selected = work.masked_select(expanded_mask)  # [N]
    local_count = selected.new_tensor(float(selected.numel()))  # []
    if reduce_stats is None and statistics_dtype == torch.float32:
        if selected.numel() <= correction:
            raise ValueError("advantage count must exceed the variance correction")
        # Native local mean/std and division retain reference FP32 rounding.
        divisor = (
            (selected.var(correction=correction) + eps).sqrt()
            if eps_inside_sqrt
            else selected.std(correction=correction) + eps
        )
        normalized = torch.where(expanded_mask, (work - selected.mean()) / divisor, 0).to(advantages.dtype)
        validate_finite(normalized, "normalized advantages")
        return normalized

    def reduce_pair(pair: torch.Tensor, stage: str) -> torch.Tensor:  # pair: [2], returns [2]
        reduced = reduce_stats(pair) if reduce_stats is not None else pair  # [2]
        if not isinstance(reduced, torch.Tensor):
            raise TypeError(f"reduce_stats must return a torch.Tensor, got {type(reduced).__name__}")
        if reduced.shape != (2,):
            raise ValueError(f"reduce_stats must return shape [2], got {tuple(reduced.shape)}")
        validate_finite(reduced, f"reduced advantage {stage} statistics")
        return reduced

    sum_and_count = reduce_pair(torch.stack((selected.sum(), local_count)), "mean")  # [2]
    count = sum_and_count[1]  # []
    if float(count.item()) <= 0.0:
        raise ValueError("normalize_advantages mask selected zero values globally")
    mean = sum_and_count[0] / count  # []
    centered_sum_and_count = reduce_pair(
        torch.stack(((selected - mean).square().sum(), local_count)),
        "variance",
    )  # [2]
    centered_count = centered_sum_and_count[1]  # []
    if float(centered_count.item()) <= 0.0 or float(centered_count.item()) != float(count.item()):
        raise ValueError("reduce_stats returned inconsistent global advantage counts")
    if float(count) <= correction:
        raise ValueError("advantage count must exceed the variance correction")
    variance = centered_sum_and_count[0].clamp_min(0.0) / (count - correction)  # []
    inverse_std = torch.rsqrt(variance + eps) if eps_inside_sqrt else (variance.sqrt() + eps).reciprocal()  # []
    normalized_work = (work - mean) * inverse_std  # [...]
    zeros = torch.zeros_like(normalized_work)  # [...]
    masked_normalized = torch.where(expanded_mask, normalized_work, zeros)  # [...]
    normalized = masked_normalized.to(dtype=advantages.dtype)  # [...]
    validate_finite(normalized, "normalized advantages")
    return normalized


def sampled_k3_kl(log_reference: torch.Tensor, log_old: torch.Tensor) -> torch.Tensor:  # tensors: [...], returns [...]
    """Return the non-negative sampled K3 reference penalty per token."""
    _require_same_shape({"log_reference": log_reference, "log_old": log_old})
    if log_reference.numel() == 0:
        raise ValueError("sampled_k3_kl does not accept empty tensors")
    validate_finite(log_reference, "log_reference")
    validate_finite(log_old, "log_old")
    work_dtype = _stable_math_dtype(log_reference, log_old)
    reference_fixed = log_reference.detach().to(dtype=work_dtype)  # [...]
    old_fixed = log_old.detach().to(dtype=work_dtype)  # [...]
    penalties = policy_gradient.sampled_k3_penalty(reference_fixed, old_fixed)  # [...]
    validate_finite(penalties, "sampled K3 penalties")
    return penalties


def categorical_entropy(
    logits: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:  # logits: [...,V], mask: [...], returns []
    """Return masked mean categorical entropy from unnormalized logits."""
    if logits.ndim == 0 or logits.shape[-1] == 0:
        raise ValueError(f"logits must have a non-empty vocabulary dimension, got shape {tuple(logits.shape)}")
    validate_finite(logits, "logits")
    work_dtype = _stable_math_dtype(logits)
    logits_work = logits.to(dtype=work_dtype)  # [...,V]
    maxima = logits_work.amax(dim=-1, keepdim=True)  # [...,1]
    centered = logits_work - maxima  # [...,V]
    # For opposite finite dtype extrema (for example bf16 max and -max), the
    # subtraction above is -inf in fp32. Such categories are already saturated;
    # clamping at log(tiny) keeps their negligible entropy contribution and its
    # gradient finite without perturbing ordinary logits.
    minimum_centered = math.log(torch.finfo(work_dtype).tiny)
    stable_centered = centered.clamp_min(minimum_centered)  # [...,V]
    log_probs = torch.log_softmax(stable_centered, dim=-1)  # [...,V]
    probs = log_probs.exp()  # [...,V]
    per_category_entropy = -(probs * log_probs)  # [...,V]
    entropy = per_category_entropy.sum(dim=-1)  # [...]
    validate_finite(entropy, "categorical entropy")
    return masked_mean(entropy, mask)  # []
