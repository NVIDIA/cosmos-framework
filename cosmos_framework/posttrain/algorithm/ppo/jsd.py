# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Stable generalized JSD primitives for PPO policy optimization."""

from __future__ import annotations

import math
from typing import Any, Literal

import torch

from cosmos_framework.posttrain.algorithm.primitives.numerics import validate_finite

JSDConvention = Literal["normalized", "opd"]


def _validate_distribution(probabilities: torch.Tensor, name: str) -> None:  # probabilities: [...,V]
    if not probabilities.is_floating_point():
        raise TypeError(f"{name} must be floating point, got dtype={probabilities.dtype}")
    if probabilities.ndim == 0 or probabilities.shape[-1] == 0 or probabilities.numel() == 0:
        raise ValueError(f"{name} must have a non-empty final distribution dimension")
    validate_finite(probabilities, name)
    negative = probabilities < 0.0  # [...,V]
    if bool(negative.any().item()):
        raise ValueError(f"{name} probabilities must be non-negative")
    totals = probabilities.sum(dim=-1)  # [...]
    ones = torch.ones_like(totals)  # [...]
    tolerance = 1e-3 if probabilities.dtype in (torch.float16, torch.bfloat16) else 1e-6
    if not torch.allclose(totals, ones, atol=tolerance, rtol=tolerance):
        raise ValueError(f"{name} probabilities must sum to 1 along the final dimension")


def _log_probabilities(probabilities: torch.Tensor) -> torch.Tensor:  # probabilities: [...,V], returns [...,V]
    """Take logs without creating a ``log(0)`` backward singularity."""
    positive = probabilities > 0.0  # [...,V]
    safe_input = torch.where(positive, probabilities, torch.ones_like(probabilities))  # [...,V]
    finite_logs = safe_input.log()  # [...,V]
    negative_infinity = torch.full_like(finite_logs, -float("inf"))  # [...,V]
    return torch.where(positive, finite_logs, negative_infinity)  # [...,V]


def _kl_from_logs(
    source_probs: torch.Tensor,
    source_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    *,
    source_support: torch.Tensor | None = None,
) -> torch.Tensor:  # inputs: [...,V], returns [...]
    """Compute KL from log masses while preserving true support mismatches."""
    active = source_probs > 0.0 if source_support is None else source_support  # [...,V]
    target_missing = active & torch.isneginf(target_log_probs)  # [...,V]
    finite_active = active & ~target_missing  # [...,V]
    zeros = torch.zeros_like(source_probs)  # [...,V]
    safe_source_logs = torch.where(finite_active, source_log_probs, zeros)  # [...,V]
    safe_target_logs = torch.where(finite_active, target_log_probs, zeros)  # [...,V]
    finite_terms = source_probs * (safe_source_logs - safe_target_logs)  # [...,V]
    active_terms = torch.where(finite_active, finite_terms, zeros)  # [...,V]
    infinite_terms = torch.full_like(active_terms, float("inf"))  # [...,V]
    terms = torch.where(target_missing, infinite_terms, active_terms)  # [...,V]
    return terms.sum(dim=-1)  # [...]


def _negative_probability_weighted_log(
    log_probabilities: torch.Tensor,
    log_values: torch.Tensor,
    log_weight: float,
) -> torch.Tensor:  # inputs: [...,V], returns [...,V]
    """Return ``-weight * exp(log_probabilities) * log_values`` in signed log space."""
    active = torch.isfinite(log_probabilities) & (log_values != 0.0)  # [...,V]
    ones = torch.ones_like(log_values)  # [...,V]
    safe_absolute_logs = torch.where(active, log_values.abs(), ones)  # [...,V]
    log_magnitudes = log_probabilities + safe_absolute_logs.log() + log_weight  # [...,V]
    magnitudes = torch.where(active, log_magnitudes.exp(), torch.zeros_like(log_magnitudes))  # [...,V]
    return -log_values.sign() * magnitudes  # [...,V]


def _interior_weight_parameters(beta: float, convention: JSDConvention) -> tuple[float, float, float]:
    """Return policy/reference log weights and their policy-gradient correction."""
    if convention == "normalized":
        return -math.log(beta), -math.log1p(-beta), 0.0
    if convention == "opd":
        return math.log(beta), math.log1p(-beta), 2.0 * beta - 1.0
    raise ValueError(f"convention must be 'normalized' or 'opd', got {convention!r}")


def _interior_jsd_and_policy_log_gradient(
    policy_log_probs: torch.Tensor,
    reference_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    beta: float,
    convention: JSDConvention,
) -> tuple[torch.Tensor, torch.Tensor]:  # distributions: [...,V], returns divergence [...], gradient [...,V]
    """Stable strict-interior forward and exact derivative with respect to policy log masses.

    For KL weights ``a`` on ``KL(p || m)`` and ``b`` on ``KL(q || m)``, the derivative is
    ``a*p*log(p/m) + (a*beta - (1-beta)*b)*p*q/m``. The second term vanishes only for the
    normalized convention; retaining it is what makes direct OPD exact away from ``beta=0.5``.
    """
    policy_support = torch.isfinite(policy_log_probs)  # [...,V]
    reference_support = reference_probs > 0.0  # [...,V]
    both_present = policy_support & reference_support  # [...,V]
    policy_only = policy_support & ~reference_support  # [...,V]
    reference_only = ~policy_support & reference_support  # [...,V]
    policy_probs = policy_log_probs.exp()  # [...,V]
    zeros = torch.zeros_like(reference_probs)  # [...,V]
    safe_policy_logs = torch.where(policy_support, policy_log_probs, zeros)  # [...,V]
    safe_reference_logs = torch.where(reference_support, reference_log_probs, zeros)  # [...,V]
    machine_epsilon = torch.finfo(policy_log_probs.dtype).eps
    correction_log_threshold = math.log(16.0 * machine_epsilon)
    policy_log_weight, reference_log_weight, gradient_correction = _interior_weight_parameters(
        beta,
        convention,
    )

    if beta <= 0.5:
        reference_policy_log_ratio = safe_reference_logs - safe_policy_logs  # [...,V]
        weighted_reference_ratio_logs = reference_policy_log_ratio + math.log(beta)  # [...,V]
        mixture_policy_ratio_logs = torch.logaddexp(
            torch.full_like(reference_policy_log_ratio, math.log1p(-beta)),
            weighted_reference_ratio_logs,
        )  # [...,V]
        regular_policy_terms = _negative_probability_weighted_log(
            policy_log_probs,
            mixture_policy_ratio_logs,
            policy_log_weight,
        )  # [...,V]
        regular_reference_terms = _negative_probability_weighted_log(
            reference_log_probs,
            mixture_policy_ratio_logs - reference_policy_log_ratio,
            reference_log_weight,
        )  # [...,V]
        shared_masses = torch.where(
            both_present,
            (safe_reference_logs - mixture_policy_ratio_logs).exp(),
            zeros,
        )  # [...,V]
        regular_policy_gradient = regular_policy_terms + gradient_correction * shared_masses  # [...,V]

        # A subnormal beta cannot represent beta*(q/p-1) with useful relative
        # precision when q/p is ordinary. In exactly that regime, the strict
        # interior expression is indistinguishable in float64 from its beta->0
        # limit, category by category: p-q + q*log(q/p). Keep the regular log
        # mixture when beta*q/p is material, including finite support-mismatch
        # cases represented by very small but nonzero policy log masses.
        use_zero_limit = (beta < torch.finfo(policy_log_probs.dtype).tiny) & (
            weighted_reference_ratio_logs < correction_log_threshold
        )  # [...,V]
        zero_limit_contributions = (
            policy_probs - reference_probs + reference_probs * reference_policy_log_ratio
        )  # [...,V]
        zero_limit_policy_gradient = policy_probs - reference_probs  # [...,V]
        both_contributions = torch.where(
            use_zero_limit,
            zero_limit_contributions,
            regular_policy_terms + regular_reference_terms,
        )  # [...,V]
        both_policy_gradient = torch.where(
            use_zero_limit,
            zero_limit_policy_gradient,
            regular_policy_gradient,
        )  # [...,V]
        policy_only_contributions = _negative_probability_weighted_log(
            policy_log_probs,
            torch.full_like(policy_log_probs, math.log1p(-beta)),
            policy_log_weight,
        )  # [...,V]
        reference_only_contributions = _negative_probability_weighted_log(
            reference_log_probs,
            torch.full_like(reference_log_probs, math.log(beta)),
            reference_log_weight,
        )  # [...,V]
    else:
        one_minus_beta = 1.0 - beta
        policy_reference_log_ratio = safe_policy_logs - safe_reference_logs  # [...,V]
        weighted_policy_ratio_logs = policy_reference_log_ratio + math.log(one_minus_beta)  # [...,V]
        mixture_reference_ratio_logs = torch.logaddexp(
            torch.full_like(policy_reference_log_ratio, math.log(beta)),
            weighted_policy_ratio_logs,
        )  # [...,V]
        regular_policy_terms = _negative_probability_weighted_log(
            policy_log_probs,
            mixture_reference_ratio_logs - policy_reference_log_ratio,
            policy_log_weight,
        )  # [...,V]
        regular_reference_terms = _negative_probability_weighted_log(
            reference_log_probs,
            mixture_reference_ratio_logs,
            reference_log_weight,
        )  # [...,V]
        shared_masses = torch.where(
            both_present,
            (safe_policy_logs - mixture_reference_ratio_logs).exp(),
            zeros,
        )  # [...,V]
        regular_policy_gradient = regular_policy_terms + gradient_correction * shared_masses  # [...,V]

        # Mirror the beta->0 treatment around beta=1. The nearest few float64
        # values to one lose alpha*(p/q-1) in logaddexp, where alpha=1-beta.
        # When that correction is below rounding precision, use the exact
        # category-wise endpoint limit p*log(p/q) + q-p.
        use_one_limit = (one_minus_beta <= 16.0 * machine_epsilon) & (
            weighted_policy_ratio_logs < correction_log_threshold
        )  # [...,V]
        one_limit_policy_gradient = policy_probs * policy_reference_log_ratio  # [...,V]
        one_limit_contributions = one_limit_policy_gradient + reference_probs - policy_probs  # [...,V]
        both_contributions = torch.where(
            use_one_limit,
            one_limit_contributions,
            regular_policy_terms + regular_reference_terms,
        )  # [...,V]
        both_policy_gradient = torch.where(
            use_one_limit,
            one_limit_policy_gradient,
            regular_policy_gradient,
        )  # [...,V]
        policy_only_contributions = _negative_probability_weighted_log(
            policy_log_probs,
            torch.full_like(policy_log_probs, math.log(one_minus_beta)),
            policy_log_weight,
        )  # [...,V]
        reference_only_contributions = _negative_probability_weighted_log(
            reference_log_probs,
            torch.full_like(reference_log_probs, math.log(beta)),
            reference_log_weight,
        )  # [...,V]

    contributions = torch.where(
        both_present,
        both_contributions,
        torch.where(
            policy_only,
            policy_only_contributions,
            torch.where(reference_only, reference_only_contributions, zeros),
        ),
    )  # [...,V]
    policy_log_gradient = torch.where(
        both_present,
        both_policy_gradient,
        torch.where(policy_only, policy_only_contributions, zeros),
    )  # [...,V]
    divergence = contributions.sum(dim=-1).clamp_min(0.0)  # [...]
    return divergence, policy_log_gradient  # [...], [...,V]


def _generalized_jsd_from_log_policy(
    policy_log_probs: torch.Tensor,
    reference_probs: torch.Tensor,
    beta: float,
    convention: JSDConvention,
) -> torch.Tensor:  # inputs: [...,V], returns [...]
    """Evaluate generalized JSD from stable policy log masses.

    The calculation is intentionally float64 only on the already-coarsened
    distribution. The returned forward value follows the defining KL formula,
    while a first-order surrogate installs its closed-form policy-log gradient.
    This avoids ``0 * inf`` and subnormal-beta cancellation without clipping
    gradients before AMP unscaling.
    """
    policy_logs_work = policy_log_probs.to(dtype=torch.float64)  # [...,V]
    reference_work = reference_probs.detach().to(dtype=torch.float64)  # [...,V]
    reference_logs_work = _log_probabilities(reference_work)  # [...,V]
    policy_support = torch.isfinite(policy_logs_work)  # [...,V]
    policy_work = policy_logs_work.exp()  # [...,V]
    zeros = torch.zeros_like(policy_logs_work)  # [...,V]

    if beta == 0.0:
        reference_support = reference_work > 0.0  # [...,V]
        both_present = policy_support & reference_support  # [...,V]
        policy_missing = ~policy_support & reference_support  # [...,V]
        safe_policy_logs = torch.where(policy_support, policy_logs_work, zeros)  # [...,V]
        safe_reference_logs = torch.where(reference_support, reference_logs_work, zeros)  # [...,V]
        reference_policy_log_ratio = safe_reference_logs - safe_policy_logs  # [...,V]
        active_log_ratio = torch.where(both_present, reference_policy_log_ratio, zeros)  # [...,V]
        # Mirror the beta=1 endpoint with the reverse-KL Bregman form.
        # ``q*log(q/p) + p - q`` sums to KL for normalized distributions,
        # remains non-negative at low-precision equality, and has the stable
        # policy-log-mass derivative ``p-q``.
        finite_contributions = reference_work * active_log_ratio + policy_work - reference_work  # [...,V]
        infinite_contributions = torch.full_like(finite_contributions, float("inf"))  # [...,V]
        contributions = torch.where(policy_missing, infinite_contributions, finite_contributions)  # [...,V]
        forward = contributions.sum(dim=-1).clamp_min(0.0)  # [...]
        policy_log_gradient = torch.where(policy_support, policy_work - reference_work, zeros)  # [...,V]
    elif beta == 1.0:
        reference_support = reference_work > 0.0  # [...,V]
        both_present = policy_support & reference_support  # [...,V]
        reference_missing = policy_support & ~reference_support  # [...,V]
        safe_policy_logs = torch.where(policy_support, policy_logs_work, zeros)  # [...,V]
        safe_reference_logs = torch.where(reference_support, reference_logs_work, zeros)  # [...,V]
        policy_reference_log_ratio = safe_policy_logs - safe_reference_logs  # [...,V]
        active_log_ratio = torch.where(both_present, policy_reference_log_ratio, zeros)  # [...,V]
        # Category-wise ``p*log(p/q) + q - p`` sums to KL for normalized
        # distributions and avoids the first-order cancellation that can make
        # a low-precision equality case negative. Its derivative with respect
        # to a policy log mass is the equally stable ``p*log(p/q)``.
        finite_contributions = policy_work * active_log_ratio + reference_work - policy_work  # [...,V]
        infinite_contributions = torch.full_like(finite_contributions, float("inf"))  # [...,V]
        contributions = torch.where(reference_missing, infinite_contributions, finite_contributions)  # [...,V]
        forward = contributions.sum(dim=-1).clamp_min(0.0)  # [...]
        policy_log_gradient = torch.where(both_present, policy_work * active_log_ratio, zeros)  # [...,V]
    else:
        forward, policy_log_gradient = _interior_jsd_and_policy_log_gradient(
            policy_logs_work,
            reference_work,
            reference_logs_work,
            beta,
            convention,
        )  # [...], [...,V]

    surrogate_inputs = torch.where(policy_support, policy_logs_work, zeros)  # [...,V]
    surrogate = (surrogate_inputs * policy_log_gradient.detach()).sum(dim=-1)  # [...]
    return forward.detach() + surrogate - surrogate.detach()  # [...]


def generalized_jsd(
    policy_probs: torch.Tensor,
    reference_probs: torch.Tensor,
    beta: float,
    *,
    convention: JSDConvention = "normalized",
) -> torch.Tensor:  # probabilities: [...,V], returns [...]
    """Return generalized JSD over the final distribution dimension.

    For interior ``beta``, ``m = (1-beta) p + beta q``. ``normalized`` uses
    ``KL(p || m) / beta + KL(q || m) / (1-beta)`` for PPO, while ``opd`` uses
    the historical direct objective ``beta * KL(p || m) + (1-beta) * KL(q || m)``.
    Both conventions use ``KL(q || p)`` at zero and ``KL(p || q)`` at one.
    Reference probabilities are always treated as fixed targets.
    """
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be finite and in [0, 1], got {beta}")
    if convention not in ("normalized", "opd"):
        raise ValueError(f"convention must be 'normalized' or 'opd', got {convention!r}")
    if policy_probs.shape != reference_probs.shape:
        raise ValueError(
            f"policy_probs and reference_probs shapes must match, got "
            f"{tuple(policy_probs.shape)} and {tuple(reference_probs.shape)}"
        )
    if policy_probs.device != reference_probs.device:
        raise ValueError("policy_probs and reference_probs must be on the same device")
    _validate_distribution(policy_probs, "policy_probs")
    reference_fixed = reference_probs.detach()  # [...,V]
    _validate_distribution(reference_fixed, "reference_probs")

    policy_work = policy_probs.to(dtype=torch.float64)  # [...,V]
    policy_logs_work = _log_probabilities(policy_work)  # [...,V]
    reference_work = reference_fixed.to(dtype=torch.float64)  # [...,V]
    divergence_work = _generalized_jsd_from_log_policy(  # [...]
        policy_logs_work,
        reference_work,
        beta,
        convention,
    )
    result_dtype = torch.float64 if policy_probs.dtype == torch.float64 else torch.float32
    return divergence_work.to(dtype=result_dtype)  # [...]


def _broadcast_to(
    tensor: torch.Tensor, shape: tuple[int, ...], name: str
) -> torch.Tensor:  # tensor: [...], returns [...]
    try:
        return torch.broadcast_to(tensor, shape)  # [...]
    except RuntimeError as error:
        raise ValueError(f"{name} shape {tuple(tensor.shape)} is not broadcast-compatible with {shape}") from error


class _CoarsePolicyLogProbs(torch.autograd.Function):
    """Coarsen logits without retaining another full-vocabulary activation.

    The support consists of explicit token IDs and one residual bucket. Forward
    log-normalizers are accumulated over bounded vocabulary chunks, and backward
    reconstructs the exact log-softmax/coarse-residual Jacobian chunk by chunk.
    """

    @staticmethod
    def forward(
        ctx: Any,
        policy_logits: torch.Tensor,
        support_ids: torch.Tensor,
        chunk_size: int,
        temperature: float,
    ) -> torch.Tensor:  # logits: [...,V], ids: [...,K], returns [...,K+1]
        prefix = tuple(policy_logits.shape[:-1])
        vocabulary_size = policy_logits.shape[-1]
        support_size = support_ids.shape[-1]
        row_count = math.prod(prefix) if prefix else 1
        work_dtype = torch.float64 if policy_logits.dtype == torch.float64 else torch.float32
        inverse_temperature = 1.0 / temperature
        logits_flat = policy_logits.reshape(row_count, vocabulary_size)  # [N,V]
        ids_flat = support_ids.reshape(row_count, support_size)  # [N,K]
        log_normalizer = torch.full(
            (row_count,),
            -float("inf"),
            device=policy_logits.device,
            dtype=work_dtype,
        )  # [N]
        residual_log_normalizer = torch.full_like(log_normalizer, -float("inf"))  # [N]

        for start in range(0, vocabulary_size, chunk_size):
            end = min(start + chunk_size, vocabulary_size)
            chunk = logits_flat[:, start:end].to(dtype=work_dtype) * inverse_temperature  # [N,C]
            chunk_log_normalizer = torch.logsumexp(chunk, dim=-1)  # [N]
            log_normalizer = torch.logaddexp(log_normalizer, chunk_log_normalizer)  # [N]

            residual_chunk = chunk.clone()  # [N,C]
            in_chunk = (ids_flat >= start) & (ids_flat < end)  # [N,K]
            matches = in_chunk.nonzero(as_tuple=False)  # [M,2]
            if matches.numel() > 0:
                match_rows = matches[:, 0]  # [M]
                match_support = matches[:, 1]  # [M]
                match_columns = ids_flat[match_rows, match_support] - start  # [M]
                residual_chunk[match_rows, match_columns] = -float("inf")  # [N,C]
            chunk_residual_log_normalizer = torch.logsumexp(residual_chunk, dim=-1)  # [N]
            residual_log_normalizer = torch.logaddexp(
                residual_log_normalizer,
                chunk_residual_log_normalizer,
            )  # [N]

        support_logits = logits_flat.gather(dim=-1, index=ids_flat).to(dtype=work_dtype) * inverse_temperature  # [N,K]
        support_log_probs = support_logits - log_normalizer.unsqueeze(-1)  # [N,K]
        residual_log_probs = residual_log_normalizer - log_normalizer  # [N]
        coarse_log_probs = torch.cat(
            (support_log_probs, residual_log_probs.unsqueeze(-1)),
            dim=-1,
        )  # [N,K+1]

        ctx.chunk_size = chunk_size
        ctx.input_dtype = policy_logits.dtype
        ctx.inverse_temperature = inverse_temperature
        ctx.save_for_backward(policy_logits, support_ids, log_normalizer, residual_log_normalizer)
        return coarse_log_probs.reshape(prefix + (support_size + 1,))  # [...,K+1]

    @staticmethod
    def backward(
        ctx: Any,
        grad_coarse_log_probs: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, None, None, None]:  # gradient: [...,K+1]
        if grad_coarse_log_probs is None:
            return None, None, None, None

        policy_logits, support_ids, log_normalizer, residual_log_normalizer = ctx.saved_tensors
        prefix = tuple(policy_logits.shape[:-1])
        vocabulary_size = policy_logits.shape[-1]
        support_size = support_ids.shape[-1]
        row_count = math.prod(prefix) if prefix else 1
        work_dtype = log_normalizer.dtype
        logits_flat = policy_logits.reshape(row_count, vocabulary_size)  # [N,V]
        ids_flat = support_ids.reshape(row_count, support_size)  # [N,K]
        grad_coarse = grad_coarse_log_probs.reshape(row_count, support_size + 1).to(dtype=work_dtype)  # [N,K+1]
        support_gradients = grad_coarse[:, :support_size]  # [N,K]
        residual_gradients = grad_coarse[:, support_size]  # [N]
        total_gradients = grad_coarse.sum(dim=-1)  # [N]
        grad_logits = torch.empty_like(policy_logits).reshape(row_count, vocabulary_size)  # [N,V]
        residual_exists = torch.isfinite(residual_log_normalizer)  # [N]
        safe_residual_log_normalizer = torch.where(
            residual_exists,
            residual_log_normalizer,
            torch.zeros_like(residual_log_normalizer),
        )  # [N]

        for start in range(0, vocabulary_size, ctx.chunk_size):
            end = min(start + ctx.chunk_size, vocabulary_size)
            chunk = logits_flat[:, start:end].to(dtype=work_dtype) * ctx.inverse_temperature  # [N,C]
            full_probabilities = (chunk - log_normalizer.unsqueeze(-1)).exp()  # [N,C]
            residual_probabilities = (chunk - safe_residual_log_normalizer.unsqueeze(-1)).exp()  # [N,C]
            residual_probabilities = torch.where(
                residual_exists.unsqueeze(-1),
                residual_probabilities,
                torch.zeros_like(residual_probabilities),
            )  # [N,C]
            in_chunk = (ids_flat >= start) & (ids_flat < end)  # [N,K]
            matches = in_chunk.nonzero(as_tuple=False)  # [M,2]
            if matches.numel() > 0:
                match_rows = matches[:, 0]  # [M]
                match_support = matches[:, 1]  # [M]
                match_columns = ids_flat[match_rows, match_support] - start  # [M]
                residual_probabilities[match_rows, match_columns] = 0.0  # [N,C]
            chunk_gradient = (
                -total_gradients.unsqueeze(-1) * full_probabilities
                + residual_gradients.unsqueeze(-1) * residual_probabilities
            )  # [N,C]
            if matches.numel() > 0:
                chunk_gradient[match_rows, match_columns] += support_gradients[
                    match_rows,
                    match_support,
                ]  # [M]
            grad_logits[:, start:end] = (chunk_gradient * ctx.inverse_temperature).to(dtype=ctx.input_dtype)  # [N,C]

        return grad_logits.reshape(policy_logits.shape), None, None, None  # [...,V]


def _coarse_policy_log_probs(
    policy_logits: torch.Tensor,
    support_ids: torch.Tensor,
    *,
    chunk_size: int = 8192,
    temperature: float = 1.0,
) -> torch.Tensor:  # logits: [...,V], ids: [...,K], returns [...,K+1]
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return _CoarsePolicyLogProbs.apply(policy_logits, support_ids, chunk_size, temperature)  # [...,K+1]


def topk_residual_jsd(
    policy_logits: torch.Tensor,
    support_ids: torch.Tensor,
    reference_support_probs: torch.Tensor,
    reference_residual_mass: torch.Tensor,
    beta: float,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:  # logits: [...,V], support/reference: [...,K], residual: [...], returns [...]
    """Compute generalized JSD on fixed support tokens plus one residual bucket."""
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be finite and in [0, 1], got {beta}")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {temperature}")
    if not policy_logits.is_floating_point():
        raise TypeError(f"policy_logits must be floating point, got dtype={policy_logits.dtype}")
    if policy_logits.ndim == 0 or policy_logits.shape[-1] == 0 or policy_logits.numel() == 0:
        raise ValueError("policy_logits must have a non-empty vocabulary dimension")
    validate_finite(policy_logits, "policy_logits")
    if support_ids.ndim == 0 or support_ids.shape[-1] == 0:
        raise ValueError("support_ids must have a non-empty final support dimension")
    if support_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"support_ids must contain integers, got dtype={support_ids.dtype}")

    vocabulary_size = policy_logits.shape[-1]
    support_size = support_ids.shape[-1]
    if support_size > vocabulary_size:
        raise ValueError(f"support size {support_size} exceeds vocabulary size {vocabulary_size}")
    prefix = tuple(policy_logits.shape[:-1])
    support_shape = prefix + (support_size,)
    ids_input = support_ids.to(device=policy_logits.device, dtype=torch.long)  # [...,K]
    ids = _broadcast_to(ids_input, support_shape, "support_ids")  # [...,K]
    out_of_range = (ids < 0) | (ids >= vocabulary_size)  # [...,K]
    if bool(out_of_range.any().item()):
        raise ValueError(f"support_ids values must be in range [0, {vocabulary_size})")
    sorted_ids = torch.sort(ids, dim=-1).values  # [...,K]
    duplicates = sorted_ids[..., 1:] == sorted_ids[..., :-1]  # [...,K-1]
    if bool(duplicates.any().item()):
        raise ValueError("support_ids must be unique along the final dimension")

    if not reference_support_probs.is_floating_point() or not reference_residual_mass.is_floating_point():
        raise TypeError("reference_support_probs and reference_residual_mass must be floating point")
    reference_input = reference_support_probs.detach().to(
        device=policy_logits.device,
        dtype=torch.float64,
    )  # [...,K]
    reference_support = _broadcast_to(reference_input, support_shape, "reference_support_probs")  # [...,K]
    residual_input = reference_residual_mass.detach().to(
        device=policy_logits.device,
        dtype=torch.float64,
    )  # [...] or [...,1]
    if residual_input.ndim == policy_logits.ndim and residual_input.shape[-1] == 1:
        residual_input = residual_input.squeeze(-1)  # [...]
    reference_residual = _broadcast_to(residual_input, prefix, "reference_residual_mass")  # [...]
    validate_finite(reference_support, "reference_support_probs")
    validate_finite(reference_residual, "reference_residual_mass")
    negative_reference = (reference_support < 0.0).any() | (reference_residual < 0.0).any()  # []
    if bool(negative_reference.item()):
        raise ValueError("reference support probabilities and residual mass must be non-negative")
    reference_total = reference_support.sum(dim=-1) + reference_residual  # [...]
    reference_ones = torch.ones_like(reference_total)  # [...]
    low_precision_reference = reference_support_probs.dtype in (torch.float16, torch.bfloat16) or (
        reference_residual_mass.dtype in (torch.float16, torch.bfloat16)
    )
    tolerance = 1e-3 if low_precision_reference else 1e-6
    if not torch.allclose(reference_total, reference_ones, atol=tolerance, rtol=tolerance):
        raise ValueError("reference support probabilities plus residual mass must sum to 1")

    if support_size == vocabulary_size and bool((reference_residual != 0.0).any().item()):
        raise ValueError("reference residual mass must be zero when support covers the full vocabulary")
    # Validation accepts the representational tolerance of low-precision input,
    # then fixes the detached coarse target to an exactly normalized distribution.
    reference_normalizer = reference_total.unsqueeze(-1)  # [...,1]
    normalized_reference_support = reference_support / reference_normalizer  # [...,K]
    normalized_reference_residual = reference_residual / reference_total  # [...]
    coarse_reference = torch.cat(
        (normalized_reference_support, normalized_reference_residual.unsqueeze(-1)),
        dim=-1,
    )  # [...,K+1]
    coarse_policy_logs = _coarse_policy_log_probs(policy_logits, ids, temperature=temperature)  # [...,K+1]
    divergence_work = _generalized_jsd_from_log_policy(  # [...]
        coarse_policy_logs,
        coarse_reference,
        beta,
        "normalized",
    )
    result_dtype = torch.float64 if policy_logits.dtype == torch.float64 else torch.float32
    return divergence_work.to(dtype=result_dtype)  # [...]
