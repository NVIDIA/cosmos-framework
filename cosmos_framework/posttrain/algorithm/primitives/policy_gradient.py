# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Policy-gradient loss and reduction primitives shared by online algorithms."""

from __future__ import annotations

from typing import Any

import torch

POLICY_GRADIENT_LOSS_TYPES: tuple[str, ...] = (
    "token-mean",
    "seq-mean-token-sum",
    "seq-mean-token-mean",
)
FIXED_RESPONSE_POLICY_GRADIENT_LOSS_TYPE = "seq-mean-token-sum-fixed"
# Granularity of the clipped surrogate's importance ratio: one ratio per token (GRPO/PPO) or one
# per response (GSPO, arXiv 2507.18071). Same vocabulary as TRL's ``importance_sampling_level``.
IMPORTANCE_SAMPLING_LEVELS: tuple[str, ...] = ("token", "sequence")
# The token log-ratio is clamped symmetrically before exp; the sequence log-ratio only from above,
# matching the reference GSPO implementations (verl, cosmos-rl). See ``sequence_log_ratio``.
TOKEN_LOG_RATIO_CLAMP = 20.0
SEQUENCE_LOG_RATIO_CLAMP_MAX = 10.0


def sequence_sums(
    values: torch.Tensor, token_counts: torch.Tensor
) -> torch.Tensor:  # values: [N], token_counts: [B], returns [B]
    """Sum a flat per-token vector over its ragged per-sequence spans.

    Plain slice sums, not ``index_add``/``scatter``: deterministic (no atomics), and the same
    arithmetic as a per-sequence ``.sum()`` loop. Gradient flows through when ``values`` carries
    one; callers wanting a detached statistic detach the input.
    """
    offsets = [0]
    for count in token_counts.tolist():  # [B] -> list[int]
        offsets.append(offsets[-1] + int(count))
    if offsets[-1] != values.numel():
        raise ValueError(f"token_counts sum to {offsets[-1]} but the flat vector holds {values.numel()} tokens")
    return torch.stack([values[offsets[index] : offsets[index + 1]].sum() for index in range(len(offsets) - 1)])  # [B]


def sequence_mean_log_ratio(
    current: torch.Tensor,
    behavior: torch.Tensor,
    token_counts: torch.Tensor,
) -> torch.Tensor:  # current/behavior: [N], token_counts: [B], returns [B] (detached)
    """Per-sequence mean of the token log-ratios, z_i = (1/L_i) Σ_t (log π_θ − log π_old).

    Subtract FIRST, then reduce: ``mean(current) - mean(behavior)`` cancels catastrophically because
    each side is O(1..10) per token while the difference is O(1e-4..1e-3). An empty span (L_i == 0)
    yields 0, i.e. a ratio of exactly 1. Detached: this is the statistic GSPO clips on and the
    diagnostic both levels report; the gradient path is built by ``sequence_log_ratio``.
    """
    with torch.no_grad():
        delta = current - behavior  # [N]
        lengths = token_counts.to(delta.dtype).clamp_min(1)  # [B]
        return sequence_sums(delta, token_counts) / lengths  # [B]


def sequence_log_ratio(
    current: torch.Tensor,
    behavior: torch.Tensor,
    token_counts: torch.Tensor,
    *,
    clamp_max: float = SEQUENCE_LOG_RATIO_CLAMP_MAX,
) -> torch.Tensor:  # current/behavior: [N], token_counts: [B], returns [N]
    """GSPO's sequence-level log-ratio, broadcast to every token of its response.

    Value: every token of response i carries ``z_i = mean_t(log π_θ − log π_old)`` over that
    response's ``L_i`` loss-masked tokens (GSPO eq. 7; the ratio is ``exp`` of this). Gradient: the
    GSPO-token form ``sg[z_i] + (c_t − sg[c_t])`` (eq. 14), so ``∂/∂c_t = 1`` for the token's own
    logprob and 0 for its neighbours. The parenthesisation is load-bearing: ``c_t − c_t`` is exactly
    ``0.0`` in IEEE arithmetic, so the VALUE is bit-identical to ``z_i``; ``z − sg[c] + c`` is not.

    For a sequence-uniform advantage this is identical in value, clip decision and parameter
    gradient to the full-gradient form that broadcasts ``z_i`` with ``expand`` and differentiates
    through the mean: there ``∂z_i/∂c_t = 1/L_i`` on each of the ``L_i`` identical per-token losses,
    and the copies sum back to one. The two forms differ only for per-token advantages, where the
    full-gradient form averages the credit over the response and this one keeps it per token.

    ``clamp_max`` bounds the log-ratio from above only (reference implementations use 10). It is
    inert for a positive advantage, which the epsilon clip already handles, and it is the ONLY bound
    on a far-off-policy negative-advantage response, because the dual clip is undefined here: it
    caps that response's loss at ``exp(clamp_max)·|A|`` and zeroes its gradient. Underflow on the
    other side is benign (loss and gradient go to 0 for A > 0; the clipped branch is selected for
    A < 0), so there is no lower clamp.
    """
    seq_mean = sequence_mean_log_ratio(current, behavior, token_counts)  # [B]
    per_token_mean = torch.repeat_interleave(seq_mean, token_counts)  # [N]
    log_ratio = per_token_mean + (current - current.detach())  # [N]
    return torch.clamp(log_ratio, max=clamp_max)  # [N]


def sampled_k3_penalty(
    log_reference: torch.Tensor,
    log_behavior: torch.Tensor,
    *,
    log_ratio_limit: float | None = TOKEN_LOG_RATIO_CLAMP,
    coeff_clamp: float | None = None,
) -> torch.Tensor:  # log probabilities: [N], returns [N]
    """Return the sampled K3 penalty ``exp(z) - z - 1`` for ``z=log_reference-log_behavior``.

    ``coeff_clamp`` bounds the penalty's GRADIENT without touching its VALUE (``None`` = off, the
    exact estimator, bit-identical to before the knob existed). The exact derivative
    ``d k3 / d log_behavior = 1 - exp(z)`` is unbounded below inside the ``log_ratio_limit`` band
    (−4.85e8 just inside ±20 nats), so one token whose reference forward disagrees with the policy
    forward by a few nats — routing nondeterminism on a sparse-MoE policy, not policy drift — can own
    the whole step's gradient and the clip norm's budget. With ``coeff_clamp = c`` the returned tensor
    carries the exact ``k3`` value and the gradient of the LINEAR EXTENSION of ``k3`` beyond
    ``|z| = c``: the coefficient becomes ``1 - exp(clamp(z, -c, c))``, bounded by ``exp(c) - 1`` in
    magnitude and still pulling toward the reference past the band (a saturating clamp would zero
    it, which is what the outer ``log_ratio_limit`` already does and keeps doing). Detach-and-add
    keeps the value bit-identical: ``surrogate - surrogate.detach()`` is exactly ``0.0``.
    """
    delta = log_reference - log_behavior  # [N]
    if log_ratio_limit is not None:
        delta = delta.clamp(min=-log_ratio_limit, max=log_ratio_limit)  # [N]
    k3 = torch.expm1(delta) - delta  # [N]
    if coeff_clamp is None:
        return k3
    if not coeff_clamp > 0.0:
        raise ValueError(f"coeff_clamp must be None or a positive number of nats, got {coeff_clamp!r}")
    bounded = delta.clamp(min=-coeff_clamp, max=coeff_clamp)  # [N]
    # d surrogate / d delta: inside the band the clamp passes the gradient and `delta - bounded` has
    # slope 0, giving expm1(delta) as before; outside, the clamp blocks the first term and
    # `delta - bounded` has slope 1, giving the boundary coefficient expm1(±c).
    slope = torch.expm1(bounded)  # [N]
    surrogate = slope - bounded + slope.detach() * (delta - bounded)  # [N]
    return k3.detach() + (surrogate - surrogate.detach())  # [N]


def clipped_policy_gradient_loss(
    current: torch.Tensor,
    behavior: torch.Tensor,
    advantage: torch.Tensor,
    eps_low: float,
    eps_high: float,
    reference: torch.Tensor | None = None,
    kl_beta: float = 0.0,
    lower_bound_ratio: float | None = 3.0,
    stats: dict | None = None,
    log_ratio_limit: float | None = TOKEN_LOG_RATIO_CLAMP,
    *,
    importance_sampling_level: str = "token",
    token_counts: torch.Tensor | None = None,
    kl_coeff_clamp: float | None = None,
    token_weight: torch.Tensor | None = None,
    centering: torch.Tensor | None = None,
) -> torch.Tensor:  # current/behavior/advantage/reference/token_weight/centering: [N], token_counts: [B], returns [N]
    """Compute the clipped surrogate, optional dual clip, and k3 KL at token or sequence level.

    ``behavior=current.detach()`` gives the on-policy policy-gradient estimator. A negative
    epsilon disables that side of the clip, matching cosmos-rl. A positive ``lower_bound_ratio``
    caps the unbounded negative-advantage branch; ``None`` disables it.

    ``importance_sampling_level="token"`` is GRPO/PPO: one ratio per token. ``"sequence"`` is GSPO:
    one ratio per response, shared by all its tokens (see ``sequence_log_ratio``), so the epsilon
    clip admits or drops whole responses. Sequence level requires ``token_counts`` and refuses a
    dual-clip bound, which no GSPO reference defines; the caller must pass ``None``, not rely on it
    being ignored. On-policy the two levels agree exactly in value and gradient, because every
    log-ratio is 0 either way.

    ``log_ratio_limit`` bounds token-level ratios before exponentiation. ``None`` is reserved for
    callers such as PPO that have already replaced every clipped branch with its exact boundary;
    sequence-level GSPO keeps its own one-sided bound in ``sequence_log_ratio``.

    ``kl_coeff_clamp`` bounds the KL term's per-token gradient coefficient at ``exp(c) - 1`` while
    the penalty's value (and every ``kl_*`` statistic) stays the exact k3; see ``sampled_k3_penalty``.
    ``None`` keeps the exact gradient.

    ``token_weight`` and ``centering`` are score centering's two hooks (``rl.score_centering``;
    ``algorithm/primitives/score_centering.py``), both ``None`` by default, in which case the returned
    tensor is byte-identical to before they existed. ``token_weight`` (detached, ``[N]``) scales the
    clipped surrogate — ``sg[w_{y_t}]`` of the composed estimator, exactly ``1.0`` for centering alone —
    and ``centering`` (``[N]``, live through the trainer's head logprobs) is ADDED as
    ``advantage * centering``: the surrogate's gradient at the on-policy point is ``−A ∇log p_{y_t}``,
    so the sum has the gradient of ``−A (sg[w] log p_{y_t} − correction)`` (Eq. 14 of the reference)
    while the surrogate's value, clip diagnostics and the KL term stay what they were.

    ``stats`` (when given) is filled with DETACHED diagnostic SUMS — ``clip_count`` (tokens whose
    per-token loss the clamp/dual-clip actually CHANGED), ``ratio_absdev_sum`` (Σ``|ratio-1|``) and,
    when ``kl_beta > 0``, ``kl_sum`` (Σ of the per-token K3 penalty, before the ``kl_beta`` scale)
    plus ``kl_max``, the ONE non-sum: the largest per-token penalty on this shard, which is the
    number that separates "the anchor is pulling everywhere" from "one token owns the step".
    Without them a clip that never binds is indistinguishable from one that does; the loss looks
    healthy either way, which is exactly how an inert clip goes unnoticed. With ``token_counts`` two
    sequence-level sums are added at EITHER level: ``seq_clip_count`` (responses with at least one
    changed token) and ``seq_log_ratio_abs_sum`` (Σ_i |z_i| over responses, the magnitude GSPO's
    clip compares against ``ln(1±ε)``; at token level it previews what a sequence ratio would see).
    """
    if importance_sampling_level not in IMPORTANCE_SAMPLING_LEVELS:
        raise ValueError(
            f"importance_sampling_level must be one of {IMPORTANCE_SAMPLING_LEVELS}, got {importance_sampling_level!r}"
        )
    if importance_sampling_level == "sequence":
        if token_counts is None:
            raise ValueError("importance_sampling_level='sequence' needs token_counts to locate each response's tokens")
        if lower_bound_ratio is not None:
            raise ValueError(
                "importance_sampling_level='sequence' takes no dual-clip bound: GSPO defines none, so pass "
                f"lower_bound_ratio=None instead of {lower_bound_ratio!r}"
            )
        log_ratio = sequence_log_ratio(current, behavior, token_counts)  # [N]
    else:
        log_ratio = current - behavior  # [N]
        if log_ratio_limit is not None:
            log_ratio = log_ratio.clamp(min=-log_ratio_limit, max=log_ratio_limit)  # [N]
    ratio = torch.exp(log_ratio)  # [N]
    min_clipped = 1.0 - eps_low if eps_low >= 0.0 else None
    max_clipped = 1.0 + eps_high if eps_high >= 0.0 else None
    if min_clipped is None and max_clipped is None:
        clipped = ratio  # [N]
    else:
        clipped = torch.clamp(ratio, min=min_clipped, max=max_clipped)  # [N]
    loss = -torch.min(ratio * advantage, clipped * advantage)  # [N]
    if lower_bound_ratio is not None:
        loss = torch.where(
            advantage < 0,
            torch.min(-lower_bound_ratio * advantage, loss),
            loss,
        )  # [N]
    if stats is not None:
        with torch.no_grad():
            # "Did the clip CHANGE this token's loss", not "was the ratio out of band". Band
            # membership over-counts: where torch.min selects the UNCLIPPED term the clamp is inert,
            # so counting it reports a trust region doing work it is not. Exact inequality is right —
            # where nothing clipped, loss == -(ratio * advantage); this also picks up the dual-clip
            # branch for free, since that too rewrites loss. Computed BEFORE the KL term is folded in
            # (the KL would make every token differ). SUMS, never rank-local means: a mean here
            # becomes a mean-of-means after the cross-rank reduction, so the caller forms the ratio
            # (Σcount/Σtokens) after reduction instead.
            changed = loss != -(ratio * advantage)  # [N]
            stats["clip_count"] = float(changed.sum())
            stats["ratio_absdev_sum"] = float((ratio - 1.0).abs().sum())
            if token_counts is not None:
                # Response-level view of the same two questions. A response counts as clipped when
                # ANY of its tokens changed; at sequence level that is all-or-none by construction,
                # at token level it says how many responses the token clip touched at all.
                stats["seq_clip_count"] = float((sequence_sums(changed.to(ratio.dtype), token_counts) > 0).sum())
                stats["seq_log_ratio_abs_sum"] = float(
                    sequence_mean_log_ratio(current, behavior, token_counts).abs().sum()
                )
    if token_weight is not None:
        if token_weight.shape != loss.shape:
            raise ValueError(
                f"token_weight must be [N] like the loss, got {tuple(token_weight.shape)} vs {tuple(loss.shape)}"
            )
        loss = loss * token_weight.detach()  # [N]  sg[w_{y_t}] · surrogate (composed IS; 1.0 for centering alone)
    if centering is not None:
        if centering.shape != loss.shape:
            raise ValueError(
                f"centering must be [N] like the loss, got {tuple(centering.shape)} vs {tuple(loss.shape)}"
            )
        # + A · Σ_H sg[q_v w_v − α p_v] log p_v: subtracts the sampler-expected score from the gradient.
        loss = loss + advantage * centering  # [N]
    if kl_beta > 0.0:
        if reference is None:
            raise ValueError("kl_beta>0 requires reference per-token logprobs")
        # Reference-anchored K3 penalty, ADDED to the per-token loss (DeepSeekMath GRPO eq. 4;
        # cosmos-rl `kl_beta * kl` in grpo_trainer.py):
        #     z = log pi_ref(y_t) - log pi_theta(y_t);   k3 = exp(z) - z - 1 >= 0
        # `k3` is the sampled estimator of KL(pi_theta || pi_ref) on tokens drawn from pi_theta, so
        # MINIMIZING the loss pulls the current policy back toward the reference:
        #     d k3 / d log pi_theta = 1 - exp(z)  (< 0 when pi_theta under-weights the token vs the
        # reference => descent raises it; > 0 when it over-weights => descent lowers it). Value and
        # gradient name two different divergences: the VALUE (what `kl_sum`/`kl_max` report) is the
        # forward-KL estimator above, while the expected GRADIENT of this sampled loss, with the
        # samples held fixed, is E_{pi_theta}[(1 - pi_ref/pi_theta) grad log pi_theta] =
        # -E_{pi_ref}[grad log pi_theta] = grad KL(pi_ref || pi_theta) — the mode-covering pull toward
        # the reference (the DeepSeekMath convention, kept here deliberately). The
        # reference is DETACHED here so the gradient can only flow through `current`, whatever the
        # caller produced it under. The log-ratio is clamped at +/-TOKEN_LOG_RATIO_CLAMP nats before
        # `exp`, exactly as PPO's reward-shaping form `algorithm/ppo/losses.py::sampled_k3_kl` does:
        # beyond the clamp the per-token penalty saturates and its gradient is zero, which keeps a
        # collapsed token (log-ratio in the hundreds) from overflowing fp32 into a non-finite loss.
        # Inside that band the gradient coefficient 1 - exp(z) is still unbounded below; a positive
        # `kl_coeff_clamp` caps it at exp(c) - 1 without changing the penalty's value.
        kl = sampled_k3_penalty(reference.detach(), current, coeff_clamp=kl_coeff_clamp)  # [N]
        loss = loss + kl_beta * kl  # [N]
        if stats is not None:
            # Detached SUM (not mean) for the same reason as `clip_count`: the caller divides by the
            # aggregated token count after the cross-rank reduction. The MAX is the one statistic
            # that composes by max, not by sum; the caller's adapter reduces it that way.
            detached_kl = kl.detach()  # [N]
            stats["kl_sum"] = float(detached_kl.sum())
            stats["kl_max"] = float(detached_kl.max()) if detached_kl.numel() else 0.0
    return loss


def gather_behavior_logprobs(
    behavior_logprobs: list[Any],
    selection: torch.Tensor,
    current: torch.Tensor,
) -> torch.Tensor:  # selection: [B,L-1], current/returns: [N]
    """Flatten stored completion logprobs into ``current``'s masked token order.

    Stored values cover the full completion. The selection mask may exclude a shared completion
    prefix, so each sequence contributes its trailing selected-token span.
    """
    counts = selection.sum(dim=1).tolist()  # [B] -> list[int]
    flat: list[float] = []
    for sequence_logprobs, count in zip(behavior_logprobs, counts):
        if sequence_logprobs is None:
            raise ValueError("rollout_logprobs present for some sequences but missing for others")
        values = sequence_logprobs.tolist() if hasattr(sequence_logprobs, "tolist") else list(sequence_logprobs)
        if len(values) < count:
            raise ValueError(
                f"rollout_logprobs length {len(values)} < masked completion token count {count} "
                "(pi_old shorter than the masked completion)"
            )
        if count:
            flat.extend(float(value) for value in values[-count:])
    return torch.tensor(flat, device=current.device, dtype=current.dtype)  # [N]


def reduce_policy_gradient_loss(
    per_token_loss: torch.Tensor,
    token_counts: torch.Tensor,
    loss_type: str,
    *,
    max_response_length: int | None = None,
) -> tuple[torch.Tensor, int]:  # per_token_loss: [N], token_counts: [B], returns ([], units)
    """Reduce ragged per-token losses to a local numerator and normalization-unit count.

    The caller owns distributed normalization: divide the numerator by the DP-mean unit count so
    it composes with FSDP's mean-reduced gradients under uneven shards.
    """
    per_sequence = sequence_sums(per_token_loss, token_counts)  # [B]
    if loss_type == "token-mean":
        return per_sequence.sum(), int(token_counts.sum())  # [], tokens
    if loss_type == "seq-mean-token-sum":
        return per_sequence.sum(), len(per_sequence)  # [], sequences
    if loss_type == "seq-mean-token-mean":
        lengths = token_counts.to(per_token_loss.dtype).clamp_min(1)  # [B]
        return (per_sequence / lengths).sum(), len(per_sequence)  # [], sequences
    if loss_type == FIXED_RESPONSE_POLICY_GRADIENT_LOSS_TYPE:
        if isinstance(max_response_length, bool) or not isinstance(max_response_length, int):
            raise TypeError(
                f"{FIXED_RESPONSE_POLICY_GRADIENT_LOSS_TYPE} requires a non-bool integer "
                f"max_response_length, got {max_response_length!r}"
            )
        if max_response_length < 1:
            raise ValueError(
                f"{FIXED_RESPONSE_POLICY_GRADIENT_LOSS_TYPE} requires max_response_length > 0, "
                f"got {max_response_length!r}"
            )
        fixed_numerator = per_sequence.sum() / max_response_length  # []
        return fixed_numerator, len(per_sequence)  # [], sequences
    supported = (*POLICY_GRADIENT_LOSS_TYPES, FIXED_RESPONSE_POLICY_GRADIENT_LOSS_TYPE)
    raise ValueError(f"loss_type must be one of {supported}, got {loss_type!r}")
