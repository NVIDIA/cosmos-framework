# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Score centering for the sampler/trainer mismatch (top-k head form), pure tensor math.

The policy gradient ``E_q[A · ∇ log p(y_t)]`` is taken over tokens drawn from the SAMPLER ``q``
(vLLM, bf16, TP) while the score ``∇ log p`` is the TRAINER's. When ``q ≠ p`` the expected score
``E_q[∇ log p] = Σ_v q_v ∇ log p_v`` is no longer zero, so a constant advantage still moves the
policy — a drift term proportional to ``q − p``. Centering subtracts that expectation:

    ∇ L_t = −A · ( w_{y_t} ∇ log p_{y_t} − Σ_{v∈H} (q_v w_v − α p_v) ∇ log p_v )          (Eq. 14)

with the head ``H`` = the sampler's top-k ids, ``ρ = q_tail / p_tail`` the tail-mass ratio that
models the unseen sampler tail as ``ρ · p`` (Eq. 9; the tail's own score sum is recovered from
``E_p[∇ log p] = 0``, Eq. 10, so only head gathers are needed), ``w = f(p/q)`` an optional
importance weight (``f = 1`` for score centering alone; truncated / masked IS when composed) and
``α = ρ f(1/ρ)`` its constant value on the modelled tail (Eq. 12). Realised as a loss term whose
COEFFICIENTS are detached and whose ``log p`` factors are live:

    correction_t = Σ_{v∈H} sg[q_v w_v − α p_v] · log p_v ;   loss_t = −A (sg[w_{y_t}] log p_{y_t} − correction_t)

Reference: the ``score_centering_loss`` snippet quoted in the design note (README form). Rules that
travel with it and are load-bearing (each breaks the zero-drift identity when violated, measured):
float32 logprobs on both sides; the head is NOT renormalised; the sampled token's logprob is kept
even when it falls outside the head. With ``q == p`` exactly, ``ρ == 1`` and every residual is
exactly ``0.0``, so the correction vanishes bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

SCORE_CENTERING_COMPOSE: tuple[str, ...] = ("none", "tis", "mis")
# Tail masses are floored here (README ``eps``): ``1 − Σ_H p`` is a small difference of O(1) numbers.
SCORE_CENTERING_TAIL_EPS = 1e-6


@dataclass(frozen=True)
class ScoreCenteringSpec:
    """Validated, immutable view of ``rl.score_centering`` the objective and the primitive share."""

    top_k: int
    compose: str = "none"
    tis_clip: float = 2.0
    mis_low: float = 0.5
    mis_high: float = 5.0
    tail_eps: float = SCORE_CENTERING_TAIL_EPS

    def __post_init__(self) -> None:
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError(f"score_centering.top_k must be a positive integer, got {self.top_k!r}")
        if self.compose not in SCORE_CENTERING_COMPOSE:
            raise ValueError(f"score_centering.compose must be one of {SCORE_CENTERING_COMPOSE}, got {self.compose!r}")
        if not (self.tis_clip >= 1.0):
            # f(1) must equal 1 so a sampler that agrees with the trainer leaves the estimator untouched.
            raise ValueError(f"score_centering.tis_clip must be >= 1 (f(1) == 1), got {self.tis_clip!r}")
        if not (0.0 <= self.mis_low <= 1.0 <= self.mis_high) or self.mis_low >= self.mis_high:
            raise ValueError(
                "score_centering.mis_low/mis_high must satisfy 0 <= low <= 1 <= high and low < high (the drop "
                f"band must contain the on-policy ratio 1), got {self.mis_low!r}/{self.mis_high!r}"
            )
        if not (self.tail_eps > 0.0):
            raise ValueError(f"score_centering tail_eps must be positive, got {self.tail_eps!r}")


def resolve_score_centering(cfg: Any) -> ScoreCenteringSpec | None:
    """``rl.score_centering`` → spec, or None when the knob is absent or disabled."""
    if cfg is None or not bool(getattr(cfg, "enabled", False)):
        return None
    return ScoreCenteringSpec(
        top_k=int(cfg.top_k),
        compose=str(cfg.compose),
        tis_clip=float(cfg.tis_clip),
        mis_low=float(cfg.mis_low),
        mis_high=float(cfg.mis_high),
    )


def importance_weight(ratio: torch.Tensor, spec: ScoreCenteringSpec) -> torch.Tensor:  # ratio: [...], returns [...]
    """``f(r)`` for ``r = p/q``: identity-free ``1`` (none), ``min(r, c)`` (TIS), ``r·1[lo ≤ r ≤ hi]`` (MIS)."""
    if spec.compose == "none":
        return torch.ones_like(ratio)
    if spec.compose == "tis":
        return ratio.clamp_max(spec.tis_clip)
    if spec.compose == "mis":
        return torch.where((ratio >= spec.mis_low) & (ratio <= spec.mis_high), ratio, torch.zeros_like(ratio))
    raise ValueError(f"unknown score_centering compose {spec.compose!r}")


@dataclass(frozen=True)
class ScoreCenteringTerms:
    """Per-token pieces of Eq. 14 plus detached diagnostics (rank-local SUMS and one MAX)."""

    correction: torch.Tensor  # [N] live through the trainer's head logprobs; coefficients detached
    token_weight: torch.Tensor  # [N] detached sg[w_{y_t}]; exactly 1.0 under compose="none"
    stats: dict[str, float]


def score_centering_terms(
    head_logp: torch.Tensor,
    head_sampler_logp: torch.Tensor,
    token_logp: torch.Tensor,
    sampler_token_logp: torch.Tensor,
    *,
    spec: ScoreCenteringSpec,
) -> ScoreCenteringTerms:  # head_*: [N,k]; token_*: [N]
    """The centering correction and the sampled-token weight for one flat batch of tokens.

    Args:
        head_logp: trainer ``log p`` gathered at the SAMPLER's top-k ids, ``[N, k]``, live (the only
            tensor the correction back-propagates through). Float32.
        head_sampler_logp: sampler ``log q`` at the same ids, ``[N, k]``, treated as constants.
        token_logp: trainer ``log p`` of the sampled tokens, ``[N]`` (only its detached value is read
            here; the caller keeps the live one in its own surrogate).
        sampler_token_logp: sampler ``log q`` of the sampled tokens, ``[N]`` — the TRUE ``q_{y_t}``,
            kept even when ``y_t`` is outside the head (README rule).
        spec: the validated ``rl.score_centering`` view.

    Returns:
        ``correction`` ``[N]`` = ``Σ_H sg[q_v w_v − α p_v] log p_v``; ``token_weight`` ``[N]`` =
        ``sg[f(p_{y_t}/q_{y_t})]``; ``stats`` with detached sums (``sc_head_mass_q_sum``,
        ``sc_head_mass_p_sum``, ``sc_rho_sum``, ``sc_kl_head_sum``, ``sc_correction_abs_sum``,
        ``sc_token_weight_sum``) and ``sc_rho_max``.
    """
    if head_logp.dim() != 2 or head_sampler_logp.shape != head_logp.shape:
        raise ValueError(
            f"score centering needs matching [N,k] head tensors, got {tuple(head_logp.shape)} and "
            f"{tuple(head_sampler_logp.shape)}"
        )
    if token_logp.shape != sampler_token_logp.shape or token_logp.shape != head_logp.shape[:1]:
        raise ValueError(
            f"score centering needs one sampled-token logprob per head row, got token {tuple(token_logp.shape)}, "
            f"sampler {tuple(sampler_token_logp.shape)}, head rows {head_logp.shape[0]}"
        )
    if head_logp.shape[1] != spec.top_k:
        raise ValueError(f"score centering head width {head_logp.shape[1]} != score_centering.top_k={spec.top_k}")
    with torch.no_grad():
        # Coefficients: every quantity below is a constant of the update (Eq. 14's sg[·]).
        p_head = head_logp.exp()  # [N,k]
        q_head = head_sampler_logp.to(head_logp.dtype).exp()  # [N,k]
        p_tail = (1.0 - p_head.sum(-1)).clamp_min(spec.tail_eps)  # [N]
        q_tail = (1.0 - q_head.sum(-1)).clamp_min(spec.tail_eps)  # [N]
        rho = q_tail / p_tail  # [N]  (== 1.0 exactly when q_head == p_head)
        alpha = rho * importance_weight(1.0 / rho, spec)  # [N]  ρ f(1/ρ); == ρ under compose="none"
        head_weight = importance_weight((head_logp - head_sampler_logp.to(head_logp.dtype)).exp(), spec)  # [N,k]
        residual = q_head * head_weight - alpha[:, None] * p_head  # [N,k]  q_v w_v − α p_v
        token_weight = importance_weight(
            (token_logp.detach() - sampler_token_logp.to(token_logp.dtype)).exp(), spec
        )  # [N]
    # LIVE only through head_logp: gradient Σ_H (q_v w_v − α p_v) ∇ log p_v = E_q̂[ŵ ∇ log p] (Eq. 12).
    correction = (residual * head_logp).sum(-1)  # [N]
    with torch.no_grad():
        kl_head = torch.where(
            q_head > 0, q_head * (head_sampler_logp.to(head_logp.dtype) - head_logp), torch.zeros_like(q_head)
        ).sum(-1)  # [N]  Σ_H q log(q/p): the sampler/trainer disagreement on the head, same temperature
        stats = {
            "sc_head_mass_q_sum": float(q_head.sum()),
            "sc_head_mass_p_sum": float(p_head.sum()),
            "sc_rho_sum": float(rho.sum()),
            "sc_rho_max": float(rho.max()) if rho.numel() else 0.0,
            "sc_kl_head_sum": float(kl_head.sum()),
            "sc_correction_abs_sum": float(correction.detach().abs().sum()),
            "sc_token_weight_sum": float(token_weight.sum()),
        }
    return ScoreCenteringTerms(correction=correction, token_weight=token_weight, stats=stats)


def gather_head_records(
    head_ids: list[Any],
    head_logprobs: list[Any],
    selection: torch.Tensor,
    *,
    top_k: int,
    device: torch.device,
    dtype: torch.dtype,
    vocab_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:  # selection: [B,L-1]; returns ([N,k] long, [N,k] dtype)
    """Flatten per-sequence ``[T_i, k]`` sampler heads into the loss's masked token order.

    ``vocab_size`` (the trainer's ``logits.shape[-1]``) makes this the ONE place an out-of-vocabulary
    head id is refused: the caller runs it inside ``run_rank_converged``, so every DP rank agrees to fail
    before any rank reaches the head ``gather`` (a rank-local CUDA index fault there would leave its
    peers waiting in the next collective until the process-group timeout).

    Mirrors ``gather_behavior_logprobs``: each sequence contributes its TRAILING ``count_i`` rows
    (a masked shared prefix drops leading completion tokens, never trailing ones), concatenated in
    sequence order, so row ``n`` pairs with ``current[n]``. Refuses a missing record, a width other
    than ``top_k``, fewer rows than masked tokens, or a non-finite/positive logprob other than
    ``-inf`` (a masked-out sampler token; ``exp`` makes it a zero mass).
    """
    counts = selection.sum(dim=1).tolist()  # [B] -> list[int]
    if len(head_ids) != len(counts) or len(head_logprobs) != len(counts):
        raise ValueError(
            f"score centering got {len(head_ids)} id records and {len(head_logprobs)} logprob records for "
            f"{len(counts)} sequences"
        )
    ids_out: list[torch.Tensor] = []
    lp_out: list[torch.Tensor] = []
    for index, (ids, lps, count) in enumerate(zip(head_ids, head_logprobs, counts)):
        if ids is None or lps is None:
            raise ValueError(
                f"score centering requires policy_info['rollout_topk_ids'/'rollout_topk_logprobs'] on every "
                f"trajectory; sequence {index} carries none (producer built without score centering?)"
            )
        ids_t = torch.as_tensor(ids)
        lps_t = torch.as_tensor(lps)
        if ids_t.dim() != 2 or ids_t.shape[1] != top_k or lps_t.shape != ids_t.shape:
            raise ValueError(
                f"score centering record {index} must be [T, {top_k}] ids with matching logprobs, got ids "
                f"{tuple(ids_t.shape)} and logprobs {tuple(lps_t.shape)}"
            )
        if ids_t.shape[0] < count:
            raise ValueError(
                f"score centering record {index} has {ids_t.shape[0]} rows < masked completion token count {count}"
            )
        lps32 = lps_t.to(torch.float32)
        bad = ~(torch.isfinite(lps32) | torch.isneginf(lps32)) | (lps32 > 1.0e-6)
        if bool(bad.any()):
            raise ValueError(f"score centering record {index} has {int(bad.sum())} invalid sampler logprob entries")
        if vocab_size is not None:
            if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1:
                raise ValueError(f"vocab_size must be a positive integer, got {vocab_size!r}")
            ids_long = ids_t.to(torch.long)
            out_of_range = (ids_long < 0) | (ids_long >= vocab_size)
            if bool(out_of_range.any()):
                worst = ids_long[out_of_range]
                raise ValueError(
                    f"score centering record {index} has {int(out_of_range.sum())} sampler head id(s) outside the trainer "
                    f"vocabulary [0, {vocab_size}) (e.g. {int(worst.min())} .. {int(worst.max())}): the rollout and trainer "
                    "vocabularies disagree (different tokenizer or vocab padding?)"
                )
        if count:
            ids_out.append(ids_t[-count:].to(torch.long))
            lp_out.append(lps_t[-count:])
    if not ids_out:
        return (
            torch.empty((0, top_k), dtype=torch.long, device=device),
            torch.empty((0, top_k), dtype=dtype, device=device),
        )
    return torch.cat(ids_out).to(device), torch.cat(lp_out).to(device=device, dtype=dtype)  # [N,k], [N,k]


def completion_token_and_head_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    selection: torch.Tensor,
    head_ids: torch.Tensor,
    *,
    temperature: float,
    dtype: torch.dtype,
) -> tuple[
    torch.Tensor, torch.Tensor
]:  # logits [B,L,V], input_ids [B,L], selection [B,L-1], head_ids [N,k] -> [N], [N,k]
    """One tempered log-softmax; the sampled-token logprobs AND the trainer logprobs at the sampler head.

    The ``[N]`` result is computed by exactly the ops of ``algorithm/grpo.py::_completion_token_logprobs``
    (``logits / temperature``, ``log_softmax`` in ``dtype`` over ``logits[:, :-1]``, gather
    ``input_ids[:, 1:]``, select ``masks[:, :-1]``), so enabling centering does not change ``current``.
    The head gather goes through a ``[B, L-1, k]`` index (zeros at unselected positions) rather than a
    ``[N, V]`` slice of the log-softmax, so no second vocab-wide tensor is materialised.
    """
    if head_ids.dim() != 2 or head_ids.shape[0] != int(selection.sum()):
        raise ValueError(
            f"head_ids must be [N,k] with N == selected positions ({int(selection.sum())}), got {tuple(head_ids.shape)}"
        )
    tempered = logits / temperature  # [B,L,V]
    logp_all = torch.log_softmax(tempered[:, :-1], dim=-1, dtype=dtype)  # [B,L-1,V]
    tok_logp = logp_all.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B,L-1]
    k = head_ids.shape[1]
    vocab = int(logp_all.shape[-1])
    if head_ids.numel() and (int(head_ids.min()) < 0 or int(head_ids.max()) >= vocab):
        raise ValueError(
            f"head_ids carry a token id outside the trainer vocabulary [0, {vocab}); gather_head_records(vocab_size=...) "
            "must have refused this inside the rank-converged preparation"
        )
    full_index = torch.zeros((*selection.shape, k), dtype=torch.long, device=logits.device)  # [B,L-1,k]
    full_index[selection] = head_ids.to(device=logits.device, dtype=torch.long)
    head_logp = logp_all.gather(-1, full_index)[selection]  # [N,k]
    return tok_logp[selection], head_logp  # [N], [N,k]
