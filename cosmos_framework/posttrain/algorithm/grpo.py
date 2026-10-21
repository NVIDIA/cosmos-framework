# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GRPO policy-gradient objectives (DESIGN §8.2 #2).

``make_objective`` is the ``legacy_pg`` surrogate: group-relative advantage plus a plain
policy-gradient loss, with no importance ratio, clip, or KL. ``make_clipped_objective`` is the
production surrogate with frozen π_old, clipping, and an optional frozen-reference K3 KL anchor
(``rl.kl_beta > 0``; see ``FrozenReference``).

token_sum reduction: the objective sums per-token `advantage · logπ` over the completion and divides
by the DP-global completion-token count (all-reduced here), so longer completions carry proportional
weight and the loss is DP-world-size invariant.

Truncation: the producer's ``policy_info["finish_reason"]`` verdict (``schema.FINISH_REASON_KEY``)
says whether the engine cut a completion at its token budget. The clipped objective reports how many
of the completions it was handed were cut and how long responses run (``train/rollout_truncated_frac``,
``train/response_len_mean``, ``train/response_len_max``), and under ``rl.mask_truncated_completions``
removes the cut ones from the loss (DAPO overlong filtering).
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import torch

from cosmos_framework.posttrain.algorithm.annotators.advantage import advantage_to_tokens, require_advantages
from cosmos_framework.posttrain.algorithm.primitives.logprob_gap import (
    NO_LOGPROB_GAP,
    logprob_gap_stats,
    selected_token_logprobs,
)
from cosmos_framework.posttrain.algorithm.primitives.objective import (
    OBJECTIVE_WEIGHT_SUM,
    MetricAdapter,
    Objective,
    Reduction,
    normalization_scale,
)
from cosmos_framework.posttrain.algorithm.primitives.policy_forward import (
    forward_policy,
    policy_logits,
    prepare_policy_capability,
    validate_collated_policy_batch,
)
from cosmos_framework.posttrain.algorithm.primitives.policy_gradient import (
    FIXED_RESPONSE_POLICY_GRADIENT_LOSS_TYPE,
    IMPORTANCE_SAMPLING_LEVELS,
    POLICY_GRADIENT_LOSS_TYPES,
    clipped_policy_gradient_loss,
    gather_behavior_logprobs,
    reduce_policy_gradient_loss,
)
from cosmos_framework.posttrain.algorithm.primitives.score_centering import (
    ScoreCenteringTerms,
    completion_token_and_head_logprobs,
    gather_head_records,
    resolve_score_centering,
    score_centering_terms,
)
from cosmos_framework.posttrain.data.packer.base import DataPacker
from cosmos_framework.posttrain.data.packer.decoder_only_llm import RLPolicyInput, shape_policy_input
from cosmos_framework.posttrain.schema import (
    ROLLOUT_TOPK_IDS_KEY,
    ROLLOUT_TOPK_LOGPROBS_KEY,
    Trajectory,
    Transition,
    is_length_truncated,
)
from cosmos_framework.posttrain.utils.dist import (
    raise_if_any_rank_failed,
    rank_max_count,
    rank_mean_count,
    run_rank_converged,
)

_GRPO_PUBLIC_METRICS: dict[str, str] = {
    "grpo_loss": "loss_avg",
    "grpo_n": "valid_sequence_count",
    "grpo_valid_tokens": "valid_token_count",
    # Clip diagnostics, emitted as per-rank RATIOS and token-weighted at aggregation (below), so the
    # reported value is Σclip_count/Σtokens across ranks -- not a mean of per-rank fractions.
    "grpo_clip_frac": "clip_frac",
    "grpo_ratio_absdev": "ratio_absdev",
    # Response-level clip diagnostics, per-rank RATIOS over sequences, sequence-weighted below.
    # `seq_clip_frac` is the fraction of responses the clip touched; `seq_log_ratio_abs` is the mean
    # over responses of |mean_t log-ratio|, the magnitude a sequence-level (GSPO) clip compares
    # against ln(1±ε). Emitted at BOTH importance-sampling levels so the key set is config-invariant.
    "grpo_seq_clip_frac": "seq_clip_frac",
    "grpo_seq_log_ratio_abs": "seq_log_ratio_abs",
    # Train-vs-rollout logprob gap per loss-masked completion token, |logπ_train(T=1) − logπ_rollout|
    # in nats (algorithm/primitives/logprob_gap.py). Emitted whenever the producer recorded rollout
    # logprobs, independently of whether the loss uses them; `logprob_gap_token_count` is the number
    # of tokens compared and is 0 (with zero-valued gap keys) when none were recorded.
    "grpo_logprob_gap_mean": "logprob_gap_mean",
    # "mean_rank_": quantiles are not composable by weighted mean, so the aggregate is the
    # token-weighted mean of per-rank p99s -- named for what it is, not for the global p99.
    "grpo_logprob_gap_p99": "mean_rank_logprob_gap_p99",
    "grpo_logprob_gap_frac": "logprob_gap_gt0p5_frac",
    "grpo_logprob_gap_tokens": "logprob_gap_token_count",
    # Reference-anchored KL (rl.kl_beta > 0, `FrozenReference`): the per-rank MEAN over loss-masked
    # completion tokens of the K3 penalty exp(z)-z-1, z = logπ_ref − logπ_θ, BEFORE the kl_beta
    # scale; token-weighted below so the aggregate is Σk3/Σtokens. 0.0 without a reference.
    # `kl_beta` is echoed beside it so a run's stats say which anchor strength produced the curve.
    "grpo_kl_mean": "kl_mean",
    "grpo_kl_beta": "kl_beta",
    # The largest per-token K3 penalty in the batch (same scale as `kl_mean`, before `kl_beta`), reduced
    # by MAX across ranks and chunks (`max_reduced` below). The mean cannot tell "the anchor pulls a
    # little everywhere" from "one token owns the step": measured on a sparse-MoE policy, a
    # reference-vs-policy forward discrepancy on a few tokens drove kl_mean to 173 against a median of
    # 0.028 and the pre-clip grad_norm to 114 against 0.029; the max names that token's penalty directly.
    "grpo_kl_max": "kl_max",
    # Response-length and truncation diagnostics over the completions the chunk was HANDED (before
    # an empty completion or `rl.mask_truncated_completions` drops one, so the rate reads the same
    # with the mask on or off): their count, the fraction the engine cut at its token budget
    # (`policy_info["finish_reason"] == "length"`, the vLLM rollout's stamp; 0.0 when the producer
    # stamps none), their mean generated length in tokens and the longest one. The two ratios are
    # count-weighted below so the aggregates are Σcut/Σcompletions and Σtokens/Σcompletions; the
    # count is summed (`SUMMED_TRAIN_COUNT_KEYS`) and the max is declared `max_reduced` below, the
    # one declaration point for every maximum: a mean of per-rank or per-chunk maxima would
    # understate the longest completion of the step, the number that says whether anything reached
    # `rollout.max_tokens`.
    "grpo_response_count": "response_count",
    "grpo_truncated_frac": "rollout_truncated_frac",
    "grpo_response_len_mean": "response_len_mean",
    "grpo_response_len_max": "response_len_max",
    # Score centering (`rl.score_centering.enabled`; algorithm/primitives/score_centering.py), over the
    # loss-masked completion tokens the correction was computed on (`sc_token_count`, the weight of
    # every mean below; 0 with zero-valued keys when centering is off). `sc_head_mass_q` / `_p` are the
    # sampler's and trainer's probability mass inside the sampler's top-k head (how much of q the
    # head form sees; the reference reports >= 0.99 at k = 128); `sc_rho_mean` / `sc_rho_max` the
    # tail-mass ratio q_tail/p_tail (1.0 when the two agree; a large max is a confident trainer against
    # an unsure sampler — the place the 1e-6 floor bites); `sc_kl_head` = Σ_H q log(q/p), the
    # sampler-vs-trainer disagreement on the head at the SAME temperature (the mismatch the
    # correction removes; compare `logprob_gap_mean`, a sampled-token statistic); `sc_correction_abs`
    # the mean |correction| per token (exactly 0 when q == p); `sc_token_weight` the mean composed
    # sampled-token weight (exactly 1.0 for compose="none"); `sc_out_of_head_frac` the fraction of
    # sampled tokens outside the sampler's own head.
    "grpo_sc_head_mass_q": "sc_head_mass_q",
    "grpo_sc_head_mass_p": "sc_head_mass_p",
    "grpo_sc_rho_mean": "sc_rho_mean",
    "grpo_sc_rho_max": "sc_rho_max",
    "grpo_sc_kl_head": "sc_kl_head",
    "grpo_sc_correction_abs": "sc_correction_abs",
    "grpo_sc_token_weight": "sc_token_weight",
    "grpo_sc_out_of_head_frac": "sc_out_of_head_frac",
    "grpo_sc_tokens": "sc_token_count",
    OBJECTIVE_WEIGHT_SUM: "weight_sum",
}
_GRPO_METRIC_ADAPTER = MetricAdapter(
    names=_GRPO_PUBLIC_METRICS,
    internal_prefix="grpo_",
    phases=frozenset({"train"}),
    max_reduced=frozenset({"grpo_kl_max", "grpo_response_len_max", "grpo_sc_rho_max"}),
    weighted_by={
        "grpo_loss": OBJECTIVE_WEIGHT_SUM,
        # Token-weight the clip fractions, exactly as grpo_loss is weighted by the DP-mean divisor,
        # so the aggregate is the global clip fraction, not a mean of per-rank fractions.
        "grpo_clip_frac": "grpo_valid_tokens",
        "grpo_ratio_absdev": "grpo_valid_tokens",
        # Sequence-weight the response-level diagnostics for the same reason.
        "grpo_seq_clip_frac": "grpo_n",
        "grpo_seq_log_ratio_abs": "grpo_n",
        # Weighted by the tokens actually compared, not by valid tokens: a shard without rollout
        # logprobs then carries zero weight instead of diluting the gap with zeros.
        "grpo_logprob_gap_mean": "grpo_logprob_gap_tokens",
        "grpo_logprob_gap_p99": "grpo_logprob_gap_tokens",
        "grpo_logprob_gap_frac": "grpo_logprob_gap_tokens",
        "grpo_kl_mean": "grpo_valid_tokens",
        "grpo_truncated_frac": "grpo_response_count",
        "grpo_response_len_mean": "grpo_response_count",
        # Weighted by the tokens the correction covered: a shard without centering weighs zero.
        "grpo_sc_head_mass_q": "grpo_sc_tokens",
        "grpo_sc_head_mass_p": "grpo_sc_tokens",
        "grpo_sc_rho_mean": "grpo_sc_tokens",
        "grpo_sc_kl_head": "grpo_sc_tokens",
        "grpo_sc_correction_abs": "grpo_sc_tokens",
        "grpo_sc_token_weight": "grpo_sc_tokens",
        "grpo_sc_out_of_head_frac": "grpo_sc_tokens",
    },
)


def make_objective(packer: DataPacker | None = None, temperature: float = 1.0) -> Objective:
    """Build the reference on-policy GRPO objective with token-sum reduction.

    Args:
        packer: Optional fallback tokenizer for trajectories without stored ``prompt_ids``.
        temperature: Positive policy-logit temperature used by the on-policy likelihood.

    Returns:
        An objective normalized by the DP-mean completion-token count. Its metrics include the
        divisor as ``OBJECTIVE_WEIGHT_SUM`` so the trainer can normalize an accumulated window once.
    """

    if isinstance(temperature, bool) or not isinstance(temperature, Real):
        raise TypeError(f"temperature must be a non-bool real number, got {temperature!r}")
    resolved_temperature = float(temperature)
    if not math.isfinite(resolved_temperature) or resolved_temperature <= 0:
        raise ValueError(f"temperature must be finite and positive, got {temperature!r}")

    def objective(model: Any, trajectories: list[Trajectory]) -> tuple[torch.Tensor, dict]:
        require_advantages(trajectories)
        net = getattr(model, "net", model)  # raw module IS its own net; HFModel owns it under .net
        device = next(net.parameters()).device
        numerator = torch.zeros((), device=device)  # []
        n_tokens, n_seq = 0, 0
        # FSDP attaches collectives to forwards, so uneven shards agree on a count and short ranks
        # issue zero-gradient padding forwards before the denominator collective.
        n_forwards = rank_max_count(len(trajectories), device)
        if n_forwards == 0:
            raise ValueError("GRPO objective got an empty batch — no trajectories to learn from.")

        def prepare() -> list[tuple[Transition, RLPolicyInput, torch.Tensor | None]]:
            shaped: list[tuple[Transition, RLPolicyInput, torch.Tensor | None]] = []
            for traj in trajectories:
                tr = traj.transitions[0]
                policy_input = shape_policy_input(
                    tr,
                    packer,
                    int(tr.algo_extra.get("n_ignore_prefix_tokens", 0)),
                )
                if getattr(policy_input, "mm_kwargs", None):
                    raise ValueError(
                        "legacy GRPO cannot score multimodal policy inputs; use the clipped "
                        "surrogate, whose policy-forward contract preserves model-specific inputs"
                    )
                # Keep scalar arithmetic unchanged for bit-for-bit parity. Tensor advantages are
                # width-checked before any forward so a rank-local data error converges safely.
                advantage_tokens = (
                    advantage_to_tokens(
                        tr.advantage,
                        sum(policy_input.logprob_masks[:-1]),
                        device=device,
                    )
                    if isinstance(tr.advantage, torch.Tensor)
                    else None
                )  # [N] or None
                shaped.append((tr, policy_input, advantage_tokens))
            return shaped

        shaped = run_rank_converged(
            prepare,
            "GRPO objective preparation",
            raise_errors=raise_if_any_rank_failed,
        )
        for tr, policy_input, advantage_tokens in shaped:
            ids = torch.tensor(policy_input.input_ids, device=device).unsqueeze(0)  # [1,L]
            logits = net(ids).logits[0]  # [L,V]
            if not any(policy_input.logprob_masks):
                numerator = numerator + logits.sum() * 0.0  # []
                continue
            mask = torch.tensor(policy_input.logprob_masks, dtype=torch.bool, device=device)  # [L]
            logp_all = torch.log_softmax(logits[:-1] / resolved_temperature, dim=-1)  # [L-1,V]
            tok_logp = logp_all.gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)  # [L-1]
            comp_logp = tok_logp[mask[:-1]]  # [N]
            if advantage_tokens is None:
                numerator = numerator + float(tr.advantage) * comp_logp.sum()  # []
            else:
                numerator = numerator + (advantage_tokens * comp_logp).sum()  # []
            n_tokens += comp_logp.numel()
            n_seq += 1
        for _ in range(n_forwards - len(trajectories)):
            pad = torch.zeros((1, 2), dtype=torch.long, device=device)  # [1,2]
            numerator = numerator + net(pad).logits.sum() * 0.0  # []
        denom = rank_mean_count(n_tokens, device)
        loss = -numerator * normalization_scale(denom)  # []
        return loss, {
            "grpo_loss": float(loss.detach()),
            "grpo_n": n_seq,
            "grpo_valid_tokens": n_tokens,
            OBJECTIVE_WEIGHT_SUM: denom,
        }

    return Objective(objective, reduction=Reduction.TOKEN_SUM, metric_adapter=_GRPO_METRIC_ADAPTER)


class OldLogprobCache:
    """π_old FROZEN at the policy that collected the batch, held across its optimizer-step window.

    WHY. The objective below recomputes π_old as `cur.detach()`, making the ratio `exp(0) == 1` so
    the epsilon-clip cannot bind. That is EXACT when a collected batch drives ONE optimizer step
    (`n_mini == 1`): nothing moves between the forward and the update. It becomes wrong the moment
    `rl.mini_batch_per_replica` splits the batch into `n_mini > 1` SEQUENTIAL updates — π_old then
    tracks the drifting policy, the clip silently stops binding, and the run takes `n_mini`
    unconstrained steps on one batch of experience.

    MEASURED, Qwen2.5-3B / GSM8K, identical recipe either side: at `n_mini=32` reward collapsed to
    exactly 0.0 at step 29 of 100 and the policy emitted token soup for the remaining 53 steps; at
    `n_mini=1` the same recipe trained cleanly to 0.94 and matched cosmos-rl's curve. `config/rl.py`
    already names this failure verbatim ("N unconstrained steps on one batch") while refusing
    `mu_iterations > 1` — inner epochs and `n_mini > 1` are the SAME hazard through different doors,
    and only that one was bolted.

    cosmos-rl does exactly this: when `batch_size > per_optimize_batch_size` it prepends a
    `TrainerPhase.OLD_LOGP_COMPUTE` pass (model in eval, no grad) filling `old_per_token_logps`
    before any update, then reads it during training
    (`policy/trainer/llm_trainer/grpo_trainer.py`). This is that pass.

    KEYED BY CALL ORDER, like cosmos-rl's `old_per_token_logps[local_mini_step]`, because the
    dispatch ingress CONSUMES `(mini, micro)` and the trainer body only ever sees `train(batch)`
    (`roles/core/dispatch.py`, spec 2026-08-12 §3.4). The record pass and the train pass iterate the
    same `(mini, micro)` product in the same order on the same rank, so chunk N of one is chunk N of
    the other. An empty shard returns before reaching π_old in BOTH passes — `select_chunk` is
    deterministic — so the two stay aligned without special-casing.
    """

    __slots__ = ("_store", "_read", "mode", "_probe")

    def __init__(self) -> None:
        self._store: list = []
        self._read = 0
        self._probe = None  # scratch slot for the determinism probe; never part of the window
        self.mode = "off"  # "off" -> today's behaviour | "record" (ahead pass) | "replay" (training)

    def begin_record(self) -> None:
        """Enter the ahead pass. Idempotent WITHIN a pass: only the first call clears."""
        if self.mode != "record":
            self._store.clear()
            self._read = 0
            self.mode = "record"

    def begin_probe(self) -> None:
        """Capture the next `old_for` into a scratch slot WITHOUT touching the recorded window.

        Used once per run to verify empirically that a train-mode forward reproduces the eval-mode
        π_old at unchanged weights — the invariant the ahead pass depends on. Kept off `_store` so
        the probe can never be replayed as if it were a chunk's π_old.
        """
        self.mode = "probe"

    def skip_chunk(self) -> None:
        """Advance the read pointer for a chunk the training pass DISCARDED without scoring.

        The window is keyed by APPEND ORDER: chunk k of the training pass replays `_store[k]`. That
        pairing holds only while both passes consume the same chunks. `LLMTrainer.train` has one path
        that breaks it -- the poisoned-window discard (`roles/trainer/llm.py`, section 5.2) returns
        BEFORE calling the objective, so `old_for` never runs and `_read` never advances, while the
        ahead pass DID record a pi_old for that chunk. Every later chunk then replays a reference
        belonging to different trajectories.

        That failure is silent and worse than not freezing at all. It under-consumes, so `old_for`'s
        exhaustion guard cannot fire and `begin_replay_window`'s count check is an upper bound only.
        And `exp(log pi_theta(tokens_A) - log pi_old(tokens_B))` is not a ratio: it saturates the
        +/-20 clamp and pins most tokens at the clip boundary -- exactly what `begin_replay`'s
        docstring means by "refusing rather than clipping against the wrong tokens". It persists to
        the end of the collected batch, since only the next `begin_record` clears the window.

        Reachable only at `n_micro > 1`: at `grad_accum == 1` the skip count is
        `max(0, 1 - 0 - 1) == 0`, so nothing is ever discarded.

        Advancing here keeps the append-order scheme, and with it the correspondence to cosmos-rl's
        `old_per_token_logps[local_mini_step]`. Outside replay it is a no-op: with no window there is
        no pairing to maintain.
        """
        if self.mode == "replay":
            self._read += 1

    @property
    def last_recorded(self):
        """The most recently recorded π_old, or None. For the determinism probe only."""
        return self._store[-1] if self._store else None

    @property
    def probe(self):
        """The tensor captured by the last `begin_probe()` window."""
        return self._probe

    def begin_replay(self, expected: int | None = None) -> None:
        """Freeze what was recorded and rewind for the training pass.

        Args:
            expected: chunks the driver walked (`n_mini * n_micro`). Checked when given.

        Raises:
            RuntimeError: the window does not hold exactly `expected` records. Records are keyed by
                APPEND ORDER alone, so an ahead-pass call delivered twice (Ray retry, actor
                restart mid-pass) appends a duplicate π_old and every subsequent chunk then replays
                its PREDECESSOR's reference — a silent wrong-reference update that the exhaustion
                guard cannot see, because the count still runs out at the right time. A short window
                (a lost call) mis-pairs the same way. Neither is recoverable here: the recorded
                references cannot be re-attributed to chunks after the fact, so this refuses rather
                than training against a reference that belongs to different tokens.
        """
        if expected is not None and len(self._store) != int(expected):
            raise RuntimeError(
                f"π_old window holds {len(self._store)} records but the driver walked {expected} "
                "chunks. Records are keyed by append order, so a duplicated or lost ahead-pass call "
                "shifts every later chunk onto the wrong reference — silently, since the count still "
                "exhausts on schedule. Refusing rather than clipping against the wrong tokens."
            )
        self._read = 0
        self.mode = "replay"

    def old_for(self, cur):
        """Return π_old for this chunk's `cur`, advancing the state machine.

        Args:
            cur: this chunk's current-policy flat logprobs `[n_tok]`.

        Returns:
            The frozen π_old when replaying, else `cur.detach()` (recording it when in the ahead
            pass). Falling back to `cur.detach()` keeps `n_mini == 1` byte-identical to before.

        Raises:
            RuntimeError: replaying past the end, i.e. the training pass asked for more chunks than
                the ahead pass recorded — a driver-order bug, and silently returning `cur.detach()`
                would restore the very failure this class exists to prevent.
        """
        if self.mode == "replay":
            if self._read >= len(self._store):
                raise RuntimeError(
                    f"OldLogprobCache exhausted: training asked for chunk {self._read} but the "
                    f"ahead pass recorded {len(self._store)}. The record and train passes must "
                    "iterate the same (mini, micro) product in the same order."
                )
            old = self._store[self._read]
            self._read += 1
            # Deliberately STAY in "replay" once exhausted rather than falling back to "off": an
            # extra train() would then silently score against `cur.detach()` — the exact unclipped
            # update this class exists to prevent, and invisible in every metric. The next batch's
            # `begin_record` clears the window instead.
            return old
        old = cur.detach()
        if self.mode == "record":
            self._store.append(old)
        elif self.mode == "probe":
            self._probe = old  # scratch: deliberately NOT appended to the window
        return old

    @property
    def recorded(self) -> int:
        return len(self._store)


def _completion_token_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    selection: torch.Tensor,
    *,
    temperature: float,
    dtype: torch.dtype,
) -> torch.Tensor:  # logits [B,L,V], input_ids [B,L], selection [B,L-1] -> [N]
    """Tempered per-token logprobs of the next token, flattened over the loss-masked positions.

    ONE implementation for the current policy and the frozen reference, so both score the same
    tokens through the same ops: logits / temperature, `log_softmax` in `dtype` (fp32 by default,
    see the note in the objective) over `logits[:, :-1]`, gather `input_ids[:, 1:]`, select
    `masks[:, :-1]`. The reference logprobs are consumed by a difference against the current ones
    (`k3(z)`, z = ref − cur), so any asymmetry here would report as KL.
    """
    tempered = logits / temperature  # [B,L,V]
    logp_all = torch.log_softmax(tempered[:, :-1], dim=-1, dtype=dtype)  # [B,L-1,V]
    tok_logp = logp_all.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B,L-1]
    return tok_logp[selection]  # [N]


class FrozenReference:
    """The frozen reference policy that anchors the clipped objective's KL term (`rl.kl_beta > 0`).

    WHAT. A second, FSDP-sharded instance of the SAME policy adapter, built by the SAME builder from
    the SAME base checkpoint the policy loads (`entrypoints/grpo.py::build_trainer`), then frozen
    (`requires_grad_(False)`, `eval()`) and never updated — `rl.ref.update_every` must be 0, the
    same frozen-anchor semantics DPO/NFT use (`config/rl.py::RefConfig`). It scores every policy
    training batch under `torch.no_grad()` through the shared forward seam
    (`prepare_policy_capability` → `forward_policy` → `policy_logits`) and returns the per-token
    logprobs of the SAME tokens under the SAME masks, temperature, and logprob dtype as the current
    policy's, which `clipped_policy_gradient_loss(..., reference=, kl_beta=)` then turns into
    `kl_beta · k3` per token. Because a resume restores only the policy, the anchor is always the
    run's step-0 weights (the "initial policy"), as for DPO's un-checkpointed `net_ref`.

    WHY A HOLDER. The objective closure is built before any model exists (`build_objective_fn`
    receives only the config and packer), so the twin is attached to this slot after construction
    and read from it by the objective (scoring) and by nothing else. It is NOT a submodule of the
    policy adapter, so checkpoint saves, resumes, and weight publication never see it
    (`OldLogprobCache` is the same plumbing precedent). Its DEVICE RESIDENCY is not this class's
    concern either: the objective is the algorithm and consumes scores, the trainer is the engine
    and owns what leaves the card under colocation, so `build_trainer` hands the same model it
    attaches here to `LLMTrainer(auxiliary_models=...)` and the trainer never reaches into the
    objective for a network. Memory: one sharded, gradient-free copy of the network per policy rank.

    COLLECTIVE PARITY. The reference forward is an FSDP forward, so it runs on EVERY rank on every
    chunk the objective scores for training — including an empty local shard (on the probe batch,
    result discarded) — exactly like the policy forward whose collectives it mirrors. Skipping it by
    any rank-local condition would strand peers in its all-gathers. The one condition it IS skipped
    on is rank-uniform: while `OldLogprobCache` is recording (the π_old ahead pass) or probing (the
    one-shot determinism probe), both of which discard the loss, no rank scores the reference; the
    driver sets that mode identically everywhere before each pass.
    """

    __slots__ = ("model", "label")

    def __init__(self, label: str = "reference") -> None:
        self.model: Any = None  # the frozen policy adapter (or raw module), None until attached
        self.label = label

    @property
    def attached(self) -> bool:
        return self.model is not None

    @property
    def net(self) -> Any:
        """The frozen raw module, or None before `attach`. The module `score` forwards through."""
        return None if self.model is None else getattr(self.model, "net", self.model)

    def attach(self, model: Any) -> "FrozenReference":
        """Freeze `model` (no grads, eval mode) and make it this objective's KL anchor.

        Args:
            model: a policy adapter (with `.net`) or raw module built like the policy, holding the
                weights the policy started from.

        Returns:
            self, for chaining.

        Raises:
            RuntimeError: a reference is already attached (the anchor is built once and never
                replaced; a second attach is a wiring bug, not a refresh).
            TypeError: `model` exposes no `torch.nn.Module` to freeze.
        """
        if self.model is not None:
            raise RuntimeError(f"a frozen {self.label} is already attached; it is built once and never replaced")
        net = getattr(model, "net", model)
        if not isinstance(net, torch.nn.Module):
            raise TypeError(f"frozen {self.label} must wrap a torch.nn.Module, got {type(net).__name__}")
        net.requires_grad_(False)
        net.eval()
        self.model = model
        return self

    def score(
        self,
        batch: dict[str, Any],
        input_ids: torch.Tensor,
        selection: torch.Tensor,
        *,
        temperature: float,
        dtype: torch.dtype,
        owner: str,
    ) -> torch.Tensor:  # input_ids [B,L], selection [B,L-1] -> [N]
        """Per-token reference logprobs of the collated batch's loss-masked completion tokens.

        Mirrors the policy path one-for-one — its own `prepare_policy_capability` (so a routing
        replay carried by the batch pins THIS adapter's blocks, not the policy's), the same forward
        seam and validation, then `_completion_token_logprobs` — under `torch.no_grad()` so the
        result carries no graph. Called on every rank whenever the objective scores a training
        chunk; never while the π_old cache records or probes (see the class docstring). Preflight
        errors are converged like the policy's so no rank enters the FSDP forward alone.
        """
        if self.model is None:
            raise RuntimeError(f"frozen {self.label} is not attached")
        model = self.model
        net = self.net
        with torch.no_grad():
            prepared = run_rank_converged(
                lambda: prepare_policy_capability(model, batch),
                f"{owner} {self.label} capability preflight",
                raise_errors=raise_if_any_rank_failed,
            )
            output = forward_policy(model, net, prepared)
            logits = run_rank_converged(
                lambda: policy_logits(output, input_ids, owner=f"{owner} {self.label}"),
                f"{owner} {self.label} forward validation",
                raise_errors=raise_if_any_rank_failed,
            )  # [B,L,V]
            return _completion_token_logprobs(logits, input_ids, selection, temperature=temperature, dtype=dtype)


def _response_length(transition: Transition, policy_input: Any) -> int:
    """Generated tokens of one completion, for the response-length diagnostics.

    The producer's ``completion_token_count`` stamp (every rollout writes one) is authoritative,
    because the shaped loss mask can exclude a reward-stage-filtered prefix. A transition without
    the stamp (hand-built batches) falls back to the shaped completion width plus that filtered
    prefix, which is the generated length whenever the packer shaped the whole completion.
    """
    stamped = transition.policy_info.get("completion_token_count")
    if isinstance(stamped, int) and not isinstance(stamped, bool) and stamped >= 0:
        return stamped
    return int(sum(policy_input.logprob_masks)) + int(transition.algo_extra.get("n_ignore_prefix_tokens", 0))


def _response_metrics(truncated: int, lengths: list[int]) -> dict[str, int | float]:
    """Rank-local response diagnostics for one chunk (see ``_GRPO_PUBLIC_METRICS``).

    Ratios are over the chunk's completion count so the adapter's count-weighted mean aggregates
    them as Σ/Σ across ranks and chunks; an empty chunk reports zeros with a zero weight, which is
    what the weighted reduction requires of it.
    """
    count = len(lengths)
    return {
        "grpo_response_count": count,
        "grpo_truncated_frac": truncated / count if count else 0.0,
        "grpo_response_len_mean": sum(lengths) / count if count else 0.0,
        "grpo_response_len_max": max(lengths) if lengths else 0,
    }


def _score_centering_metrics(
    terms: ScoreCenteringTerms | None,
    token_count: int,
    out_of_head: int,
) -> dict[str, float | int]:
    """Rank-local score-centering diagnostics, per-token MEANS over `token_count` plus the count.

    Emitted with the SAME key set whether centering is on or off (metrics are packed into one
    per-rank vector and reduced, so a config-dependent key would be a mismatched all-reduce); off,
    every value is 0.0 with `grpo_sc_tokens == 0`, so the token-weighted merge ignores the shard.
    """
    if terms is None or token_count <= 0:
        return {
            "grpo_sc_head_mass_q": 0.0,
            "grpo_sc_head_mass_p": 0.0,
            "grpo_sc_rho_mean": 0.0,
            "grpo_sc_rho_max": 0.0,
            "grpo_sc_kl_head": 0.0,
            "grpo_sc_correction_abs": 0.0,
            "grpo_sc_token_weight": 0.0,
            "grpo_sc_out_of_head_frac": 0.0,
            "grpo_sc_tokens": 0,
        }
    stats = terms.stats
    return {
        "grpo_sc_head_mass_q": stats["sc_head_mass_q_sum"] / token_count,
        "grpo_sc_head_mass_p": stats["sc_head_mass_p_sum"] / token_count,
        "grpo_sc_rho_mean": stats["sc_rho_sum"] / token_count,
        "grpo_sc_rho_max": stats["sc_rho_max"],
        "grpo_sc_kl_head": stats["sc_kl_head_sum"] / token_count,
        "grpo_sc_correction_abs": stats["sc_correction_abs_sum"] / token_count,
        "grpo_sc_token_weight": stats["sc_token_weight_sum"] / token_count,
        "grpo_sc_out_of_head_frac": out_of_head / token_count,
        "grpo_sc_tokens": int(token_count),
    }


def make_clipped_objective(
    packer: DataPacker,
    algo_cfg: Any,
    *,
    max_response_length: int | None = None,
) -> Objective:
    """Build clipped GRPO over the shared policy-gradient surrogate.

    Args:
        packer: Tokenization and batched collation seam.
        algo_cfg: Clip, KL, temperature, behavior-logprob, reduction, and overlong-filtering settings.
        max_response_length: Fixed rollout response bound for Dr-GRPO's reduction mode.

    Returns:
        The configured trainer-facing objective, including its DP-mean normalization weight.
    """

    loss_type = str(getattr(algo_cfg, "loss_type", "token-mean"))
    supported_loss_types = (*POLICY_GRADIENT_LOSS_TYPES, FIXED_RESPONSE_POLICY_GRADIENT_LOSS_TYPE)
    if loss_type not in supported_loss_types:
        raise ValueError(f"loss_type must be one of {supported_loss_types}, got {loss_type!r}")
    # Ratio granularity: "token" (GRPO) or "sequence" (GSPO). GRPOConfig validates this too; the
    # check here is the backstop for hand-built cfgs, which are how objective tests reach this seam.
    level = str(getattr(algo_cfg, "importance_sampling_level", "token"))
    if level not in IMPORTANCE_SAMPLING_LEVELS:
        raise ValueError(f"importance_sampling_level must be one of {IMPORTANCE_SAMPLING_LEVELS}, got {level!r}")
    lower_bound_ratio = float(getattr(algo_cfg, "lower_bound_ratio", 3.0))
    if level == "sequence" and lower_bound_ratio != 0.0:
        raise ValueError(
            "importance_sampling_level='sequence' applies no dual clip to the response ratio (GSPO defines "
            f"none), so lower_bound_ratio must be 0, got {lower_bound_ratio!r}"
        )
    if loss_type == FIXED_RESPONSE_POLICY_GRADIENT_LOSS_TYPE:
        if isinstance(max_response_length, bool) or not isinstance(max_response_length, int):
            raise TypeError(
                f"{FIXED_RESPONSE_POLICY_GRADIENT_LOSS_TYPE} requires rollout.max_tokens to be a "
                f"non-bool integer, got {max_response_length!r}"
            )
        if max_response_length < 1:
            raise ValueError(
                f"{FIXED_RESPONSE_POLICY_GRADIENT_LOSS_TYPE} requires rollout.max_tokens > 0, "
                f"got {max_response_length!r}"
            )

    # Attached ONLY when this objective would actually consult it. Under
    # `use_rollout_logprobs_for_loss` the `gather_behavior_logprobs` branch supplies π_old and
    # `old_for` is never reached, so a cache there would be a permanently-empty object that still made
    # `needs_old_logprobs()` answer True. That matters beyond tidiness: the trainer derives "was an
    # ahead pass expected?" from (n_mini > 1, cache is not None), and BOTH must be rank-invariant --
    # config and objective identity are, mutable cache MODE is not. Deriving the expectation from
    # mode is what let a restarted rank skip a collective its peers entered (Codex round 7).
    _uses_rollout_lp = bool(getattr(algo_cfg, "use_rollout_logprobs_for_loss", False))
    old_logp_cache = None if _uses_rollout_lp else OldLogprobCache()
    # The KL anchor's scorer slot exists iff the loss will consult it. `build_trainer` attaches the
    # frozen twin after building the policy (and registers the same model with the trainer for
    # residency); until then (and forever at kl_beta == 0) the objective scores exactly as before. A
    # positive kl_beta with nothing attached is refused at the first call, converged across ranks,
    # rather than by the primitive mid-forward on one rank.
    kl_beta = float(getattr(algo_cfg, "kl_beta", 0.0))
    if kl_beta < 0.0:
        raise ValueError(f"kl_beta must be >= 0, got {kl_beta!r}")
    reference = FrozenReference() if kl_beta > 0.0 else None
    # Bound on the KL term's per-token gradient coefficient (`rl.kl_coeff_clamp`; None = the exact
    # gradient). GRPOConfig validates the range; this is the backstop for hand-built cfgs.
    kl_coeff_clamp = getattr(algo_cfg, "kl_coeff_clamp", None)
    if kl_coeff_clamp is not None:
        if isinstance(kl_coeff_clamp, bool) or not isinstance(kl_coeff_clamp, Real):
            raise TypeError(f"kl_coeff_clamp must be None or a positive number, got {kl_coeff_clamp!r}")
        kl_coeff_clamp = float(kl_coeff_clamp)
        if not math.isfinite(kl_coeff_clamp) or kl_coeff_clamp <= 0.0:
            raise ValueError(f"kl_coeff_clamp must be None or a positive finite number, got {kl_coeff_clamp!r}")
    # DAPO overlong filtering (`rl.mask_truncated_completions`): a completion the engine cut at its
    # token budget (`policy_info["finish_reason"] == "length"`, the vLLM rollout's stamp) leaves the
    # loss the way an all-zero loss mask would -- absent from the surrogate, the KL term and their
    # token/sequence divisors alike. An unstamped batch (HF/local rollouts, older data) masks nothing.
    mask_truncated = bool(getattr(algo_cfg, "mask_truncated_completions", False))
    # Score centering (`rl.score_centering`): None when off, so every branch below that consults it
    # is skipped and the loss is byte-identical to before the block existed. GRPOConfig refuses the
    # combinations the objective cannot compose; these are the backstops for hand-built cfgs.
    score_centering = resolve_score_centering(getattr(algo_cfg, "score_centering", None))
    if score_centering is not None:
        if _uses_rollout_lp:
            raise ValueError("score_centering.enabled requires use_rollout_logprobs_for_loss=False")
        if level != "token":
            raise ValueError("score_centering.enabled requires importance_sampling_level='token'")
        if bool(getattr(algo_cfg, "needs_cached_old_logprobs", False)):
            raise ValueError("score_centering.enabled requires n_mini == 1 (no pi_old ahead pass)")

    def objective(model: Any, trajectories: list[Trajectory]) -> tuple[torch.Tensor, dict]:
        net = getattr(model, "net", model)  # raw module IS its own net; HFModel owns it under .net
        device = next(net.parameters()).device
        # Empty local shards are legal, but an all-rank empty batch is not.
        if rank_max_count(len(trajectories), device) == 0:
            raise ValueError("clipped GRPO objective got an empty batch — no trajectories to learn from.")
        require_advantages(trajectories)

        # Shape everything before the forward and converge rank-local data errors first.
        def prepare() -> tuple[
            list[Any],
            list[float],
            list[torch.Tensor] | None,
            list[Any],
            Any | None,
            dict[str, int | float],
            tuple[list[Any], list[Any]],
        ]:
            processed: list[Any] = []
            raw_advantages: list[Any] = []
            widths: list[int] = []
            behavior_logprobs: list[Any] = []
            head_ids_records: list[Any] = []
            head_logprob_records: list[Any] = []
            probe_input: Any | None = None
            truncated_count = 0
            response_lengths: list[int] = []
            for traj in trajectories:
                tr = traj.transitions[0]
                policy_input = shape_policy_input(
                    tr,
                    packer,
                    int(tr.algo_extra.get("n_ignore_prefix_tokens", 0)),
                )
                if probe_input is None:
                    # Keep the complete shaped object: a VLM probe needs its attention mask and
                    # media tensors as well as the image-placeholder token ids.
                    probe_input = policy_input
                # Diagnostics cover every completion handed in, before either drop below.
                truncated = is_length_truncated(tr.policy_info)
                truncated_count += int(truncated)
                response_lengths.append(_response_length(tr, policy_input))
                if not any(policy_input.logprob_masks):
                    continue  # empty completion -> no learnable signal
                if mask_truncated and truncated:
                    # Overlong filtering takes the same exit as an all-zero loss mask: the completion
                    # is absent from the surrogate, the KL term and the token/sequence divisors, and
                    # the pi_old ahead pass skips it identically (the stamp is batch data).
                    continue
                processed.append(policy_input)
                raw_advantages.append(tr.advantage)
                widths.append(sum(policy_input.logprob_masks[:-1]))
                behavior_logprobs.append(tr.policy_info.get("rollout_logprobs"))
                # The sampler's top-k head per generated token (score centering); None when unstamped.
                head_ids_records.append(tr.policy_info.get(ROLLOUT_TOPK_IDS_KEY))
                head_logprob_records.append(tr.policy_info.get(ROLLOUT_TOPK_LOGPROBS_KEY))
            advantage_tokens: list[torch.Tensor] | None = None
            scalar_advantages: list[float] = []
            if any(isinstance(advantage, torch.Tensor) for advantage in raw_advantages):
                if level == "sequence":
                    # One ratio per response only means one advantage per response. Per-token
                    # credit under a shared ratio is GSPO-token, which nothing here produces or
                    # qualifies; refuse inside the converged block so every rank fails together.
                    raise ValueError(
                        "importance_sampling_level='sequence' requires one scalar advantage per response; "
                        "per-token advantages are not supported by the sequence-level ratio"
                    )
                advantage_tokens = [
                    advantage_to_tokens(advantage, width, device=device)
                    for advantage, width in zip(raw_advantages, widths)
                ]  # each [N_i]
            else:
                scalar_advantages = [float(advantage) for advantage in raw_advantages]
            response_metrics = _response_metrics(truncated_count, response_lengths)
            return (
                processed,
                scalar_advantages,
                advantage_tokens,
                behavior_logprobs,
                probe_input,
                response_metrics,
                (head_ids_records, head_logprob_records),
            )

        (
            processed,
            scalar_advantages,
            advantage_tokens,
            behavior_logprobs,
            probe_input,
            response_metrics,
            head_records,
        ) = run_rank_converged(
            prepare,
            "clipped GRPO objective preparation",
            raise_errors=raise_if_any_rank_failed,
        )

        def collate_policy_inputs() -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
            """Collate locally, but converge every rank before the FSDP forward."""
            if not processed:
                if probe_input is not None:
                    probe_max_len = packer.policy_compute_max_len([probe_input])
                    batch = packer.policy_collate_fn([probe_input], probe_max_len)  # model-ready tensors [1,...]
                    input_ids = batch["input_ids"].to(device)  # [1,L]
                    masks = batch["logprob_masks"].to(device)  # [1,L]
                else:
                    input_ids = torch.zeros((1, 2), dtype=torch.long, device=device)  # [1,2]
                    masks = torch.zeros_like(input_ids, dtype=torch.bool)  # [1,2]
                    batch = {"input_ids": input_ids, "logprob_masks": masks}
            else:
                max_len = packer.policy_compute_max_len(processed)
                batch = packer.policy_collate_fn(  # model-ready tensors [B,...]
                    processed,
                    max_len,
                )
                input_ids = batch["input_ids"].to(device)  # [B,L]
                masks = batch["logprob_masks"].to(device)  # [B,L]
            expected_batch = 1 if not processed else len(processed)
            batch["input_ids"] = input_ids  # [B,L]
            batch["logprob_masks"] = masks  # [B,L]
            input_ids, masks = validate_collated_policy_batch(  # [B,L], [B,L]
                batch,
                expected_batch,
                owner="clipped GRPO",
            )
            return batch, input_ids, masks

        batch, input_ids, masks = run_rank_converged(
            collate_policy_inputs,
            "clipped GRPO policy collation",
            raise_errors=raise_if_any_rank_failed,
        )
        use_rollout_lp = bool(getattr(algo_cfg, "use_rollout_logprobs_for_loss", False))
        _lp_dtype = getattr(torch, str(getattr(algo_cfg, "logprob_dtype", "float32")).removeprefix("torch."))
        temperature = float(algo_cfg.temperature)
        selection = masks[:, :-1]  # [B,L-1] loss-masked completion positions (next-token aligned)
        # Frozen-reference logprobs for the KL anchor, [N] or None. BEFORE the policy forward and
        # under no_grad, so its [B,L,V] logits and fp32 log-softmax are freed before the policy's
        # exist (lower peak than scoring it afterwards with the policy graph alive). Same tokens
        # (`input_ids`), same positions (`selection`), same temperature and dtype as `current`
        # below. Runs on the probe batch of an empty shard too: it is an FSDP forward, so every
        # rank must enter it whenever any rank does (see FrozenReference).
        #
        # NOT run while the π_old cache is recording or probing (n_mini > 1). The ahead pass and the
        # one-shot determinism probe call this objective under no_grad and discard the loss, and the
        # reference is frozen, so the training pass re-scores exactly these tokens against exactly
        # these weights: scoring them here too would cost n_mini*n_micro + 1 discarded reference
        # forwards per collected batch (the GSPO cadence). The mode is set by the trainer driver
        # identically on every rank before each pass, so the skip is rank-uniform and strands no
        # collective. The loss then runs at kl_beta 0 for those passes, which is what π_old needs:
        # the recorded logprobs are the policy's own and the KL term never enters them.
        reference_logprobs: torch.Tensor | None = None
        pass_kl_beta = kl_beta
        if reference is not None:
            raise_if_any_rank_failed(
                None
                if reference.attached
                else ValueError(
                    f"kl_beta={kl_beta} requires a frozen reference policy but none is attached: "
                    "set rl.ref.build_ref=true so build_trainer builds the twin, or set kl_beta=0"
                ),
                "clipped GRPO frozen-reference requirement",
            )
            if old_logp_cache is not None and old_logp_cache.mode in ("record", "probe"):
                pass_kl_beta = 0.0
            else:
                reference_logprobs = reference.score(
                    batch,
                    input_ids,
                    selection,
                    temperature=temperature,
                    dtype=_lp_dtype,
                    owner="clipped GRPO",
                )  # [N]
        prepared = run_rank_converged(
            lambda: prepare_policy_capability(model, batch),
            "clipped GRPO policy capability preflight",
            raise_errors=raise_if_any_rank_failed,
        )
        forward_output = forward_policy(model, net, prepared)
        logits = run_rank_converged(
            lambda: policy_logits(forward_output, input_ids, owner="clipped GRPO policy"),
            "clipped GRPO policy forward validation",
            raise_errors=raise_if_any_rank_failed,
        )  # [B,L,V]
        if not processed:
            # Match the non-empty path's forward and collective order with zero contribution.
            loss = logits.sum() * 0.0  # []
            # Match the non-empty path's one rollout-logprob verdict collective (loss-mode
            # validation or gap-diagnostic gathering, exactly one of the two runs there).
            raise_if_any_rank_failed(None, "clipped GRPO rollout-logprob preparation")
            if score_centering is not None:
                # ... and its score-centering record collective.
                raise_if_any_rank_failed(None, "clipped GRPO score-centering record preparation")
            # Match peers' denominator collective after the probe forward.
            denom = rank_mean_count(0, device)
            # SAME key set as the non-empty return -- metrics are packed into one per-rank vector and
            # reduced, so a missing/extra key here is a mismatched all-reduce. An empty shard clipped
            # nothing, drifted nowhere, and compared no tokens: 0.0 (zero weight) for all of them.
            # The response diagnostics are the exception: they describe the completions handed in,
            # which may all have been empty or (under the mask) all cut at the token budget.
            return loss, {
                "grpo_loss": 0.0,
                "grpo_n": 0,
                "grpo_valid_tokens": 0,
                "grpo_clip_frac": 0.0,
                "grpo_ratio_absdev": 0.0,
                "grpo_seq_clip_frac": 0.0,
                "grpo_seq_log_ratio_abs": 0.0,
                "grpo_logprob_gap_mean": 0.0,
                "grpo_logprob_gap_p99": 0.0,
                "grpo_logprob_gap_frac": 0.0,
                "grpo_logprob_gap_tokens": 0,
                "grpo_kl_mean": 0.0,
                "grpo_kl_beta": kl_beta,
                "grpo_kl_max": 0.0,
                **response_metrics,
                **_score_centering_metrics(None, 0, 0),
                OBJECTIVE_WEIGHT_SUM: denom,
            }
        # Temperature-1 token logprobs for the train/rollout gap diagnostic: the engine records raw
        # (untempered) logprobs, so comparing against the tempered loss-path logprobs would report
        # the sampling temperature, not the train/infer gap. Free when rl.temperature == 1 (the loss
        # path's `current` IS the temperature-1 value); one extra chunked log-softmax otherwise,
        # skipped when no trajectory recorded logprobs (presence is rank-converged by `prepare`).
        # Under score centering the engine records PROCESSED (tempered) logprobs, so the gap pairs
        # the tempered loss-path `current` with them and this temperature-1 pass is not taken.
        gap_current = (
            selected_token_logprobs(logits, input_ids, masks, dtype=_lp_dtype)
            if temperature != 1.0 and score_centering is None and any(value is not None for value in behavior_logprobs)
            else None
        )  # [N] or None
        # LOG_SOFTMAX IN `logprob_dtype` (fp32 by default), NOT the model's compute dtype. cosmos-rl
        # does the same via `train.logprob_dtype`, fp32 INDEPENDENTLY of param_dtype -- so its
        # logprobs are fp32 even on a bf16 model. Computing them in bf16 costs precision exactly where
        # the clipped surrogate needs it: `ratio = exp(cur - old)` is a difference of two logprobs,
        # and bf16 cannot resolve the small deviations that decide whether a token crosses the
        # epsilon boundary. MEASURED at n_mini=8, matched config: bf16 logprobs made the clip engage
        # on 0.00089 of tokens vs 0.01493 in fp32 (17x). Invisible at n_mini==1 (ratio is identically
        # 1, nothing clips). `dtype=` computes in that dtype without materialising an upcast copy of
        # the [B,L,V] logits (~1GB per chunk at Qwen's 151k vocab). The reference above went through
        # the very same helper, so the KL's z = ref − cur compares like with like.
        head_logp: torch.Tensor | None = None  # [N,k] trainer log p at the sampler's head ids
        head_sampler_logp: torch.Tensor | None = None  # [N,k] sampler log q at the same ids (fp32)
        head_ids: torch.Tensor | None = None  # [N,k]
        if score_centering is None:
            current = _completion_token_logprobs(
                logits, input_ids, selection, temperature=temperature, dtype=_lp_dtype
            )  # [N]
        else:

            def prepare_head_records() -> tuple[torch.Tensor, torch.Tensor]:
                if any(value is None for value in behavior_logprobs):
                    raise ValueError(
                        "score_centering.enabled=True but a trajectory carries no policy_info['rollout_logprobs'] "
                        "(the sampled token's own sampler logprob is part of the estimator)"
                    )
                return gather_head_records(
                    head_records[0],
                    head_records[1],
                    selection,
                    top_k=score_centering.top_k,
                    device=device,
                    dtype=_lp_dtype,
                    vocab_size=int(logits.shape[-1]),  # refused HERE, rank-converged, never at the gather
                )  # [N,k], [N,k]

            head_ids, head_sampler_logp = run_rank_converged(
                prepare_head_records,
                "clipped GRPO score-centering record preparation",
                raise_errors=raise_if_any_rank_failed,
            )
            # Same ops as `_completion_token_logprobs` for `current`, plus the head gather off the one
            # log-softmax (no second [B,L,V] tensor).
            current, head_logp = completion_token_and_head_logprobs(
                logits, input_ids, selection, head_ids, temperature=temperature, dtype=_lp_dtype
            )  # [N], [N,k]
        counts = selection.sum(dim=1)  # [B]
        advantage = (
            torch.repeat_interleave(torch.tensor(scalar_advantages, device=device), counts)
            if advantage_tokens is None
            else torch.cat(advantage_tokens)
        )  # [N]
        # π_old: recomputed from the CURRENT policy by default (ratio == exp(0) == 1 on-policy),
        # matching cosmos-rl's `use_rollout_logprobs_for_loss=False`. The engine's own logprobs are
        # opt-in: they are raw temperature-1 logprobs from the INFERENCE forward, so they equal
        # log π_old only under the temperature contract (rollout.temperature == rl.temperature ==
        # 1.0, no top-p/top-k warping; control/temperature.py) and even then carry the train/infer
        # gap measured below into every ratio. See GRPOConfig.use_rollout_logprobs_for_loss.
        if use_rollout_lp:

            def prepare_behavior() -> torch.Tensor:
                if all(value is None for value in behavior_logprobs):
                    raise ValueError(
                        "use_rollout_logprobs_for_loss=True but NO trajectory carries "
                        "policy_info['rollout_logprobs'] — the producer does not record them"
                    )
                behavior = gather_behavior_logprobs(behavior_logprobs, selection, current)  # [N]
                n_bad = int((~torch.isfinite(behavior)).sum())
                if n_bad:
                    raise ValueError(
                        f"clipped GRPO: policy_info['rollout_logprobs'] has {n_bad} non-finite entries; "
                        "trim completions at the stop token or disable rollout logprobs for loss"
                    )
                return behavior

            behavior = run_rank_converged(
                prepare_behavior,
                "clipped GRPO rollout-logprob validation",
                raise_errors=raise_if_any_rank_failed,
            )
            rollout_logprobs: torch.Tensor | None = behavior  # [N]
        else:
            # `current.detach()` unless an ahead pass froze π_old for this optimizer-step window; see
            # OldLogprobCache for why n_mini > 1 makes the detached form wrong.
            behavior = old_logp_cache.old_for(current) if old_logp_cache is not None else current.detach()  # [N]

            def gather_recorded_logprobs() -> torch.Tensor | None:
                """Rollout logprobs for the gap diagnostic when the producer recorded them, else None.

                Converged like the loss-mode validation: a malformed record (mixed presence, short
                sequence) is a producer bug and must fail every rank together, not desync collectives.
                """
                if all(value is None for value in behavior_logprobs):
                    return None
                return gather_behavior_logprobs(behavior_logprobs, selection, current)  # [N]

            rollout_logprobs = run_rank_converged(
                gather_recorded_logprobs,
                "clipped GRPO rollout-logprob preparation",
                raise_errors=raise_if_any_rank_failed,
            )
        gap = (
            NO_LOGPROB_GAP
            if rollout_logprobs is None
            else logprob_gap_stats(current.detach() if gap_current is None else gap_current, rollout_logprobs)
        )
        clip_stats: dict = {}
        centering: ScoreCenteringTerms | None = None
        sc_out_of_head = 0
        if score_centering is not None:
            assert head_logp is not None and head_sampler_logp is not None and head_ids is not None
            assert rollout_logprobs is not None  # validated in prepare_head_records
            centering = score_centering_terms(
                head_logp,
                head_sampler_logp,
                current,
                rollout_logprobs,
                spec=score_centering,
            )
            with torch.no_grad():
                sampled = input_ids[:, 1:][selection]  # [N] the sampled token ids, in `current` order
                sc_out_of_head = int((~(head_ids == sampled[:, None]).any(-1)).sum())
        per_token = clipped_policy_gradient_loss(
            current,
            behavior,
            advantage,
            algo_cfg.epsilon_low,
            algo_cfg.epsilon_high,
            reference=reference_logprobs,
            kl_beta=pass_kl_beta,
            lower_bound_ratio=lower_bound_ratio if lower_bound_ratio > 0 else None,
            stats=clip_stats,
            importance_sampling_level=level,
            token_counts=counts,
            kl_coeff_clamp=kl_coeff_clamp,
            token_weight=None if centering is None else centering.token_weight,
            centering=None if centering is None else centering.correction,
        )  # [N]
        numerator, units = reduce_policy_gradient_loss(
            per_token,
            counts,
            loss_type,
            max_response_length=max_response_length,
        )  # []
        denom = rank_mean_count(units, device)
        loss = numerator * normalization_scale(denom)  # []
        valid_tokens = int(counts.sum())  # RANK-LOCAL count — diagnostics only
        token_divisor = max(valid_tokens, 1)
        sequence_divisor = max(len(processed), 1)
        return loss, {
            "grpo_loss": float(loss.detach()),
            "grpo_n": len(processed),
            "grpo_valid_tokens": valid_tokens,
            # PER-RANK fractions; the adapter token-weights the first pair and sequence-weights the
            # second, so each aggregate is Σcount/Σunits. At n_mini == 1 every ratio is identically
            # 1, so all four are 0.0 by construction. Under a sequence-level ratio `clip_frac` is a
            # length-weighted response clip fraction (a clipped response clips all its tokens).
            "grpo_clip_frac": clip_stats.get("clip_count", 0.0) / token_divisor,
            "grpo_ratio_absdev": clip_stats.get("ratio_absdev_sum", 0.0) / token_divisor,
            "grpo_seq_clip_frac": clip_stats.get("seq_clip_count", 0.0) / sequence_divisor,
            "grpo_seq_log_ratio_abs": clip_stats.get("seq_log_ratio_abs_sum", 0.0) / sequence_divisor,
            # Rank-local gap statistics; the adapter weights them by the compared-token count.
            "grpo_logprob_gap_mean": gap.mean,
            "grpo_logprob_gap_p99": gap.p99,
            "grpo_logprob_gap_frac": gap.gt_threshold_frac,
            "grpo_logprob_gap_tokens": gap.token_count,
            # Rank-local token MEAN of the unscaled K3 penalty (the primitive reports the detached
            # SUM); token-weighted by the adapter. Exactly 0.0 when no reference is wired, and on the
            # ahead-pass/probe calls that skip the reference (their metrics are discarded).
            "grpo_kl_mean": clip_stats.get("kl_sum", 0.0) / token_divisor,
            "grpo_kl_beta": kl_beta,
            # Rank-local MAX of the same penalty; the adapter declares it max-reduced.
            "grpo_kl_max": clip_stats.get("kl_max", 0.0),
            # Over every completion handed in, not only `processed`; count-weighted by the adapter.
            **response_metrics,
            # Score centering diagnostics over the tokens the correction covered (zeros, weight 0, when off).
            **_score_centering_metrics(centering, valid_tokens, sc_out_of_head),
            OBJECTIVE_WEIGHT_SUM: denom,
        }

    # Every mode is a sum divided by a DP-global valid-unit count, so micro-batches must defer
    # normalization until the optimizer boundary.
    return Objective(
        objective,
        reduction=Reduction.TOKEN_SUM,
        metric_adapter=_GRPO_METRIC_ADAPTER,
        old_logp_cache=old_logp_cache,
        reference_scorer=reference,
    )
