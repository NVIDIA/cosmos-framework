# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""On-policy-distillation objectives (OPD design 2026-08-18 §§6–8).

This module is the trainer-facing half of OPD: it shapes trajectories through the `DataPacker`,
runs the student forward, pairs each scored position with the teacher's `TeacherSignal`, and reduces
to a loss the way posttrain's GRPO objective does. Direct OPD defaults to the normalized
top-k-plus-residual-other partition; the original raw-slice divergence remains for explicit legacy
same-token advantage runs.

TWO OBJECTIVES, ONE TEACHER PATH:

- `make_direct_objective` — GKD lineage. The per-token divergence, minimized directly. Never reads
  `Transition.advantage`.
- `make_advantage_objective` — MiniLLM lineage. A same-token divergence can remain the legacy
  subtraction penalty, while cross-token DPCA supplies detached credit through a separate additive
  `TeacherCreditSource`.

Direct divergence and detached teacher credit stay separate so DPCA cannot inherit the legacy
subtraction sign by accident.

TWO DELIBERATE DISTRIBUTED CONTRACTS:

- **No single-rank restriction.** Every mode divides by a DP-global denominator via
  `rank_mean_count`, exactly as `algorithm/grpo.py::make_clipped_objective` does. `select_chunk` may
  therefore split a batch across the complete multi-rank `dp_replicate × dp_shard` mesh.
- **All reductions are DP-correct.** Token- and sequence-mean modes use their declared global unit
  count rather than a mean of rank-local means, including uneven shards.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Real
from typing import Any, Protocol

import torch

from cosmos_framework.posttrain.algorithm.annotators.advantage import advantage_to_tokens, require_advantages
from cosmos_framework.posttrain.algorithm.dpca import TeacherCreditSource
from cosmos_framework.posttrain.algorithm.grpo import OldLogprobCache
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
    POLICY_GRADIENT_LOSS_TYPES,
    clipped_policy_gradient_loss,
    reduce_policy_gradient_loss,
)
from cosmos_framework.posttrain.data.packer.base import DataPacker
from cosmos_framework.posttrain.data.packer.decoder_only_llm import shape_policy_input
from cosmos_framework.posttrain.schema import TeacherSignal, Trajectory
from cosmos_framework.posttrain.schema.trajectory import _tensor_shape
from cosmos_framework.posttrain.utils.dist import (
    raise_if_any_rank_failed,
    rank_max_count,
    rank_mean_count,
    run_rank_converged,
)

_LOSS_TYPES = POLICY_GRADIENT_LOSS_TYPES
_OPD_PUBLIC_METRICS: dict[str, str] = {
    "opd_loss": "loss_avg",
    "opd_n": "valid_sequence_count",
    "opd_valid_tokens": "valid_token_count",
    "opd_divergence": "opd_divergence",
    "opd_advantage": "opd_advantage",
    "opd_teacher_credit": "opd_teacher_credit",
    OBJECTIVE_WEIGHT_SUM: "weight_sum",
}
_OPD_DIRECT_METRIC_ADAPTER = MetricAdapter(
    names={
        "opd_loss": "loss_avg",
        "opd_n": "valid_sequence_count",
        "opd_valid_tokens": "valid_token_count",
        "opd_divergence": "opd_divergence",
        OBJECTIVE_WEIGHT_SUM: "weight_sum",
    },
    internal_prefix="opd_",
    phases=frozenset({"train"}),
    weighted_by={
        "opd_loss": OBJECTIVE_WEIGHT_SUM,
        "opd_divergence": "opd_valid_tokens",
    },
)
_OPD_ADVANTAGE_METRIC_ADAPTER = MetricAdapter(
    names=_OPD_PUBLIC_METRICS,
    internal_prefix="opd_",
    phases=frozenset({"train"}),
    weighted_by={
        "opd_loss": OBJECTIVE_WEIGHT_SUM,
        "opd_divergence": "opd_valid_tokens",
        "opd_advantage": "opd_valid_tokens",
        "opd_teacher_credit": "opd_valid_tokens",
    },
)


class DivergenceSource(Protocol):
    """Differentiable per-token teacher/student divergence for same-token OPD.

    ``TopKPlusOtherDivergence`` is the direct-OPD default. ``TopKSliceDivergence`` preserves the
    raw-slice arithmetic for explicit legacy advantage runs.

    DPCA deliberately does not implement this protocol: it consumes detached realized-token
    likelihoods and returns additive policy credit through ``TeacherCreditSource``.
    """

    def per_token(
        self,
        student_logits: "torch.Tensor",
        signals: list[TeacherSignal],
        device: "torch.device",
    ) -> "torch.Tensor": ...


def _support_logprobs(student_logits: "torch.Tensor", support_ids: "torch.Tensor") -> "torch.Tensor":
    """The student's FULL-vocabulary logprobs at `support_ids`, as `logit - logsumexp(logits)`.

    Identical to `log_softmax(logits).gather(-1, ids)` and the same quantity cosmos-rl computes
    (`utils/util.py::compute_logprobs_for_top_k_indices`), but it keeps the largest intermediate to
    one `[n_tok, 1]` column instead of a second full-vocabulary tensor.

    fp32 throughout: a bf16 reduction over a 150k-wide vocabulary loses enough precision that the
    top-k mass can round above one, which any partition-based formulation would then have to reject.
    The cast is idempotent, so a caller that already holds the fp32 copy should pass THAT — see
    `make_advantage_objective`, which needs the same tensor for the sampled-token logprob and would
    otherwise keep two.

    MEMORY. That cast is the objective's peak allocation: `[n_tok, V]` fp32, and autograd holds it
    for the logsumexp backward, so chunking the loop would not free it. Budget
    `scored_tokens x vocab x 4 bytes` per microbatch — measured, ~9.3 GiB at 16k scored tokens on a
    151936 vocabulary, which OOMs a 184 GiB GB200 that is already holding two vLLM engines. Control
    it with `rl.micro_batch_per_replica`, which splits ONE drawn batch across forwards (the
    reference's `mini_batch` does the same job), not by shrinking the batch itself.
    """
    logits = student_logits.float()
    return logits.gather(-1, support_ids) - torch.logsumexp(logits, dim=-1, keepdim=True)


def _concat_support(
    signals: list[TeacherSignal], device: torch.device, vocab_size: int | None = None
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """`(support_ids [n_tok, k], teacher_logprobs [n_tok, k])` concatenated in signal order.

    `vocab_size` is the STUDENT's logit width, and checking against it is not paranoia: a teacher
    whose `lm_head` is wider than the student's can return an id the student has no column for.
    Qwen2.5 is exactly that case — the shipped recipe pairs a 151936-wide student with a 152064-wide
    teacher, identical tokenizers (both stop at id 151664) but 128 extra padding rows on the teacher.
    Those rows are untokenizable, so a healthy teacher should never rank one in its top-k, but
    nothing enforces it. Unchecked, the id reaches `gather` below and trips a CUDA device-side
    assert, which is asynchronous, poisons the context, and leaves the peer ranks blocked in a
    collective until the NCCL watchdog fires. A named error here costs one reduction per microbatch.

    cosmos-rl has the same unguarded gather (`utils/util.py::compute_logprobs_for_top_k_indices`)
    and ships the same model pair, so this is a strict improvement over the reference with no
    numerical divergence: on in-range data the tensors are identical.
    """
    widths: set[int] = set()
    validated_ids: list[torch.Tensor] = []
    validated_logprobs: list[torch.Tensor] = []
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    for signal_index, signal in enumerate(signals):
        ids = signal.topk_ids
        logprobs = signal.topk_logprobs
        if not isinstance(ids, torch.Tensor) or not isinstance(logprobs, torch.Tensor):
            raise ValueError(
                f"TeacherSignal {signal_index} top-k payload is not materialized as tensors; "
                "the trainer must materialize teacher bulk before the objective."
            )
        if ids.ndim != 2 or logprobs.ndim != 2 or tuple(ids.shape) != tuple(logprobs.shape):
            raise ValueError(
                f"TeacherSignal {signal_index} top-k payload must be identically shaped [T,K], got "
                f"{tuple(ids.shape)} and {tuple(logprobs.shape)}"
            )
        if ids.shape[1] < 1:
            raise ValueError(f"TeacherSignal {signal_index} top-k support must have K>=1")
        if ids.dtype not in integer_dtypes:
            raise ValueError(f"TeacherSignal {signal_index} topk_ids must have integer dtype, got {ids.dtype}")
        if not torch.is_floating_point(logprobs):
            raise ValueError(
                f"TeacherSignal {signal_index} topk_logprobs must have floating dtype, got {logprobs.dtype}"
            )
        finite = torch.isfinite(logprobs)  # [T,K]
        if not bool(finite.all()):
            raise ValueError(f"TeacherSignal {signal_index} topk_logprobs contains non-finite values")
        sorted_ids = torch.sort(ids.to(dtype=torch.long), dim=-1).values  # [T,K]
        duplicate = sorted_ids[:, 1:] == sorted_ids[:, :-1]  # [T,K-1]
        if bool(duplicate.any()):
            duplicate_rows = duplicate.any(dim=-1).nonzero(as_tuple=True)[0].tolist()  # [R]
            raise ValueError(
                f"TeacherSignal {signal_index} has duplicate top-k support IDs in row(s) {duplicate_rows[:8]}"
            )
        widths.add(int(ids.shape[1]))
        validated_ids.append(ids.to(device=device, dtype=torch.long))  # [T,K]
        validated_logprobs.append(logprobs.to(device=device, dtype=torch.float32))  # [T,K]
    if len(widths) != 1:
        raise ValueError(f"TeacherSignal top-k widths must agree within a batch, got {sorted(widths)}")
    support_ids = torch.cat(validated_ids)  # [N,K]
    if support_ids.numel() and int(support_ids.min()) < 0:
        lowest = int(support_ids.min())
        raise ValueError(f"teacher support id {lowest} is negative")
    if vocab_size is not None and support_ids.numel():
        highest = int(support_ids.max())
        if highest >= vocab_size:
            raise ValueError(
                f"teacher support id {highest} is outside the student's vocabulary of {vocab_size}. "
                "The teacher's lm_head is wider than the student's, so it can return padding-row "
                "ids the student has no column for. Pair models whose id spaces agree, or the "
                "gather below trips a device-side assert instead of raising here."
            )
    teacher = torch.cat(validated_logprobs)  # [N,K]
    return support_ids, teacher


def _append_residual_logprob(
    support_logprobs: torch.Tensor,
    *,
    label: str,
    mass_tolerance: float,
) -> torch.Tensor:  # support_logprobs: [N,K], returns [N,K+1]
    """Append the categorical residual ``log(1 - sum(exp(support)))`` in fp32.

    ``support_logprobs`` are full-vocabulary logprobs, not a top-k renormalization. A tiny clamp is
    used only at the exact zero-mass endpoint so the categorical divergence has finite log inputs;
    realistic ``K < vocab_size`` rows retain their differentiable residual exactly.
    """
    if support_logprobs.ndim != 2:
        raise ValueError(f"{label} support logprobs must be [N,K], got {tuple(support_logprobs.shape)}")
    if not math.isfinite(mass_tolerance) or mass_tolerance < 0.0:
        raise ValueError(f"mass_tolerance must be finite and >= 0, got {mass_tolerance!r}")
    logprobs = support_logprobs.float()  # [N,K]
    finite = torch.isfinite(logprobs)  # [N,K]
    if not bool(finite.all()):
        raise ValueError(f"{label} support logprobs contains non-finite values")
    support_mass = logprobs.exp().sum(dim=-1)  # [N]
    excess = support_mass > 1.0 + mass_tolerance  # [N]
    if bool(excess.any()):
        rows = excess.nonzero(as_tuple=True)[0].tolist()  # [R]
        largest = float(support_mass.max())
        raise ValueError(
            f"{label} top-k support mass exceeds one beyond tolerance {mass_tolerance}: "
            f"max={largest}, row(s)={rows[:8]}"
        )
    residual = (1.0 - support_mass).clamp_min(0.0)  # [N]
    residual_logprob = residual.clamp_min(torch.finfo(logprobs.dtype).tiny).log()  # [N]
    return torch.cat([logprobs, residual_logprob.unsqueeze(-1)], dim=-1)  # [N,K+1]


def topk_slice_divergence(
    student_logprobs: "torch.Tensor",
    teacher_logprobs: "torch.Tensor",
    *,
    beta: float,
) -> "torch.Tensor":
    """Generalized JSD over unnormalized `[n_tok, k]` top-k slices.

    The interior mixture crosses its KL weights so it joins the explicit endpoint formulas
    continuously. Because the slices omit residual vocabulary mass, this legacy quantity is not
    guaranteed non-negative; new direct OPD uses the normalized `(k+1)` partition instead.

    Args:
        student_logprobs: `[n_tok, k]` full-vocabulary logprobs at the teacher's support.
        teacher_logprobs: `[n_tok, k]` the teacher's own logprobs for the same ids.
        beta: 0 is `KL(teacher || student)`, 1 is `KL(student || teacher)`, interior interpolates.

    Returns:
        `[n_tok]` divergence, summed over the k axis. NOT guaranteed non-negative at EITHER
        endpoint, nor in the interior near one.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"jsd_beta must be in [0, 1], got {beta}")
    if student_logprobs.shape != teacher_logprobs.shape:
        raise ValueError(
            f"student and teacher slices must match, got {tuple(student_logprobs.shape)} "
            f"and {tuple(teacher_logprobs.shape)}"
        )
    # `F.kl_div(input, target, log_target=True)` is `target.exp() * (target - input)`, i.e.
    # KL(target || input) — which is why the reference's argument order looks reversed against the
    # mathematical convention. Reproduced exactly, order included.
    functional = torch.nn.functional
    if beta == 0.0:
        divergence = functional.kl_div(student_logprobs, teacher_logprobs, reduction="none", log_target=True)
    elif beta == 1.0:
        divergence = functional.kl_div(teacher_logprobs, student_logprobs, reduction="none", log_target=True)
    else:
        beta_t = torch.tensor(beta, dtype=student_logprobs.dtype, device=student_logprobs.device)
        mixture = torch.logsumexp(
            torch.stack([student_logprobs + torch.log(1 - beta_t), teacher_logprobs + torch.log(beta_t)]),
            dim=0,
        )
        kl_teacher = functional.kl_div(mixture, teacher_logprobs, reduction="none", log_target=True)
        kl_student = functional.kl_div(mixture, student_logprobs, reduction="none", log_target=True)
        # Weights crossed against the mixture — see the docstring. The reference has
        # `beta_t * kl_teacher + (1 - beta_t) * kl_student`, which collapses to zero at both ends.
        divergence = (1 - beta_t) * kl_teacher + beta_t * kl_student
    return divergence.sum(-1)


def topk_plus_other_divergence(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    *,
    beta: float,
    mass_tolerance: float = 1.0e-5,
) -> torch.Tensor:  # student_logprobs/teacher_logprobs: [N,K], returns [N]
    """Generalized JSD over the teacher top-k support plus one residual-other category.

    Unlike ``topk_slice_divergence``, both inputs become normalized ``K+1``-way categorical
    partitions. This restores non-negativity and makes a matched full distribution an exact fixed
    point while keeping the student's residual differentiable through all non-support logits.
    """
    if student_logprobs.shape != teacher_logprobs.shape:
        raise ValueError(
            f"student and teacher slices must match, got {tuple(student_logprobs.shape)} "
            f"and {tuple(teacher_logprobs.shape)}"
        )
    student = _append_residual_logprob(  # [N,K+1]
        student_logprobs,
        label="student",
        mass_tolerance=mass_tolerance,
    )
    teacher = _append_residual_logprob(  # [N,K+1]
        teacher_logprobs,
        label="teacher",
        mass_tolerance=mass_tolerance,
    )
    return topk_slice_divergence(student, teacher, beta=beta)  # [N]


class TopKPlusOtherDivergence:
    """Same-token direct OPD over the top-k support and residual vocabulary mass."""

    beta: float
    mass_tolerance: float

    def __init__(self, beta: float, *, mass_tolerance: float = 1.0e-5) -> None:
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"jsd_beta must be in [0, 1], got {beta}")
        if not math.isfinite(mass_tolerance) or mass_tolerance < 0.0:
            raise ValueError(f"mass_tolerance must be finite and >= 0, got {mass_tolerance!r}")
        self.beta = float(beta)
        self.mass_tolerance = float(mass_tolerance)

    def per_token(
        self,
        student_logits: torch.Tensor,
        signals: list[TeacherSignal],
        device: torch.device,
    ) -> torch.Tensor:  # student_logits: [N,V], returns [N]
        support_ids, teacher = _concat_support(  # [N,K], [N,K]
            signals,
            device,
            vocab_size=student_logits.shape[-1],
        )
        student = _support_logprobs(student_logits, support_ids)  # [N,K]
        return topk_plus_other_divergence(  # [N]
            student,
            teacher,
            beta=self.beta,
            mass_tolerance=self.mass_tolerance,
        )


class TopKSliceDivergence:
    """Legacy plain top-k-slice JSD, with no residual "other" bucket or deduplication.

    `student_logits` is `[n_tok, V]` — the FULL-vocabulary logits at exactly the scored positions,
    row-major over sequences then positions, which is the order the signals concatenate in. The
    student slice comes from a full-vocabulary `log_softmax` gathered at the teacher support.
    """

    def __init__(self, beta: float) -> None:
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"jsd_beta must be in [0, 1], got {beta}")
        self.beta = float(beta)

    def per_token(
        self,
        student_logits: "torch.Tensor",
        signals: list[TeacherSignal],
        device: "torch.device",
    ) -> "torch.Tensor":
        support_ids, teacher = _concat_support(signals, device, vocab_size=student_logits.shape[-1])
        student = _support_logprobs(student_logits, support_ids)
        return topk_slice_divergence(student, teacher, beta=self.beta)


def _scored_signal(transition: Any, n_scored: int, *, signal_kind: str = "topk") -> TeacherSignal:
    """The transition's `TeacherSignal`, checked against the positions this objective will score.

    ONE ROW PER SCORED TOKEN. `topk_ids[j]` is the teacher's support for the j-th position the loss
    is taken over, in the same order — so the objective does no shifting: the teacher role scored
    the completion and knows which positions it scored, and doing the next-token shift there once
    beats re-deriving it here from a mask the teacher never saw.

    Strict about the count on purpose. `n_ignore_prefix_tokens` (written at reward time by
    `annotators/prefix_filter.py`) shortens the scored span, and a signal built for the full
    completion would then pair position j with position j + n_ignore's teacher distribution — a
    silent misalignment that trains confidently on the wrong distribution. There is no safe way to
    guess which end to trim, so a mismatch raises.

    Raises:
        ValueError: `targets` is not a `TeacherSignal`, carries no top-k pair, or its row count
            disagrees with the scored-token count.
    """
    signal = transition.targets
    if not isinstance(signal, TeacherSignal):
        raise ValueError(
            "OPD needs a TeacherSignal in `transition.targets`, got "
            f"{type(signal).__name__} — the teacher role writes it before the trajectory is "
            "dispatched to the trainer."
        )
    if signal_kind == "topk":
        if signal.topk_ids is None:
            raise ValueError(
                "OPD's shared-vocabulary path needs `TeacherSignal.topk_ids`/`topk_logprobs`; this "
                "signal carries the cross-tokenizer DPCA chunk payload instead."
            )
        shape = _tensor_shape(signal.topk_ids)
        if shape is None:
            raise ValueError(
                "OPD's TeacherSignal reached the objective un-materialized — `topk_ids` is still a "
                "transfer handle rather than a tensor. `materialize_payload` runs before the trainer "
                "sees a batch; a stage that forwards the packed form skips it."
            )
        rows = int(shape[0])
        if rows != n_scored:
            raise ValueError(
                f"TeacherSignal has {rows} row(s) but the objective scores {n_scored} token(s). The "
                "signal carries one row per SCORED position, in order; a completion whose "
                "`n_ignore_prefix_tokens` trimmed its span needs a signal trimmed the same way."
            )
    elif signal_kind == "chunk":
        if signal.chunk_bounds is None:
            raise ValueError(
                "DPCA needs `TeacherSignal.chunk_bounds`/`chunk_teacher_logprob`; this signal carries "
                "the same-token top-k payload instead."
            )
        if _tensor_shape(signal.chunk_bounds) is None:
            raise ValueError(
                "DPCA TeacherSignal reached the objective un-materialized — `chunk_bounds` is still "
                "a transfer handle rather than a tensor."
            )
    else:
        raise ValueError(f"signal_kind must be topk|chunk, got {signal_kind!r}")
    return signal


def _prepare(
    trajectories: list[Trajectory],
    packer: DataPacker,
    device: torch.device,
    *,
    label: str,
    with_advantage: bool,
    signal_kind: str = "topk",
) -> tuple[
    list[Any],
    list[Trajectory],
    list[TeacherSignal],
    list[int],
    list[torch.Tensor],
    Any | None,
]:
    """Shape a batch, validate its signals, and expand its advantages — CONVERGED across ranks.

    Everything that can raise on rank-local data lives here, above the caller's first forward.
    Under per-sample sharding a malformed trajectory lands on ONE rank, and a raise between
    forwards aborts that rank while its peers block in the FSDP all-gathers of a forward it will
    never issue: the run hangs instead of reporting the error. `raise_if_any_rank_failed` makes
    every rank raise together.

    Args:
        trajectories: this rank's shard.
        packer: the shaping seam.
        device: the net's device (advantage vectors are built on it).
        label: what to name this objective in a converged failure.
        with_advantage: also read and expand `Transition.advantage` per sequence.

    Returns:
        `(processed, processed_trajectories, signals, widths, advantage_tokens, probe_input)`.
        `widths[i]` is sequence i's scored-token count; `probe_input` preserves the complete
        model-ready sample for an empty-completion forward, including VLM media fields.
    """

    def prepare() -> tuple[
        list[Any],
        list[Trajectory],
        list[TeacherSignal],
        list[int],
        list[torch.Tensor],
        Any | None,
    ]:
        processed: list[Any] = []
        processed_trajectories: list[Trajectory] = []
        signals: list[TeacherSignal] = []
        widths: list[int] = []
        advantages: list[torch.Tensor] = []
        probe_input: Any | None = None
        for traj in trajectories:
            tr = traj.transitions[0]
            n_ignore = int(tr.algo_extra.get("n_ignore_prefix_tokens", 0))
            policy_input = shape_policy_input(tr, packer, n_ignore)
            if probe_input is None:
                probe_input = policy_input
            if not any(policy_input.logprob_masks):
                continue  # empty completion -> nothing to distil against
            # `sum(mask[:-1])` is this sequence's scored-token count and equals its entry in the
            # collated `counts`: the packer's mask ends in 0 and pads with 0.
            n_scored = sum(policy_input.logprob_masks[:-1])
            signals.append(_scored_signal(tr, n_scored, signal_kind=signal_kind))
            if with_advantage:
                advantage = advantage_to_tokens(tr.advantage, n_scored, device=device)  # [N_i]
                advantages.append(advantage)
            processed.append(policy_input)
            processed_trajectories.append(traj)
            widths.append(n_scored)
        return processed, processed_trajectories, signals, widths, advantages, probe_input

    return run_rank_converged(prepare, label, raise_errors=raise_if_any_rank_failed)


def _prepare_forward_inputs(
    packer: DataPacker,
    processed: list[Any],
    probe_input: Any | None,
    device: torch.device,
    *,
    context: str,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor | None]:  # returns batch, input_ids [B,L], masks [B,L] or None
    """Converge rank-local collation and shape checks before any rank enters the FSDP forward."""

    def prepare_local() -> tuple[dict[str, Any], torch.Tensor, torch.Tensor | None]:
        if not processed:
            if probe_input is not None:
                probe_max_len = packer.policy_compute_max_len([probe_input])
                batch = packer.policy_collate_fn([probe_input], probe_max_len)  # model-ready tensors [1,...]
                input_ids = batch["input_ids"].to(device)  # [1,L]
                batch["input_ids"] = input_ids
                probe_mask = batch.get("logprob_masks")
                if probe_mask is not None:
                    batch["logprob_masks"] = probe_mask.to(device)  # [1,L]
                input_ids, _probe_masks = validate_collated_policy_batch(  # [1,L], [1,L]
                    batch,
                    1,
                    owner="OPD",
                )
            else:
                # Token id 0 is valid in every vocabulary; finalization zeroes the output.
                input_ids = torch.zeros((1, 2), dtype=torch.long, device=device)  # [1,2]
                batch = {"input_ids": input_ids}
            return batch, input_ids, None

        max_len = packer.policy_compute_max_len(processed)
        batch = packer.policy_collate_fn(processed, max_len)  # {input_ids[B,L], logprob_masks[B,L]}
        batch["input_ids"] = batch["input_ids"].to(device)  # [B,L]
        batch["logprob_masks"] = batch["logprob_masks"].to(device)  # [B,L]
        input_ids, masks = validate_collated_policy_batch(  # [B,L], [B,L]
            batch,
            len(processed),
            owner="OPD",
        )
        return batch, input_ids, masks

    return run_rank_converged(prepare_local, context, raise_errors=raise_if_any_rank_failed)


def _converged_finalize(
    finalize_local: Callable[[], tuple[torch.Tensor, int, dict[str, float]]],
    device: torch.device,
    *,
    context: str,
) -> tuple[torch.Tensor, float, dict[str, float]]:  # returns loss: [], denominator, diagnostics
    """Converge all post-forward local work before entering the denominator collective.

    Selection, ``DivergenceSource.per_token``, and ragged reduction all inspect rank-local shapes and
    can raise. The callback contains every such operation, including the empty shard's zeroed loss,
    so peers either fail together at the verdict or all proceed to ``rank_mean_count``.
    """
    numerator, units, diagnostics = run_rank_converged(
        finalize_local,
        context,
        raise_errors=raise_if_any_rank_failed,
    )
    denominator = rank_mean_count(units, device)
    loss = numerator * normalization_scale(denominator)  # []
    return loss, denominator, diagnostics


def _scored_positions(
    forward_logits: torch.Tensor,
    input_ids: torch.Tensor,
    masks: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:  # returns scored logits [N,V], sampled ids [N]
    """Select full-vocabulary logits and sampled ids at scored positions after the forward.

    ``selection = masks[:, :-1]`` pairs position j with the token it predicts, j+1 — the same alignment the
    GRPO objectives use, so a signal built against one is valid against the other. The flattened
    result is row-major over sequences then positions, which is the order signals and advantage
    vectors concatenate in.
    """
    logits = forward_logits / temperature  # [B,L,V]
    selection = masks[:, :-1]  # [B,L-1]
    scored_logits = logits[:, :-1][selection]  # [N,V]
    sampled_ids = input_ids[:, 1:][selection]  # [N]
    return scored_logits, sampled_ids


def _reduce_local(
    per_token: torch.Tensor,
    widths: list[int],
    loss_type: str,
    device: torch.device,
) -> tuple[torch.Tensor, int]:  # per_token: [N], returns numerator: [], units
    """Validate and reduce one rank's ragged per-token values without a collective."""
    expected_tokens = sum(widths)
    if expected_tokens != per_token.numel():
        raise RuntimeError(
            f"OPD got {per_token.numel()} per-token value(s) for {expected_tokens} scored token(s) — "
            "the DivergenceSource must emit exactly one per scored position."
        )
    token_counts = torch.tensor(widths, dtype=torch.long, device=device)  # [B]
    return reduce_policy_gradient_loss(per_token, token_counts, loss_type)


def make_direct_objective(
    packer: DataPacker,
    *,
    jsd_beta: float = 1.0,
    loss_type: str = "token-mean",
    temperature: float = 1.0,
    divergence: DivergenceSource | None = None,
) -> Objective:
    """The differentiable OPD objective: minimize the teacher/student divergence directly.

    Keyword knobs rather than a config object, deliberately. `OPDConfig` (design §4) is a run-level
    config that also owns the teacher, the data source and the entrypoint wiring; binding this
    factory to it now would couple the objective to a type that does not exist yet. The entrypoint
    passes the fields through when it lands, and the tests construct it directly meanwhile.

    Args:
        packer: the `DataPacker` shaping seam, shared with the GRPO objectives — each trajectory
            goes through `shape_policy_input`, so stored `policy_info["prompt_ids"]` win over a
            re-tokenization exactly as they do there.
        jsd_beta: 0 is `KL(teacher || student)` (forward, mode-covering), 1 is
            `KL(student || teacher)` (reverse, mode-seeking), interior values the generalized JSD.
            Defaults to 1 to match the cosmos-rl reference config.
        loss_type: `token-mean` (every token weighted equally, the reference default),
            `seq-mean-token-sum`, or `seq-mean-token-mean`. Same three modes and the same DP-global
            denominator as `algorithm/grpo.py::make_clipped_objective` — a parity knob, not
            formatting. NOTE: the "~1.4x on a 4..160-token spread" figure carried by
            `grpo.py` and `config/rl.py` does not hold for `token-mean` vs `seq-mean-token-mean`;
            measured 0.9965 +/- 0.0247 over 200 seeds with per-token loss independent of length.
            The identity is `ratio = 1 + corr(len, per-seq mean loss) * CV_len * CV_loss`, so length
            spread ALONE moves nothing — 1.4x needs a strongly front-loaded per-token loss profile.
            The large gap is between the seq-mean and token-mean FAMILIES, and it is of order the
            mean completion length, not 1.4x.
        temperature: divides the training-forward logits before the log-softmax. SEPARATE from the
            rollout sampling temperature; set both or they desync.
        divergence: the `DivergenceSource`. Defaults to `TopKPlusOtherDivergence(jsd_beta)`; pass one
            explicitly only for a same-token alternative such as legacy raw-slice parity.

    Returns:
        An `Objective` with deferred DP-global normalization for every loss type.

    Raises:
        ValueError: unknown `loss_type`, or `jsd_beta` outside [0, 1].
    """
    if loss_type not in _LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {_LOSS_TYPES}, got {loss_type!r}")
    source = divergence if divergence is not None else TopKPlusOtherDivergence(jsd_beta)

    def objective(model: Any, trajectories: list[Trajectory]) -> tuple[torch.Tensor, dict[str, Any]]:
        net = getattr(model, "net", model)  # raw module IS its own net; HFModel owns it under .net
        device = next(net.parameters()).device
        # An empty shard is legal — `select_chunk` round-robins samples, so a rank gets zero
        # whenever the batch is smaller than the mesh. Ask the group first: all-empty is a real
        # error and raising is symmetric; otherwise this rank takes the probe path.
        if rank_max_count(len(trajectories), device) == 0:
            raise ValueError("OPD direct objective got an empty batch — no trajectories to learn from.")
        # NO `require_advantages` here. `direct` never reads `Transition.advantage`: the gradient is
        # the divergence's own, not a policy gradient weighted by one.
        processed, _, signals, widths, _, probe_input = _prepare(
            trajectories, packer, device, label="OPD direct objective preparation", with_advantage=False
        )
        policy_batch, input_ids, masks = _prepare_forward_inputs(
            packer,
            processed,
            probe_input,
            device,
            context="OPD direct objective forward preparation",
        )  # [B,L], [B,L] or None
        policy_batch = run_rank_converged(
            lambda: prepare_policy_capability(model, policy_batch),
            "OPD direct policy capability preflight",
            raise_errors=raise_if_any_rank_failed,
        )
        forward_output = forward_policy(model, net, policy_batch)

        def finalize_local() -> tuple[torch.Tensor, int, dict[str, float]]:
            forward_logits = policy_logits(forward_output, input_ids, owner="OPD policy")  # [B,L,V]
            if masks is None:
                zero_loss = forward_logits.sum() * 0.0  # []
                return zero_loss, 0, {}
            student_logits, _ = _scored_positions(
                forward_logits,
                input_ids,
                masks,
                temperature,
            )  # [N,V], [N]
            per_token = source.per_token(student_logits, signals, device)  # [N]
            numerator, units = _reduce_local(per_token, widths, loss_type, device)  # [], count
            return numerator, units, {"opd_divergence": float(per_token.detach().mean())}

        loss, denominator, diagnostics = _converged_finalize(
            finalize_local,
            device,
            context="OPD direct objective finalization",
        )
        if not processed:
            return loss, {
                "opd_loss": 0.0,
                "opd_n": 0,
                "opd_valid_tokens": 0,
                "opd_divergence": 0.0,
                OBJECTIVE_WEIGHT_SUM: denominator,
            }
        return loss, {
            "opd_loss": float(loss.detach()),
            "opd_n": len(processed),
            "opd_valid_tokens": sum(widths),
            "opd_divergence": diagnostics["opd_divergence"],
            OBJECTIVE_WEIGHT_SUM: denominator,
        }

    return Objective(objective, reduction=Reduction.TOKEN_SUM, metric_adapter=_OPD_DIRECT_METRIC_ADAPTER)


def make_advantage_objective(
    packer: DataPacker,
    *,
    jsd_beta: float | None = 1.0,
    kl_penalty_coef: float = 1.0,
    kl_discount_factor: float = 0.0,
    loss_type: str = "token-mean",
    temperature: float = 1.0,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    kl_beta: float = 0.0,
    lower_bound_ratio: float = 3.0,
    divergence: DivergenceSource | None = None,
    teacher_credit: TeacherCreditSource | None = None,
) -> Objective:
    """OPD through either legacy divergence penalty or detached additive teacher credit.

    The group advantage comes from the reward stage, as it does for GRPO — this objective reads
    `Transition.advantage` and does not recompute it. Same-token legacy mode subtracts differentiable
    divergence, ``A_reward - coef * D``. DPCA instead adds detached credit,
    ``A_reward + coef * A_DPCA``; the two interfaces are mutually exclusive.

    **This is the objective that needs the widened advantage channel.** The group estimate is one
    scalar per sample and the penalty is per token, so the sum is per token; `advantage_to_tokens`
    broadcasts the first to meet the second (`schema/trajectory.py::Transition`).

    **IN LEGACY SAME-TOKEN MODE THE DIVERGENCE MUST KEEP ITS GRADIENT — that is where that mode's
    distillation signal lives.** At the
    on-policy `ratio == 1` the loss differentiates into `-(A_group - c*D_t)*∇log π(a_t) + c*∇D_t`,
    and only the second term carries teacher signal. The first term's divergence part is a
    state-dependent baseline: `D_t` is a function of the prefix and never of the sampled `a_t`, so
    `E[D_t * ∇log π] = 0` exactly. Cutting `∇D_t` therefore does not weaken the penalty, it deletes
    it, leaving plain GRPO plus injected variance. A `detach_divergence` flag used to be exposed
    here and was removed for that reason: no setting of it was sound, because the textbook MiniLLM
    form needs the accumulated FUTURE divergence (`kl_discount_factor`, refused below), not the
    local one. cosmos-rl's `top_k > 0` path — the one the reference config runs — likewise does not
    detach (`grpo_trainer.py:1691`, no `no_grad`); its `top_k == 0` branch does, soundly, because
    there the divergence is a sampled-token reverse KL and so IS action-dependent.

    Args:
        packer: the `DataPacker` shaping seam.
        jsd_beta: divergence direction, as in `make_direct_objective`. Must be ``None`` when
            ``teacher_credit`` is set, because DPCA has no shared-vocabulary JSD.
        kl_penalty_coef: finite non-negative weight on the selected teacher signal: a subtraction
            scale for legacy divergence or an addition scale for detached DPCA credit.
        kl_discount_factor: NOT IMPLEMENTED, see Raises. 0.0 (off) in the reference config.
        loss_type: as in `make_direct_objective`.
        temperature: divides the training-forward logits.
        epsilon_low, epsilon_high: PPO clip bounds. A negative value disables that side, matching
            cosmos-rl — the distillation configs pass -1.0 for both.
        kl_beta: KL-to-reference weight on the surrogate. 0 (off) in the reference config; a
            non-zero value is refused here because this objective computes no reference logprobs.
        lower_bound_ratio: dual-clip bound; <= 0 disables it. Values <= 1 make the dual clip bind at
            `ratio == 1` on negative-advantage tokens, which drops the policy-gradient route
            entirely; the reference passes 100.0.
        divergence: the legacy same-token `DivergenceSource`; defaults to
            `TopKSliceDivergence(jsd_beta)`. Tests inject a detaching wrapper here to pin that the
            divergence really is differentiable.
        teacher_credit: an explicit detached additive credit source such as DPCA. When set, the
            objective computes `A_reward + kl_penalty_coef * credit`; it never routes the credit
            through the legacy divergence-subtraction path.

    Returns:
        An `Objective` with deferred DP-global normalization for every loss type.

    Raises:
        ValueError: unknown `loss_type`, `jsd_beta` outside [0, 1], a negative/non-finite
            `kl_penalty_coef`, or `kl_beta > 0`.
        NotImplementedError: `kl_discount_factor != 0`. It is the return-to-go knob — the reference's
            `discounted_future_sum_loop` builds `R[t] = -c * sum_{s>=t} gamma^(s-t) * D_s`, which is
            what would make the policy-gradient route carry teacher signal at all. But it rebuilds
            the result from a Python list (`torch.tensor(...tolist()...)`), silently DETACHING the
            penalty, so enabling it REPLACES the differentiable route rather than adding to it.
            Copying that blindly would trade the working signal for a worse estimator. Left loud.
    """
    if loss_type not in _LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {_LOSS_TYPES}, got {loss_type!r}")
    if (
        isinstance(kl_beta, bool)
        or not isinstance(kl_beta, Real)
        or not math.isfinite(float(kl_beta))
        or float(kl_beta) != 0.0
    ):
        raise ValueError(
            "make_advantage_objective requires kl_beta=0 exactly because it computes no reference "
            f"logprobs, got {kl_beta!r}."
        )
    if (
        isinstance(kl_penalty_coef, bool)
        or not isinstance(kl_penalty_coef, Real)
        or not math.isfinite(float(kl_penalty_coef))
        or kl_penalty_coef < 0
    ):
        raise ValueError(f"kl_penalty_coef must be a non-bool finite real >= 0, got {kl_penalty_coef!r}")
    if kl_discount_factor != 0.0:
        raise NotImplementedError(
            f"kl_discount_factor={kl_discount_factor} is not implemented (see the docstring); the "
            "reference configs leave it at 0.0."
        )
    if divergence is not None and teacher_credit is not None:
        raise ValueError("divergence and teacher_credit are mutually exclusive teacher-signal interfaces")
    if teacher_credit is not None:
        if jsd_beta is not None:
            raise ValueError("teacher_credit requires jsd_beta=None because cross-token DPCA has no shared support")
        source: DivergenceSource | None = None
    else:
        if jsd_beta is None:
            raise ValueError("legacy same-token advantage OPD requires jsd_beta in [0, 1]")
        source = divergence if divergence is not None else TopKSliceDivergence(jsd_beta)
    # FROZEN pi_old, exactly as the GRPO clipped surrogate does it (`algorithm/grpo.py`).
    #
    # Without this the objective passed `current.detach()`, making the ratio identically 1 -- correct
    # only when a collected batch drives ONE optimizer step. At `n_mini > 1` the reference then
    # tracks the drifting policy, the epsilon-clip stops binding, and the run takes n_mini
    # unconstrained steps on one batch: the exact hazard the determinism attestation is demanded for.
    # `OPDConfig._objective_consults_pi_old()` returns True in this mode and the config gate demands
    # that attestation, so an operator who supplied it was given a guarantee this could not keep.
    old_logp_cache = OldLogprobCache()

    def objective(model: Any, trajectories: list[Trajectory]) -> tuple[torch.Tensor, dict[str, Any]]:
        net = getattr(model, "net", model)
        device = next(net.parameters()).device
        if rank_max_count(len(trajectories), device) == 0:
            raise ValueError("OPD advantage objective got an empty batch — no trajectories to learn from.")
        # The group advantage is written at REWARD time and travels with the trajectory; read, do
        # not recompute. This also converges its own validation across ranks.
        require_advantages(trajectories)
        processed, processed_trajectories, signals, widths, advantages, probe_input = _prepare(
            trajectories,
            packer,
            device,
            label="OPD advantage objective preparation",
            with_advantage=True,
            signal_kind="chunk" if teacher_credit is not None else "topk",
        )
        credit_tokens: torch.Tensor | None = None
        if teacher_credit is not None:

            def prepare_credit() -> torch.Tensor:  # returns [N]
                credit = teacher_credit.per_token_credit(  # [N]
                    processed_trajectories,
                    signals,
                    device,
                )
                if not isinstance(credit, torch.Tensor) or credit.ndim != 1:
                    raise ValueError(
                        f"TeacherCreditSource must return a rank-1 tensor, got {type(credit).__name__} "
                        f"with shape {getattr(credit, 'shape', None)}"
                    )
                if credit.numel() != sum(widths):
                    raise ValueError(
                        f"TeacherCreditSource returned {credit.numel()} value(s) for {sum(widths)} "
                        "scored policy token(s)"
                    )
                if credit.requires_grad:
                    raise ValueError("TeacherCreditSource output must be detached from autograd")
                finite = torch.isfinite(credit)  # [N]
                if not bool(finite.all()):
                    raise ValueError("TeacherCreditSource returned non-finite credit")
                return credit.to(device=device, dtype=torch.float32)  # [N]

            credit_tokens = run_rank_converged(
                prepare_credit,
                "OPD teacher-credit preparation",
                raise_errors=raise_if_any_rank_failed,
            )
        policy_batch, input_ids, masks = _prepare_forward_inputs(
            packer,
            processed,
            probe_input,
            device,
            context="OPD advantage objective forward preparation",
        )  # [B,L], [B,L] or None
        policy_batch = run_rank_converged(
            lambda: prepare_policy_capability(model, policy_batch),
            "OPD advantage policy capability preflight",
            raise_errors=raise_if_any_rank_failed,
        )
        forward_output = forward_policy(model, net, policy_batch)

        # π(a_t) for the token actually generated, from the SAME logits the divergence reads — one
        # forward, two consumers. `logit - logsumexp` rather than a full log-softmax, for the reason
        # in `TopKSliceDivergence`.
        #
        # ONE fp32 cast, both consumers. `_support_logprobs` casts too, but `.float()` on an
        # already-fp32 tensor returns the SAME object, so handing it `logits` rather than
        # `student_logits` keeps a single `[n_tok, V]` copy alive instead of two. Passing the bf16
        # original here silently doubled this path's peak — the objective's largest allocation, and
        # one autograd holds through the backward — against the figure `_support_logprobs` documents.
        def finalize_local() -> tuple[torch.Tensor, int, dict[str, float]]:
            forward_logits = policy_logits(forward_output, input_ids, owner="OPD policy")  # [B,L,V]
            if masks is None:
                zero_loss = forward_logits.sum() * 0.0  # []
                return zero_loss, 0, {}
            student_logits, sampled_ids = _scored_positions(
                forward_logits,
                input_ids,
                masks,
                temperature,
            )  # [N,V], [N]
            logits = student_logits.float()  # [N,V]
            current = logits.gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)  # [N]
            reward_advantage = torch.cat(advantages)  # [N]
            if credit_tokens is None:
                if source is None:
                    raise RuntimeError("legacy OPD divergence source is missing")
                divergence_tokens = source.per_token(logits, signals, device)  # [N]
                advantage = reward_advantage - kl_penalty_coef * divergence_tokens  # [N]
            else:
                divergence_tokens = torch.zeros_like(credit_tokens)  # [N]
                advantage = reward_advantage + kl_penalty_coef * credit_tokens  # [N]
            # pi_old from the ahead pass when one ran, else this forward detached. At n_mini == 1
            # the cache replays what `old_for` just recorded, so the ratio is still identically 1 and
            # behaviour is unchanged; at n_mini > 1 it is the FROZEN reference and the clip binds.
            per_token = clipped_policy_gradient_loss(
                current,
                old_logp_cache.old_for(current),
                advantage,
                epsilon_low,
                epsilon_high,
                kl_beta=0.0,
                lower_bound_ratio=lower_bound_ratio if lower_bound_ratio > 0 else None,
            )  # [N]
            numerator, units = _reduce_local(per_token, widths, loss_type, device)  # [], count
            return (
                numerator,
                units,
                {
                    "opd_divergence": float(divergence_tokens.detach().mean()),
                    "opd_advantage": float(advantage.detach().mean()),
                    "opd_teacher_credit": (0.0 if credit_tokens is None else float(credit_tokens.detach().mean())),
                },
            )

        loss, denominator, diagnostics = _converged_finalize(
            finalize_local,
            device,
            context="OPD advantage objective finalization",
        )
        if not processed:
            return loss, {
                "opd_loss": 0.0,
                "opd_n": 0,
                "opd_valid_tokens": 0,
                "opd_divergence": 0.0,
                "opd_advantage": 0.0,
                "opd_teacher_credit": 0.0,
                OBJECTIVE_WEIGHT_SUM: denominator,
            }
        return loss, {
            "opd_loss": float(loss.detach()),
            "opd_n": len(processed),
            "opd_valid_tokens": sum(widths),
            "opd_divergence": diagnostics["opd_divergence"],
            "opd_advantage": diagnostics["opd_advantage"],
            "opd_teacher_credit": diagnostics["opd_teacher_credit"],
            OBJECTIVE_WEIGHT_SUM: denominator,
        }

    # Handing the cache to the Objective is what makes `LLMTrainer.needs_old_logprobs()` answer True,
    # which is how the driver decides to run the ahead pass at all. Creating the cache without
    # publishing it here would leave the objective reading an object nothing ever fills -- which is
    # exactly the state this mode shipped in: the config gate demanded a determinism attestation for
    # an ahead pass that could never run.
    return Objective(
        objective,
        reduction=Reduction.TOKEN_SUM,
        metric_adapter=_OPD_ADVANTAGE_METRIC_ADAPTER,
        old_logp_cache=old_logp_cache,
    )
