# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Exact dual-pointer chunk alignment and detached DPCA teacher credit.

DPCA aligns two tokenizations of the same realized completion by exact decoded text. It never
force-closes an unmatched chunk or attaches an unmatched tail. An exactly zero student chunk uses
the reference's unit-ratio convention and therefore contributes zero credit; nonzero near-zero
denominators still fail closed. The teacher signal stores full student-completion coordinates;
prefix filtering is applied only after full-completion credit has been constructed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

import torch

from cosmos_framework.posttrain.schema import TeacherSignal, Trajectory

_LOGPROB_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class AlignmentChunk:
    """One exact decoded-text match in student and teacher token coordinates."""

    student_start: int
    student_end: int
    teacher_start: int
    teacher_end: int


class TeacherCreditSource(Protocol):
    """Detached per-token teacher credit added to the ordinary reward advantage."""

    def per_token_credit(
        self,
        trajectories: list[Trajectory],
        signals: list[TeacherSignal],
        device: torch.device,
    ) -> torch.Tensor:  # returns [N]
        ...


def _decoded_text(decoder: Callable[[list[int]], str], token_ids: list[int], *, label: str) -> str:
    """Decode one token slice and require an exact string result."""
    try:
        text = decoder(token_ids)
    except Exception as error:  # noqa: BLE001 -- normalize arbitrary tokenizer backend failures
        raise ValueError(f"{label} decode failed: {error}") from error
    if not isinstance(text, str):
        raise ValueError(f"{label} decoder must return str, got {type(text).__name__}")
    return text


def align_completion_chunks(
    student_token_ids: Sequence[int],
    teacher_token_ids: Sequence[int],
    student_decode: Callable[[list[int]], str],
    teacher_decode: Callable[[list[int]], str],
    *,
    max_chunk_tokens: int = 64,
) -> list[AlignmentChunk]:
    """Align two full completion tokenizations by exact decoded-text chunks.

    The shorter decoded side advances until the strings match. Equal-length unequal strings cannot
    be repaired by appending to only one side and therefore fail immediately. ``max_chunk_tokens``
    is a rejection guard on either tokenizer's chunk width, never a force-close rule.
    """
    if isinstance(max_chunk_tokens, bool) or not isinstance(max_chunk_tokens, int) or max_chunk_tokens < 1:
        raise ValueError(f"max_chunk_tokens must be a non-bool positive integer, got {max_chunk_tokens!r}")
    student_ids = [int(token_id) for token_id in student_token_ids]
    teacher_ids = [int(token_id) for token_id in teacher_token_ids]
    if not student_ids or not teacher_ids:
        raise ValueError(
            f"DPCA alignment requires non-empty tokenizations, got student={len(student_ids)} "
            f"teacher={len(teacher_ids)}"
        )
    student_full = _decoded_text(student_decode, student_ids, label="full student completion")
    teacher_full = _decoded_text(teacher_decode, teacher_ids, label="full teacher completion")
    if student_full != teacher_full:
        raise ValueError("DPCA full student and teacher completion texts do not match exactly")

    student_index = 0
    teacher_index = 0
    chunks: list[AlignmentChunk] = []
    while student_index < len(student_ids) and teacher_index < len(teacher_ids):
        student_start = student_index
        teacher_start = teacher_index
        student_index += 1
        teacher_index += 1
        while True:
            student_width = student_index - student_start
            teacher_width = teacher_index - teacher_start
            if student_width > max_chunk_tokens or teacher_width > max_chunk_tokens:
                raise ValueError(
                    "DPCA alignment exceeded max_chunk_tokens without an exact match: "
                    f"student_width={student_width}, teacher_width={teacher_width}, "
                    f"limit={max_chunk_tokens}"
                )
            student_text = _decoded_text(
                student_decode,
                student_ids[student_start:student_index],
                label=f"student chunk [{student_start}:{student_index}]",
            )
            teacher_text = _decoded_text(
                teacher_decode,
                teacher_ids[teacher_start:teacher_index],
                label=f"teacher chunk [{teacher_start}:{teacher_index}]",
            )
            if student_text == teacher_text:
                chunks.append(
                    AlignmentChunk(
                        student_start=student_start,
                        student_end=student_index,
                        teacher_start=teacher_start,
                        teacher_end=teacher_index,
                    )
                )
                break
            if len(student_text) < len(teacher_text):
                if student_index >= len(student_ids):
                    raise ValueError("DPCA student tokens ended before the current teacher chunk matched")
                student_index += 1
            elif len(teacher_text) < len(student_text):
                if teacher_index >= len(teacher_ids):
                    raise ValueError("DPCA teacher tokens ended before the current student chunk matched")
                teacher_index += 1
            else:
                raise ValueError(
                    "DPCA decoded chunks have equal length but different text; exact alignment is impossible"
                )

    if student_index != len(student_ids) or teacher_index != len(teacher_ids):
        raise ValueError(
            "DPCA alignment left an unmatched tail: "
            f"student={student_index}/{len(student_ids)}, teacher={teacher_index}/{len(teacher_ids)}"
        )
    _validate_alignment_cover(chunks, len(student_ids), len(teacher_ids))
    return chunks


def _validate_alignment_cover(
    chunks: Sequence[AlignmentChunk],
    student_length: int,
    teacher_length: int,
) -> None:
    """Require ordered, gap-free, non-overlapping coverage of both tokenizations."""
    expected_student = 0
    expected_teacher = 0
    for chunk_index, chunk in enumerate(chunks):
        if chunk.student_start != expected_student or chunk.teacher_start != expected_teacher:
            raise ValueError(
                f"DPCA chunk {chunk_index} does not continue the exact cover: expected student/teacher "
                f"starts {expected_student}/{expected_teacher}, got {chunk.student_start}/{chunk.teacher_start}"
            )
        if chunk.student_end <= chunk.student_start or chunk.teacher_end <= chunk.teacher_start:
            raise ValueError(f"DPCA chunk {chunk_index} is empty or reversed")
        if chunk.student_end > student_length or chunk.teacher_end > teacher_length:
            raise ValueError(f"DPCA chunk {chunk_index} extends beyond its completion")
        expected_student = chunk.student_end
        expected_teacher = chunk.teacher_end
    if expected_student != student_length or expected_teacher != teacher_length:
        raise ValueError(
            "DPCA chunks do not cover both full completions: "
            f"student={expected_student}/{student_length}, teacher={expected_teacher}/{teacher_length}"
        )


def build_dpca_signal(
    student_token_ids: Sequence[int],
    teacher_token_ids: Sequence[int],
    teacher_token_logprobs: Sequence[float] | torch.Tensor,
    student_decode: Callable[[list[int]], str],
    teacher_decode: Callable[[list[int]], str],
    *,
    max_chunk_tokens: int = 64,
) -> TeacherSignal:
    """Build a chunk signal by exact alignment and teacher-logprob summation."""
    chunks = align_completion_chunks(
        student_token_ids,
        teacher_token_ids,
        student_decode,
        teacher_decode,
        max_chunk_tokens=max_chunk_tokens,
    )
    teacher_logprobs = torch.as_tensor(teacher_token_logprobs).detach().to(dtype=torch.float32)  # [Tt]
    if teacher_logprobs.ndim != 1 or teacher_logprobs.numel() != len(teacher_token_ids):
        raise ValueError(
            "teacher_token_logprobs must contain exactly one value per teacher completion token, got "
            f"shape={tuple(teacher_logprobs.shape)} tokens={len(teacher_token_ids)}"
        )
    finite = torch.isfinite(teacher_logprobs)  # [Tt]
    if not bool(finite.all()):
        raise ValueError("teacher_token_logprobs contains non-finite values")
    bounds = torch.tensor(  # [C,2]
        [[chunk.student_start, chunk.student_end] for chunk in chunks],
        dtype=torch.int64,
    )
    chunk_values: list[torch.Tensor] = []
    for chunk in chunks:
        chunk_value = teacher_logprobs[chunk.teacher_start : chunk.teacher_end].sum()  # []
        chunk_values.append(chunk_value)
    chunk_logprobs = torch.stack(chunk_values).to(dtype=torch.float32)  # [C]
    return TeacherSignal(chunk_bounds=bounds, chunk_teacher_logprob=chunk_logprobs)


class DPCATeacherCreditSource:
    """Construct Eq. 9 credit from full-completion rollout likelihoods and chunk signals."""

    min_abs_student_chunk_logprob: float

    def __init__(self, *, min_abs_student_chunk_logprob: float = 1.0e-8) -> None:
        if (
            isinstance(min_abs_student_chunk_logprob, bool)
            or not isinstance(min_abs_student_chunk_logprob, Real)
            or not math.isfinite(float(min_abs_student_chunk_logprob))
            or min_abs_student_chunk_logprob <= 0.0
        ):
            raise ValueError(
                "min_abs_student_chunk_logprob must be a non-bool finite real > 0, got "
                f"{min_abs_student_chunk_logprob!r}"
            )
        self.min_abs_student_chunk_logprob = float(min_abs_student_chunk_logprob)

    def per_token_credit(
        self,
        trajectories: list[Trajectory],
        signals: list[TeacherSignal],
        device: torch.device,
    ) -> torch.Tensor:  # returns [N]
        if len(trajectories) != len(signals):
            raise ValueError(
                f"DPCA needs one signal per scored trajectory, got {len(signals)} signal(s) for "
                f"{len(trajectories)} trajectory(s)"
            )
        credits: list[torch.Tensor] = []
        for trajectory_index, (trajectory, signal) in enumerate(zip(trajectories, signals)):
            credit = self._trajectory_credit(  # [Ti]
                trajectory,
                signal,
                trajectory_index,
                device,
            )
            credits.append(credit)  # [Ti]
        if not credits:
            return torch.empty(0, dtype=torch.float32, device=device)  # [0]
        return torch.cat(credits)  # [N]

    def _trajectory_credit(
        self,
        trajectory: Trajectory,
        signal: TeacherSignal,
        trajectory_index: int,
        device: torch.device,
    ) -> torch.Tensor:  # returns [Ti-n_ignore]
        if len(trajectory.transitions) != 1:
            raise ValueError(
                f"DPCA trajectory {trajectory_index} must contain exactly one transition, got "
                f"{len(trajectory.transitions)}"
            )
        transition = trajectory.transitions[0]
        action = transition.action
        if isinstance(action, (str, bytes)) or not hasattr(action, "__len__"):
            raise ValueError(f"DPCA trajectory {trajectory_index} action must carry completion token IDs")
        action_ids = torch.as_tensor(action).detach()  # [T]
        integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if action_ids.ndim != 1 or action_ids.dtype not in integer_dtypes:
            raise ValueError(f"DPCA trajectory {trajectory_index} action must be a rank-1 integer token tensor")
        raw_completion_ids = transition.policy_info.get("completion_ids")
        if isinstance(raw_completion_ids, (str, bytes)) or raw_completion_ids is None:
            raise ValueError(f"DPCA trajectory {trajectory_index} is missing inline completion_ids")
        try:
            completion_ids = list(raw_completion_ids)
        except (TypeError, ValueError) as error:
            raise ValueError(f"DPCA trajectory {trajectory_index} completion_ids must be an inline iterable") from error
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0 for token_id in completion_ids
        ):
            raise ValueError(f"DPCA trajectory {trajectory_index} completion_ids contains an invalid token id")
        inline_ids = torch.tensor(completion_ids, dtype=torch.long)  # [T]
        normalized_action_ids = action_ids.to(device="cpu", dtype=torch.long)  # [T]
        if not torch.equal(normalized_action_ids, inline_ids):
            raise ValueError(
                f"DPCA trajectory {trajectory_index} materialized action differs from inline completion_ids"
            )
        completion_length = len(completion_ids)
        if completion_length < 1:
            raise ValueError(f"DPCA trajectory {trajectory_index} has an empty completion")

        raw_rollout_logprobs = transition.policy_info.get("rollout_logprobs")
        if raw_rollout_logprobs is None:
            raise ValueError(f"DPCA trajectory {trajectory_index} is missing policy_info['rollout_logprobs']")
        rollout_logprobs = (  # [T]
            torch.as_tensor(raw_rollout_logprobs, device=device).detach().to(dtype=torch.float32)
        )
        if rollout_logprobs.ndim != 1 or rollout_logprobs.numel() != completion_length:
            raise ValueError(
                f"DPCA trajectory {trajectory_index} needs exactly one rollout logprob per full completion "
                f"token, got shape={tuple(rollout_logprobs.shape)} tokens={completion_length}"
            )
        rollout_finite = torch.isfinite(rollout_logprobs)  # [T]
        if not bool(rollout_finite.all()):
            raise ValueError(f"DPCA trajectory {trajectory_index} rollout logprobs contains non-finite values")
        positive_rollout_logprobs = rollout_logprobs > _LOGPROB_TOLERANCE  # [T]
        if bool(positive_rollout_logprobs.any()):
            raise ValueError(f"DPCA trajectory {trajectory_index} rollout logprobs contains a positive value")

        bounds = signal.chunk_bounds
        raw_teacher_logprobs = signal.chunk_teacher_logprob
        if not isinstance(bounds, torch.Tensor) or not isinstance(raw_teacher_logprobs, torch.Tensor):
            raise ValueError(f"DPCA trajectory {trajectory_index} chunk signal is not materialized as tensors")
        if bounds.dtype not in integer_dtypes:
            raise ValueError(f"DPCA trajectory {trajectory_index} chunk_bounds must have integer dtype")
        bounds = bounds.detach().to(device=device, dtype=torch.long)  # [C,2]
        teacher_logprobs = raw_teacher_logprobs.detach().to(device=device, dtype=torch.float32)  # [C]
        if bounds.ndim != 2 or bounds.shape[1] != 2 or teacher_logprobs.shape != (bounds.shape[0],):
            raise ValueError(
                f"DPCA trajectory {trajectory_index} needs chunk_bounds [C,2] and teacher logprobs [C], "
                f"got {tuple(bounds.shape)} and {tuple(teacher_logprobs.shape)}"
            )
        if bounds.shape[0] < 1:
            raise ValueError(f"DPCA trajectory {trajectory_index} has no alignment chunks")
        teacher_finite = torch.isfinite(teacher_logprobs)  # [C]
        if not bool(teacher_finite.all()):
            raise ValueError(f"DPCA trajectory {trajectory_index} teacher chunk logprobs contains non-finite values")
        positive_teacher_logprobs = teacher_logprobs > _LOGPROB_TOLERANCE  # [C]
        if bool(positive_teacher_logprobs.any()):
            raise ValueError(f"DPCA trajectory {trajectory_index} teacher chunk logprobs contains a positive value")

        full_credit = torch.empty_like(rollout_logprobs)  # [T]
        expected_start = 0
        for chunk_index in range(bounds.shape[0]):
            start = int(bounds[chunk_index, 0])
            end = int(bounds[chunk_index, 1])
            if start != expected_start:
                raise ValueError(
                    f"DPCA trajectory {trajectory_index} chunk {chunk_index} starts at {start}, "
                    f"expected contiguous start {expected_start}"
                )
            if end <= start or end > completion_length:
                raise ValueError(
                    f"DPCA trajectory {trajectory_index} chunk {chunk_index} has invalid bounds "
                    f"[{start},{end}) for completion length {completion_length}"
                )
            student_chunk = rollout_logprobs[start:end]  # [Tc]
            student_sum = student_chunk.sum()  # []
            if not bool(torch.isfinite(student_sum)):
                raise ValueError(f"DPCA trajectory {trajectory_index} chunk {chunk_index} has non-finite L_S")
            student_sum_value = float(student_sum)
            if student_sum_value == 0.0:
                # Cosmos-RL defines the indeterminate exact-zero chunk as ratio=1. This occurs for
                # deterministic sampled EOS tokens (log p == 0) and makes their token credit zero.
                full_credit[start:end] = torch.zeros_like(student_chunk)  # [Tc]
                expected_start = end
                continue
            if abs(student_sum_value) <= self.min_abs_student_chunk_logprob:
                raise ValueError(
                    f"DPCA trajectory {trajectory_index} chunk {chunk_index} has near-zero L_S="
                    f"{student_sum_value} at floor {self.min_abs_student_chunk_logprob}"
                )
            teacher_sum = teacher_logprobs[chunk_index]  # []
            ratio = teacher_sum / student_sum  # []
            chunk_credit = (ratio - 1.0) * student_chunk  # [Tc]
            full_credit[start:end] = chunk_credit  # [Tc]
            expected_start = end
        if expected_start != completion_length:
            raise ValueError(
                f"DPCA trajectory {trajectory_index} chunks cover {expected_start}/{completion_length} "
                "student completion tokens"
            )

        raw_ignore = transition.algo_extra.get("n_ignore_prefix_tokens", 0)
        if isinstance(raw_ignore, bool) or not isinstance(raw_ignore, int):
            raise ValueError(f"DPCA trajectory {trajectory_index} n_ignore_prefix_tokens must be a non-bool integer")
        if not 0 <= raw_ignore <= completion_length:
            raise ValueError(
                f"DPCA trajectory {trajectory_index} n_ignore_prefix_tokens={raw_ignore} is outside "
                f"completion length {completion_length}"
            )
        effective_credit = full_credit[raw_ignore:]  # [Ti-n_ignore]
        return effective_credit
