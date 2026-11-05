# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Teacher role contracts for synchronous, ranked OPD scoring.

A teacher takes trajectories the rollout produced and attaches a `TeacherSignal` to each, in place
on `transition.targets`, before the batch is dispatched to the trainer:

    score(trajectories) -> list[Trajectory]        # same objects, now carrying TeacherSignal

A ROLE, NOT AN ANNOTATOR. `RewardWorker(annotators=...)` looks like the obvious home — an annotator
is exactly `Callable[[list[Trajectory]], None]` and runs over a whole batch at reward time — but an
annotator is a synchronous in-process call, and a teacher is either an HTTP service or a second
inference engine with its own placement and its own parallelism. Keeping it a role is what lets it
have replicas, its own device budget, and (later) a disaggregated home.

THE ALIGNMENT CONTRACT, which is the part that must not drift. `TeacherSignal.topk_ids[i]` is the
teacher's support for the **i-th SCORED token**, in order — see `algorithm/opd.py::_scored_signal`,
which refuses a row count that disagrees rather than guessing which end to trim. Bodies therefore
derive the scored span with the SAME `shape_policy_input` the objective uses, never by re-deriving
prompt/completion boundaries themselves. That single shared call is what makes the alignment correct
by construction instead of by review.

For a ranked vLLM teacher, every TP lane enters the same scoring call. The replica leader returns
the trajectories carrying tensor payloads; non-leaders return only :class:`TeacherRankStatus` so
duplicate ``[batch,tokens,top_k]`` results never enter replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from cosmos_framework.posttrain.schema import Trajectory


@dataclass(frozen=True)
class TeacherRankStatus:
    """Small successful result returned by a non-leader teacher TP lane.

    The status deliberately contains no trajectory or tensor field. It is suitable for an
    all-rank completion barrier without publishing duplicate teacher bulk to replay.
    """

    rank: int
    world_size: int
    request_digest: str
    signal_digest: str
    attempts: int
    scored_sequences: int
    scored_tokens: int


TeacherScoreResult = list[Trajectory] | TeacherRankStatus


@runtime_checkable
class TeacherBody(Protocol):
    """Attaches a `TeacherSignal` to every transition it is given.

    Called once per reward batch, on the same whole-group batch the reward stage sees. The replica
    leader returns the same trajectory objects, mutated atomically after whole-batch validation;
    non-leader TP lanes return :class:`TeacherRankStatus`.
    """

    def score(self, trajectories: list[Trajectory]) -> TeacherScoreResult: ...


class TeacherContractError(RuntimeError):
    """The teacher could not produce a usable signal for a batch.

    Raised rather than returning a partial batch. cosmos-rl degrades instead — a failed score
    becomes `None`, and its trainer logs "All teacher logprobs are None ... skipping distillation
    loss calculation" and carries on as plain GRPO. That is a silently un-distilled run; posttrain
    fails loud, because a distillation run that quietly stops distilling is worse than one that
    stops.
    """
