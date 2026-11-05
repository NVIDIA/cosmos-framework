# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Strict shared accessors for reward production and consumption.

The framework request builder reads terminal completion text here; candidate scorers validate their
typed `RewardContext` inputs with the task-specific helpers below. Missing values raise rather than
default because a wiring failure must not become a plausible zero reward.

The terminal-final accessor is shared by the advantage estimator, shared-prefix filter, and
reward-batch reporting. A missing reward must not become a plausible ``0.0`` in any of them: that
would make a wiring failure indistinguishable from a legitimate zero-reward completion in both the
gradient and diagnostics.

Names keep their leading underscore because these accessors are internal to the reward package.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from math import isfinite
from numbers import Integral, Real

from cosmos_framework.posttrain.schema import FINAL_REWARD_KEY, Trajectory, Transition


class RewardContractError(ValueError):
    """A reward cannot be evaluated because its declared runtime contract was violated."""


def _terminal_transition(traj: Trajectory) -> Transition:
    if not traj.transitions:
        raise RewardContractError("reward consumer needs a trajectory with at least one transition")
    return traj.transitions[-1]


def _exact_token_ids(value: object, *, field: str) -> tuple[int, ...]:
    """Normalize one exact one-dimensional token-ID sequence without importing tensor libraries."""
    if value is None or isinstance(value, (str, bytes)):
        raise RewardContractError(f"{field} must carry exact completion action token IDs, got {value!r}")
    if isinstance(value, Sequence):
        token_ids = tuple(value)
    else:
        tolist = getattr(value, "tolist", None)
        if not callable(tolist):
            raise RewardContractError(f"{field} must carry exact completion action token IDs, got {value!r}")
        token_ids = tolist()
        if not isinstance(token_ids, list):
            raise RewardContractError(f"{field} must carry one-dimensional completion action token IDs")
        token_ids = tuple(token_ids)
    if not token_ids:
        raise RewardContractError(f"{field} must carry at least one exact completion action token ID")
    if any(isinstance(token_id, bool) or not isinstance(token_id, Integral) or token_id < 0 for token_id in token_ids):
        raise RewardContractError(f"{field} must carry non-negative integer completion action token IDs")
    return tuple(int(token_id) for token_id in token_ids)


def _is_materialized_token_id_sequence(value: object) -> bool:
    """Whether ``value`` exposes IDs that must be validated rather than treated as an opaque handle."""
    if value is None or isinstance(value, (str, bytes)):
        return False
    return isinstance(value, Sequence) or callable(getattr(value, "tolist", None))


def _completion_action_token_ids(traj: Trajectory) -> tuple[int, ...]:
    """Resolve exact terminal completion IDs from inline metadata or a materialized action.

    Reward actors may receive either a list/tuple, a tensor-like action, or an opaque bulk handle
    accompanied by rollout-stamped ``completion_ids``. A count-only metadata hint is insufficient:
    token-aligned rewards require evidence of the exact realized token sequence.
    """
    transition = _terminal_transition(traj)
    inline_token_ids = transition.policy_info.get("completion_ids")
    normalized_inline_ids = (
        None if inline_token_ids is None else _exact_token_ids(inline_token_ids, field="policy_info['completion_ids']")
    )
    normalized_action_ids = (
        _exact_token_ids(transition.action, field="transition.action")
        if _is_materialized_token_id_sequence(transition.action)
        else None
    )
    if (
        normalized_inline_ids is not None
        and normalized_action_ids is not None
        and normalized_inline_ids != normalized_action_ids
    ):
        raise RewardContractError(
            f"trajectory {traj.id!r} inline completion_ids {normalized_inline_ids} differ from "
            f"materialized action token IDs {normalized_action_ids}"
        )
    if normalized_inline_ids is not None:
        return normalized_inline_ids
    if normalized_action_ids is not None:
        return normalized_action_ids
    raise RewardContractError(
        f"trajectory {traj.id!r} needs exact completion action token IDs to align token reward terms"
    )


def _completion_action_token_count(traj: Trajectory) -> int:
    """Return the exact terminal completion width after validating all available token identities."""
    return len(_completion_action_token_ids(traj))


def _terminal_reward(traj: Trajectory) -> float:
    """Return a trajectory's explicit finite terminal ``reward['final']``.

    This is the one strict interpretation used wherever a scored trajectory becomes either a
    training signal or a runtime diagnostic. Genuine numeric zero is valid. Missing, bool,
    non-numeric, and non-finite values raise instead of being coerced into a healthy-looking report.

    Args:
        traj: The scored trajectory to validate.

    Returns:
        The finite terminal final reward as a float.

    Raises:
        RewardContractError: The trajectory has no transition or no usable terminal final reward.
    """
    transition = _terminal_transition(traj)
    reward = transition.reward
    if reward is None or FINAL_REWARD_KEY not in reward:
        raise RewardContractError("GRPO needs terminal reward['final']")
    value = reward[FINAL_REWARD_KEY]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RewardContractError(f"GRPO needs terminal reward['final'] to be numeric, got {value!r}")
    try:
        result = float(value)
    except OverflowError as exc:
        raise RewardContractError(f"GRPO needs finite terminal reward['final'], got {value!r}") from exc
    if not isfinite(result):
        raise RewardContractError(f"GRPO needs finite terminal reward['final'], got {result!r}")
    return result


def _completion_text(traj: Trajectory) -> str:
    transition = _terminal_transition(traj)
    if "completion_text" not in transition.policy_info or transition.policy_info["completion_text"] is None:
        raise RewardContractError("reward provider needs policy_info['completion_text']")
    completion = transition.policy_info["completion_text"]
    if not isinstance(completion, str):
        raise RewardContractError(
            f"reward provider needs completion_text to be a string, got {type(completion).__name__}"
        )
    return completion


def _scalar_answer_text(answer: object) -> str:
    if isinstance(answer, bool) or not isinstance(answer, (str, int, float)):
        raise RewardContractError(f"exact-match reward needs a string or finite numeric answer, got {answer!r}")
    if isinstance(answer, float) and not isfinite(answer):
        raise RewardContractError(f"exact-match reward needs a string or finite numeric answer, got {answer!r}")
    return str(answer)


def _integer_answer(answer: object) -> int:
    if isinstance(answer, bool):
        raise RewardContractError(f"arithmetic reward needs an integer answer, got {answer!r}")
    if isinstance(answer, int):
        return answer
    if isinstance(answer, float):
        if isfinite(answer) and answer.is_integer():
            return int(answer)
        raise RewardContractError(f"arithmetic reward needs an integer answer, got {answer!r}")
    if isinstance(answer, str) and re.fullmatch(r"[+-]?\d+", answer.strip()):
        return int(answer)
    raise RewardContractError(f"arithmetic reward needs an integer answer, got {answer!r}")
