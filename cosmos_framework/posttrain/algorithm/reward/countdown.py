# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Countdown verifiable reward — cosmos-rl's `custom_reward_fn`, ported VERBATIM.

Source: `packages/cosmos-rl/cosmos_rl/tools/dataset/countdown_distill.py::custom_reward_fn`, the
verifier the OPD reference config scores with. Ported rather than reimplemented for the same reason
`gsm8k.py` is: an alignment run compares against a number this function produced, so any
"improvement" here shows up as an implementation difference in the comparison.

The task: given a multiset of numbers and a target, write an arithmetic expression using each number
exactly once that equals the target. A completion scores 1.0 only if all three hold —

1. the equation is TRUE (`simplify(left - right) == 0`, sympy, so `6*(4+3)` == `42` counts);
2. the numbers used on the left are the given multiset, sorted-compared so order does not matter and
   a repeated number cannot be dropped or invented;
3. the right-hand side is the target.

`<answer>...</answer>` is stripped when present, matching the reference, which lets a model wrap its
final line in tags without being punished for it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from cosmos_framework.posttrain.algorithm.reward.group import RewardCandidate, RewardContext
from cosmos_framework.posttrain.algorithm.reward.result import RewardProviderResult

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_INTEGERS = re.compile(r"\d+")


def countdown_reference(answer: object) -> tuple[list[int], int]:
    """Validate and unpack the Countdown reference `{"nums": [...], "target": N}`.

    Countdown's ground truth is STRUCTURED, unlike every other built-in reward's scalar answer — the
    verifier has to know both which numbers were offered and what they had to reach. Validated here
    so a malformed row fails with what is wrong rather than scoring 0.0, which would be
    indistinguishable from a model that never solves anything.

    Args:
        reference: the value at `obs.representations['reference']`.

    Returns:
        `(nums, target)`.

    Raises:
        RewardContractError: not a mapping, or missing/non-integer `nums`/`target`.
    """
    from cosmos_framework.posttrain.algorithm.reward.contract import RewardContractError

    if not isinstance(answer, Mapping) or "nums" not in answer or "target" not in answer:
        raise RewardContractError(
            f"countdown reward needs obs.representations['reference'] = {{'nums': [...], 'target': int}}, got {answer!r}"
        )
    nums, target = answer["nums"], answer["target"]
    try:
        nums = [int(n) for n in nums]
        target = int(target)
    except (TypeError, ValueError) as exc:
        raise RewardContractError(f"countdown reward needs integer nums/target, got {answer!r}") from exc
    if not nums:
        raise RewardContractError(f"countdown reward needs a non-empty nums list, got {answer!r}")
    return nums, target


def countdown_reward_fn(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
    """Countdown verifiable reward (cosmos-rl `custom_reward_fn`, VERBATIM logic).

    Args:
        candidate: The immutable candidate completion.
        context: The common group context carrying the structured reference.

    Returns:
        A result whose ``task`` term is 1.0 when the equation is true, uses exactly the offered
        numbers, and reaches the target; 0.0 otherwise, including on any parse failure.
    """
    # Contract first, as in `gsm8k.py`: a missing reference is a wiring error and must raise. A
    # silent zero there is indistinguishable from a wrong answer, so a misconfigured source would
    # look like a model that never gets anything right.
    completion = candidate.completion
    nums, target = countdown_reference(context.reference)
    # Above the guard on purpose: an image without sympy is a wiring error, not a wrong answer. Under
    # the guard, an ImportError would score every completion 0.0 and the run would train happily on
    # an all-zero reward — the same failure mode `TeacherContractError` exists to prevent.
    from sympy import simplify
    from sympy.parsing.sympy_parser import parse_expr

    try:
        tagged = _ANSWER_TAG.search(completion)
        student_answer = tagged.group(1).strip() if tagged else completion.strip()
        left_str, right_str = student_answer.split("=")
        left = parse_expr(left_str.strip())
        right = parse_expr(right_str.strip())
        is_true = simplify(left - right) == 0

        used = [int(x) for x in _INTEGERS.findall(left_str)]
        result = int(_INTEGERS.findall(right_str)[0])
        if sorted(used) != sorted(nums):
            return RewardProviderResult(terms={"task": 0.0})
        if result != target:
            return RewardProviderResult(terms={"task": 0.0})
        return RewardProviderResult(terms={"task": 1.0 if is_true else 0.0})
    except Exception:  # noqa: BLE001 — parity with cosmos-rl's guard: any parse failure scores 0
        return RewardProviderResult(terms={"task": 0.0})
