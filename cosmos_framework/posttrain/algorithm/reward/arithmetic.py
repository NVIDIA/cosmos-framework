# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Toy verifiable reward for `roles/env/dataset.py::ArithmeticEnv`.

TASK-SPECIFIC, and here for the same reason as `gsm8k.py`: paired with one env, not with the reward
machinery. The placement/topology tests (`tests/test_control_grpo_topologies.py`,
`tests/test_comm_weight_sync_asym.py`, ...) use this pair to get a non-zero GRPO gradient without a
real model or dataset.

A SMOKE verifier, not a production answer parser: it takes the first integer anywhere in the
completion. Deliberate -- the point is cheap reward VARIANCE within a group so the dispatch and
gradient plumbing can be exercised. Do not reach for it as a general numeric reward.
"""

from __future__ import annotations

import re

from cosmos_framework.posttrain.algorithm.reward.contract import _integer_answer
from cosmos_framework.posttrain.algorithm.reward.group import RewardCandidate, RewardContext
from cosmos_framework.posttrain.algorithm.reward.result import RewardProviderResult


def arithmetic_reward_fn(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
    """Toy rule for ArithmeticEnv: compare the completion's first integer to ground truth.

    This smoke verifier is intentionally simple and is not a production final-answer parser.
    Hit -> 1.0, else 0.0.
    """
    expected = _integer_answer(context.reference)
    match = re.search(r"-?\d+", candidate.completion)
    if match is None:
        return RewardProviderResult(terms={"task": 0.0})
    return RewardProviderResult(terms={"task": 1.0 if int(match.group()) == expected else 0.0})
