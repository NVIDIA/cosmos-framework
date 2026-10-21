# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Reward — unified abstraction (DESIGN §1.5).

THE public surface of this package: the reward FUNCTIONS a recipe can select, plus the machinery
that runs them. Nothing else. `extract_gsm8k_answer` is deliberately absent — it is a helper of
`gsm8k_reward_fn`, not a reward, and listing it here made the export list read as a grab-bag rather
than as "these are the rewards available". Reach for it at `reward.gsm8k.extract_gsm8k_answer` if you
genuinely need the parser; importing from the defining module is the honest way to use an internal.

Each name is imported from the module that actually defines it, not laundered through `reward.py`.
That mattered concretely: while the task rules were re-exported from there, `reward.py` had to keep
imports alive purely to be re-exported, and its `__all__` was the only thing using one of them —
remove the list and the import became dead, breaking the package. `reward.py` no longer carries an
`__all__`; this file is the single source of truth for what callers may use, and every module below
imports only what it needs.

    contract.py    terminal trajectory accessors and shared reward failure semantics
    group.py       typed group request/result correlation and request construction
    result.py      optimized terms and isolated typed diagnostics
    reward.py      the group-native provider protocol, pointwise adapter, combiner, and registry
    gsm8k.py       GSM8K — paired with configs/grpo_qwen3b_gsm8k.py's prompt, cosmos-rl parity
    arithmetic.py  toy — paired with ArithmeticEnv
"""

from cosmos_framework.posttrain.algorithm.reward.arithmetic import arithmetic_reward_fn
from cosmos_framework.posttrain.algorithm.reward.contract import RewardContractError
from cosmos_framework.posttrain.algorithm.reward.group import (
    RewardCandidate,
    RewardCandidateResult,
    RewardContext,
    RewardGroupRequest,
    RewardGroupResult,
)
from cosmos_framework.posttrain.algorithm.reward.gsm8k import gsm8k_reward_fn
from cosmos_framework.posttrain.algorithm.reward.result import RewardProviderResult
from cosmos_framework.posttrain.algorithm.reward.reward import (
    DEFAULT_REWARD_PROVIDER_CATALOG,
    RULE_REWARDS,
    BoundRewardProvider,
    CandidateRewardFn,
    RewardCombiner,
    RewardProvider,
    RewardProviderCatalog,
    RewardProviderSpec,
    RuleRewardSpec,
    bind_candidate_provider,
    constant_reward_fn,
    exact_match_reward_fn,
    map_candidates,
    nonempty_reward_fn,
    resolve_reward_provider_catalog,
    rule_reward_spec,
)

__all__ = [
    "DEFAULT_REWARD_PROVIDER_CATALOG",
    "RULE_REWARDS",
    "BoundRewardProvider",
    "CandidateRewardFn",
    "RewardCombiner",
    "RewardContractError",
    "RewardCandidate",
    "RewardCandidateResult",
    "RewardContext",
    "RewardGroupRequest",
    "RewardGroupResult",
    "RewardProvider",
    "RewardProviderCatalog",
    "RewardProviderSpec",
    "RewardProviderResult",
    "RuleRewardSpec",
    "arithmetic_reward_fn",
    "bind_candidate_provider",
    "constant_reward_fn",
    "exact_match_reward_fn",
    "gsm8k_reward_fn",
    "map_candidates",
    "nonempty_reward_fn",
    "resolve_reward_provider_catalog",
    "rule_reward_spec",
]
