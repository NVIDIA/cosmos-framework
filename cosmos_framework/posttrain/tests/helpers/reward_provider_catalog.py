# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Importable custom reward provider catalog used by CPU wiring tests."""

from __future__ import annotations

from functools import partial

from cosmos_framework.posttrain.algorithm.reward import (
    RewardCandidate,
    RewardContext,
    RewardProviderResult,
    RewardProviderSpec,
    map_candidates,
)
from cosmos_framework.posttrain.schema import MetricContribution


def custom_prompt_reward(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
    """Small pointwise scorer proving that recipes can add rewards outside the framework package."""
    del candidate, context
    return RewardProviderResult(
        terms={"custom": 0.25},
        diagnostics={"accepted": MetricContribution.sum(1)},
    )


def custom_reward_provider_catalog() -> dict[str, RewardProviderSpec]:
    """Return one provider with a fixed term, diagnostic, provenance, and input contract."""
    return {
        "custom_prompt": RewardProviderSpec(
            factory=partial(map_candidates, custom_prompt_reward),
            needs=("prompt",),
            revision="test-v1",
            diagnostic_names=("accepted",),
            term_names=("custom",),
        )
    }
