# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""roles.reward — reward WG body variants (DESIGN §1.2/§1.5). `worker` (in-process);
an external reward service (body=(b)) drops in as a sibling."""

from cosmos_framework.posttrain.roles.reward.diagnostics import ProviderMetricReport
from cosmos_framework.posttrain.roles.reward.ppo import PPORewardRole
from cosmos_framework.posttrain.roles.reward.worker import RewardWorker, reward_state_transaction

__all__ = ["PPORewardRole", "ProviderMetricReport", "RewardWorker", "reward_state_transaction"]
