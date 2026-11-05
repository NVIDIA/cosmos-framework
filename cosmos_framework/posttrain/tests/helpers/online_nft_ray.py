# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Model-free two-rank role mesh Ray fixture for online NFT."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch

from cosmos_framework.posttrain.algorithm.annotators.advantage import distributed_group_relative
from cosmos_framework.posttrain.algorithm.nft import NFTObjectiveConfig, make_objective
from cosmos_framework.posttrain.algorithm.reward import RewardCombiner
from cosmos_framework.posttrain.config.rl import VideoRewardConfig
from cosmos_framework.posttrain.control import ExperienceBuffer
from cosmos_framework.posttrain.roles.env import WFMDataEnv
from cosmos_framework.posttrain.roles.reward import RewardWorker
from cosmos_framework.posttrain.roles.reward.video import (
    VideoRewardRequest,
    VideoRewardResult,
    build_video_reward_provider,
)
from cosmos_framework.posttrain.roles.rollout import WFMOnlineRollout
from cosmos_framework.posttrain.roles.trainer import DiffusionTrainer
from cosmos_framework.posttrain.tests.helpers.toy_model import ToyVideoFlowModel


class _Loader:
    def __init__(self) -> None:
        self.start = 0

    def set_start_iteration(self, iteration: int) -> None:
        self.start = iteration

    def __iter__(self) -> Iterator[dict[str, object]]:
        index = self.start
        while True:
            yield {
                "video": [torch.zeros(1, 2)],  # one sample [C,T]
                "ai_caption": [f"ray-prompt-{index}"],
                "x": torch.tensor([[0.3 + index, -0.2]]),  # [B=1,D]
            }
            index += 1


class _Client:
    poll_attempts = 1
    transport_retries = 0

    def score(self, request: VideoRewardRequest, *, deadline_monotonic: float) -> VideoRewardResult:
        del deadline_monotonic
        rank_reward = 1.0 if "_r1_" in request.sample_id else 0.0
        return VideoRewardResult(
            score=rank_reward,
            service_id=f"service-{request.sample_id}",
        )


def setup_online_nft_bodies() -> dict[str, Any]:
    """Build five bodies around one actor-local toy policy."""
    model = ToyVideoFlowModel(dim=2, seed=3)
    trainer = DiffusionTrainer(
        model,
        make_objective(NFTObjectiveConfig()),
        torch.optim.AdamW(model.net.parameters(), lr=1e-3),
        update_old_every_iter=2,
    )
    rollout = WFMOnlineRollout(
        model,
        sampler=object(),
        num_steps=2,
        shift=5.0,
        guidance=3.0,
        group_size=2,
    )
    provider = build_video_reward_provider(
        VideoRewardConfig(provider_revision="test-dance-grpo-v1"),
        _Client(),
        model.video_reward_decoder(),
        run_id="ray-test",
    )
    if rollout.policy is not trainer.model:
        raise RuntimeError("Ray NFT fixture failed to colocate rollout/trainer policy identity")
    return {
        "data": WFMDataEnv(_Loader()),
        "rollout": rollout,
        "rm": RewardWorker(
            [provider],
            RewardCombiner(),
            annotators=[distributed_group_relative(eps=1e-4, group_size=2, correction=1)],
            training_group_size=1,
        ),
        "replay": ExperienceBuffer(exact_batch_size=1),
        "policy": trainer,
    }
