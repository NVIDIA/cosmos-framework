# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Frozen evaluation-only reference role for synchronous PPO."""

from __future__ import annotations

from typing import Any

import torch

from cosmos_framework.posttrain.algorithm.ppo.views import (
    PPOEpisode,
    ReferenceScore,
    attach_reference_score,
    processor_inputs_for_role,
    trajectory_to_episode,
)
from cosmos_framework.posttrain.model.ppo import (
    episode_model_inputs,
    prepare_deterministic_ppo_forward,
)
from cosmos_framework.posttrain.model.value import (
    extract_causal_logits,
    forward_causal_model,
    model_device,
    response_action_logits,
    sampled_action_logprobs,
)
from cosmos_framework.posttrain.schema import Trajectory
from cosmos_framework.posttrain.utils.dist import run_rank_converged


class PPOReferenceRole:
    """Own a frozen reference model and return aligned fixed policy targets."""

    optimizer: None

    def __init__(self, model: Any, *, top_k: int = 128, temperature: float = 1.0) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
        if not 0.01 <= float(temperature) < float("inf"):
            raise ValueError(f"temperature must be finite and >= 0.01 to match vLLM, got {temperature}")
        self.model = model
        self.top_k = top_k
        self.temperature = float(temperature)
        self.optimizer = None
        net = getattr(model, "net", model)
        net.requires_grad_(False)
        net.eval()

    def score(
        self,
        trajectories: list[Trajectory],
    ) -> tuple[Trajectory, ...]:
        """Score rollout trajectories and return reference overlays."""
        if not trajectories:
            raise ValueError("reference score received an empty episode batch")
        if any(not isinstance(item, Trajectory) for item in trajectories):
            raise TypeError("reference score accepts only Trajectory values")
        rollout_trajectories = tuple(trajectories)
        episodes = tuple(trajectory_to_episode(item) for item in rollout_trajectories)
        inputs_by_episode = {item.id: processor_inputs_for_role(item, "reference") for item in rollout_trajectories}
        prepare_deterministic_ppo_forward(self.model)
        device = model_device(self.model)
        prepared_inputs = run_rank_converged(
            lambda: tuple(episode_model_inputs(episode, device, inputs_by_episode) for episode in episodes),
            "reference input preparation",
        )
        results: list[ReferenceScore] = []
        with torch.inference_mode():
            for episode_index, (episode, inputs) in enumerate(zip(episodes, prepared_inputs, strict=True)):
                output = forward_causal_model(self.model, inputs)
                results.append(
                    run_rank_converged(
                        lambda episode=episode, output=output: self._score_episode(
                            episode,
                            extract_causal_logits(output),
                            device,
                        ),
                        f"reference scoring episode {episode_index}",
                    )
                )
        return tuple(
            attach_reference_score(trajectory, result)
            for trajectory, result in zip(rollout_trajectories, results, strict=True)
        )

    def _score_episode(
        self,
        episode: PPOEpisode,
        logits: torch.Tensor,  # [1,P+T,V]
        device: torch.device,
    ) -> ReferenceScore:
        """Postprocess one completed rank-local model forward under convergence."""
        action_logits = response_action_logits(
            logits,
            len(episode.prompt_token_ids),
            len(episode.action_token_ids),
        )  # [1,T,V]
        action_ids = torch.tensor([episode.action_token_ids], dtype=torch.long, device=device)  # [1,T]
        action_logprobs = sampled_action_logprobs(
            action_logits,
            action_ids,
            temperature=self.temperature,
        )[0]  # [T]
        support_size = min(self.top_k, action_logits.shape[-1])
        support_ids = torch.topk(action_logits[0], k=support_size, dim=-1, sorted=True).indices  # [T,K]
        if not bool(torch.isfinite(action_logprobs).all().item()):
            raise ValueError(f"reference produced non-finite action logprobs for {episode.episode_id!r}")
        support_id_rows: list[tuple[int, ...]] = []
        support_probability_rows: list[tuple[float, ...]] = []
        residual_rows: list[float] = []
        reference_ids = support_ids.cpu().tolist()
        behavior_ids = episode.behavior_topk_token_ids or ((),) * len(episode.action_token_ids)
        for token_index, (behavior_row, reference_row) in enumerate(zip(behavior_ids, reference_ids)):
            union_ids = tuple(dict.fromkeys((*behavior_row, *reference_row)))
            if any(token_id >= action_logits.shape[-1] for token_id in union_ids):
                raise ValueError(f"behavior top-k token ID exceeds the reference vocabulary for {episode.episode_id!r}")
            row_logits = action_logits[0, token_index].detach().to(dtype=torch.float64)  # [V]
            row_logits = row_logits / self.temperature  # [V]
            log_normalizer = torch.logsumexp(row_logits, dim=0)  # []
            if not bool(torch.isfinite(log_normalizer).item()):
                raise ValueError(f"reference produced non-finite probabilities for {episode.episode_id!r}")
            union_tensor = torch.tensor(union_ids, dtype=torch.long, device=device)  # [K_union]
            union_log_probabilities = row_logits.gather(0, union_tensor) - log_normalizer  # [K_union]
            union_probabilities = union_log_probabilities.exp()  # [K_union]
            excluded_logits = row_logits.clone()  # [V]
            excluded_logits[union_tensor] = -torch.inf  # [V]
            residual_log_probability = torch.logsumexp(excluded_logits, dim=0) - log_normalizer  # []
            residual = residual_log_probability.exp()  # []
            support_id_rows.append(union_ids)
            support_probability_rows.append(tuple(float(value) for value in union_probabilities.cpu().tolist()))
            residual_rows.append(float(residual.cpu()))
        return ReferenceScore(
            episode_id=episode.episode_id,
            policy_version=episode.policy_version,
            rendered_prompt=episode.rendered_prompt,
            prompt_token_ids=episode.prompt_token_ids,
            action_token_ids=episode.action_token_ids,
            action_mask=episode.action_mask,
            media=episode.media,
            action_logprobs=tuple(float(value) for value in action_logprobs.cpu().tolist()),
            support_token_ids=tuple(support_id_rows),
            support_probabilities=tuple(support_probability_rows),
            residual_probabilities=tuple(residual_rows),
        )


__all__ = ["PPOReferenceRole"]
