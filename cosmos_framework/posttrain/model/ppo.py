# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Model-input and forward-mode helpers shared by PPO roles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from cosmos_framework.posttrain.algorithm.ppo.views import PPOEpisode

_PPO_FORCE_TRAIN_FORWARD_MODULES = frozenset(
    {
        "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextExperts",
    }
)


def selected_transition_logprobs(
    sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, *, score_horizon: int
) -> torch.Tensor:  # [B,H,D] -> [B]
    """Score the executed seven-coordinate prefix in native precision before FP32 reduction."""
    deterministic = std == 0
    safe_std = torch.where(deterministic, torch.ones_like(std), std)
    constant = -torch.log(safe_std) - 0.5 * torch.log(2 * torch.pi * torch.ones_like(sample))
    exponent = -0.5 * torch.pow((sample - mean) / safe_std, 2)
    cells = torch.where(deterministic, torch.zeros_like(sample), constant + exponent)
    return cells[:, :score_horizon, :7].float().sum(dim=(1, 2))


def prepare_deterministic_ppo_forward(
    model: Any,
    *,
    enable_gradient_checkpointing: bool = False,
) -> None:
    """Select deterministic role-forward modes without changing gradient state.

    PPO compares against immutable behavior scores and reuses one batch for
    several epochs, so train-mode dropout would corrupt the ratios. A blanket
    ``eval()`` is also wrong for modules whose training flag selects a sparse,
    deterministic compute path. Start in evaluation mode, zero scalar
    functional-dropout rates, then re-enable only exact compatible compute-mode
    modules and optional HF gradient-checkpointing containers without changing
    their stochastic child modules back to train mode.
    """
    net = getattr(model, "net", model)
    if not isinstance(net, torch.nn.Module):
        raise TypeError(f"PPO model must expose a torch.nn.Module, got {type(net).__name__}")
    net.eval()
    for module in net.modules():
        for attribute, value in tuple(vars(module).items()):
            normalized = attribute.lower()
            if (
                ("dropout" in normalized or normalized.endswith("pdrop"))
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                setattr(module, attribute, 0.0)
        module_type = type(module)
        qualified_type = f"{module_type.__module__}.{module_type.__qualname__}"
        if qualified_type in _PPO_FORCE_TRAIN_FORWARD_MODULES or (
            enable_gradient_checkpointing and getattr(module, "gradient_checkpointing", False) is True
        ):
            module.training = True


def prepare_deterministic_ppo_update(model: Any) -> None:
    """Prepare a deterministic PPO forward with gradient checkpointing enabled."""
    prepare_deterministic_ppo_forward(model, enable_gradient_checkpointing=True)


def episode_model_inputs(
    episode: PPOEpisode,
    device: torch.device,
    inputs_by_episode: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Materialize one role input and verify its exact prompt/action token identity."""
    expected_ids = episode.prompt_token_ids + episode.action_token_ids
    if inputs_by_episode is None:
        if episode.media:
            raise ValueError(
                f"multimodal episode {episode.episode_id!r} requires explicit processor inputs; "
                "text-only fallback would drop its ordered media"
            )
        return {"input_ids": torch.tensor([expected_ids], dtype=torch.long, device=device)}
    if episode.episode_id not in inputs_by_episode:
        raise ValueError(f"missing processor inputs for episode {episode.episode_id!r}")
    raw_inputs = inputs_by_episode[episode.episode_id]
    if "input_ids" not in raw_inputs:
        raise ValueError(f"processor inputs for episode {episode.episode_id!r} do not contain input_ids")
    input_ids = raw_inputs["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("processor input_ids must be a torch.Tensor")
    expected = torch.tensor([expected_ids], dtype=torch.long, device=input_ids.device)  # [1,P+T]
    if input_ids.shape != expected.shape or not torch.equal(input_ids.to(dtype=torch.long), expected):
        raise ValueError(f"processor input_ids mismatch for episode {episode.episode_id!r}")
    return {
        key: value.to(device=device) if isinstance(value, torch.Tensor) else value for key, value in raw_inputs.items()
    }


__all__ = [
    "episode_model_inputs",
    "prepare_deterministic_ppo_forward",
    "prepare_deterministic_ppo_update",
    "selected_transition_logprobs",
]
