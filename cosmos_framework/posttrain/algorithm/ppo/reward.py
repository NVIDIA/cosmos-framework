# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""PPO reward providers bound through the shared typed provider catalog."""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from cosmos_framework.posttrain.algorithm.ppo.wire import _finite_reward_scalar
from cosmos_framework.posttrain.algorithm.reward import (
    RewardCandidate,
    RewardCandidateResult,
    RewardContext,
    RewardContractError,
    RewardGroupRequest,
    RewardGroupResult,
    RewardProviderCatalog,
    RewardProviderResult,
    RewardProviderSpec,
    map_candidates,
    rule_reward_spec,
)
from cosmos_framework.posttrain.algorithm.reward.contract import _completion_action_token_ids
from cosmos_framework.posttrain.config.rl import PPO_BUILTIN_TERMINALS, PPORewardConfig
from cosmos_framework.posttrain.schema import ObsBundle, Trajectory

PPO_TERMINAL_REWARD_TERM = "ppo_terminal"
PPO_PROCESS_REWARD_TERM = "ppo_process"
PPO_REWARD_PROVIDER_NAMES = ("ppo_terminal", "ppo_process")
_PPO_REWARD_REVISION = "ppo-v2"
_PPO_REWARD_SEMANTIC_IDENTITY = {"request_enrichment": "ppo-context-v1"}


@dataclass(frozen=True)
class PPORewardContext:
    """Frozen legacy callback envelope over one trusted local materialized observation."""

    observation: ObsBundle
    episode_id: str
    policy_version: int
    rendered_prompt: str
    completion_text: str
    action_token_ids: tuple[int, ...]
    terminated: bool
    truncated: bool


@dataclass(frozen=True, kw_only=True)
class _PPORewardGroupRequest(RewardGroupRequest):
    """Canonical shared request plus the validated PPO callback contexts."""

    ppo_contexts: tuple[PPORewardContext, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        contexts = tuple(self.ppo_contexts)
        if len(contexts) != len(self.candidates) or any(
            not isinstance(context, PPORewardContext) for context in contexts
        ):
            raise RewardContractError("PPO reward request needs one typed context per candidate")
        for candidate, context in zip(self.candidates, contexts, strict=True):
            if candidate.trajectory_id != context.episode_id:
                raise RewardContractError("PPO reward request context identity does not match its candidate")
            if candidate.completion != context.completion_text:
                raise RewardContractError("PPO reward request completion does not match its callback context")
            if candidate.completion_token_count != len(context.action_token_ids):
                raise RewardContractError("PPO reward request token count does not match its callback context")
        object.__setattr__(self, "ppo_contexts", contexts)


def terminal_reward(observation: ObsBundle, completion_text: str, config: PPORewardConfig) -> float:
    """Return the built-in terminal scalar without coupling reward logic to rollout."""
    if config.terminal == "none":
        return 0.0
    expected = observation.representations.get("reference")
    match = re.fullmatch(r"\s*([+-]?\d+)\s*", completion_text or "")
    correct = expected is not None and match is not None and int(match.group()) == int(expected)
    return float(config.correct_reward if correct else config.incorrect_reward)


def process_rewards(action_length: int, config: PPORewardConfig) -> tuple[float, ...]:
    """Return exactly one built-in process reward per sampled action token."""
    if isinstance(action_length, bool) or not isinstance(action_length, int) or action_length < 1:
        raise ValueError(f"action_length must be a positive integer, got {action_length!r}")
    if config.process == "none":
        return (0.0,) * action_length
    scale = float(config.process_reward_scale)
    return tuple(scale * (index + 1) / action_length for index in range(action_length))


def _context(trajectory: Trajectory) -> PPORewardContext:
    if not trajectory.transitions:
        raise RewardContractError("PPO reward provider needs a terminal transition")
    transition = trajectory.transitions[-1]
    payload = transition.algo_extra.get("ppo")
    if not isinstance(payload, Mapping):
        raise RewardContractError("PPO reward provider needs transition.algo_extra['ppo'] metadata")
    termination = payload.get("termination")
    if not isinstance(termination, Mapping):
        raise RewardContractError("PPO reward provider needs terminal/truncation metadata")
    policy_version = transition.policy_info.get("weight_version")
    if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
        raise RewardContractError(f"PPO reward provider needs a non-negative policy version, got {policy_version!r}")
    rendered_prompt = payload.get("rendered_prompt")
    if not isinstance(rendered_prompt, str):
        raise RewardContractError("PPO reward provider needs a rendered prompt string")
    completion_text = transition.policy_info.get("completion_text", payload.get("completion_text"))
    if not isinstance(completion_text, str):
        raise RewardContractError("PPO reward provider needs a completion text string")
    terminated = termination.get("terminated")
    truncated = termination.get("truncated")
    if not isinstance(terminated, bool) or not isinstance(truncated, bool) or terminated == truncated:
        raise RewardContractError("PPO reward provider needs exactly one of terminated/truncated")
    return PPORewardContext(
        observation=transition.obs,
        episode_id=trajectory.id,
        policy_version=policy_version,
        rendered_prompt=rendered_prompt,
        completion_text=completion_text,
        action_token_ids=_completion_action_token_ids(trajectory),
        terminated=terminated,
        truncated=truncated,
    )


def _enrich_ppo_reward_request(
    request: RewardGroupRequest,
    trajectories: tuple[Trajectory, ...],
) -> RewardGroupRequest:
    """Attach exact PPO callback contexts through RewardWorker's framework-only seam."""
    contexts = tuple(_context(trajectory) for trajectory in trajectories)
    return _PPORewardGroupRequest(
        group_id=request.group_id,
        candidates=request.candidates,
        context=request.context,
        ppo_contexts=contexts,
    )


class _PPORewardProvider:
    """Consume PPO's trusted enrichment without widening the shared request fields."""

    @staticmethod
    def _contexts(request: RewardGroupRequest) -> tuple[PPORewardContext, ...]:
        if not isinstance(request, _PPORewardGroupRequest):
            raise RewardContractError("PPO reward provider needs its validated PPO group request")
        return request.ppo_contexts


def _load_callable(path: str, role: str) -> Callable[..., Any] | None:
    if not path:
        return None
    module_name, _, attribute_path = path.partition(":")
    try:
        provider: Any = importlib.import_module(module_name)
        for attribute in attribute_path.split("."):
            provider = getattr(provider, attribute)
    except (AssertionError, AttributeError, ImportError, ModuleNotFoundError) as error:
        raise ValueError(f"{role} reward provider {path!r} is not callable") from error
    if not callable(provider):
        raise ValueError(f"{role} reward provider {path!r} is not callable")
    return provider


class _TerminalRewardProvider(_PPORewardProvider):
    _config: PPORewardConfig
    _callable: Callable[..., Any] | None
    _kwargs: dict[str, Any]

    def __init__(self, config: PPORewardConfig) -> None:
        self._config = config
        self._callable = _load_callable(config.terminal_callable, "terminal")
        self._kwargs = dict(config.terminal_kwargs)

    def evaluate(self, request: RewardGroupRequest) -> RewardGroupResult:
        contexts = self._contexts(request)
        return RewardGroupResult(
            group_id=request.group_id,
            items=tuple(
                RewardCandidateResult(
                    trajectory_id=candidate.trajectory_id,
                    result=RewardProviderResult(
                        terms={
                            PPO_TERMINAL_REWARD_TERM: _finite_reward_scalar(
                                terminal_reward(context.observation, context.completion_text, self._config)
                                if self._callable is None
                                else self._callable(context, **self._kwargs),
                                "PPO terminal reward callback",
                            )
                        }
                    ),
                )
                for candidate, context in zip(request.candidates, contexts, strict=True)
            ),
        )


class _ProcessRewardProvider(_PPORewardProvider):
    _config: PPORewardConfig
    _callable: Callable[..., Any] | None
    _kwargs: dict[str, Any]

    def __init__(self, config: PPORewardConfig) -> None:
        self._config = config
        self._callable = _load_callable(config.process_callable, "process")
        self._kwargs = dict(config.process_kwargs)

    def _result(self, context: PPORewardContext) -> RewardProviderResult:
        values: Any = (
            process_rewards(len(context.action_token_ids), self._config)
            if self._callable is None
            else self._callable(context, **self._kwargs)
        )
        if not isinstance(values, Sequence) and callable(getattr(values, "tolist", None)):
            values = values.detach().cpu().tolist() if callable(getattr(values, "detach", None)) else values.tolist()
        return RewardProviderResult(token_terms={PPO_PROCESS_REWARD_TERM: values})

    def evaluate(self, request: RewardGroupRequest) -> RewardGroupResult:
        contexts = self._contexts(request)
        return RewardGroupResult(
            group_id=request.group_id,
            items=tuple(
                RewardCandidateResult(
                    trajectory_id=candidate.trajectory_id,
                    result=self._result(context),
                )
                for candidate, context in zip(request.candidates, contexts, strict=True)
            ),
        )


def _shared_terminal_terms(
    score: Callable[[RewardCandidate, RewardContext], RewardProviderResult],
    config: PPORewardConfig,
    candidate: RewardCandidate,
    context: RewardContext,
) -> RewardProviderResult:
    """Re-emit a shared rule's hit/miss on PPO's reward scale under PPO's terminal term name.

    The term NAME is PPO's invariant, not the rule's: ``seal_ppo_reward_batch`` accepts exactly
    ``{ppo_terminal, ppo_process, final}`` while every shared rule reports ``task``. The SCALE is
    PPO's too -- ``correct_reward``/``incorrect_reward`` stay meaningful for a bound rule.
    """
    hit = float(score(candidate, context).terms["task"])
    value = config.correct_reward if hit else config.incorrect_reward
    return RewardProviderResult(terms={PPO_TERMINAL_REWARD_TERM: _finite_reward_scalar(value, "PPO terminal reward")})


def _terminal_provider_spec(config: PPORewardConfig) -> RewardProviderSpec:
    """Bind PPO's own matcher, or a shared rule reward named by ``reward.terminal``.

    A shared name reuses the registry GRPO selects from, so PPO gains no private reward mechanism
    and inherits the rule's declared ``needs`` -- which is what makes the framework populate
    ``RewardContext.reference`` and what the manifest reads to require ground truth.
    """
    if config.terminal in PPO_BUILTIN_TERMINALS:
        return RewardProviderSpec(
            factory=partial(_TerminalRewardProvider, config),
            revision=_PPO_REWARD_REVISION,
            term_names=(PPO_TERMINAL_REWARD_TERM,),
            token_term_names=(),
            semantic_identity=_PPO_REWARD_SEMANTIC_IDENTITY,
        )
    rule = rule_reward_spec(config.terminal)
    return RewardProviderSpec(
        factory=partial(map_candidates, partial(_shared_terminal_terms, rule.fn, config)),
        revision=_PPO_REWARD_REVISION,
        term_names=(PPO_TERMINAL_REWARD_TERM,),
        token_term_names=(),
        needs=rule.needs,
        semantic_identity={**_PPO_REWARD_SEMANTIC_IDENTITY, "terminal_rule": config.terminal},
    )


def build_ppo_reward_catalog(config: PPORewardConfig) -> RewardProviderCatalog:
    """Return PPO's deterministic provider catalog without constructing providers eagerly."""
    if not isinstance(config, PPORewardConfig):
        raise TypeError(f"PPO reward catalog needs PPORewardConfig, got {type(config).__name__}")
    frozen_config = PPORewardConfig(
        terminal=config.terminal,
        process=config.process,
        correct_reward=float(config.correct_reward),
        incorrect_reward=float(config.incorrect_reward),
        process_reward_scale=float(config.process_reward_scale),
        terminal_callable=config.terminal_callable,
        process_callable=config.process_callable,
        terminal_kwargs=dict(config.terminal_kwargs),
        process_kwargs=dict(config.process_kwargs),
    )
    return RewardProviderCatalog(
        {
            PPO_REWARD_PROVIDER_NAMES[0]: _terminal_provider_spec(frozen_config),
            PPO_REWARD_PROVIDER_NAMES[1]: RewardProviderSpec(
                factory=partial(_ProcessRewardProvider, frozen_config),
                revision=_PPO_REWARD_REVISION,
                term_names=(),
                token_term_names=(PPO_PROCESS_REWARD_TERM,),
                semantic_identity=_PPO_REWARD_SEMANTIC_IDENTITY,
            ),
        }
    )


__all__ = [
    "PPO_PROCESS_REWARD_TERM",
    "PPO_REWARD_PROVIDER_NAMES",
    "PPO_TERMINAL_REWARD_TERM",
    "PPORewardContext",
    "build_ppo_reward_catalog",
    "process_rewards",
    "terminal_reward",
]
