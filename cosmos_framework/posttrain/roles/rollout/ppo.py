# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Strict synchronous PPO collection on top of the live vLLM weight receiver.

The adapter deliberately keeps vLLM generation separate from policy/reference/
critic scoring. It records the immutable behavior-policy token identities and
log-probabilities, then builds three equivalent Hugging Face input mappings for
the scoring and update phases.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from cosmos_framework.posttrain.algorithm.ppo.views import (
    PPOEpisode,
    batch_digest,
    episode_to_trajectory,
)
from cosmos_framework.posttrain.data.packer.base import DataPacker
from cosmos_framework.posttrain.roles.rollout.sampler_head import SamplerHeadSpec, parse_ranked_rows
from cosmos_framework.posttrain.roles.rollout.vllm import VLLMRollout
from cosmos_framework.posttrain.schema import ObsBundle, Trajectory

_PPO_SAMPLING_KEYS: frozenset[str] = frozenset({"temperature", "top_p", "top_k", "repetition_penalty", "max_tokens"})
_PPO_MAX_SAMPLING_SEED: int = (1 << 63) - 1


def _ppo_prompt_sampling_seed(base_seed: int, policy_version: int, episode_id: str) -> int:
    """Derive a stable vLLM request seed from exact on-policy identity."""
    digest = hashlib.sha256(b"cosmos3-posttrain-ppo-sampling-v1")
    for component in (str(base_seed).encode("ascii"), str(policy_version).encode("ascii"), episode_id.encode("utf-8")):
        digest.update(len(component).to_bytes(8, byteorder="big"))
        digest.update(component)
    return int.from_bytes(digest.digest()[:8], byteorder="big") & _PPO_MAX_SAMPLING_SEED


def _clone_mapping(inputs: Mapping[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for name, value in inputs.items():
        cloned[name] = value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
    return cloned


@dataclass(frozen=True)
class _PreparedPrompt:
    observation: ObsBundle
    rendered_prompt: str
    prompt_token_ids: tuple[int, ...]
    vllm_input: str | dict[str, Any]


def _exact_token_ids(values: Any, name: str) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of exact integer token IDs")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError(f"{name} must contain non-negative exact integer token IDs")
    return tuple(values)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class PPOVLLMRollout(VLLMRollout):
    """Collect one completion per prompt with immutable PPO behavior statistics.

    The vLLM engine must be constructed with ``processed_logprobs`` so the
    returned behavior scores include the shared sampling temperature, and with
    ``max_logprobs >= behavior_top_k``. Those are engine-level settings rather
    than ``SamplingParams`` fields.
    """

    seed: int | None

    def __init__(
        self,
        model: Any,
        *,
        packer: DataPacker,
        sampling: dict[str, Any] | None = None,
        behavior_top_k: int = 128,
        seed: int | None = None,
        engine_config: dict[str, Any] | None = None,
    ) -> None:
        normalized_sampling: dict[str, Any] = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "max_tokens": 16,
        }
        if sampling is not None:
            unexpected = sorted((key for key in sampling if key not in _PPO_SAMPLING_KEYS), key=repr)
            if unexpected:
                raise ValueError(f"unexpected PPO sampling keys {unexpected!r}")
            normalized_sampling.update(sampling)
        temperature = normalized_sampling.get("temperature")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or float(temperature) < 0.01
        ):
            raise ValueError(f"PPO sampling temperature must be finite and >= 0.01 to match vLLM, got {temperature!r}")
        top_p = normalized_sampling.get("top_p")
        if (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not math.isfinite(float(top_p))
            or float(top_p) != 1.0
        ):
            raise ValueError(f"PPO sampling top_p must equal 1.0, got {top_p!r}")
        top_k = normalized_sampling.get("top_k")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k > 0:
            raise ValueError(f"PPO sampling top_k must be a non-positive integer (disabled), got {top_k!r}")
        normalized_sampling["top_k"] = -1
        repetition_penalty = normalized_sampling.get("repetition_penalty")
        if (
            isinstance(repetition_penalty, bool)
            or not isinstance(repetition_penalty, (int, float))
            or not math.isfinite(float(repetition_penalty))
            or float(repetition_penalty) != 1.0
        ):
            raise ValueError(f"PPO sampling repetition_penalty must equal 1.0 (disabled), got {repetition_penalty!r}")
        max_tokens = normalized_sampling.get("max_tokens")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError(f"PPO sampling max_tokens must be a positive integer, got {max_tokens!r}")
        if isinstance(behavior_top_k, bool) or not isinstance(behavior_top_k, int) or behavior_top_k < 1:
            raise ValueError(f"behavior_top_k must be a positive integer, got {behavior_top_k!r}")
        if seed is not None and type(seed) is not int:
            raise ValueError(f"PPO sampling seed must be an exact integer or None, got {seed!r}")
        # The behavior head is the sampling distribution only under processed logprobs with a wide enough
        # max_logprobs: refuse construction unless the engine identity proves it (shared contract with GRPO's
        # score-centering head; `roles/rollout/sampler_head.py`).
        SamplerHeadSpec(behavior_top_k).validate_engine_config(engine_config, owner="PPOVLLMRollout(behavior_top_k)")
        super().__init__(model, packer=packer, sampling=normalized_sampling, engine_config=engine_config)
        self.behavior_top_k = behavior_top_k
        self.seed = seed

    def generate(
        self,
        prompts: list[ObsBundle],
        *,
        policy_version: int,
    ) -> list[Trajectory]:
        """Collect a complete, synchronous ``n=1`` on-policy batch.

        The live engine's stamped weight version must exactly match the explicit
        controller version. Any malformed request aborts the whole call; partial
        batches are never returned.
        """

        def prepare() -> tuple[tuple[list[_PreparedPrompt], int], list[Any], Any, dict[str, Any]]:
            from vllm import SamplingParams

            if not prompts:
                raise ValueError("PPO rollout collection received an empty prompt batch")
            if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
                raise ValueError(f"policy_version must be a non-negative integer, got {policy_version!r}")
            if policy_version != self.weight_version():
                raise ValueError(
                    f"stale rollout weights: engine holds policy_version {self.weight_version()}, "
                    f"collection requested {policy_version}"
                )
            episode_ids = [prompt.id for prompt in prompts]
            if any(not isinstance(episode_id, str) or not episode_id for episode_id in episode_ids):
                raise ValueError("every PPO prompt must carry a non-empty string episode_id in ObsBundle.id")
            if len(set(episode_ids)) != len(episode_ids):
                raise ValueError("duplicate episode_id in PPO prompt batch")

            prepared = [self._prepare_prompt(prompt) for prompt in prompts]
            requests = [prompt.vllm_input for prompt in prepared]
            seeds = (
                [None] * len(episode_ids)
                if self.seed is None
                else [_ppo_prompt_sampling_seed(self.seed, policy_version, episode_id) for episode_id in episode_ids]
            )
            if self.seed is None:
                params: Any = SamplingParams(n=1, logprobs=self.behavior_top_k, **self.sampling)
            else:
                params = [
                    SamplingParams(
                        n=1,
                        logprobs=self.behavior_top_k,
                        seed=seed,
                        **self.sampling,
                    )
                    for seed in seeds
                ]
            identity = self._training_request_identity(prompts, requests, dict(self.sampling), 1)
            identity.update(
                {
                    "policy_version": policy_version,
                    "behavior_top_k": self.behavior_top_k,
                    "seeds": seeds,
                }
            )
            return (prepared, policy_version), requests, params, identity

        def parse(state: tuple[list[_PreparedPrompt], int], request_outputs: list[Any]) -> list[Trajectory]:
            prepared, version = state
            if (
                not isinstance(request_outputs, Sequence)
                or isinstance(request_outputs, (str, bytes))
                or len(request_outputs) != len(prepared)
            ):
                actual = None if not isinstance(request_outputs, Sequence) else len(request_outputs)
                raise RuntimeError(
                    f"vLLM returned {actual} request outputs for {len(prepared)} PPO prompts; "
                    "partial batches are not allowed"
                )
            return [
                self._collect_request(prompt, request, version)
                for prompt, request in zip(prepared, request_outputs, strict=True)
            ]

        return self._execute_training_generation(
            prepare,
            parse=parse,
            output_identity=lambda trajectories: {"batch_digest": batch_digest(trajectories)},
            context="PPO vLLM training",
        )

    def _prepare_prompt(self, observation: ObsBundle) -> _PreparedPrompt:
        rollout_input = self.packer.get_rollout_input(observation)
        if isinstance(rollout_input, str):
            rendered_prompt = rollout_input
            prompt_token_ids: tuple[int, ...] = ()
            vllm_input: str | dict[str, Any] = rollout_input
        elif isinstance(rollout_input, Mapping):
            rendered_prompt = rollout_input.get("prompt")
            if not isinstance(rendered_prompt, str):
                raise TypeError("multimodal rollout input must contain a rendered prompt")
            raw_prompt_ids = rollout_input.get("prompt_token_ids")
            prompt_token_ids = () if raw_prompt_ids is None else _exact_token_ids(raw_prompt_ids, "prompt token IDs")
            vllm_input = _clone_mapping(rollout_input)
        else:
            raise TypeError("PPO DataPacker rollout input must be a string or mapping")
        # vLLM's request output is the source of truth for text prompt IDs. In
        # particular, a Llama tokenizer may add BOS inside vLLM even though the
        # packer's add_special_tokens=False probe does not. Scoring the returned
        # request IDs preserves exact behavior-policy identity for both cases.
        return _PreparedPrompt(
            observation=observation,
            rendered_prompt=rendered_prompt,
            prompt_token_ids=prompt_token_ids,
            vllm_input=vllm_input,
        )

    def _collect_request(
        self,
        prompt: _PreparedPrompt,
        request: Any,
        policy_version: int,
    ) -> Trajectory:
        request_prompt = getattr(request, "prompt", None)
        if not isinstance(request_prompt, str):
            raise TypeError(f"vLLM returned a non-string prompt for episode {prompt.observation.id!r}")
        if request_prompt != prompt.rendered_prompt:
            raise ValueError(
                f"vLLM rendered prompt mismatch for episode {prompt.observation.id!r}: "
                f"expected {prompt.rendered_prompt!r}, got {request_prompt!r}"
            )
        request_prompt_ids = _exact_token_ids(
            getattr(request, "prompt_token_ids", None),
            "vLLM prompt token IDs",
        )
        if not request_prompt_ids:
            raise ValueError(f"vLLM returned empty prompt token IDs for episode {prompt.observation.id!r}")
        if prompt.prompt_token_ids and request_prompt_ids != prompt.prompt_token_ids:
            raise ValueError(f"vLLM prompt token identity mismatch for episode {prompt.observation.id!r}")
        outputs = getattr(request, "outputs", None)
        if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)) or len(outputs) != 1:
            actual = None if not isinstance(outputs, Sequence) else len(outputs)
            raise RuntimeError(
                f"vLLM must return exactly one completion (n=1) for episode {prompt.observation.id!r}, got {actual}"
            )
        output = outputs[0]
        action_token_ids = _exact_token_ids(getattr(output, "token_ids", None), "sampled action token IDs")
        if not action_token_ids:
            raise ValueError(f"vLLM returned an empty action for episode {prompt.observation.id!r}")
        completion_text = getattr(output, "text", None)
        if not isinstance(completion_text, str):
            raise TypeError("vLLM completion text must be a string")
        terminated, truncated = self._termination(output)
        old_logprobs, behavior_topk_token_ids = self._behavior_scores(output, action_token_ids)
        policy_inputs, reference_inputs, critic_inputs = self._role_inputs(
            prompt,
            request_prompt_ids,
            action_token_ids,
            old_logprobs,
        )
        episode = PPOEpisode(
            episode_id=prompt.observation.id,
            policy_version=policy_version,
            rendered_prompt=prompt.rendered_prompt,
            prompt_token_ids=request_prompt_ids,
            action_token_ids=action_token_ids,
            action_mask=(True,) * len(action_token_ids),
            terminated=terminated,
            truncated=truncated,
            old_logprobs=old_logprobs,
            media=(),
            terminal_reward=0.0,
            process_rewards=(0.0,) * len(action_token_ids),
            behavior_topk_token_ids=behavior_topk_token_ids,
        )
        return episode_to_trajectory(
            episode,
            observation=prompt.observation,
            processor_inputs_by_role={
                "policy": policy_inputs,
                "reference": reference_inputs,
                "critic": critic_inputs,
            },
            completion_text=completion_text,
        )

    @staticmethod
    def _termination(output: Any) -> tuple[bool, bool]:
        finish_reason = getattr(output, "finish_reason", None)
        if finish_reason == "stop":
            return True, False
        if finish_reason == "length":
            return False, True
        raise ValueError(
            f"vLLM completion has ambiguous finish_reason {finish_reason!r}; "
            "expected 'stop' (terminal) or 'length' (truncated)"
        )

    def _behavior_scores(
        self,
        output: Any,
        action_token_ids: tuple[int, ...],
    ) -> tuple[tuple[float, ...], tuple[tuple[int, ...], ...]]:
        """The sampled-action logprobs and the rank-ordered behavior head per action token.

        The vLLM ranked-logprob protocol (row count == action length, mapping rows keyed by non-negative
        integer ids, positive integer ranks, exactly the ranks 1..k once each, the sampled token's own
        finite entry present) is `roles/rollout/sampler_head.py::parse_ranked_rows`, shared with GRPO's
        score-centering head; a violation raises `SamplerHeadProtocolError` (a ValueError).
        """
        rows = getattr(output, "logprobs", None)
        ids_rows, _lp_rows, sampled = parse_ranked_rows(rows, self.behavior_top_k, sampled_token_ids=action_token_ids)
        assert sampled is not None
        return tuple(sampled), tuple(tuple(row) for row in ids_rows)

    def _role_inputs(
        self,
        prompt: _PreparedPrompt,
        prompt_token_ids: tuple[int, ...],
        action_token_ids: tuple[int, ...],
        old_logprobs: tuple[float, ...],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        del old_logprobs
        sample = self.packer.get_policy_input(
            prompt.observation,
            list(action_token_ids),
            prompt_ids=list(prompt_token_ids),
        )
        max_length = self.packer.policy_compute_max_len([sample])
        inputs = self.packer.policy_collate_fn([sample], max_length)
        return _clone_mapping(inputs), _clone_mapping(inputs), _clone_mapping(inputs)


__all__ = ["PPOVLLMRollout"]
