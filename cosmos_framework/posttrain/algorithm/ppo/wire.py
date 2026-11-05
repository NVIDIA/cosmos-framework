# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Strict PPO wire conversions behind the package's public algorithm-local views.

PPO worker groups exchange only ``Trajectory`` objects.  These frozen records are
worker-local validation views: they make token identity and lifecycle invariants
convenient without creating a second transport schema.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
import re
import struct
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import torch

from cosmos_framework.posttrain.algorithm.ppo.records import (
    CriticScore,
    MediaIdentity,
    PPOCriticTargetReceipt,
    PPOEpisode,
    PPOPreparedBatchReceipt,
    PPOTrainingEpisode,
    ReferenceScore,
    ScoredPPOEpisode,
    validate_critic_score,
    validate_reference_score,
)
from cosmos_framework.posttrain.algorithm.reward.contract import RewardContractError, _completion_action_token_ids
from cosmos_framework.posttrain.schema.trajectory import FINAL_REWARD_KEY, ObsBundle, Status, Trajectory, Transition

PPO_WIRE_SCHEMA_VERSION = 1
PPO_PROCESS_REWARD_KEY = "ppo/process"
PPO_ALGO_KEY = "ppo"
_PROCESSOR_ROLES = ("policy", "reference", "critic")
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _digest_parts(tag: str, parts: list[bytes | memoryview]) -> list[bytes | memoryview]:
    encoded_tag = tag.encode("utf-8")
    size = sum(len(part) for part in parts)
    header = len(encoded_tag).to_bytes(4, "big") + encoded_tag + size.to_bytes(8, "big")
    return [header, *parts]


def _canonical_parts(value: Any) -> list[bytes | memoryview]:
    """Frame the existing wire bytes without concatenating tensor payloads.

    This deliberately has no ``repr`` or pickle fallback: a new processor value
    type must define stable byte semantics before it can influence PPO identity.
    Buffer views retain their owners only for this digest call; there is no
    identity cache that could hide a subsequent mutation of the input.
    """
    if value is None:
        return _digest_parts("none", [])
    if type(value) is bool:
        return _digest_parts("bool", [b"1" if value else b"0"])
    if type(value) is int:
        return _digest_parts("int", [str(value).encode("ascii")])
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical PPO digest rejects non-finite floats")
        return _digest_parts("float64", [struct.pack(">d", value)])
    if isinstance(value, str):
        return _digest_parts("str", [value.encode("utf-8")])
    if isinstance(value, bytes):
        return _digest_parts("bytes", [value])
    if isinstance(value, bytearray):
        return _digest_parts("bytearray", [bytes(value)])
    if isinstance(value, memoryview):
        return _digest_parts("memoryview", [value.tobytes()])
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        metadata = _canonical_parts(str(tensor.dtype)) + _canonical_parts(tuple(tensor.shape))
        try:
            array = tensor.numpy()
        except (TypeError, RuntimeError):
            array = tensor.view(torch.uint8).numpy()
        raw = memoryview(array.reshape(-1)).cast("B")
        return _digest_parts("torch-tensor", metadata + _digest_parts("data", [raw]))
    # NumPy and PIL are optional runtime dependencies; detect their stable public
    # surface without importing them into the shared schema layer.
    module = type(value).__module__
    if module.startswith("numpy") and all(hasattr(value, name) for name in ("dtype", "shape", "tobytes")):
        contiguous = value
        try:
            import numpy as np

            contiguous = np.ascontiguousarray(value)
        except ImportError:  # pragma: no cover - a numpy object implies numpy is importable
            pass
        try:
            raw = memoryview(contiguous.reshape(-1)).cast("B")
        except (TypeError, ValueError):
            # NumPy supports dtypes (e.g. datetime64) without a buffer format.
            raw = contiguous.tobytes(order="C")
        payload = (
            _canonical_parts(str(contiguous.dtype))
            + _canonical_parts(tuple(int(dim) for dim in contiguous.shape))
            + _digest_parts("data", [raw])
        )
        return _digest_parts("numpy-array", payload)
    if module.startswith("PIL.") and all(hasattr(value, name) for name in ("mode", "size", "tobytes")):
        payload = (
            _canonical_parts(value.mode)
            + _canonical_parts(tuple(value.size))
            + _digest_parts("data", [value.tobytes()])
        )
        return _digest_parts("pil-image", payload)
    if isinstance(value, Mapping):
        entries = [
            (
                _canonical_bytes(key),
                _canonical_parts(item),
            )
            for key, item in value.items()
        ]
        entries.sort(key=lambda pair: pair[0])
        return _digest_parts(
            "mapping", [part for key, item in entries for part in _digest_parts("entry", [key, *item])]
        )
    if isinstance(value, tuple):
        return _digest_parts(
            "tuple",
            [part for item in value for part in _digest_parts("item", _canonical_parts(item))],
        )
    if isinstance(value, list):
        return _digest_parts(
            "list",
            [part for item in value for part in _digest_parts("item", _canonical_parts(item))],
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        identity = f"{type(value).__module__}.{type(value).__qualname__}"
        payload = _canonical_parts(identity) + [
            part
            for field in dataclasses.fields(value)
            for part in _digest_parts(field.name, _canonical_parts(getattr(value, field.name)))
        ]
        return _digest_parts("dataclass", payload)
    raise TypeError(f"unsupported PPO canonical digest type {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    """Materialize canonical keys for the existing lexicographic mapping order."""
    return b"".join(_canonical_parts(value))


def canonical_digest(value: Any) -> str:
    """Return one type-tagged SHA256 digest for a supported PPO value graph."""
    digest = hashlib.sha256()
    for part in _canonical_parts(value):
        digest.update(part)
    return "sha256:" + digest.hexdigest()


def _tensor_vector(values: Sequence[Any], *, dtype: torch.dtype) -> torch.Tensor:
    """Encode one numeric wire vector as a tensor eligible for the shared bulk transport."""
    return torch.tensor(tuple(values), dtype=dtype)  # [N]


def _tensor_rows(
    rows: Sequence[Sequence[Any]],
    *,
    dtype: torch.dtype,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Encode rectangular rows densely, with tensor-row fallback for genuinely ragged input."""
    normalized = tuple(tuple(row) for row in rows)
    if not normalized:
        return torch.empty((0, 0), dtype=dtype)  # [0,0]
    if len({len(row) for row in normalized}) == 1:
        return torch.tensor(normalized, dtype=dtype)  # [R,C]
    return tuple(torch.tensor(row, dtype=dtype) for row in normalized)  # each [C_i]


def _wire_vector(value: Any, name: str) -> tuple[Any, ...]:
    """Decode a tensor wire vector while accepting the pre-tensor tuple schema."""
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(f"PPO wire field {name} must be one-dimensional, got shape {tuple(value.shape)!r}")
        decoded = value.detach().cpu().tolist()  # [N]
        return tuple(decoded)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"PPO wire field {name} must be a numeric sequence")
    return tuple(value)


def _wire_rows(value: Any, name: str) -> tuple[tuple[Any, ...], ...]:
    """Decode rectangular or ragged tensor rows into the algorithm's immutable tuple view."""
    if isinstance(value, torch.Tensor):
        if value.ndim != 2:
            raise ValueError(f"PPO wire field {name} must be two-dimensional, got shape {tuple(value.shape)!r}")
        decoded = value.detach().cpu().tolist()  # [R,C]
        return tuple(tuple(row) for row in decoded)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"PPO wire field {name} must be a sequence of numeric rows")
    return tuple(_wire_vector(row, f"{name}[{index}]") for index, row in enumerate(value))


def _exact_processor_mapping(value: Any, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{role} processor payload must be a mapping")
    return value


def _mapping_input_ids(inputs: Mapping[str, Any], role: str) -> tuple[int, ...]:
    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError(f"{role} inputs must carry tensor input_ids")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError(f"{role} input_ids must have shape [1,L] with L > 0")
    return tuple(int(token_id) for token_id in input_ids[0].detach().cpu().tolist())


def _processor_catalog(
    episode: PPOEpisode,
    processor_inputs_by_role: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(processor_inputs_by_role) != set(_PROCESSOR_ROLES):
        raise ValueError(
            f"PPO processor inputs must exactly cover policy/reference/critic, got {sorted(processor_inputs_by_role)!r}"
        )
    expected_ids = (*episode.prompt_token_ids, *episode.action_token_ids)
    catalog: dict[str, Mapping[str, Any]] = {}
    aliases: dict[str, str] = {}
    for role in _PROCESSOR_ROLES:
        inputs = _exact_processor_mapping(processor_inputs_by_role[role], role)
        if _mapping_input_ids(inputs, role) != expected_ids:
            raise ValueError(f"{role} input_ids do not match exact prompt and action token identity")
        digest = canonical_digest(inputs)
        prior = catalog.get(digest)
        if prior is not None and canonical_digest(prior) != digest:
            raise ValueError(f"processor catalog digest collision for {digest}")
        if prior is None:
            catalog[digest] = inputs
        aliases[role] = digest
    return {"catalog": catalog, "roles": aliases}


def episode_to_trajectory(
    episode: PPOEpisode,
    *,
    observation: ObsBundle,
    processor_inputs_by_role: Mapping[str, Mapping[str, Any]],
    completion_text: str = "",
) -> Trajectory:
    """Encode one validated rollout view as the sole shared PPO wire object."""
    if not isinstance(episode, PPOEpisode):
        raise TypeError("episode_to_trajectory requires a PPOEpisode")
    if not isinstance(observation, ObsBundle):
        raise TypeError("PPO rollout observation must be an ObsBundle")
    if observation.id and observation.id != episode.episode_id:
        raise ValueError("PPO observation ID does not match episode_id")
    if not isinstance(completion_text, str):
        raise TypeError("PPO rollout completion_text must be a string")
    processor = _processor_catalog(episode, processor_inputs_by_role)
    processor_alias_digests = dict(processor["roles"])
    rollout_digest = canonical_digest(
        {
            "episode": episode,
            "processor_aliases": processor_alias_digests,
        }
    )
    transition = Transition(
        obs=copy.deepcopy(observation),
        action=episode.action_token_ids,
        policy_info={
            "prompt_ids": _tensor_vector(episode.prompt_token_ids, dtype=torch.long),  # [P]
            "rollout_logprobs": episode.old_logprobs,
            "weight_version": episode.policy_version,
            "behavior_topk_token_ids": _tensor_rows(  # [T,K] or ragged rows
                episode.behavior_topk_token_ids,
                dtype=torch.long,
            ),
            "completion_ids": episode.action_token_ids,
            "completion_token_count": len(episode.action_token_ids),
            "completion_text": completion_text,
        },
        reward={FINAL_REWARD_KEY: episode.terminal_reward, PPO_PROCESS_REWARD_KEY: episode.process_rewards},
        mask=episode.action_mask,
        # A length-cut completion still ENDS the episode, so it is `done`; `truncated` carries the
        # reason, per the `schema.Transition` contract and `embodied.sink.is_terminated`. Reporting
        # `done=episode.terminated` conflated the two and made every truncated rollout fail the
        # shared group-reward contract, which requires a done terminal transition.
        done=bool(episode.terminated or episode.truncated),
        truncated=episode.truncated,
        algo_extra={
            PPO_ALGO_KEY: {
                "schema_version": PPO_WIRE_SCHEMA_VERSION,
                "rendered_prompt": episode.rendered_prompt,
                "termination": {"terminated": episode.terminated, "truncated": episode.truncated},
                "media": episode.media,
                "processor": processor,
                "processor_alias_digests": processor_alias_digests,
                "rollout_digest": rollout_digest,
            }
        },
    )
    trajectory = Trajectory(
        transitions=[transition],
        id=episode.episode_id,
        lineage=(f"ppo-policy-{episode.policy_version}",),
        status=Status.COMPLETED,
        group_id=episode.episode_id,
    )
    transition.algo_extra[PPO_ALGO_KEY]["reward_context_digest"] = _computed_reward_context_digest(trajectory)
    return trajectory


def _ppo_payload(trajectory: Trajectory) -> tuple[Transition, Mapping[str, Any]]:
    if not isinstance(trajectory, Trajectory):
        raise TypeError(f"PPO wire batch must contain Trajectory, got {type(trajectory).__name__}")
    if len(trajectory.transitions) != 1:
        raise ValueError("PPO trajectory must contain exactly one transition")
    transition = trajectory.transitions[0]
    payload = transition.algo_extra.get(PPO_ALGO_KEY)
    if not isinstance(payload, Mapping):
        raise ValueError("PPO trajectory is missing versioned algo_extra metadata")
    if payload.get("schema_version") != PPO_WIRE_SCHEMA_VERSION:
        raise ValueError(f"unsupported PPO trajectory schema_version {payload.get('schema_version')!r}")
    if trajectory.id == "":
        raise ValueError("PPO trajectory ID must be non-empty")
    return transition, payload


def _validated_processor_alias_digests(payload: Mapping[str, Any]) -> Mapping[str, str]:
    aliases = payload.get("processor_alias_digests")
    if not isinstance(aliases, Mapping) or set(aliases) != set(_PROCESSOR_ROLES):
        raise ValueError("PPO trajectory is missing its complete processor alias digests")
    if any(not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None for value in aliases.values()):
        raise ValueError("PPO trajectory contains an invalid processor alias digest")

    processor = payload.get("processor")
    if processor is None:
        return aliases
    if not isinstance(processor, Mapping):
        raise ValueError("PPO trajectory processor payload must be a mapping")
    live_aliases = processor.get("roles")
    catalog = processor.get("catalog")
    if not isinstance(live_aliases, Mapping) or not live_aliases:
        raise ValueError("PPO trajectory processor payload is missing live role aliases")
    if not isinstance(catalog, Mapping):
        raise ValueError("PPO trajectory processor payload is missing its catalog")
    if not set(live_aliases).issubset(_PROCESSOR_ROLES):
        raise ValueError("PPO trajectory processor payload contains an unsupported live role alias")
    for role, digest in live_aliases.items():
        if digest != aliases[role]:
            raise ValueError(f"PPO live processor alias for {role} does not match its stamped digest")
        if digest not in catalog:
            raise ValueError(f"PPO live processor alias for {role} names a missing catalog entry")
    return aliases


def _reward_context_identity(trajectory: Trajectory) -> Mapping[str, Any]:
    """Return every stable input visible to a PPO reward provider."""
    transition, payload = _ppo_payload(trajectory)
    if not isinstance(transition.obs, ObsBundle):
        raise TypeError("PPO reward context observation must be an ObsBundle")
    if not isinstance(trajectory.id, str) or not trajectory.id:
        raise ValueError("PPO reward context episode_id must be a non-empty string")
    policy_version = transition.policy_info.get("weight_version")
    if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
        raise ValueError("PPO reward context policy version must be a non-negative exact integer")
    rendered_prompt = payload.get("rendered_prompt")
    completion_text = transition.policy_info.get("completion_text")
    if not isinstance(rendered_prompt, str) or not isinstance(completion_text, str):
        raise TypeError("PPO reward context prompt and completion text must be strings")
    if "completion_ids" not in transition.policy_info:
        raise ValueError("PPO reward context requires exact inline completion IDs")
    try:
        action_token_ids = _completion_action_token_ids(trajectory)
    except RewardContractError as error:
        raise ValueError("PPO reward context has invalid action/completion IDs") from error
    termination = payload.get("termination")
    if not isinstance(termination, Mapping):
        raise ValueError("PPO reward context is missing termination metadata")
    terminated = termination.get("terminated")
    truncated = termination.get("truncated")
    if not isinstance(terminated, bool) or not isinstance(truncated, bool) or terminated == truncated:
        raise ValueError("PPO reward context requires exactly one of terminated/truncated")
    aliases = _validated_processor_alias_digests(payload)
    media = tuple(payload.get("media", ()))
    if any(not isinstance(identity, MediaIdentity) for identity in media):
        raise TypeError("PPO reward context media identities must be MediaIdentity records")
    return {
        "observation": transition.obs,
        "episode_id": trajectory.id,
        "policy_version": policy_version,
        "rendered_prompt": rendered_prompt,
        "completion_text": completion_text,
        "action_token_ids": action_token_ids,
        "completion_ids": action_token_ids,
        "termination": {"terminated": terminated, "truncated": truncated},
        "processor_aliases": dict(aliases),
        "media": media,
    }


def _computed_reward_context_digest(trajectory: Trajectory) -> str:
    return canonical_digest(_reward_context_identity(trajectory))


def _validated_reward_context_digest(trajectory: Trajectory) -> str:
    _transition, payload = _ppo_payload(trajectory)
    stamped_digest = payload.get("reward_context_digest")
    if not isinstance(stamped_digest, str) or _SHA256_RE.fullmatch(stamped_digest) is None:
        raise ValueError("PPO trajectory is missing its exact reward context digest")
    if _computed_reward_context_digest(trajectory) != stamped_digest:
        raise ValueError("PPO reward context digest does not match the current trajectory payload")
    return stamped_digest


def _stamped_reward_context_digest(trajectory: Trajectory) -> str:
    _transition, payload = _ppo_payload(trajectory)
    stamped_digest = payload.get("reward_context_digest")
    if not isinstance(stamped_digest, str) or _SHA256_RE.fullmatch(stamped_digest) is None:
        raise ValueError("PPO trajectory is missing its exact reward context digest")
    return stamped_digest


def rollout_trajectory_digest(trajectory: Trajectory) -> str:
    """Recompute and validate producer-stamped identity independent of BulkRefs."""
    _transition, payload = _ppo_payload(trajectory)
    stamped_digest = payload.get("rollout_digest")
    if not isinstance(stamped_digest, str) or _SHA256_RE.fullmatch(stamped_digest) is None:
        raise ValueError("PPO trajectory is missing its exact rollout digest")
    _stamped_reward_context_digest(trajectory)
    aliases = _validated_processor_alias_digests(payload)
    episode = trajectory_to_episode(trajectory)
    computed_digest = canonical_digest(
        {
            "episode": episode,
            "processor_aliases": dict(aliases),
        }
    )
    if computed_digest != stamped_digest:
        raise ValueError("PPO rollout digest does not match the current trajectory payload")
    return stamped_digest


def trajectory_to_episode(trajectory: Trajectory) -> PPOEpisode:
    """Decode and strictly validate one rollout ``Trajectory`` as a local view."""
    transition, payload = _ppo_payload(trajectory)
    policy_info = transition.policy_info
    reward = transition.reward
    termination = payload.get("termination")
    if not isinstance(reward, Mapping) or not isinstance(termination, Mapping):
        raise ValueError("PPO trajectory is missing reward or termination metadata")
    return PPOEpisode(
        episode_id=trajectory.id,
        policy_version=policy_info.get("weight_version"),
        rendered_prompt=payload.get("rendered_prompt"),
        prompt_token_ids=_wire_vector(policy_info.get("prompt_ids"), "prompt_ids"),
        action_token_ids=transition.action,
        action_mask=transition.mask,
        terminated=termination.get("terminated"),
        truncated=termination.get("truncated"),
        old_logprobs=policy_info.get("rollout_logprobs"),
        media=payload.get("media", ()),
        terminal_reward=reward.get(FINAL_REWARD_KEY),
        process_rewards=reward.get(PPO_PROCESS_REWARD_KEY, ()),
        behavior_topk_token_ids=_wire_rows(
            policy_info.get("behavior_topk_token_ids", ()),
            "behavior_topk_token_ids",
        ),
    )


def processor_inputs_for_role(trajectory: Trajectory, role: str) -> Mapping[str, Any]:
    """Return one already-materialized processor mapping selected by its alias."""
    if role not in _PROCESSOR_ROLES:
        raise ValueError(f"unsupported PPO processor role {role!r}")
    rollout_trajectory_digest(trajectory)
    _transition, payload = _ppo_payload(trajectory)
    processor = payload.get("processor")
    if not isinstance(processor, Mapping):
        raise ValueError("PPO trajectory is missing its processor catalog")
    aliases = processor.get("roles")
    catalog = processor.get("catalog")
    if not isinstance(aliases, Mapping) or not isinstance(catalog, Mapping) or role not in aliases:
        raise ValueError(f"PPO trajectory has no processor payload for {role}")
    digest = aliases[role]
    if digest not in catalog:
        raise ValueError(f"PPO processor alias for {role} names a missing catalog entry")
    value = catalog[digest]
    inputs = _exact_processor_mapping(value, role)
    if canonical_digest(inputs) != digest:
        raise ValueError(f"{role} processor payload digest mismatch")
    episode = trajectory_to_episode(trajectory)
    expected = (*episode.prompt_token_ids, *episode.action_token_ids)
    if _mapping_input_ids(inputs, role) != expected:
        raise ValueError(f"{role} processor input_ids drifted from trajectory token identity")
    return inputs


def trajectory_to_reference_score(trajectory: Trajectory) -> ReferenceScore:
    """Decode one reference overlay and revalidate its rollout identity."""
    episode = trajectory_to_episode(trajectory)
    _transition, payload = _ppo_payload(trajectory)
    reference = payload.get("reference")
    if not isinstance(reference, Mapping):
        raise ValueError("PPO trajectory is missing its reference overlay")
    score = ReferenceScore(
        episode_id=episode.episode_id,
        policy_version=episode.policy_version,
        rendered_prompt=episode.rendered_prompt,
        prompt_token_ids=episode.prompt_token_ids,
        action_token_ids=episode.action_token_ids,
        action_mask=episode.action_mask,
        media=episode.media,
        action_logprobs=reference.get("action_logprobs"),
        support_token_ids=_wire_rows(reference.get("support_token_ids"), "reference.support_token_ids"),
        support_probabilities=_wire_rows(
            reference.get("support_probabilities"),
            "reference.support_probabilities",
        ),
        residual_probabilities=reference.get("residual_probabilities"),
    )
    # Construction validates score shape; the shared validator catches identity
    # changes against the base trajectory.
    validate_reference_score(episode, score)
    return score


def trajectory_to_critic_score(trajectory: Trajectory) -> CriticScore:
    """Decode one critic overlay and revalidate its rollout identity."""
    episode = trajectory_to_episode(trajectory)
    _transition, payload = _ppo_payload(trajectory)
    critic = payload.get("critic")
    if not isinstance(critic, Mapping):
        raise ValueError("PPO trajectory is missing its critic overlay")
    score = CriticScore(
        episode_id=episode.episode_id,
        policy_version=episode.policy_version,
        rendered_prompt=episode.rendered_prompt,
        prompt_token_ids=episode.prompt_token_ids,
        action_token_ids=episode.action_token_ids,
        action_mask=episode.action_mask,
        media=episode.media,
        values=critic.get("values"),
    )
    validate_critic_score(episode, score)
    return score


def _attach_score_payload(trajectory: Trajectory, name: str, values: Mapping[str, Any]) -> Trajectory:
    """Attach one validated score payload without duplicating Trajectory copy mechanics."""
    transition, payload = _ppo_payload(trajectory)
    copied_payload = copy.deepcopy(dict(payload))
    copied_payload[name] = dict(values)
    return dataclasses.replace(
        trajectory,
        transitions=[
            dataclasses.replace(
                transition,
                policy_info=copy.deepcopy(transition.policy_info),
                reward=copy.deepcopy(transition.reward),
                algo_extra={**copy.deepcopy(transition.algo_extra), PPO_ALGO_KEY: copied_payload},
            )
        ],
        lineage=tuple(trajectory.lineage),
    )


def attach_reference_score(trajectory: Trajectory, score: ReferenceScore) -> Trajectory:
    """Attach one frozen-reference result to a complete PPO trajectory."""
    episode = trajectory_to_episode(trajectory)
    validate_reference_score(episode, score)
    return _attach_score_payload(
        trajectory,
        "reference",
        {
            "action_logprobs": score.action_logprobs,
            "support_token_ids": _tensor_rows(score.support_token_ids, dtype=torch.long),
            "support_probabilities": _tensor_rows(score.support_probabilities, dtype=torch.float64),
            "residual_probabilities": score.residual_probabilities,
        },
    )


def attach_critic_score(trajectory: Trajectory, score: CriticScore) -> Trajectory:
    """Attach one critic result to a complete PPO trajectory."""
    episode = trajectory_to_episode(trajectory)
    validate_critic_score(episode, score)
    return _attach_score_payload(trajectory, "critic", {"values": score.values})


def trajectory_ppo_scores(trajectory: Trajectory) -> tuple[PPOEpisode, ReferenceScore, CriticScore]:
    """Read and validate the PPO scores attached by the reference and critic roles."""
    episode = trajectory_to_episode(trajectory)
    reference = trajectory_to_reference_score(trajectory)
    critic = trajectory_to_critic_score(trajectory)
    return episode, reference, critic


def _finite_reward_scalar(value: Any, name: str) -> float:
    """Normalize one shared reward value without Python bool/string coercion."""
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise TypeError(f"{name} must be a finite non-bool scalar, got shape {tuple(value.shape)!r}")
        value = value.detach().cpu().item()  # []
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-bool scalar, got {value!r}")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite non-bool scalar") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite non-bool scalar, got {value!r}")
    return result


def seal_ppo_reward_batch(
    rollout: Sequence[Trajectory],
    reward: Sequence[Trajectory],
) -> list[Trajectory]:
    """Seal shared reward-worker channels into fresh PPO wire trajectories.

    The reward overlay is intentionally processor-free. Every row is validated before any sealed
    result is returned, so one late identity/schema failure cannot expose a partially scored batch.
    """
    from cosmos_framework.posttrain.algorithm.ppo.reward import (
        PPO_PROCESS_REWARD_TERM,
        PPO_TERMINAL_REWARD_TERM,
    )

    rollout_items = tuple(rollout)
    reward_items = tuple(reward)
    if not rollout_items or len(reward_items) != len(rollout_items):
        raise ValueError("PPO reward overlay must align one row per non-empty rollout batch")
    if tuple(item.id for item in reward_items) != tuple(item.id for item in rollout_items):
        raise ValueError("PPO reward overlay episode IDs/order do not exactly match rollout")

    planned: list[Trajectory] = []
    for base, overlay in zip(rollout_items, reward_items, strict=True):
        if not isinstance(base, Trajectory) or not isinstance(overlay, Trajectory):
            raise TypeError("PPO reward sealing accepts only Trajectory rows")
        base_transition, base_payload = _ppo_payload(base)
        overlay_transition, overlay_payload = _ppo_payload(overlay)
        source_digest = rollout_trajectory_digest(base)
        if overlay_payload.get("rollout_digest") != source_digest:
            raise ValueError("PPO reward overlay rollout digest does not match source identity")
        source_context_digest = _stamped_reward_context_digest(base)
        overlay_context_digest = overlay_payload.get("reward_context_digest")
        if overlay_context_digest != source_context_digest:
            raise ValueError("PPO reward overlay reward context digest does not match source identity")
        if "processor" in overlay_payload:
            raise ValueError("PPO reward overlay must not retain processor payloads")
        if tuple(overlay_transition.action) != tuple(base_transition.action):
            raise ValueError("PPO reward overlay action token identity does not match source")
        if canonical_digest(overlay_transition.policy_info) != canonical_digest(base_transition.policy_info):
            raise ValueError("PPO reward overlay policy identity does not match source")
        reward_mapping = overlay_transition.reward
        expected_names = {PPO_TERMINAL_REWARD_TERM, PPO_PROCESS_REWARD_TERM, FINAL_REWARD_KEY}
        if not isinstance(reward_mapping, Mapping) or set(reward_mapping) != expected_names:
            actual_names = [] if not isinstance(reward_mapping, Mapping) else sorted(reward_mapping)
            raise ValueError(
                f"PPO reward overlay schema must be exactly {sorted(expected_names)!r}, got {actual_names!r}"
            )
        terminal_value = reward_mapping[PPO_TERMINAL_REWARD_TERM]
        final_value = reward_mapping[FINAL_REWARD_KEY]
        terminal_scalar = _finite_reward_scalar(terminal_value, "PPO terminal reward term")
        final_scalar = _finite_reward_scalar(final_value, "PPO shared final reward")
        if final_scalar != terminal_scalar:
            raise ValueError("PPO terminal reward term and shared final reward must be equal finite non-bool scalars")
        process_values = reward_mapping[PPO_PROCESS_REWARD_TERM]
        if isinstance(process_values, (str, bytes)) or not isinstance(process_values, Sequence):
            raise TypeError("PPO process reward term must be a token-aligned sequence")
        process_tuple = tuple(
            _finite_reward_scalar(value, f"PPO process reward term[{index}]")
            for index, value in enumerate(process_values)
        )
        action_ids = tuple(base_transition.action)
        if len(process_tuple) != len(action_ids):
            raise ValueError(
                f"PPO process reward width {len(process_tuple)} must equal action token count {len(action_ids)}"
            )

        copied_payload = copy.deepcopy(dict(base_payload))
        copied_transition = dataclasses.replace(
            base_transition,
            reward={FINAL_REWARD_KEY: final_scalar, PPO_PROCESS_REWARD_KEY: process_tuple},
            policy_info=copy.deepcopy(base_transition.policy_info),
            algo_extra={**copy.deepcopy(base_transition.algo_extra), PPO_ALGO_KEY: copied_payload},
        )
        sealed = dataclasses.replace(base, transitions=[copied_transition], lineage=tuple(base.lineage))
        episode = trajectory_to_episode(sealed)
        aliases = _validated_processor_alias_digests(copied_payload)
        copied_payload["rollout_digest"] = canonical_digest(
            {
                "episode": episode,
                "processor_aliases": dict(aliases),
            }
        )
        planned.append(sealed)
    return planned


def compact_ppo_reward_batch(trajectories: Sequence[Trajectory]) -> list[Trajectory]:
    """Drop observation/media/processor payloads after shared reward evaluation."""
    compacted: list[Trajectory] = []
    for trajectory in trajectories:
        transition, payload = _ppo_payload(trajectory)
        batch_id = payload.get("batch_id")
        rollout_digest = payload.get("rollout_digest")
        reward_context_digest = _validated_reward_context_digest(trajectory)
        if not isinstance(batch_id, str) or _SHA256_RE.fullmatch(batch_id) is None:
            raise ValueError("PPO reward result is missing its source batch_id")
        if not isinstance(rollout_digest, str) or _SHA256_RE.fullmatch(rollout_digest) is None:
            raise ValueError("PPO reward result is missing its source rollout digest")
        compact_payload = {
            "schema_version": PPO_WIRE_SCHEMA_VERSION,
            "batch_id": batch_id,
            "rollout_digest": rollout_digest,
            "reward_context_digest": reward_context_digest,
        }
        compacted.append(
            dataclasses.replace(
                trajectory,
                transitions=[
                    dataclasses.replace(
                        transition,
                        obs=ObsBundle(id=trajectory.id),
                        policy_info=copy.deepcopy(transition.policy_info),
                        reward=copy.deepcopy(transition.reward),
                        algo_extra={PPO_ALGO_KEY: compact_payload},
                    )
                ],
                lineage=tuple(trajectory.lineage),
            )
        )
    return compacted


def validate_ppo_reward_context_batch(trajectories: Sequence[Trajectory]) -> None:
    """Validate materialized provider inputs against their producer stamps before scoring."""
    for trajectory in trajectories:
        _validated_reward_context_digest(trajectory)


def _advantage_channel_tensor(advantages: Sequence[float]) -> torch.Tensor:
    """Encode one episode's advantages for the shared per-token advantage channel.

    WIDTH is the full action length ``[T]`` with zeros at masked positions, not the mask-selected
    ``[N]``: PPO's objective masks inside the loss, so keeping ``[T]`` leaves its arithmetic
    untouched, and ``T`` is exactly the completion-token count the channel documents.

    DTYPE is float64, and that is load-bearing rather than incidental. ``PPOTrainingEpisode``
    advantages are float64 and `_computed_training_target_digest` seals them with ``struct.pack(">d")``
    on the producing rank; the consuming rank recomputes that digest from what it decodes off this
    channel. A float32 channel would round-trip to a different float64 value and break that digest
    against its own stamp -- an identity guarantee lost, not a precision detail. The objective casts
    to float32 itself (`algorithm/ppo/objective.py`), so the loss is unaffected either way.
    """
    return torch.tensor(tuple(advantages), dtype=torch.float64)  # [T]


def _decoded_advantage_channel(transition: Transition, action_length: int, episode_id: str) -> tuple[float, ...]:
    """Decode `Transition.advantage` back into the algorithm's per-action view.

    An absent channel or a width that no longer matches the episode's action tokens is rejected
    instead of being interpreted as a valid training target.
    """
    advantage = transition.advantage
    if advantage is None:
        raise ValueError(f"PPO trajectory {episode_id!r} carries no Transition.advantage")
    if not isinstance(advantage, torch.Tensor):
        raise TypeError(
            f"PPO trajectory {episode_id!r} advantage must be a per-token tensor, got {type(advantage).__name__}"
        )
    if advantage.ndim != 1 or advantage.numel() != action_length:
        raise ValueError(
            f"PPO trajectory {episode_id!r} advantage has shape {tuple(advantage.shape)!r}; "
            f"the channel width must be the full action length [{action_length}]"
        )
    return tuple(float(value) for value in advantage.detach().cpu().tolist())


def training_episode_to_trajectory(base: Trajectory, item: PPOTrainingEpisode) -> Trajectory:
    """Install policy-computed targets as one strict Trajectory overlay."""
    episode = trajectory_to_episode(base)
    if item.scored.episode != episode:
        raise ValueError("PPO target overlay episode identity does not match its rollout")
    transition, payload = _ppo_payload(base)
    copied_payload = copy.deepcopy(dict(payload))
    copied_payload["reference"] = {
        "action_logprobs": item.scored.reference.action_logprobs,
        "support_token_ids": _tensor_rows(item.scored.reference.support_token_ids, dtype=torch.long),
        "support_probabilities": _tensor_rows(item.scored.reference.support_probabilities, dtype=torch.float64),
        "residual_probabilities": item.scored.reference.residual_probabilities,
    }
    copied_payload["critic"] = {"values": item.scored.critic.values}
    # `advantages` rides `Transition.advantage`, the ONE advantage channel GRPO, OPD and PPO share
    # (`schema/trajectory.py::Transition`). What stays in the algorithm payload are the critic's
    # targets and its reporting terms -- `action_rewards`, `returns`, `value_advantages` are not
    # advantages the policy-gradient surrogate reads.
    targets = {
        "ppo": {
            "action_rewards": item.action_rewards,
            "returns": item.returns,
            "value_advantages": item.value_advantages,
        }
    }
    return dataclasses.replace(
        base,
        transitions=[
            dataclasses.replace(
                transition,
                obs=copy.deepcopy(transition.obs),
                policy_info=copy.deepcopy(transition.policy_info),
                reward=copy.deepcopy(transition.reward),
                targets=targets,
                advantage=_advantage_channel_tensor(item.advantages),  # [T]
                algo_extra={**copy.deepcopy(transition.algo_extra), PPO_ALGO_KEY: copied_payload},
            )
        ],
    )


def _computed_training_target_digest(item: PPOTrainingEpisode) -> str:
    return canonical_digest(
        {
            "episode_id": item.scored.episode.episode_id,
            "policy_version": item.scored.episode.policy_version,
            "action_rewards": item.action_rewards,
            "advantages": item.advantages,
            "returns": item.returns,
            "value_advantages": item.value_advantages,
        }
    )


def training_target_digest(value: PPOTrainingEpisode | Trajectory) -> str:
    """Return one processor/reference-independent digest for policy targets."""
    if isinstance(value, PPOTrainingEpisode):
        return _computed_training_target_digest(value)
    if not isinstance(value, Trajectory):
        raise TypeError("PPO training target digest requires PPOTrainingEpisode or Trajectory")
    item = trajectory_to_training_episode(value)
    computed_digest = _computed_training_target_digest(item)
    _transition, payload = _ppo_payload(value)
    stamped_digest = payload.get("training_target_digest")
    if not isinstance(stamped_digest, str) or _SHA256_RE.fullmatch(stamped_digest) is None:
        raise ValueError("PPO trajectory is missing its exact training target digest")
    if computed_digest != stamped_digest:
        raise ValueError("PPO training target digest does not match the current trajectory targets")
    return stamped_digest


def training_target_to_trajectory(base: Trajectory, item: PPOTrainingEpisode) -> Trajectory:
    """Build a processor-free policy target overlay for critic installation."""
    target = training_episode_to_trajectory(base, item)
    transition, payload = _ppo_payload(target)
    copied_payload = copy.deepcopy(dict(payload))
    processor = copied_payload.pop("processor", None)
    if not isinstance(processor, Mapping) or not isinstance(processor.get("roles"), Mapping):
        raise ValueError("policy training target is missing its processor alias")
    source_digest = processor["roles"].get("policy")
    if not isinstance(source_digest, str) or _SHA256_RE.fullmatch(source_digest) is None:
        raise ValueError("policy training target has an invalid processor digest")
    copied_payload["source_processor_digest"] = source_digest
    copied_payload["training_target_digest"] = _computed_training_target_digest(item)
    return dataclasses.replace(
        target,
        transitions=[
            dataclasses.replace(
                transition,
                obs=ObsBundle(id=target.id),
                algo_extra={**copy.deepcopy(transition.algo_extra), PPO_ALGO_KEY: copied_payload},
            )
        ],
    )


def validate_training_target_overlay(
    base: Trajectory,
    target: Trajectory,
    *,
    batch_id: str,
) -> PPOTrainingEpisode:
    """Bind one policy target to the exact cached rollout before critic install."""
    if not isinstance(batch_id, str) or _SHA256_RE.fullmatch(batch_id) is None:
        raise ValueError("PPO training target batch_id must be an exact SHA256 digest")
    _base_transition, base_payload = _ppo_payload(base)
    _target_transition, target_payload = _ppo_payload(target)
    if base.id != target.id:
        raise ValueError("PPO training target episode ID does not match the cached critic input")
    for role, payload in (("critic cache", base_payload), ("policy target", target_payload)):
        if payload.get("batch_id") != batch_id:
            raise ValueError(f"PPO {role} batch_id does not match the active batch")
    if "processor" in target_payload:
        raise ValueError("PPO policy training target must be processor-free")
    aliases = base_payload.get("processor_alias_digests")
    if not isinstance(aliases, Mapping):
        raise ValueError("PPO cached critic input is missing processor alias digests")
    expected_policy_digest = aliases.get("policy")
    if target_payload.get("source_processor_digest") != expected_policy_digest:
        raise ValueError("PPO training target policy processor digest does not match the cached rollout")
    if rollout_trajectory_digest(target) != rollout_trajectory_digest(base):
        raise ValueError("PPO training target rollout digest does not match the cached critic input")
    item = trajectory_to_training_episode(target)
    if trajectory_to_episode(base) != item.scored.episode:
        raise ValueError(f"critic target episode identity mismatch for {base.id!r}")
    training_target_digest(target)
    return item


def trajectory_to_training_episode(trajectory: Trajectory) -> PPOTrainingEpisode:
    """Decode policy targets from a cached Trajectory."""
    episode = trajectory_to_episode(trajectory)
    reference = trajectory_to_reference_score(trajectory)
    critic = trajectory_to_critic_score(trajectory)
    transition, _payload = _ppo_payload(trajectory)
    targets = transition.targets
    if not isinstance(targets, Mapping) or not isinstance(targets.get("ppo"), Mapping):
        raise ValueError("PPO trajectory is missing policy-computed training targets")
    ppo_targets = targets["ppo"]
    if "advantages" in ppo_targets:
        raise ValueError("PPO trajectory carries a private advantage payload; advantages ride Transition.advantage")
    return PPOTrainingEpisode(
        scored=ScoredPPOEpisode(episode=episode, reference=reference, critic=critic),
        action_rewards=ppo_targets.get("action_rewards"),
        advantages=_decoded_advantage_channel(transition, len(episode.action_token_ids), trajectory.id),
        returns=ppo_targets.get("returns"),
        value_advantages=ppo_targets.get("value_advantages", ()),
    )


def prepared_batch_receipt(
    batch_id: str,
    local_training: Sequence[PPOTrainingEpisode],
) -> PPOPreparedBatchReceipt:
    """All-gather local algorithm views and emit identical global metadata on every rank."""
    local = tuple(local_training)
    if not local:
        raise ValueError("cannot summarize an empty local PPO training shard")
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        gathered: list[tuple[PPOTrainingEpisode, ...] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        if any(shard is None for shard in gathered):
            raise RuntimeError("prepared PPO batch all-gather returned an incomplete shard set")
        global_training = tuple(item for shard in gathered if shard is not None for item in shard)
    else:
        global_training = local
    episode_ids = tuple(item.scored.episode.episode_id for item in global_training)
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("prepared PPO batch all-gather duplicated episode IDs")
    policy_versions = {item.scored.episode.policy_version for item in global_training}
    if len(policy_versions) != 1:
        raise ValueError("prepared PPO batch must contain exactly one policy_version")
    from cosmos_framework.posttrain.algorithm.ppo.batch import ppo_training_batch_metrics

    return PPOPreparedBatchReceipt(
        batch_id=batch_id,
        episode_ids=episode_ids,
        policy_version=next(iter(policy_versions)),
        target_digests=tuple(training_target_digest(item) for item in global_training),
        batch_metrics=ppo_training_batch_metrics(global_training),
    )


def project_rollout_for_role(
    trajectories: Sequence[Trajectory],
    role: str,
    *,
    batch_id: str | None = None,
) -> list[Trajectory]:
    """Copy a rollout skeleton while retaining only one processor alias/catalog value."""
    if role not in (*_PROCESSOR_ROLES, "reward"):
        raise ValueError(f"unsupported PPO projection role {role!r}")
    copied_trajectories = tuple(trajectories)
    resolved_batch_id = batch_digest(copied_trajectories) if batch_id is None else batch_id
    if not isinstance(resolved_batch_id, str) or _SHA256_RE.fullmatch(resolved_batch_id) is None:
        raise ValueError("PPO role projection batch_id must be an exact SHA256 digest")
    projected: list[Trajectory] = []
    for trajectory in copied_trajectories:
        transition, payload = _ppo_payload(trajectory)
        if role == "reward":
            rollout_trajectory_digest(trajectory)
            copied_payload = copy.deepcopy(dict(payload))
            copied_payload.pop("processor", None)
            copied_payload["batch_id"] = resolved_batch_id
            copied_transition = dataclasses.replace(
                transition,
                obs=copy.deepcopy(transition.obs),
                policy_info=copy.deepcopy(transition.policy_info),
                reward=copy.deepcopy(transition.reward),
                algo_extra={**copy.deepcopy(transition.algo_extra), PPO_ALGO_KEY: copied_payload},
            )
            projected.append(
                dataclasses.replace(
                    trajectory,
                    transitions=[copied_transition],
                    lineage=tuple(trajectory.lineage),
                )
            )
            continue
        processor = payload.get("processor")
        if not isinstance(processor, Mapping):
            raise ValueError("PPO trajectory is missing its processor catalog")
        aliases = processor.get("roles")
        catalog = processor.get("catalog")
        if not isinstance(aliases, Mapping) or not isinstance(catalog, Mapping) or role not in aliases:
            raise ValueError(f"PPO trajectory has no processor payload for {role}")
        digest = aliases[role]
        if digest not in catalog:
            raise ValueError(f"PPO processor alias for {role} names a missing catalog entry")
        copied_payload = copy.deepcopy(dict(payload))
        copied_payload["batch_id"] = resolved_batch_id
        # Preserve the opaque payload/ref itself while rebuilding every mutable
        # shell.  This keeps one canonical catalog value and avoids mutating the
        # rollout object during three role projections.
        copied_payload["processor"] = {"catalog": {digest: catalog[digest]}, "roles": {role: digest}}
        copied_transition = dataclasses.replace(
            transition,
            obs=ObsBundle(id=trajectory.id),
            policy_info=copy.deepcopy(transition.policy_info),
            reward=copy.deepcopy(transition.reward),
            algo_extra={**copy.deepcopy(transition.algo_extra), PPO_ALGO_KEY: copied_payload},
        )
        projected.append(
            dataclasses.replace(
                trajectory,
                transitions=[copied_transition],
                lineage=tuple(trajectory.lineage),
            )
        )
    return projected


def trajectory_digest(trajectory: Trajectory) -> str:
    """Digest stable PPO identity/targets without processor bulk residence.

    Catalog keys are content digests computed before packing.  Hashing those keys,
    rather than a local/remote ``BulkRef``, keeps the result invariant across
    projection and transport placement.

    ``advantage`` is hashed alongside ``targets`` because the advantage moved OUT of the algorithm
    payload onto the shared `Transition.advantage` channel. Hashing only ``targets`` after that move
    would silently stop covering the one target the policy surrogate actually multiplies.
    """
    transition, payload = _ppo_payload(trajectory)
    processor = payload.get("processor")
    if processor is None:
        source_digest = payload.get("source_processor_digest")
        if not isinstance(source_digest, str) or not source_digest.startswith("sha256:"):
            raise ValueError("PPO score overlay is missing its source processor digest")
        aliases: Mapping[str, Any] = {"source": source_digest}
    else:
        if not isinstance(processor, Mapping):
            raise ValueError("PPO trajectory has an invalid processor catalog")
        aliases = processor.get("roles")
        if not isinstance(aliases, Mapping):
            raise ValueError("PPO trajectory is missing processor role aliases")
    target_metadata = transition.targets
    return canonical_digest(
        {
            "rollout_digest": rollout_trajectory_digest(trajectory),
            "processor_aliases": dict(aliases),
            "reference": payload.get("reference"),
            "critic": payload.get("critic"),
            "targets": target_metadata,
            "advantage": transition.advantage,
        }
    )


def _validated_rollout_digests(trajectories: Sequence[Trajectory]) -> tuple[str, ...]:
    """Return each trajectory's re-verified producer stamp, in the caller's order."""
    copied = tuple(trajectories)
    if not copied:
        raise ValueError("cannot digest an empty PPO batch")
    ids = [trajectory.id for trajectory in copied]
    if any(not isinstance(episode_id, str) or not episode_id for episode_id in ids):
        raise ValueError("PPO batch contains an invalid trajectory ID")
    if len(set(ids)) != len(ids):
        raise ValueError("PPO batch contains duplicate trajectory IDs")
    return tuple(rollout_trajectory_digest(trajectory) for trajectory in copied)


def batch_digest(trajectories: Sequence[Trajectory]) -> str:
    """Digest an ordered rollout batch independently of role projection residence."""
    return canonical_digest(_validated_rollout_digests(trajectories))


def global_batch_digest(local_trajectories: Sequence[Trajectory]) -> str:
    """Digest the COMPLETE batch from one model-DP shard of it.

    ``batch_digest`` names exactly the trajectories it is handed. Under model DP the trainable
    roles are fed disjoint shards (the ``select_model_dp`` ingress on the ``prepare_batch`` /
    ``predict`` edges), so a per-shard digest names a DIFFERENT batch on every rank while every
    other field of the surrounding receipt is all-gathered and therefore global. That mismatch is
    not cosmetic: the controller compares whole receipts with ``resolve_identical`` and then hands
    one rank's ``batch_id`` back to every rank as the key into its own shard cache, so the
    identity has to denote the same batch everywhere.

    Gathering in rank order reproduces the batch exactly, because the ingress split is contiguous
    and ``concat_batches`` preserves producer order: this returns the same value ``batch_digest``
    would return over the unsharded batch. Without a process group it degrades to that call.
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return canonical_digest(_validated_rollout_digests(local_trajectories))
    from cosmos_framework.posttrain.utils.dist import run_rank_converged

    # Converge on the stamp check BEFORE the gather: a payload fault on one rank would otherwise
    # leave its peers blocked in `all_gather_object` until the process group times out.
    local = run_rank_converged(
        lambda: _validated_rollout_digests(local_trajectories),
        "PPO batch identity stamp validation",
    )
    gathered: list[tuple[str, ...] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    if any(shard is None for shard in gathered):
        raise RuntimeError("PPO batch identity all-gather returned an incomplete shard set")
    digests = tuple(digest for shard in gathered if shard is not None for digest in shard)
    if len(set(digests)) != len(digests):
        raise ValueError(
            "PPO batch identity all-gather saw the same trajectory on more than one model-DP rank; "
            "the ingress must hand every rank a disjoint shard"
        )
    return canonical_digest(digests)


__all__ = [
    "CriticScore",
    "MediaIdentity",
    "PPOEpisode",
    "PPOCriticTargetReceipt",
    "PPOPreparedBatchReceipt",
    "PPOTrainingEpisode",
    "ReferenceScore",
    "ScoredPPOEpisode",
    "batch_digest",
    "canonical_digest",
    "compact_ppo_reward_batch",
    "episode_to_trajectory",
    "global_batch_digest",
    "processor_inputs_for_role",
    "prepared_batch_receipt",
    "project_rollout_for_role",
    "rollout_trajectory_digest",
    "seal_ppo_reward_batch",
    "trajectory_digest",
    "trajectory_to_critic_score",
    "trajectory_to_episode",
    "trajectory_to_reference_score",
    "trajectory_to_training_episode",
    "training_episode_to_trajectory",
    "training_target_digest",
    "training_target_to_trajectory",
    "validate_ppo_reward_context_batch",
    "validate_training_target_overlay",
]
