# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Minimal numerical, parity, role-loop, and model-DP coverage for synchronous PPO."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging
import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.algorithm.ppo import ppo_policy_loss, ppo_value_loss
from cosmos_framework.posttrain.algorithm.ppo.batch import PPOUpdatePlan
from cosmos_framework.posttrain.algorithm.ppo.gae import compute_gae
from cosmos_framework.posttrain.algorithm.ppo.jsd import generalized_jsd, topk_residual_jsd
from cosmos_framework.posttrain.algorithm.ppo.views import (
    CriticScore,
    PPOEpisode,
    PPOTrainingEpisode,
    ReferenceScore,
    ScoredPPOEpisode,
    attach_critic_score,
    attach_reference_score,
    episode_to_trajectory,
    trajectory_to_episode,
    trajectory_to_reference_score,
)
from cosmos_framework.posttrain.comm.bulk import BulkRef, BulkStore
from cosmos_framework.posttrain.comm.codec import materialize_payload, pack_payload
from cosmos_framework.posttrain.comm.transfer.movers import LocalStoreMover
from cosmos_framework.posttrain.model.value import DenseValueModel, response_action_logits, sampled_action_logprobs
from cosmos_framework.posttrain.roles.critic import PPOCriticRole
from cosmos_framework.posttrain.roles.reference import PPOReferenceRole
from cosmos_framework.posttrain.roles.trainer.ppo import PPOPolicyTrainer
from cosmos_framework.posttrain.schema import ObsBundle, Trajectory

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _ToyLM(torch.nn.Module):
    config: SimpleNamespace
    embedding: torch.nn.Embedding
    lm_head: torch.nn.Linear

    def __init__(self, hidden_size: int = 8, vocabulary_size: int = 9) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = torch.nn.Embedding(vocabulary_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocabulary_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,  # [B,L]
        *,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        use_cache: bool = True,
        **_: object,
    ) -> SimpleNamespace:
        del return_dict
        assert not use_cache
        hidden = self.embedding(input_ids)  # [B,L,H]
        logits = self.lm_head(hidden)  # [B,L,V]
        hidden_states = (hidden,) if output_hidden_states else None
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)


def _episode(
    episode_id: str = "episode-0",
    policy_version: int = 0,
    *,
    old_logprobs: tuple[float, ...] = (-1.25, -0.75),
    terminal_reward: float = 1.0,
) -> PPOEpisode:
    return PPOEpisode(
        episode_id=episode_id,
        policy_version=policy_version,
        rendered_prompt="1 + 1 =",
        prompt_token_ids=(5, 6, 7),
        action_token_ids=(2, 3),
        action_mask=(True, True),
        terminated=True,
        truncated=False,
        old_logprobs=old_logprobs,
        terminal_reward=terminal_reward,
        process_rewards=(0.1, 0.2),
    )


def _scores(episode: PPOEpisode) -> tuple[ReferenceScore, CriticScore]:
    reference = ReferenceScore(
        episode_id=episode.episode_id,
        policy_version=episode.policy_version,
        rendered_prompt=episode.rendered_prompt,
        prompt_token_ids=episode.prompt_token_ids,
        action_token_ids=episode.action_token_ids,
        action_mask=episode.action_mask,
        media=episode.media,
        action_logprobs=(-1.0, -0.5),
        support_token_ids=((0, 1), (1, 2)),
        support_probabilities=((0.2, 0.3), (0.1, 0.6)),
        residual_probabilities=(0.5, 0.3),
    )
    critic = CriticScore(
        episode_id=episode.episode_id,
        policy_version=episode.policy_version,
        rendered_prompt=episode.rendered_prompt,
        prompt_token_ids=episode.prompt_token_ids,
        action_token_ids=episode.action_token_ids,
        action_mask=episode.action_mask,
        media=episode.media,
        values=(0.4, 0.5, 0.6),
    )
    return reference, critic


def _training_item(episode: PPOEpisode) -> PPOTrainingEpisode:
    reference, critic = _scores(episode)
    return PPOTrainingEpisode(
        scored=ScoredPPOEpisode(episode=episode, reference=reference, critic=critic),
        action_rewards=(0.1, episode.terminal_reward),
        advantages=(0.6, -0.25),
        returns=(1.0, 0.25),
        value_advantages=(0.6, -0.25),
    )


def _trajectory(episode: PPOEpisode) -> Trajectory:
    input_ids = torch.tensor([episode.prompt_token_ids + episode.action_token_ids], dtype=torch.long)  # [1,P+T]
    inputs = {"input_ids": input_ids}
    return episode_to_trajectory(
        episode,
        observation=ObsBundle(
            representations={"prompt": episode.rendered_prompt, "reference": 2},
            id=episode.episode_id,
        ),
        processor_inputs_by_role=dict.fromkeys(("policy", "reference", "critic"), inputs),
        completion_text="2",
    )


def _scored_trajectory(episode: PPOEpisode) -> Trajectory:
    reference, critic = _scores(episode)
    return attach_critic_score(attach_reference_score(_trajectory(episode), reference), critic)


def test_ppo_wire_tensorizes_large_numeric_fields_and_preserves_record_views() -> None:
    """Large PPO vectors use the existing tensor bulk path without changing local tuple records."""
    episode = dataclasses.replace(
        _episode(),
        behavior_topk_token_ids=((0, 1), (1, 2)),
    )
    trajectory = _trajectory(episode)
    policy_info = trajectory.transitions[0].policy_info

    assert isinstance(policy_info["prompt_ids"], torch.Tensor)
    assert tuple(policy_info["prompt_ids"].shape) == (3,)
    assert isinstance(policy_info["behavior_topk_token_ids"], torch.Tensor)
    assert tuple(policy_info["behavior_topk_token_ids"].shape) == (2, 2)
    assert trajectory_to_episode(trajectory) == episode

    reference, _critic = _scores(episode)
    scored = attach_reference_score(trajectory, reference)
    reference_wire = scored.transitions[0].algo_extra["ppo"]["reference"]
    assert isinstance(reference_wire["support_token_ids"], torch.Tensor)
    assert tuple(reference_wire["support_token_ids"].shape) == (2, 2)
    assert isinstance(reference_wire["support_probabilities"], torch.Tensor)
    assert tuple(reference_wire["support_probabilities"].shape) == (2, 2)
    assert trajectory_to_reference_score(scored) == reference


def test_ppo_large_prompt_and_topk_fields_use_bulk_refs_and_round_trip() -> None:
    """Production-sized top-k rows aggregate before thresholding instead of staying inline per row."""
    token_count = 16
    top_k = 128
    episode = PPOEpisode(
        episode_id="bulk-episode",
        policy_version=0,
        rendered_prompt="long prompt",
        prompt_token_ids=tuple(range(256)),
        action_token_ids=tuple(range(token_count)),
        action_mask=(True,) * token_count,
        terminated=True,
        truncated=False,
        old_logprobs=(-1.0,) * token_count,
        behavior_topk_token_ids=tuple(tuple(range(top_k)) for _ in range(token_count)),
    )
    support_ids = tuple(tuple(range(top_k)) for _ in range(token_count))
    support_probabilities = tuple((1.0 / (2 * top_k),) * top_k for _ in range(token_count))
    reference = ReferenceScore(
        episode_id=episode.episode_id,
        policy_version=episode.policy_version,
        rendered_prompt=episode.rendered_prompt,
        prompt_token_ids=episode.prompt_token_ids,
        action_token_ids=episode.action_token_ids,
        action_mask=episode.action_mask,
        media=episode.media,
        action_logprobs=(-1.0,) * token_count,
        support_token_ids=support_ids,
        support_probabilities=support_probabilities,
        residual_probabilities=(0.5,) * token_count,
    )
    scored = attach_reference_score(_trajectory(episode), reference)
    store = BulkStore()

    packed = pack_payload((scored,), store)

    assert isinstance(packed, tuple)
    transition = packed[0].transitions[0]
    assert isinstance(transition.policy_info["prompt_ids"], BulkRef)
    assert isinstance(transition.policy_info["behavior_topk_token_ids"], BulkRef)
    reference_wire = transition.algo_extra["ppo"]["reference"]
    assert isinstance(reference_wire["support_token_ids"], BulkRef)
    assert isinstance(reference_wire["support_probabilities"], BulkRef)

    materialized = materialize_payload(packed, LocalStoreMover())
    assert isinstance(materialized, tuple)
    assert trajectory_to_episode(materialized[0]) == episode
    assert trajectory_to_reference_score(materialized[0]) == reference
    assert len(store) == 0


class _SharedModelRollout:
    model: torch.nn.Module
    version: int
    pending_version: int | None

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.version = -1
        self.pending_version = None

    def generate(self, prompts: list[ObsBundle], *, policy_version: int) -> list[Trajectory]:
        assert policy_version == self.version
        return [_trajectory(_episode(prompt.id, policy_version)) for prompt in prompts]

    def prepare_weight_version(self, version: int, producer_identity: str | None) -> None:
        from cosmos_framework.posttrain.comm.weight_transfer import shared_model_identity

        assert producer_identity == shared_model_identity(self.model)
        self.pending_version = version

    def commit_weight_version(self, version: int) -> None:
        assert version == self.pending_version
        self.version = version
        self.pending_version = None


def test_entrypoint_declares_and_places_the_seven_shared_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    """PPO adds role interactions without introducing a PPO-specific placement backend."""
    from cosmos_framework.posttrain.control.placement import ActorGroup
    from cosmos_framework.posttrain.entrypoints import ppo
    from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

    grid = SimpleNamespace(
        policy_shape=lambda: (1, 2),
        rollout_shape=lambda: (1, 1),
        reference_ranks=1,
        critic_ranks=2,
    )
    monkeypatch.setattr(ppo, "_model_parallelism", lambda _config, _role, ranks: (1, ranks))

    groups = ppo.declare_wgs(SimpleNamespace(grid=grid))

    assert {name: group.role for name, group in groups.items()} == {
        "data": "env",
        "rollout": "rollout",
        "rm": "reward",
        "replay": "buffer",
        "reference": "reference",
        "critic": "critic",
        "policy": "trainer",
    }
    actors = {name: ActorGroup((WorkerGroup(name, kind),), lambda: {}, name=name) for name, kind in ppo.ROLE_SPECS}
    colocated = ppo._device_groups(actors, "colocated")
    disaggregated = ppo._device_groups(actors, "disaggregated")
    assert colocated[1].actor_groups == (actors["rollout"], actors["policy"])
    assert [group.name for group in disaggregated] == [
        "ppo_control",
        "ppo_experience_gpus",
        "ppo_policy_gpus",
        "ppo_reference",
        "ppo_critic",
    ]


def test_scoring_parity_matches_rollout_logprobs() -> None:
    """HF recomputation must recover exact rollout scores before the first PPO update."""
    torch.manual_seed(7)
    model = _ToyLM()
    template = _episode()
    with torch.inference_mode():
        input_ids = torch.tensor(  # [1,P+T]
            [[*template.prompt_token_ids, *template.action_token_ids]],
            dtype=torch.long,
        )
        logits = model(input_ids=input_ids, use_cache=False).logits  # [1,P+T,V]
        action_logits = response_action_logits(  # [T,V]
            logits,
            len(template.prompt_token_ids),
            len(template.action_token_ids),
        )[0]
        action_ids = torch.tensor(template.action_token_ids, dtype=torch.long)  # [T]
        rollout_logprobs = sampled_action_logprobs(action_logits, action_ids)  # [T]
    episode = dataclasses.replace(template, old_logprobs=tuple(float(value) for value in rollout_logprobs))
    role = PPOPolicyTrainer(model, torch.optim.SGD(model.parameters(), lr=0.0))

    metrics = role.score_on_policy_metrics([_training_item(episode)], policy_clip=0.2)

    expected = {
        "behavior_logprob_delta_mean": 0.0,
        "behavior_logprob_delta_max": 0.0,
        "behavior_logprob_multiplicative_mean": 1.0,
        "initial_ratio_mean": 1.0,
        "initial_ratio_min": 1.0,
        "initial_ratio_max": 1.0,
        "initial_clip_fraction": 0.0,
        "initial_importance_ess": 1.0,
    }
    assert {name: metrics[name] for name in expected} == pytest.approx(expected, abs=1e-7)


def test_ppo_numerics_match_independent_closed_forms() -> None:
    """GAE, clipped policy/value losses, and full-support JSD match direct definitions."""
    rewards = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float64)  # [B,T]
    values = torch.tensor([[0.0, 0.5, 2.0], [0.0, 0.5, 2.0]], dtype=torch.float64)  # [B,T+1]
    mask = torch.ones((2, 2), dtype=torch.bool)  # [B,T]
    terminated = torch.tensor([True, False], dtype=torch.bool)  # [B]
    truncated = torch.tensor([False, True], dtype=torch.bool)  # [B]
    advantages, returns = compute_gae(  # [B,T], [B,T]
        rewards,
        values,
        mask,
        terminated,
        truncated,
        gamma=1.0,
        gae_lambda=1.0,
    )
    expected_advantages = torch.tensor([[1.0, 0.5], [3.0, 2.5]], dtype=torch.float64)  # [B,T]
    expected_returns = torch.tensor([[1.0, 1.0], [3.0, 3.0]], dtype=torch.float64)  # [B,T]
    torch.testing.assert_close(advantages, expected_advantages)
    torch.testing.assert_close(returns, expected_returns)

    current = torch.tensor(  # [T]
        [math.log(1.5), math.log(1.1), math.log(0.5), math.log(0.9)],
        dtype=torch.float64,
        requires_grad=True,
    )
    reference_current = current.detach().clone().requires_grad_(True)  # [T]
    old = torch.zeros(4, dtype=torch.float64)  # [T]
    policy_advantages = torch.tensor([2.0, 1.0, -3.0, -1.0], dtype=torch.float64)  # [T]
    token_mask = torch.ones(4, dtype=torch.bool)  # [T]
    actual_policy, policy_metrics = ppo_policy_loss(current, old, policy_advantages, token_mask, 0.2)  # []
    ratio = reference_current.exp()  # [T]
    expected_policy = -torch.minimum(  # []
        ratio * policy_advantages,
        ratio.clamp(0.8, 1.2) * policy_advantages,
    ).mean()
    actual_policy.backward()
    expected_policy.backward()
    torch.testing.assert_close(actual_policy, expected_policy)
    torch.testing.assert_close(current.grad, reference_current.grad)
    assert float(policy_metrics["clip_fraction"]) == pytest.approx(0.5)

    current_values = torch.tensor([1.0, 0.1], dtype=torch.float64, requires_grad=True)  # [T]
    reference_values = current_values.detach().clone().requires_grad_(True)  # [T]
    old_values = torch.zeros(2, dtype=torch.float64)  # [T]
    targets = torch.ones(2, dtype=torch.float64)  # [T]
    actual_value, value_metrics = ppo_value_loss(current_values, old_values, targets, token_mask[:2], 0.2)  # []
    clipped_values = old_values + (reference_values - old_values).clamp(-0.2, 0.2)  # [T]
    expected_value = (
        0.5
        * torch.maximum(  # []
            (reference_values - targets).square(),
            (clipped_values - targets).square(),
        ).mean()
    )
    actual_value.backward()
    expected_value.backward()
    torch.testing.assert_close(actual_value, expected_value)
    torch.testing.assert_close(current_values.grad, reference_values.grad)
    assert float(value_metrics["value_clip_fraction"]) == pytest.approx(0.5)

    policy_logits = torch.tensor([0.1, 1.0, -0.3, 0.4], dtype=torch.float64)  # [V]
    reference_logits = torch.tensor([0.3, -0.1, 0.7, 0.2], dtype=torch.float64)  # [V]
    reference_probs = reference_logits.softmax(dim=-1)  # [V]
    support_ids = torch.arange(policy_logits.numel())  # [V]
    residual = torch.tensor(0.0, dtype=torch.float64)  # []
    actual_jsd = topk_residual_jsd(policy_logits, support_ids, reference_probs, residual, 0.5)  # []
    expected_jsd = generalized_jsd(policy_logits.softmax(dim=-1), reference_probs, 0.5)  # []
    torch.testing.assert_close(actual_jsd, expected_jsd, atol=1e-12, rtol=1e-12)


def test_ppo_role_loop_runs_two_updates_and_publishes_weights(tmp_path: Path) -> None:
    """The public loop crosses every PPO role and commits policy, critic, and source state."""
    from cosmos_framework.posttrain.config.rl import PPOConfig
    from cosmos_framework.posttrain.control import ExperienceBuffer
    from cosmos_framework.posttrain.entrypoints.ppo import build_reward, loop
    from cosmos_framework.posttrain.roles.env import DatasetEnv, DeterministicArithmeticDataset

    torch.manual_seed(19)
    policy_model = _ToyLM()
    reference_model = copy.deepcopy(policy_model)
    reference_before = {name: value.detach().clone() for name, value in reference_model.state_dict().items()}
    critic_model = DenseValueModel(_ToyLM())
    config = PPOConfig(
        max_steps=2,
        save_every=1,
        resume="off",
        checkpoint_format="consolidated",
        prompts_per_iteration=2,
        ppo_epochs=1,
        num_minibatches=1,
        microbatch_size=1,
    )
    rollout = _SharedModelRollout(policy_model)
    policy = PPOPolicyTrainer(
        policy_model,
        torch.optim.SGD(policy_model.parameters(), lr=0.01),
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        reference_kl_coef=config.reference_kl_coef,
    )
    critic = PPOCriticRole(critic_model, torch.optim.SGD(critic_model.parameters(), lr=0.01))
    units = {
        "data": DatasetEnv(DeterministicArithmeticDataset(seed=4, length=4, max_operand=2), repeat=True),
        "rollout": rollout,
        "rm": build_reward(config),
        "replay": ExperienceBuffer(exact_batch_size=config.prompts_per_iteration),
        "reference": PPOReferenceRole(reference_model, top_k=4),
        "critic": critic,
        "policy": policy,
    }

    summary = loop(
        units,
        SimpleNamespace(rl=config, job=SimpleNamespace(wandb_mode="disabled")),
        str(tmp_path),
    )

    assert summary["policy_version"] == policy.policy_version == rollout.version == 2
    assert policy.optimizer_step == critic.optimizer_step == 2
    assert all(torch.equal(value, reference_model.state_dict()[name]) for name, value in reference_before.items())
    assert (tmp_path / "checkpoints/iter_000002/_SUCCESS").is_file()
    assert (tmp_path / "checkpoints/iter_000002/policy/data.pt").is_file()


def _exact_digest(value: Any) -> str:
    """Digest a nested state structure from its EXACT tensor bytes.

    ``torch.equal`` would answer "did the values change", which is not the question a warmup
    iteration has to answer: the actor's optimizer state and LR schedule must be untouched, and
    those are dictionaries of tensors, ints and floats whose SHAPE can change too (an SGD optimizer
    grows a momentum entry the first time it steps). Hashing the traversal covers all of it.
    """
    hasher = hashlib.sha256()

    def visit(path: str, item: Any) -> None:
        hasher.update(path.encode("utf-8"))
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            hasher.update(f"{array.dtype}{array.shape}".encode("ascii"))
            hasher.update(array.tobytes())
        elif isinstance(item, Mapping):
            for key in sorted(item, key=repr):
                visit(f"{path}.{key!r}", item[key])
        elif isinstance(item, (list, tuple)):
            for index, entry in enumerate(item):
                visit(f"{path}[{index}]", entry)
        else:
            hasher.update(repr(item).encode("utf-8"))

    visit("", value)
    return hasher.hexdigest()


def _actor_state(policy: PPOPolicyTrainer) -> dict[str, str]:
    """Digest every piece of actor state a critic-only iteration must leave untouched."""
    return {
        "model": _exact_digest(policy.model.state_dict()),
        "optimizer": _exact_digest(policy.opt.state_dict()),
        "lr_scheduler": _exact_digest(policy.lr_scheduler.state_dict()),
    }


def _optimizer_internal_steps(optimizer: torch.optim.Optimizer) -> tuple[int, ...]:
    """Report each parameter's OWN step counter, which is not the trainer's optimizer clock.

    A stateful optimizer keeps its own per-parameter `step`, and AdamW divides by the bias
    correction derived from it. `LLMTrainer.save()` under `checkpoint_format="dcp"` extracts
    optimizer state through `get_optimizer_state_dict`, which materializes an untouched
    optimizer by running a zero-gradient, zero-LR `step()`. Parameters do not move, so no
    model digest can see it -- but the counter advances, and the next real update is then
    bias-corrected for the wrong step. The trainer's own `optimizer_step` stays 0 throughout,
    so only this counter can distinguish a pristine actor from a silently initialized one.
    """
    steps: list[int] = []
    for parameter in sorted(optimizer.state, key=id):
        step = optimizer.state[parameter].get("step")
        if step is not None:
            steps.append(int(step.item() if isinstance(step, torch.Tensor) else step))
    return tuple(steps)


def _as_int(value: Any) -> int:
    """Read a step counter that may be a python int, a 0-d tensor, or a 1-element tensor."""
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    return int(value)


def _optimizer_group_state(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Report the optimizer state that lives on the GROUP rather than per parameter.

    `_optimizer_internal_steps` reads `optimizer.state`, which is empty for an implementation
    that keeps one shared counter per param group. `utils/fused_adam.py::FusedAdam` is exactly
    that: `step()` creates `param_groups["step"]` and lazily builds `param_groups_master`.
    Clearing `optimizer.state` therefore restores nothing for it, and the next save sees a
    pristine optimizer again and increments the group counter a second time.
    """
    return {
        "group_steps": tuple(_as_int(group["step"]) for group in optimizer.param_groups if "step" in group),
        "has_master": getattr(optimizer, "param_groups_master", None) is not None,
    }


class _GroupStateAdamW(torch.optim.AdamW):
    """An AdamW that keeps its step counter and master weights where FusedAdam keeps them.

    FusedAdam needs apex CUDA kernels and cannot be constructed on a CPU test host, so this
    models the part of it that matters here: state owned by the param GROUP and by an
    optimizer attribute, neither of which lives in `optimizer.state`. The rollback has to be
    defined over everything a `step()` creates, not over one dictionary.
    """

    param_groups_master: list[dict[str, Any]] | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.param_groups_master = None

    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
        if self.param_groups_master is None:
            self.param_groups_master = [
                {"params": [p.detach().clone().float() for p in g["params"]]} for g in self.param_groups
            ]
        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1
        return super().step(closure)


class _TracingRollout(_SharedModelRollout):
    """Record what each outer iteration observed BEFORE that iteration could change anything."""

    observed_versions: list[int]
    observed_actor: list[str]

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        self.observed_versions = []
        self.observed_actor = []

    def generate(self, prompts: list[ObsBundle], *, policy_version: int) -> list[Trajectory]:
        self.observed_versions.append(policy_version)
        self.observed_actor.append(_exact_digest(self.model.state_dict()))
        return super().generate(prompts, policy_version=policy_version)


@dataclasses.dataclass(frozen=True)
class _PPORunResult:
    """Everything an outer-iteration/actor-update separation has to be judged on."""

    summary: dict[str, Any]
    observed_versions: tuple[int, ...]
    observed_actor: tuple[str, ...]
    initial_actor: dict[str, str]
    final_actor: dict[str, str]
    initial_critic: str
    final_critic: str
    policy_optimizer_step: int
    critic_optimizer_step: int
    rollout_version: int
    actor_parameters: dict[str, torch.Tensor]
    actor_optimizer_internal_steps: tuple[int, ...]


def _run_ppo_loop(
    out_dir: Path,
    *,
    max_steps: int,
    resume: str,
    warmup: int | None = None,
    save_every: int = 1,
    checkpoint_format: str = "consolidated",
    actor_optimizer: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer] | None = None,
) -> _PPORunResult:
    """Run the real PPO loop over toy roles; ``warmup=None`` leaves the config field defaulted.

    ``checkpoint_format`` and ``actor_optimizer`` are open because the checkpoint format decides
    HOW optimizer state is extracted and the optimizer decides whether that extraction is
    observable: the DCP path materializes an untouched optimizer, and only a stateful optimizer
    carries the step counter that records it. The defaults keep the original consolidated/SGD
    recipe, under which neither is true.
    """
    from cosmos_framework.posttrain.config.rl import PPOConfig
    from cosmos_framework.posttrain.control import ExperienceBuffer
    from cosmos_framework.posttrain.entrypoints.ppo import build_reward, loop
    from cosmos_framework.posttrain.roles.env import DatasetEnv, DeterministicArithmeticDataset

    torch.manual_seed(19)
    policy_model = _ToyLM()
    reference_model = copy.deepcopy(policy_model)
    critic_model = DenseValueModel(_ToyLM())
    warmup_kwargs = {} if warmup is None else {"critic_warmup_iterations": warmup}
    config = PPOConfig(
        max_steps=max_steps,
        save_every=save_every,
        resume=resume,
        checkpoint_format=checkpoint_format,
        prompts_per_iteration=2,
        ppo_epochs=1,
        num_minibatches=1,
        microbatch_size=1,
        **warmup_kwargs,
    )
    optimizer = (
        torch.optim.SGD(policy_model.parameters(), lr=0.01)
        if actor_optimizer is None
        else actor_optimizer(policy_model.parameters())
    )
    policy = PPOPolicyTrainer(
        policy_model,
        optimizer,
        # A visibly decaying schedule: one actor optimizer step would halve the LR, so a frozen
        # scheduler is evidence rather than a coincidence of a constant-LR recipe.
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5),
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        reference_kl_coef=config.reference_kl_coef,
    )
    critic = PPOCriticRole(critic_model, torch.optim.SGD(critic_model.parameters(), lr=0.01))
    # Each role saves through its OWN checkpoint manager, which is where the format lives. In
    # production `build_policy` hands it one built from this same config field; these roles are
    # assembled by hand, so the config value alone would leave them on the default format and the
    # DCP extraction path would never run.
    policy.checkpoint_format = checkpoint_format
    critic.checkpoint_format = checkpoint_format
    rollout = _TracingRollout(policy_model)
    units = {
        "data": DatasetEnv(DeterministicArithmeticDataset(seed=4, length=16, max_operand=2), repeat=True),
        "rollout": rollout,
        "rm": build_reward(config),
        "replay": ExperienceBuffer(exact_batch_size=config.prompts_per_iteration),
        "reference": PPOReferenceRole(reference_model, top_k=4),
        "critic": critic,
        "policy": policy,
    }
    initial_actor = _actor_state(policy)
    initial_critic = _exact_digest(critic_model.state_dict())

    summary = loop(
        units,
        SimpleNamespace(rl=config, job=SimpleNamespace(wandb_mode="disabled")),
        str(out_dir),
    )

    return _PPORunResult(
        summary=summary,
        observed_versions=tuple(rollout.observed_versions),
        observed_actor=tuple(rollout.observed_actor),
        initial_actor=initial_actor,
        final_actor=_actor_state(policy),
        initial_critic=initial_critic,
        final_critic=_exact_digest(critic_model.state_dict()),
        policy_optimizer_step=policy.optimizer_step,
        critic_optimizer_step=critic.optimizer_step,
        rollout_version=rollout.version,
        actor_parameters={name: value.detach().clone() for name, value in policy_model.state_dict().items()},
        actor_optimizer_internal_steps=_optimizer_internal_steps(optimizer),
    )


def test_critic_warmup_freezes_actor_state_while_the_critic_learns(tmp_path: Path) -> None:
    """A run that never leaves warmup trains the critic and leaves every actor clock untouched."""
    run = _run_ppo_loop(tmp_path, max_steps=2, warmup=2, resume="off")

    assert run.final_actor == run.initial_actor
    assert run.final_critic != run.initial_critic
    assert run.policy_optimizer_step == 0
    assert run.critic_optimizer_step == 2
    # The actor never published, so the rollout stays on the version it was seeded with and every
    # collection round is on-policy against unchanged weights.
    assert run.observed_versions == (0, 0)
    assert run.observed_actor == (run.initial_actor["model"],) * 2
    assert run.rollout_version == 0
    assert run.summary["outer_iteration"] == 2
    assert run.summary["policy_version"] == 0
    assert run.summary["final_metrics"]["train/actor_warmup"] == 1.0
    assert "optim/policy_optimizer_step" not in run.summary["final_metrics"]
    assert run.summary["final_metrics"]["optim/critic_optimizer_step"] == 2.0
    # Warmup progress is durable in its own right: the boundary is named by the outer iteration.
    assert (tmp_path / "checkpoints/iter_000002/_SUCCESS").is_file()


def test_actor_updates_begin_exactly_at_the_iteration_after_warmup(tmp_path: Path) -> None:
    """Iterations 1..N are critic-only; iteration N+1 is the first one that moves the actor."""
    run = _run_ppo_loop(tmp_path, max_steps=4, warmup=2, resume="off")

    frozen = run.initial_actor["model"]
    # Observed at the START of iterations 1, 2, 3 and 4: the actor is still frozen entering
    # iteration 3 (the first non-warmup one) and has moved only by the start of iteration 4.
    assert run.observed_actor[:3] == (frozen, frozen, frozen)
    assert run.observed_actor[3] != frozen
    assert run.observed_versions == (0, 0, 0, 1)
    assert run.policy_optimizer_step == 2
    assert run.critic_optimizer_step == 4
    assert run.summary["outer_iteration"] == 4
    assert run.summary["policy_version"] == 2
    assert run.rollout_version == 2
    assert run.summary["final_metrics"]["train/actor_warmup"] == 0.0


def test_zero_warmup_preserves_the_unconditional_update_sequence(tmp_path: Path) -> None:
    """N=0 -- explicit or defaulted -- updates the actor every iteration, as it always did."""
    explicit = _run_ppo_loop(tmp_path / "explicit", max_steps=3, warmup=0, resume="off")
    defaulted = _run_ppo_loop(tmp_path / "defaulted", max_steps=3, warmup=None, resume="off")

    assert explicit.observed_versions == defaulted.observed_versions == (0, 1, 2)
    assert explicit.observed_actor == defaulted.observed_actor
    assert len(set(explicit.observed_actor)) == 3
    assert explicit.final_actor == defaulted.final_actor
    assert explicit.final_actor != explicit.initial_actor
    assert explicit.policy_optimizer_step == explicit.critic_optimizer_step == 3
    assert explicit.summary["outer_iteration"] == explicit.summary["policy_version"] == 3
    assert explicit.rollout_version == 3
    assert explicit.summary["final_metrics"]["train/actor_warmup"] == 0.0
    assert explicit.summary["final_metrics"]["optim/policy_optimizer_step"] == 3.0


@pytest.mark.parametrize("interrupt_after", [1, 2], ids=["mid-warmup", "at-warmup-boundary"])
def test_resume_during_and_at_the_warmup_boundary_matches_one_uninterrupted_run(
    tmp_path: Path,
    interrupt_after: int,
) -> None:
    """Interrupting inside warmup, or exactly on its last iteration, repeats and skips nothing."""
    uninterrupted = _run_ppo_loop(tmp_path / "whole", max_steps=4, warmup=2, resume="off")

    split = tmp_path / "split"
    first = _run_ppo_loop(split, max_steps=interrupt_after, warmup=2, resume="off")
    second = _run_ppo_loop(split, max_steps=4, warmup=2, resume="auto")

    assert first.observed_versions + second.observed_versions == uninterrupted.observed_versions
    assert first.observed_actor + second.observed_actor == uninterrupted.observed_actor
    assert second.summary["outer_iteration"] == uninterrupted.summary["outer_iteration"] == 4
    assert second.summary["policy_version"] == uninterrupted.summary["policy_version"] == 2
    assert second.final_actor == uninterrupted.final_actor
    assert second.final_critic == uninterrupted.final_critic
    assert second.policy_optimizer_step == uninterrupted.policy_optimizer_step
    assert second.critic_optimizer_step == uninterrupted.critic_optimizer_step


def test_resume_rejects_a_boundary_that_disagrees_with_the_configured_warmup(tmp_path: Path) -> None:
    """Changing the warmup length under a committed boundary is a fail-closed resume, not a silent
    renumbering of the actor's publication clock."""
    _run_ppo_loop(tmp_path, max_steps=2, warmup=2, resume="off")

    with pytest.raises(RuntimeError, match="critic_warmup_iterations"):
        _run_ppo_loop(tmp_path, max_steps=4, warmup=0, resume="auto")


@pytest.mark.parametrize("lengthened", [3, 5, 10], ids=["one-longer", "longer", "far-longer"])
def test_resume_rejects_a_lengthened_warmup_the_actor_clock_cannot_distinguish(
    tmp_path: Path,
    lengthened: int,
) -> None:
    """A boundary committed INSIDE warmup is pinned by the warmup it recorded, not by one inferred.

    An actor that has never published reports version 0, and ``max(0, iteration - N)`` is 0 for
    EVERY ``N >= iteration``, so a check derived from that counter accepts any lengthened warmup
    over such a boundary. Accepting it slides the first actor update silently -- from iteration 3 to
    4, 6 or 11 here -- and the run continues under a schedule it did not start with.
    """
    _run_ppo_loop(tmp_path, max_steps=2, warmup=2, resume="off")

    with pytest.raises(RuntimeError, match="critic_warmup_iterations"):
        _run_ppo_loop(tmp_path, max_steps=4, warmup=lengthened, resume="auto")


def test_resume_with_the_recorded_warmup_is_accepted_and_completes_the_schedule(tmp_path: Path) -> None:
    """The boundary names the warmup it ran under, and that same value resumes without complaint."""
    _run_ppo_loop(tmp_path, max_steps=2, warmup=2, resume="off")

    marker = json.loads((tmp_path / "checkpoints/iter_000002/_SUCCESS").read_text(encoding="utf-8"))
    assert marker["loop_state"] == {"critic_warmup_iterations": 2}

    resumed = _run_ppo_loop(tmp_path, max_steps=4, warmup=2, resume="auto")

    assert resumed.summary["outer_iteration"] == 4
    assert resumed.summary["policy_version"] == 2
    # Warmup is over exactly at the resumed boundary, so the first resumed iteration is the first
    # one to publish and the second observes that publication.
    assert resumed.observed_versions == (0, 1)


def test_resume_restores_a_boundary_committed_before_the_warmup_was_recorded(tmp_path: Path) -> None:
    """Back-compat: a boundary written before the marker carried loop state still resumes.

    It recorded no warmup, so absence is UNKNOWN rather than a mismatch and the run continues under
    exactly the check that boundary shipped with, instead of every existing boundary failing closed.
    """
    _run_ppo_loop(tmp_path, max_steps=2, warmup=2, resume="off")
    marker_path = tmp_path / "checkpoints/iter_000002/_SUCCESS"
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    payload.pop("loop_state")
    marker_path.write_text(json.dumps(payload), encoding="utf-8")

    resumed = _run_ppo_loop(tmp_path, max_steps=4, warmup=2, resume="auto")

    assert resumed.summary["outer_iteration"] == 4
    assert resumed.summary["policy_version"] == 2
    assert resumed.observed_versions == (0, 1)


def _adamw(parameters: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    """A STATEFUL optimizer, which is what makes a warmup-time checkpoint observable at all."""
    return torch.optim.AdamW(parameters, lr=0.01)


def test_warmup_checkpoint_leaves_the_frozen_actor_optimizer_uninitialized_under_dcp(
    tmp_path: Path,
) -> None:
    """Checkpointing a frozen actor must not initialize its optimizer.

    `get_optimizer_state_dict` materializes an untouched optimizer by stepping it with zero
    gradients at zero LR. Parameters do not move, so the model digest and the trainer's own
    `optimizer_step` both still read "nothing happened" -- while AdamW's internal step has gone
    to 1 and the first REAL update is bias-corrected as though it were the second.
    """
    run = _run_ppo_loop(
        tmp_path,
        max_steps=2,
        warmup=2,
        resume="off",
        checkpoint_format="dcp",
        actor_optimizer=_adamw,
    )

    assert run.policy_optimizer_step == 0
    assert run.final_actor == run.initial_actor
    assert run.actor_optimizer_internal_steps == ()


def test_warmup_numerics_do_not_depend_on_how_often_the_run_checkpoints(tmp_path: Path) -> None:
    """Save frequency is an operational knob; it may not reach the weights.

    Identical seeds and inputs, one warmup iteration, then a real actor update: saving every
    iteration and not saving at all have to produce bit-identical actors. They did not, because
    the warmup-iteration save advanced AdamW's step counter before the first real update used it.
    """
    saving = _run_ppo_loop(
        tmp_path / "every-iteration",
        max_steps=2,
        warmup=1,
        resume="off",
        save_every=1,
        checkpoint_format="dcp",
        actor_optimizer=_adamw,
    )
    not_saving = _run_ppo_loop(
        tmp_path / "never",
        max_steps=2,
        warmup=1,
        resume="off",
        save_every=0,
        checkpoint_format="dcp",
        actor_optimizer=_adamw,
    )

    assert saving.policy_optimizer_step == not_saving.policy_optimizer_step == 1
    assert (
        saving.actor_optimizer_internal_steps
        == not_saving.actor_optimizer_internal_steps
        == (1,) * len(saving.actor_optimizer_internal_steps)
    )
    assert saving.actor_parameters.keys() == not_saving.actor_parameters.keys()
    for name, value in saving.actor_parameters.items():
        assert torch.equal(value, not_saving.actor_parameters[name]), f"{name} diverged with save_every=1"


def test_resume_inside_warmup_restores_a_still_uninitialized_actor_optimizer(tmp_path: Path) -> None:
    """A mid-warmup resume must hand back a pristine actor optimizer, not a re-initialized one.

    Restore has the same hazard as save in both directions: `_load_dcp` extracts a load TARGET
    through `get_optimizer_state_dict`, and the state on disk was written by an equally
    materializing save. Either one alone leaves the resumed actor at step 1, so the run that
    resumes trains different weights from the run that never stopped.
    """
    uninterrupted = _run_ppo_loop(
        tmp_path / "whole",
        max_steps=3,
        warmup=2,
        resume="off",
        checkpoint_format="dcp",
        actor_optimizer=_adamw,
    )

    split = tmp_path / "split"
    _run_ppo_loop(split, max_steps=1, warmup=2, resume="off", checkpoint_format="dcp", actor_optimizer=_adamw)
    resumed = _run_ppo_loop(
        split,
        max_steps=3,
        warmup=2,
        resume="auto",
        checkpoint_format="dcp",
        actor_optimizer=_adamw,
    )

    assert resumed.summary["outer_iteration"] == uninterrupted.summary["outer_iteration"] == 3
    assert resumed.policy_optimizer_step == uninterrupted.policy_optimizer_step == 1
    assert resumed.actor_optimizer_internal_steps == uninterrupted.actor_optimizer_internal_steps
    for name, value in uninterrupted.actor_parameters.items():
        assert torch.equal(value, resumed.actor_parameters[name]), f"{name} diverged across the resume"


def _pristine_policy(tmp_path: Path, optimizer_factory: Callable[..., torch.optim.Optimizer]) -> PPOPolicyTrainer:
    """A DCP policy trainer whose optimizer has never stepped, with a visibly decaying schedule."""
    torch.manual_seed(19)
    model = _ToyLM()
    optimizer = optimizer_factory(model.parameters(), lr=0.01)
    policy = PPOPolicyTrainer(
        model,
        optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5),
    )
    policy.checkpoint_format = "dcp"
    return policy


def _step_once(policy: PPOPolicyTrainer) -> None:
    """Advance the optimizer and its schedule the way one completed update would."""
    for parameter in policy.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    policy.opt.step()
    policy.lr_scheduler.step()
    policy._iter = 1
    policy._attempt_id = 1


@pytest.mark.parametrize("reuse_a_trained_optimizer", [False, True], ids=["fresh", "already-trained"])
def test_a_group_state_optimizer_round_trips_a_pristine_checkpoint(
    tmp_path: Path,
    reuse_a_trained_optimizer: bool,
) -> None:
    """Save and load must agree on the group schema, not just on the per-parameter state.

    A pristine save strips the probe-created group `step` from what it writes. Shaping the LOAD
    target runs the same materializing extraction, so that key comes back -- and DCP's planner is
    strict: a target field the checkpoint intentionally omits fails the load outright, before any
    later reset can undo it. The two sides have to be sanitized the same way.

    The reuse case matters separately: a checkpoint that asserts pristine can be loaded into an
    optimizer that has really trained, whose group already carries a `step` of its own.
    """
    source = _pristine_policy(tmp_path, _GroupStateAdamW)
    path = str(tmp_path / "iter_000001")
    source.save(path, publish=False, update_latest=False)

    target = _pristine_policy(tmp_path, _GroupStateAdamW)
    if reuse_a_trained_optimizer:
        _step_once(target)
        assert _optimizer_group_state(target.opt)["group_steps"] == (1,)

    target.load(path)

    assert _optimizer_group_state(target.opt) == {"group_steps": (), "has_master": False}
    assert target.optimizer_is_pristine()
    assert [group["lr"] for group in target.opt.param_groups] == [0.01]


def test_warmup_save_rolls_back_group_owned_optimizer_state_too(tmp_path: Path) -> None:
    """The rollback has to cover every place a `step()` writes, not just `optimizer.state`.

    PPO accepts `fusedadam` (`config_ppo.py::_validate_optimizer`), and that implementation keeps
    one shared step counter on the param GROUP and builds `param_groups_master` lazily. Clearing
    `optimizer.state` restores neither, so a second save would see a pristine optimizer again and
    advance the group counter once more -- checkpoint COUNT reaching the weights by another route.
    """
    policy = _pristine_policy(tmp_path, _GroupStateAdamW)
    assert _optimizer_group_state(policy.opt) == {"group_steps": (), "has_master": False}

    policy.save(str(tmp_path / "first"))
    assert _optimizer_group_state(policy.opt) == {"group_steps": (), "has_master": False}
    assert policy.optimizer_is_pristine()

    policy.save(str(tmp_path / "second"))
    assert _optimizer_group_state(policy.opt) == {"group_steps": (), "has_master": False}
    assert _optimizer_internal_steps(policy.opt) == ()


def test_an_optimizer_that_stepped_without_per_parameter_state_is_not_called_pristine(
    tmp_path: Path,
) -> None:
    """A real step that leaves `optimizer.state` empty must not be mistaken for no step at all.

    FusedAdam skips `state[p]` for any parameter whose grad is None but still advances the shared
    group counter, so a group of frozen parameters trains with an empty per-parameter dictionary.
    Reading only that dictionary would mark a stepped optimizer `optimizer_pristine`, and restore
    honors the marker by discarding optimizer state -- losing real training state, silently.
    """
    policy = _pristine_policy(tmp_path, _GroupStateAdamW)
    policy.opt.step()  # a REAL step; no parameter has a gradient, so no per-parameter state
    policy.opt.step()

    assert policy.opt.state == {}, "precondition: this optimizer keeps nothing per parameter"
    assert _optimizer_group_state(policy.opt)["group_steps"] == (2,)
    assert not policy.optimizer_is_pristine()

    # The marker is what restore acts on, so the checkpoint must not carry it here.
    assert "optimizer_pristine" not in policy._update_state_dict()


def test_a_pristine_checkpoint_restores_the_saved_optimizer_hyperparameters(tmp_path: Path) -> None:
    """An empty per-parameter state does not mean there is nothing to restore.

    `param_groups` still carries LR, betas and weight decay, and `lr_scheduler.load_state_dict`
    restores the scheduler's own bookkeeping WITHOUT writing the LR back into the optimizer. A
    restore that skips the optimizer component entirely therefore leaves the live optimizer on
    whatever hyperparameters it happened to hold.
    """
    policy = _pristine_policy(tmp_path, torch.optim.AdamW)
    path = str(tmp_path / "pristine")
    saved_lr = [group["lr"] for group in policy.opt.param_groups]
    policy.save(path)

    _step_once(policy)
    assert [group["lr"] for group in policy.opt.param_groups] != saved_lr

    policy.load(path)

    assert [group["lr"] for group in policy.opt.param_groups] == saved_lr
    assert policy.optimizer_is_pristine(), "restoring group options must not materialize moments"
    assert policy.lr_scheduler.last_epoch == 0


def test_a_pristine_resume_trains_the_same_as_a_fresh_process_resume(tmp_path: Path) -> None:
    """Resuming in-process and resuming fresh must agree on the first update after restore.

    This is the consequence of the hyperparameter gap: the reused trainer kept the decayed LR,
    so its next update moved the weights by a different amount than a fresh process loading the
    identical checkpoint. Same checkpoint, same code, two answers.
    """
    policy = _pristine_policy(tmp_path, torch.optim.AdamW)
    path = str(tmp_path / "pristine")
    policy.save(path)

    _step_once(policy)
    policy.load(path)
    _step_once(policy)
    reused = {name: value.detach().clone() for name, value in policy.model.state_dict().items()}

    fresh = _pristine_policy(tmp_path, torch.optim.AdamW)
    fresh.load(path)
    _step_once(fresh)

    for name, value in fresh.model.state_dict().items():
        assert torch.equal(value, reused[name]), f"{name} depends on whether the trainer was reused"


class _CapturedLog(logging.Handler):
    """Collect records off the posttrain logger, which does not propagate to ``caplog``'s root."""

    records: list[logging.LogRecord]

    def __init__(self) -> None:
        super().__init__()
        self.records = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_a_boundary_without_a_recorded_warmup_keeps_its_old_reach_and_reports_it(tmp_path: Path) -> None:
    """The cost of accepting a legacy boundary is reported, not hidden.

    Its derived check cannot discriminate any warmup at least as long as the boundary's iteration
    count, so a lengthened warmup over such a boundary stays accepted exactly as it was before the
    warmup became part of the marker -- and says so, naming the value it could not verify.
    """
    from cosmos_framework.posttrain.utils.log import logger as posttrain_logger

    _run_ppo_loop(tmp_path, max_steps=2, warmup=2, resume="off")
    marker_path = tmp_path / "checkpoints/iter_000002/_SUCCESS"
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    payload.pop("loop_state")
    marker_path.write_text(json.dumps(payload), encoding="utf-8")

    captured = _CapturedLog()
    posttrain_logger.addHandler(captured)
    try:
        resumed = _run_ppo_loop(tmp_path, max_steps=4, warmup=3, resume="auto")
    finally:
        posttrain_logger.removeHandler(captured)

    assert resumed.summary["outer_iteration"] == 4
    warnings = [record.getMessage() for record in captured.records if record.levelno == logging.WARNING]
    assert any("critic_warmup_iterations=3" in message and "cannot be verified" in message for message in warnings)


@pytest.mark.parametrize(
    "value",
    [-1, 3.0, True, "2"],
    ids=["negative", "float", "bool", "string"],
)
def test_ppo_config_rejects_a_warmup_length_that_is_not_a_whole_count(value: Any) -> None:
    """The typed schema is the first gate: a warmup length is a non-negative exact integer."""
    from cosmos_framework.posttrain.config.rl import PPOConfig

    with pytest.raises(ValueError, match="critic_warmup_iterations"):
        PPOConfig(critic_warmup_iterations=value)


@pytest.mark.parametrize("value", [-1, 1.5, True], ids=["negative", "float", "bool"])
def test_validate_ppo_config_rejects_an_overridden_warmup_before_workers_start(value: Any) -> None:
    """``--validate`` reads the UNINSTANTIATED root, so a dotlist override needs its own gate."""
    from cosmos_framework.posttrain.config.config_ppo import PosttrainPPOConfig, validate_ppo_config

    config = PosttrainPPOConfig(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )
    validate_ppo_config(config)

    config.rl.critic_warmup_iterations = value
    with pytest.raises(ValueError, match="critic_warmup_iterations"):
        validate_ppo_config(config)


def test_validate_ppo_config_rejects_custom_all_reduce_before_workers_start() -> None:
    """PPO rollout actors keep one visible GPU, so vLLM's custom all-reduce cannot map TP peers."""
    from cosmos_framework.posttrain.config.config_ppo import PosttrainPPOConfig, validate_ppo_config

    config = PosttrainPPOConfig(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )
    validate_ppo_config(config)

    config.rollout.custom_all_reduce = True
    with pytest.raises(ValueError, match="PPO does not support rollout.custom_all_reduce"):
        validate_ppo_config(config)


def _distributed_ppo_worker(rank: int, world_size: int, rendezvous_path: str, output_queue: Any) -> None:
    import traceback

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from cosmos_framework.posttrain.tests.helpers.distributed import init_gloo_process_group
    from cosmos_framework.posttrain.workers.dispatch import select_model_dp

    try:
        init_gloo_process_group(rank, world_size, rendezvous_path, timeout_seconds=10.0)
        torch.manual_seed(23)
        model = DistributedDataParallel(_ToyLM())
        role = PPOPolicyTrainer(model, torch.optim.SGD(model.parameters(), lr=0.01))
        global_batch = [
            _scored_trajectory(_episode(f"episode-{index}", terminal_reward=float(index + 1))) for index in range(4)
        ]
        local_batch = list(select_model_dp(global_batch, dp_rank=rank, dp_world=world_size))

        role.prepare_batch(local_batch)
        prepared = role.prepared_batch_receipt()
        update = role.train_batch(
            prepared.batch_id,
            PPOUpdatePlan(epochs=1, num_minibatches=2, microbatch_size=1, seed=5),
        )
        output_queue.put(
            (
                rank,
                {
                    "local_ids": [trajectory.id for trajectory in local_batch],
                    "global_ids": list(prepared.episode_ids),
                    "batch_id": prepared.batch_id,
                    "optimizer_steps": update.optimizer_steps,
                    "optimizer_step": update.optimizer_step,
                },
            )
        )
    except BaseException:  # noqa: BLE001 -- the parent must receive a traceback instead of hanging
        output_queue.put((rank, {"worker_error": traceback.format_exc()}))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.serial
def test_two_rank_model_dp_uses_one_global_batch_and_disjoint_local_shards(tmp_path: Path) -> None:
    """Two real ranks agree on batch identity while training disjoint cache shards."""
    from cosmos_framework.posttrain.tests.helpers.distributed import run_spawned_ranks

    results = run_spawned_ranks(
        _distributed_ppo_worker,
        world_size=2,
        rendezvous_path=tmp_path / "ppo-model-dp.rendezvous",
        timeout_seconds=45.0,
        context="minimal distributed PPO",
    )

    assert not set(results[0]["local_ids"]) & set(results[1]["local_ids"])
    assert results[0]["global_ids"] == results[1]["global_ids"]
    assert results[0]["batch_id"] == results[1]["batch_id"]
    assert {result["optimizer_steps"] for result in results.values()} == {2}
    assert {result["optimizer_step"] for result in results.values()} == {2}


def test_validate_ppo_config_refuses_an_emergency_plan_its_boundary_cannot_honour() -> None:
    """Review of c2af017642d (thread 0069d0e0): `rl.emergency_checkpoint_components` is a shared `RunConfig` field, but
    the multi-role boundary writes every role with the full plan at the cadence and at a termination trigger alike,
    so "run" would be ignored; the validator refuses it instead."""
    from cosmos_framework.posttrain.config.config_ppo import PosttrainPPOConfig, validate_ppo_config

    config = PosttrainPPOConfig(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )
    validate_ppo_config(config)  # the default "full" is what the boundary writes
    config.rl.emergency_checkpoint_components = "run"
    with pytest.raises(ValueError, match="PPO ignores rl.emergency_checkpoint_components='run'.*multi-role"):
        validate_ppo_config(config)
