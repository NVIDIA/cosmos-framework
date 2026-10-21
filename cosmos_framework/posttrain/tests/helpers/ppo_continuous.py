# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Small Gaussian action/value model and local decision environment for PPO integration tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_ppo import make_model_config
from cosmos_framework.posttrain.embodied.window import ChunkResult, collect_window
from cosmos_framework.posttrain.model.gr00t.policy import Gr00tN1d7TrainPolicy
from cosmos_framework.posttrain.model.gr00t.ppo import Gr00tPPOCondition, Gr00tPPOHeads, Gr00tPPOScores, Gr00tPPOTrace
from cosmos_framework.posttrain.model.protocol import PPOParameterPartition
from cosmos_framework.posttrain.roles.rollout.decision import DecisionRolloutMixin


def parameter_failure_release(learner, stage: str) -> dict:
    """Inject a real parameter-read failure and keep its entire exception chain alive."""
    import gc
    import traceback
    import weakref
    from unittest.mock import patch

    from torch.distributed.checkpoint import state_dict

    from cosmos_framework.posttrain.algorithm.ppo import wire

    refs = []

    def source_failure(value):
        tensors = value.values() if isinstance(value, dict) else (value,)
        refs.extend(weakref.ref(tensor) for tensor in tensors)
        raise ValueError("injected parameter reader failure")

    def wrap_failure(value):
        try:
            source_failure(value)
        except ValueError as error:
            raise RuntimeError("wrapped parameter reader failure") from error

    if stage == "hash":
        original = wire._canonical_parts

        def fail(value):
            if isinstance(value, torch.Tensor):
                wrap_failure(value)
            return original(value)

        target, name = wire, "_canonical_parts"
    elif stage == "state_dict":
        original = state_dict.get_model_state_dict

        def fail(*args, **kwargs):
            wrap_failure(original(*args, **kwargs))

        target, name = state_dict, "get_model_state_dict"
    else:
        raise ValueError(stage)
    retained = None
    with patch.object(target, name, fail):
        try:
            learner.evaluation_state()
        except RuntimeError as error:
            retained = error
    if retained is None or not isinstance(retained.__cause__, ValueError) or not refs:
        raise AssertionError("parameter reader failure was not reached")
    gc.collect()
    if any(ref() is not None for ref in refs):
        raise AssertionError("retained exception kept a borrowed parameter binding")
    if not traceback.extract_tb(retained.__traceback__) or not traceback.extract_tb(retained.__cause__.__traceback__):
        raise AssertionError("parameter cleanup discarded the exception chain")
    return {"stage": stage, "released_bindings": len(refs), "exception_chain_preserved": True}


class GaussianHeads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_head = nn.Module()
        self.action_head.actor = nn.Linear(2, 1)
        self.action_head.value_head = nn.Linear(2, 1)

    def forward(self, condition: Gr00tPPOCondition, trace: Gr00tPPOTrace) -> Gr00tPPOScores:
        features = condition.backbone_features[:, 0]
        mean = self.action_head.actor(features)
        logprob = torch.distributions.Normal(mean[:, None], 1).log_prob(trace.latent_after[:, :16, :7]).sum((1, 2))
        value = self.action_head.value_head(features)[:, 0]
        return Gr00tPPOScores(logprob, value)


class GaussianPolicy(DecisionRolloutMixin):
    validate_ppo_record = Gr00tN1d7TrainPolicy.validate_ppo_record
    score_ppo_decision = Gr00tN1d7TrainPolicy.score_ppo_decision

    ppo_logprob_tolerance = 0.02
    evaluation_mode = "ode"
    ppo_frozen_state = Gr00tN1d7TrainPolicy.ppo_frozen_state

    def ppo_identity(self) -> dict:
        return self.config.identity()

    def sample_decisions(self, observations, *, sources, seed, weight_version):
        from cosmos_framework.posttrain.model.gr00t.ppo import sampled_decisions

        return sampled_decisions(
            observations,
            self.infer_batch(observations, seed=seed, weight_version=weight_version),
            sources,
            seed=seed,
            version=weight_version,
        )

    def bootstrap_values(self, observations, *, seed, weight_version):
        return tuple(
            row["old_value"] for row in self.infer_batch(observations, seed=seed, weight_version=weight_version)
        )

    def _evaluation_actions(self, observations, *, seed, weight_version):
        return [
            row["action"]
            for row in self.infer_batch(
                observations, seed=seed, weight_version=weight_version, sampling_mode=self.evaluation_mode
            )
        ]

    def __init__(self, *, distributed: bool = False) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            self.net = GaussianHeads()
        self.config = make_model_config("test-gaussian")
        self.precision = SimpleNamespace(transfer="float32")
        self.ppo_parameters = PPOParameterPartition(
            tuple(name for name, _ in self.net.named_parameters() if ".value_head." not in name),
            tuple(name for name, _ in self.net.named_parameters() if ".value_head." in name),
        )
        self.version = 0
        self._forward = torch.nn.parallel.DistributedDataParallel(self.net) if distributed else self.net

    def ppo_score(self, condition: Gr00tPPOCondition, trace: Gr00tPPOTrace) -> Gr00tPPOScores:
        return self._forward(condition, trace)

    def infer_batch(
        self, observations: list[dict], *, seed: int, weight_version: int, sampling_mode: str | None = None
    ) -> list[dict]:
        if weight_version != self.version:
            raise ValueError("stale Gaussian policy version")
        if sampling_mode is not None:
            if sampling_mode not in ("ode", "native_ode"):
                raise ValueError("unsupported Gaussian evaluation mode")
            with torch.no_grad():
                features = torch.tensor([[observation["position"], 1.0] for observation in observations])
                mean = self.net.action_head.actor(features).tanh()
                return [{"action": value.expand(self.config.execution_horizon, 7).numpy().copy()} for value in mean]
        generator = torch.Generator().manual_seed(seed)
        result = []
        with torch.no_grad():
            batch = len(observations)
            features = torch.tensor(
                [[observation["position"], 1.0] for observation in observations], dtype=torch.float32
            )
            condition = Gr00tPPOCondition(
                features[:, None],
                torch.ones((batch, 1), dtype=torch.bool),
                torch.ones((batch, 1), dtype=torch.bool),
                torch.zeros((batch, 1, 132)),
                torch.zeros(batch, dtype=torch.long),
            )
            latent = torch.randn((batch, 40, 132), generator=generator) + self.net.action_head.actor(features)[:, None]
            trace = Gr00tPPOTrace(
                torch.zeros_like(latent), latent, torch.zeros(batch, dtype=torch.long), torch.zeros(batch)
            )
            scores = self.ppo_score(condition, trace)
            for index in range(batch):
                result.append(
                    {
                        "condition": condition,
                        "sample_index": index,
                        "trace": trace,
                        "old_logprob": float(scores.logprobs[index]),
                        "old_value": float(scores.values[index]),
                        "action": latent[index, :16, :7].tanh().numpy(),
                    }
                )
        return result


class IntegratorSession:
    n_lanes = 2
    session_id = "integrator"

    def __init__(self, n_lanes: int = 2, terminal_decision: int = 1, lane_offset: int = 0) -> None:
        self.n_lanes, self.terminal_decision, self.lane_offset = n_lanes, terminal_decision, lane_offset
        self.positions = np.linspace(0.3, -0.7, n_lanes)
        self.step = 0

    def observe(self) -> list[dict]:
        return [{"position": float(value)} for value in self.positions]

    def sources(self, window: int) -> list[dict]:
        return [
            {
                "sample_id": f"{window}:{lane}:{self.step}",
                "episode_id": f"{window}:{lane}",
                "fragment_id": f"{window}:{lane}",
                "step_idx": self.step,
                "session": self.session_id,
            }
            for lane in range(self.lane_offset, self.lane_offset + self.n_lanes)
        ]

    def step_chunk(self, actions: list[np.ndarray]) -> list[ChunkResult]:
        results = []
        for lane, action in enumerate(actions):
            rewards = []
            for control in action:
                self.positions[lane] += float(control[0]) * 0.1
                rewards.append(-float(self.positions[lane] ** 2))
            obs = {"position": float(self.positions[lane])}
            results.append(
                ChunkResult(action.copy(), obs, obs, tuple(rewards), self.step == self.terminal_decision, False)
            )
        self.step += 1
        return results

    def close(self) -> None:
        self.positions = np.empty(0)


class LocalWindowGateway:
    def __init__(self, n_lanes: int = 2, terminal_decision: int = 1, lane_offset: int = 0) -> None:
        self.n_lanes, self.terminal_decision, self.lane_offset = n_lanes, terminal_decision, lane_offset
        self.closed = False

    def run_window(self, *, policy: object, version: int, decisions: int) -> dict:
        if self.closed:
            raise RuntimeError("collection after gateway close")
        session = IntegratorSession(self.n_lanes, self.terminal_decision, self.lane_offset)
        try:
            return collect_window(session, policy, version=version, decisions=decisions)
        finally:
            session.close()

    def close(self) -> None:
        self.closed = True


class LinearVelocityHeads(Gr00tPPOHeads):
    """Analytical linear velocity field for checking the production ODE integrator."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            self.velocity = nn.Linear(4, 132)

    def _features(self, condition):
        return condition.backbone_features, condition.state

    def _velocity(self, condition, vl, state, latent, index):
        return self.velocity(vl.mean(dim=1))[:, None] + latent * 0.1


class EvaluationIntegratorSession(IntegratorSession):
    def __init__(self, case: dict, seed: int) -> None:
        super().__init__(n_lanes=1, terminal_decision=0)
        self.case = case
        self.positions = np.random.default_rng(seed + case["init_state_id"]).uniform(-1, 1, size=1)

    def step_chunk(self, actions: list[np.ndarray]) -> list[ChunkResult]:
        result = super().step_chunk(actions)[0]
        return [
            replace(
                result,
                episode={
                    **self.case,
                    "success": result.terminated,
                    "steps": len(result.rewards),
                    "return": sum(result.rewards),
                },
            )
        ]


class EvaluationIntegratorGateway(LocalWindowGateway):
    """Fixed-case sessions run the same public episode driver as the simulator server."""

    def __init__(self) -> None:
        super().__init__()
        self.evaluated: list[tuple[dict, int]] = []

    def open_evaluation(self, case: dict, seed: int) -> EvaluationIntegratorSession:
        self.evaluated.append((case, seed))
        return EvaluationIntegratorSession(case, seed)

    def run_evaluation(self, *, policy: object, cases: list[dict], version: int, seed: int, decisions: int) -> dict:
        from cosmos_framework.posttrain.embodied.window import evaluate_episodes

        if self.closed:
            raise RuntimeError("evaluation after gateway close")
        # This fixture owns no persistent physics outside the independently closed sessions.
        return {
            **evaluate_episodes(self, policy, cases=cases, version=version, seed=seed, decisions=decisions),
            "training_session_unchanged": True,
        }


def make_collector(rollout, gateway, recipe, replica, n_lanes, *, environment=None, evaluation_gateway_factory=None):
    """Drive production simulator, collector and rollout Roles with the small test simulator."""
    from cosmos_framework.posttrain.embodied.sims.libero.adapter import LiberoEnvironment
    from cosmos_framework.posttrain.embodied.sims.libero.config import LiberoConfig
    from cosmos_framework.posttrain.tests.helpers.window_control import LocalWindowControl

    if environment is None:
        horizon = rollout.config.execution_horizon
        environment = LiberoEnvironment(
            LiberoConfig(
                suite="libero_10" if horizon == 8 else "libero_spatial",
                execution_horizon=horizon,
                max_episode_steps=720 if horizon == 8 else 256,
            )
        )
    return LocalWindowControl(
        rollout,
        gateway,
        recipe,
        replica,
        n_lanes,
        environment=environment,
        evaluation_gateway_factory=evaluation_gateway_factory,
    )


def collect_scored(collector, version):
    """Use the production reward transaction and FIFO replay before invoking the learner."""
    from cosmos_framework.posttrain.algorithm.reward.decision import build_decision_reward_provider
    from cosmos_framework.posttrain.control.buffer import ExperienceBuffer
    from cosmos_framework.posttrain.embodied.sims.libero.reward import LiberoDecisionReward
    from cosmos_framework.posttrain.roles.reward.worker import RewardWorker

    reward = RewardWorker(
        [build_decision_reward_provider(LiberoDecisionReward(collector.environment.config.suite))],
        training_group_size=1,
    )
    replay = ExperienceBuffer(exact_batch_size=collector.n_lanes * collector.recipe.window_decisions)
    replay.put_samples(reward.score(collector.collect_window(version)))
    return replay.get_exact_samples()
