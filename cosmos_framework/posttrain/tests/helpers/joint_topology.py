# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU window simulator and policy for tests of the production role bindings."""

from __future__ import annotations

import os
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.posttrain.algorithm.ppo.wire import canonical_digest
from cosmos_framework.posttrain.config.build import REPLICA_INDEX_ENV
from cosmos_framework.posttrain.embodied.embodiment import EmbodimentMap, register_embodiment
from cosmos_framework.posttrain.embodied.registry import register_sim, register_sink
from cosmos_framework.posttrain.embodied.sim import PROTOCOL_VERSION, SimCapabilities
from cosmos_framework.posttrain.roles.env.collector import CollectorBody
from cosmos_framework.posttrain.roles.rollout.decision import DecisionRolloutMixin
from cosmos_framework.posttrain.tests.helpers.joint_ppo import GaussianEnvironment, GaussianPolicy, GaussianSession

SIM_NAME = "joint_topology_test"


class TopologySession(GaussianSession):
    def __init__(self, *, replica: int, evaluation: bool = False, failure_once: bool = False) -> None:
        super().__init__(n_lanes=1 if evaluation else 2)
        self.replica, self.evaluation, self.failure_once = replica, evaluation, failure_once
        self.session_id = f"joint-topology-{replica}-{'eval' if evaluation else 'train'}"
        self.closed = False

    def observe(self):
        return [
            {**row, "image": np.full((64, 64, 3), self.replica + self.turn, dtype=np.uint8)}
            for row in super().observe()
        ]

    def sources(self, window):
        return [
            {**source, **{key: f"r{self.replica}:{source[key]}" for key in ("sample_id", "episode_id", "fragment_id")}}
            for source in super().sources(window)
        ]

    def step_chunk(self, actions):
        if self.failure_once and self.turn == 1:
            self.failure_once = False
            raise RuntimeError("injected topology simulator failure")
        results = super().step_chunk(actions)
        if self.evaluation:
            return [replace(row, terminated=True, episode={"success": True, "steps": 1}) for row in results]
        return results

    def close(self):
        self.closed = True


class TopologySimulator:
    def __init__(
        self,
        *,
        embodiment: str,
        replica: int = 0,
        output: str,
        failure_once: bool = False,
        extras=("decision_chunks", "fixed_case_evaluation"),
        **_options,
    ):
        self.replica = replica
        self.extras = tuple(extras)
        self.session = TopologySession(replica=replica, failure_once=failure_once and replica == 0)
        self.evaluations = []
        self.resource = (Path(output) / f"sim-{os.getpid()}-{replica}.txt").open("w")

    def capabilities(self):
        return SimCapabilities(
            protocol_version=PROTOCOL_VERSION,
            backend=SIM_NAME,
            backend_version="test-v1",
            embodiment=SIM_NAME,
            max_lanes=2,
            camera_shapes={"image": (64, 64)},
            proprio_width=2,
            extras=self.extras,
        )

    def probe(self):
        if self.resource.closed:
            raise RuntimeError("simulator is closed")

    def gpu_uuid(self):
        return None

    def open_window(self, version):
        self.probe()
        return self.session

    def open_evaluation(self, case, seed):
        self.probe()
        session = TopologySession(replica=self.replica, evaluation=True)
        self.evaluations.append(session)
        return session

    def online_state_digest(self):
        return canonical_digest({"turn": self.session.turn, "closed": self.session.closed})

    def close(self):
        self.session.close()
        self.resource.close()


def register_test_simulator():
    from cosmos_framework.posttrain.embodied.sinks.null import NullSink

    register_sink("null", NullSink)
    register_embodiment(EmbodimentMap(SIM_NAME, {"image": "image"}, "features", action_dim=2, state_dim=2))
    register_sim(SIM_NAME, TopologySimulator)


class TopologyEnvironment(GaussianEnvironment):
    evaluation_decisions = 2

    def evaluation_seed(self, context):
        return context["seed"]


class TopologyPolicy(DecisionRolloutMixin, GaussianPolicy):
    def __init__(self):
        super().__init__()
        self.closed = False

    def _evaluation_actions(self, observations, *, seed, weight_version):
        # Deliberate RNG use proves that the wrapper protects the process executing the model.
        random.random()
        np.random.random()
        torch.rand(1)
        with torch.no_grad():
            means, _ = self.net(torch.tensor(np.stack([row["features"] for row in observations])))
        return [row[None].numpy() for row in means]

    def rng_state(self):
        from cosmos_framework.posttrain.checkpoint import collect_rng_state

        return canonical_digest(collect_rng_state())

    def close(self):
        self.closed = True


class InspectableSimBody(CollectorBody):
    def state(self):
        return {
            "closed": self._sim.resource.closed,
            "turn": self._sim.session.turn,
            "evaluation_sessions_closed": all(session.closed for session in self._sim.evaluations),
        }


def setup_roles(names, recipe, output, failure_once=False):
    from cosmos_framework.posttrain.control.buffer import ExperienceBuffer
    from cosmos_framework.posttrain.roles.env.ppo_window import PPOWindowCollector
    from cosmos_framework.posttrain.roles.reward.worker import RewardWorker
    from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer
    from cosmos_framework.posttrain.tests.helpers.joint_ppo import reward_provider

    register_test_simulator()
    replica = int(os.environ.get(REPLICA_INDEX_ENV, "0"))
    bodies = {}
    for name in names:
        if name == "sim":
            bodies[name] = InspectableSimBody(
                sim=SIM_NAME,
                sink="null",
                embodiment=SIM_NAME,
                sim_options={"replica": replica, "output": output, "failure_once": failure_once},
            )
        elif name == "rollout":
            bodies[name] = TopologyPolicy()
        elif name == "collector":
            bodies[name] = PPOWindowCollector(recipe, replica, 2, environment=TopologyEnvironment())
        elif name == "reward":
            bodies[name] = RewardWorker([reward_provider()], training_group_size=1)
        elif name == "replay":
            bodies[name] = ExperienceBuffer(exact_batch_size=2 * recipe.window_decisions)
        elif name == "policy":
            bodies[name] = JointPPOTrainer(TopologyPolicy(), recipe)
    return bodies


# Gateway preloading performs the same registration inside the isolated simulator interpreter.
register_test_simulator()
