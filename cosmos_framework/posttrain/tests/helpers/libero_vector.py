# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Deterministic native-style physics for actual spawned LIBERO vector lifecycle tests."""

import os
import random
import time
from functools import partial
from types import SimpleNamespace

import numpy as np

from cosmos_framework.posttrain.embodied.sims.libero.runtime import LiberoChunkSession
from cosmos_framework.posttrain.embodied.sims.libero.vector import CAMERAS, LiberoLane, LiberoVectorEnv


class ToyPhysics:
    def __init__(self, bddl):
        if "fail-open" in bddl:
            raise ValueError("injected native construction failure")
        self.steps, self.closed = 0, False
        self.position = np.zeros(3, dtype=np.float64)
        self.env = SimpleNamespace(_observables={name: SimpleNamespace(_enabled=True) for name in CAMERAS})
        self.images = {name: np.zeros((8, 8, 3), dtype=np.uint8) for name in CAMERAS}
        self.render_count = 0

    def seed(self, seed):
        self.seed_value = seed
        np.random.seed(seed)
        random.seed(seed)

    def reset(self):
        self.steps = 0
        self.position[:] = 0

    def set_init_state(self, state):
        self.position[:] = float(np.asarray(state))

    def step(self, action):
        if action[-1] == 99:
            os._exit(17)
        if action[-1] == 98:
            time.sleep(30)
        if action[-1] == 97:
            raise ValueError("injected native step failure")
        self.steps += 1
        self.position += action[:3] + np.random.uniform(-1e-6, 1e-6, 3)
        observation = {"position": self.position.copy(), "physics_steps": self.steps}
        for name in CAMERAS:
            if self.env._observables[name]._enabled:
                self.images[name].fill(self.steps % 256)
                self.render_count += 1
                observation[name] = self.images[name].copy()
        success = self.steps == 18
        return observation, float(success), success, {}

    def get_sim_state(self):
        return {
            "steps": self.steps,
            "position": self.position.copy(),
            "seed": self.seed_value,
            "render_count": self.render_count,
        }

    def close(self):
        self.closed = True


class ToyLane(LiberoLane):
    def _open(self, bddl):
        return ToyPhysics(bddl)


class ToyBenchmark:
    n_tasks = 10

    def get_task_init_states(self, task):
        return np.arange(50)

    def get_task(self, task):
        return SimpleNamespace(problem_folder=str(task), bddl_file="task.bddl", language=f"task {task}")


class ToySession(LiberoChunkSession):
    def _open_vector(self):
        factory = partial(ToyLane, reset_steps=self._protocol.reset_steps, reset_gripper=self._protocol.reset_gripper)
        return LiberoVectorEnv([factory] * self.n_lanes)

    def _views(self, raw, prompt, lane):
        return {**raw, "prompt": prompt}
