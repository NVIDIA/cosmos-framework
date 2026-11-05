# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Spawned native LIBERO lanes over the frozen package's ordered vector scheduler."""

from __future__ import annotations

import multiprocessing
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from libero.libero.envs.venv import BaseVectorEnv, CloudpickleWrapper, EnvWorker, SubprocEnvWorker, _worker

from cosmos_framework.posttrain.embodied.sim import SimProtocolError

CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
LaneSnapshot = tuple[
    np.ndarray,
    tuple[str, np.ndarray, int, int, float],
    tuple[int, tuple[int, ...], float | None],
    str | None,
    tuple[bool, ...],
]


@dataclass(frozen=True)
class LaneReset:
    bddl: str
    state: np.ndarray
    seed: int


class LiberoLane:
    """Own physics only; the parent session chooses cases and owns episode statistics."""

    def __init__(self, *, reset_steps: int, reset_gripper: float) -> None:
        self.reset_steps, self.reset_gripper = reset_steps, reset_gripper
        self.env = None
        self._bddl = None
        self.reset_request: LaneReset | None = None
        self._camera_flags: tuple[bool, ...] | None = None

    @property
    def unwrapped(self) -> LiberoLane:
        return self

    def _open(self, bddl: str):
        from libero.libero.envs import OffScreenRenderEnv

        return OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=256,
            camera_widths=256,
            render_gpu_device_id=0,
            ignore_done=True,
            horizon=100000,
        )

    def reset(self) -> dict:
        request = self.reset_request
        if request is None:
            raise SimProtocolError("LIBERO lane reset is missing its parent-selected case")
        try:
            if self._bddl != request.bddl:
                self.close()
                self.env = self._open(request.bddl)
                self._bddl = request.bddl
            self.env.seed(request.seed)
            self.env.reset()
            self.env.set_init_state(request.state)
            zero = np.zeros(7, dtype=np.float32)
            zero[-1] = self.reset_gripper
            for _ in range(self.reset_steps):
                observation, _, _, _ = self.env.step(zero)
            # Binding an enabled optimization must fail on a different simulator surface.
            for name in CAMERAS:
                self.env.env._observables[name]
            return observation
        except BaseException:
            self.close()
            raise

    @property
    def camera_enabled(self) -> bool:
        return self._camera_flags is None

    @camera_enabled.setter
    def camera_enabled(self, enabled: bool) -> None:
        if enabled and self._camera_flags is None:
            return
        observables = self.env.env._observables
        if not enabled and self._camera_flags is None:
            self._camera_flags = tuple(observables[name]._enabled for name in CAMERAS)
            for name in CAMERAS:
                # Observable.set_enabled resets its sampling clock; RLinf preserves it.
                observables[name]._enabled = False
        elif enabled and self._camera_flags is not None:
            for name, original in zip(CAMERAS, self._camera_flags, strict=True):
                observables[name]._enabled = original
            self._camera_flags = None

    def step(self, action: np.ndarray):
        try:
            return self.env.step(action)
        except BaseException:
            self.close()
            raise

    @property
    def state_snapshot(self) -> LaneSnapshot:
        return (
            self.env.get_sim_state(),
            np.random.get_state(),
            random.getstate(),
            self._bddl,
            tuple(self.env.env._observables[name]._enabled for name in CAMERAS),
        )

    def close(self) -> None:
        if self.env is not None:
            try:
                self.camera_enabled = True
            finally:
                env, self.env = self.env, None
                self._bddl = None
                env.close()


class SpawnLiberoWorker(SubprocEnvWorker):
    """Keep vendor RPCs, with explicit spawn and bounded ownership of the child."""

    def __init__(self, factory: Callable, lane: int) -> None:
        self.lane = lane
        context = multiprocessing.get_context("spawn")
        self.parent_remote, self.child_remote = context.Pipe()
        self.share_memory, self.buffer = False, None
        self.process = context.Process(
            target=_worker,
            args=(self.parent_remote, self.child_remote, CloudpickleWrapper(factory), None),
            daemon=True,
        )
        EnvWorker.__init__(self, factory)
        try:
            self.process.start()
            self.child_remote.close()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        if not self.parent_remote.poll(120):
            raise SimProtocolError(f"LIBERO lane {self.lane} did not reply within 120 seconds")
        try:
            return self.parent_remote.recv()
        except (EOFError, ConnectionResetError) as error:
            raise SimProtocolError(
                f"LIBERO lane {self.lane} process exited during an operation (exitcode={self.process.exitcode})"
            ) from error

    def recv(self):
        return self._receive()

    def get_env_attr(self, key: str):
        self.parent_remote.send(["getattr", key])
        return self._receive()

    def close_env(self) -> None:
        try:
            if self.process.pid is not None and self.process.is_alive():
                self.parent_remote.send(["close", None])
                if self.parent_remote.poll(2):
                    self.parent_remote.recv()
        except (BrokenPipeError, EOFError, ConnectionResetError):
            pass
        finally:
            self.parent_remote.close()
            self.child_remote.close()
            if self.process.pid is not None:
                self.process.join(timeout=2)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=2)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=2)


class LiberoVectorEnv(BaseVectorEnv):
    """One process per lane; the vendor owns synchronous ordered step/reset scheduling."""

    def __init__(self, factories: Sequence[Callable]) -> None:
        started: list[SpawnLiberoWorker] = []

        def worker(factory: Callable) -> SpawnLiberoWorker:
            result = SpawnLiberoWorker(factory, len(started))
            started.append(result)
            return result

        try:
            super().__init__(list(factories), worker)
        except BaseException:
            for item in started:
                item.close()
            raise

    def reset_lanes(self, requests: dict[int, LaneReset]) -> list[dict]:
        lanes = list(requests)
        for lane, request in requests.items():
            self.set_env_attr("reset_request", request, id=lane)
        return list(self.reset(id=lanes))

    def cameras(self, enabled: bool) -> None:
        self.set_env_attr("camera_enabled", enabled)
        states = self.get_env_attr("camera_enabled")
        if any(state != enabled for state in states):
            raise SimProtocolError("LIBERO workers did not acknowledge their camera state")

    def state_snapshots(self) -> list[LaneSnapshot]:
        return self.get_env_attr("state_snapshot")

    def close(self) -> None:
        if self.is_closed:
            return
        self.is_closed = True
        errors = []
        for worker in self.workers:
            try:
                worker.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]
