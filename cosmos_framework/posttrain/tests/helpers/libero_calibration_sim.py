# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Test-side LIBERO backend emitting NVIDIA-native observation views beside the posttrain ones.

Baseline calibration (`tests/gr00t_baseline_cli.py`) loads this module in the simulator process
through ``SimGatewayBody(preload=...)``; importing it registers ``libero_calibration``. Production
recipes never name that backend, so the native ``LiberoEnv`` dependency stays out of
``embodied/sims/libero/runtime.py``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from cosmos_framework.posttrain.embodied.registry import register_sim
from cosmos_framework.posttrain.embodied.sim import SimCapabilities
from cosmos_framework.posttrain.embodied.sims.libero.runtime import LiberoChunkSession, LiberoSimulator

NATIVE_VIEWS = ("raw_image", "raw_wrist_image", "native_image", "native_wrist_image")


class CalibrationLiberoChunkSession(LiberoChunkSession):
    """Borrow NVIDIA's ``LiberoEnv`` simulator so native and posttrain views share one physics state."""

    def __init__(self, n_lanes: int, *args: Any, **kwargs: Any) -> None:
        self._native_owners: list[Any] = [None] * n_lanes
        super().__init__(n_lanes, *args, **kwargs)

    def _open_env(self, lane: int, bddl: str, language: str) -> Any:
        from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv

        owner = LiberoEnv(bddl, language)
        self._native_owners[lane] = owner
        # Borrow its raw simulator; only the native Gym owner closes it.
        return owner._env

    def _close_env(self, lane: int) -> None:
        owner, self._native_owners[lane] = self._native_owners[lane], None
        self._envs[lane] = None
        owner.close()

    def _views(self, raw: dict, prompt: str, lane: int) -> dict:
        observation = super()._views(raw, prompt, lane)
        native = self._native_owners[lane]._process_observation(raw)
        observation.update(
            raw_image=np.ascontiguousarray(raw["agentview_image"]),
            raw_wrist_image=np.ascontiguousarray(raw["robot0_eye_in_hand_image"]),
            raw_eef_pos=np.asarray(raw["robot0_eef_pos"]).copy(),
            raw_eef_quat=np.asarray(raw["robot0_eef_quat"]).copy(),
            raw_gripper=np.asarray(raw["robot0_gripper_qpos"]).copy(),
            native_image=np.ascontiguousarray(native["video.image"]),
            native_wrist_image=np.ascontiguousarray(native["video.wrist_image"]),
        )
        for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper"):
            observation[f"native_{key}"] = np.asarray(native[f"state.{key}"], dtype=np.float32)
        return observation


class CalibrationLiberoSimulator(LiberoSimulator):
    session_class = CalibrationLiberoChunkSession

    def capabilities(self) -> SimCapabilities:
        # The shared gateway sizes its observation buffer from these declarations.
        base = super().capabilities()
        return replace(base, camera_shapes={**base.camera_shapes, **{key: (256, 256) for key in NATIVE_VIEWS}})


register_sim("libero_calibration", lambda **kwargs: CalibrationLiberoSimulator(**kwargs))
