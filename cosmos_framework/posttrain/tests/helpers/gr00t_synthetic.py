# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""A deterministic, simulator-free source of processed GR00T samples and probe observations.

Not a dataset: per-step `Trajectory` recording from `SimBody` is blocked on simulator integration, so these
samples exist to give the parity gates identical inputs on both sides and the weight-plane smoke test
something to compute gradients from. Two properties matter and are what the L1 test checks: one seed gives
one batch, and the normalized actions stay inside the processor's q01/q99 clipping band (a clipped action
is constant and would make the loss insensitive to most of the head).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cosmos_framework.posttrain.config.model import Gr00tN1d7ModelConfig

# Per-step perturbation scales, small enough that the normalized values stay inside the q01/q99 band.
_TRANSLATION_STEP = 0.002  # metres
_ROTATION_STEP = 0.002  # radians
_JOINT_STEP = 0.002  # radians


def probe_image(rng: np.random.Generator, kind: int) -> np.ndarray:
    """One deterministic 224x224 frame: gradient (crop offset), checkerboard (resampling), or noise."""
    size = 224
    if kind == 0:
        ramp = np.linspace(0, 255, size, dtype=np.float32)
        image = np.stack(
            [ramp[None, :].repeat(size, 0), ramp[:, None].repeat(size, 1), np.full((size, size), 128.0)], -1
        )
        return image.astype(np.uint8)
    if kind == 1:
        cell = 8
        board = (((np.arange(size) // cell)[:, None] + (np.arange(size) // cell)[None, :]) % 2) * 255
        return np.repeat(board[:, :, None], 3, axis=2).astype(np.uint8)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def build_observations(config: Any, batch: int, seed: int = 0) -> list[dict[str, Any]]:
    """Deterministic observations shaped from the recipe's contract. Not zeros: an all-zero image hides a
    normalization difference and an all-zero `eef_9d` makes the processor's orthonormalization divide by zero."""
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for index in range(batch):
        obs: dict[str, Any] = {config.prompt_key: "pick up the cube"}
        for camera_index, env_key in enumerate(config.camera_map.values()):
            obs[env_key] = probe_image(rng, (index + camera_index) % 3)
        for gr00t_key, env_key in config.state_map.items():
            if gr00t_key == "eef_9d":
                # [translation(3), rot6d(6)]: the identity pose is zero translation then e_x, e_y.
                obs[env_key] = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
            elif gr00t_key == "joint_position":
                obs[env_key] = rng.uniform(-1.0, 1.0, size=7).astype(np.float32)
            else:
                obs[env_key] = np.array([0.0], dtype=np.float32)
        out.append(obs)
    return out


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' formula: a proper rotation, so the rot6d rows stay orthonormal."""
    axis = axis / np.linalg.norm(axis)
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _eef_chunk(reference: np.ndarray, horizon: int, rng: np.random.Generator) -> np.ndarray:
    """Absolute `[xyz(3), rot6d(6)]` chunk drifting from `reference`. rot6d is the first two rows of a
    rotation matrix and the processor orthonormalizes it, so it is generated from a real rotation."""
    start_translation = np.asarray(reference[:3], dtype=np.float64)
    start_rotation = np.asarray(reference[3:9], dtype=np.float64).reshape(2, 3)
    row1 = start_rotation[0] / np.linalg.norm(start_rotation[0])
    row2 = start_rotation[1] - np.dot(row1, start_rotation[1]) * row1
    row2 = row2 / np.linalg.norm(row2)
    base = np.vstack([row1, row2, np.cross(row1, row2)])

    axis = rng.normal(size=3)
    direction = rng.normal(size=3)
    direction = direction / np.linalg.norm(direction)

    chunk = np.zeros((horizon, 9), dtype=np.float32)
    for step in range(horizon):
        chunk[step, :3] = start_translation + direction * _TRANSLATION_STEP * (step + 1)
        chunk[step, 3:9] = (_rotation_matrix(axis, _ROTATION_STEP * (step + 1)) @ base)[:2, :].flatten()
    return chunk


def synthetic_step_data(config: Gr00tN1d7ModelConfig, observation: dict, seed: int) -> Any:
    """One `VLAStepData` with ABSOLUTE actions: the processor's `apply_action` does the relative conversion."""
    from gr00t.data.types import VLAStepData

    rng = np.random.default_rng(seed)
    horizon = config.model_action_horizon
    images = {
        gr00t_key: [np.asarray(observation[env_key], dtype=np.uint8)]
        for gr00t_key, env_key in config.camera_map.items()
    }
    states = {
        gr00t_key: np.atleast_1d(np.asarray(observation[env_key], dtype=np.float32)).reshape(1, -1)
        for gr00t_key, env_key in config.state_map.items()
    }
    actions: dict[str, np.ndarray] = {}
    for gr00t_key, state in states.items():
        reference = state[-1]
        if gr00t_key == "eef_9d":
            actions[gr00t_key] = _eef_chunk(reference, horizon, rng)
        elif gr00t_key == "gripper_position":
            # Absolute and bounded: a smooth close over the chunk, kept off the 0/1 rails.
            actions[gr00t_key] = np.linspace(0.2, 0.8, horizon, dtype=np.float32).reshape(horizon, 1)
        else:
            drift = rng.normal(size=reference.shape).astype(np.float32)
            drift /= max(float(np.linalg.norm(drift)), 1e-6)
            steps = np.arange(1, horizon + 1, dtype=np.float32)[:, None]
            actions[gr00t_key] = (reference[None, :] + drift[None, :] * _JOINT_STEP * steps).astype(np.float32)
    return VLAStepData(
        images=images,
        states=states,
        actions=actions,
        text=observation[config.prompt_key],
        embodiment=config.resolve_embodiment_tag(),
    )


def synthetic_samples(processor: Any, config: Gr00tN1d7ModelConfig, n: int, seed: int = 0) -> list[dict]:
    """`n` processed per-sample dicts. `processor.eval()` selects the deterministic image transform and
    disables processor-level state dropout; the head's own dropout stays live and is seeded by the gate."""
    from gr00t.data.types import MessageType

    processor.eval()
    return [
        processor(
            [
                {
                    "type": MessageType.EPISODE_STEP.value,
                    "content": synthetic_step_data(config, observation, seed + index),
                }
            ]
        )
        for index, observation in enumerate(build_observations(config, n, seed=seed))
    ]
