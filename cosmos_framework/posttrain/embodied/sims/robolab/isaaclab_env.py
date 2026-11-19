# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native RoboLab start_isaaclab_app and scene registration. Imports Isaac only when explicitly booted."""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any


def configure_event_reporting(enabled: bool) -> None:
    """Wire RoboLab's CLI flag into registration and its pinned recorder."""
    import robolab.constants

    # Environment registration reads this global; parsing --enable-subtask alone
    # does not enable the progress/event terms recorded in episode_results.jsonl.
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = enabled
    if not enabled:
        return

    # REVERT-WHEN-UPSTREAM: the pinned RoboLab recorder indexes CUDA uint16
    # status tensors directly, which torch does not support. Keep the workaround
    # local and idempotent until RoboLab carries the same int32 round-trip guard.
    import torch
    from robolab.core.logging import recorder_manager

    if torch.uint16 in getattr(recorder_manager, "_UNINDEXABLE_DTYPES", ()):
        return  # Newer native images already handle unsigned tensor indexing.
    slice_to_envs = recorder_manager._slice_to_envs
    if getattr(slice_to_envs, "_cosmos3_uint16_guard", False):
        return

    def _slice_to_envs_with_uint16_guard(value: Any, env_ids: Any) -> Any:
        if isinstance(value, torch.Tensor) and value.is_cuda and value.dtype == torch.uint16:
            return value.to(torch.int32)[env_ids].to(torch.uint16)
        return slice_to_envs(value, env_ids)

    setattr(_slice_to_envs_with_uint16_guard, "_cosmos3_uint16_guard", True)
    recorder_manager._slice_to_envs = _slice_to_envs_with_uint16_guard


def start_isaaclab_app(*, event_reports: bool) -> Any:
    # Ray's uvloop is incompatible with Kit. This code runs on the simulator main thread.
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    asyncio.set_event_loop(asyncio.new_event_loop())
    import cv2  # noqa: F401 -- load OpenCV before Isaac Lab's shared libraries
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--headless", "--enable_cameras"])
    app = AppLauncher(args).app
    # Registration reads this global, so it must be configured before any task registration.
    configure_event_reporting(event_reports)
    return app


def validate_installed_task_metadata(task: str, expected_sha: str) -> None:
    from robolab.constants import TASK_DIR

    from cosmos_framework.posttrain.embodied.sims.robolab.tasks import validate_task_metadata

    validate_task_metadata(
        task=task, metadata=os.path.join(TASK_DIR, "_metadata", "task_metadata.json"), expected_sha=expected_sha
    )


def register_task_environments(tasks: list[str]) -> None:
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

    auto_register_droid_envs(task=tasks, cameras=WRIST_LEFT_RIGHT_HEAD)


def create_task_environment(
    task: str, *, seed: int, n_lanes: int, instruction_type: str, output_dir: str
) -> tuple[Any, Any]:
    from robolab.constants import set_output_dir
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.runtime import create_env
    from robolab.core.task.task import resolve_instruction

    # Each unit gets its own native artifacts without mutating PACKAGE_DIR or shared output links.
    os.makedirs(output_dir, exist_ok=True)
    set_output_dir(output_dir)
    scenes = get_envs(task=[task])
    if len(scenes) != 1:
        raise ValueError(f"RoboLab task {task!r} needs one explicit scene, got {scenes}")
    env, cfg = create_env(
        scenes[0],
        device="cuda:0",
        seed=seed,
        num_envs=n_lanes,
        instruction_type=instruction_type,
        policy="posttrain",
        renderer="realtime",
        rendering_mode=None,
    )
    try:
        # Older images retain a selected instruction dictionary; newer ones return
        # a string. Policies must receive the same text for either native version.
        cfg.instruction = resolve_instruction(cfg.instruction, instruction_type)
    except Exception:
        env.close()
        raise
    return env, cfg
