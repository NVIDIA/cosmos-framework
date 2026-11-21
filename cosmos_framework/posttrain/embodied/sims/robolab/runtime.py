# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native RoboLab Simulator; policy execution belongs to the shared episode driver."""

from __future__ import annotations

import os
import uuid
from typing import Any

from cosmos_framework.posttrain.embodied.sim import PROTOCOL_VERSION, SimCapabilities, SimProtocolError
from cosmos_framework.posttrain.embodied.sims.robolab.interface import get_embodiment_map, validate_control_freq
from cosmos_framework.posttrain.embodied.sims.robolab.session import RoboLabSession
from cosmos_framework.posttrain.embodied.unit import WorkUnit


class RoboLabSimulator:
    def __init__(
        self,
        *,
        embodiment: str,
        robot: str = "droid",
        control_freq: int = 15,
        n_lanes: int = 1,
        event_reports: bool = True,
        instruction_type: str = "default",
        output_dir: str = "",
        emit_extrinsics: bool = False,
        dense_caption: bool = False,
        control_mode: str = "joint_pos",
    ) -> None:
        validate_control_freq(robot, control_freq)
        self._dense_caption = dense_caption
        self._control_mode = control_mode
        self._embodiment_map = get_embodiment_map(robot, control_mode)
        if embodiment != self._embodiment_map.name or emit_extrinsics or n_lanes < 1:
            raise ValueError("RoboLab requires its matching control interface, positive lanes and no extrinsics")
        self._robot, self._control_freq, self._n_lanes = robot, control_freq, n_lanes
        self._event_reports, self._instruction_type = event_reports, instruction_type
        self._output_dir = output_dir or os.path.join(os.environ["EVAL_OUTPUT_DIR"], "native", "robolab")
        self._app: Any = None
        self._session: RoboLabSession | None = None
        self._registered: set[str] = set()
        self._generation = 0

    def capabilities(self) -> SimCapabilities:
        camera_hw = (720, 1280)
        cameras = {name: (480, 864) if name == "viewport" else camera_hw for name in self._embodiment_map.cameras}
        return SimCapabilities(
            PROTOCOL_VERSION,
            "robolab",
            os.environ.get("ROBOLAB_REVISION", ""),
            self._embodiment_map.name,
            self._n_lanes,
            cameras,
            self._embodiment_map.state_dim,
            # render=False skips CPU image readback, not Isaac's native rendering.
            supports_render_skip=True,
        )

    def probe(self) -> None:
        from cosmos_framework.posttrain.embodied.sims.robolab.isaaclab_env import start_isaaclab_app

        if self._app is None:
            self._app = start_isaaclab_app(event_reports=self._event_reports)

    def open(self, unit: WorkUnit) -> RoboLabSession:
        from cosmos_framework.posttrain.embodied.sims.robolab.isaaclab_env import (
            create_task_environment,
            register_task_environments,
            validate_installed_task_metadata,
        )

        if unit.suite != "robolab" or unit.scenario["native_batch_size"] > self._n_lanes:
            raise SimProtocolError("RoboLab work unit differs from simulator capabilities")
        if self._session is not None:
            self._session.close()
        self.probe()
        task = str(unit.scenario["task"])
        validate_installed_task_metadata(task, str(unit.scenario["native_task_sha"]))
        if task not in self._registered:
            register_task_environments([task])
            self._registered.add(task)
        env, cfg = create_task_environment(
            task,
            seed=int(unit.scenario["seed"]),
            n_lanes=int(unit.scenario["native_batch_size"]),
            instruction_type=self._instruction_type,
            output_dir=os.path.join(self._output_dir, unit.unit_id, uuid.uuid4().hex),
        )
        self._generation += 1
        try:
            self._session = RoboLabSession(
                unit,
                env,
                cfg,
                robot=self._robot,
                control_freq=self._control_freq,
                generation=self._generation,
                dense_caption=self._dense_caption,
                control_mode=self._control_mode,
            )
        except BaseException:
            env.close()
            raise
        return self._session

    def wants_restart(self) -> bool:
        return False

    def gpu_uuid(self) -> str | None:
        return os.environ.get("CUDA_VISIBLE_DEVICES")

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        # Isaac app.close hard-exits its process. The gateway owns process teardown;
        # only release task-owned environments here so results can be acknowledged first.
