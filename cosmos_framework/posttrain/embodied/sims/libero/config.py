# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""LIBERO configuration and source-boundary validation without model dependencies."""

from __future__ import annotations

from dataclasses import dataclass

from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
    DEFAULT_EXECUTION_PROFILE,
    LEGACY_EVAL_PROFILE,
    LIBERO_RUNTIME,
    LIBERO_SUITES,
    SHARED_PROFILE,
    evaluation_state_ids,
    execution_protocol,
    load_partition,
)


@dataclass(slots=True)
class LiberoConfig:
    """Task/evaluation configuration; the simulator runtime is fixed by ``LIBERO_RUNTIME``.

    ``slots`` makes a retired knob such as ``environment`` an error instead of an ignored attribute.
    """

    suite: str = "libero_spatial"
    execution_horizon: int = 16
    max_episode_steps: int = 256
    train_manifest: str = ""
    eval_manifest: str = ""
    eval_profile: str = "legacy_100"
    sim_python: str = ""
    execution_profile: str = DEFAULT_EXECUTION_PROFILE

    def __post_init__(self) -> None:
        if self.suite not in LIBERO_SUITES:
            raise ValueError(f"environment.config.suite unsupported: {self.suite!r}")
        protocol = execution_protocol(self.suite, self.execution_profile)
        if (self.execution_horizon, self.max_episode_steps) != (protocol.execution_horizon, protocol.max_episode_steps):
            raise ValueError("environment execution horizon / episode limit differ from its execution protocol")
        evaluation_state_ids(self.eval_profile)
        if protocol.fixed_episode and self.eval_profile != SHARED_PROFILE:
            raise ValueError("environment.config.execution_profile requires shared_500 manifests")
        if self.suite == "libero_spatial" and self.eval_profile not in (LEGACY_EVAL_PROFILE, SHARED_PROFILE):
            raise ValueError("Spatial requires legacy_100 or shared_500")

    @property
    def simulator_options(self) -> dict:
        return {
            "suite": self.suite,
            "max_steps": self.max_episode_steps,
            "execution_horizon": self.execution_horizon,
            "execution_profile": self.execution_profile,
        }

    def identity(self) -> dict:
        partition = load_partition(self.train_manifest, self.eval_manifest) if self.train_manifest else ()
        return {
            "kind": "libero",
            "suite": self.suite,
            "partition": [manifest.digest for manifest in partition],
            "protocol": execution_protocol(self.suite, self.execution_profile).identity(),
            "environment": LIBERO_RUNTIME.identity(),
            "eval_profile": self.eval_profile,
        }

    def validate_sources(self) -> None:
        self.__post_init__()
        train, evaluation = load_partition(self.train_manifest, self.eval_manifest)
        if train.suite != self.suite or evaluation.profile != self.eval_profile:
            raise ValueError("LIBERO source manifests differ from the configured suite/evaluation profile")
