# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""LIBERO rendering and task-source probe, independent of the model interpreter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cosmos_framework.posttrain.embodied.sims.libero.config import LiberoConfig
from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
    DEFAULT_EXECUTION_PROFILE,
    LIBERO_RUNTIME,
    execution_protocol,
    load_partition,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-profile", default=DEFAULT_EXECUTION_PROFILE)
    args = parser.parse_args()
    import numpy as np

    from cosmos_framework.posttrain.embodied.sims.libero.runtime import LiberoSimulator

    train, evaluation = load_partition(args.train_manifest, args.eval_manifest)
    protocol = execution_protocol(train.suite, args.execution_profile)
    config = LiberoConfig(
        suite=train.suite,
        execution_horizon=protocol.execution_horizon,
        max_episode_steps=protocol.max_episode_steps,
        execution_profile=args.execution_profile,
        eval_profile=evaluation.profile,
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
    )
    config.validate_sources()
    sim = LiberoSimulator(
        embodiment="libero_sim",
        n_lanes=1,
        train_manifest=args.train_manifest,
        **config.simulator_options,
    )
    try:
        sim.probe()
        session = sim.open_window(0)
        chunk = session.step_chunk([np.zeros((protocol.execution_horizon, 7), dtype=np.float32)])[0]
        if (
            chunk.next_observation["image"].shape != (256, 256, 3)
            or not 1 <= len(chunk.rewards) <= protocol.execution_horizon
        ):
            raise RuntimeError("LIBERO rendering/chunk preflight failed")
        result = {
            "runtime": LIBERO_RUNTIME.identity(),
            "train_manifest": train.digest,
            "eval_manifest": evaluation.digest,
            "executed_steps": len(chunk.rewards),
            "protocol": protocol.identity(),
        }
    finally:
        sim.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=4) + "\n")


if __name__ == "__main__":
    main()
