# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Run the model and environment owners' allocation probes for one baked recipe."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    from cosmos_framework.utils.config import load_config
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.config.config_joint_ppo import resolve_joint_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = resolve_joint_config(load_config(args.config, []))
    environment = instantiate(config.environment)
    output = Path(os.environ["GR00T_RUNTIME_DIR"])
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cosmos_framework.posttrain.model.gr00t.preflight",
            "--checkpoint",
            config.model.checkpoint_path,
            "--output",
            str(output / "model-preflight.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            environment.config.sim_python,
            "-m",
            "cosmos_framework.posttrain.embodied.sims.libero.preflight",
            "--train-manifest",
            environment.config.train_manifest,
            "--eval-manifest",
            environment.config.eval_manifest,
            "--output",
            str(output / "sim-preflight.json"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
