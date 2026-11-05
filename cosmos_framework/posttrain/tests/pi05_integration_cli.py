# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Qualify π0.5 Spatial through the production Joint PPO entry and real LIBERO sessions.

``prepare`` writes shared_500 manifests, an isolated LIBERO path configuration,
and one real reset observation via the simulator interpreter. ``train`` defaults
to three complete windows / nine optimizer updates and full 500-case validation.
``train --resume ITER_DIRECTORY`` keeps the identities and runs one more window.
The output argument is the artifact root; the shared entry chooses its normal
project/group/name/run-timestamp directory beneath it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path


def ensure_manifests(output: Path) -> tuple[Path, Path]:
    from cosmos_framework.posttrain.embodied.sims.libero.protocol import LiberoManifest, fixed_manifest

    directory = output / "manifests"
    plans = [
        (directory / f"libero_spatial-shared_500-{split}.json", fixed_manifest(split, profile="shared_500"))
        for split in ("train", "eval")
    ]
    for path, expected in plans:
        if path.exists() and LiberoManifest.read(path) != expected:
            raise ValueError(f"existing integration manifest differs: {path}")
    directory.mkdir(parents=True, exist_ok=True)
    for path, manifest in plans:
        if not path.exists():
            manifest.write(path)
    return plans[0][0], plans[1][0]


def configure_libero(output: Path, reference_root: Path, assets: Path | None) -> Path:
    import yaml

    benchmark = reference_root / "venv/lib/python3.12/site-packages/libero/libero"
    if assets is None:
        assets = reference_root / (
            "cache/huggingface/hub/datasets--RLinf--LIBERO-assets/snapshots/3ba78404b48e8c70fa4ac9782d5aac8e5b46d55f"
        )
    payload = {
        "benchmark_root": str(benchmark),
        "bddl_files": str(benchmark / "bddl_files"),
        "init_states": str(benchmark / "init_files"),
        "datasets": str(benchmark.parent / "datasets"),
        "assets": str(assets),
    }
    for key in ("benchmark_root", "bddl_files", "init_states", "assets"):
        if not Path(payload[key]).is_dir():
            raise FileNotFoundError(f"LIBERO {key}: {payload[key]}")
    directory = output / "libero-config"
    path = directory / "config.yaml"
    if path.exists():
        if yaml.safe_load(path.read_text()) != payload:
            raise ValueError(f"existing LIBERO path configuration differs: {path}")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=True))
    return directory


def make_integration_config(args, manifests: tuple[Path, Path]):
    from cosmos_framework.posttrain.checkpoint.manifest import iteration_name, iteration_number
    from cosmos_framework.posttrain.configs.pi05_libero_spatial_ppo import make_config
    from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
        PI05_SPATIAL_EXECUTION_PROFILE,
        execution_protocol,
    )

    config = make_config()
    config.model.checkpoint_path = str(args.checkpoint)
    config.model.tokenizer_path = str(args.tokenizer)
    config.model.norm_stats_path = str(args.norm_stats)
    config.environment.config.train_manifest = str(manifests[0])
    config.environment.config.eval_manifest = str(manifests[1])
    config.environment.config.sim_python = str(args.sim_python)
    protocol = execution_protocol("libero_spatial", PI05_SPATIAL_EXECUTION_PROFILE)
    config.rl.num_envs = 8
    config.rl.window_decisions = protocol.max_episode_steps // protocol.execution_horizon
    config.rl.global_minibatch = 128
    config.rl.microbatch = 2
    updates = config.rl.decisions // config.rl.global_minibatch
    config.rl.save_every = updates
    config.rl.validation.enabled = True
    config.rl.validation.before_training = True
    config.rl.validation.every_steps = 3 * updates
    config.rl.validation.save_checkpoint = True
    config.rl.keep_optim_last = 0
    config.rl.keep_model_last = 0
    config.rl.max_steps = 3 * updates
    config.rl.resume = "off"
    if args.resume is not None:
        config.rl.resume = str(args.resume)
        restored_step = iteration_number(iteration_name(str(args.resume)))
        if restored_step % updates:
            raise ValueError("resume checkpoint is not at a complete integration window boundary")
        config.rl.max_steps = restored_step + updates
    config.job.group = "pi05_integration"
    config.job.name = "spatial_small"
    config.job.wandb_mode = "disabled"
    return config


def export_observation(args) -> None:
    import numpy as np

    from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
        PI05_SPATIAL_EXECUTION_PROFILE,
        LiberoCase,
        execution_protocol,
    )
    from cosmos_framework.posttrain.embodied.sims.libero.runtime import LiberoSimulator

    protocol = execution_protocol("libero_spatial", PI05_SPATIAL_EXECUTION_PROFILE)
    case = LiberoCase(args.task_id, args.state_id)
    with ExitStack() as cleanup:
        simulator = LiberoSimulator(
            embodiment="libero_sim",
            n_lanes=1,
            execution_horizon=protocol.execution_horizon,
            max_steps=protocol.max_episode_steps,
            execution_profile=PI05_SPATIAL_EXECUTION_PROFILE,
        )
        cleanup.callback(simulator.close)
        session = simulator.open_evaluation(asdict(case), seed=args.seed)
        cleanup.callback(session.close)
        observation = session.observe()[0]
        np.savez_compressed(
            args.output,
            image=np.asarray(observation["image"])[None],
            wrist_image=np.asarray(observation["wrist_image"])[None],
            state=np.asarray(observation["state"])[None],
            prompt=np.asarray([observation["prompt"]]),
        )
    metadata = {"case": asdict(case), "seed": args.seed, "execution_profile": PI05_SPATIAL_EXECUTION_PROFILE}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"observation": str(args.output), **metadata}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    for name in ("prepare", "train"):
        phase = phases.add_parser(name)
        phase.add_argument("--sim-python", type=Path, required=True)
        phase.add_argument("--output", type=Path, required=True)
        phase.add_argument("--reference-root", type=Path, default=Path("/reference"))
        phase.add_argument("--libero-assets", type=Path)
        if name == "prepare":
            phase.add_argument("--task-id", type=int, default=0)
            phase.add_argument("--state-id", type=int, default=0)
            phase.add_argument("--seed", type=int, default=0)
        else:
            for resource in ("checkpoint", "tokenizer", "norm-stats"):
                phase.add_argument(f"--{resource}", type=Path, required=True)
            phase.add_argument("--resume", type=Path)
            phase.add_argument("--validate-only", action="store_true")
    observe = phases.add_parser("_observe", help="Internal simulator subprocess")
    observe.add_argument("--output", type=Path, required=True)
    observe.add_argument("--task-id", type=int, required=True)
    observe.add_argument("--state-id", type=int, required=True)
    observe.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    if args.phase == "_observe":
        export_observation(args)
        return 0
    args.output = args.output.absolute()
    args.sim_python = args.sim_python.absolute()
    args.reference_root = args.reference_root.absolute()
    if args.libero_assets is not None:
        args.libero_assets = args.libero_assets.absolute()
    temporary = args.output / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temporary)
    manifests = ensure_manifests(args.output)
    directory = configure_libero(args.output, args.reference_root, args.libero_assets)
    os.environ["LIBERO_CONFIG_PATH"] = str(directory)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if args.phase == "prepare":
        subprocess.run(
            [
                str(args.sim_python),
                "-m",
                "cosmos_framework.posttrain.tests.pi05_integration_cli",
                "_observe",
                "--output",
                str(args.output / "libero-observation.npz"),
                "--task-id",
                str(args.task_id),
                "--state-id",
                str(args.state_id),
                "--seed",
                str(args.seed),
            ],
            env=dict(os.environ),
            check=True,
        )
        return 0
    from cosmos_framework.posttrain.entrypoints.joint_ppo import run, validate_config
    from cosmos_framework.posttrain.utils.mesh import detect_mesh

    for name in ("checkpoint", "tokenizer", "norm_stats", "resume"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.absolute())
    config = make_integration_config(args, manifests)
    if args.validate_only:
        nodes, gpus = detect_mesh()
        validate_config(config, nodes * gpus)
        print("Pi05 Spatial integration configuration and 500-case manifests are valid")
        return 0
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(args.output)
    summary = run(config)
    expected_windows = 1 if args.resume is not None else 3
    if summary["stop_reason"] is None and summary["windows"] != expected_windows:
        raise RuntimeError(f"integration completed {summary['windows']} windows, expected {expected_windows}")
    (args.output / f"integration-summary-step-{summary['optimizer_step']}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
