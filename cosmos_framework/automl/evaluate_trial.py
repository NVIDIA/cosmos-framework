# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Evaluate only the checkpoint committed by this trial's native checkpointer."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import tomli_w

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def read_toml(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def validate_dcp(checkpoint: Path, *, components: tuple[str, ...] = ("model",)) -> None:
    """Check native DCP file structure without importing torch or unpickling metadata.

    Native loading still validates the serialized state. Framework deduplicates
    replicated state onto the lowest rank, so other ranks may write empty shards.
    Each required component must contain data, but not every rank-local file must.
    """
    root = checkpoint.resolve()
    for component in components:
        directory = checkpoint / component
        metadata = directory / ".metadata"
        shards = list(directory.glob("*.distcp"))
        if not metadata.is_file() or not metadata.resolve().is_relative_to(root) or not metadata.stat().st_size:
            raise ValueError(f"DCP {component} requires nonempty metadata within the trial checkpoint")
        if not shards:
            raise ValueError(f"DCP {component} checkpoint lacks shards")
        if any(not shard.is_file() or not shard.resolve().is_relative_to(root) for shard in shards):
            raise ValueError(f"DCP {component} shards must be regular files within the trial checkpoint")
        if not any(shard.stat().st_size for shard in shards):
            raise ValueError(f"DCP {component} requires nonempty shard data")


def prepare(run_dir: Path, evaluation: dict, output: Path, expected_iteration: int) -> tuple[dict, Path]:
    checkpoint_root = run_dir / "checkpoints"
    name = (checkpoint_root / "latest_checkpoint.txt").read_text().strip()
    checkpoint = (checkpoint_root / name).resolve()
    if not name or checkpoint.parent != checkpoint_root.resolve():
        raise ValueError("Latest-checkpoint pointer must select a direct child of this trial's checkpoints")
    if name != f"iter_{expected_iteration:09d}":
        raise ValueError("The latest checkpoint does not match the approved fixed training budget")
    validate_dcp(checkpoint)
    config_file = run_dir / "config.yaml"
    if not config_file.is_file():
        raise ValueError("Training did not produce its resolved config.yaml")
    evaluation.setdefault("model", {}).update(
        model_name=str(checkpoint),
        config_file=str(config_file),
        export_dir=str(output / "hf"),
        save_folder="metrics",
    )
    evaluation.setdefault("evaluation", {}).update(skip_saved=False, total_shard=1, shard_id=0)
    evaluation["results_dir"] = str(output)
    return evaluation, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-iteration", type=int, required=True)
    args = parser.parse_args()
    config, checkpoint = prepare(
        args.run_dir, read_toml(args.evaluation_config), args.output_dir, args.expected_iteration
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config_path = args.output_dir / "resolved.toml"
    config_path.write_text(tomli_w.dumps(config))
    subprocess.run(
        [sys.executable, "-m", "cosmos_framework.evaluation.reasoner.evaluate", "--config", str(config_path)],
        check=True,
    )
    (args.output_dir / "checkpoint.json").write_text(json.dumps({"checkpoint": str(checkpoint)}))


if __name__ == "__main__":
    main()
