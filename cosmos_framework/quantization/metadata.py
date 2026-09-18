# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Record quantization provenance separately from source checkpoint configuration."""

import copy
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

METADATA_FILENAME = "quantization_metadata.json"


def collect_quantization_metadata(*, source: str | Path, input_dir: Path, recipe: dict, prompts: list[str]) -> dict:
    """Capture the current environment, source identity, and calibration inputs."""
    packages = {}
    for name in (
        "cosmos-framework",
        "torch",
        "transformers",
        "nvidia-modelopt",
        "diffusers",
        "accelerate",
        "numpy",
        "tokenizers",
        "safetensors",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None

    package_dir = Path(__file__).resolve().parent
    repo_dir = package_dir.parents[1]
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repo_dir), "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None

    input_dir = Path(input_dir).resolve()
    configs = {}
    for name in ("config.json", "generation_config.json", "transformer/config.json", "scheduler/scheduler_config.json"):
        path = input_dir / name
        if path.is_file():
            configs[name] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "transformers_version": json.loads(path.read_text()).get("transformers_version"),
            }
    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "packages": packages},
        "framework": {
            "git_revision": revision,
            "git_dirty": dirty,
            "quantization_source_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(package_dir.glob("*.py"))
            },
        },
        "source": {
            "requested": str(source),
            "resolved_path": str(input_dir),
            "snapshot_revision": input_dir.name if input_dir.parent.name == "snapshots" else None,
            "configs": configs,
        },
        "recipe": copy.deepcopy(recipe),
        "calibration_prompts": list(prompts),
    }


def write_quantization_metadata(output_dir: Path, metadata: dict) -> None:
    """Write a new sidecar without modifying an inherited source symlink."""
    path = Path(output_dir) / METADATA_FILENAME
    if path.is_symlink():
        path.unlink()
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
