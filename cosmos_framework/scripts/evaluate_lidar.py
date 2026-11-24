# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Score LiDAR tokenizer reconstructions and generated point clouds."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def doctor() -> dict[str, Any]:
    """Report available packages without constructing models or downloading weights."""
    versions: dict[str, str | None] = {}
    for package in ("numpy", "scipy", "safetensors", "torch", "cosmos-framework", "utonia", "mmdet3d"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "packages": versions,
        "framework_importable": importlib.util.find_spec("cosmos_framework") is not None,
        "base_metrics_need_learned_model": False,
        "optional_models": {
            "utonia": "Only feature extraction needs local pinned weights, Utonia code and CUDA; see docs/lidar_evaluation_models.md",
            "omnihd": "Only detector inference needs local pinned weights and separate MMDetection3D environment",
            "tokenizer": "Framework encode/decode needs the user's exported tokenizer and its runtime",
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("tokenizer", "clouds", "objects", "detections"):
        child = commands.add_parser(name)
        child.add_argument("--manifest", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        if name == "tokenizer":
            child.add_argument("--aggregation", choices=("macro", "micro"), default="macro")
            child.add_argument("--projection", type=Path)
            child.add_argument("--chamfer-max-points", type=int, default=50_000)
    features = commands.add_parser("features")
    for name in ("generated", "reference", "sample-keys", "output"):
        features.add_argument(f"--{name}", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--projection", type=Path)
    prepare.add_argument("--select-reference-frames", action="store_true")
    prepare.add_argument("--pool-reference", action="store_true")
    native = commands.add_parser("framework-tokenizer")
    for name in ("artifact", "manifest", "output-dir"):
        native.add_argument(f"--{name}", type=Path, required=True)
    native.add_argument("--device", default="cuda")
    native.add_argument("--chamfer-max-points", type=int, default=50_000)
    commands.add_parser("doctor")
    args = parser.parse_args(argv)
    result: dict[str, Any]
    if args.command == "doctor":
        result = doctor()
    elif args.command == "tokenizer":
        from cosmos_framework.evaluation.lidar.tokenizer import evaluate_tokenizer

        result = evaluate_tokenizer(args.manifest, args.aggregation, args.projection, args.chamfer_max_points)
    elif args.command == "framework-tokenizer":
        from cosmos_framework.evaluation.lidar.framework_tokenizer import evaluate_framework_tokenizer

        result = evaluate_framework_tokenizer(
            args.artifact,
            args.manifest,
            args.output_dir,
            args.device,
            args.chamfer_max_points,
        )
    elif args.command == "prepare":
        from cosmos_framework.evaluation.lidar.io import prepare_framework_outputs

        result = prepare_framework_outputs(
            args.manifest,
            args.output_dir,
            args.projection,
            args.select_reference_frames,
            args.pool_reference,
        )
    elif args.command == "clouds":
        from cosmos_framework.evaluation.lidar.generation import evaluate_clouds

        result = evaluate_clouds(args.manifest)
    elif args.command == "features":
        from cosmos_framework.evaluation.lidar.generation import evaluate_features

        result = evaluate_features(args.generated, args.reference, args.sample_keys)
    elif args.command == "objects":
        from cosmos_framework.evaluation.lidar.objects import evaluate_objects

        result = evaluate_objects(args.manifest)
    else:
        from cosmos_framework.evaluation.lidar.detection import evaluate_detections

        result = evaluate_detections(args.manifest)
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if getattr(args, "output", None) is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
