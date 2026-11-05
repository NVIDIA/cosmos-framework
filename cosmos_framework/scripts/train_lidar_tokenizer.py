# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Native LiDAR tokenizer V1.2 continuation and serving export entry point."""

import argparse
import json
from pathlib import Path

from cosmos_framework.configs.tokenizer.lidar_v1p2 import LidarV1P2Recipe
from cosmos_framework.data.tokenizer.lidar import read_manifest
from cosmos_framework.trainer.tokenizer.lidar import export_checkpoint, sha256, train
from cosmos_framework.trainer.tokenizer.lidar_evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train")
    training.add_argument("--train-manifest", type=Path, required=True)
    training.add_argument("--eval-manifest", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    initialization = training.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--warm-start", type=Path)
    initialization.add_argument("--resume", type=Path)
    training.add_argument("--checkpoint-sha256", required=True)
    training.add_argument("--stop-after", type=int, default=30_000)
    training.add_argument("--workers", type=int, default=4)
    training.add_argument("--dryrun", action="store_true")
    exporting = subparsers.add_parser("export")
    exporting.add_argument("--checkpoint", type=Path, required=True)
    exporting.add_argument("--output", type=Path, required=True)
    exporting.add_argument("--weights", choices=("live", "ema"), required=True)
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--artifact", type=Path, required=True)
    evaluation.add_argument("--manifest", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "export":
        export_checkpoint(args.checkpoint, args.output, args.weights)
        return
    if args.command == "evaluate":
        evaluate(args.artifact, args.manifest, args.output, args.device)
        return
    recipe = LidarV1P2Recipe()
    samples, heldout = read_manifest(args.train_manifest), read_manifest(args.eval_manifest)
    if set(samples) & set(heldout):
        parser.error("Train and evaluation manifests must be disjoint; split by recording/session")
    checkpoint = args.warm_start or args.resume
    if sha256(checkpoint) != args.checkpoint_sha256.lower():
        parser.error("Checkpoint SHA256 does not match the selected immutable artifact")
    if not 0 < args.stop_after <= recipe.max_steps or args.workers < 0:
        parser.error("stop-after must be in 1..30000 and workers must be nonnegative")
    if args.dryrun:
        print(
            json.dumps(
                {
                    **recipe.manifest(),
                    "train_clips": len(samples),
                    "eval_clips": len(heldout),
                    "initialization": "resume" if args.resume else "network-only warm start",
                    "checkpoint_sha256": args.checkpoint_sha256,
                    "stop_after": args.stop_after,
                },
                indent=2,
            )
        )
        return
    train(
        recipe,
        samples,
        args.output,
        heldout=heldout,
        warm_start=args.warm_start,
        resume=args.resume,
        stop_after=args.stop_after,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
