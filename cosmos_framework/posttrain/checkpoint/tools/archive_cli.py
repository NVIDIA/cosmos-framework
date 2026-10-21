# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Thin CLI adapter for the archive library and verified local retention."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from cosmos_framework.posttrain.checkpoint.archive.publisher import publish_artifact
from cosmos_framework.posttrain.checkpoint.archive.reader import fetch_archived_artifact, verify_archived_artifact
from cosmos_framework.posttrain.checkpoint.retention import prune_verified_artifacts


def main(argv: Sequence[str] | None = None) -> int:
    """Publish, verify, fetch, or prune checkpoint artifacts; 0 on success, 1 on refusal or failed verification."""
    from cosmos_framework.posttrain.config.rl import CheckpointUploadConfig

    defaults = CheckpointUploadConfig()
    parser = argparse.ArgumentParser(
        prog="python -m cosmos_framework.posttrain.checkpoint.tools.archive_cli",
        description="Digest-bound remote lifecycle of checkpoint_artifacts/ payloads.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish", help="upload one local artifact and write its .uploaded marker")
    publish.add_argument("--artifact-dir", required=True, help="<run>/<ts>/checkpoint_artifacts/iter_*")
    publish.add_argument(
        "--upload-prefix",
        required=True,
        help="the run's rl.upload.prefix (<project>/<group>/<run>/<ts>/checkpoints); the artifact lands under "
        "<dirname>/checkpoint_artifacts/<iter>",
    )
    publish.add_argument("--bucket", default=defaults.bucket, help=f"rl.upload.bucket (default {defaults.bucket})")
    publish.add_argument("--delete-local", action="store_true", help="release the payload after live verification")
    verify = commands.add_parser("verify", help="HEAD-verify every object the artifact's marker records")
    verify.add_argument("--artifact-dir", required=True)
    fetch = commands.add_parser("fetch", help="stage one archived artifact into <dst>/<iter_*>")
    fetch.add_argument("--artifact-uri", required=True, help="s3://<bucket>/<...>/checkpoint_artifacts/iter_*")
    fetch.add_argument("--dst", required=True, help="staging directory; the artifact lands in <dst>/<iter_*>")
    prune = commands.add_parser("prune-verified", help="release local payloads whose archived copies verify live")
    prune.add_argument("--paths", required=True, help="comma-separated artifact directories or payload files")
    for command in (publish, verify, fetch, prune):
        command.add_argument("--credentials", default=defaults.credentials, help="JSON credentials file")
    args = parser.parse_args(argv)
    try:
        if args.command == "publish":
            report = publish_artifact(
                args.artifact_dir,
                {"bucket": args.bucket, "prefix": args.upload_prefix, "credentials": args.credentials},
            )
            if args.delete_local:
                pruned = prune_verified_artifacts([args.artifact_dir], credentials=args.credentials)
                if pruned["kept"]:
                    raise RuntimeError(f"local artifact retained: {pruned['kept']}")
                report["removed"] = pruned["removed"].get(args.artifact_dir, [])
            print(json.dumps(report, sort_keys=True), flush=True)
        elif args.command == "verify":
            verified = verify_archived_artifact(args.artifact_dir, credentials=args.credentials)
            print("verified" if verified else "not verified", flush=True)
            return 0 if verified else 1
        elif args.command == "fetch":
            print(fetch_archived_artifact(args.artifact_uri, args.dst, credentials=args.credentials), flush=True)
        else:
            paths = [path for path in args.paths.split(",") if path]
            print(json.dumps(prune_verified_artifacts(paths, credentials=args.credentials), sort_keys=True), flush=True)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"{args.command}: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
