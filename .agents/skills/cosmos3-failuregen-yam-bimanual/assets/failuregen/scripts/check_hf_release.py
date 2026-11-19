#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run the three examples using a fresh HF download, then offline inference."""

import argparse
import json
import os
from pathlib import Path

from failuregen.notebook import run_streamed
from failuregen.release import materialize_run
from failuregen.showcase import prepare
from huggingface_hub import snapshot_download


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upload-result", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--token-file", type=Path, required=True)
    p.add_argument("--framework", type=Path, required=True)
    p.add_argument("--cookbook", type=Path, required=True)
    args = p.parse_args()
    result = json.loads(args.upload_result.read_text())
    token = args.token_file.read_text().strip()
    bundle = Path(
        snapshot_download(
            result["repo"],
            repo_type="model",
            revision=result["revision"],
            token=token,
            cache_dir=args.output / "download-cache",
            max_workers=4,
        )
    )
    release = json.loads((bundle / "release.json").read_text())
    runtime = materialize_run(bundle, args.output)
    directory = args.output / "showcase"
    prepared = prepare(
        bundle / release["prepared"],
        bundle / release["dataset"],
        runtime,
        bundle / release["checkpoint"],
        args.cookbook / "configs/showcase.json",
        directory,
    )
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HOME=str(args.output / "empty-offline-cache"),
        PYTHONHASHSEED="42",
        PYTHONUNBUFFERED="1",
    )
    for job in prepared["jobs"]:
        run_streamed(
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=4",
            "-m",
            "failuregen.cli",
            "--profile",
            directory / "profile.yaml",
            "run",
            "--manifest",
            directory / job["manifest"],
            "--out-root",
            directory / job["out_root"],
            "--checkpoint",
            bundle / release["checkpoint"],
            "--seed",
            job["seed"],
            cwd=args.framework,
            env=env,
        )
    import av

    videos = []
    for example in prepared["examples"]:
        video = directory / example["video"]
        with av.open(str(video)) as stream:
            count = sum(1 for _ in stream.decode(video=0))
        if count != 17:
            raise ValueError(f"Expected 17 frames: {video}, got {count}")
        videos.append(dict(label=example["label"], video=str(video), frames=count))
    report = dict(
        repo=result["repo"],
        revision=result["revision"],
        fresh_download=True,
        inference_offline=True,
        seed=0,
        videos=videos,
    )
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FRESH_DOWNLOAD_INFERENCE_VERIFIED", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
